"""SAE training loop.

Workflow
--------
1. Load a frozen model (any registered architecture + checkpoint).
2. Hook into a named layer to extract activations for every training batch.
3. Train a :class:`~sparse_ae.model.SparseAutoencoder` on those activations.

Usage (CLI)
-----------
    python -m sparse_ae.train \\
        --model_config ../../config/HydraDNA_cVQVAEPlus.yaml \\
        --model_weights ../../results/models/model.pth \\
        --hook_layer  encoder_gru \\
        --fasta       ../../data/deepSTARR/Sequences_Train.fa \\
        --activity    ../../data/deepSTARR/Sequences_activity_Train.txt \\
        --dict_size   1024 \\
        --l1_coeff    1e-3 \\
        --epochs      50 \\
        --output_dir  ../../results/sae/
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import load_config, load_fasta_with_strand_correction, one_hot_encode_dna
from models.registry import build_model
from .model import SparseAutoencoder, TopKSparseAutoencoder


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def _register_hook(module, activations_out: list):
    """Attach a forward hook that appends pooled activations to *activations_out*."""
    def hook(_module, _inp, output):
        if isinstance(output, tuple):
            output = output[0]
        if output.dim() == 3:          # [B, C, L] → [B, C]
            output = output.mean(dim=2)
        activations_out.append(output.detach().cpu())
    return module.register_forward_hook(hook)


def load_activations(
    model: torch.nn.Module,
    module: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Collect activations from *module* given all batches in *dataloader*.

    Returns
    -------
    torch.Tensor, shape ``(N, D)``
    """
    activations: list = []
    handle = _register_hook(module, activations)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            try:
                model(X)
            except Exception:
                pass  # forward may fail after hook fires; that's OK
    handle.remove()
    return torch.cat(activations, dim=0)


def _resolve_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Return a submodule by dot-separated name (e.g. ``'encoder_gru'``)."""
    parts = name.split(".")
    m = model
    for p in parts:
        m = getattr(m, p)
    return m


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_sae(
    sae,
    activations: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
    output_dir: str = ".",
    seed: int = 42,
    resample_interval: int = 0,
    resample_threshold: float = 1e-5,
) -> dict:
    """Train *sae* on pre-extracted *activations*.

    Parameters
    ----------
    sae : SparseAutoencoder | TopKSparseAutoencoder
        Model to train.
    activations : torch.Tensor, shape ``(N, D)``
        Pre-computed frozen-model activations.
    epochs : int
    batch_size : int
    lr : float
    device : torch.device
    output_dir : str
        Where to write checkpoints and the training history JSON.
    seed : int
    resample_interval : int
        If > 0, re-initialise dead encoder rows every this many epochs.
        0 disables resampling.
    resample_threshold : float
        Activation threshold below which a feature is treated as dead.

    Returns
    -------
    history : dict
        Keys ``epoch``, ``total_loss``, ``recon_loss``, ``l1_loss``,
        ``dead_fraction``, ``avg_active_pct``, ``n_resampled``.
    """
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    sae = sae.to(device)

    dataset    = TensorDataset(activations)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimiser  = optim.Adam(sae.parameters(), lr=lr)

    history = {k: [] for k in ("epoch", "total_loss", "recon_loss", "l1_loss",
                                "dead_fraction", "avg_active_pct", "n_resampled")}

    for epoch in range(1, epochs + 1):
        sae.train()
        tot, rec, sp = 0.0, 0.0, 0.0

        for (x,) in loader:
            x = x.to(device)
            optimiser.zero_grad()
            total, recon, l1 = sae.loss(x)
            total.backward()
            optimiser.step()
            sae._normalise_decoder()
            tot += total.item()
            rec += recon.item()
            sp  += l1.item()

        n    = len(loader)
        probe = activations.to(device)
        dead  = sae.dead_feature_fraction(probe)

        # Average active features per sequence (% of dict_size)
        with torch.no_grad():
            sae.eval()
            feats_probe = sae.encode(probe)                           # [B, dict_size]
            active_per_seq = (feats_probe > 1e-5).float().sum(dim=1)
            avg_active_pct = (active_per_seq.mean().item() / sae.dict_size) * 100
            sae.train()

        # Dead-neuron resampling
        n_resampled = 0
        if resample_interval > 0 and epoch % resample_interval == 0 and epoch < epochs * 0.75:
            sae.eval()
            n_resampled = sae.resample_dead_features(
                probe, threshold=resample_threshold)
            if n_resampled:
                print(f"           Resampled {n_resampled} dead features")

        msg = (f"Epoch {epoch:>3}/{epochs}  total={tot/n:.4f}  "
               f"recon={rec/n:.4f}  l1={sp/n:.4f}  dead={dead:.2%}  "
               f"avg_active={avg_active_pct:.1f}%")
        print(msg)

        history["epoch"].append(epoch)
        history["total_loss"].append(tot / n)
        history["recon_loss"].append(rec / n)
        history["l1_loss"].append(sp / n)
        history["dead_fraction"].append(dead)
        history["avg_active_pct"].append(avg_active_pct)
        history["n_resampled"].append(n_resampled)

    torch.save(sae.state_dict(), os.path.join(output_dir, "sae.pth"))
    with open(os.path.join(output_dir, "sae_history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"[SAE] Saved checkpoint → {output_dir}/sae.pth")
    return history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train a sparse autoencoder on frozen model activations.")
    ap.add_argument("--model_config",  required=True,  help="YAML config of the frozen model.")
    ap.add_argument("--model_weights", required=True,  help="Path to frozen model .pth checkpoint.")
    ap.add_argument("--hook_layer",    default="encoder_gru",
                    help="Dot-separated submodule name to hook (default: encoder_gru).")
    ap.add_argument("--fasta",         required=True,  help="FASTA file for training sequences.")
    ap.add_argument("--activity",      required=True,  help="Tab-separated activity file.")
    ap.add_argument("--dict_size",     type=int,   default=1024, help="SAE feature dictionary size.")
    ap.add_argument("--l1_coeff",      type=float, default=1e-3, help="L1 sparsity coefficient.")
    ap.add_argument("--epochs",        type=int,   default=50)
    ap.add_argument("--batch_size",    type=int,   default=256)
    ap.add_argument("--lr",            type=float, default=3e-4)
    ap.add_argument("--output_dir",    default="../../results/sae/")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--topk",          type=int,   default=0,
                    help="If > 0 use TopKSparseAutoencoder with this k instead of L1 SAE.")
    ap.add_argument("--resample_interval", type=int, default=10,
                    help="Re-initialise dead features every N epochs (0 = off).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen model
    config = load_config(args.model_config)
    model  = build_model(config).to(device)
    model.load_state_dict(
        torch.load(args.model_weights, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Build dataloader
    seqs    = load_fasta_with_strand_correction(args.fasta)
    X       = one_hot_encode_dna(seqs)
    X_t     = torch.from_numpy(X).float()
    dataset  = TensorDataset(X_t)
    loader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Extract activations
    hook_module = _resolve_module(model, args.hook_layer)
    print(f"[SAE] Extracting activations from layer '{args.hook_layer}' …")
    acts = load_activations(model, hook_module, loader, device)
    print(f"[SAE] Activations shape: {acts.shape}")

    # Build + train SAE
    if args.topk > 0:
        sae = TopKSparseAutoencoder(input_dim=acts.shape[1], dict_size=args.dict_size,
                                    k=args.topk)
    else:
        sae = SparseAutoencoder(input_dim=acts.shape[1], dict_size=args.dict_size,
                                l1_coeff=args.l1_coeff)
    train_sae(sae, acts, epochs=args.epochs, batch_size=args.batch_size,
              lr=args.lr, device=device, output_dir=args.output_dir,
              seed=args.seed, resample_interval=args.resample_interval)


if __name__ == "__main__":
    main()
