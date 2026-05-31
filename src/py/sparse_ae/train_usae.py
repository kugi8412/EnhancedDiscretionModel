"""Universal Sparse Autoencoder (USAE) training loop.

Based on Thasarathan et al. 2025 (arXiv:2502.03714).
Trains a single overcomplete dictionary that simultaneously reconstructs 
the activations of multiple models.
"""

import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.cuda import seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
# Ensure we can import from the main project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import load_config, load_fasta_with_strand_correction, one_hot_encode_dna
from models.registry import build_model
from .train import load_activations, _resolve_module


# ===========================================================================
# Universal TopK Sparse Autoencoder
# ===========================================================================


def resolve_topk(k, dict_size):
    """Resolve TopK value: if 0 < k < 1, treat as fraction of dict_size."""
    if isinstance(k, float) and 0.0 < k < 1.0:
        return max(1, int(k * dict_size))
    return int(k)


class UniversalTopKSAE(nn.Module):
    """Universal SAE using Top-K hard sparsity.
    
    Shares a single latent dictionary across multiple models.
    Each model has its own encoder/decoder projecting to/from the shared space.
    
    Architecture per model i (Thasarathan et al. 2025):
        Encoder: Linear(d_i, m) → BatchNorm(m) → ReLU → TopK
        Decoder: Z @ D(i)  (dictionary matrix, no bias)
    """
    def __init__(self, input_dims: list[int], dict_size: int, k: int = 64):
        super().__init__()
        self.num_models = len(input_dims)
        self.dict_size = dict_size
        self.k = resolve_topk(k, dict_size)

        # Model-specific encoders (linear + batch norm) and decoders
        self.encoders = nn.ModuleList([
            nn.Linear(dim, dict_size) for dim in input_dims
        ])
        self.encoder_bns = nn.ModuleList([
            nn.BatchNorm1d(dict_size) for _ in input_dims
        ])
        self.decoders = nn.ModuleList([
            nn.Linear(dict_size, dim, bias=False) for dim in input_dims
        ])
        self.pre_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(dim)) for dim in input_dims
        ])

        # Kaiming init
        for enc, dec in zip(self.encoders, self.decoders):
            nn.init.kaiming_uniform_(enc.weight)
            nn.init.kaiming_uniform_(dec.weight)
            
        self.normalise_decoders()

    def normalise_decoders(self):
        """Ensure decoder column norms are <= 1."""
        with torch.no_grad():
            for dec in self.decoders:
                norms = dec.weight.norm(dim=0, keepdim=True).clamp(min=1.0)
                dec.weight.div_(norms)

    def encode(self, x: torch.Tensor, model_idx: int) -> torch.Tensor:
        """Encode activations from a specific model into the universal latent space."""
        x_c = x - self.pre_biases[model_idx]
        h = self.encoders[model_idx](x_c)
        h = self.encoder_bns[model_idx](h)
        pre = F.relu(h)
        
        # Top-K hard sparsity
        topk_v, topk_i = pre.topk(self.k, dim=1)
        out = torch.zeros_like(pre)
        out.scatter_(1, topk_i, topk_v)
        return out

    def decode(self, z: torch.Tensor, model_idx: int) -> torch.Tensor:
        """Decode universal latent vector back into a specific model's activation space."""
        return self.decoders[model_idx](z) + self.pre_biases[model_idx]

    def resample_dead_features(self, probe_acts_list: list[torch.Tensor], threshold: float = 1e-5, noise_scale: float = 0.2) -> int:
        """Resample dead features across all universal encoders/decoders."""
        with torch.no_grad():
            # 1. Znajdź neurony, które są martwe we WSZYSTKICH modelach
            is_active = torch.zeros(self.dict_size, dtype=torch.bool, device=probe_acts_list[0].device)
            for m in range(self.num_models):
                Z = self.encode(probe_acts_list[m], m)
                active_m = (Z.max(dim=0).values > threshold)
                is_active = is_active | active_m
            
            is_dead = ~is_active
            n_dead = int(is_dead.sum().item())
            if n_dead == 0:
                return 0

            dead_idx = torch.where(is_dead)[0]

            # 2. Zresetuj je dla każdego modelu z osobna, kierując na trudne przykłady
            for m in range(self.num_models):
                acts = probe_acts_list[m]
                Z = self.encode(acts, m)
                x_hat = self.decode(Z, m)
                
                # Oblicz błąd dla każdego przykładu
                losses = F.mse_loss(x_hat, acts, reduction='none').mean(dim=1)
                
                # Wybierz przykłady z najwyższym błędem
                top_idx = torch.argsort(losses, descending=True)[:n_dead]
                candidates = acts[top_idx]
                
                # Zabezpieczenie, jeśli mamy więcej martwych neuronów niż przykładów w probe (rzadkie)
                if len(candidates) < n_dead:
                    repeats = (n_dead // len(candidates)) + 1
                    candidates = candidates.repeat(repeats, 1)[:n_dead]

                new_dir = F.normalize(candidates, dim=1)
                noise = torch.randn_like(new_dir) * noise_scale
                resampled_weights = new_dir + noise

                # Aktualizacja Kodera
                self.encoders[m].weight.data[dead_idx] = resampled_weights
                self.encoders[m].bias.data[dead_idx] = 0.0
                
                # Reset BatchNorm stats for dead features
                self.encoder_bns[m].running_mean[dead_idx] = 0.0
                self.encoder_bns[m].running_var[dead_idx] = 1.0
                
                # Aktualizacja Dekodera (kolumny)
                self.decoders[m].weight.data[:, dead_idx] = resampled_weights.T

            self.normalise_decoders()
            return n_dead


# ===========================================================================
# Training Pipeline
# ===========================================================================

def train_usae(
    usae: UniversalTopKSAE,
    acts_list: list[torch.Tensor],
    model_names: list[str],
    epochs: int = 300,
    batch_size: int = 512,
    max_lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
    output_dir: str = ".",
    resample_interval: int = 25,
    seed: int = 42
):
    """Main USAE training loop.
    
    Follows Thasarathan et al. 2025:
    - Each (encoder_i, decoder_i) pair has an independent Adam optimizer.
    - Each iteration: randomly select model i, encode through encoder i,
      decode through ALL decoders, compute L1 reconstruction loss, but only
      step optimizer i (updating encoder_i and decoder_i).
    - Uses L1 loss (not MSE) per Appendix A.1.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    usae = usae.to(device)

    # TensorDataset accepts multiple tensors -> spits out aligned batches!
    dataset = TensorDataset(*acts_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    total_steps = epochs * len(loader)
    
    # Independent Adam optimizer per (encoder_i, decoder_i) pair (Appendix A.1)
    optimizers = []
    schedulers = []
    for i in range(usae.num_models):
        params_i = (
            list(usae.encoders[i].parameters()) +
            list(usae.encoder_bns[i].parameters()) +
            list(usae.decoders[i].parameters()) +
            [usae.pre_biases[i]]
        )
        opt_i = optim.Adam(params_i, lr=max_lr)
        sched_i = optim.lr_scheduler.OneCycleLR(
            opt_i,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=1000.0
        )
        optimizers.append(opt_i)
        schedulers.append(sched_i)

    history = {
        "epoch": [],
        "total_loss": [],
        "dead_fraction": []
    }
    
    # Track cross-reconstruction per model pair
    for i in range(usae.num_models):
        for j in range(usae.num_models):
            history[f"l1_{model_names[i]}_to_{model_names[j]}"] = []

    print("\n[USAE] Starting Universal SAE Training...")
    
    for epoch in range(1, epochs + 1):
        usae.train()
        epoch_loss = 0.0
        
        # Track L1 specifically for monitoring
        l1_tracking = np.zeros((usae.num_models, usae.num_models))

        for batch_acts in loader:
            batch_acts = [act.to(device) for act in batch_acts]
            
            # Zero ALL optimizers to clear stale gradients
            for opt in optimizers:
                opt.zero_grad()
            
            # Step 1: Randomly select ONE encoder (Section 3.2)
            enc_idx = random.randint(0, usae.num_models - 1)
            
            # Step 2: Encode into universal space Z
            Z = usae.encode(batch_acts[enc_idx], enc_idx)
            
            # Step 3: Decode through ALL decoders and sum L1 losses
            total_loss = 0.0
            for dec_idx in range(usae.num_models):
                x_hat = usae.decode(Z, dec_idx)
                loss = F.l1_loss(x_hat, batch_acts[dec_idx])
                total_loss += loss
                
                with torch.no_grad():
                    l1_tracking[enc_idx, dec_idx] += loss.item()
            
            # Step 4: Backprop, but only step optimizer for encoder i (Section 3.2)
            total_loss.backward()
            optimizers[enc_idx].step()
            schedulers[enc_idx].step()
            
            usae.normalise_decoders()
            epoch_loss += total_loss.item()

        # Analytics
        n_batches = len(loader)
        avg_loss = epoch_loss / n_batches
        l1_tracking /= (n_batches / usae.num_models) # Approximate avg per selected encoder

        # Dead fraction check
        with torch.no_grad():
            usae.eval()
            probe_acts = [acts[:2000].to(device) for acts in acts_list]
            is_active = torch.zeros(usae.dict_size, dtype=torch.bool, device=device)
            for m in range(usae.num_models):
                Z_probe = usae.encode(probe_acts[m], m)
                is_active = is_active | (Z_probe.max(dim=0).values > 1e-5)
            
            dead_frac = 1.0 - is_active.float().mean().item()

        if epoch % resample_interval == 0 and epoch < (epochs * 0.75):
            n_resampled = usae.resample_dead_features(probe_acts)
            if n_resampled > 0:
                print(f"           [!] Reset {n_resampled} dead neurons!")

        if epoch % 10 == 0 or epoch == 1:
            lr_curr = optimizers[0].param_groups[0]['lr']
            print(f"Epoch {epoch:>3}/{epochs} | LR: {lr_curr:.2e} | Total L1: {avg_loss:.4f} | Dead: {dead_frac:.1%}")

        history["epoch"].append(epoch)
        history["total_loss"].append(avg_loss)
        history["dead_fraction"].append(dead_frac)
        for i in range(usae.num_models):
            for j in range(usae.num_models):
                history[f"l1_{model_names[i]}_to_{model_names[j]}"].append(l1_tracking[i, j])

    # Save
    torch.save(usae.state_dict(), os.path.join(output_dir, "usae.pth"))
    with open(os.path.join(output_dir, "usae_history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"\n[USAE] Training complete! Saved to {output_dir}")
    return usae


# ===========================================================================
# CLI & Execution
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Train a Universal SAE across multiple models.")
    ap.add_argument("--models", nargs="+", required=True, 
                    help="List of models in format 'config_path:weights_path:layer_name'")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--activity", required=True)
    ap.add_argument("--dict_size", type=int, default=1024)
    ap.add_argument("--top_k", type=float, default=64,
                    help="TopK: integer for absolute count, float <1 for fraction (e.g. 0.1 = 10%%)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Parse models
    model_configs = []
    for m in args.models:
        parts = m.split(":")
        if len(parts) != 3:
            raise ValueError(f"Model arg must be config:weights:layer. Got: {m}")
        model_configs.append({"config": parts[0], "weights": parts[1], "layer": parts[2]})
        
    model_names = [Path(m["config"]).stem for m in model_configs]
    print(f"[USAE] Initialising Universal SAE for {len(model_names)} models: {model_names}")

    # 2. Data Preparation
    seqs = load_fasta_with_strand_correction(args.fasta)
    X = one_hot_encode_dna(seqs)
    X_t = torch.from_numpy(X).float()
    dataset = TensorDataset(X_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Extract and standardise activations
    extracted_acts = []
    norm_stats = {"means": [], "stds": []}
    
    for idx, (m_info, name) in enumerate(zip(model_configs, model_names)):
        print(f"[{name}] Loading and extracting layer '{m_info['layer']}'...")
        cfg = load_config(m_info["config"])
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(m_info["weights"], map_location=device, weights_only=True))
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
            
        hook_module = _resolve_module(model, m_info["layer"])
        acts = load_activations(model, hook_module, loader, device)
        
        # USAE Requirement: Standardise activations
        mean = acts.mean(dim=0, keepdim=True)
        std = acts.std(dim=0, keepdim=True).clamp(min=1e-8)
        acts = (acts - mean) / std
        
        extracted_acts.append(acts)
        norm_stats["means"].append(mean.cpu())
        norm_stats["stds"].append(std.cpu())
        
        print(f"[{name}] Activations shaped {acts.shape} standardised (mean=0, std=1).")
        
        # Clear memory
        del model
        torch.cuda.empty_cache()

    # Save standardisation statistics so analyze_usae.py can use them!
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(norm_stats, os.path.join(args.output_dir, "usae_norm_stats.pth"))

    input_dims = [acts.shape[1] for acts in extracted_acts]

    # 4. Build and train
    usae = UniversalTopKSAE(input_dims=input_dims, dict_size=args.dict_size, k=args.top_k)
    
    train_usae(
        usae=usae,
        acts_list=extracted_acts,
        model_names=model_names,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_lr=args.lr,
        device=device,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
