"""SAE feature → motif analysis.

For every SAE feature dimension, compute:
- Mean activation over the dataset
- Correlation with Dev / Hk activity scores
- Top-k sequences that maximally activate this feature
- JASPAR motif enrichment in the top-k sequences (via FIMO / biopython / built-in PWMS)

Usage (CLI)
-----------
    python -m sparse_ae.analyze \\
        --sae_checkpoint ../../results/sae/sae.pth \\
        --model_config   ../../config/HydraDNA_cVQVAEPlus.yaml \\
        --model_weights  ../../results/models/model.pth \\
        --hook_layer     encoder_gru \\
        --fasta          ../../data/deepSTARR/Sequences_Test.fa \\
        --activity       ../../data/deepSTARR/Sequences_activity_Test.txt \\
        --output_dir     ../../results/sae/analysis/ \\
        --top_k          500
"""

from __future__ import annotations

import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import load_config, load_fasta_with_strand_correction, one_hot_encode_dna
from models.registry import build_model
from .model import SparseAutoencoder
from .train import _resolve_module, load_activations


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_feature_stats(
    features: np.ndarray,           # (N, F)
    y_dev:    np.ndarray,            # (N,)
    y_hk:     np.ndarray,            # (N,)
) -> pd.DataFrame:
    """Compute per-feature statistics and correlations with activity.

    Returns
    -------
    pd.DataFrame with columns:
        feature_idx, mean_activation, frac_active,
        pcc_dev, pcc_hk, scc_dev, scc_hk
    """
    records = []
    F = features.shape[1]
    for fi in range(F):
        feat = features[:, fi]
        mask = feat > 0
        row = {
            "feature_idx":    fi,
            "mean_activation": float(feat.mean()),
            "frac_active":     float(mask.mean()),
        }
        if mask.sum() > 10:
            row["pcc_dev"] = float(pearsonr(feat, y_dev)[0]) if np.std(feat) > 0 else 0.0
            row["pcc_hk"]  = float(pearsonr(feat, y_hk)[0])  if np.std(feat) > 0 else 0.0
            row["scc_dev"] = float(spearmanr(feat, y_dev)[0])
            row["scc_hk"]  = float(spearmanr(feat, y_hk)[0])
        else:
            row.update({"pcc_dev": 0.0, "pcc_hk": 0.0, "scc_dev": 0.0, "scc_hk": 0.0})
        records.append(row)
    return pd.DataFrame(records)


def top_k_sequences(features: np.ndarray, sequences: list[str], k: int = 500) -> dict:
    """For each feature, return incides of the top-*k* activating sequences."""
    F = features.shape[1]
    top_k = {}
    for fi in range(F):
        indices = np.argsort(-features[:, fi])[:k]
        top_k[fi] = indices.tolist()
    return top_k


def _kmer_counts(seqs: list[str], k: int = 6) -> pd.DataFrame:
    """Compute k-mer frequency matrix (rows = sequences, cols = k-mers)."""
    from itertools import product
    bases    = "ACGT"
    kmers    = ["".join(p) for p in product(bases, repeat=k)]
    kmer_idx = {km: i for i, km in enumerate(kmers)}

    mat = np.zeros((len(seqs), len(kmers)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        seq = seq.upper()
        for j in range(len(seq) - k + 1):
            km = seq[j:j + k]
            if km in kmer_idx:
                mat[i, kmer_idx[km]] += 1
        row_sum = mat[i].sum()
        if row_sum > 0:
            mat[i] /= row_sum  # TPM normalise

    return pd.DataFrame(mat, columns=kmers)


def enriched_kmers(
    top_seqs:  list[str],
    all_seqs:  list[str],
    k:         int = 6,
    top_n:     int = 20,
) -> pd.DataFrame:
    """Fisher-exact enrichment of k-mers in *top_seqs* vs *all_seqs* background.

    Returns
    -------
    pd.DataFrame sorted by p-value with columns ``kmer``, ``enrichment``, ``pvalue``.
    """
    from scipy.stats import fisher_exact

    top_counts = _kmer_counts(top_seqs, k).values.sum(axis=0)
    bg_counts  = _kmer_counts(all_seqs,  k).values.sum(axis=0)
    kmers_all  = ["".join(p) for p in __import__("itertools").product("ACGT", repeat=k)]

    records = []
    for ki, km in enumerate(kmers_all):
        a = top_counts[ki]
        b = bg_counts[ki]
        table = [[a, len(top_seqs) - a], [b, len(all_seqs) - b]]
        odds, p = fisher_exact(table, alternative="greater")
        records.append({"kmer": km, "enrichment": float(odds), "pvalue": float(p)})

    df = pd.DataFrame(records).sort_values("pvalue").head(top_n)
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    sae:            SparseAutoencoder,
    features:       np.ndarray,     # (N, F)
    sequences:      list[str],
    y_dev:          np.ndarray,
    y_hk:           np.ndarray,
    output_dir:     str,
    top_k:          int = 500,
    kmer_size:      int = 6,
    min_frac_active: float = 0.02,
):
    """End-to-end: compute stats, save CSV, compute k-mer enrichment for top features."""
    os.makedirs(output_dir, exist_ok=True)

    stats = compute_feature_stats(features, y_dev, y_hk)
    stats.to_csv(os.path.join(output_dir, "feature_stats.csv"), index=False)
    print(f"[SAE] Feature stats → {output_dir}/feature_stats.csv")

    active_features = stats[stats["frac_active"] >= min_frac_active]["feature_idx"].tolist()
    print(f"[SAE] Analysing {len(active_features)} features with frac_active ≥ {min_frac_active}")

    top_k_idx  = top_k_sequences(features, sequences, k=top_k)
    enrichments = {}
    for fi in active_features:
        top_seqs = [sequences[i] for i in top_k_idx[fi]]
        enr = enriched_kmers(top_seqs, sequences, k=kmer_size, top_n=20)
        enrichments[fi] = enr.to_dict("records")

    with open(os.path.join(output_dir, "kmer_enrichments.json"), "w") as fh:
        json.dump(enrichments, fh, indent=2)
    print(f"[SAE] k-mer enrichments → {output_dir}/kmer_enrichments.json")
    return stats, enrichments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyse SAE features for motif enrichment.")
    ap.add_argument("--sae_checkpoint",  required=True)
    ap.add_argument("--model_config",    required=True)
    ap.add_argument("--model_weights",   required=True)
    ap.add_argument("--hook_layer",      default="encoder_gru")
    ap.add_argument("--fasta",           required=True)
    ap.add_argument("--activity",        required=True)
    ap.add_argument("--output_dir",      default="../../results/sae/analysis/")
    ap.add_argument("--top_k",           type=int,   default=500)
    ap.add_argument("--kmer_size",       type=int,   default=6)
    ap.add_argument("--dict_size",       type=int,   default=1024)
    ap.add_argument("--input_dim",       type=int,   default=256,
                    help="Activation dimension (must match training).")
    ap.add_argument("--l1_coeff",        type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.model_config)
    model  = build_model(config).to(device)
    model.load_state_dict(
        torch.load(args.model_weights, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    fasta_seqs = load_fasta_with_strand_correction(args.fasta)
    act_df  = pd.read_csv(args.activity, sep='\t')
    y_dev   = act_df.iloc[:, 0].values.astype(np.float32)
    y_hk    = act_df.iloc[:, 1].values.astype(np.float32)

    X_t      = torch.from_numpy(one_hot_encode_dna(fasta_seqs)).float()
    dataset  = TensorDataset(X_t)
    loader   = DataLoader(dataset, batch_size=256, shuffle=False)
    hook_mod = _resolve_module(model, args.hook_layer)
    acts     = load_activations(model, hook_mod, loader, device)
    print(f"[SAE] Activations: {acts.shape}")

    sae = SparseAutoencoder(args.input_dim, args.dict_size, args.l1_coeff)
    sae.load_state_dict(
        torch.load(args.sae_checkpoint, map_location="cpu", weights_only=True))
    sae.eval()

    with torch.no_grad():
        features = sae.encode(acts.to(device)).cpu().numpy()

    run_analysis(sae, features, fasta_seqs, y_dev, y_hk, args.output_dir,
                 top_k=args.top_k, kmer_size=args.kmer_size)


if __name__ == "__main__":
    main()
