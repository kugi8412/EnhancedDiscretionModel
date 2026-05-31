"""Universal SAE analysis script.

Based on Thasarathan et al. 2025 (arXiv:2502.03714).
Computes Cross-reconstruction R^2, Firing Entropy (FE), Co-Fire Proportion (CFP),
Concept Energy, and extracts k-mer enrichments for Universal vs Model-Specific concepts.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import load_config, load_fasta_with_strand_correction, one_hot_encode_dna
from models.registry import build_model
from .train import load_activations, _resolve_module
from .train_usae import UniversalTopKSAE
from .analyze import _kmer_counts, enriched_kmers, top_k_sequences


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def compute_usae_metrics(
    Z_list: list[np.ndarray], 
    y_dev: np.ndarray, 
    y_hk: np.ndarray
) -> pd.DataFrame:
    """Compute FE, CFP, Energy, and PCC for USAE concepts.
    
    Parameters
    ----------
    Z_list : list of np.ndarray
        List of encoded latent spaces [N, dict_size] for each model.
    """
    num_models = len(Z_list)
    dict_size = Z_list[0].shape[1]
    
    # 1. Calculate Activation Frequencies (for Firing Entropy)
    # fire_freqs shape: [num_models, dict_size]
    fire_freqs = np.array([(Z > 0).mean(axis=0) for Z in Z_list])
    total_freqs = fire_freqs.sum(axis=0) + 1e-10
    
    # Firing Entropy (FE) = - sum(p * log(p)) / log(M)
    p = fire_freqs / total_freqs
    p_safe = np.where(p > 0, p, 1e-10)
    FE = -np.sum(p_safe * np.log(p_safe), axis=0) / np.log(num_models)
    
    # 2. Co-Fire Proportion (CFP)
    # Fraction of times concept fires simultaneously in ALL models, 
    # given that it fired in AT LEAST ONE model.
    fired = np.stack([(Z > 0) for Z in Z_list], axis=0) # [M, N, dict_size]
    fired_all = fired.all(axis=0).sum(axis=0) # [dict_size]
    fired_any = fired.any(axis=0).sum(axis=0) + 1e-10
    CFP = fired_all / fired_any
    
    # 3. Concept Energy
    # Energy(k) = || E[Z_k] * D_k ||_2^2
    # Since we normalize decoders, D_k has norm 1. 
    # We take the mean activation across all models for each concept.
    mean_Z = np.mean([Z.mean(axis=0) for Z in Z_list], axis=0) # [dict_size]
    energy = mean_Z ** 2
    
    # 4. Correlation with Target (PCC)
    # We use the average Z across models as the "universal" activation strength
    Z_avg = np.mean(Z_list, axis=0) # [N, dict_size]
    
    records = []
    for fi in range(dict_size):
        feat = Z_avg[:, fi]
        row = {
            "feature_idx": fi,
            "mean_activation": float(mean_Z[fi]),
            "firing_entropy": float(FE[fi]),
            "co_fire_prop": float(CFP[fi]),
            "energy": float(energy[fi]),
        }
        
        if feat.std() > 0:
            row["pcc_dev"] = float(pearsonr(feat, y_dev)[0])
            row["pcc_hk"] = float(pearsonr(feat, y_hk)[0])
        else:
            row["pcc_dev"] = 0.0
            row["pcc_hk"] = 0.0
            
        # Classify concept based on Thasarathan et al. thresholds
        if FE[fi] > 0.9:
            row["classification"] = "Universal"
        elif FE[fi] < 0.2:
            # Find which model it is specific to
            specific_to = np.argmax(fire_freqs[:, fi])
            row["classification"] = f"Specific_M{specific_to}"
        else:
            row["classification"] = "Mixed"
            
        records.append(row)
        
    return pd.DataFrame(records), Z_avg


def main():
    ap = argparse.ArgumentParser(description="Analyse Universal SAE (USAE) features.")
    ap.add_argument("--usae_checkpoint", required=True)
    ap.add_argument("--models", nargs="+", required=True, 
                    help="Config:Weights:Layer matching training.")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--activity", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dict_size", type=int, default=2048)
    ap.add_argument("--top_k", type=int, default=500, help="For motif matching")
    ap.add_argument("--kmer_size", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Parse models
    model_configs = []
    for m in args.models:
        parts = m.split(":")
        model_configs.append({"config": parts[0], "weights": parts[1], "layer": parts[2]})
    model_names = [Path(m["config"]).stem for m in model_configs]
    num_models = len(model_names)

    # 2. Load Data
    print("[USAE] Loading FASTA and standardisation stats...")
    seqs = load_fasta_with_strand_correction(args.fasta)
    X = torch.from_numpy(one_hot_encode_dna(seqs)).float()
    
    act_df = pd.read_csv(args.activity, sep='\t')
    y_dev = act_df.iloc[:, 0].values.astype(np.float32)
    y_hk = act_df.iloc[:, 1].values.astype(np.float32)

    loader = DataLoader(TensorDataset(X), batch_size=512, shuffle=False)

    # Load Standardisation Stats from training
    train_dir = str(Path(args.usae_checkpoint).parent)
    stats_path = os.path.join(train_dir, "usae_norm_stats.pth")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing normalisation stats at {stats_path}. Run train_usae.py first.")
    norm_stats = torch.load(stats_path, map_location="cpu", weights_only=True)

    # 3. Extract & Standardise Activations
    extracted_acts = []
    for idx, (m_info, name) in enumerate(zip(model_configs, model_names)):
        cfg = load_config(m_info["config"])
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(m_info["weights"], map_location=device, weights_only=True))
        model.eval()

        hook_module = _resolve_module(model, m_info["layer"])
        acts = load_activations(model, hook_module, loader, device)
        acts = acts.to(device)
        
        # Standardise using TRAINING stats
        mean = norm_stats["means"][idx].to(device)
        std = norm_stats["stds"][idx].to(device)
        acts = (acts - mean) / std
        extracted_acts.append(acts)
        
        del model
        torch.cuda.empty_cache()

    # 4. Load USAE
    input_dims = [acts.shape[1] for acts in extracted_acts]
    # top_k in SAE definition is different from top_k sequences, assuming 64 as per Phase 7
    usae = UniversalTopKSAE(input_dims=input_dims, dict_size=args.dict_size, k=64).to(device)
    usae.load_state_dict(torch.load(args.usae_checkpoint, map_location=device, weights_only=True))
    usae.eval()

    # 5. Compute R^2 Matrix and encode latents
    print("[USAE] Computing Latents and Cross-Reconstruction R^2...")
    r2_matrix = np.zeros((num_models, num_models))
    Z_list = []

    with torch.no_grad():
        # Get latents
        for idx in range(num_models):
            # Process in chunks if memory is an issue, but standard test sets fit on GPU
            Z = usae.encode(extracted_acts[idx], model_idx=idx).cpu().numpy()
            Z_list.append(Z)
            
        # Get R^2
        for enc_idx in range(num_models):
            Z_tensor = torch.from_numpy(Z_list[enc_idx]).to(device)
            for dec_idx in range(num_models):
                x_hat = usae.decode(Z_tensor, model_idx=dec_idx).cpu().numpy()
                true_acts = extracted_acts[dec_idx].cpu().numpy()
                r2_matrix[enc_idx, dec_idx] = compute_r2(true_acts, x_hat)

    # Save R^2 Matrix
    r2_df = pd.DataFrame(r2_matrix, index=model_names, columns=model_names)
    r2_df.to_csv(os.path.join(args.output_dir, "cross_reconstruction_r2.csv"))
    print("\n--- Cross-Reconstruction R^2 ---")
    print(r2_df.round(3))
    print("--------------------------------\n")

    # 6. Compute Feature Metrics (FE, CFP, Energy)
    print("[USAE] Calculating Firing Entropy, Co-Fire Proportion, and Energy...")
    stats_df, Z_avg = compute_usae_metrics(Z_list, y_dev, y_hk)
    
    # Replace Specific_M0 with actual model names
    for i, name in enumerate(model_names):
        stats_df["classification"] = stats_df["classification"].replace(f"Specific_M{i}", f"Specific_{name}")
        
    stats_df.to_csv(os.path.join(args.output_dir, "usae_feature_stats.csv"), index=False)
    
    # Print summary
    counts = stats_df["classification"].value_counts()
    print("[USAE] Concept Classification Summary:")
    for cat, count in counts.items():
        print(f"  - {cat}: {count} concepts")

    # 7. K-mer Enrichment (For Universal and Top Specific Concepts)
    print(f"\n[USAE] Running k-mer enrichment (k={args.kmer_size}) for concepts...")
    bg_counts = _kmer_counts(seqs, k=args.kmer_size).values.sum(axis=0)
    total_bg_seqs = len(seqs)
    
    top_k_idx = top_k_sequences(Z_avg, seqs, k=args.top_k)
    enrichments = {}
    
    # Select features to analyze: All universal + top 100 specific
    target_features = stats_df[
        (stats_df["classification"] == "Universal") | 
        (stats_df["energy"] > stats_df["energy"].quantile(0.8)) # High energy concepts
    ]["feature_idx"].tolist()
    
    for i, fi in enumerate(target_features):
        top_seqs = [seqs[idx] for idx in top_k_idx[fi]]
        enr = enriched_kmers(top_seqs, bg_counts, total_bg_seqs, k=args.kmer_size, top_n=20)
        enrichments[fi] = enr.to_dict("records")
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(target_features)} features...")

    with open(os.path.join(args.output_dir, "usae_kmer_enrichments.json"), "w") as fh:
        json.dump(enrichments, fh, indent=2)
        
    print(f"\n[USAE] Analysis complete! All files saved to {args.output_dir}")


if __name__ == "__main__":
    main()
