"""Universal SAE visualisation suite.

Based on Thasarathan et al. 2025 (Phase 7c).
Generates publication-ready plots for Cross-Reconstruction, Firing Entropy, 
Concept Energy, K-mer enrichment, and consistency with independent SAEs.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cross_recon_heatmap(analysis_dir: str, output_dir: str):
    """Plot Cross-reconstruction R^2 confusion matrix."""
    csv_path = os.path.join(analysis_dir, "cross_reconstruction_r2.csv")
    if not os.path.exists(csv_path):
        print(f"[Warning] {csv_path} not found. Skipping heatmap.")
        return

    df = pd.read_csv(csv_path, index_col=0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=0, vmax=1.0, fmt=".3f", 
                linewidths=0.5, cbar_kws={'label': r'Reconstruction $R^2$'})
    plt.title("Cross-Model Reconstruction $R^2$", pad=20)
    plt.ylabel("Encoder Model (Source)")
    plt.xlabel("Decoder Model (Target)")
    
    # Move x-axis labels to top for better matrix feel
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "usae_cross_recon_r2.png"), dpi=300)
    plt.close()


def plot_firing_entropy(analysis_dir: str, output_dir: str):
    """Plot Firing Entropy (FE) bimodal histogram."""
    csv_path = os.path.join(analysis_dir, "usae_feature_stats.csv")
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    sns.histplot(df["firing_entropy"], bins=50, color="indigo", kde=False)
    
    # Add threshold lines
    plt.axvline(0.2, color='red', linestyle='--', label='Specific (<0.2)')
    plt.axvline(0.9, color='green', linestyle='--', label='Universal (>0.9)')
    
    plt.title("Concept Firing Entropy (FE) Distribution")
    plt.xlabel("Firing Entropy (0 = Model-Specific, 1 = Universal)")
    plt.ylabel("Number of Concepts")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "usae_firing_entropy.png"), dpi=300)
    plt.close()


def plot_cfp_vs_energy(analysis_dir: str, output_dir: str):
    """Plot Co-Fire Proportion vs Concept Energy."""
    csv_path = os.path.join(analysis_dir, "usae_feature_stats.csv")
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 7))
    
    # Sort so 'Mixed' is plotted first (at the bottom), and Universal/Specific on top
    df_sorted = df.sort_values("classification", key=lambda x: x == "Mixed")
    
    sns.scatterplot(
        data=df_sorted, 
        x="co_fire_prop", 
        y="energy", 
        hue="classification",
        palette="Set1",
        alpha=0.7,
        s=40
    )
    
    plt.yscale("log")
    plt.title("Concept Importance vs Universality")
    plt.xlabel("Co-Fire Proportion (CFP)")
    plt.ylabel("Concept Energy (Log Scale)")
    
    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "usae_cfp_vs_energy.png"), dpi=300)
    plt.close()


def plot_kmer_enrichments(analysis_dir: str, output_dir: str):
    """Plot top enriched k-mers for the highest energy Universal and Specific concepts."""
    csv_path = os.path.join(analysis_dir, "usae_feature_stats.csv")
    json_path = os.path.join(analysis_dir, "usae_kmer_enrichments.json")
    
    if not os.path.exists(json_path):
        print(f"[Warning] {json_path} not found. Skipping k-mer plots.")
        return

    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        enrichments = json.load(f)

    # Find Top 1 Universal and Top 1 Specific per model (by energy)
    targets = []
    
    univ_df = df[df["classification"] == "Universal"]
    if not univ_df.empty:
        targets.append(univ_df.nlargest(1, "energy").iloc[0])
        
    for model_name in df["classification"].unique():
        if "Specific" in model_name:
            spec_df = df[df["classification"] == model_name]
            if not spec_df.empty:
                targets.append(spec_df.nlargest(1, "energy").iloc[0])

    kmer_dir = os.path.join(output_dir, "kmers")
    os.makedirs(kmer_dir, exist_ok=True)

    for target in targets:
        fi = str(int(target["feature_idx"]))
        if fi not in enrichments:
            continue
            
        records = enrichments[fi][:15] # Top 15 k-mers
        kmers = [r["kmer"] for r in records]
        odds = [r["enrichment"] for r in records]
        
        plt.figure(figsize=(8, 6))
        colors = ["#2ecc71" if target["classification"] == "Universal" else "#e74c3c" for _ in odds]
        
        plt.barh(kmers[::-1], odds[::-1], color=colors)
        plt.axvline(1.0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Enrichment (Odds Ratio)")
        plt.title(f"[{target['classification']}] Concept {fi} (Energy: {target['energy']:.2f})")
        
        plt.tight_layout()
        plt.savefig(os.path.join(kmer_dir, f"concept_{fi}_{target['classification']}.png"), dpi=150)
        plt.close()


def plot_consistency(usae_checkpoint: str, indep_sae_dirs: list[str], output_dir: str):
    """Compare USAE dictionary with Independent SAE dictionaries via Cosine Similarity."""
    # We load USAE state dict directly to avoid needing to instantiate the class with exact dims
    usae_state = torch.load(usae_checkpoint, map_location="cpu", weights_only=True)
    
    # Extract USAE decoder weights
    usae_decoders = []
    idx = 0
    while True:
        key = f"decoders.{idx}.weight"
        if key in usae_state:
            usae_decoders.append(usae_state[key])
            idx += 1
        else:
            break
            
    if not usae_decoders:
        print("[Error] Could not find decoder weights in USAE checkpoint.")
        return

    consistency_results = []
    
    plt.figure(figsize=(10, 6))

    for i, sae_dir in enumerate(indep_sae_dirs):
        model_name = os.path.basename(os.path.normpath(sae_dir))
        sae_path = os.path.join(sae_dir, "sae.pth")
        
        if not os.path.exists(sae_path):
            print(f"[Warning] SAE not found at {sae_path}")
            continue
            
        if i >= len(usae_decoders):
            print(f"[Warning] More independent SAEs than USAE decoders. Skipping {model_name}")
            break

        # Load Independent SAE weights
        indep_state = torch.load(sae_path, map_location="cpu", weights_only=True)
        # Handle both TopK (encoder/decoder) and L1 SAE architectures
        w_key = "decoder.weight" if "decoder.weight" in indep_state else "W_dec"
        
        if w_key not in indep_state:
            print(f"[Warning] Could not find decoder weights in {sae_path}")
            continue
            
        W_indep = indep_state[w_key] # [input_dim, indep_dict_size]
        W_usae = usae_decoders[i]    # [input_dim, usae_dict_size]
        
        # Normalize columns (features)
        W_indep_norm = F.normalize(W_indep, dim=0)
        W_usae_norm = F.normalize(W_usae, dim=0)
        
        # Cosine similarity matrix [indep_dict_size, usae_dict_size]
        sim_matrix = W_indep_norm.T @ W_usae_norm
        
        # For each feature in independent SAE, find the max similarity in USAE
        max_sims = sim_matrix.max(dim=1).values.numpy()
        
        # Metrics
        matched = (max_sims > 0.5).sum()
        total = len(max_sims)
        match_pct = (matched / total) * 100
        
        consistency_results.append({
            "Model": model_name,
            "Matched Concepts (>0.5)": f"{matched} / {total} ({match_pct:.1f}%)",
            "Mean Max Sim": f"{max_sims.mean():.3f}"
        })
        
        sns.kdeplot(max_sims, label=f"{model_name} (Matches: {match_pct:.1f}%)", fill=True, alpha=0.3)

    # Save summary table
    if consistency_results:
        pd.DataFrame(consistency_results).to_csv(os.path.join(output_dir, "consistency_summary.csv"), index=False)
        print("\n--- Consistency with Independent SAEs ---")
        print(pd.DataFrame(consistency_results).to_string(index=False))
        print("-----------------------------------------\n")

    plt.axvline(0.5, color="red", linestyle="--", label="Match Threshold (0.5)")
    plt.title("USAE vs Independent SAE Consistency (Max Cosine Similarity)")
    plt.xlabel("Max Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "usae_consistency_kde.png"), dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Generate plots for Universal SAE analysis.")
    ap.add_argument("--usae_checkpoint", required=True)
    ap.add_argument("--analysis_dir", required=True)
    ap.add_argument("--indep_sae_dirs", nargs="+", required=True,
                    help="Paths to folders containing independent sae.pth files")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("[Viz] Plotting Cross-Reconstruction Heatmap...")
    plot_cross_recon_heatmap(args.analysis_dir, args.output_dir)
    
    print("[Viz] Plotting Firing Entropy Distribution...")
    plot_firing_entropy(args.analysis_dir, args.output_dir)
    
    print("[Viz] Plotting Co-Fire Proportion vs Energy...")
    plot_cfp_vs_energy(args.analysis_dir, args.output_dir)
    
    print("[Viz] Plotting K-mer Enrichments...")
    plot_kmer_enrichments(args.analysis_dir, args.output_dir)
    
    print("[Viz] Computing and Plotting Consistency with Independent SAEs...")
    plot_consistency(args.usae_checkpoint, args.indep_sae_dirs, args.output_dir)
    
    print(f"\n[Viz] All plots generated successfully in: {args.output_dir}")


if __name__ == "__main__":
    main()
