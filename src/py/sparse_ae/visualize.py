"""SAE visualization utilities.

Functions
---------
plot_feature_stats(stats_csv, output_dir)
    Scatter and histogram plots of mean activation, fraction active, and PCC.
plot_feature_activity_correlation(stats_csv, output_dir)
    Dev vs Hk PCC per feature, coloured by fraction active.
plot_kmer_enrichment(enrichments_json, feature_idx, output_dir)
    Bar chart of top enriched k-mers for a given feature.
plot_activation_heatmap(features, subset, output_dir)
    Heatmap of feature activations for a random subset of sequences.
compare_model_features(stats_list, model_names, output_dir)
    Overlaid histogram of across-model PCC distributions.
"""

from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_feature_stats(stats_csv: str, output_dir: str) -> None:
    """Summary plots of per-feature statistics."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(stats_csv)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].hist(df["frac_active"], bins=50, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Fraction active")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Feature sparsity distribution")

    axes[1].hist(df["pcc_dev"], bins=50, color="coral",   edgecolor="white", alpha=0.7, label="Dev")
    axes[1].hist(df["pcc_hk"],  bins=50, color="seagreen", edgecolor="white", alpha=0.7, label="Hk")
    axes[1].set_xlabel("Pearson r with activity")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Feature–activity PCC")
    axes[1].legend()

    axes[2].scatter(df["pcc_dev"], df["pcc_hk"],
                    c=df["frac_active"], cmap="viridis",
                    s=6, alpha=0.6)
    axes[2].set_xlabel("PCC Dev")
    axes[2].set_ylabel("PCC Hk")
    axes[2].set_title("Dev vs Hk correlation (colour = frac active)")
    plt.colorbar(axes[2].collections[0], ax=axes[2], label="frac active")

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_stats.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[viz] Saved {path}")


def plot_kmer_enrichment(enrichments_json: str, feature_idx: int, output_dir: str) -> None:
    """Horizontal bar chart of k-mer enrichment for feature *feature_idx*."""
    os.makedirs(output_dir, exist_ok=True)
    with open(enrichments_json) as f:
        enrichments = json.load(f)

    key    = str(feature_idx)
    if key not in enrichments:
        print(f"[viz] Feature {feature_idx} not found in enrichments JSON.")
        return

    records = enrichments[key][:20]
    kmers   = [r["kmer"]       for r in records]
    odds    = [r["enrichment"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["steelblue" if o >= 1 else "salmon" for o in odds]
    ax.barh(kmers[::-1], odds[::-1], color=colors[::-1])
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Enrichment (odds ratio)")
    ax.set_title(f"Feature {feature_idx} — top enriched k-mers")
    plt.tight_layout()
    path = os.path.join(output_dir, f"feature_{feature_idx}_kmers.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[viz] Saved {path}")


def plot_activation_heatmap(
    features: np.ndarray,
    output_dir: str,
    subset_size: int = 200,
    top_features: int = 50,
) -> None:
    """Heatmap of activations for a random subset of sequences × top features."""
    os.makedirs(output_dir, exist_ok=True)

    rng    = np.random.default_rng(0)
    idx    = rng.choice(features.shape[0], min(subset_size, features.shape[0]), replace=False)
    sub    = features[idx]

    # Select top features by mean activation
    top_fi = np.argsort(-sub.mean(axis=0))[:top_features]
    sub    = sub[:, top_fi]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(sub.T, ax=ax, cmap="viridis", xticklabels=False,
                yticklabels=[str(f) for f in top_fi])
    ax.set_xlabel("Sequence")
    ax.set_ylabel("Feature index")
    ax.set_title("SAE activation heatmap (random subset)")
    plt.tight_layout()
    path = os.path.join(output_dir, "activation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[viz] Saved {path}")


def compare_model_features(
    stats_list: list[str],
    model_names: list[str],
    output_dir: str,
    metric: str = "pcc_dev",
) -> None:
    """Overlaid histogram of a correlation metric across multiple SAE runs."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    palette = cm.tab10.colors
    for i, (csv_path, name) in enumerate(zip(stats_list, model_names)):
        df = pd.read_csv(csv_path)
        ax.hist(df[metric].abs(), bins=40, alpha=0.55,
                label=name, color=palette[i % 10], edgecolor="white", density=True)

    ax.set_xlabel(f"|{metric}|")
    ax.set_ylabel("Density")
    ax.set_title(f"Cross-model comparison — {metric}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"compare_{metric}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[viz] Saved {path}")
