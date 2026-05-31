#!/usr/bin/env python
"""
Sequence selection for fine-tuning based on PixelCNN uncertainty.

Uses a trained PixelCNN (conditioned or unconditional) to compute per-sequence
entropy and reconstruction error, then generates plots to help select sequences
that would benefit most from fine-tuning.

Metrics computed per sequence:
  - Mean entropy: avg entropy of softmax across all positions
  - Max entropy positions: number of positions with entropy > threshold
  - Reconstruction accuracy: % positions correctly predicted
  - Per-position entropy profile: identifies uncertain regions

Plots generated:
  1. Entropy histogram — identifies bimodal distribution (easy vs hard sequences)
  2. Entropy vs Expression scatter — reveals which expression ranges are uncertain
  3. Top-N most uncertain sequences (for targeted fine-tuning)
  4. Positional entropy heatmap — uncertain regions across sequences

Usage:
    python select_sequences.py \\
        --pixelcnn_config ../../config/PixelCNN_Conditioned.yaml \\
        --pixelcnn_weights train_logs/DNA_PixelCNN_Conditioned_v1_seed42.pth \\
        --fasta ../../data/deepSTARR/Sequences_Train.fa \\
        --activity ../../data/deepSTARR/Sequences_activity_Train.txt \\
        --output_dir outputs/sequence_selection/ \\
        --top_k 500
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_config, load_fasta_sequences, one_hot_encode_dna
from models.registry import build_model


def compute_sequence_uncertainty(model, latent_model, dataloader, device):
    """Compute per-sequence entropy and reconstruction metrics.

    Returns
    -------
    results : dict with keys:
        'mean_entropy': np.ndarray [N] — avg entropy per sequence
        'max_entropy_positions': np.ndarray [N] — count of high-entropy positions
        'recon_accuracy': np.ndarray [N] — % correct per sequence
        'positional_entropy': np.ndarray [N, L] — full entropy map
    """
    model.eval()
    if latent_model is not None:
        latent_model.eval()

    all_mean_entropy = []
    all_max_ent_pos = []
    all_recon_acc = []
    all_pos_entropy = []

    entropy_threshold = np.log(4) * 0.5  # 50% of max entropy (uniform over 4 bases)

    with torch.no_grad():
        for (X_batch,) in dataloader:
            X_batch = X_batch.to(device)

            # Get conditioning latent if available
            latent = None
            if latent_model is not None:
                latent, _ = latent_model.encode_to_latent(X_batch)

            # Forward pass
            if latent is not None:
                logits = model(X_batch, latent=latent)
            else:
                logits = model(X_batch)

            # Compute per-position entropy
            probs = F.softmax(logits, dim=1)  # [B, 4, L]
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=1)  # [B, L]

            # Reconstruction accuracy
            true_seq = X_batch.argmax(dim=1)  # [B, L]
            pred_seq = logits.argmax(dim=1)   # [B, L]
            recon_acc = (pred_seq == true_seq).float().mean(dim=1) * 100.0  # [B]

            entropy_np = entropy.cpu().numpy()
            all_pos_entropy.append(entropy_np)
            all_mean_entropy.append(entropy_np.mean(axis=1))
            all_max_ent_pos.append((entropy_np > entropy_threshold).sum(axis=1))
            all_recon_acc.append(recon_acc.cpu().numpy())

    return {
        'mean_entropy': np.concatenate(all_mean_entropy),
        'max_entropy_positions': np.concatenate(all_max_ent_pos),
        'recon_accuracy': np.concatenate(all_recon_acc),
        'positional_entropy': np.concatenate(all_pos_entropy, axis=0),
    }


def plot_entropy_histogram(mean_entropy, output_dir):
    """Plot 1: Distribution of mean per-sequence entropy."""
    plt.figure(figsize=(10, 6))
    plt.hist(mean_entropy, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
    plt.axvline(np.percentile(mean_entropy, 90), color='red', linestyle='--',
                label=f'90th percentile ({np.percentile(mean_entropy, 90):.3f})')
    plt.axvline(np.percentile(mean_entropy, 95), color='darkred', linestyle='--',
                label=f'95th percentile ({np.percentile(mean_entropy, 95):.3f})')
    plt.xlabel('Mean Per-Position Entropy (nats)')
    plt.ylabel('Number of Sequences')
    plt.title('Sequence Uncertainty Distribution (PixelCNN)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_histogram.png'), dpi=200)
    plt.close()


def plot_entropy_vs_expression(mean_entropy, y_dev, y_hk, output_dir):
    """Plot 2: Entropy vs expression level — where is the model uncertain?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc0 = axes[0].scatter(y_dev, mean_entropy, c=mean_entropy, cmap='YlOrRd',
                          s=3, alpha=0.4, vmin=0)
    axes[0].set_xlabel('Dev log2 enrichment')
    axes[0].set_ylabel('Mean Entropy')
    axes[0].set_title('Uncertainty vs Developmental Expression')
    plt.colorbar(sc0, ax=axes[0], label='Entropy')

    sc1 = axes[1].scatter(y_hk, mean_entropy, c=mean_entropy, cmap='YlOrRd',
                          s=3, alpha=0.4, vmin=0)
    axes[1].set_xlabel('Hk log2 enrichment')
    axes[1].set_ylabel('Mean Entropy')
    axes[1].set_title('Uncertainty vs Housekeeping Expression')
    plt.colorbar(sc1, ax=axes[1], label='Entropy')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_vs_expression.png'), dpi=200)
    plt.close()


def plot_top_uncertain_heatmap(positional_entropy, indices, sequences, output_dir, n_show=50):
    """Plot 3: Positional entropy heatmap for most uncertain sequences."""
    n_show = min(n_show, len(indices))
    top_entropy = positional_entropy[indices[:n_show]]

    plt.figure(figsize=(14, max(6, n_show * 0.15)))
    sns.heatmap(top_entropy, cmap='YlOrRd', xticklabels=False,
                yticklabels=[f"#{i}" for i in indices[:n_show]],
                cbar_kws={'label': 'Entropy (nats)'})
    plt.xlabel('Position (bp)')
    plt.ylabel('Sequence Index')
    plt.title(f'Top {n_show} Most Uncertain Sequences — Positional Entropy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_uncertain_heatmap.png'), dpi=150)
    plt.close()


def plot_selection_summary(df, output_dir, top_k):
    """Plot 4: Summary scatter of selected vs non-selected sequences."""
    plt.figure(figsize=(10, 7))

    mask = df['selected']
    plt.scatter(df.loc[~mask, 'recon_accuracy'], df.loc[~mask, 'mean_entropy'],
                s=4, alpha=0.3, c='grey', label='Not selected')
    plt.scatter(df.loc[mask, 'recon_accuracy'], df.loc[mask, 'mean_entropy'],
                s=12, alpha=0.7, c='crimson', label=f'Selected (top {top_k})')

    plt.xlabel('Reconstruction Accuracy (%)')
    plt.ylabel('Mean Entropy (nats)')
    plt.title('Sequence Selection for Fine-Tuning')
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'selection_summary.png'), dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Select sequences for fine-tuning via PixelCNN uncertainty.")
    ap.add_argument('--pixelcnn_config', required=True, help="YAML config for the PixelCNN model")
    ap.add_argument('--pixelcnn_weights', required=True, help="Trained PixelCNN weights (.pth)")
    ap.add_argument('--fasta', required=True, help="FASTA file of sequences to score")
    ap.add_argument('--activity', required=True, help="Activity TSV (Dev/Hk columns)")
    ap.add_argument('--output_dir', default='outputs/sequence_selection/')
    ap.add_argument('--top_k', type=int, default=500, help="Number of sequences to select")
    ap.add_argument('--batch_size', type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config and model
    config = load_config(args.pixelcnn_config)
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.pixelcnn_weights, map_location=device, weights_only=True))
    model.eval()

    # Load latent conditioning model if specified
    latent_model = None
    latent_cfg = config.get('latent_model', None)
    if latent_cfg and latent_cfg.get('config_path'):
        print(f"[INFO] Loading latent model: {latent_cfg['config_path']}")
        latent_config = load_config(latent_cfg['config_path'])
        latent_model = build_model(latent_config).to(device)
        latent_model.load_state_dict(
            torch.load(latent_cfg['weights_path'], map_location=device, weights_only=True))
        latent_model.eval()

    # Load data
    print("[INFO] Loading sequences...")
    sequences = load_fasta_sequences(args.fasta)
    X = torch.from_numpy(one_hot_encode_dna(sequences)).float()
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    activity = pd.read_csv(args.activity, sep='\t')
    y_dev = activity.iloc[:, 0].values
    y_hk = activity.iloc[:, 1].values

    # Compute uncertainty
    print("[INFO] Computing per-sequence uncertainty...")
    results = compute_sequence_uncertainty(model, latent_model, loader, device)

    # Rank by uncertainty (highest entropy = most uncertain = best for fine-tuning)
    ranking = np.argsort(results['mean_entropy'])[::-1]
    selected_indices = ranking[:args.top_k]

    # Build results DataFrame
    df = pd.DataFrame({
        'seq_idx': np.arange(len(sequences)),
        'mean_entropy': results['mean_entropy'],
        'max_entropy_positions': results['max_entropy_positions'],
        'recon_accuracy': results['recon_accuracy'],
        'dev_expression': y_dev,
        'hk_expression': y_hk,
        'selected': False,
    })
    df.loc[selected_indices, 'selected'] = True
    df['rank'] = 0
    df.loc[selected_indices, 'rank'] = np.arange(1, args.top_k + 1)

    # Save results
    df.to_csv(os.path.join(args.output_dir, 'sequence_scores.csv'), index=False)

    # Save selected sequence indices and FASTA
    selected_df = df[df['selected']].sort_values('rank')
    selected_df.to_csv(os.path.join(args.output_dir, 'selected_sequences.csv'), index=False)

    fasta_path = os.path.join(args.output_dir, 'selected_sequences.fa')
    with open(fasta_path, 'w') as f:
        for _, row in selected_df.iterrows():
            idx = int(row['seq_idx'])
            f.write(f">seq_{idx}_entropy={row['mean_entropy']:.4f}_"
                    f"dev={row['dev_expression']:.2f}_hk={row['hk_expression']:.2f}\n")
            f.write(f"{sequences[idx]}\n")

    # Generate plots
    print("[INFO] Generating selection plots...")
    plot_entropy_histogram(results['mean_entropy'], args.output_dir)
    plot_entropy_vs_expression(results['mean_entropy'], y_dev, y_hk, args.output_dir)
    plot_top_uncertain_heatmap(results['positional_entropy'], selected_indices,
                              sequences, args.output_dir)
    plot_selection_summary(df, args.output_dir, args.top_k)

    # Print summary
    print(f"\n{'='*60}")
    print(f" SEQUENCE SELECTION SUMMARY")
    print(f"{'='*60}")
    print(f" Total sequences scored:    {len(sequences)}")
    print(f" Selected for fine-tuning:  {args.top_k}")
    print(f" Entropy threshold (90th):  {np.percentile(results['mean_entropy'], 90):.4f}")
    print(f" Selected mean entropy:     {results['mean_entropy'][selected_indices].mean():.4f}")
    print(f" Non-selected mean entropy: {results['mean_entropy'][ranking[args.top_k:]].mean():.4f}")
    print(f" Selected recon accuracy:   {results['recon_accuracy'][selected_indices].mean():.1f}%")
    print(f" Non-selected recon acc:    {results['recon_accuracy'][ranking[args.top_k:]].mean():.1f}%")
    print(f"{'='*60}")
    print(f" Output: {args.output_dir}")
    print(f"   - sequence_scores.csv    (all sequences with metrics)")
    print(f"   - selected_sequences.csv (top-{args.top_k} for fine-tuning)")
    print(f"   - selected_sequences.fa  (FASTA for fine-tuning)")
    print(f"   - entropy_histogram.png")
    print(f"   - entropy_vs_expression.png")
    print(f"   - top_uncertain_heatmap.png")
    print(f"   - selection_summary.png")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
