#!/usr/bin/env python
"""
Cross Sparse Autoencoder (CrossSAE) for comparing internal representations
across different trained DNA enhancer prediction models.

Trains a sparse autoencoder on each model's penultimate features, then
compares learned dictionary atoms. Handles models with different feature
dimensions via linear projection to a shared latent space.

Can be driven by a YAML config (--sae_config) OR CLI flags.

Usage:
    python cross_sae.py --sae_config config/CrossSAE.yaml
    python cross_sae.py \\
        --configs config1.yaml config2.yaml \\
        --weights weights1.pth weights2.pth \\
        --data_config base_config.yaml \\
        --output_dir results/cross_sae
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
import yaml

from utils import load_config, prepare_input, set_global_seed
from models.registry import build_model


# ==========================================
# 1. SPARSE AUTOENCODER
# ==========================================

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with L1 penalty on the hidden representation.
    Learns an overcomplete dictionary of features from model activations.
    """
    def __init__(self, input_dim, hidden_dim, l1_coeff=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coeff = l1_coeff

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Tied init
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

    def encode(self, x):
        return F.relu(self.encoder(x))

    def forward(self, x):
        h = self.encode(x)
        x_recon = self.decoder(h)
        return x_recon, h

    def loss(self, x):
        x_recon, h = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x)
        sparsity_loss = self.l1_coeff * h.abs().mean()
        return recon_loss + sparsity_loss, recon_loss, sparsity_loss


# ==========================================
# 2. PROJECTION LAYER (different dims)
# ==========================================

class FeatureProjector(nn.Module):
    """Project features to a shared dimension when models have different feature sizes."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        if in_dim == out_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
            )

    def forward(self, x):
        return self.proj(x)


# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================

@torch.no_grad()
def extract_features(model, dataloader, device):
    """Extract penultimate features from a trained model."""
    model.eval()
    all_features = []
    all_targets_dev = []
    all_targets_hk = []

    for X_batch, Y_dev, Y_hk in dataloader:
        X_batch = X_batch.to(device)
        features = model.get_features(X_batch)
        all_features.append(features.cpu())
        all_targets_dev.append(Y_dev)
        all_targets_hk.append(Y_hk)

    features = torch.cat(all_features, dim=0)
    targets_dev = torch.cat(all_targets_dev, dim=0)
    targets_hk = torch.cat(all_targets_hk, dim=0)

    return features, targets_dev, targets_hk


# ==========================================
# 4. TRAIN SPARSE AUTOENCODER
# ==========================================

def train_sae(features, hidden_dim=512, l1_coeff=1e-3, epochs=200,
              lr=1e-3, batch_size=256, device='cpu'):
    """Train a Sparse Autoencoder on extracted features."""
    input_dim = features.shape[1]
    sae = SparseAutoencoder(input_dim, hidden_dim, l1_coeff).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(features)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    history = {'total': [], 'recon': [], 'sparsity': []}

    for epoch in range(epochs):
        epoch_total, epoch_recon, epoch_sparsity = 0.0, 0.0, 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            total_loss, recon_loss, sparsity_loss = sae.loss(batch)
            total_loss.backward()
            optimizer.step()

            # Normalize decoder weights to unit norm
            with torch.no_grad():
                norms = sae.decoder.weight.norm(dim=0, keepdim=True)
                sae.decoder.weight.div_(norms.clamp(min=1e-8))

            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_sparsity += sparsity_loss.item()
            n_batches += 1

        history['total'].append(epoch_total / n_batches)
        history['recon'].append(epoch_recon / n_batches)
        history['sparsity'].append(epoch_sparsity / n_batches)

        if (epoch + 1) % 50 == 0:
            print(f"  SAE Epoch {epoch+1}/{epochs} | Loss: {history['total'][-1]:.6f} "
                  f"| Recon: {history['recon'][-1]:.6f} | Sparse: {history['sparsity'][-1]:.6f}")

    return sae, history


# ==========================================
# 5. CROSS-MODEL COMPARISON
# ==========================================

def compute_activation_correlation(sae_a, sae_b, features_a, features_b,
                                   device='cpu', activity_threshold=0.05):
    """
    Compare SAE activations on the same input data passed through different models.
    Handles different input_dim via FeatureProjector if SAE input dims differ.
    """
    sae_a.eval()
    sae_b.eval()

    with torch.no_grad():
        h_a = sae_a.encode(features_a.to(device)).cpu().numpy()
        h_b = sae_b.encode(features_b.to(device)).cpu().numpy()

    # Active features: non-zero for > threshold fraction of samples
    active_a = (h_a > 0).mean(axis=0) > activity_threshold
    active_b = (h_b > 0).mean(axis=0) > activity_threshold

    h_a_active = h_a[:, active_a]
    h_b_active = h_b[:, active_b]

    n_a, n_b = h_a_active.shape[1], h_b_active.shape[1]
    corr_matrix = np.zeros((n_a, n_b))

    for i in range(n_a):
        for j in range(n_b):
            std_a = np.std(h_a_active[:, i])
            std_b = np.std(h_b_active[:, j])
            if std_a > 0 and std_b > 0:
                corr_matrix[i, j] = pearsonr(
                    h_a_active[:, i], h_b_active[:, j]
                )[0]

    return corr_matrix, active_a, active_b, h_a, h_b


def compute_feature_expression_correlation(hidden_activations, targets_dev, targets_hk):
    """Correlate each SAE hidden unit with expression targets."""
    n_hidden = hidden_activations.shape[1]
    dev_corrs = np.zeros(n_hidden)
    hk_corrs = np.zeros(n_hidden)
    dev_np = targets_dev.numpy() if torch.is_tensor(targets_dev) else targets_dev
    hk_np = targets_hk.numpy() if torch.is_tensor(targets_hk) else targets_hk

    for i in range(n_hidden):
        h_i = hidden_activations[:, i]
        if np.std(h_i) > 0:
            dev_corrs[i] = pearsonr(h_i, dev_np)[0]
            hk_corrs[i] = pearsonr(h_i, hk_np)[0]
    return dev_corrs, hk_corrs


# ==========================================
# 6. VISUALIZATION
# ==========================================

def plot_training_curves(histories, names, save_path):
    """Plot SAE training loss curves for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    titles = ['Total Loss', 'Reconstruction Loss', 'Sparsity Loss']
    keys = ['total', 'recon', 'sparsity']

    for ax, title, key in zip(axes, titles, keys):
        for name, hist in zip(names, histories):
            ax.plot(hist[key], label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_activation_histograms(hidden_acts_dict, save_path):
    """Plot activation value distributions for each model's SAE."""
    n = len(hidden_acts_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, h) in zip(axes, hidden_acts_dict.items()):
        nonzero = h[h > 0].flatten()
        ax.hist(nonzero, bins=100, color='steelblue', alpha=0.8,
                edgecolor='white', linewidth=0.3)
        sparsity = (h == 0).mean() * 100
        ax.set_title(f'{name}\nSparsity: {sparsity:.1f}%')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Count')
        ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('SAE Activation Distributions (non-zero only)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_expression_scatter(dev_corrs, hk_corrs, name, save_path):
    """Scatter of each SAE feature's correlation with Dev vs Hk expression."""
    fig, ax = plt.subplots(figsize=(7, 6))

    active = (np.abs(dev_corrs) > 0.05) | (np.abs(hk_corrs) > 0.05)
    ax.scatter(dev_corrs[~active], hk_corrs[~active], s=10, alpha=0.3,
               color='gray', label='Inactive')
    ax.scatter(dev_corrs[active], hk_corrs[active], s=15, alpha=0.6,
               c=dev_corrs[active] - hk_corrs[active], cmap='RdBu_r',
               edgecolors='none')

    ax.axhline(0, color='gray', ls='--', lw=0.7, alpha=0.5)
    ax.axvline(0, color='gray', ls='--', lw=0.7, alpha=0.5)
    ax.plot([-1, 1], [-1, 1], 'k--', lw=0.5, alpha=0.3)

    ax.set_xlabel('Correlation with Dev expression')
    ax.set_ylabel('Correlation with Hk expression')
    ax.set_title(f'{name} — SAE Feature × Expression Correlation')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison(corr_matrix, name_a, name_b, save_path,
                    title="Activation Correlation"):
    """Heatmap + histogram of cross-model activation correlations."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    im = axes[0].imshow(corr_matrix, aspect='auto', cmap='RdBu_r',
                         vmin=-1, vmax=1)
    axes[0].set_xlabel(f'{name_b} features')
    axes[0].set_ylabel(f'{name_a} features')
    axes[0].set_title(title)
    plt.colorbar(im, ax=axes[0])

    max_sim_a = np.max(np.abs(corr_matrix), axis=1)
    max_sim_b = np.max(np.abs(corr_matrix), axis=0)

    axes[1].hist(max_sim_a, bins=50, alpha=0.7,
                  label=f'{name_a} → {name_b}', color='steelblue')
    axes[1].hist(max_sim_b, bins=50, alpha=0.7,
                  label=f'{name_b} → {name_a}', color='coral')
    axes[1].set_xlabel('Max |Correlation|')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Best-Match Distribution')
    axes[1].legend()
    axes[1].axvline(0.8, color='gray', ls='--', alpha=0.5)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_shared_features(corr_matrix, name_a, name_b, save_path, threshold=0.7):
    """Bar chart of shared vs unique features."""
    high_corr_pairs = np.argwhere(np.abs(corr_matrix) > threshold)

    n_shared = len(high_corr_pairs)
    n_unique_a = (corr_matrix.shape[0] - len(set(high_corr_pairs[:, 0]))
                  if n_shared > 0 else corr_matrix.shape[0])
    n_unique_b = (corr_matrix.shape[1] - len(set(high_corr_pairs[:, 1]))
                  if n_shared > 0 else corr_matrix.shape[1])

    fig, ax = plt.subplots(figsize=(8, 6))
    categories = [f'Shared\n(|r|>{threshold})', f'{name_a}\nUnique', f'{name_b}\nUnique']
    counts = [n_shared, n_unique_a, n_unique_b]
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    bars = ax.bar(categories, counts, color=colors, edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.5,
                str(count), ha='center', fontweight='bold')

    ax.set_ylabel('Number of Features')
    ax.set_title(f'Shared vs Unique SAE Features: {name_a} vs {name_b}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_top_matched_features(corr_matrix, name_a, name_b,
                              h_a, h_b, save_path, top_k=8):
    """Show activation profiles of the top-k most correlated feature pairs."""
    # Find top-k pairs by absolute correlation
    flat_idx = np.argsort(np.abs(corr_matrix).flatten())[::-1][:top_k]
    rows, cols = np.unravel_index(flat_idx, corr_matrix.shape)

    fig, axes = plt.subplots(2, top_k // 2, figsize=(4 * (top_k // 2), 8))
    axes = axes.flatten()

    for idx, (r, c) in enumerate(zip(rows, cols)):
        if idx >= len(axes):
            break
        ax = axes[idx]
        corr_val = corr_matrix[r, c]
        ax.scatter(h_a[:, r], h_b[:, c], s=4, alpha=0.3, rasterized=True)
        ax.set_xlabel(f'{name_a} feat {r}')
        ax.set_ylabel(f'{name_b} feat {c}')
        ax.set_title(f'r={corr_val:.3f}', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(f'Top-{top_k} Matched Feature Pairs', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ==========================================
# 7. CONFIG LOADING
# ==========================================

def load_sae_config(path):
    """Load a CrossSAE YAML config."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ==========================================
# 8. MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross Sparse Autoencoder for Model Comparison"
    )
    # YAML-based config (preferred)
    parser.add_argument('--sae_config', type=str, default=None,
                        help="Path to CrossSAE YAML config (overrides CLI flags)")
    # CLI flags (fallback)
    parser.add_argument('--configs', nargs='+', default=None,
                        help="YAML config paths for each model")
    parser.add_argument('--weights', nargs='+', default=None,
                        help="Weight paths for each model")
    parser.add_argument('--names', nargs='+', default=None,
                        help="Model display names")
    parser.add_argument('--data_config', type=str, default=None,
                        help="Base data config for shared dataloader")
    parser.add_argument('--output_dir', type=str, default='results/cross_sae')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--l1_coeff', type=float, default=1e-3)
    parser.add_argument('--sae_epochs', type=int, default=200)
    parser.add_argument('--sae_lr', type=float, default=1e-3)
    parser.add_argument('--sae_batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # If a YAML config is provided, it overrides CLI
    if args.sae_config:
        cfg = load_sae_config(args.sae_config)
        model_configs = cfg['models']
        configs = [m['config'] for m in model_configs]
        weights = [m['weights'] for m in model_configs]
        names = [m.get('name', None) for m in model_configs]
        data_config_path = cfg.get('data_config', configs[0])
        output_dir = cfg.get('output_dir', 'results/cross_sae')
        sae_params = cfg.get('sae', {})
        hidden_dim = sae_params.get('hidden_dim', 512)
        l1_coeff = sae_params.get('l1_coeff', 1e-3)
        sae_epochs = sae_params.get('epochs', 200)
        sae_lr = sae_params.get('lr', 1e-3)
        sae_batch_size = sae_params.get('batch_size', 256)
        seed = cfg.get('seed', 42)
    else:
        assert args.configs and args.weights, \
            "Provide --sae_config OR both --configs and --weights"
        configs = args.configs
        weights = args.weights
        names = args.names
        data_config_path = args.data_config or configs[0]
        output_dir = args.output_dir
        hidden_dim = args.hidden_dim
        l1_coeff = args.l1_coeff
        sae_epochs = args.sae_epochs
        sae_lr = args.sae_lr
        sae_batch_size = args.sae_batch_size
        seed = args.seed

    assert len(configs) == len(weights), "Must provide equal configs and weights"
    assert len(configs) >= 2, "Need at least 2 models to compare"

    set_global_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    # Resolve model names
    if names is None or all(n is None for n in names):
        names = [load_config(c)['model']['name'] for c in configs]

    # Shared dataloader
    data_config = load_config(data_config_path)
    val_loader = prepare_input('Val', data_config, shuffle=False)

    # ---- PHASE 1: Feature Extraction ----
    print("=" * 60)
    print("PHASE 1: Feature Extraction")
    print("=" * 60)

    all_features = {}
    targets_dev, targets_hk = None, None
    feature_dims = {}

    for name, config_path, weight_path in zip(names, configs, weights):
        print(f"\n[{name}] Loading model...")
        config = load_config(config_path)
        model = build_model(config).to(device)
        model.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )
        model.eval()

        features, t_dev, t_hk = extract_features(model, val_loader, device)
        all_features[name] = features
        feature_dims[name] = features.shape[1]
        targets_dev, targets_hk = t_dev, t_hk

        print(f"[{name}] Feature shape: {features.shape}")
        del model
        torch.cuda.empty_cache()

    # ---- Project features to shared dim if needed ----
    dims = list(feature_dims.values())
    shared_dim = max(dims)  # project smaller → larger
    projectors = {}

    if len(set(dims)) > 1:
        print(f"\nFeature dimensions differ: {feature_dims}")
        print(f"Projecting all to shared_dim={shared_dim}")
        for name in names:
            proj = FeatureProjector(feature_dims[name], shared_dim).to(device)
            # Quick train of projector: minimize reconstruction after PCA
            if feature_dims[name] != shared_dim:
                opt = optim.Adam(proj.parameters(), lr=1e-3)
                feats = all_features[name].to(device)
                for _ in range(100):
                    opt.zero_grad()
                    out = proj(feats)
                    # identity-like loss: projected features should preserve distances
                    loss = F.mse_loss(out[:, :feature_dims[name]], feats)
                    loss.backward()
                    opt.step()
                proj.eval()
                with torch.no_grad():
                    all_features[name] = proj(feats).cpu()
                print(f"  [{name}] Projected {feature_dims[name]} → {shared_dim}")
            projectors[name] = proj

    # ---- PHASE 2: Train SAEs ----
    print("\n" + "=" * 60)
    print("PHASE 2: Training Sparse Autoencoders")
    print("=" * 60)

    saes = {}
    histories = []
    hidden_acts = {}

    for name in names:
        features = all_features[name]
        in_dim = features.shape[1]
        print(f"\n[{name}] Training SAE (in={in_dim}, hidden={hidden_dim})...")
        sae, history = train_sae(
            features, hidden_dim=hidden_dim, l1_coeff=l1_coeff,
            epochs=sae_epochs, lr=sae_lr, batch_size=sae_batch_size,
            device=device,
        )
        saes[name] = sae
        histories.append(history)

        # Cache hidden activations
        sae.eval()
        with torch.no_grad():
            hidden_acts[name] = sae.encode(features.to(device)).cpu().numpy()

        sae_path = os.path.join(output_dir, f'sae_{name}.pth')
        torch.save(sae.state_dict(), sae_path)

    # ---- Visualizations per model ----
    plot_training_curves(
        histories, names,
        os.path.join(output_dir, 'training_curves.png'),
    )
    plot_activation_histograms(
        hidden_acts,
        os.path.join(output_dir, 'activation_histograms.png'),
    )

    for name in names:
        dev_corrs, hk_corrs = compute_feature_expression_correlation(
            hidden_acts[name], targets_dev, targets_hk,
        )
        plot_feature_expression_scatter(
            dev_corrs, hk_corrs, name,
            os.path.join(output_dir, f'feature_expression_{name}.png'),
        )

    # ---- PHASE 3: Pairwise comparison ----
    print("\n" + "=" * 60)
    print("PHASE 3: Cross-Model Comparison")
    print("=" * 60)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            print(f"\n--- Comparing {name_a} vs {name_b} ---")

            sae_a, sae_b = saes[name_a], saes[name_b]
            feat_a, feat_b = all_features[name_a], all_features[name_b]

            corr_matrix, active_a, active_b, h_a, h_b = \
                compute_activation_correlation(
                    sae_a, sae_b, feat_a, feat_b, device=device,
                )

            print(f"  Active: {name_a}={active_a.sum()}, {name_b}={active_b.sum()}")
            n_shared = (np.abs(corr_matrix) > 0.7).sum()
            print(f"  Highly correlated pairs (|r| > 0.7): {n_shared}")

            tag = f'{name_a}_vs_{name_b}'
            plot_comparison(
                corr_matrix, name_a, name_b,
                os.path.join(output_dir, f'correlation_{tag}.png'),
            )
            plot_shared_features(
                corr_matrix, name_a, name_b,
                os.path.join(output_dir, f'shared_features_{tag}.png'),
            )
            plot_top_matched_features(
                corr_matrix, name_a, name_b,
                h_a[:, active_a], h_b[:, active_b],
                os.path.join(output_dir, f'top_matches_{tag}.png'),
            )

    # ---- Summary report ----
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Cross Sparse Autoencoder Comparison Summary\n")
        f.write("=" * 50 + "\n\n")
        for name in names:
            feat = all_features[name]
            h = hidden_acts[name]
            sparsity = (h == 0).mean() * 100
            f.write(f"Model: {name}\n")
            f.write(f"  Original feature dim: {feature_dims[name]}\n")
            f.write(f"  Feature mean: {feat.mean():.4f}, std: {feat.std():.4f}\n")
            f.write(f"  SAE sparsity: {sparsity:.1f}%\n")
            f.write(f"  Active features (>5%): {(h > 0).mean(axis=0).sum():.0f}/{hidden_dim}\n\n")

    print(f"\nSummary: {summary_path}")
    print("Done.")


if __name__ == '__main__':
    main()
