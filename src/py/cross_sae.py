#!/usr/bin/env python
"""
Cross Sparse Autoencoder (CrossSAE) for comparing internal representations
across different trained DNA enhancer prediction models.

Supports two modes:

1. **Penultimate mode** (default): Train SAE on each model's ``get_features()``
   output and compare learned dictionary atoms.
2. **Multi-layer mode** (``--multilayer``): Extract activations from every named
   layer via forward hooks, compute CKA similarity across all layer pairs, and
   optionally train per-layer SAEs for deep comparison.

Can be driven by a YAML config (``--sae_config``) OR CLI flags.

Usage (penultimate):
    python cross_sae.py --sae_config config/CrossSAE.yaml

Usage (multi-layer):
    python cross_sae.py --sae_config config/CrossSAE_MultiLayer.yaml --multilayer
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


def resolve_topk(k, hidden_dim):
    """Resolve TopK value: if 0 < k < 1, treat as fraction of hidden_dim."""
    if isinstance(k, float) and 0.0 < k < 1.0:
        return max(1, int(k * hidden_dim))
    return int(k)


class SparseAutoencoder(nn.Module):
    """
    TopK Sparse Autoencoder — hard sparsity via top-k mask.
    Keeps exactly `k` largest activations per sample, zeros the rest.
    No L1 tuning required; sparsity = 1 - k/hidden_dim.
    """
    def __init__(self, input_dim, hidden_dim, k=32, l1_coeff=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = resolve_topk(k, hidden_dim)

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.encoder_bn = nn.BatchNorm1d(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Tied init
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

    def encode(self, x):
        pre = F.relu(self.encoder_bn(self.encoder(x)))
        topk_vals, topk_idx = pre.topk(self.k, dim=1)
        out = torch.zeros_like(pre)
        out.scatter_(1, topk_idx, topk_vals)
        return out

    def forward(self, x):
        h = self.encode(x)
        x_recon = self.decoder(h)
        return x_recon, h

    def loss(self, x):
        x_recon, h = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x)
        # No L1 needed — sparsity is structural via TopK
        return recon_loss, recon_loss, torch.tensor(0.0, device=x.device)


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
# 3b. HOOK-BASED MULTI-LAYER EXTRACTION
# ==========================================

class LayerwiseExtractor:
    """Capture activations from multiple named layers via forward hooks.

    Works with any ``nn.Module``.  Layers are identified by the dotted name
    returned by ``model.named_modules()`` (e.g. ``"seqextractor.blc0"``).

    Parameters
    ----------
    model : nn.Module
        The model to attach hooks to.
    layer_names : list[str] or None
        Specific layer names to capture.  If ``None``, every child module
        that contains at least one parameter is captured.
    """

    def __init__(self, model, layer_names=None):
        self.model = model
        self.activations = {}
        self._hooks = []

        if layer_names is None:
            layer_names = self._auto_select_layers(model)

        self.layer_names = layer_names
        name_to_module = dict(model.named_modules())
        for name in layer_names:
            if name not in name_to_module:
                print(f"  [WARNING] Layer '{name}' not found, skipping.")
                continue
            hook = name_to_module[name].register_forward_hook(
                self._make_hook(name)
            )
            self._hooks.append(hook)

    @staticmethod
    def _auto_select_layers(model):
        """Select layers that are leaf modules with parameters."""
        selected = []
        for name, module in model.named_modules():
            if name == '':
                continue
            children = list(module.children())
            has_params = any(True for _ in module.parameters(recurse=False))
            if not children and has_params:
                selected.append(name)
        # For very deep models, subsample to ≤ 30 layers
        if len(selected) > 30:
            step = max(1, len(selected) // 30)
            selected = selected[::step]
        return selected

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations[name] = output.detach()
        return hook_fn

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        self.activations.clear()


def _pool_activation(x):
    """Reduce an activation tensor to [B, D] via global average pooling.

    Handles 2D (B, D), 3D (B, C, L), and 4D (B, C, H, W) tensors.
    """
    if x.dim() == 2:
        return x
    elif x.dim() == 3:
        return x.mean(dim=2)
    elif x.dim() == 4:
        return x.mean(dim=(2, 3))
    else:
        return x.reshape(x.shape[0], -1)


@torch.no_grad()
def extract_layerwise_features(model, dataloader, device, layer_names=None,
                                max_samples=5000):
    """Extract activations from multiple layers of a model.

    Returns
    -------
    layer_features : dict[str, Tensor]
        Mapping from layer name to ``[N, D]`` feature matrix.
    layer_names : list[str]
        Ordered layer names that were actually captured.
    targets_dev, targets_hk : Tensor
        Expression targets.
    """
    model.eval()
    extractor = LayerwiseExtractor(model, layer_names)

    layer_accum = {n: [] for n in extractor.layer_names}
    all_dev, all_hk = [], []
    n_collected = 0

    for X_batch, Y_dev, Y_hk in dataloader:
        X_batch = X_batch.to(device)
        extractor.clear()

        # Trigger forward pass (we don't need the output)
        try:
            model(X_batch)
        except Exception:
            pass

        for name in extractor.layer_names:
            if name in extractor.activations:
                pooled = _pool_activation(extractor.activations[name]).cpu()
                layer_accum[name].append(pooled)

        all_dev.append(Y_dev)
        all_hk.append(Y_hk)
        n_collected += X_batch.shape[0]
        if n_collected >= max_samples:
            break

    extractor.remove_hooks()

    layer_features = {}
    valid_names = []
    for name in extractor.layer_names:
        if layer_accum[name]:
            layer_features[name] = torch.cat(layer_accum[name], dim=0)
            valid_names.append(name)

    targets_dev = torch.cat(all_dev, dim=0)[:n_collected]
    targets_hk = torch.cat(all_hk, dim=0)[:n_collected]

    return layer_features, valid_names, targets_dev, targets_hk


# ==========================================
# 3c. CKA (CENTERED KERNEL ALIGNMENT)
# ==========================================

def _centering_matrix_transform(K):
    """Apply centering: H @ K @ H where H = I - 1/n * 11^T."""
    n = K.shape[0]
    unit = torch.ones(n, n, device=K.device, dtype=K.dtype) / n
    return K - unit @ K - K @ unit + unit @ K @ unit


def linear_CKA(X, Y):
    """Compute linear CKA between two feature matrices.

    Parameters
    ----------
    X, Y : Tensor
        Feature matrices of shape ``[N, D1]`` and ``[N, D2]``.

    Returns
    -------
    float
        CKA similarity in ``[0, 1]``.
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # HSIC via linear kernel: HSIC(X,Y) = || Y^T X ||_F^2 / (n-1)^2
    YtX = Y.T @ X
    hsic_xy = (YtX * YtX).sum()

    XtX = X.T @ X
    hsic_xx = (XtX * XtX).sum()

    YtY = Y.T @ Y
    hsic_yy = (YtY * YtY).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy).clamp(min=1e-12)
    return (hsic_xy / denom).item()


def compute_cka_matrix(features_a, features_b, names_a, names_b,
                       max_samples=3000):
    """Compute pairwise CKA between all layers of two models.

    Returns
    -------
    cka_matrix : ndarray of shape ``[len(names_a), len(names_b)]``
    """
    n = min(max_samples, min(
        next(iter(features_a.values())).shape[0],
        next(iter(features_b.values())).shape[0],
    ))

    cka = np.zeros((len(names_a), len(names_b)))
    for i, na in enumerate(names_a):
        Xa = features_a[na][:n].float()
        for j, nb in enumerate(names_b):
            Xb = features_b[nb][:n].float()
            cka[i, j] = linear_CKA(Xa, Xb)
    return cka


# ==========================================
# 4. TRAIN SPARSE AUTOENCODER
# ==========================================

def train_sae(features, hidden_dim=512, l1_coeff=1e-3, k=32, epochs=200,
              lr=1e-3, batch_size=256, device='cpu'):
    """Train a TopK Sparse Autoencoder on extracted features."""
    input_dim = features.shape[1]
    sae = SparseAutoencoder(input_dim, hidden_dim, k=k).to(device)
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
# 6b. MULTI-LAYER VISUALIZATIONS
# ==========================================

def plot_cka_heatmap(cka_matrix, names_a, names_b, model_name_a, model_name_b,
                     save_path):
    """Cross-model CKA heatmap showing layer-to-layer similarity."""
    fig, ax = plt.subplots(figsize=(max(8, len(names_b) * 0.45),
                                    max(6, len(names_a) * 0.40)))

    im = ax.imshow(cka_matrix, cmap='magma', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Linear CKA')

    # Tick labels: use short names
    short_a = [n.split('.')[-1] if '.' in n else n for n in names_a]
    short_b = [n.split('.')[-1] if '.' in n else n for n in names_b]

    ax.set_xticks(range(len(names_b)))
    ax.set_xticklabels(short_b, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names_a)))
    ax.set_yticklabels(short_a, fontsize=7)

    ax.set_xlabel(model_name_b, fontsize=11)
    ax.set_ylabel(model_name_a, fontsize=11)
    ax.set_title(f'Linear CKA: {model_name_a} vs {model_name_b}', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cka_self(cka_matrix, layer_names, model_name, save_path):
    """Within-model CKA heatmap (self-similarity across layers)."""
    fig, ax = plt.subplots(figsize=(max(8, len(layer_names) * 0.4),
                                    max(6, len(layer_names) * 0.35)))

    im = ax.imshow(cka_matrix, cmap='magma', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Linear CKA')

    short = [n.split('.')[-1] if '.' in n else n for n in layer_names]
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(short, fontsize=7)

    ax.set_title(f'Self-CKA: {model_name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_layer_progression(cka_matrix, names_a, names_b,
                           model_name_a, model_name_b, save_path):
    """Plot the diagonal-like CKA progression (best-match per layer)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Best match: for each layer in A, which layer in B is most similar?
    best_b_for_a = np.argmax(cka_matrix, axis=1)
    best_cka_a = np.max(cka_matrix, axis=1)

    short_a = [n.split('.')[-1] if '.' in n else n for n in names_a]
    colors_a = plt.cm.magma(best_cka_a)

    axes[0].barh(range(len(names_a)), best_cka_a, color=colors_a)
    axes[0].set_yticks(range(len(names_a)))
    axes[0].set_yticklabels(short_a, fontsize=7)
    axes[0].set_xlabel('Best CKA')
    axes[0].set_title(f'{model_name_a} layers → best match in {model_name_b}')
    axes[0].set_xlim(0, 1)
    for i, (val, idx) in enumerate(zip(best_cka_a, best_b_for_a)):
        short_match = names_b[idx].split('.')[-1] if '.' in names_b[idx] else names_b[idx]
        axes[0].text(val + 0.01, i, short_match, va='center', fontsize=6)
    axes[0].invert_yaxis()

    # Best match: for each layer in B, which in A?
    best_a_for_b = np.argmax(cka_matrix, axis=0)
    best_cka_b = np.max(cka_matrix, axis=0)

    short_b = [n.split('.')[-1] if '.' in n else n for n in names_b]
    colors_b = plt.cm.magma(best_cka_b)

    axes[1].barh(range(len(names_b)), best_cka_b, color=colors_b)
    axes[1].set_yticks(range(len(names_b)))
    axes[1].set_yticklabels(short_b, fontsize=7)
    axes[1].set_xlabel('Best CKA')
    axes[1].set_title(f'{model_name_b} layers → best match in {model_name_a}')
    axes[1].set_xlim(0, 1)
    for i, (val, idx) in enumerate(zip(best_cka_b, best_a_for_b)):
        short_match = names_a[idx].split('.')[-1] if '.' in names_a[idx] else names_a[idx]
        axes[1].text(val + 0.01, i, short_match, va='center', fontsize=6)
    axes[1].invert_yaxis()

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(f'Layer Best-Match Progression', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_layerwise_sae_grid(sae_corr_results, model_name_a, model_name_b,
                            save_path):
    """Summary plot of per-layer-pair SAE comparison results."""
    if not sae_corr_results:
        return

    n = len(sae_corr_results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (pair_key, info) in enumerate(sae_corr_results.items()):
        ax = axes[idx]
        corr_matrix = info['corr_matrix']

        max_per_row = np.max(np.abs(corr_matrix), axis=1) if corr_matrix.size else np.array([])
        if max_per_row.size > 0:
            ax.hist(max_per_row, bins=30, color='steelblue', alpha=0.8,
                    edgecolor='white', linewidth=0.3)
        ax.axvline(0.7, color='red', ls='--', alpha=0.6)
        ax.set_title(pair_key, fontsize=8)
        ax.set_xlabel('Max |r|')
        ax.set_xlim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Per-Layer SAE Correlation: {model_name_a} vs {model_name_b}',
                 fontsize=12)
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
    parser.add_argument('--topk', type=float, default=32,
                        help='TopK: integer for absolute count, float <1 for fraction (e.g. 0.1 = 10%%)')
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
        topk = sae_params.get('topk', 32)
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
        topk = args.topk
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
            features, hidden_dim=hidden_dim, k=topk,
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


# ==========================================
# 9. MULTI-LAYER MAIN
# ==========================================

def main_multilayer():
    """Multi-layer comparison: CKA heatmaps + optional per-layer SAE analysis."""
    parser = argparse.ArgumentParser(
        description="Multi-Layer Cross Sparse Autoencoder Comparison"
    )
    parser.add_argument('--sae_config', type=str, required=True,
                        help="YAML config with model definitions")
    parser.add_argument('--multilayer', action='store_true')
    parser.add_argument('--layers', nargs='*', default=None,
                        help="Layer names to capture (default: auto-select)")
    parser.add_argument('--max_samples', type=int, default=5000,
                        help="Max samples for feature extraction")
    parser.add_argument('--sae_per_layer', action='store_true',
                        help="Train SAEs on top-CKA layer pairs")
    parser.add_argument('--top_k_pairs', type=int, default=5,
                        help="Number of top CKA layer pairs for SAE analysis")
    args = parser.parse_args()

    cfg = load_sae_config(args.sae_config)
    model_configs = cfg['models']
    configs = [m['config'] for m in model_configs]
    weights = [m['weights'] for m in model_configs]
    names = [m.get('name', None) for m in model_configs]
    data_config_path = cfg.get('data_config', configs[0])
    output_dir = cfg.get('output_dir', 'results/cross_sae_multilayer')
    seed = cfg.get('seed', 42)
    sae_params = cfg.get('sae', {})

    # Per-model layer overrides from config
    layer_overrides = {}
    for m in model_configs:
        if 'layers' in m:
            layer_overrides[m.get('name', m['config'])] = m['layers']

    set_global_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    if names is None or all(n is None for n in names):
        names = [load_config(c)['model']['name'] for c in configs]

    data_config = load_config(data_config_path)
    val_loader = prepare_input('Val', data_config, shuffle=False)

    # ---- PHASE 1: Multi-Layer Feature Extraction ----
    print("=" * 60)
    print("PHASE 1: Multi-Layer Feature Extraction")
    print("=" * 60)

    all_layer_features = {}  # name -> {layer_name -> Tensor}
    all_layer_names = {}     # name -> [layer_names]
    targets_dev, targets_hk = None, None

    for name, config_path, weight_path in zip(names, configs, weights):
        print(f"\n[{name}] Loading model...")
        config = load_config(config_path)
        model = build_model(config).to(device)
        model.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )
        model.eval()

        # Determine layer selection
        layers_for_model = (
            args.layers or
            layer_overrides.get(name, None) or
            None  # auto-detect
        )

        layer_feats, valid_names, t_dev, t_hk = extract_layerwise_features(
            model, val_loader, device,
            layer_names=layers_for_model,
            max_samples=args.max_samples,
        )
        all_layer_features[name] = layer_feats
        all_layer_names[name] = valid_names
        targets_dev, targets_hk = t_dev, t_hk

        print(f"[{name}] Captured {len(valid_names)} layers:")
        for ln in valid_names:
            print(f"    {ln}: {layer_feats[ln].shape}")

        del model
        torch.cuda.empty_cache()

    # ---- PHASE 2: Self-CKA (within each model) ----
    print("\n" + "=" * 60)
    print("PHASE 2: Self-CKA (within-model layer similarity)")
    print("=" * 60)

    for name in names:
        feats = all_layer_features[name]
        lnames = all_layer_names[name]
        cka_self = compute_cka_matrix(feats, feats, lnames, lnames)
        plot_cka_self(
            cka_self, lnames, name,
            os.path.join(output_dir, f'self_cka_{name}.png'),
        )

    # ---- PHASE 3: Cross-Model CKA ----
    print("\n" + "=" * 60)
    print("PHASE 3: Cross-Model CKA")
    print("=" * 60)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            feats_a = all_layer_features[na]
            feats_b = all_layer_features[nb]
            lnames_a = all_layer_names[na]
            lnames_b = all_layer_names[nb]

            print(f"\n--- CKA: {na} ({len(lnames_a)} layers) "
                  f"vs {nb} ({len(lnames_b)} layers) ---")

            cka = compute_cka_matrix(feats_a, feats_b, lnames_a, lnames_b)

            tag = f'{na}_vs_{nb}'
            plot_cka_heatmap(
                cka, lnames_a, lnames_b, na, nb,
                os.path.join(output_dir, f'cka_{tag}.png'),
            )
            plot_layer_progression(
                cka, lnames_a, lnames_b, na, nb,
                os.path.join(output_dir, f'progression_{tag}.png'),
            )

            # ---- PHASE 4 (optional): SAE on top-CKA pairs ----
            if args.sae_per_layer:
                print(f"\n  Training SAEs on top-{args.top_k_pairs} CKA pairs...")
                flat_idx = np.argsort(cka.flatten())[::-1][:args.top_k_pairs]
                rows, cols = np.unravel_index(flat_idx, cka.shape)

                sae_results = {}
                hidden_dim = sae_params.get('hidden_dim', 256)
                topk = sae_params.get('topk', 32)
                sae_epochs = sae_params.get('epochs', 100)
                sae_lr = sae_params.get('lr', 1e-3)
                sae_bs = sae_params.get('batch_size', 256)

                for r, c in zip(rows, cols):
                    layer_a = lnames_a[r]
                    layer_b = lnames_b[c]
                    cka_val = cka[r, c]
                    pair_key = (f"{layer_a.split('.')[-1]}"
                                f" ↔ {layer_b.split('.')[-1]}")
                    print(f"    [{pair_key}] CKA={cka_val:.3f}")

                    fa = feats_a[layer_a].float()
                    fb = feats_b[layer_b].float()

                    sae_a, _ = train_sae(
                        fa, hidden_dim=hidden_dim, k=topk,
                        epochs=sae_epochs, lr=sae_lr,
                        batch_size=sae_bs, device=device,
                    )
                    sae_b, _ = train_sae(
                        fb, hidden_dim=hidden_dim, k=topk,
                        epochs=sae_epochs, lr=sae_lr,
                        batch_size=sae_bs, device=device,
                    )

                    corr, act_a, act_b, h_a, h_b = \
                        compute_activation_correlation(
                            sae_a, sae_b, fa, fb, device=device,
                        )
                    n_shared = (np.abs(corr) > 0.7).sum()
                    print(f"      Active: {act_a.sum()}/{act_b.sum()}, "
                          f"Shared (|r|>0.7): {n_shared}")

                    sae_results[pair_key] = {
                        'corr_matrix': corr,
                        'cka': cka_val,
                        'n_shared': n_shared,
                    }

                plot_layerwise_sae_grid(
                    sae_results, na, nb,
                    os.path.join(output_dir, f'layer_sae_{tag}.png'),
                )

    # ---- Summary ----
    summary_path = os.path.join(output_dir, 'multilayer_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Multi-Layer Cross SAE Comparison Summary\n")
        f.write("=" * 50 + "\n\n")
        for name in names:
            lnames = all_layer_names[name]
            f.write(f"Model: {name}\n")
            f.write(f"  Layers captured: {len(lnames)}\n")
            for ln in lnames:
                feat = all_layer_features[name][ln]
                f.write(f"    {ln}: dim={feat.shape[1]}\n")
            f.write("\n")

    print(f"\nSummary: {summary_path}")
    print("Done.")


if __name__ == '__main__':
    # Check if --multilayer flag is present
    import sys
    if '--multilayer' in sys.argv:
        main_multilayer()
    else:
        main()
