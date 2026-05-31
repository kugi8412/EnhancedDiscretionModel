#!/usr/bin/env python
"""
Crosscoder Sparse Autoencoder — Shared Dictionary Learning Across Models.

Implements the crosscoder framework from Lindsey et al. (2024)
"Sparse Crosscoders for Cross-Layer Features and Model Diffing" adapted
for comparing DNA sequence models on DeepSTARR data.

Key idea: Instead of training separate SAEs per model and correlating
activations post-hoc, train a SINGLE shared dictionary that encodes
activations from ALL models simultaneously. The loss naturally separates:
  - **Shared features**: decoder norms are similar across models
  - **Model-specific features**: decoder norm high in one model only

Additionally provides k-mer enrichment and motif analysis to biologically
interpret learned dictionary atoms.

Architecture (for M models):
    Encoder:  f(x) = ReLU( sum_m  W_enc_m @ a_m(x)  +  b_enc )
    Decoder:  a_m'(x) = W_dec_m @ f(x) + b_dec_m    (per model)

Loss:
    L = sum_m ||a_m - a_m'||^2  +  sum_i f_i * (sum_m ||W_dec_i_m||)
    (L1-of-norms: encourages layer/model sparsity per feature)

Usage:
    python crosscoder_sae.py --config config/Crosscoder.yaml
"""

import os
import sys
import argparse
from collections import defaultdict, Counter
from itertools import product as iter_product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
import yaml

from utils import load_config, prepare_input, set_global_seed
from models.registry import build_model


# ==========================================================================
# 1. CROSSCODER ARCHITECTURE
# ==========================================================================

class CrosscoderSAE(nn.Module):
    """Sparse Crosscoder that learns a shared dictionary across M models.

    Encoder reads from all models' representations simultaneously:
        f(x) = ReLU(sum_m W_enc_m @ a_m(x) + b_enc)

    Decoder reconstructs each model separately:
        a_m'(x) = W_dec_m @ f(x) + b_dec_m

    Loss uses L1-of-norms regularization which encourages features to be
    sparse across models (surfacing model-specific vs shared features).

    Parameters
    ----------
    input_dims : dict[str, int]
        Mapping from model name to feature dimension.
    hidden_dim : int
        Number of dictionary atoms (features) in the shared dictionary.
    l1_coeff : float
        L1 sparsity coefficient.
    normalize_inputs : bool
        Whether to normalize each model's input to unit variance before encoding.
    """

    def __init__(self, input_dims, hidden_dim=1024, l1_coeff=1e-3,
                 normalize_inputs=True):
        super().__init__()
        self.model_names = list(input_dims.keys())
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.l1_coeff = l1_coeff
        self.normalize_inputs = normalize_inputs
        self.n_models = len(self.model_names)

        # Per-model encoder weights
        self.encoders = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim, bias=False)
            for name, dim in input_dims.items()
        })
        # Shared encoder bias
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Per-model decoder weights + bias
        self.decoders = nn.ModuleDict({
            name: nn.Linear(hidden_dim, dim, bias=True)
            for name, dim in input_dims.items()
        })

        # Input normalization (learned per model)
        if normalize_inputs:
            self.input_scales = nn.ParameterDict({
                name: nn.Parameter(torch.ones(dim))
                for name, dim in input_dims.items()
            })
        else:
            self.input_scales = None

        # Initialize with tied weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize encoders/decoders with small random weights, tied."""
        for name in self.model_names:
            nn.init.xavier_uniform_(self.encoders[name].weight)
            nn.init.xavier_uniform_(self.decoders[name].weight)
            nn.init.zeros_(self.decoders[name].bias)

    def encode(self, activations):
        """Encode activations from all models into shared hidden space.

        Parameters
        ----------
        activations : dict[str, Tensor]
            {model_name: (batch, dim_m)} feature tensors.

        Returns
        -------
        Tensor of shape (batch, hidden_dim): sparse feature activations.
        """
        # Normalize inputs
        if self.normalize_inputs and self.input_scales is not None:
            activations = {
                name: act * self.input_scales[name].unsqueeze(0)
                for name, act in activations.items()
            }

        # Sum encoder contributions from all models
        h = self.encoder_bias.unsqueeze(0)  # (1, hidden_dim)
        for name in self.model_names:
            if name in activations:
                h = h + self.encoders[name](activations[name])

        # ReLU activation for sparsity
        return F.relu(h)

    def decode(self, f):
        """Decode hidden features back to each model's space.

        Parameters
        ----------
        f : Tensor
            (batch, hidden_dim) sparse feature activations.

        Returns
        -------
        dict[str, Tensor]: {model_name: reconstructed (batch, dim_m)}
        """
        return {
            name: self.decoders[name](f)
            for name in self.model_names
        }

    def forward(self, activations):
        """Full forward: encode -> decode.

        Returns (reconstructions_dict, hidden_activations).
        """
        f = self.encode(activations)
        recons = self.decode(f)
        return recons, f

    def loss(self, activations):
        """Compute crosscoder loss with L1-of-norms regularization.

        L = sum_m ||a_m - a_m'||^2
          + l1_coeff * sum_i f_i * (sum_m ||W_dec_i_m||_2)

        The per-feature penalty is weighted by the L1 norm of per-model
        decoder vector norms (not L2-of-norms), following the paper's
        recommendation for surfacing model-specific features.

        Returns
        -------
        total_loss, recon_loss, sparsity_loss : Tensor
        """
        recons, f = self.forward(activations)

        # Reconstruction loss (MSE per model, summed)
        recon_loss = torch.tensor(0.0, device=f.device)
        for name in self.model_names:
            if name in activations:
                recon_loss = recon_loss + F.mse_loss(
                    recons[name], activations[name], reduction='mean'
                )

        # L1-of-norms sparsity: f_i * sum_m ||W_dec_i_m||_2
        # W_dec shape per model: (dim_m, hidden_dim) → column i has norm
        decoder_norms_sum = torch.zeros(self.hidden_dim, device=f.device)
        for name in self.model_names:
            # Column norms of decoder weight: ||W_dec[:,i]||_2 for each feature i
            col_norms = self.decoders[name].weight.norm(dim=0)  # (hidden_dim,)
            decoder_norms_sum = decoder_norms_sum + col_norms

        # Weighted sparsity: mean over batch of f_i * decoder_norms_sum_i
        sparsity_loss = self.l1_coeff * (f * decoder_norms_sum.unsqueeze(0)).mean()

        total_loss = recon_loss + sparsity_loss
        return total_loss, recon_loss, sparsity_loss

    def get_feature_model_norms(self):
        """Get per-feature, per-model decoder norms for model diffing.

        Returns
        -------
        ndarray of shape (hidden_dim, n_models)
            Row i = feature i's decoder norm in each model.
        """
        norms = np.zeros((self.hidden_dim, self.n_models))
        with torch.no_grad():
            for j, name in enumerate(self.model_names):
                col_norms = self.decoders[name].weight.norm(dim=0).cpu().numpy()
                norms[:, j] = col_norms
        return norms

    def classify_features(self, threshold=0.1):
        """Classify features as shared or model-specific.

        A feature is "model-specific" if > (1-threshold) of its total decoder
        norm is concentrated in a single model. Otherwise it's "shared".

        Returns
        -------
        dict with keys:
            'shared': list of feature indices
            '{model_name}_specific': list of feature indices per model
            'norms': (hidden_dim, n_models) array
        """
        norms = self.get_feature_model_norms()
        total_norms = norms.sum(axis=1, keepdims=True).clip(min=1e-10)
        relative_norms = norms / total_norms

        result = {'shared': [], 'norms': norms}
        for j, name in enumerate(self.model_names):
            result[f'{name}_specific'] = []

        for i in range(self.hidden_dim):
            max_relative = relative_norms[i].max()
            if max_relative > (1.0 - threshold):
                # Model-specific
                best_model = self.model_names[relative_norms[i].argmax()]
                result[f'{best_model}_specific'].append(i)
            else:
                result['shared'].append(i)

        return result


# ==========================================================================
# 2. TRAINING
# ==========================================================================

def train_crosscoder(all_features, hidden_dim=1024, l1_coeff=1e-3,
                     epochs=300, lr=1e-3, batch_size=256, device='cpu',
                     normalize_inputs=True, log_interval=50):
    """Train a Crosscoder SAE on features from multiple models.

    Parameters
    ----------
    all_features : dict[str, Tensor]
        {model_name: (N, dim_m)} features extracted from validation data.
    hidden_dim : int
        Number of shared dictionary atoms.
    l1_coeff : float
        Sparsity coefficient.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Batch size.
    device : str
        Device to train on.

    Returns
    -------
    crosscoder : CrosscoderSAE
    history : dict with loss curves
    """
    input_dims = {name: feats.shape[1] for name, feats in all_features.items()}
    model_names = list(all_features.keys())
    N = all_features[model_names[0]].shape[0]

    crosscoder = CrosscoderSAE(
        input_dims, hidden_dim=hidden_dim, l1_coeff=l1_coeff,
        normalize_inputs=normalize_inputs
    ).to(device)

    optimizer = optim.Adam(crosscoder.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Create joint dataloader (same indices for all models)
    tensors = [all_features[name] for name in model_names]
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {'total': [], 'recon': [], 'sparsity': [], 'l0': []}

    for epoch in range(epochs):
        epoch_total, epoch_recon, epoch_sparse, epoch_l0 = 0., 0., 0., 0.
        n_batches = 0

        for batch_tensors in loader:
            batch_dict = {
                name: batch_tensors[i].to(device)
                for i, name in enumerate(model_names)
            }

            optimizer.zero_grad()
            total_loss, recon_loss, sparsity_loss = crosscoder.loss(batch_dict)
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(crosscoder.parameters(), 1.0)
            optimizer.step()

            # Normalize decoder columns to unit norm (per paper)
            with torch.no_grad():
                for name in model_names:
                    W = crosscoder.decoders[name].weight
                    norms = W.norm(dim=0, keepdim=True).clamp(min=1e-8)
                    W.div_(norms)

            # Track L0 (number of active features per sample)
            with torch.no_grad():
                f = crosscoder.encode(batch_dict)
                l0 = (f > 0).float().sum(dim=1).mean().item()

            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_sparse += sparsity_loss.item()
            epoch_l0 += l0
            n_batches += 1

        scheduler.step()

        history['total'].append(epoch_total / n_batches)
        history['recon'].append(epoch_recon / n_batches)
        history['sparsity'].append(epoch_sparse / n_batches)
        history['l0'].append(epoch_l0 / n_batches)

        if (epoch + 1) % log_interval == 0:
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"Loss: {history['total'][-1]:.5f} | "
                  f"Recon: {history['recon'][-1]:.5f} | "
                  f"Sparse: {history['sparsity'][-1]:.5f} | "
                  f"L0: {history['l0'][-1]:.1f}")

    return crosscoder, history


# ==========================================================================
# 3. K-MER AND MOTIF ANALYSIS
# ==========================================================================

def count_kmers(sequence, k=6):
    """Count k-mers in a DNA sequence string."""
    counts = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if 'N' not in kmer:
            counts[kmer] += 1
    return counts


def onehot_to_sequence(x):
    """Convert one-hot tensor (4, L) or (L, 4) to DNA string."""
    mapping = 'ACGT'
    if x.shape[0] == 4:
        indices = x.argmax(dim=0)
    else:
        indices = x.argmax(dim=1)
    return ''.join(mapping[i] for i in indices.tolist())


def compute_kmer_enrichment(crosscoder, model, dataloader, device,
                            model_name, k=6, top_features=50,
                            max_samples=2000):
    """Compute k-mer enrichment for top-activating features.

    For each of the top dictionary features (by mean activation), finds
    the sequences that maximally activate it and computes k-mer frequencies
    in the activating regions vs background.

    Parameters
    ----------
    crosscoder : CrosscoderSAE
    model : nn.Module (trained model to extract features from)
    dataloader : DataLoader yielding (X, Y_dev, Y_hk)
    device : torch.device
    model_name : str
    k : int
        k-mer length (default 6, captures most TF binding sites)
    top_features : int
        Number of features to analyze.
    max_samples : int
        Max sequences to process.

    Returns
    -------
    kmer_results : dict
        {feature_idx: {'top_kmers': [...], 'enrichment': {...}, ...}}
    """
    crosscoder.eval()
    model.eval()

    # Collect features and original sequences
    all_model_feats = []
    all_sequences = []
    n_collected = 0

    with torch.no_grad():
        for X_batch, _, _ in dataloader:
            X_batch = X_batch.to(device)
            feats = model.get_features(X_batch)
            all_model_feats.append(feats.cpu())
            # Store raw sequences for k-mer analysis
            for i in range(X_batch.shape[0]):
                all_sequences.append(onehot_to_sequence(X_batch[i].cpu()))
            n_collected += X_batch.shape[0]
            if n_collected >= max_samples:
                break

    model_feats = torch.cat(all_model_feats, dim=0)[:max_samples]
    all_sequences = all_sequences[:max_samples]

    # Get crosscoder hidden activations
    with torch.no_grad():
        activations = {model_name: model_feats.to(device)}
        hidden = crosscoder.encode(activations).cpu().numpy()  # (N, hidden_dim)

    # Select top features by mean activation
    mean_activation = hidden.mean(axis=0)
    top_feat_indices = np.argsort(mean_activation)[::-1][:top_features]

    # Background k-mer frequencies
    bg_kmers = Counter()
    for seq in all_sequences:
        bg_kmers.update(count_kmers(seq, k))
    total_bg = sum(bg_kmers.values()) or 1

    # Per-feature k-mer enrichment
    kmer_results = {}
    for feat_idx in top_feat_indices:
        feat_acts = hidden[:, feat_idx]

        # Top 10% activating sequences for this feature
        threshold = np.percentile(feat_acts[feat_acts > 0], 90) if (feat_acts > 0).sum() > 10 else 0
        top_seq_mask = feat_acts > max(threshold, 1e-6)

        if top_seq_mask.sum() < 5:
            continue

        # K-mer counts in top-activating sequences
        top_kmers = Counter()
        for idx in np.where(top_seq_mask)[0]:
            top_kmers.update(count_kmers(all_sequences[idx], k))
        total_top = sum(top_kmers.values()) or 1

        # Compute enrichment (log2 fold-change vs background)
        enrichment = {}
        for kmer, count in top_kmers.most_common(100):
            freq_top = count / total_top
            freq_bg = bg_kmers.get(kmer, 1) / total_bg
            enrichment[kmer] = np.log2(freq_top / freq_bg + 1e-10)

        # Sort by enrichment
        sorted_kmers = sorted(enrichment.items(), key=lambda x: -x[1])

        kmer_results[int(feat_idx)] = {
            'mean_activation': float(mean_activation[feat_idx]),
            'n_activating': int(top_seq_mask.sum()),
            'top_kmers': sorted_kmers[:20],
            'enrichment': dict(sorted_kmers[:50]),
        }

    return kmer_results


def compute_positional_importance(crosscoder, model, dataloader, device,
                                  model_name, top_features=20,
                                  max_samples=500):
    """Compute positional importance of input nucleotides per feature.

    Uses gradient-based attribution: for each feature, compute
    d(feature_activation) / d(input) to find which positions matter.

    Returns
    -------
    importance_maps : dict
        {feature_idx: ndarray of shape (4, seq_len)}
    """
    model.eval()
    crosscoder.eval()

    # Collect a batch
    X_all = []
    for X_batch, _, _ in dataloader:
        X_all.append(X_batch)
        if sum(x.shape[0] for x in X_all) >= max_samples:
            break
    X_all = torch.cat(X_all, dim=0)[:max_samples].to(device)

    # Get feature activations to find top features
    with torch.no_grad():
        model_feats = model.get_features(X_all)
        activations = {model_name: model_feats}
        hidden = crosscoder.encode(activations).cpu().numpy()

    mean_act = hidden.mean(axis=0)
    top_feat_indices = np.argsort(mean_act)[::-1][:top_features]

    importance_maps = {}

    for feat_idx in top_feat_indices:
        # Find top-activating samples for this feature
        feat_acts = hidden[:, feat_idx]
        top_samples = np.argsort(feat_acts)[-min(50, max_samples):]

        X_subset = X_all[top_samples].detach().requires_grad_(True)

        # Forward with gradient
        model_feats_grad = model.get_features(X_subset)
        act_dict = {model_name: model_feats_grad}
        h = crosscoder.encode(act_dict)

        # Gradient of this specific feature w.r.t. input
        target = h[:, feat_idx].sum()
        target.backward()

        grad = X_subset.grad.detach().cpu().numpy()  # (N_sub, 4, L)
        # Average absolute gradient across samples
        mean_grad = np.abs(grad).mean(axis=0)  # (4, L)
        importance_maps[int(feat_idx)] = mean_grad

        # Clear
        X_subset.grad = None

    return importance_maps


def generate_motif_pwm(crosscoder, model, dataloader, device,
                       model_name, feature_idx, max_samples=2000,
                       top_frac=0.1):
    """Generate Position Weight Matrix (PWM) for a feature.

    Finds sequences that maximally activate the feature, then computes
    nucleotide frequencies at each position of the activating region.

    Returns
    -------
    pwm : ndarray of shape (4, seq_len) — frequencies
    ic : ndarray of shape (seq_len,) — information content per position
    """
    model.eval()
    crosscoder.eval()

    X_all, seqs = [], []
    with torch.no_grad():
        for X_batch, _, _ in dataloader:
            X_all.append(X_batch)
            if sum(x.shape[0] for x in X_all) >= max_samples:
                break
    X_all = torch.cat(X_all, dim=0)[:max_samples]

    with torch.no_grad():
        model_feats = model.get_features(X_all.to(device))
        act_dict = {model_name: model_feats}
        hidden = crosscoder.encode(act_dict).cpu().numpy()

    feat_acts = hidden[:, feature_idx]
    n_top = max(10, int(len(feat_acts) * top_frac))
    top_idx = np.argsort(feat_acts)[-n_top:]

    # Get one-hot sequences for top-activating samples
    top_seqs = X_all[top_idx].numpy()  # (n_top, 4, L)

    # PWM = average one-hot (nucleotide frequency per position)
    pwm = top_seqs.mean(axis=0)  # (4, L)

    # Information content: 2 - H(position)
    # H = -sum(p * log2(p))
    eps = 1e-10
    H = -(pwm * np.log2(pwm + eps)).sum(axis=0)
    ic = 2.0 - H  # bits

    return pwm, ic


# ==========================================================================
# 4. VISUALIZATION
# ==========================================================================

def plot_training_history(history, save_path):
    """Plot crosscoder training curves."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].plot(history['total'], 'b-')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')

    axes[1].plot(history['recon'], 'g-')
    axes[1].set_title('Reconstruction Loss')
    axes[1].set_xlabel('Epoch')

    axes[2].plot(history['sparsity'], 'r-')
    axes[2].set_title('Sparsity Loss')
    axes[2].set_xlabel('Epoch')

    axes[3].plot(history['l0'], 'm-')
    axes[3].set_title('L0 (Active Features)')
    axes[3].set_xlabel('Epoch')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_model_diffing(crosscoder, save_path, threshold=0.1):
    """Plot feature decoder norms across models (the key crosscoder result).

    Shows trimodal distribution: shared, model_A-specific, model_B-specific.
    """
    norms = crosscoder.get_feature_model_norms()  # (hidden_dim, n_models)
    total = norms.sum(axis=1, keepdims=True).clip(min=1e-10)
    relative = norms / total

    n_models = len(crosscoder.model_names)

    if n_models == 2:
        # Classic 2-model diffing: histogram of relative norm
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        ratio = relative[:, 0]  # fraction belonging to model 0
        # Filter features with nonzero total norm
        active_mask = norms.sum(axis=1) > 0.01
        ratio_active = ratio[active_mask]

        axes[0].hist(ratio_active, bins=100, color='steelblue', edgecolor='white',
                     linewidth=0.3)
        axes[0].axvline(threshold, color='red', ls='--', alpha=0.6, label=f'threshold={threshold}')
        axes[0].axvline(1 - threshold, color='red', ls='--', alpha=0.6)
        axes[0].set_xlabel(f'Relative decoder norm in {crosscoder.model_names[0]}')
        axes[0].set_ylabel('Feature count')
        axes[0].set_title('Feature Distribution Across Models')
        axes[0].legend()

        # Classify and show counts
        classification = crosscoder.classify_features(threshold=threshold)
        n_shared = len(classification['shared'])
        labels, counts, colors = [], [], []
        labels.append('Shared')
        counts.append(n_shared)
        colors.append('#2ecc71')
        for name in crosscoder.model_names:
            n_specific = len(classification[f'{name}_specific'])
            labels.append(f'{name}\nSpecific')
            counts.append(n_specific)
            colors.append('#3498db' if name == crosscoder.model_names[0] else '#e74c3c')

        bars = axes[1].bar(labels, counts, color=colors, edgecolor='white')
        for bar, c in zip(bars, counts):
            axes[1].text(bar.get_x() + bar.get_width()/2, c + 1,
                         str(c), ha='center', fontweight='bold', fontsize=9)
        axes[1].set_ylabel('Feature count')
        axes[1].set_title('Shared vs Model-Specific Features')

        # Scatter of decoder norms
        axes[2].scatter(norms[active_mask, 0], norms[active_mask, 1],
                        s=8, alpha=0.3, c=ratio_active, cmap='coolwarm',
                        edgecolors='none')
        axes[2].set_xlabel(f'{crosscoder.model_names[0]} decoder norm')
        axes[2].set_ylabel(f'{crosscoder.model_names[1]} decoder norm')
        axes[2].set_title('Per-Feature Decoder Norms')
        axes[2].plot([0, norms.max()], [0, norms.max()], 'k--', alpha=0.3)

        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    else:
        # Multi-model: heatmap of relative norms
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Sort features by dominant model
        dominant = relative.argmax(axis=1)
        sort_idx = np.lexsort((relative.max(axis=1), dominant))

        im = axes[0].imshow(relative[sort_idx].T, aspect='auto',
                            cmap='hot', vmin=0, vmax=1)
        axes[0].set_yticks(range(n_models))
        axes[0].set_yticklabels(crosscoder.model_names, fontsize=9)
        axes[0].set_xlabel('Feature index (sorted by model dominance)')
        axes[0].set_title('Relative Decoder Norms Across Models')
        plt.colorbar(im, ax=axes[0])

        # Bar chart of classification
        classification = crosscoder.classify_features(threshold=threshold)
        labels = ['Shared']
        counts = [len(classification['shared'])]
        for name in crosscoder.model_names:
            labels.append(f'{name}')
            counts.append(len(classification[f'{name}_specific']))

        axes[1].bar(labels, counts, color=plt.cm.Set2(range(len(labels))),
                    edgecolor='white')
        axes[1].set_ylabel('Feature count')
        axes[1].set_title('Feature Classification')
        axes[1].tick_params(axis='x', rotation=45)

        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_kmer_enrichment(kmer_results, model_name, save_path, top_k=10):
    """Plot top enriched k-mers for each analyzed feature."""
    n_features = min(12, len(kmer_results))
    if n_features == 0:
        return

    feature_indices = list(kmer_results.keys())[:n_features]
    cols = min(4, n_features)
    rows = (n_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_features == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    for idx, feat_idx in enumerate(feature_indices):
        ax = axes[idx]
        info = kmer_results[feat_idx]
        top_kmers = info['top_kmers'][:top_k]

        if not top_kmers:
            ax.set_visible(False)
            continue

        kmers = [x[0] for x in top_kmers]
        enrichments = [x[1] for x in top_kmers]

        colors = ['#e74c3c' if e > 1 else '#3498db' for e in enrichments]
        ax.barh(range(len(kmers)), enrichments, color=colors,
                edgecolor='white', linewidth=0.3)
        ax.set_yticks(range(len(kmers)))
        ax.set_yticklabels(kmers, fontfamily='monospace', fontsize=8)
        ax.set_xlabel('log2 enrichment')
        ax.set_title(f'Feature {feat_idx}\n(n={info["n_activating"]}, '
                     f'act={info["mean_activation"]:.3f})', fontsize=9)
        ax.axvline(0, color='gray', ls='-', lw=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()

    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'K-mer Enrichment — {model_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_motif_logo(pwm, ic, feature_idx, save_path):
    """Plot a sequence logo from PWM and information content.

    Simple text-based logo: letter height = frequency * IC at that position.
    """
    seq_len = pwm.shape[1]
    # Only show the most informative 30bp window
    if seq_len > 30:
        # Find the window with highest total IC
        window = 30
        best_start = 0
        best_ic = 0
        for s in range(seq_len - window):
            w_ic = ic[s:s+window].sum()
            if w_ic > best_ic:
                best_ic = w_ic
                best_start = s
        pwm = pwm[:, best_start:best_start+window]
        ic = ic[best_start:best_start+window]
        seq_len = window

    fig, ax = plt.subplots(figsize=(max(8, seq_len * 0.3), 2.5))

    letters = 'ACGT'
    colors = {'A': '#2ecc71', 'C': '#3498db', 'G': '#f1c40f', 'T': '#e74c3c'}

    for pos in range(seq_len):
        # Sort letters by height (smallest first)
        heights = [(pwm[nt, pos] * ic[pos], letters[nt]) for nt in range(4)]
        heights.sort(key=lambda x: x[0])

        y_offset = 0
        for height, letter in heights:
            if height > 0.01:
                ax.text(pos + 0.5, y_offset + height/2, letter,
                        ha='center', va='center',
                        fontsize=max(6, int(height * 12)),
                        fontweight='bold', color=colors[letter],
                        fontfamily='monospace')
            y_offset += height

    ax.set_xlim(0, seq_len)
    ax.set_ylim(0, 2)
    ax.set_xlabel('Position')
    ax.set_ylabel('bits')
    ax.set_title(f'Feature {feature_idx} — Motif Logo')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_feature_expression_correlation(crosscoder, all_features, targets_dev,
                                        targets_hk, device, save_path):
    """Correlate crosscoder features with Dev/Hk expression across all models."""
    crosscoder.eval()

    with torch.no_grad():
        act_dict = {name: feats.to(device) for name, feats in all_features.items()}
        hidden = crosscoder.encode(act_dict).cpu().numpy()

    dev_np = targets_dev.numpy() if torch.is_tensor(targets_dev) else targets_dev
    hk_np = targets_hk.numpy() if torch.is_tensor(targets_hk) else targets_hk

    n_hidden = hidden.shape[1]
    dev_corrs = np.zeros(n_hidden)
    hk_corrs = np.zeros(n_hidden)

    for i in range(n_hidden):
        if np.std(hidden[:, i]) > 0:
            dev_corrs[i] = pearsonr(hidden[:, i], dev_np)[0]
            hk_corrs[i] = pearsonr(hidden[:, i], hk_np)[0]

    # Classification overlay
    classification = crosscoder.classify_features()

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot all
    ax.scatter(dev_corrs, hk_corrs, s=8, alpha=0.2, color='gray', label='All')

    # Highlight shared
    shared_idx = classification['shared']
    if shared_idx:
        ax.scatter(dev_corrs[shared_idx], hk_corrs[shared_idx],
                   s=15, alpha=0.5, color='#2ecc71', label='Shared', zorder=3)

    # Highlight model-specific
    colors_specific = plt.cm.tab10(range(crosscoder.n_models))
    for j, name in enumerate(crosscoder.model_names):
        spec_idx = classification[f'{name}_specific']
        if spec_idx:
            ax.scatter(dev_corrs[spec_idx], hk_corrs[spec_idx],
                       s=15, alpha=0.5, color=colors_specific[j],
                       label=f'{name}-specific', zorder=3)

    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.axvline(0, color='gray', ls='--', lw=0.5)
    ax.plot([-1, 1], [-1, 1], 'k--', lw=0.5, alpha=0.3)
    ax.set_xlabel('Correlation with Dev expression')
    ax.set_ylabel('Correlation with Hk expression')
    ax.set_title('Crosscoder Features × Expression')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_sparsity_comparison(crosscoder, all_features, device, save_path):
    """Compare activation sparsity patterns across models."""
    crosscoder.eval()
    model_names = crosscoder.model_names

    fig, axes = plt.subplots(1, len(model_names) + 1, figsize=(5 * (len(model_names) + 1), 4))

    # Joint encoding (all models together)
    with torch.no_grad():
        act_dict = {name: feats.to(device) for name, feats in all_features.items()}
        hidden_joint = crosscoder.encode(act_dict).cpu().numpy()

    axes[0].hist((hidden_joint > 0).sum(axis=1), bins=50, color='purple',
                 alpha=0.8, edgecolor='white')
    axes[0].set_xlabel('Active features per sample')
    axes[0].set_title(f'Joint (all models)\nMean L0={hidden_joint.mean(axis=0).sum():.0f}')

    # Per-model encoding (only one model at a time)
    for idx, name in enumerate(model_names):
        with torch.no_grad():
            single_dict = {name: all_features[name].to(device)}
            # Zero out other models' contributions
            hidden_single = crosscoder.encode(single_dict).cpu().numpy()

        axes[idx + 1].hist((hidden_single > 0).sum(axis=1), bins=50,
                           color=plt.cm.tab10(idx), alpha=0.8, edgecolor='white')
        axes[idx + 1].set_xlabel('Active features per sample')
        axes[idx + 1].set_title(f'{name} only\nMean L0={hidden_single.mean(axis=0).sum():.0f}')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Sparsity: Joint vs Single-Model Encoding', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ==========================================================================
# 5. FEATURE EXTRACTION FROM MODELS
# ==========================================================================

@torch.no_grad()
def extract_features_from_model(model, dataloader, device, max_samples=None):
    """Extract penultimate features from a trained model."""
    model.eval()
    all_features = []
    all_dev, all_hk = [], []
    n = 0

    for X_batch, Y_dev, Y_hk in dataloader:
        X_batch = X_batch.to(device)
        features = model.get_features(X_batch)
        all_features.append(features.cpu())
        all_dev.append(Y_dev)
        all_hk.append(Y_hk)
        n += X_batch.shape[0]
        if max_samples and n >= max_samples:
            break

    features = torch.cat(all_features, dim=0)
    if max_samples:
        features = features[:max_samples]
    return features, torch.cat(all_dev)[:features.shape[0]], torch.cat(all_hk)[:features.shape[0]]


# ==========================================================================
# 6. MAIN
# ==========================================================================

def load_crosscoder_config(path):
    """Load crosscoder YAML config."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Crosscoder SAE: Shared Dictionary Learning Across Models"
    )
    parser.add_argument('--config', type=str, required=True,
                        help="Path to Crosscoder YAML config")
    parser.add_argument('--skip_motifs', action='store_true',
                        help="Skip k-mer/motif analysis (faster)")
    parser.add_argument('--k', type=int, default=6,
                        help="k-mer length for enrichment analysis")
    args = parser.parse_args()

    cfg = load_crosscoder_config(args.config)

    # Parse config
    model_configs = cfg['models']
    configs = [m['config'] for m in model_configs]
    weights = [m['weights'] for m in model_configs]
    names = [m.get('name', None) for m in model_configs]
    data_config_path = cfg.get('data_config', configs[0])
    output_dir = cfg.get('output_dir', 'results/crosscoder')
    seed = cfg.get('seed', 42)

    cc_params = cfg.get('crosscoder', {})
    hidden_dim = cc_params.get('hidden_dim', 1024)
    l1_coeff = cc_params.get('l1_coeff', 1e-3)
    epochs = cc_params.get('epochs', 300)
    lr = cc_params.get('lr', 1e-3)
    batch_size = cc_params.get('batch_size', 256)
    normalize = cc_params.get('normalize_inputs', True)

    motif_params = cfg.get('motif_analysis', {})
    k = args.k or motif_params.get('k', 6)
    top_features = motif_params.get('top_features', 50)
    max_samples = motif_params.get('max_samples', 3000)

    set_global_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    # Resolve model names
    if names is None or all(n is None for n in names):
        names = [load_config(c)['model']['name'] for c in configs]

    # Dataloader
    data_config = load_config(data_config_path)
    val_loader = prepare_input('Val', data_config, shuffle=False)

    # ========================================
    # PHASE 1: Feature Extraction
    # ========================================
    print("=" * 60)
    print("PHASE 1: Feature Extraction from All Models")
    print("=" * 60)

    all_features = {}
    models = {}
    targets_dev, targets_hk = None, None

    for name, config_path, weight_path in zip(names, configs, weights):
        print(f"\n[{name}] Loading model from {weight_path}...")
        config = load_config(config_path)
        model = build_model(config).to(device)
        model.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )
        model.eval()

        features, t_dev, t_hk = extract_features_from_model(
            model, val_loader, device, max_samples=max_samples
        )
        all_features[name] = features
        models[name] = model  # keep for motif analysis
        targets_dev, targets_hk = t_dev, t_hk
        print(f"  [{name}] Features: {features.shape}")

    # ========================================
    # PHASE 2: Train Crosscoder
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 2: Training Crosscoder SAE")
    print(f"  Models: {names}")
    print(f"  Hidden dim: {hidden_dim}, L1: {l1_coeff}, Epochs: {epochs}")
    print("=" * 60)

    crosscoder, history = train_crosscoder(
        all_features, hidden_dim=hidden_dim, l1_coeff=l1_coeff,
        epochs=epochs, lr=lr, batch_size=batch_size, device=device,
        normalize_inputs=normalize,
    )

    # Save crosscoder
    cc_path = os.path.join(output_dir, 'crosscoder.pth')
    torch.save(crosscoder.state_dict(), cc_path)
    print(f"\nCrosscoder saved: {cc_path}")

    # ========================================
    # PHASE 3: Model Diffing
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 3: Model Diffing & Feature Classification")
    print("=" * 60)

    classification = crosscoder.classify_features()
    print(f"\n  Shared features: {len(classification['shared'])}")
    for name in names:
        n_spec = len(classification[f'{name}_specific'])
        print(f"  {name}-specific: {n_spec}")

    plot_training_history(history, os.path.join(output_dir, 'training.png'))
    plot_model_diffing(crosscoder, os.path.join(output_dir, 'model_diffing.png'))
    plot_feature_expression_correlation(
        crosscoder, all_features, targets_dev, targets_hk, device,
        os.path.join(output_dir, 'feature_expression.png')
    )
    plot_sparsity_comparison(
        crosscoder, all_features, device,
        os.path.join(output_dir, 'sparsity.png')
    )

    # ========================================
    # PHASE 4: K-mer & Motif Analysis
    # ========================================
    if not args.skip_motifs:
        print("\n" + "=" * 60)
        print("PHASE 4: K-mer Enrichment & Motif Analysis")
        print("=" * 60)

        for name in names:
            print(f"\n[{name}] Computing {k}-mer enrichment...")
            kmer_results = compute_kmer_enrichment(
                crosscoder, models[name], val_loader, device,
                model_name=name, k=k, top_features=top_features,
                max_samples=max_samples,
            )
            plot_kmer_enrichment(
                kmer_results, name,
                os.path.join(output_dir, f'kmer_{name}.png'),
            )

            # Save k-mer results
            kmer_save_path = os.path.join(output_dir, f'kmer_results_{name}.yaml')
            with open(kmer_save_path, 'w') as f:
                # Convert numpy types for yaml
                save_data = {}
                for feat_idx, info in kmer_results.items():
                    save_data[feat_idx] = {
                        'mean_activation': info['mean_activation'],
                        'n_activating': info['n_activating'],
                        'top_kmers': [(kmer, float(enrich))
                                      for kmer, enrich in info['top_kmers']],
                    }
                yaml.dump(save_data, f, default_flow_style=False)

            # Motif logos for top 5 features
            print(f"  [{name}] Generating motif logos...")
            for feat_idx in list(kmer_results.keys())[:5]:
                pwm, ic = generate_motif_pwm(
                    crosscoder, models[name], val_loader, device,
                    model_name=name, feature_idx=feat_idx,
                    max_samples=max_samples,
                )
                plot_motif_logo(
                    pwm, ic, feat_idx,
                    os.path.join(output_dir, f'motif_{name}_feat{feat_idx}.png'),
                )

    # ========================================
    # Summary
    # ========================================
    summary_path = os.path.join(output_dir, 'crosscoder_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Crosscoder SAE — Model Comparison Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Models compared: {names}\n")
        f.write(f"Hidden dimension: {hidden_dim}\n")
        f.write(f"L1 coefficient: {l1_coeff}\n")
        f.write(f"Training epochs: {epochs}\n")
        f.write(f"Final loss: {history['total'][-1]:.6f}\n")
        f.write(f"Final L0: {history['l0'][-1]:.1f}\n\n")

        f.write("Feature Classification:\n")
        f.write(f"  Shared: {len(classification['shared'])}\n")
        for name in names:
            f.write(f"  {name}-specific: {len(classification[f'{name}_specific'])}\n")
        f.write(f"\nFeature dimensions:\n")
        for name in names:
            f.write(f"  {name}: {all_features[name].shape[1]}\n")

    print(f"\nSummary: {summary_path}")
    print("Done.")


if __name__ == '__main__':
    main()
