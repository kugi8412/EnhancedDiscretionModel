"""Sparse Autoencoder (SAE) model definitions.

Two variants
------------
``SparseAutoencoder``
    L1-regularised autoencoder (Bricken et al. 2023 style).
    Training objective: MSE(x̂, x) + λ · mean(|f|₁)

``TopKSparseAutoencoder``
    Hard-sparsity variant that keeps only the top-k activations per sample
    (Conmy & Nanda 2024 style).  No L1 coefficient needed.
    Training objective: MSE(x̂, x)  with hard zeroing of all but k features.

Both variants
-------------
- Decoder column normalisation after every gradient step.
- Dead-feature detection and resampling utility.
- ``get_active_features(x, threshold)`` → boolean mask.
- ``feature_cosine_similarity()`` → off-diagonal cosine matrix (monosemanticity).

References
----------
.. Bricken et al., 2023, "Towards Monosemanticity"
.. Conmy & Nanda, 2024, "Activation Addition"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Single-layer sparse autoencoder.

    Parameters
    ----------
    input_dim : int
        Dimension of the frozen-model activations used as input.
    dict_size : int
        Number of learnable features (typically 4–16× ``input_dim``).
    l1_coeff : float
        L1 sparsity coefficient λ.
    """

    def __init__(self, input_dim: int, dict_size: int, l1_coeff: float = 1e-3):
        super().__init__()
        self.input_dim  = input_dim
        self.dict_size  = dict_size
        self.l1_coeff   = l1_coeff

        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)

        # Bias for pre-processing (replicate-of-mean centering trick)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        self._normalise_decoder()

    # ------------------------------------------------------------------

    def _normalise_decoder(self):
        """Project decoder column norms to ≤ 1 (in-place, no gradient)."""
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1.0)
            self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return sparse feature activations ``[B, dict_size]``."""
        x_c = x - self.pre_bias
        return F.relu(self.encoder(x_c))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from feature activations."""
        return self.decoder(features) + self.pre_bias

    def forward(self, x: torch.Tensor):
        """Full pass: encode then decode.

        Returns
        -------
        x_hat : torch.Tensor, same shape as *x*
            Reconstruction.
        features : torch.Tensor, shape ``[B, dict_size]``
            Sparse activations.
        """
        features = self.encode(x)
        x_hat    = self.decode(features)
        return x_hat, features

    def loss(self, x: torch.Tensor):
        """Compute combined reconstruction + sparsity loss.

        Returns
        -------
        total : scalar tensor
        recon : scalar tensor (MSE)
        l1    : scalar tensor (mean L1 norm)
        """
        x_hat, features = self.forward(x)
        recon = F.mse_loss(x_hat, x)
        l1    = features.abs().mean()
        total = recon + self.l1_coeff * l1
        return total, recon, l1

    def dead_feature_fraction(self, x: torch.Tensor, threshold: float = 1e-5) -> float:
        """Fraction of features that never activate above *threshold* on *x*."""
        with torch.no_grad():
            features = self.encode(x)
            active   = (features.max(dim=0).values > threshold)
        return 1.0 - active.float().mean().item()

    def get_active_features(self, x: torch.Tensor, threshold: float = 1e-5) -> torch.Tensor:
        """Boolean mask ``[dict_size]`` — True where feature activates on any sample."""
        with torch.no_grad():
            features = self.encode(x)
        return features.max(dim=0).values > threshold

    def feature_cosine_similarity(self) -> torch.Tensor:
        """Compute pairwise cosine similarity of decoder columns ``[dict_size, dict_size]``.

        Low off-diagonal values indicate monosemantic (non-redundant) features.
        """
        W = self.decoder.weight          # [input_dim, dict_size]
        W_n = F.normalize(W, dim=0)     # normalise along input dimension
        return W_n.T @ W_n              # [dict_size, dict_size]

    def resample_dead_features(
        self,
        activations: torch.Tensor,
        threshold: float = 1e-5,
        noise_scale: float = 0.2,
    ) -> int:
        """Re-initialise dead encoder rows from high-loss input samples.

        Parameters
        ----------
        activations : torch.Tensor, shape ``(N, input_dim)``
            A batch of activations to identify high-loss samples from.
        threshold : float
            Activation threshold below which a feature is considered dead.
        noise_scale : float
            Small Gaussian noise added to the resampled direction.

        Returns
        -------
        int
            Number of features resampled.
        """
        with torch.no_grad():
            feats     = self.encode(activations)
            is_dead   = (feats.max(dim=0).values <= threshold)  # [dict_size]
            n_dead    = int(is_dead.sum().item())
            if n_dead == 0:
                return 0

            # Rank samples by reconstruction loss
            x_hat = self.decode(feats)
            losses = F.mse_loss(x_hat, activations, reduction='none').mean(dim=1)  # [N]
            top_idx = torch.argsort(losses, descending=True)[:n_dead]
            candidates = activations[top_idx]              # [n_dead, input_dim]

            dead_idx = torch.where(is_dead)[0]             # [n_dead]
            new_dir  = F.normalize(candidates, dim=1)      # unit length
            noise    = torch.randn_like(new_dir) * noise_scale

            self.encoder.weight.data[dead_idx] = new_dir + noise
            self.encoder.bias.data[dead_idx]   = 0.0

        return n_dead


# ---------------------------------------------------------------------------
# TopK Sparse Autoencoder
# ---------------------------------------------------------------------------

class TopKSparseAutoencoder(nn.Module):
    """Hard-sparsity SAE: exactly *k* features are active per sample.

    Parameters
    ----------
    input_dim : int
        Dimension of the frozen-model activations.
    dict_size : int
        Total feature dictionary size (typically 8–32× ``input_dim``).
    k : int
        Number of active features per sample.
    """

    def __init__(self, input_dim: int, dict_size: int, k: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k         = k

        self.encoder  = nn.Linear(input_dim, dict_size)
        self.decoder  = nn.Linear(dict_size, input_dim, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        self._normalise_decoder()

    def _normalise_decoder(self):
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1.0)
            self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return top-k sparse activations ``[B, dict_size]``."""
        x_c    = x - self.pre_bias
        pre    = F.relu(self.encoder(x_c))       # [B, dict_size]
        topk_v, topk_i = pre.topk(self.k, dim=1)
        out    = torch.zeros_like(pre)
        out.scatter_(1, topk_i, topk_v)
        return out

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features) + self.pre_bias

    def forward(self, x: torch.Tensor):
        features = self.encode(x)
        x_hat    = self.decode(features)
        return x_hat, features

    def loss(self, x: torch.Tensor):
        """MSE reconstruction loss only (sparsity is hard-enforced)."""
        x_hat, features = self.forward(x)
        recon = F.mse_loss(x_hat, x)
        return recon, recon, torch.tensor(0.0, device=x.device)

    def dead_feature_fraction(self, x: torch.Tensor, threshold: float = 1e-5) -> float:
        with torch.no_grad():
            features = self.encode(x)
            active   = (features.max(dim=0).values > threshold)
        return 1.0 - active.float().mean().item()

    def get_active_features(self, x: torch.Tensor, threshold: float = 1e-5) -> torch.Tensor:
        with torch.no_grad():
            features = self.encode(x)
        return features.max(dim=0).values > threshold

    def feature_cosine_similarity(self) -> torch.Tensor:
        W   = self.decoder.weight
        W_n = F.normalize(W, dim=0)
        return W_n.T @ W_n

    def resample_dead_features(
        self,
        activations: torch.Tensor,
        threshold: float = 1e-5,
        noise_scale: float = 0.2,
    ) -> int:
        return SparseAutoencoder.resample_dead_features(
            self, activations, threshold, noise_scale)
