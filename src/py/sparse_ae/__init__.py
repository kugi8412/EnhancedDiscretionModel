"""Sparse Autoencoder (SAE) pipeline for interpretable motif-feature analysis."""

from .model import SparseAutoencoder
from .train import train_sae, load_activations

__all__ = ["SparseAutoencoder", "train_sae", "load_activations"]
