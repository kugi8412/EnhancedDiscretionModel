#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ensemble prediction with Monte Carlo Dropout uncertainty estimation.

Supports two complementary uncertainty strategies that can be combined:

**Multi-model ensemble**
    Loads N independently trained models (each with its own YAML config and
    weight file).  Prediction mean and variance are computed over model
    outputs.  Works for any model registered in ``models.registry``.

**Monte Carlo Dropout (MC Dropout)**
    Runs K stochastic forward passes through a single model with dropout
    layers kept in training mode.  Variance over passes approximates the
    epistemic uncertainty.  No re-training is required.

Both methods can be applied jointly: MC Dropout is run on every ensemble
member, and the resulting distributions are pooled before computing
aggregate statistics.

Usage
-----
From the command line::

    python ensemble.py \\
        --configs config/DeepSTARR.yaml config/LegNetPlus.yaml \\
        --weights train_logs/ds.pth       train_logs/lg.pth \\
        --fasta   ../../data/deepSTARR/Sequences_Test.fa \\
        --output  outputs/ensemble/results.tsv

From Python::

    predictor = EnsemblePredictor.from_configs(
        [("config/DeepSTARR.yaml", "train_logs/ds.pth"),
         ("config/LegNet.yaml",    "train_logs/lg.pth")],
        mc_passes=30,
    )
    results = predictor.predict(sequences)
    flagged = predictor.flag_uncertain(results, var_threshold=0.5)
"""

import argparse
import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from Bio import SeqIO

# ---------------------------------------------------------------------------
# Project imports (work from src/py/)
# ---------------------------------------------------------------------------
from models.registry import build_model
from utils import load_config, one_hot_encode_dna


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Container for ensemble prediction outputs.

    Attributes
    ----------
    sequences : list[str]
        Original DNA strings used for prediction.
    mean_dev : np.ndarray, shape (N,)
        Mean predicted developmental log2 enrichment.
    mean_hk : np.ndarray, shape (N,)
        Mean predicted housekeeping log2 enrichment.
    var_dev : np.ndarray, shape (N,)
        Predictive variance for developmental activity.
    var_hk : np.ndarray, shape (N,)
        Predictive variance for housekeeping activity.
    std_dev : np.ndarray, shape (N,)
        Predictive standard deviation for developmental activity.
    std_hk : np.ndarray, shape (N,)
        Predictive standard deviation for housekeeping activity.
    ci95_dev : np.ndarray, shape (N, 2)
        95 % confidence interval [lower, upper] for developmental activity.
    ci95_hk : np.ndarray, shape (N, 2)
        95 % confidence interval [lower, upper] for housekeeping activity.
    n_models : int
        Total number of stochastic forward passes pooled.
    source : str
        Description of uncertainty source ('ensemble', 'mc_dropout', or
        'ensemble+mc_dropout').
    """

    sequences: List[str]
    mean_dev: np.ndarray
    mean_hk: np.ndarray
    var_dev: np.ndarray
    var_hk: np.ndarray
    std_dev: np.ndarray
    std_hk: np.ndarray
    ci95_dev: np.ndarray
    ci95_hk: np.ndarray
    n_models: int
    source: str

    def to_dataframe(self) -> pd.DataFrame:
        """Export results as a pandas DataFrame."""
        return pd.DataFrame({
            "sequence":  self.sequences,
            "mean_dev":  self.mean_dev,
            "mean_hk":   self.mean_hk,
            "var_dev":   self.var_dev,
            "var_hk":    self.var_hk,
            "std_dev":   self.std_dev,
            "std_hk":    self.std_hk,
            "ci95_dev_low":  self.ci95_dev[:, 0],
            "ci95_dev_high": self.ci95_dev[:, 1],
            "ci95_hk_low":   self.ci95_hk[:, 0],
            "ci95_hk_high":  self.ci95_hk[:, 1],
        })

    def flag_uncertain(
        self,
        var_threshold_dev: float = 0.5,
        var_threshold_hk: float = 0.5,
    ) -> np.ndarray:
        """Return a boolean mask of sequences flagged as uncertain.

        A sequence is flagged when its variance in *either* developmental or
        housekeeping activity exceeds the corresponding threshold.

        Parameters
        ----------
        var_threshold_dev : float
            Variance threshold for developmental activity.
        var_threshold_hk : float
            Variance threshold for housekeeping activity.

        Returns
        -------
        np.ndarray, shape (N,), dtype bool
            ``True`` for sequences with high predictive uncertainty.
        """
        return (self.var_dev > var_threshold_dev) | (self.var_hk > var_threshold_hk)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enable_mc_dropout(model: nn.Module) -> None:
    """Set all ``nn.Dropout`` layers (and variants) to training mode."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train()


def _model_predict(
    model: nn.Module,
    X: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a single deterministic forward pass over *X* in mini-batches.

    Parameters
    ----------
    model : nn.Module
        Model in evaluation mode.
    X : torch.Tensor, shape (N, 4, L)
        One-hot encoded sequences (on CPU — batches are moved to *device*).
    batch_size : int
        Mini-batch size.
    device : torch.device
        Target device.

    Returns
    -------
    preds_dev, preds_hk : np.ndarray, shape (N,)
    """
    preds_dev, preds_hk = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = X[start:start + batch_size].to(device)
            out = model(batch)
            if isinstance(out, (list, tuple)):
                d, h = out[0], out[1]
            else:
                d, h = out[:, 0:1], out[:, 1:2]
            preds_dev.append(d.squeeze(-1).cpu().numpy())
            preds_hk.append(h.squeeze(-1).cpu().numpy())
    return np.concatenate(preds_dev), np.concatenate(preds_hk)


def _mc_predict(
    model: nn.Module,
    X: torch.Tensor,
    n_passes: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run *n_passes* stochastic MC Dropout forward passes.

    Returns arrays of shape ``(n_passes, N)`` for pointwise statistics
    across passes.
    """
    model.eval()
    _enable_mc_dropout(model)
    all_dev = np.zeros((n_passes, X.shape[0]))
    all_hk  = np.zeros((n_passes, X.shape[0]))
    with torch.no_grad():
        for k in range(n_passes):
            d, h = _model_predict(model, X, batch_size, device)
            all_dev[k] = d
            all_hk[k]  = h
    return all_dev, all_hk


def _pool_statistics(
    all_dev: np.ndarray,
    all_hk: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Compute mean, variance, std, and 95 % CI from a ``(K, N)`` array."""
    mean_dev = all_dev.mean(axis=0)
    mean_hk  = all_hk.mean(axis=0)
    var_dev  = all_dev.var(axis=0, ddof=1) if all_dev.shape[0] > 1 else np.zeros(all_dev.shape[1])
    var_hk   = all_hk.var(axis=0,  ddof=1) if all_hk.shape[0]  > 1 else np.zeros(all_hk.shape[1])
    std_dev  = np.sqrt(var_dev)
    std_hk   = np.sqrt(var_hk)
    ci95_dev = np.stack([
        np.percentile(all_dev, 2.5,  axis=0),
        np.percentile(all_dev, 97.5, axis=0),
    ], axis=1)
    ci95_hk = np.stack([
        np.percentile(all_hk, 2.5,  axis=0),
        np.percentile(all_hk, 97.5, axis=0),
    ], axis=1)
    return mean_dev, mean_hk, var_dev, var_hk, std_dev, std_hk, ci95_dev, ci95_hk


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    """Ensemble + MC Dropout predictor for enhancer activity.

    Parameters
    ----------
    models : list[tuple[nn.Module, torch.device]]
        List of ``(model, device)`` pairs.  Models must already be loaded
        and in evaluation mode.
    mc_passes : int
        Number of MC Dropout stochastic forward passes per model.
        Set to ``1`` (or ``0``) to disable MC Dropout (deterministic run).
    batch_size : int
        Mini-batch size for inference.
    """

    def __init__(
        self,
        models: List[Tuple[nn.Module, torch.device]],
        mc_passes: int = 30,
        batch_size: int = 256,
    ):
        if not models:
            raise ValueError("At least one model is required.")
        self.models     = models
        self.mc_passes  = max(mc_passes, 1)
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Constructor helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_configs(
        cls,
        model_specs: List[Tuple[str, str]],
        mc_passes: int = 30,
        batch_size: int = 256,
        device: Optional[str] = None,
    ) -> "EnsemblePredictor":
        """Build an ensemble from a list of (config_path, weights_path) pairs.

        Parameters
        ----------
        model_specs : list[tuple[str, str]]
            Each element is ``(config_path, weights_path)``.
        mc_passes : int
            MC Dropout forward passes per model.
        batch_size : int
            Inference batch size.
        device : str or None
            Torch device string.  Auto-detects CUDA when ``None``.

        Returns
        -------
        EnsemblePredictor
        """
        target = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        loaded = []
        for cfg_path, w_path in model_specs:
            cfg   = load_config(cfg_path)
            cfg["data"]["augment"] = False          # always eval mode
            model = build_model(cfg).to(target)
            state = torch.load(w_path, map_location=target)
            model.load_state_dict(state)
            model.eval()
            print(f"[ensemble] Loaded {cfg['model']['name']} from {w_path}")
            loaded.append((model, target))
        return cls(loaded, mc_passes=mc_passes, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, sequences: List[str]) -> PredictionResult:
        """Run ensemble + MC Dropout prediction on a list of DNA strings.

        Parameters
        ----------
        sequences : list[str]
            DNA sequences.  Must be of equal length (or will be padded to
            the longest via 'N' right-padding).

        Returns
        -------
        PredictionResult
            Full statistics including mean, variance, std, and 95 % CI.
        """
        # One-hot encode
        X = torch.tensor(
            one_hot_encode_dna(sequences), dtype=torch.float32
        )  # [N, 4, L]

        n_models   = len(self.models)
        n_passes   = self.mc_passes
        use_mc     = n_passes > 1

        per_model_dev: List[np.ndarray] = []
        per_model_hk:  List[np.ndarray] = []

        for model, device in self.models:
            if use_mc:
                d, h = _mc_predict(model, X, n_passes, self.batch_size, device)
            else:
                d0, h0 = _model_predict(model, X, self.batch_size, device)
                d, h   = d0[np.newaxis], h0[np.newaxis]
            per_model_dev.append(d)   # (n_passes, N)
            per_model_hk.append(h)

        # Pool across all models × all passes → (n_models * n_passes, N)
        all_dev = np.concatenate(per_model_dev, axis=0)
        all_hk  = np.concatenate(per_model_hk,  axis=0)

        stats = _pool_statistics(all_dev, all_hk)
        mean_dev, mean_hk, var_dev, var_hk, std_dev, std_hk, ci95_dev, ci95_hk = stats

        source = (
            "ensemble+mc_dropout" if (n_models > 1 and use_mc) else
            "mc_dropout"          if (n_models == 1 and use_mc) else
            "ensemble"            if n_models > 1 else
            "single_model"
        )

        return PredictionResult(
            sequences=sequences,
            mean_dev=mean_dev,
            mean_hk=mean_hk,
            var_dev=var_dev,
            var_hk=var_hk,
            std_dev=std_dev,
            std_hk=std_hk,
            ci95_dev=ci95_dev,
            ci95_hk=ci95_hk,
            n_models=n_models * n_passes,
            source=source,
        )

    def flag_uncertain(
        self,
        result: PredictionResult,
        var_threshold_dev: float = 0.5,
        var_threshold_hk: float = 0.5,
    ) -> pd.DataFrame:
        """Return a DataFrame of sequences that exceed the variance thresholds.

        Parameters
        ----------
        result : PredictionResult
        var_threshold_dev, var_threshold_hk : float
            Variance thresholds above which a sequence is considered
            problematic.

        Returns
        -------
        pd.DataFrame
            Subset of *result.to_dataframe()* for flagged sequences, with an
            additional ``reason`` column explaining which threshold was exceeded.
        """
        mask = result.flag_uncertain(var_threshold_dev, var_threshold_hk)
        df   = result.to_dataframe()[mask].copy()
        df["reason"] = np.where(
            (result.var_dev[mask] > var_threshold_dev) &
            (result.var_hk[mask]  > var_threshold_hk),
            "high_var_both",
            np.where(result.var_dev[mask] > var_threshold_dev, "high_var_dev", "high_var_hk"),
        )
        return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ensemble / MC Dropout inference for enhancer activity prediction"
    )
    p.add_argument(
        "--configs", nargs="+", required=True,
        help="YAML config files for each model (space-separated)",
    )
    p.add_argument(
        "--weights", nargs="+", required=True,
        help="Weight files (.pth) for each model (same order as --configs)",
    )
    p.add_argument(
        "--fasta", required=True,
        help="FASTA file of sequences to predict",
    )
    p.add_argument(
        "--output", default="outputs/ensemble/results.tsv",
        help="Output TSV path (default: outputs/ensemble/results.tsv)",
    )
    p.add_argument(
        "--mc_passes", type=int, default=30,
        help="MC Dropout forward passes per model (1 = disabled, default: 30)",
    )
    p.add_argument(
        "--batch_size", type=int, default=256,
        help="Inference batch size (default: 256)",
    )
    p.add_argument(
        "--var_threshold_dev", type=float, default=0.5,
        help="Dev variance threshold for flagging uncertain sequences",
    )
    p.add_argument(
        "--var_threshold_hk", type=float, default=0.5,
        help="Hk variance threshold for flagging uncertain sequences",
    )
    p.add_argument(
        "--device", default=None,
        help="Torch device string, e.g. 'cuda:0' (auto-detected when omitted)",
    )
    return p


def main() -> None:
    """CLI entry point."""
    args = _build_arg_parser().parse_args()

    if len(args.configs) != len(args.weights):
        raise ValueError("--configs and --weights must have the same number of entries.")

    model_specs = list(zip(args.configs, args.weights))

    predictor = EnsemblePredictor.from_configs(
        model_specs,
        mc_passes=args.mc_passes,
        batch_size=args.batch_size,
        device=args.device,
    )

    sequences = [
        str(rec.seq).upper()
        for rec in SeqIO.parse(args.fasta, "fasta")
    ]
    print(f"[ensemble] Predicting {len(sequences)} sequences ...")

    result = predictor.predict(sequences)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = result.to_dataframe()
    df.to_csv(args.output, sep="\t", index=False)
    print(f"[ensemble] Results written to {args.output}")

    flagged = predictor.flag_uncertain(
        result, args.var_threshold_dev, args.var_threshold_hk
    )
    if len(flagged):
        flag_path = args.output.replace(".tsv", "_flagged.tsv")
        flagged.to_csv(flag_path, sep="\t", index=False)
        print(f"[ensemble] {len(flagged)} uncertain sequences written to {flag_path}")
    else:
        print("[ensemble] No sequences flagged as uncertain.")


if __name__ == "__main__":
    main()
