#!/usr/bin/env python
"""Streamlit app — Ensemble + MC Dropout uncertainty explorer.

Run
---
    streamlit run src/py/streamlit/ensemble_app.py

Features
--------
- Upload multiple (config YAML + checkpoint) pairs to form an ensemble.
- Upload a FASTA file with sequences to evaluate.
- See per-sequence Dev / Hk mean predictions with 95% CI from MC Dropout.
- Flag high-uncertainty sequences above user-defined variance thresholds.
- Download results as CSV.
"""

import os
import sys
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ensemble import EnsemblePredictor
from utils import one_hot_encode_dna
from models.registry import build_model

# ---------------------------------------------------------------------------
try:
    st.set_page_config(page_title="Ensemble Prediction & Uncertainty", layout="wide")
except st.errors.StreamlitAPIException:
    pass
st.title("Ensemble Prediction & MC Dropout Uncertainty")
st.markdown(
    "Upload model configs + weights to build an ensemble, then evaluate sequences "
    "with predictive uncertainty estimates from MC Dropout."
)

# ---------------------------------------------------------------------------
# Sidebar — model upload
# ---------------------------------------------------------------------------
st.sidebar.header("1.  Model files")
st.sidebar.markdown("Upload one or more **config YAML** + **checkpoint (.pth)** pairs.")

uploaded_configs  = st.sidebar.file_uploader(
    "YAML config files",  type=["yaml", "yml"], accept_multiple_files=True, key="cfgs")
uploaded_weights  = st.sidebar.file_uploader(
    ".pth weight files",  type=["pth"],          accept_multiple_files=True, key="wts")
mc_passes         = st.sidebar.slider("MC Dropout passes",    min_value=5,   max_value=100, value=30, step=5)
batch_size        = st.sidebar.slider("Batch size",           min_value=32,  max_value=1024, value=256, step=32)
var_thresh_dev    = st.sidebar.number_input("Variance threshold Dev", min_value=0.0, value=0.5, step=0.05, format="%.3f")
var_thresh_hk     = st.sidebar.number_input("Variance threshold Hk",  min_value=0.0, value=0.5, step=0.05, format="%.3f")

# Sidebar — sequence upload
st.sidebar.header("2.  Sequences")
uploaded_fasta = st.sidebar.file_uploader("FASTA file", type=["fa", "fasta", "txt"], key="fasta")

# ---------------------------------------------------------------------------
# Parse / validate uploads
# ---------------------------------------------------------------------------

@st.cache_resource
def _build_ensemble(cfg_bytes_list, wt_bytes_list, mc_passes, batch_size):
    import yaml
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    for cfg_bytes, wt_bytes in zip(cfg_bytes_list, wt_bytes_list):
        cfg   = yaml.safe_load(cfg_bytes)
        model = build_model(cfg).to(device)
        model.load_state_dict(
            torch.load(io.BytesIO(wt_bytes), map_location=device, weights_only=True))
        model.eval()
        models.append((model, device))
    return EnsemblePredictor(models, mc_passes=mc_passes, batch_size=batch_size)


def _parse_fasta(fasta_bytes: bytes) -> list[str]:
    sequences = []
    current_seq = []
    for line in fasta_bytes.decode(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_seq:
                sequences.append("".join(current_seq).upper())
                current_seq = []
        else:
            current_seq.append(line)
    if current_seq:
        sequences.append("".join(current_seq).upper())
    return sequences


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

run_button = st.button("Run ensemble prediction", type="primary")

if run_button:
    if not uploaded_configs or not uploaded_weights:
        st.error("Please upload at least one config YAML and one .pth weight file.")
        st.stop()
    if len(uploaded_configs) != len(uploaded_weights):
        st.error(f"Number of configs ({len(uploaded_configs)}) and weights "
                 f"({len(uploaded_weights)}) must match.")
        st.stop()
    if not uploaded_fasta:
        st.error("Please upload a FASTA file with sequences.")
        st.stop()

    cfg_bytes_list = [f.read() for f in uploaded_configs]
    wt_bytes_list  = [f.read() for f in uploaded_weights]
    fasta_bytes    = uploaded_fasta.read()
    sequences      = _parse_fasta(fasta_bytes)

    if not sequences:
        st.error("No sequences parsed from the uploaded FASTA file.")
        st.stop()

    n_seqs = len(sequences)
    st.info(f"Loaded {n_seqs} sequences.  Building ensemble ({len(cfg_bytes_list)} models) …")

    with st.spinner("Loading models …"):
        predictor = _build_ensemble(
            tuple(cfg_bytes_list), tuple(wt_bytes_list), mc_passes, batch_size)

    with st.spinner(f"Running predictions (MC passes = {mc_passes}) …"):
        result = predictor.predict(sequences)

    # Build display dataframe
    df = result.to_dataframe()
    df["sequence_id"] = [f"seq_{i}" for i in range(n_seqs)]
    df["uncertain"]   = (
        (df["var_dev"] > var_thresh_dev) | (df["var_hk"] > var_thresh_hk)
    )
    n_uncertain = df["uncertain"].sum()

    st.success(
        f"Done. {n_uncertain} / {n_seqs} sequences flagged as high-uncertainty "
        f"(var_dev > {var_thresh_dev:.3f} OR var_hk > {var_thresh_hk:.3f})."
    )

    # --- Scatter: mean prediction ± CI ----------------------------------------
    col1, col2 = st.columns(2)
    for col, (task, mn, lo, hi) in zip(
        [col1, col2],
        [("Dev",  "mean_dev", "ci95_dev_low", "ci95_dev_high"),
         ("Hk",   "mean_hk",  "ci95_hk_low",  "ci95_hk_high")],
    ):
        with col:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors  = ["red" if u else "steelblue" for u in df["uncertain"]]
            ax.errorbar(
                range(n_seqs),
                df[mn],
                yerr=[df[mn] - df[lo], df[hi] - df[mn]],
                fmt="none", alpha=0.3, color="gray", linewidth=0.5,
            )
            ax.scatter(range(n_seqs), df[mn], s=4, c=colors, zorder=3)
            ax.set_xlabel("Sequence index")
            ax.set_ylabel("Predicted log2-enrichment")
            ax.set_title(f"{task} predictions (red = uncertain)")
            plt.tight_layout()
            st.pyplot(fig)

    # --- Variance histogram ---------------------------------------------------
    st.subheader("Prediction variance distribution")
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.hist(df["var_dev"], bins=60, color="coral", edgecolor="white")
    a1.axvline(var_thresh_dev, color="black", linestyle="--", label=f"threshold = {var_thresh_dev}")
    a1.set_title("Variance Dev")
    a1.set_xlabel("Var")
    a1.legend()

    a2.hist(df["var_hk"], bins=60, color="seagreen", edgecolor="white")
    a2.axvline(var_thresh_hk, color="black", linestyle="--", label=f"threshold = {var_thresh_hk}")
    a2.set_title("Variance Hk")
    a2.set_xlabel("Var")
    a2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    # --- Rank-order comparison (Dev vs Hk mean) --------------------------------
    st.subheader("Rank-order: Dev vs Hk predictions")
    rank_dev = np.argsort(np.argsort(-df["mean_dev"]))   # rank from highest
    rank_hk  = np.argsort(np.argsort(-df["mean_hk"]))
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sc = ax3.scatter(rank_dev, rank_hk, s=5,
                     c=["red" if u else "steelblue" for u in df["uncertain"]],
                     alpha=0.5)
    lim = n_seqs
    ax3.plot([0, lim], [0, lim], 'k--', lw=1, label='rank parity')
    ax3.set_xlabel("Rank by Dev activity")
    ax3.set_ylabel("Rank by Hk activity")
    ax3.set_title("Rank-order agreement (Dev vs Hk)")
    ax3.legend(fontsize=8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)
    rank_scc = np.corrcoef(rank_dev, rank_hk)[0, 1]
    st.caption(f"Spearman rank correlation (Dev vs Hk): **{rank_scc:.4f}**")

    # --- Data table + download -----------------------------------------------
    st.subheader("Results table")
    display_cols = ["sequence_id", "mean_dev", "mean_hk",
                    "var_dev",  "var_hk", "uncertain"]
    st.dataframe(df[display_cols].style.highlight_max(
        subset=["var_dev", "var_hk"], color="#ffdddd"))

    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("Download full results CSV", data=csv_bytes,
                       file_name="ensemble_predictions.csv", mime="text/csv")
