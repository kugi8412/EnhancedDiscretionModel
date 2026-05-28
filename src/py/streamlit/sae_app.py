#!/usr/bin/env python
"""Streamlit app — Sparse Autoencoder feature explorer.

Run
---
    streamlit run src/py/streamlit/sae_app.py

Features
--------
- Upload frozen model config + weights + SAE checkpoint (.pth).
- Upload a FASTA file + activity file.
- Browse per-feature statistics (mean activation, fraction active, PCC with Dev/Hk).
- Visualise top enriched k-mers for any selected feature.
- Compare two SAE runs (different models / layer depths) side-by-side.
"""

import os
import sys
import json
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import streamlit as st
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import load_config, one_hot_encode_dna
from models.registry import build_model
from sparse_ae.model import SparseAutoencoder
from sparse_ae.train import _resolve_module, load_activations
from sparse_ae.analyze import compute_feature_stats, top_k_sequences, enriched_kmers, _kmer_counts
from sparse_ae.visualize import plot_feature_stats, plot_kmer_enrichment, plot_activation_heatmap

# ---------------------------------------------------------------------------
try:
    st.set_page_config(page_title="SAE Feature Explorer", layout="wide")
except st.errors.StreamlitAPIException:
    pass
st.title("Sparse Autoencoder Feature Explorer")
st.markdown(
    "Load a frozen backbone model + trained SAE to explore which sequence motifs "
    "drive individual feature activations."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("1.  Run A — Primary model + SAE")

cfg_a   = st.sidebar.file_uploader("Model config (YAML)",   type=["yaml", "yml"], key="cfg_a")
wts_a   = st.sidebar.file_uploader("Model weights (.pth)",  type=["pth"],         key="wts_a")
sae_a   = st.sidebar.file_uploader("SAE checkpoint (.pth)", type=["pth"],         key="sae_a")
layer_a = st.sidebar.text_input("Hook layer (dot-separated)", value="encoder_gru", key="layer_a")

st.sidebar.header("2.  Run B — (optional) Compare second SAE")
cfg_b   = st.sidebar.file_uploader("Model config (YAML)",   type=["yaml", "yml"], key="cfg_b")
wts_b   = st.sidebar.file_uploader("Model weights (.pth)",  type=["pth"],         key="wts_b")
sae_b   = st.sidebar.file_uploader("SAE checkpoint (.pth)", type=["pth"],         key="sae_b")
layer_b = st.sidebar.text_input("Hook layer (dot-separated)", value="encoder_gru", key="layer_b")

st.sidebar.header("3.  Data")
fasta_up    = st.sidebar.file_uploader("FASTA file",    type=["fa", "fasta", "txt"], key="fasta")
activity_up = st.sidebar.file_uploader("Activity file (TSV, cols: Dev Hk)",
                                        type=["txt", "tsv"], key="act")

st.sidebar.header("4.  Analysis settings")
dict_size  = st.sidebar.number_input("SAE dict_size (must match checkpoint)", value=1024, step=64)
input_dim  = st.sidebar.number_input("Activation dimension (must match training)",  value=256, step=32)
top_k      = st.sidebar.slider("Top-K sequences for k-mer analysis", 50, 1000, 300, 50)
kmer_k     = st.sidebar.selectbox("k-mer length", [4, 5, 6], index=2)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_fasta(fasta_bytes: bytes):
    seqs = []
    current_seq = []
    for line in fasta_bytes.decode(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_seq:
                seqs.append("".join(current_seq).upper())
                current_seq = []
        else:
            current_seq.append(line)
    if current_seq:
        seqs.append("".join(current_seq).upper())
    return seqs


def _parse_activity(act_bytes: bytes):
    """Return y_dev, y_hk as numpy arrays."""
    df = pd.read_csv(io.BytesIO(act_bytes), sep="\t")
    y_dev = df.iloc[:, 0].values.astype(np.float32)
    y_hk  = df.iloc[:, 1].values.astype(np.float32)
    return y_dev, y_hk


@st.cache_resource
def _extract_features(cfg_bytes, wts_bytes, sae_bytes, layer_name, seq_bytes,
                       dict_size, input_dim):
    import yaml, tempfile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg    = yaml.safe_load(cfg_bytes)
    model  = build_model(cfg).to(device)
    model.load_state_dict(
        torch.load(io.BytesIO(wts_bytes), map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    sequences = _parse_fasta(seq_bytes)
    X         = one_hot_encode_dna(sequences)
    X_t       = torch.from_numpy(X).float()
    loader    = DataLoader(TensorDataset(X_t), batch_size=256, shuffle=False)

    hook_mod = _resolve_module(model, layer_name)
    acts     = load_activations(model, hook_mod, loader, device)

    sae = SparseAutoencoder(int(input_dim), int(dict_size))
    sae.load_state_dict(
        torch.load(io.BytesIO(sae_bytes), map_location="cpu", weights_only=True))
    sae.eval()

    with torch.no_grad():
        features = sae.encode(acts.to(device)).cpu().numpy()

    return features, sequences


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
run_btn = st.button("Analyse features", type="primary")

if run_btn:
    missing = []
    for name, up in [("Model config A", cfg_a), ("Model weights A", wts_a),
                     ("SAE checkpoint A", sae_a), ("FASTA", fasta_up), ("Activity", activity_up)]:
        if not up:
            missing.append(name)
    if missing:
        st.error("Missing uploads: " + ", ".join(missing))
        st.stop()

    fasta_bytes = fasta_up.read()
    y_dev, y_hk = _parse_activity(activity_up.read())
    sequences   = _parse_fasta(fasta_bytes)

    # --- Run A ---------------------------------------------------------------
    st.header("Run A")
    with st.spinner("Extracting activations and computing SAE features (Run A) …"):
        feat_a, seqs_a = _extract_features(
            cfg_a.read(), wts_a.read(), sae_a.read(),
            layer_a, fasta_bytes, dict_size, input_dim)

    stats_a = compute_feature_stats(feat_a, y_dev[:len(seqs_a)], y_hk[:len(seqs_a)])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total features", stats_a.shape[0])
    col2.metric("Median |PCC Dev|", f"{stats_a['pcc_dev'].abs().median():.3f}")
    col3.metric("Median |PCC Hk|",  f"{stats_a['pcc_hk'].abs().median():.3f}")

    # Feature stats scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(stats_a["frac_active"], bins=50, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Fraction active")
    axes[0].set_title("Sparsity distribution")

    sc = axes[1].scatter(stats_a["pcc_dev"], stats_a["pcc_hk"],
                         c=stats_a["frac_active"], cmap="viridis", s=6, alpha=0.7)
    axes[1].set_xlabel("PCC Dev")
    axes[1].set_ylabel("PCC Hk")
    axes[1].set_title("Dev vs Hk correlation")
    plt.colorbar(sc, ax=axes[1], label="frac active")
    plt.tight_layout()
    st.pyplot(fig)

    # Feature browser
    st.subheader("Individual feature explorer")
    top_by_pcc = stats_a.assign(max_pcc=stats_a[["pcc_dev", "pcc_hk"]].abs().max(axis=1))
    top_by_pcc = top_by_pcc.sort_values("max_pcc", ascending=False)

    selected_fi = st.selectbox(
        "Select a feature to inspect (sorted by max |PCC|)",
        top_by_pcc["feature_idx"].tolist(),
        format_func=lambda fi: (
            f"Feature {fi}  |  PCC_dev={stats_a.loc[stats_a['feature_idx']==fi,'pcc_dev'].values[0]:.3f}"
            f"  PCC_hk={stats_a.loc[stats_a['feature_idx']==fi,'pcc_hk'].values[0]:.3f}"
            f"  frac={stats_a.loc[stats_a['feature_idx']==fi,'frac_active'].values[0]:.2%}"
        ),
    )

    if selected_fi is not None:
        top_idx  = np.argsort(-feat_a[:, selected_fi])[:top_k]
        top_seqs = [seqs_a[i] for i in top_idx]
        bg_counts = _kmer_counts(seqs_a, k=kmer_k).values.sum(axis=0)
        enr      = enriched_kmers(top_seqs, bg_counts, len(seqs_a), k=kmer_k, top_n=20)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        colors = ["steelblue" if o >= 1 else "salmon" for o in enr["enrichment"]]
        ax2.barh(enr["kmer"][::-1], enr["enrichment"][::-1], color=colors[::-1])
        ax2.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax2.set_xlabel("Enrichment (odds ratio)")
        ax2.set_title(f"Feature {selected_fi} — top-{kmer_k}-mer enrichment")
        plt.tight_layout()
        st.pyplot(fig2)

        st.dataframe(enr.reset_index(drop=True))

    # --- Activation heatmap --------------------------------------------------
    st.subheader("Activation heatmap (random subset)")
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    rng     = np.random.default_rng(0)
    sub_idx = rng.choice(feat_a.shape[0], min(200, feat_a.shape[0]), replace=False)
    top_fi  = np.argsort(-feat_a.mean(axis=0))[:50]
    import seaborn as sns
    sns.heatmap(feat_a[sub_idx][:, top_fi].T, ax=ax3, cmap="viridis",
                xticklabels=False, yticklabels=[str(f) for f in top_fi])
    ax3.set_xlabel("Sequence")
    ax3.set_ylabel("Feature")
    ax3.set_title("Top-50 features × 200 random sequences")
    plt.tight_layout()
    st.pyplot(fig3)

    # --- Run B comparison ----------------------------------------------------
    if cfg_b and wts_b and sae_b:
        st.header("Run B — comparison")
        with st.spinner("Run B …"):
            feat_b, _seqs_b = _extract_features(
                cfg_b.read(), wts_b.read(), sae_b.read(),
                layer_b, fasta_bytes, dict_size, input_dim)
        stats_b = compute_feature_stats(feat_b, y_dev[:len(_seqs_b)], y_hk[:len(_seqs_b)])

        fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4))
        for stat, color, label in [
            (stats_a, "steelblue", "Run A"),
            (stats_b, "coral",     "Run B"),
        ]:
            axes4[0].hist(stat["pcc_dev"].abs(), bins=40, alpha=0.6,
                          color=color, label=label, edgecolor="white", density=True)
            axes4[1].hist(stat["pcc_hk"].abs(),  bins=40, alpha=0.6,
                          color=color, label=label, edgecolor="white", density=True)
        for ax, title in zip(axes4, ["Dev |PCC|", "Hk |PCC|"]):
            ax.set_title(title)
            ax.set_xlabel("|PCC|")
            ax.set_ylabel("Density")
            ax.legend()
        plt.tight_layout()
        st.pyplot(fig4)

    # --- Full stats table download -------------------------------------------
    st.subheader("Run A — feature statistics table")
    st.dataframe(stats_a.sort_values("frac_active", ascending=False))
    csv_bytes = stats_a.to_csv(index=False).encode()
    st.download_button("Download feature_stats.csv", data=csv_bytes,
                       file_name="feature_stats.csv", mime="text/csv")
