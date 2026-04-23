#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streamlit application for evaluating and comparing DNA expression prediction models.

Features
--------
- Upload any number of (config YAML + weights) pairs for side-by-side comparison.
- Predicted vs Observed hexbins with PCC / SCC metrics.
- Forward vs Reverse-Complement strand-bias analysis.
- Dev vs Hk correlation structure.
- Residual / error distribution with Q-Q plot.
- Multi-model benchmark summary table with radar chart.
- Download all plots as PNG.

Run
---
    streamlit run src/py/streamlit/results_app.py
"""

import os
import sys
import io

import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, probplot
import streamlit as st
from Bio import SeqIO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.registry import build_model
from utils import get_reverse_complement, one_hot_encode_dna


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DNA Model Evaluation", layout="wide",
                   initial_sidebar_state="expanded")

st.title("DNA Model Evaluation & Multi-Model Comparison")
st.caption("Upload one or more model configs + weights and a test FASTA + labels to compare.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _fig_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    buf.seek(0)
    return buf.read()


@st.cache_resource(show_spinner=False)
def _load_model(config_bytes: bytes, weights_bytes: bytes):
    cfg    = yaml.safe_load(config_bytes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(cfg).to(device)
    model.load_state_dict(
        torch.load(io.BytesIO(weights_bytes), map_location=device, weights_only=True))
    model.eval()
    return model, device, cfg.get("model", {}).get("name", "model")

    model  = build_model(cfg).to(device)
    model.load_state_dict(
        torch.load(io.BytesIO(weights_bytes), map_location=device, weights_only=True))
    model.eval()
    return model, device, cfg.get("model", {}).get("name", "model")


@st.cache_data(show_spinner=False)
def _predict(_model, sequences, _device, batch_size=256):
    all_dev, all_hk = [], []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i: i + batch_size]
        X = torch.from_numpy(one_hot_encode_dna(batch)).float().to(_device)
        with torch.no_grad():
            out = _model(X)
        if isinstance(out, (list, tuple)):
            d, h = out[0], out[1]
        else:
            d, h = out[:, 0:1], out[:, 1:2]
        all_dev.append(d.cpu().squeeze().numpy())
        all_hk.append(h.cpu().squeeze().numpy())
    return (np.concatenate(all_dev).ravel(),
            np.concatenate(all_hk).ravel())


def _metrics(true, pred):
    pcc = pearsonr(true, pred)[0]
    scc = spearmanr(true, pred)[0]
    mse = float(np.mean((pred - true) ** 2))
    return pcc, scc, mse


# ---------------------------------------------------------------------------
# Sidebar — uploads
# ---------------------------------------------------------------------------
st.sidebar.header("1.  Model files")
st.sidebar.markdown(
    "Upload **config YAML** and **weights (.pth)** for each model you want to compare.  "
    "Files are matched by upload order.")
cfg_files = st.sidebar.file_uploader("Config YAMLs", type=["yaml", "yml"],
                                      accept_multiple_files=True, key="cfgs")
wt_files  = st.sidebar.file_uploader("Weights (.pth)", type=["pth"],
                                      accept_multiple_files=True, key="wts")

st.sidebar.header("2.  Test data")
fasta_up   = st.sidebar.file_uploader("FASTA file", type=["fa", "fasta"], key="fa")
labels_up  = st.sidebar.file_uploader("Activity labels (.txt / .tsv)", type=["txt", "tsv"], key="lbl")

st.sidebar.header("3.  Options")
strand_rc  = st.sidebar.checkbox("Compute RC-strand predictions too", value=True)
batch_size = st.sidebar.slider("Batch size", 32, 1024, 256, 32)
cmap_hex   = st.sidebar.selectbox("Hexbin colour map", ["Blues", "YlOrRd", "viridis", "plasma"], index=0)

run_btn = st.sidebar.button("Run analysis", type="primary")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if run_btn:
    if not cfg_files or not wt_files:
        st.error("Upload at least one config YAML and one .pth file.")
        st.stop()
    if len(cfg_files) != len(wt_files):
        st.error(f"Mismatch: {len(cfg_files)} configs vs {len(wt_files)} weight files.")
        st.stop()
    if not fasta_up or not labels_up:
        st.error("Upload both a FASTA file and an activity labels file.")
        st.stop()

    # ---- load sequences with strand correction
    fasta_bytes = fasta_up.read()
    raw_io      = io.StringIO(fasta_bytes.decode(errors="replace"))
    records     = list(SeqIO.parse(raw_io, "fasta"))
    sequences   = []
    for r in records:
        seq = str(r.seq).upper()
        if "_-_" in r.id:
            seq = get_reverse_complement(seq)
        sequences.append(seq)
    sequences_rc = [get_reverse_complement(s) for s in sequences]

    # ---- load labels
    try:
        df_lbl   = pd.read_csv(io.BytesIO(labels_up.read()), sep="\t")
        true_dev = df_lbl["Dev_log2_enrichment"].values.astype(float)
        true_hk  = df_lbl["Hk_log2_enrichment"].values.astype(float)
    except Exception as exc:
        st.error(f"Could not parse labels file: {exc}")
        st.stop()

    n = min(len(sequences), len(true_dev))
    sequences    = sequences[:n]
    sequences_rc = sequences_rc[:n]
    true_dev, true_hk = true_dev[:n], true_hk[:n]

    # ---- load all models + predict
    PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
               "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
    models_data = []
    prog = st.progress(0, text="Loading models…")
    for i, (cfg_f, wt_f) in enumerate(zip(cfg_files, wt_files)):
        prog.progress((i + 1) / len(cfg_files), text=f"Loading model {i+1}/{len(cfg_files)}…")
        model, device, mname = _load_model(cfg_f.read(), wt_f.read())
        existing = [d["name"] for d in models_data]
        tag = mname if mname not in existing else f"{mname}_{i+1}"
        with st.spinner(f"Predicting with {tag}…"):
            p_dev, p_hk       = _predict(model, sequences,    device, batch_size)
            p_dev_rc, p_hk_rc = (_predict(model, sequences_rc, device, batch_size)
                                  if strand_rc else (None, None))
        models_data.append({
            "name": tag, "model": model, "device": device,
            "p_dev": p_dev, "p_hk": p_hk,
            "p_dev_rc": p_dev_rc, "p_hk_rc": p_hk_rc,
        })
    prog.empty()

    n_models = len(models_data)
    colours  = {d["name"]: PALETTE[i % len(PALETTE)] for i, d in enumerate(models_data)}

    # ====================================================================
    # SUMMARY TABLE
    # ====================================================================
    st.header("Model benchmark summary")
    rows = []
    for d in models_data:
        pcc_d, scc_d, mse_d = _metrics(true_dev, d["p_dev"])
        pcc_h, scc_h, mse_h = _metrics(true_hk,  d["p_hk"])
        rows.append({
            "Model":    d["name"],
            "Dev PCC":  round(pcc_d, 4), "Dev SCC": round(scc_d, 4), "Dev MSE": round(mse_d, 4),
            "Hk PCC":   round(pcc_h, 4), "Hk SCC":  round(scc_h, 4), "Hk MSE":  round(mse_h, 4),
            "Avg PCC":  round((pcc_d + pcc_h) / 2, 4),
        })
    df_summary = pd.DataFrame(rows).sort_values("Avg PCC", ascending=False)
    st.dataframe(
        df_summary.style
            .highlight_max(subset=["Dev PCC", "Dev SCC", "Hk PCC", "Hk SCC", "Avg PCC"], color="#d4edda")
            .highlight_min(subset=["Dev MSE", "Hk MSE"], color="#d4edda")
            .format(precision=4),
        use_container_width=True,
    )
    st.download_button("Download summary CSV",
                       df_summary.to_csv(index=False).encode(),
                       "benchmark_summary.csv", "text/csv")

    # ====================================================================
    # RADAR CHART
    # ====================================================================
    if n_models > 1:
        st.subheader("Radar chart — multi-model comparison")
        metrics_radar = ["Dev PCC", "Dev SCC", "Hk PCC", "Hk SCC"]
        N_cat  = len(metrics_radar)
        angles = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
        angles += angles[:1]

        fig_r, ax_r = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        for i, d in enumerate(models_data):
            vals = [df_summary.loc[df_summary["Model"] == d["name"], c].values[0]
                    for c in metrics_radar]
            vals += vals[:1]
            ax_r.plot(angles, vals, color=colours[d["name"]], linewidth=2, label=d["name"])
            ax_r.fill(angles, vals, color=colours[d["name"]], alpha=0.12)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(metrics_radar, fontsize=9)
        ax_r.set_ylim(0, 1)
        ax_r.set_title("PCC / SCC comparison", pad=16)
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
        col_r, _ = st.columns([1, 1])
        col_r.pyplot(fig_r)

    # ====================================================================
    # TABS
    # ====================================================================
    tab_pred, tab_strand, tab_struct, tab_err, tab_rank = st.tabs([
        "Predicted vs Observed",
        "Strand Bias",
        "Dev vs Hk Structure",
        "Residuals",
        "Rank-order",
    ])

    # ---- Tab 1: Predicted vs Observed ----
    with tab_pred:
        for d in models_data:
            with st.expander(f"**{d['name']}**", expanded=(n_models == 1)):
                pcc_d = pearsonr(true_dev, d["p_dev"])[0]
                pcc_h = pearsonr(true_hk,  d["p_hk"])[0]
                scc_d = spearmanr(true_dev, d["p_dev"])[0]
                scc_h = spearmanr(true_hk,  d["p_hk"])[0]
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                for ax, truth, pred, title in [
                    (axes[0], true_dev, d["p_dev"], f"Developmental  PCC={pcc_d:.3f}  SCC={scc_d:.3f}"),
                    (axes[1], true_hk,  d["p_hk"],  f"Housekeeping   PCC={pcc_h:.3f}  SCC={scc_h:.3f}"),
                ]:
                    hb = ax.hexbin(truth, pred, bins="log", cmap=cmap_hex, gridsize=60)
                    lim = [min(truth.min(), pred.min()), max(truth.max(), pred.max())]
                    ax.plot(lim, lim, "r--", lw=1, alpha=0.7)
                    ax.set_xlabel("Observed log₂ enrichment")
                    ax.set_ylabel("Predicted log₂ enrichment")
                    ax.set_title(title, fontsize=10)
                    fig.colorbar(hb, ax=ax, label="log₁₀ count")
                    _style_axes(ax)
                fig.suptitle(d["name"], fontsize=13, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Dev PCC", f"{pcc_d:.4f}")
                c2.metric("Dev SCC", f"{scc_d:.4f}")
                c3.metric("Hk PCC",  f"{pcc_h:.4f}")
                c4.metric("Hk SCC",  f"{scc_h:.4f}")
                st.download_button(f"Download {d['name']} plot",
                                   _fig_bytes(fig), f"{d['name']}_pred_vs_obs.png", "image/png",
                                   key=f"dl_pred_{d['name']}")

    # ---- Tab 2: Strand Bias ----
    with tab_strand:
        if not strand_rc:
            st.info("Enable 'Compute RC-strand predictions' in the sidebar.")
        else:
            for d in models_data:
                with st.expander(f"**{d['name']}**", expanded=(n_models == 1)):
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    for ax, fwd, rc, label in [
                        (axes[0], d["p_dev"], d["p_dev_rc"], "Developmental"),
                        (axes[1], d["p_hk"],  d["p_hk_rc"],  "Housekeeping"),
                    ]:
                        hb = ax.hexbin(fwd, rc, bins="log", cmap=cmap_hex, gridsize=60)
                        lim = [min(fwd.min(), rc.min()), max(fwd.max(), rc.max())]
                        ax.plot(lim, lim, "k--", lw=1)
                        r = pearsonr(fwd, rc)[0]
                        ax.set_xlabel("Forward strand prediction")
                        ax.set_ylabel("RC strand prediction")
                        ax.set_title(f"{label}  r(fwd,rc)={r:.3f}", fontsize=10)
                        fig.colorbar(hb, ax=ax, label="log₁₀ count")
                        _style_axes(ax)
                    fig.suptitle(f"{d['name']} — Strand Bias", fontsize=13, fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.download_button(f"Download {d['name']} strand plot",
                                       _fig_bytes(fig), f"{d['name']}_strand_bias.png", "image/png",
                                       key=f"dl_strand_{d['name']}")

    # ---- Tab 3: Dev vs Hk structure ----
    with tab_struct:
        fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))
        if n_models == 0:
            axes = [axes]
        hb_o = axes[0].hexbin(true_dev, true_hk, bins="log", cmap="Greys", gridsize=60)
        axes[0].set_xlabel("Dev log₂"); axes[0].set_ylabel("Hk log₂")
        r_o = pearsonr(true_dev, true_hk)[0]
        axes[0].set_title(f"Observed (r={r_o:.3f})")
        fig.colorbar(hb_o, ax=axes[0], label="log₁₀ count")
        _style_axes(axes[0])
        for i, d in enumerate(models_data):
            ax = axes[i + 1]
            hb = ax.hexbin(d["p_dev"], d["p_hk"], bins="log", cmap=cmap_hex, gridsize=60)
            r = pearsonr(d["p_dev"], d["p_hk"])[0]
            ax.set_xlabel("Pred Dev log₂"); ax.set_ylabel("Pred Hk log₂")
            ax.set_title(f"{d['name']} (r={r:.3f})", fontsize=10)
            fig.colorbar(hb, ax=ax, label="log₁₀ count")
            _style_axes(ax)
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("Download structure plot", _fig_bytes(fig), "dev_hk_structure.png", "image/png")

    # ---- Tab 4: Residuals ----
    with tab_err:
        for d in models_data:
            with st.expander(f"**{d['name']}**", expanded=(n_models == 1)):
                fig, axes = plt.subplots(2, 2, figsize=(12, 9))
                for col_i, (truth, pred, label, color) in enumerate([
                    (true_dev, d["p_dev"], "Developmental", colours[d["name"]]),
                    (true_hk,  d["p_hk"],  "Housekeeping",  colours[d["name"]]),
                ]):
                    err = pred - truth
                    axes[0, col_i].hist(err, bins=80, color=color, edgecolor="white", alpha=0.8)
                    axes[0, col_i].axvline(0, color="red", lw=1.2, ls="--")
                    axes[0, col_i].set_xlabel("Residual")
                    axes[0, col_i].set_ylabel("Count")
                    axes[0, col_i].set_title(f"{label}  μ={err.mean():.3f}  σ={err.std():.3f}")
                    _style_axes(axes[0, col_i])
                    (osm, osr), (slope, intercept, _) = probplot(err, dist="norm")
                    axes[1, col_i].scatter(osm, osr, s=3, alpha=0.4, color=color)
                    axes[1, col_i].plot([osm.min(), osm.max()],
                                        [slope * osm.min() + intercept,
                                         slope * osm.max() + intercept], "r--", lw=1)
                    axes[1, col_i].set_xlabel("Theoretical quantiles")
                    axes[1, col_i].set_ylabel("Sample quantiles")
                    axes[1, col_i].set_title(f"{label} Q-Q")
                    _style_axes(axes[1, col_i])
                fig.suptitle(f"{d['name']} — Residual Analysis", fontsize=13, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                st.download_button(f"Download {d['name']} residuals",
                                   _fig_bytes(fig), f"{d['name']}_residuals.png", "image/png",
                                   key=f"dl_err_{d['name']}")

    # ---- Tab 5: Rank-order ----
    with tab_rank:
        st.markdown("Sequences ordered by **observed** activity; lines show model rank-order predictions.")
        for task_label, truth in [("Developmental", true_dev), ("Housekeeping", true_hk)]:
            st.subheader(task_label)
            sort_idx = np.argsort(truth)
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(truth[sort_idx], color="black", lw=1.5, label="Observed", zorder=n_models + 1)
            for i, d in enumerate(models_data):
                p = d["p_dev"] if task_label == "Developmental" else d["p_hk"]
                ax.plot(p[sort_idx], color=PALETTE[i % len(PALETTE)], lw=0.9,
                        alpha=0.8, label=d["name"])
            ax.set_xlabel("Sequence rank (by observed activity)")
            ax.set_ylabel("log₂ enrichment")
            ax.set_title(f"{task_label} — rank-ordered predictions")
            ax.legend(fontsize=8, ncol=min(n_models + 1, 4))
            _style_axes(ax)
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button(f"Download {task_label} rank plot",
                               _fig_bytes(fig), f"rank_{task_label}.png", "image/png",
                               key=f"dl_rank_{task_label}")

else:
    st.info("Configure uploads in the sidebar and click **Run analysis** to begin.")
    st.markdown("""
**What to upload:**
- **Config YAML** — model configuration (e.g. `config/LegNetPlus.yaml`)
- **Weights (.pth)** — trained checkpoint
- **FASTA** — test sequences (`data/deepSTARR/Sequences_Test.fa`)
- **Activity labels** — TSV with `Dev_log2_enrichment` and `Hk_log2_enrichment` columns

Upload multiple config + weight pairs to compare several models simultaneously.
""")
