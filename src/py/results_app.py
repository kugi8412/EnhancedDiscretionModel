#!/usr/bin/env python
"""
Streamlit app for DNA model evaluation and visualization.
Auto-detects model architecture from weights — no config YAML needed.

Usage:
    streamlit run results_app.py
"""

import os
import sys
import io
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import streamlit as st
from Bio import SeqIO

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.registry import MODELS, build_model
from utils import get_reverse_complement, one_hot_encode_dna

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="DNA Model Evaluation", layout="wide", page_icon="🧬")


# ============================================================
# MODEL AUTO-DETECTION
# ============================================================

# Key patterns in state_dict keys that uniquely identify each architecture
_MODEL_SIGNATURES = {
    'LegNetPlus': ['stem.branch_k3', 'glub.gate_proj'],
    'LegNetV2': ['stem.0.weight', 'main_blocks.0.effblock'],
    'LegNet': ['stem.0.weight', 'main_blocks.0.local_block'],
    'ConvNeXt_DNA': ['downsample_layers', 'stages'],
    'SEResNet': ['layer1', 'layer2', 'se_block'],
    'DeepSTARR': ['conv_block1', 'conv_block2', 'dense_block'],
    'ReverseNet_SuperKernel': ['conv_block1', 'rev_conv_block'],
    'BassetNetwork': ['conv_layers', 'fc_layers'],
    'CustomNetwork': ['conv_layers', 'fc_layers', 'dropout'],
    'HydraDNA_cVQVAE': ['gru', 'vq_layer'],
    'LegNet_VQVAE': ['encoder', 'vq_layer', 'decoder'],
    'DNA_PixelCNN': ['input_conv', 'gated_blocks'],
}


def _match_signature(state_keys, patterns):
    """Check if all pattern substrings appear in at least one key."""
    for pat in patterns:
        if not any(pat in k for k in state_keys):
            return False
    return True


def detect_model_name(state_dict):
    """Detect model name from state_dict key patterns."""
    keys = list(state_dict.keys())
    for name, patterns in _MODEL_SIGNATURES.items():
        if _match_signature(keys, patterns):
            return name
    return None


def _infer_kwargs_from_state(name, state_dict):
    """Try to infer constructor kwargs from weight shapes."""
    kwargs = {}

    if name in ('LegNet', 'LegNetV2', 'LegNetPlus'):
        # stem first conv -> in_ch
        for k, v in state_dict.items():
            if 'stem' in k and 'weight' in k and v.dim() == 3:
                kwargs['in_ch'] = v.shape[1]
                kwargs['stem_ch'] = v.shape[0]
                break

    elif name == 'DeepSTARR':
        for k, v in state_dict.items():
            if 'conv_block1' in k and 'weight' in k and v.dim() == 3:
                kwargs['num_filters'] = v.shape[0]
                break

    elif name == 'ConvNeXt_DNA':
        for k, v in state_dict.items():
            if 'downsample_layers.0' in k and 'weight' in k and v.dim() == 3:
                kwargs['in_ch'] = v.shape[1]
                kwargs['stem_ch'] = v.shape[0]
                break

    elif name == 'SEResNet':
        for k, v in state_dict.items():
            if 'conv1.weight' in k and v.dim() == 3:
                kwargs['in_channels'] = v.shape[1]
                break

    return kwargs


def auto_load_model(state_dict, device):
    """Auto-detect architecture, build model, load weights."""
    name = detect_model_name(state_dict)
    if name is None:
        raise ValueError(
            "Could not auto-detect model architecture from weight keys. "
            f"Registered models: {list(MODELS.keys())}. "
            "Upload a config YAML for manual loading."
        )
    if name not in MODELS:
        raise ValueError(f"Detected '{name}' but it is not in the registry.")

    model_cls = MODELS[name]
    inferred = _infer_kwargs_from_state(name, state_dict)

    # Try with inferred kwargs + multiple seq_len candidates
    for seq_len in [249, 269, 200, 300]:
        try:
            kw = {**inferred, 'seq_len': seq_len}
            # Remove seq_len for models that don't accept it
            try:
                model = model_cls(**kw)
            except TypeError:
                kw.pop('seq_len', None)
                model = model_cls(**kw)
            model.load_state_dict(state_dict, strict=True)
            model.to(device).eval()
            return name, model
        except Exception:
            continue

    # Fallback: default kwargs
    try:
        model = model_cls()
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()
        return name, model
    except Exception as e:
        raise ValueError(
            f"Detected architecture '{name}' but could not load weights: {e}"
        )


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource
def load_model(weights_bytes, config_dict=None):
    """Load model — auto-detect or use config."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(
        io.BytesIO(weights_bytes), map_location=device, weights_only=True
    )

    if config_dict is not None:
        import yaml
        model = build_model(config_dict).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        name = config_dict.get('model', {}).get('name', 'Unknown')
        return name, model, device

    name, model = auto_load_model(state_dict, device)
    return name, model, device


@st.cache_data
def run_predictions(_model, sequences, _device, batch_size=512):
    all_dev, all_hk = [], []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        encoded = one_hot_encode_dna(batch)
        tensor = torch.tensor(encoded, dtype=torch.float32).to(_device)
        with torch.no_grad():
            out = _model(tensor)
            if isinstance(out, (list, tuple)):
                d, h = out[0], out[1]
            else:
                d, h = out[:, 0], out[:, 1]
        all_dev.extend(d.cpu().numpy().flatten())
        all_hk.extend(h.cpu().numpy().flatten())
    return np.array(all_dev), np.array(all_hk)


def gc_content(sequences):
    return np.array(
        [(s.count('G') + s.count('C')) / len(s) * 100 for s in sequences]
    )


def adjust_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("1. Model")

config_mode = st.sidebar.radio(
    "Loading mode:",
    ["Auto-detect (weights only)", "Manual (config + weights)"],
    help="Auto-detect infers architecture from weight file keys."
)

config_file = None
if config_mode.startswith("Manual"):
    config_file = st.sidebar.file_uploader("Config YAML", type=["yaml", "yml"])

weights_file = st.sidebar.file_uploader("Model weights (.pth)", type=["pth"])

st.sidebar.header("2. Test Data")
fasta_file = st.sidebar.file_uploader("Sequences (.fasta)", type=["fa", "fasta"])
targets_file = st.sidebar.file_uploader(
    "Activity file (.txt/.tsv)", type=["txt", "csv", "tsv"]
)

st.sidebar.header("3. Visualization Options")
color_option = st.sidebar.selectbox(
    "Color overlay (strand bias tab):",
    ["Density (log count)", "GC content (%)", "MSE error"],
)


# ============================================================
# MAIN
# ============================================================
st.title("🧬 DNA Enhancer Model Evaluation")

if weights_file and fasta_file and targets_file:
    # ---- LOAD MODEL ----
    try:
        cfg = None
        if config_file:
            import yaml
            cfg = yaml.safe_load(config_file)
        model_name, model, device = load_model(weights_file.getvalue(), cfg)
        st.sidebar.success(f"Model: **{model_name}**")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # ---- LOAD DATA ----
    fasta_text = fasta_file.getvalue().decode("utf-8")
    records = list(SeqIO.parse(io.StringIO(fasta_text), "fasta"))
    sequences = [str(r.seq).upper() for r in records]

    try:
        df_targets = pd.read_csv(
            io.StringIO(targets_file.getvalue().decode("utf-8")), sep='\t'
        )
        true_dev = df_targets['Dev_log2_enrichment'].values
        true_hk = df_targets['Hk_log2_enrichment'].values
    except Exception as e:
        st.error(f"Error reading activity file: {e}")
        st.stop()

    n = min(len(sequences), len(true_dev))
    sequences, true_dev, true_hk = sequences[:n], true_dev[:n], true_hk[:n]

    # ---- PREDICTIONS ----
    with st.spinner("Running forward predictions..."):
        pred_dev, pred_hk = run_predictions(model, sequences, device)
    with st.spinner("Running reverse complement predictions..."):
        seqs_rc = [get_reverse_complement(s) for s in sequences]
        pred_dev_rc, pred_hk_rc = run_predictions(model, seqs_rc, device)

    gc_vals = gc_content(sequences)
    mse_dev = (pred_dev - true_dev) ** 2
    mse_hk = (pred_hk - true_hk) ** 2

    # ---- METRICS ----
    pcc_dev = pearsonr(true_dev, pred_dev)[0]
    pcc_hk = pearsonr(true_hk, pred_hk)[0]
    scc_dev = spearmanr(true_dev, pred_dev)[0]
    scc_hk = spearmanr(true_hk, pred_hk)[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dev PCC", f"{pcc_dev:.4f}")
    col2.metric("Hk PCC", f"{pcc_hk:.4f}")
    col3.metric("Dev SCC", f"{scc_dev:.4f}")
    col4.metric("Hk SCC", f"{scc_hk:.4f}")

    # ---- TABS ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "Predicted vs Observed",
        "Forward vs RC Strand",
        "Dev vs Hk Correlation",
        "Error Distribution",
    ])

    # ========== TAB 1: PREDICTED vs OBSERVED ==========
    with tab1:
        st.subheader("Predicted vs Observed Expression")

        fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))

        hb1 = axes1[0].hexbin(
            true_dev, pred_dev, gridsize=60, bins='log', cmap='viridis'
        )
        axes1[0].set_title(f'Developmental (PCC = {pcc_dev:.3f})')
        axes1[0].set_xlabel('Observed [log2]')
        axes1[0].set_ylabel('Predicted [log2]')
        adjust_axes(axes1[0])
        fig1.colorbar(hb1, ax=axes1[0], label='log10(count)')

        hb2 = axes1[1].hexbin(
            true_hk, pred_hk, gridsize=60, bins='log', cmap='viridis'
        )
        axes1[1].set_title(f'Housekeeping (PCC = {pcc_hk:.3f})')
        axes1[1].set_xlabel('Observed [log2]')
        adjust_axes(axes1[1])
        fig1.colorbar(hb2, ax=axes1[1], label='log10(count)')

        fig1.suptitle(f'{model_name} — Predictions vs Observed', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig1)
        st.download_button(
            "Download plot", fig_to_bytes(fig1),
            f"{model_name}_pred_vs_obs.png", "image/png",
        )

    # ========== TAB 2: FWD vs RC ==========
    with tab2:
        st.subheader("Forward vs Reverse Complement Strand Bias")
        pcc_rc_dev = pearsonr(pred_dev, pred_dev_rc)[0]
        pcc_rc_hk = pearsonr(pred_hk, pred_hk_rc)[0]

        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

        if color_option.startswith("GC"):
            c_d, c_h, cmap, clabel = gc_vals, gc_vals, 'viridis', 'GC (%)'
        elif color_option.startswith("MSE"):
            c_d, c_h, cmap, clabel = mse_dev, mse_hk, 'inferno_r', 'MSE'
        else:
            c_d, c_h, cmap, clabel = None, None, None, None

        if c_d is not None:
            sc1 = axes2[0].scatter(
                pred_dev, pred_dev_rc, c=c_d, cmap=cmap,
                alpha=0.5, s=8, rasterized=True,
            )
            sc2 = axes2[1].scatter(
                pred_hk, pred_hk_rc, c=c_h, cmap=cmap,
                alpha=0.5, s=8, rasterized=True,
            )
            fig2.colorbar(sc1, ax=axes2[0], label=clabel)
            fig2.colorbar(sc2, ax=axes2[1], label=clabel)
        else:
            hb1 = axes2[0].hexbin(pred_dev, pred_dev_rc, gridsize=50, bins='log')
            hb2 = axes2[1].hexbin(pred_hk, pred_hk_rc, gridsize=50, bins='log')
            fig2.colorbar(hb1, ax=axes2[0], label='log10(count)')
            fig2.colorbar(hb2, ax=axes2[1], label='log10(count)')

        for ax in axes2:
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]
            ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
            adjust_axes(ax)

        axes2[0].set_title(f'Dev (RC PCC = {pcc_rc_dev:.3f})')
        axes2[1].set_title(f'Hk (RC PCC = {pcc_rc_hk:.3f})')
        axes2[0].set_xlabel('Forward')
        axes2[0].set_ylabel('Reverse Complement')
        axes2[1].set_xlabel('Forward')

        fig2.suptitle(f'{model_name} — Strand Bias Analysis', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig2)
        st.download_button(
            "Download plot", fig_to_bytes(fig2),
            f"{model_name}_strand_bias.png", "image/png",
        )

    # ========== TAB 3: DEV vs HK ==========
    with tab3:
        st.subheader("Developmental vs Housekeeping Correlation")
        pcc_cross_true = pearsonr(true_dev, true_hk)[0]
        pcc_cross_pred = pearsonr(pred_dev, pred_hk)[0]

        fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))

        hb1 = axes3[0].hexbin(
            true_dev, true_hk, gridsize=60, bins='log', cmap='viridis'
        )
        axes3[0].set_title(f'Observed (PCC = {pcc_cross_true:.3f})')
        axes3[0].set_xlabel('Dev [log2]')
        axes3[0].set_ylabel('Hk [log2]')
        adjust_axes(axes3[0])
        fig3.colorbar(hb1, ax=axes3[0])

        hb2 = axes3[1].hexbin(
            pred_dev, pred_hk, gridsize=60, bins='log', cmap='viridis'
        )
        axes3[1].set_title(f'Predicted (PCC = {pcc_cross_pred:.3f})')
        axes3[1].set_xlabel('Pred Dev [log2]')
        axes3[1].set_ylabel('Pred Hk [log2]')
        adjust_axes(axes3[1])
        fig3.colorbar(hb2, ax=axes3[1])

        for ax in axes3:
            ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.4)
            ax.axvline(0, color='gray', ls='--', lw=0.8, alpha=0.4)

        fig3.suptitle(f'{model_name} — Dev vs Hk Structure', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig3)
        st.download_button(
            "Download plot", fig_to_bytes(fig3),
            f"{model_name}_dev_vs_hk.png", "image/png",
        )

    # ========== TAB 4: ERROR DISTRIBUTION ==========
    with tab4:
        st.subheader("Prediction Error Distribution")
        fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))

        residuals_dev = pred_dev - true_dev
        residuals_hk = pred_hk - true_hk

        axes4[0].hist(
            residuals_dev, bins=80, color='steelblue', alpha=0.8,
            edgecolor='white', linewidth=0.5,
        )
        axes4[0].axvline(0, color='red', ls='--', lw=1)
        axes4[0].set_title(
            f'Dev Residuals (mean={np.mean(residuals_dev):.3f}, '
            f'std={np.std(residuals_dev):.3f})'
        )
        axes4[0].set_xlabel('Predicted - Observed')
        adjust_axes(axes4[0])

        axes4[1].hist(
            residuals_hk, bins=80, color='coral', alpha=0.8,
            edgecolor='white', linewidth=0.5,
        )
        axes4[1].axvline(0, color='red', ls='--', lw=1)
        axes4[1].set_title(
            f'Hk Residuals (mean={np.mean(residuals_hk):.3f}, '
            f'std={np.std(residuals_hk):.3f})'
        )
        axes4[1].set_xlabel('Predicted - Observed')
        adjust_axes(axes4[1])

        fig4.suptitle(f'{model_name} — Error Distribution', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig4)
        st.download_button(
            "Download plot", fig_to_bytes(fig4),
            f"{model_name}_errors.png", "image/png",
        )

else:
    st.info(
        "Upload model weights (.pth), FASTA sequences, and activity file "
        "in the sidebar to begin. Config YAML is optional — "
        "the app will auto-detect the model architecture from weight keys."
    )
