#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streamlit application for explainable AI analysis of DNA expression models.

Supports Saliency, SmoothGrad, Integrated Gradients, DeepLift,
GradientShap, Feature Ablation, Grad-CAM, and In-silico Mutagenesis.
"""

import os
import sys
import io
import yaml
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*hooks and attributes.*")

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from captum.attr import (
    IntegratedGradients, LayerGradCam, GradientShap,
    Saliency, DeepLift, DeepLiftShap, FeatureAblation, NoiseTunnel
)

# Ensure parent directory is on sys.path for project imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.registry import build_model
from utils import get_reverse_complement, one_hot_encode_dna

# ---------------------------------------------------------------------------
# DNA base colour palette (UCSC convention)
# ---------------------------------------------------------------------------
BASE_COLORS = {'A': '#109618', 'C': '#3366CC', 'T': '#DC3912', 'G': '#FF9900'}
BASE_ORDER  = ['A', 'C', 'T', 'G']


def _draw_seq_logo(attr_4xL: np.ndarray, ax, title: str = "", max_cols: int = 250):
    """Render a 4×L attribution matrix as a two-sided sequence logo.

    Positive attributions are stacked above zero, negative below.
    Each letter is coloured by base type and scaled by attribution magnitude.
    """
    L = min(attr_4xL.shape[1], max_cols)
    arr = attr_4xL[:, :L].copy()
    max_h = max(np.abs(arr).max(), 1e-8)
    arr /= max_h                           # normalise to [-1, 1]
    for pos in range(L):
        pos_stack = sorted([(arr[i, pos], BASE_ORDER[i]) for i in range(4) if arr[i, pos] >= 0], reverse=True)
        neg_stack = sorted([(arr[i, pos], BASE_ORDER[i]) for i in range(4) if arr[i, pos] < 0])
        y = 0.0
        for h, base in pos_stack:
            fs = max(3, int(8 * h))
            ax.text(pos + 0.5, y + h * 0.5, base,
                    ha='center', va='center', fontsize=fs,
                    color=BASE_COLORS[base], fontweight='bold', fontfamily='monospace',
                    clip_on=True)
            y += h
        y = 0.0
        for h, base in neg_stack:
            fs = max(3, int(8 * abs(h)))
            ax.text(pos + 0.5, y + h * 0.5, base,
                    ha='center', va='center', fontsize=fs,
                    color=BASE_COLORS[base], fontweight='bold', fontfamily='monospace',
                    clip_on=True)
            y += h
    ax.set_xlim(0, L)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("Position")
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _ism_heatmap(ism_matrix: np.ndarray, seq_str: str, ax, title: str = ""):
    """Render a 4×L ISM matrix as a heatmap (RdBu_r) with original-sequence annotation."""
    L = ism_matrix.shape[1]
    v  = max(float(np.abs(ism_matrix).max()), 1e-8)
    im = ax.imshow(ism_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-v, vmax=v, interpolation='nearest')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(BASE_ORDER, fontsize=10)
    fs = 4 if L > 120 else 6
    for pos, base in enumerate(seq_str[:L]):
        ax.text(pos, -0.72, base, ha='center', va='center',
                fontsize=fs, color=BASE_COLORS.get(base, 'black'), fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Δ activity')
    ax.set_title(title)
    ax.set_xlabel("Position")


class ModelWrapper(nn.Module):
    def __init__(self, model, target_neuron):
        super().__init__()
        self.model = model
        self.target_neuron = target_neuron

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            stacked = torch.cat([out[0], out[1]], dim=1)
        else:
            stacked = out
        return stacked[:, self.target_neuron].unsqueeze(1)


@st.cache_resource
def load_dynamic_model(config_dict, weights_bytes):
    """Load model from YAML config and weight buffer."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config_dict).to(device)
    model.load_state_dict(torch.load(io.BytesIO(weights_bytes), map_location=device, weights_only=True))
    model.eval()
    return model, device


def generate_combined_figure(plot_items, is_heatmap):
    """Generate a combined attribution plot from collected items."""
    num_plots = len(plot_items)
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 3 * num_plots))
    if num_plots == 1:
        axes = [axes]
        
    for ax, item in zip(axes, plot_items):
        attr = item['data']
        title = item['title']
        color = item.get('color', 'black')
        
        seq_len = attr.shape[-1]
        
        if is_heatmap:
            # Expand 1D to 2D for Grad-CAM in heatmap mode
            if len(attr.shape) == 1:
                attr = np.tile(attr, (4, 1))
            
            v_max = np.max(np.abs(attr))
            v_min = -v_max if v_max > 0 else -1e-5
            im = ax.imshow(attr, aspect='auto', cmap='RdBu_r', vmin=v_min, vmax=v_max)
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(['A', 'C', 'T', 'G'])
            ax.set_title(title)
        else:
            # 1D line mode
            if len(attr.shape) == 2:
                attr_1d = np.sum(attr, axis=0)
            else:
                attr_1d = attr
                
            ax.plot(range(seq_len), attr_1d, color=color if color else 'black', lw=1)
            ax.fill_between(range(seq_len), 0, attr_1d, where=(attr_1d >= 0), color='red', alpha=0.5)
            ax.fill_between(range(seq_len), 0, attr_1d, where=(attr_1d < 0), color='blue', alpha=0.5)
            ax.set_xlim(0, seq_len)
            ax.set_title(title)
            
    plt.tight_layout()
    return fig


def run_ism(model, input_tensor, target_idx):
    """Run In-silico Mutagenesis for all positions and bases."""
    model.eval()
    with torch.no_grad():
        out = model(input_tensor)
        orig_val = (out[target_idx] if isinstance(out, (list, tuple)) else out[:, target_idx]).item()

    seq_len = input_tensor.shape[2]
    ism_matrix = np.zeros((4, seq_len))
    
    for pos in range(seq_len):
        for base in range(4):
            mutated_input = input_tensor.clone()
            mutated_input[0, :, pos] = 0
            mutated_input[0, base, pos] = 1
            
            with torch.no_grad():
                m_out = model(mutated_input)
                m_val = (m_out[target_idx] if isinstance(m_out, (list, tuple)) else m_out[:, target_idx]).item()
                ism_matrix[base, pos] = m_val - orig_val
                
    return ism_matrix


# --- Streamlit UI Configuration ---
try:
    st.set_page_config(page_title="Universal DNA Model XAI Explorer", layout="wide")
except st.errors.StreamlitAPIException:
    pass
st.title("Universal DNA Model XAI Explorer")

st.sidebar.header("1. Upload Files")
config_file = st.sidebar.file_uploader("Upload Config (.yaml)", type=["yaml", "yml"])
weights_file = st.sidebar.file_uploader("Upload Weights (.pth)", type=["pth"])
fasta_file = st.sidebar.file_uploader("Upload Sequences (.fasta)", type=["fa", "fasta"])

st.sidebar.header("2. XAI Settings")
view_mode = st.sidebar.selectbox("Visualization Mode", ["2D (Heatmap)", "1D (Line)", "Sequence Logo"])
target_choice = st.sidebar.radio("Analysis Target", ["Developmental (0)", "Housekeeping (1)"])
target_idx = 0 if "Developmental" in target_choice else 1
batch_mode = st.sidebar.checkbox("Batch mode (compare multiple sequences)", value=False)

st.sidebar.subheader("Algorithm Parameters")
num_ig_steps = st.sidebar.number_input("IG Steps", 25, 500, 100)
num_baselines_2d = st.sidebar.number_input("GradientShap Baselines", 5, 200, 25)
sg_samples = st.sidebar.number_input("SmoothGrad Samples", 5, 100, 10)
sg_stdev = st.sidebar.slider("SmoothGrad Noise (σ)", 0.01, 0.5, 0.15)


if config_file and weights_file and fasta_file:
    # Load model from YAML
    try:
        config = yaml.safe_load(config_file)
        model, device = load_dynamic_model(config, weights_file.getvalue())
        wrapped_model = ModelWrapper(model, target_idx)
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

    fasta_content = fasta_file.getvalue().decode("utf-8")
    records = list(SeqIO.parse(io.StringIO(fasta_content), "fasta"))
    record_names = [f"{r.id} ({len(r.seq)} bp)" for r in records]
    
    if batch_mode:
        selected_indices = st.multiselect(
            "Select sequences (batch)",
            range(len(records)),
            default=[0],
            format_func=lambda i: record_names[i],
        )
        if not selected_indices:
            st.warning("Select at least one sequence.")
            st.stop()
        selected_idx = selected_indices[0]   # primary for full XAI
    else:
        selected_indices = None
        selected_idx = st.selectbox("Select DNA Sequence", range(len(records)),
                                    format_func=lambda i: record_names[i])
    selected_record = records[selected_idx]
    
    seq_str = str(selected_record.seq).upper()
    seq_id = selected_record.id
    st.session_state['seq_id'] = seq_id

    # Compute reverse complement
    seq_str_rc = get_reverse_complement(seq_str)
    
    # Tensors
    input_fwd = torch.from_numpy(one_hot_encode_dna([seq_str])).float().to(device)
    input_fwd.requires_grad = True
    
    input_rc = torch.from_numpy(one_hot_encode_dna([seq_str_rc])).float().to(device)
    input_rc.requires_grad = True

    if st.button("Run XAI Analysis"):
        with st.spinner("Calculating Attributions (Forward & RC)..."):
            
            # --- FORWARD STRAND ---
            plot_items_fwd = []
            
            # Saliency
            sl = Saliency(wrapped_model)
            attr_sl_fwd = sl.attribute(input_fwd).squeeze().cpu().detach().numpy()
            plot_items_fwd.append({'data': attr_sl_fwd, 'title': 'Saliency', 'color': 'gray'})
            
            # 2. SmoothGrad
            nt = NoiseTunnel(Saliency(wrapped_model))
            attr_sg_fwd = nt.attribute(input_fwd, nt_type='smoothgrad', nt_samples=sg_samples, stdevs=sg_stdev).squeeze().cpu().detach().numpy()
            plot_items_fwd.append({'data': attr_sg_fwd, 'title': 'SmoothGrad', 'color': 'darkgray'})

            # 3. IG
            ig = IntegratedGradients(wrapped_model)
            attr_ig_fwd = ig.attribute(input_fwd, n_steps=num_ig_steps).squeeze().cpu().detach().numpy()
            plot_items_fwd.append({'data': attr_ig_fwd, 'title': 'Integrated Gradients', 'color': 'black'})
            
            # 4. DeepLift
            dl = DeepLift(wrapped_model)
            attr_dl_fwd = dl.attribute(input_fwd).squeeze().cpu().detach().numpy()
            plot_items_fwd.append({'data': attr_dl_fwd, 'title': 'DeepLift', 'color': 'blue'})
            
            # 4b. DeepSHAP (DeepLiftShap) — multiple random baselines
            dls = DeepLiftShap(wrapped_model)
            bg_baselines = torch.zeros(20, *input_fwd.shape[1:], device=input_fwd.device)
            attr_dls_fwd = dls.attribute(input_fwd, baselines=bg_baselines).squeeze().cpu().detach().numpy()
            plot_items_fwd.append({'data': attr_dls_fwd, 'title': 'DeepSHAP', 'color': 'navy'})
            
            # 5. GradientShap
            gs = GradientShap(wrapped_model)
            attr_gs_fwd = gs.attribute(input_fwd, baselines=torch.zeros_like(input_fwd), n_samples=num_baselines_2d).squeeze().cpu().detach().numpy()
            plot_items_fwd.append({'data': attr_gs_fwd, 'title': 'GradientShap', 'color': 'green'})
            
            # 6. Feature Ablation
            fa = FeatureAblation(wrapped_model)
            attr_fa_fwd = fa.attribute(input_fwd).squeeze().cpu().detach().numpy()
            plot_items_fwd.append({'data': attr_fa_fwd, 'title': 'Feature Ablation', 'color': 'red'})
            
            # Layer Grad-CAM (find Conv1d layers)
            conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv1d)]
            if len(conv_layers) >= 2:
                lgc_first = LayerGradCam(wrapped_model, conv_layers[0])
                attr_gc_first_fwd = LayerGradCam.interpolate(lgc_first.attribute(input_fwd), input_fwd.shape[2]).squeeze().cpu().detach().numpy()
                
                lgc_last = LayerGradCam(wrapped_model, conv_layers[-1])
                attr_gc_last_fwd = LayerGradCam.interpolate(lgc_last.attribute(input_fwd), input_fwd.shape[2]).squeeze().cpu().detach().numpy()
                
                plot_items_fwd.append({'data': attr_gc_first_fwd, 'title': 'Grad-CAM (First Conv)', 'color': 'mediumorchid'})
                plot_items_fwd.append({'data': attr_gc_last_fwd, 'title': 'Grad-CAM (Last Conv)', 'color': 'purple'})
            
            st.session_state['plot_items_fwd'] = plot_items_fwd


            # --- REVERSE COMPLEMENT STRAND ---
            plot_items_rc = []
            
            attr_sl_rc = sl.attribute(input_rc).squeeze().cpu().detach().numpy()
            plot_items_rc.append({'data': attr_sl_rc, 'title': 'Saliency (RC)', 'color': 'gray'})
            
            attr_sg_rc = nt.attribute(input_rc, nt_type='smoothgrad', nt_samples=sg_samples, stdevs=sg_stdev).squeeze().cpu().detach().numpy()
            plot_items_rc.append({'data': attr_sg_rc, 'title': 'SmoothGrad (RC)', 'color': 'darkgray'})
            
            attr_ig_rc = ig.attribute(input_rc, n_steps=num_ig_steps).squeeze().cpu().detach().numpy()
            plot_items_rc.append({'data': attr_ig_rc, 'title': 'Integrated Gradients (RC)', 'color': 'black'})
            
            attr_dl_rc = dl.attribute(input_rc).squeeze().cpu().detach().numpy()
            plot_items_rc.append({'data': attr_dl_rc, 'title': 'DeepLift (RC)', 'color': 'blue'})
            
            bg_baselines_rc = torch.zeros(20, *input_rc.shape[1:], device=input_rc.device)
            attr_dls_rc = dls.attribute(input_rc, baselines=bg_baselines_rc).squeeze().cpu().detach().numpy()
            plot_items_rc.append({'data': attr_dls_rc, 'title': 'DeepSHAP (RC)', 'color': 'navy'})
            
            attr_gs_rc = gs.attribute(input_rc, baselines=torch.zeros_like(input_rc), n_samples=num_baselines_2d).squeeze().cpu().detach().numpy()
            plot_items_rc.append({'data': attr_gs_rc, 'title': 'GradientShap (RC)', 'color': 'green'})
            
            attr_fa_rc = fa.attribute(input_rc).squeeze().cpu().detach().numpy()
            plot_items_rc.append({'data': attr_fa_rc, 'title': 'Feature Ablation (RC)', 'color': 'red'})
            
            if len(conv_layers) >= 2:
                attr_gc_first_rc = LayerGradCam.interpolate(lgc_first.attribute(input_rc), input_rc.shape[2]).squeeze().cpu().detach().numpy()
                attr_gc_last_rc = LayerGradCam.interpolate(lgc_last.attribute(input_rc), input_rc.shape[2]).squeeze().cpu().detach().numpy()
                
                plot_items_rc.append({'data': attr_gc_first_rc, 'title': 'Grad-CAM (First Conv) [RC]', 'color': 'mediumorchid'})
                plot_items_rc.append({'data': attr_gc_last_rc, 'title': 'Grad-CAM (Last Conv) [RC]', 'color': 'purple'})
                
            st.session_state['plot_items_rc'] = plot_items_rc

            # --- ISM ANALYSIS ---
            st.session_state['ism_matrix_fwd'] = run_ism(model, input_fwd, target_idx)
            st.session_state['ism_matrix_rc'] = run_ism(model, input_rc, target_idx)
            
            st.session_state['view_mode'] = view_mode


    # --- Display Results ---
    if 'plot_items_fwd' in st.session_state:
        st.subheader(f"Results for: `{st.session_state['seq_id']}`")
        is_hm   = "Heatmap" in st.session_state['view_mode']
        is_logo = "Logo"    in st.session_state['view_mode']

        _tab_labels = ["Forward Strand XAI", "Reverse Complement XAI", "In-silico Mutagenesis"]
        if batch_mode and selected_indices and len(selected_indices) > 1:
            _tab_labels.append("Batch ISM Comparison")
        tab_fwd, tab_rc, tab_ism, *_extra_tabs = st.tabs(_tab_labels)
        tab_batch = _extra_tabs[0] if _extra_tabs else None
        
        # TAB 1: FORWARD
        with tab_fwd:
            if is_logo:
                # Find IG item for logo rendering
                ig_item = next((x for x in st.session_state['plot_items_fwd']
                                if 'Integrated' in x['title']), None)
                if ig_item is not None and len(ig_item['data'].shape) == 2:
                    fig_logo, ax_logo = plt.subplots(figsize=(min(len(ig_item['data'][0]) * 0.08 + 2, 22), 3))
                    _draw_seq_logo(ig_item['data'], ax_logo,
                                   title=f"IG Attribution Logo — {st.session_state['seq_id']}")
                    st.pyplot(fig_logo)
                    buf_l = io.BytesIO()
                    fig_logo.savefig(buf_l, format='png', bbox_inches='tight', dpi=180)
                    buf_l.seek(0)
                    st.download_button("Download logo PNG", buf_l,
                                       f"logo_{st.session_state['seq_id']}.png", "image/png",
                                       key="dl_logo_fwd")
                st.markdown("---")
            combined_fig_fwd = generate_combined_figure(st.session_state['plot_items_fwd'], is_hm)
            st.pyplot(combined_fig_fwd)
            
            buf_fwd = io.BytesIO()
            combined_fig_fwd.savefig(buf_fwd, format="png", bbox_inches='tight', dpi=150)
            buf_fwd.seek(0)
            st.download_button(
                label="Download Combined Plot FWD (PNG)",
                data=buf_fwd,
                file_name=f"XAI_FWD_{st.session_state['seq_id']}.png",
                mime="image/png"
            )

        # TAB 2: REVERSE COMPLEMENT
        with tab_rc:
            combined_fig_rc = generate_combined_figure(st.session_state['plot_items_rc'], is_hm)
            st.pyplot(combined_fig_rc)
            
            buf_rc = io.BytesIO()
            combined_fig_rc.savefig(buf_rc, format="png", bbox_inches='tight', dpi=150)
            buf_rc.seek(0)
            st.download_button(
                label="Download Combined Plot RC (PNG)",
                data=buf_rc,
                file_name=f"XAI_RC_{st.session_state['seq_id']}.png",
                mime="image/png"
            )

        # TAB 3: ISM
        with tab_ism:
            st.markdown("### Forward Strand ISM")
            ism_fwd = st.session_state['ism_matrix_fwd']
            fig_ism_fwd, axes_ism_fwd = plt.subplots(2, 1, figsize=(max(len(seq_str) * 0.06 + 2, 15), 6))
            _ism_heatmap(ism_fwd, seq_str,
                         axes_ism_fwd[0], title="ISM heatmap — Δ activity per mutation (Forward)")
            _draw_seq_logo(ism_fwd, axes_ism_fwd[1],
                           title="ISM attribution logo (Forward) — blue=enhancing, red=disruptive")
            plt.tight_layout()
            st.pyplot(fig_ism_fwd)
            
            # Przetwarzanie ISM FWD do DataFrame
            bases = ['A', 'C', 'T', 'G']
            data_fwd = []
            for pos in range(ism_fwd.shape[1]):
                orig_base = bases[np.argmax(input_fwd[0, :, pos].cpu().detach().numpy())]
                for b_idx, mut_base in enumerate(bases):
                    if mut_base != orig_base:
                        data_fwd.append({'Position': pos, 'Original': orig_base, 'Mutated': mut_base, 'Difference': ism_fwd[b_idx, pos]})
            ism_df_fwd = pd.DataFrame(data_fwd).sort_values(by='Difference', ascending=False)
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("🔴 **Top 10 Destructive Mutations (FWD)**")
                st.dataframe(ism_df_fwd.tail(10).iloc[::-1][['Position', 'Original', 'Mutated', 'Difference']], width=500)
            with col_t2:
                st.markdown("🟢 **Top 10 Enhancing Mutations (FWD)**")
                st.dataframe(ism_df_fwd.head(10)[['Position', 'Original', 'Mutated', 'Difference']], width=500)


            st.markdown("---")
            st.markdown("### Reverse Complement ISM")
            ism_rc = st.session_state['ism_matrix_rc']
            fig_ism_rc, axes_ism_rc = plt.subplots(2, 1, figsize=(max(len(seq_str_rc) * 0.06 + 2, 15), 6))
            _ism_heatmap(ism_rc, seq_str_rc,
                         axes_ism_rc[0], title="ISM heatmap — Δ activity per mutation (RC)")
            _draw_seq_logo(ism_rc, axes_ism_rc[1],
                           title="ISM attribution logo (RC)")
            plt.tight_layout()
            st.pyplot(fig_ism_rc)
            
            # Przetwarzanie ISM RC do DataFrame
            data_rc = []
            for pos in range(ism_rc.shape[1]):
                orig_base_rc = bases[np.argmax(input_rc[0, :, pos].cpu().detach().numpy())]
                for b_idx, mut_base_rc in enumerate(bases):
                    if mut_base_rc != orig_base_rc:
                        data_rc.append({'Position': pos, 'Original': orig_base_rc, 'Mutated': mut_base_rc, 'Difference': ism_rc[b_idx, pos]})
            ism_df_rc = pd.DataFrame(data_rc).sort_values(by='Difference', ascending=False)
            
            col_t1_rc, col_t2_rc = st.columns(2)
            with col_t1_rc:
                st.markdown("🔴 **Top 10 Destructive Mutations (RC)**")
                st.dataframe(ism_df_rc.tail(10).iloc[::-1][['Position', 'Original', 'Mutated', 'Difference']], width=500)
            with col_t2_rc:
                st.markdown("🟢 **Top 10 Enhancing Mutations (RC)**")
                st.dataframe(ism_df_rc.head(10)[['Position', 'Original', 'Mutated', 'Difference']], width=500)

        # TAB 4 (optional): BATCH ISM COMPARISON
        if tab_batch is not None and batch_mode and selected_indices and len(selected_indices) > 1:
            with tab_batch:
                st.markdown("### Batch ISM comparison — selected sequences")
                n_batch = len(selected_indices)
                fig_b, axes_b = plt.subplots(n_batch, 1,
                                             figsize=(max(len(seq_str) * 0.06 + 2, 15), 4 * n_batch))
                if n_batch == 1:
                    axes_b = [axes_b]
                for ax_b, rec_idx in zip(axes_b, selected_indices):
                    rec      = records[rec_idx]
                    s        = str(rec.seq).upper()
                    if "_-_" in rec.id:
                        s = get_reverse_complement(s)
                    inp_b = torch.from_numpy(one_hot_encode_dna([s])).float().to(device)
                    with st.spinner(f"ISM for {rec.id}…"):
                        ism_b = run_ism(model, inp_b, target_idx)
                    _ism_heatmap(ism_b, s, ax_b,
                                 title=f"{rec.id} — ISM Δ activity")
                plt.tight_layout()
                st.pyplot(fig_b)
                buf_b = io.BytesIO()
                fig_b.savefig(buf_b, format='png', bbox_inches='tight', dpi=150)
                buf_b.seek(0)
                st.download_button("Download batch ISM PNG", buf_b,
                                   "batch_ism.png", "image/png", key="dl_batch_ism")

else:
    st.info("👈 Please upload the YAML config, weights (.pth) and a FASTA file in the sidebar to begin.")
