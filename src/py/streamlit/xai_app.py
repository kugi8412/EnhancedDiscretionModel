#!/usr/bin/env python
# -*- coding: utf-8 -*-
# xai_app.py

"""
This application build with Streamlit is designed to show different explainable
machine learning methods to show how model 'see' regulatory sequences.
"""

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
    Saliency, DeepLift, FeatureAblation, NoiseTunnel
)

from models.registry import build_model
from utils import get_reverse_complement, one_hot_encode_dna


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
    """Ładuje model na podstawie YAML i pakuje wagi z bufora."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config_dict).to(device)
    model.load_state_dict(torch.load(io.BytesIO(weights_bytes), map_location=device, weights_only=False))
    model.eval()
    return model, device


def generate_combined_figure(plot_items, is_heatmap):
    """Generuje połączony wykres na podstawie zebranych atrybucji."""
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
            # Rozszerzenie 1D do 2D dla Grad-CAM w trybie Heatmap
            if len(attr.shape) == 1:
                attr = np.tile(attr, (4, 1))
            
            v_max = np.max(np.abs(attr))
            v_min = -v_max if v_max > 0 else -1e-5
            im = ax.imshow(attr, aspect='auto', cmap='RdBu_r', vmin=v_min, vmax=v_max)
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(['A', 'C', 'T', 'G'])
            ax.set_title(title)
        else:
            # Tryb 1D (Line)
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
    """Prosta In-silico Mutagenesis."""
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


# --- KONFIGURACJA UI STREAMLIT ---
st.set_page_config(page_title="Universal DNA Model XAI Explorer", layout="wide")
st.title("🧬 Universal DNA Model XAI Explorer")

st.sidebar.header("📂 1. Wczytaj Pliki")
config_file = st.sidebar.file_uploader("Upload Config (.yaml)", type=["yaml", "yml"])
weights_file = st.sidebar.file_uploader("Upload Weights (.pth)", type=["pth"])
fasta_file = st.sidebar.file_uploader("Upload Sequences (.fasta)", type=["fa", "fasta"])

st.sidebar.header("⚙️ 2. Ustawienia XAI")
view_mode = st.sidebar.selectbox("Visualization Mode", ["2D (Heatmap)", "1D (Line)"])
target_choice = st.sidebar.radio("Analysis Target", ["Devanand (0)", "Housekeeping (1)"])
target_idx = 0 if "Devanand" in target_choice else 1

st.sidebar.subheader("Algorithm Parameters")
num_ig_steps = st.sidebar.number_input("IG Steps", 25, 500, 100)
num_baselines_2d = st.sidebar.number_input("GradientShap Baselines", 5, 200, 25)
sg_samples = st.sidebar.number_input("SmoothGrad Samples", 5, 100, 10)
sg_stdev = st.sidebar.slider("SmoothGrad Noise (σ)", 0.01, 0.5, 0.15)


if config_file and weights_file and fasta_file:
    # Ładowanie modelu z YAML
    try:
        config = yaml.safe_load(config_file)
        model, device = load_dynamic_model(config, weights_file.getvalue())
        wrapped_model = ModelWrapper(model, target_idx)
    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        st.stop()

    fasta_content = fasta_file.getvalue().decode("utf-8")
    records = list(SeqIO.parse(io.StringIO(fasta_content), "fasta"))
    record_names = [f"{r.id} ({len(r.seq)} bp)" for r in records]
    
    selected_idx = st.selectbox("Select DNA Sequence", range(len(records)), format_func=lambda i: record_names[i])
    selected_record = records[selected_idx]
    
    seq_str = str(selected_record.seq).upper()
    seq_id = selected_record.id
    st.session_state['seq_id'] = seq_id

    # Obliczanie Reverse Complement
    seq_str_rc = get_reverse_complement(seq_str)
    
    # Tensory
    input_fwd = torch.from_numpy(one_hot_encode_dna([seq_str])).float().to(device)
    input_fwd.requires_grad = True
    
    input_rc = torch.from_numpy(one_hot_encode_dna([seq_str_rc])).float().to(device)
    input_rc.requires_grad = True

    if st.button("Run XAI Analysis"):
        with st.spinner("Calculating Attributions (Forward & RC)..."):
            
            # --- FORWARD STRAND ---
            plot_items_fwd = []
            
            # 1. Saliency
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
            
            # 5. GradientShap
            gs = GradientShap(wrapped_model)
            attr_gs_fwd = gs.attribute(input_fwd, baselines=torch.zeros_like(input_fwd), n_samples=num_baselines_2d).squeeze().cpu().detach().numpy()
            plot_items_fwd.append({'data': attr_gs_fwd, 'title': 'GradientShap', 'color': 'green'})
            
            # 6. Feature Ablation
            fa = FeatureAblation(wrapped_model)
            attr_fa_fwd = fa.attribute(input_fwd).squeeze().cpu().detach().numpy()
            plot_items_fwd.append({'data': attr_fa_fwd, 'title': 'Feature Ablation', 'color': 'red'})
            
            # 7. Layer Grad-CAM (Szukanie warstw Conv1d)
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


    # --- WYSWIETLANIE WYNIKÓW ---
    if 'plot_items_fwd' in st.session_state:
        st.subheader(f"Results for: `{st.session_state['seq_id']}`")
        is_hm = "Heatmap" in st.session_state['view_mode']
        
        tab_fwd, tab_rc, tab_ism = st.tabs(["Forward Strand XAI", "Reverse Complement XAI", "In-silico Mutagenesis"])
        
        # TAB 1: FORWARD
        with tab_fwd:
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

        # TAB 3: ISM (Z TABELAMI)
        with tab_ism:
            st.markdown("### Forward Strand ISM")
            ism_fwd = st.session_state['ism_matrix_fwd']
            fig_ism_fwd, ax_fwd = plt.subplots(figsize=(15, 3))
            v_max = np.max(np.abs(ism_fwd))
            im_fwd = ax_fwd.imshow(ism_fwd, aspect='auto', cmap='PiYG', vmin=-v_max, vmax=v_max)
            ax_fwd.set_yticks([0, 1, 2, 3])
            ax_fwd.set_yticklabels(['A', 'C', 'T', 'G'])
            plt.colorbar(im_fwd, ax=ax_fwd)
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
            fig_ism_rc, ax_rc = plt.subplots(figsize=(15, 3))
            v_max_rc = np.max(np.abs(ism_rc))
            im_rc = ax_rc.imshow(ism_rc, aspect='auto', cmap='PiYG', vmin=-v_max_rc, vmax=v_max_rc)
            ax_rc.set_yticks([0, 1, 2, 3])
            ax_rc.set_yticklabels(['A', 'C', 'T', 'G'])
            plt.colorbar(im_rc, ax=ax_rc)
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

else:
    st.info("👈 Please upload the YAML config, weights (.pth) and a FASTA file in the sidebar to begin.")
