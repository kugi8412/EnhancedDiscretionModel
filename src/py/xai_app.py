#!/usr/bin/env python
# -*- coding: utf-8 -*-
# xai_app.py

"""
This application build with Streamlit is designed to show different explainable
machine learning methods to show how model (DeepSTARR) 'see' regulatory sequences.
"""


NUM_IG_STEPS = 100
NUM_BASELINES_2D = 25

import io
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
from captum.attr import IntegratedGradients, LayerGradCam, GradientShap, Saliency, DeepLift, FeatureAblation, DeepLiftShap

from models import DeepSTARR
from params import PARAMS, NUM_IG_STEPS, NUM_BASELINES_2D
from utils import get_reverse_complement, one_hot_encode_dna


class ModelWrapper(nn.Module):
    def __init__(self, model, target_neuron):
        super().__init__()
        self.model = model
        self.target_neuron = target_neuron

    def forward(self, x):
        out_dev, out_hk = self.model(x)
        stacked = torch.cat([out_dev, out_hk], dim=1)
        return stacked[:, self.target_neuron].unsqueeze(1)

@st.cache_resource
def load_model(weights_file):
    model = DeepSTARR(PARAMS)
    model.load_state_dict(torch.load(weights_file, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True))
    model.eval()
    return model

def parse_fasta(fasta_file):
    stringio = io.StringIO(fasta_file.getvalue().decode("utf-8"))
    sequences, ids = [], []
    for record in SeqIO.parse(stringio, "fasta"):
        sequences.append(str(record.seq).upper())
        ids.append(record.id)
    return ids, sequences

def process_data(data, normalize):
    if normalize:
        max_val = np.max(np.abs(data))
        if max_val > 0: return data / max_val
    return data

def calculate_ism(seq_str, wrapper):
    """
    Compute In-Silico Mutagenesis (all single-nucleotide mutations)
    and return a DataFrame with results sorted by impact.
    """
    nucs = ['A', 'C', 'T', 'G']
    wt_encoded = one_hot_encode_dna([seq_str])
    wt_tensor = torch.tensor(wt_encoded, dtype=torch.float32)
    
    wrapper.eval()
    with torch.no_grad():
        wt_pred = wrapper(wt_tensor).item()
        
    mutants = []
    mut_info = []
    
    for pos in range(len(seq_str)):
        orig_nuc = seq_str[pos]
        for alt_nuc in nucs:
            if alt_nuc != orig_nuc:
                mut_seq = seq_str[:pos] + alt_nuc + seq_str[pos+1:]
                mutants.append(mut_seq)
                mut_info.append({'Position': pos, 'Original': orig_nuc, 'Mutated': alt_nuc})
                
    mut_encoded = one_hot_encode_dna(mutants)
    mut_tensor = torch.tensor(mut_encoded, dtype=torch.float32)
    
    preds = []
    with torch.no_grad():
        for i in range(0, len(mut_tensor), 128):
            batch = mut_tensor[i:i+128]
            out = wrapper(batch).squeeze(-1).numpy()
            preds.extend(out)
            
    df = pd.DataFrame(mut_info)
    df['WT_Prediction'] = wt_pred
    df['Mutant_Prediction'] = preds
    df['Difference'] = df['Mutant_Prediction'] - df['WT_Prediction']
    df = df.sort_values(by='Difference').reset_index(drop=True)
    return df

def compute_xai_for_sequence(seq_str, wrapper, custom_bg_arr, is_heatmap, should_normalize, sel_methods):
    seq_encoded = one_hot_encode_dna([seq_str]) 
    input_tensor = torch.tensor(seq_encoded, dtype=torch.float32, requires_grad=True)
    
    # 1D Background (IG; DeepLIFT; Ablation)
    custom_bg = torch.tensor(custom_bg_arr).view(1, 4, 1).repeat(1, 1, input_tensor.shape[2])

    # Background x 25 (DeepSHAP; GradSHAP)
    if sel_methods.get('DeepSHAP') or sel_methods.get('GradSHAP'):
        probs = custom_bg_arr / np.sum(custom_bg_arr)
        bg_seqs = ["".join(np.random.choice(['A', 'C', 'T', 'G'], size=len(seq_str), p=probs)) for _ in range(NUM_BASELINES_2D)]
        bg_tensor = torch.tensor(one_hot_encode_dna(bg_seqs), dtype=torch.float32)
    
    results = {}
    
    def format_output(tensor_val):
        if is_heatmap:
            return process_data(tensor_val.squeeze().detach().numpy(), should_normalize)
        else:
            return process_data(torch.sum(tensor_val * input_tensor, dim=1).squeeze().detach().numpy(), should_normalize)

    if sel_methods.get('Saliency'):
        _ = Saliency(wrapper)
        wrapper.zero_grad()
        out = wrapper(input_tensor)
        out.backward()
        results['Saliency'] = format_output(input_tensor.grad.detach())
        
    if sel_methods.get('IG'):
        ig = IntegratedGradients(wrapper)
        results['Integrated Gradients'] = format_output(ig.attribute(input_tensor,
                                                                     baselines=custom_bg,
                                                                     n_steps=NUM_IG_STEPS)
                                                        )
        
    if sel_methods.get('DeepLIFT'):
        dl = DeepLift(wrapper)
        results['DeepLIFT'] = format_output(dl.attribute(input_tensor, baselines=custom_bg))
        
    if sel_methods.get('DeepSHAP'):
        dlshap = DeepLiftShap(wrapper)
        attr_dlshap = dlshap.attribute(input_tensor, baselines=bg_tensor)
        results['DeepLIFT-Shap'] = format_output(attr_dlshap)

    if sel_methods.get('GradSHAP'):
        gshap = GradientShap(wrapper)
        attr_gshap = gshap.attribute(input_tensor, baselines=bg_tensor)
        if attr_gshap.shape == (1, input_tensor.shape[2], 4): attr_gshap = attr_gshap.transpose(1, 2)
        results['Gradient SHAP'] = format_output(attr_gshap)
        
    if sel_methods.get('Ablation'):
        ablation = FeatureAblation(wrapper)
        results['Feature Ablation'] = format_output(ablation.attribute(input_tensor, baselines=custom_bg))

    if sel_methods.get('GradCAM'):
        conv_modules = [m for m in wrapper.model.modules() if isinstance(m, nn.Conv1d)]
        if len(conv_modules) > 0:
            for idx, layer_name in zip([0, -1], ['Conv1', 'Conv4']):
                layer_gc = LayerGradCam(wrapper, conv_modules[idx])
                attr_gc_up = F.interpolate(layer_gc.attribute(input_tensor), size=input_tensor.shape[2], mode='linear', align_corners=False).squeeze().detach().numpy()
                attr_gc_up = np.maximum(attr_gc_up, 0)
                
                if is_heatmap: results[f'Grad-CAM ({layer_name})'] = process_data(np.tile(attr_gc_up, (4, 1)), should_normalize)
                else: results[f'Grad-CAM ({layer_name})'] = process_data(attr_gc_up, should_normalize)

    return results

def draw_1d_axis(ax, data, title, color, is_normalized):
    ax.bar(range(len(data)), data, color=color, alpha=0.8, width=1.0)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Attribution")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlim(0, len(data))
    if is_normalized: ax.set_ylim(-1.05, 1.05)

def draw_heatmap_axis(fig, ax, data, title, is_normalized):
    vmax = 1.0 if is_normalized else np.max(np.abs(data))
    if vmax == 0: vmax = 1e-6 
    cax = ax.imshow(data, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['A', 'C', 'T', 'G'], fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.colorbar(cax, ax=ax, orientation='vertical', pad=0.01, label="Attribution")

def get_color_for_method(method_name):
    colors = {
        'Saliency': 'teal', 'Integrated Gradients': 'forestgreen',
        'DeepLIFT': 'royalblue', 'DeepLIFT-Shap': 'dodgerblue',
        'Gradient SHAP': 'darkorange', 'Feature Ablation': 'crimson',
        'Grad-CAM (Conv1)': 'mediumorchid', 'Grad-CAM (Conv4)': 'purple'
    }
    return colors.get(method_name, 'gray')

def generate_vertical_figure(plots_data, is_heatmap, is_normalized):
    methods = list(plots_data.keys())
    fig = plt.figure(figsize=(16, 3.5 * len(methods)))
    gs = fig.add_gridspec(len(methods), 1, hspace=0.45)
    
    for row, method in enumerate(methods):
        ax = fig.add_subplot(gs[row, 0])
        color = get_color_for_method(method)
        if is_heatmap: draw_heatmap_axis(fig, ax, plots_data[method], method, is_normalized)
        else: draw_1d_axis(ax, plots_data[method], method, color, is_normalized)
        if row == len(methods) - 1: ax.set_xlabel("Sequence Position (bp)", fontsize=12)
            
    return fig

def generate_symmetrical_figure(plots_fwd, plots_rc, is_heatmap, is_normalized):
    methods = list(plots_fwd.keys())
    fig = plt.figure(figsize=(20, 3.5 * len(methods)))
    gs = fig.add_gridspec(len(methods), 2, hspace=0.45, wspace=0.15)
    
    for row, method in enumerate(methods):
        color = get_color_for_method(method)
        ax_fwd = fig.add_subplot(gs[row, 0])
        ax_rc = fig.add_subplot(gs[row, 1])
        
        if is_heatmap:
            draw_heatmap_axis(fig, ax_fwd, plots_fwd[method], f"{method} (FWD)", is_normalized)
            draw_heatmap_axis(fig, ax_rc, plots_rc[method], f"{method} (RC)", is_normalized)
        else:
            draw_1d_axis(ax_fwd, plots_fwd[method], f"{method} (FWD)", color, is_normalized)
            draw_1d_axis(ax_rc, plots_rc[method], f"{method} (RC)", color, is_normalized)
            
        if row == len(methods) - 1:
            ax_fwd.set_xlabel("Sequence Position (bp)", fontsize=12)
            ax_rc.set_xlabel("Sequence Position (bp)", fontsize=12)
            
    return fig


def generate_tsv(plots_fwd, plots_rc, is_heatmap, seq_id):
    records = []
    def add_to_records(res_dict, direction):
        if res_dict is None:
            return None

        for method, data in res_dict.items():
            if is_heatmap:
                for pos in range(data.shape[1]):
                    records.append({'SeqID': seq_id,
                                    'Method': method,
                                    'Direction': direction,
                                    'Position': pos,
                                    'A': data[0, pos],
                                    'C': data[1, pos],
                                    'T': data[2, pos],
                                    'G': data[3, pos]}
                                )
            else:
                for pos in range(len(data)):
                    records.append({'SeqID': seq_id, 'Method': method, 'Direction': direction, 'Position': pos, 'Attribution': data[pos]})

    add_to_records(plots_fwd, 'FWD')
    add_to_records(plots_rc, 'RC')
    return pd.DataFrame(records).to_csv(index=False, sep='\t')


st.set_page_config(layout="wide", page_title="DeepSTARR XAI Explorer")
st.title("🧬 DeepSTARR XAI Explorer")

if 'plots_fwd' not in st.session_state:
    st.session_state['plots_fwd'] = None

if 'plots_rc' not in st.session_state:
    st.session_state['plots_rc'] = None

if 'ism_df' not in st.session_state:
    st.session_state['ism_df'] = None

if 'ism_df_rc' not in st.session_state:
    st.session_state['ism_df_rc'] = None

st.sidebar.header("1. Upload Files")
weights_file = st.sidebar.file_uploader("DeepSTARR Weights (.pth)", type=['pth'])
fasta_file = st.sidebar.file_uploader("Sequence Dataset (FASTA)", type=['fa', 'fasta'])

if weights_file is not None and fasta_file is not None:
    with st.spinner("Loading model..."):
        base_model = load_model(weights_file)
    
    ids, sequences = parse_fasta(fasta_file)

    st.sidebar.header("2. Experiment Settings")
    selected_idx = st.sidebar.selectbox("Select Sequence:", range(len(ids)), format_func=lambda x: f"{ids[x]}")
    target_neuron = st.sidebar.selectbox("Target Activity:", [0, 1], format_func=lambda x: "Developmental" if x == 0 else "Housekeeping")
    
    st.sidebar.header("3. Visualization Mode")
    view_mode = st.sidebar.radio("Display Option:",
                                 ["1D Profile (Present Nucleotides Only)",
                                  "Heatmap (Full 4x249 Matrix)"]
                                )
    is_heatmap = "Heatmap" in view_mode
    should_normalize = st.sidebar.checkbox("Normalize values to [-1, 1]", value=True)

    st.sidebar.header("4. Custom Baseline (A, C, T, G)")
    st.sidebar.caption("Write probability of each nucleotide")
    def parse_baseline(val_str, default=0.25):
        try:
            return float(val_str.replace(',', '.'))

        except ValueError: return default

    col1, col2, col3, col4 = st.sidebar.columns(4)
    base_a = parse_baseline(col1.text_input("A", value="0.25"))
    base_c = parse_baseline(col2.text_input("C", value="0.25"))
    base_t = parse_baseline(col3.text_input("T", value="0.25"))
    base_g = parse_baseline(col4.text_input("G", value="0.25"))
    custom_bg_arr = np.array([base_a, base_c, base_t, base_g], dtype=np.float32)

    st.sidebar.header("5. Select XAI Methods")
    sel_methods = {
        'Saliency': st.sidebar.checkbox("Saliency", value=True),
        'IG': st.sidebar.checkbox("Integrated Gradients", value=True),
        'DeepLIFT': st.sidebar.checkbox("DeepLIFT", value=True),
        'DeepSHAP': st.sidebar.checkbox("DeepLIFT-Shap", value=True),
        'GradSHAP': st.sidebar.checkbox("Gradient SHAP", value=True),
        'Ablation': st.sidebar.checkbox("Feature Ablation", value=True),
        'GradCAM': st.sidebar.checkbox("Grad-CAM", value=True)
    }

    if st.sidebar.button("Run Analysis", type="primary"):
        seq_str = sequences[selected_idx]
        seq_str_rc = get_reverse_complement(seq_str)
        wrapper = ModelWrapper(base_model, target_neuron)

        with st.spinner("Calculating XAI for Forward Sequence..."):
            st.session_state['plots_fwd'] = compute_xai_for_sequence(seq_str,
                                                                     wrapper,
                                                                     custom_bg_arr,
                                                                     is_heatmap,
                                                                     should_normalize,
                                                                     sel_methods
                                                                     )

        with st.spinner("Calculating XAI for Reverse Complement..."):
            st.session_state['plots_rc'] = compute_xai_for_sequence(seq_str_rc,
                                                                    wrapper,
                                                                    custom_bg_arr,
                                                                    is_heatmap,
                                                                    should_normalize,
                                                                    sel_methods
                                                                    )

        with st.spinner("Performing In Silico Mutagenesis (ISM)..."):
            st.session_state['ism_df'] = calculate_ism(seq_str, wrapper)
            st.session_state['ism_df_rc'] = calculate_ism(seq_str_rc, wrapper)

        st.session_state['seq_id'] = ids[selected_idx]
        st.session_state['run_view_mode'] = view_mode
        st.session_state['run_is_normalized'] = should_normalize

    if st.session_state['plots_fwd'] is not None:
        if (st.session_state['run_view_mode'] != view_mode or st.session_state['run_is_normalized'] != should_normalize):
            st.warning("⚠️ Display options have changed! Please click **'Run Analysis'** in the sidebar to rerun!")
        else:
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Single Sequence (Vertical)", 
                "🧬 FWD vs RC (Paired Comparison)", 
                "🔬 In Silico Mutagenesis (Landscape)",
                "🔬 In Silico Mutagenesis (RC Landscape)"
            ])

            with tab1:
                st.subheader(f"Analysis for: `{st.session_state['seq_id']}` (Forward Only)")
                fig1 = generate_vertical_figure(st.session_state['plots_fwd'], is_heatmap, should_normalize)
                st.pyplot(fig1)
                
                buf1 = io.BytesIO()
                fig1.savefig(buf1, format="png", bbox_inches='tight', dpi=150)
                buf1.seek(0)
                
                c1, c2 = st.columns(2)
                c1.download_button("📥 Download Plot (PNG)",
                                   data=buf1,
                                   file_name=f"XAI_{st.session_state['seq_id']}_FWD.png",
                                   mime="image/png",
                                   width="stretch"
                                   )
                tsv_data = generate_tsv(st.session_state['plots_fwd'], None, is_heatmap, st.session_state['seq_id'])
                c2.download_button("📥 Download Data (TSV)", data=tsv_data,
                                   file_name=f"XAI_{st.session_state['seq_id']}_FWD.tsv",
                                   mime="text/tab-separated-values",
                                   width="stretch"
                                   )

            with tab2:
                st.subheader(f"Symmetrical Analysis: FWD (Left) vs RC (Right)")
                fig2 = generate_symmetrical_figure(st.session_state['plots_fwd'], st.session_state['plots_rc'], is_heatmap, should_normalize)
                st.pyplot(fig2)
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format="png", bbox_inches='tight', dpi=300)
                buf2.seek(0)
                
                c3, c4 = st.columns(2)
                c3.download_button("📥 Download Paired Plot (PNG)",
                                   data=buf2,
                                   file_name=f"XAI_{st.session_state['seq_id']}_Paired.png",
                                   mime="image/png",
                                   width="stretch"
                                   )
                tsv_data_paired = generate_tsv(st.session_state['plots_fwd'],
                                               st.session_state['plots_rc'],
                                               is_heatmap,
                                               st.session_state['seq_id']
                                            )
                c4.download_button("📥 Download Paired Data (TSV)", data=tsv_data_paired, file_name=f"XAI_{st.session_state['seq_id']}_Paired.tsv", mime="text/tab-separated-values", width="stretch")

            with tab3:
                st.subheader("Evolvability Landscape (In Silico Mutagenesis)")
                st.markdown("Sorted prediction differences for all possible mutations.")
                ism_df = st.session_state['ism_df']
                fig_ism, ax_ism = plt.subplots(figsize=(16, 5))
                colors = ['crimson' if val > 0 else 'royalblue' for val in ism_df['Difference']]
                ax_ism.bar(range(len(ism_df)), ism_df['Difference'], color=colors, width=1.0)
                ax_ism.set_ylabel("Prediction Difference (Mutant - Original)",
                                  fontsize=14, fontweight='bold')
                ax_ism.set_xlabel("Mutations (Sorted from Worst to Best)",
                                  fontsize=14, fontweight='bold')
                ax_ism.axhline(0, color='black', linewidth=0.8)
                ax_ism.set_xticks([]) 
                plt.tight_layout()

                st.pyplot(fig_ism)
                st.divider()

                col_btn_ism1, col_btn_ism2 = st.columns(2)
                buf_ism = io.BytesIO()
                fig_ism.savefig(buf_ism, format="png", bbox_inches='tight', dpi=300)
                buf_ism.seek(0)
                
                with col_btn_ism1:
                    st.download_button(
                        label="📥 Download Landscape Plot (PNG)",
                        data=buf_ism,
                        file_name=f"ISM_Landscape_{st.session_state['seq_id']}.png",
                        mime="image/png",
                        width="stretch"
                    )
                
                csv_ism = ism_df.to_csv(index=False, sep='\t')

                with col_btn_ism2:
                    st.download_button(
                        label="📥 Download Mutations Data (TSV)", 
                        data=csv_ism, 
                        file_name=f"ISM_{st.session_state['seq_id']}.tsv", 
                        mime="text/tab-separated-values",
                        width="stretch"
                    )
                
                col_t1, col_t2 = st.columns(2)

                with col_t1:
                    st.markdown("🔴 **Top 10 Destructive Mutations** (Decreases Activity)")
                    st.dataframe(ism_df.head(10)[['Position', 'Original', 'Mutated', 'Difference']],
                                 width="stretch")
                with col_t2:
                    st.markdown("🟢 **Top 10 Enhancing Mutations** (Increases Activity)")
                    st.dataframe(ism_df.tail(10).iloc[::-1][['Position', 'Original', 'Mutated', 'Difference']],
                                 width="stretch")
            
            with tab4:
                if st.session_state['ism_df_rc'] is not None:
                    st.subheader("Evolvability Landscape (Reverse Complement)")
                    st.markdown("Sorted prediction differences for all possible mutations.")

                    ism_df_rc = st.session_state['ism_df_rc']
                    fig_ism_rc, ax_ism_rc = plt.subplots(figsize=(16, 5))
                    colors_rc = ['crimson' if val > 0 else 'royalblue' for val in ism_df_rc['Difference']]
                    ax_ism_rc.bar(range(len(ism_df_rc)), ism_df_rc['Difference'], color=colors_rc, width=1.0)
                    ax_ism_rc.set_ylabel("Prediction Difference (Mutant - Original)",
                                         fontsize=14, fontweight='bold')
                    ax_ism_rc.set_xlabel("Mutations (Sorted from Worst to Best)",
                                         fontsize=14, fontweight='bold')
                    ax_ism_rc.axhline(0, color='black', linewidth=0.8)
                    ax_ism_rc.set_xticks([]) 
                    plt.tight_layout()

                    st.pyplot(fig_ism_rc)
                    st.divider()

                    col_btn_ism_rc1, col_btn_ism_rc2 = st.columns(2)
                    buf_ism_rc = io.BytesIO()
                    fig_ism_rc.savefig(buf_ism_rc, format="png", bbox_inches='tight', dpi=300)
                    buf_ism_rc.seek(0)
                    
                    with col_btn_ism_rc1:
                        st.download_button(
                            label="📥 Download RC Landscape Plot (PNG)",
                            data=buf_ism_rc,
                            file_name=f"ISM_Landscape_{st.session_state['seq_id']}_RC.png",
                            mime="image/png",
                            width="stretch"
                        )
                    
                    csv_ism_rc = ism_df_rc.to_csv(index=False, sep='\t')
                    with col_btn_ism_rc2:
                        st.download_button(
                            label="📥 Download Mutations Data (TSV)", 
                            data=csv_ism_rc, 
                            file_name=f"ISM_{st.session_state['seq_id']}_RC.tsv", 
                            mime="text/tab-separated-values",
                            width="stretch"
                        )
                    
                    col_t1_rc, col_t2_rc = st.columns(2)
                    with col_t1_rc:
                        st.markdown("🔴 **Top 10 Destructive Mutations (RC)**")
                        st.dataframe(ism_df_rc.head(10)[['Position', 'Original', 'Mutated', 'Difference']],
                                     width="stretch")
                    with col_t2_rc:
                        st.markdown("🟢 **Top 10 Enhancing Mutations (RC)**")
                        st.dataframe(ism_df_rc.tail(10).iloc[::-1][['Position', 'Original', 'Mutated', 'Difference']],
                                     width="stretch")

else:
    st.info("👈 Please upload the model weights (.pth) and a FASTA file in the sidebar to begin.")
