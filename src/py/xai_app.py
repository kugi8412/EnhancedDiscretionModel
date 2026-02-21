#!/usr/bin/env python
# -*- coding: utf-8 -*-
# xai_app.py

"""
This application build with Streamlit is designed to show different explainable
machine learning methods to show how model (DeepSTARR) 'see' regulatory sequences.
"""

import io
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from Bio import SeqIO
from captum.attr import IntegratedGradients, LayerGradCam, GradientShap, Saliency


PARAMS = {
    'batch_size': 128,
    'epochs': 100,
    'early_stop': 12,
    'kernel_size1': 7,
    'kernel_size2': 3,
    'kernel_size3': 5,
    'kernel_size4': 3,
    'lr': 1e-4,
    'num_filters': 256,
    'num_filters2': 60,
    'num_filters3': 60,
    'num_filters4': 120,
    'n_conv_layer': 4,
    'n_add_layer': 2,
    'dropout_prob': 0.4,
    'dense_neurons1': 256,
    'dense_neurons2': 256,
    'pad': 'same',
    'vocab_size': 1024
}


class DeepSTARR(nn.Module):
    def __init__(self, params, permute_before_flatten=False):
        super(DeepSTARR, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=params['num_filters'],
                               kernel_size=params['kernel_size1'], padding=params['pad'])
        self.bn1 = nn.BatchNorm1d(params['num_filters'], eps=1e-3, momentum=0.01)
  
        self.conv2 = nn.Conv1d(in_channels=params['num_filters'], out_channels=params['num_filters2'],
                               kernel_size=params['kernel_size2'], padding=params['pad'])
        self.bn2 = nn.BatchNorm1d(params['num_filters2'], eps=1e-3, momentum=0.01)
        
        self.conv3 = nn.Conv1d(in_channels=params['num_filters2'], out_channels=params['num_filters3'],
                               kernel_size=params['kernel_size3'], padding=params['pad'])
        self.bn3 = nn.BatchNorm1d(params['num_filters3'], eps=1e-3, momentum=0.01)
        
        self.conv4 = nn.Conv1d(in_channels=params['num_filters3'], out_channels=params['num_filters4'],
                               kernel_size=params['kernel_size4'], padding=params['pad'])
        self.bn4 = nn.BatchNorm1d(params['num_filters4'], eps=1e-3, momentum=0.01)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.fc1 = nn.Linear(120 * (249 // (2**4)), params['dense_neurons1'])
        self.bn_fc1 = nn.BatchNorm1d(params['dense_neurons1'], eps=1e-3, momentum=0.01)
        
        self.fc2 = nn.Linear(params['dense_neurons1'], params['dense_neurons2'])
        self.bn_fc2 = nn.BatchNorm1d(params['dense_neurons2'], eps=1e-3, momentum=0.01)
        
        # Heads per task (developmental and housekeeping enhancer activities)
        self.fc_dev = nn.Linear(params['dense_neurons2'], 1)
        self.fc_hk = nn.Linear(params['dense_neurons2'], 1)
        
        self.dropout = nn.Dropout(params['dropout_prob'])
        self.permute_before_flatten = permute_before_flatten
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        if self.permute_before_flatten:
            x = x.permute(0, 2, 1)  # flatten the original code in Keras does
        x = x.reshape(x.shape[0], -1)

        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        
        out_dev = self.fc_dev(x)
        out_hk = self.fc_hk(x)
        
        return out_dev, out_hk


def one_hot_encode_dna(sequences):
    """One-hot encode a list of DNA sequences as a NumPy array of shape (B, 4, L)."""
    
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'T': [0, 0, 1, 0],
        'G': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }
    mapping.update({k.lower(): v for k, v in mapping.items()})  # extend to lowercase

    encoded = np.array([
        [mapping.get(base, [0, 0, 0, 0]) for base in seq]
        for seq in sequences
    ])  # shape: (B, L, 4)

    return encoded.transpose(0, 2, 1)  # (B, 4, L)


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
    model.load_state_dict(torch.load(
        weights_file, 
        map_location='cuda' if torch.cuda.is_available() else 'cpu',
        weights_only=True
    ))
    model.eval()
    return model


def parse_fasta(fasta_file):
    stringio = io.StringIO(fasta_file.getvalue().decode("utf-8"))
    sequences, ids = [], []
    for record in SeqIO.parse(stringio, "fasta"):
        sequences.append(str(record.seq).upper())
        ids.append(record.id)
    return ids, sequences


def draw_1d_axis(ax, data, title, color):
    ax.bar(range(len(data)), data, color=color, alpha=0.8, width=1.0)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Attribution")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlim(0, len(data))


def draw_heatmap_axis(fig, ax, data, title):
    vmax = np.max(np.abs(data))
    if vmax == 0: vmax = 1e-6 
    cax = ax.imshow(data, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['A', 'C', 'T', 'G'], fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.colorbar(cax, ax=ax, orientation='vertical', pad=0.01)


def generate_combined_figure(plot_items, is_heatmap):
    n_plots = len(plot_items)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 3.5 * n_plots), sharex=True)
    if n_plots == 1: axes = [axes]
    
    for ax, item in zip(axes, plot_items):
        if is_heatmap:
            draw_heatmap_axis(fig, ax, item['data'], item['title'])
        else:
            draw_1d_axis(ax, item['data'], item['title'], item['color'])
    
    axes[-1].set_xlabel("Sequence Position (bp)", fontsize=12)
    plt.tight_layout()
    return fig


st.set_page_config(layout="wide", page_title="DeepSTARR XAI Explorer")
st.title("🧬 DeepSTARR XAI Explorer")

if 'plot_items' not in st.session_state:
    st.session_state['plot_items'] = None
if 'view_mode' not in st.session_state:
    st.session_state['view_mode'] = None

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
    view_mode = st.sidebar.radio("Display Option:", [
        "1D Profile (Present Nucleotides Only)", 
        "Heatmap (Full 4x249 Matrix)"
    ])
    is_heatmap = "Heatmap" in view_mode

    if st.sidebar.button("Run Analysis", type="primary"):
        seq_str = sequences[selected_idx]
        seq_encoded = one_hot_encode_dna([seq_str]) 
        input_tensor = torch.tensor(seq_encoded, dtype=torch.float32)
        input_tensor.requires_grad = True

        wrapper = ModelWrapper(base_model, target_neuron)
        plot_items = []

        with st.spinner("Calculating Saliency..."):
            saliency = Saliency(wrapper)
            wrapper.zero_grad()
            out = wrapper(input_tensor)
            out.backward()
            raw_grads = input_tensor.grad.detach()
            
            if is_heatmap:
                plot_items.append({'data': raw_grads.squeeze().numpy(), 'title': 'Saliency (Vanilla Gradients)', 'color': None})
            else:
                sal_1d = torch.sum(raw_grads * input_tensor, dim=1).squeeze().detach().numpy()
                plot_items.append({'data': sal_1d, 'title': 'Saliency (Vanilla Gradients)', 'color': 'teal'})

        with st.spinner("Calculating Integrated Gradients..."):
            ig = IntegratedGradients(wrapper)
            attr_ig_0 = ig.attribute(input_tensor, baselines=0.0, n_steps=50)
            attr_ig_25 = ig.attribute(input_tensor, baselines=0.25, n_steps=50)
            
            if is_heatmap:
                plot_items.append({'data': attr_ig_0.squeeze().detach().numpy(), 'title': 'Integrated Gradients (Baseline 0.00)', 'color': None})
                plot_items.append({'data': attr_ig_25.squeeze().detach().numpy(), 'title': 'Integrated Gradients (Baseline 0.25)', 'color': None})
            else:
                ig_0_1d = torch.sum(attr_ig_0 * input_tensor, dim=1).squeeze().detach().numpy()
                ig_25_1d = torch.sum(attr_ig_25 * input_tensor, dim=1).squeeze().detach().numpy()
                plot_items.append({'data': ig_0_1d, 'title': 'Integrated Gradients (Baseline 0.00)', 'color': 'tomato'})
                plot_items.append({'data': ig_25_1d, 'title': 'Integrated Gradients (Baseline 0.25)', 'color': 'forestgreen'})

        with st.spinner("Calculating Gradient SHAP..."):
            gshap = GradientShap(wrapper)
            bg_0 = torch.zeros_like(input_tensor)
            attr_gshap_0 = gshap.attribute(input_tensor, baselines=bg_0)

            if attr_gshap_0.shape == (1, 249, 4):
                attr_gshap_0 = attr_gshap_0.transpose(1, 2)
            
            bg_25 = torch.full_like(input_tensor, 0.25)
            attr_gshap_25 = gshap.attribute(input_tensor, baselines=bg_25)

            if attr_gshap_25.shape == (1, 249, 4):
                attr_gshap_25 = attr_gshap_25.transpose(1, 2)
            
            if is_heatmap:
                plot_items.append({'data': attr_gshap_0.squeeze().detach().numpy(), 'title': 'Gradient SHAP (Baseline 0.00)', 'color': None})
                plot_items.append({'data': attr_gshap_25.squeeze().detach().numpy(), 'title': 'Gradient SHAP (Baseline 0.25)', 'color': None})
            else:
                shap_0_1d = torch.sum(attr_gshap_0 * input_tensor, dim=1).squeeze().detach().numpy()
                shap_25_1d = torch.sum(attr_gshap_25 * input_tensor, dim=1).squeeze().detach().numpy()
                plot_items.append({'data': shap_0_1d, 'title': 'Gradient SHAP (Baseline 0.00)', 'color': 'goldenrod'})
                plot_items.append({'data': shap_25_1d, 'title': 'Gradient SHAP (Baseline 0.25)', 'color': 'darkorange'})

        with st.spinner("Calculating Grad-CAM..."):
            conv_modules = [m for m in wrapper.model.modules() if isinstance(m, nn.Conv1d)]
            
            if len(conv_modules) > 0:
                conv_first = conv_modules[0]
                conv_last = conv_modules[-1]
                
                layer_gc_1 = LayerGradCam(wrapper, conv_first)
                layer_gc_last = LayerGradCam(wrapper, conv_last)
                
                attr_gc_1 = layer_gc_1.attribute(input_tensor)
                attr_gc_last = layer_gc_last.attribute(input_tensor)
                
                attr_gc_1_up = F.interpolate(attr_gc_1, size=249, mode='linear', align_corners=False).squeeze().detach().numpy()
                attr_gc_last_up = F.interpolate(attr_gc_last, size=249, mode='linear', align_corners=False).squeeze().detach().numpy()
                
                attr_gc_1_up = np.maximum(attr_gc_1_up, 0)
                attr_gc_last_up = np.maximum(attr_gc_last_up, 0)
                
                if is_heatmap:
                    plot_items.append({'data': np.tile(attr_gc_1_up, (4, 1)), 'title': 'Grad-CAM (First Conv Layer)', 'color': None})
                    plot_items.append({'data': np.tile(attr_gc_last_up, (4, 1)), 'title': 'Grad-CAM (Last Conv Layer)', 'color': None})
                else:
                    plot_items.append({'data': attr_gc_1_up, 'title': 'Grad-CAM (First Conv Layer)', 'color': 'mediumorchid'})
                    plot_items.append({'data': attr_gc_last_up, 'title': 'Grad-CAM (Last Conv Layer)', 'color': 'purple'})
            else:
                st.error("Could not find any Conv1d layers in the model.")

        st.session_state['plot_items'] = plot_items
        st.session_state['view_mode'] = view_mode
        st.session_state['seq_id'] = ids[selected_idx]

    if st.session_state['plot_items'] is not None:
        st.subheader(f"Results for: `{st.session_state['seq_id']}`")
        is_hm = "Heatmap" in st.session_state['view_mode']
        combined_fig = generate_combined_figure(st.session_state['plot_items'], is_hm)
        buf = io.BytesIO()
        combined_fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
        buf.seek(0)
        
        st.download_button(
            label="Download Combined Plot (PNG)",
            data=buf,
            file_name=f"XAI_analysis_{st.session_state['seq_id']}.png",
            mime="image/png",
            type="primary"
        )
        
        st.pyplot(combined_fig)

else:
    st.info("Please upload the DeepSTARR weights (.pth) and a FASTA file in the sidebar to begin.")
