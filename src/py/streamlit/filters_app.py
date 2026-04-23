#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streamlit application for CNN filter analysis and JASPAR motif matching."""

import os
import sys
import io
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# Ensure parent directory is on sys.path for project imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.registry import build_model


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="CNN Filters & Motif Analysis", layout="wide")
st.title("CNN Filter Analysis & JASPAR Motif Matching")

# --- Helper Functions ---

@st.cache_resource
def load_dynamic_model(config_dict, weights_bytes):
    """Load model from YAML config and weight buffer."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config_dict).to(device)
    model.load_state_dict(torch.load(io.BytesIO(weights_bytes), map_location=device, weights_only=False))
    model.eval()
    return model

@st.cache_data
def fetch_jaspar_motifs():
    """Fetch and parse the JASPAR CORE Insects database (includes D. melanogaster)."""
    import requests
    url = "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_insects_non-redundant_pfms_jaspar.txt"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text
    except Exception as e:
        st.error(f"Failed to download JASPAR database: {e}")
        return {}

    motifs = {}
    current_name = ""
    current_matrix = []
    
    for line in text.strip().split('\n'):
        if line.startswith(">"):
            # Linia np: >MA0023.1 twi
            parts = line.strip().split()
            current_name = parts[1] if len(parts) > 1 else parts[0][1:]
            current_matrix = []
        elif line.startswith(("A", "C", "G", "T")):
            # Linia np: A  [ 2.00  0.00  13.00 ]
            vals = line.split('[')[1].split(']')[0].split()
            current_matrix.append([float(v) for v in vals])
            
            if line.startswith("T"):
                # Convert PFM (count matrix) to PPM (probability matrix)
                pfm = np.array(current_matrix)
                ppm = pfm / np.maximum(np.sum(pfm, axis=0), 1e-6)
                motifs[current_name] = ppm
                
    return motifs

def filter_to_ppm(weight_matrix):
    """Convert convolutional filter weights (4, L) to probabilities via softmax."""
    # Subtract max for numerical stability
    w = weight_matrix - np.max(weight_matrix, axis=0, keepdims=True)
    exp_w = np.exp(w)
    return exp_w / np.sum(exp_w, axis=0)

def match_motif(filter_ppm, jaspar_ppm):
    """Compute sliding-window Pearson correlation between filter and JASPAR motif."""
    l1 = filter_ppm.shape[1]
    l2 = jaspar_ppm.shape[1]
    
    # Pad both matrices with background probability (0.25) for sliding
    padded_len = l1 + l2
    p1 = np.full((4, padded_len), 0.25)
    p2 = np.full((4, padded_len), 0.25)
    
    # Insert filter in the centre
    p1[:, l2//2 : l2//2 + l1] = filter_ppm
    
    best_corr = -1.0
    # Slide JASPAR motif along the padded array
    for offset in range(padded_len - l2 + 1):
        p2_window = np.full((4, padded_len), 0.25)
        p2_window[:, offset : offset + l2] = jaspar_ppm
        
        corr, _ = pearsonr(p1.flatten(), p2_window.flatten())
        if corr > best_corr:
            best_corr = corr
            
    return best_corr

# --- Sidebar: Configuration ---
st.sidebar.header("1. Load Model")
config_file = st.sidebar.file_uploader("Upload Config YAML", type=["yaml", "yml"])
weights_file = st.sidebar.file_uploader("Upload Weights (.pth)", type=["pth"])

st.sidebar.header("2. Analysis Settings")
frobenius_threshold = st.sidebar.number_input(
    "Frobenius Norm Threshold (filter activity cutoff)",
    min_value=0.0, max_value=1.0, value=0.1, step=0.05)
top_n_matches = st.sidebar.slider("Number of JASPAR matches to show", 1, 5, 3)

# --- MAIN LOGIC ---
if config_file and weights_file:
    # 1. Model Initialisation
    try:
        config = yaml.safe_load(config_file)
        model = load_dynamic_model(config, weights_file.getvalue())
        st.sidebar.success("Model loaded.")
    except Exception as e:
        st.error(f"Initialisation error: {e}")
        st.stop()

    # 2. Find first convolutional layer
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv1d) or type(module).__name__ == "RCConv1d":
            first_conv = module
            break

    if first_conv is None:
        st.error("No convolutional layer (nn.Conv1d) found in this model.")
        st.stop()

    # Wagi: [out_channels, in_channels (4), kernel_size]
    weights = first_conv.weight.data.cpu().numpy()
    num_filters = weights.shape[0]
    kernel_size = weights.shape[2]
    
    # 3. Compute Frobenius norms and filter active filters
    norms = np.linalg.norm(weights, ord='fro', axis=(1, 2))
    active_indices = np.where(norms >= frobenius_threshold)[0]
    active_weights = weights[active_indices]
    num_active = len(active_indices)
    
    st.markdown(
        f"**First Layer Summary:** Found **{num_filters}** filters of size {kernel_size} bp. "
        f"After applying Frobenius norm threshold (>= {frobenius_threshold}): "
        f"**{num_active}** active filters.")

    if num_active < 2:
        st.warning("Too few active filters for PCA. Lower the Frobenius norm threshold.")
        st.stop()

    # --- Tabs ---
    tab1, tab2 = st.tabs(["Filter PCA (2D)", "JASPAR Motif Matching"])

    with tab1:
        st.subheader("PCA of Active Convolutional Filters")
        
        # Flatten to vectors (N, 4 * kernel_size)
        flattened_weights = active_weights.reshape(num_active, -1)
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(flattened_weights)
        explained_variance = pca.explained_variance_ratio_ * 100
        
        # Rysowanie Wykresu PCA
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=norms[active_indices], cmap='viridis', s=60, alpha=0.8, edgecolors='k')
        
        # Filter index annotations
        for i, idx in enumerate(active_indices):
            ax.annotate(f"F{idx}", (pca_result[i, 0], pca_result[i, 1]),
                        fontsize=8, alpha=0.7, xytext=(3, 3), textcoords='offset points')
            
        ax.set_xlabel(f"PC1 ({explained_variance[0]:.2f}% variance)")
        ax.set_ylabel(f"PC2 ({explained_variance[1]:.2f}% variance)")
        ax.set_title("First Layer Filter Space (PCA)")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Frobenius Norm (filter activity)')
        
        st.pyplot(fig)

    with tab2:
        st.subheader("Filter Matching to JASPAR (Drosophila melanogaster)")
        
        if st.button("Run Motif Scanning (may take a moment)"):
            with st.spinner("Fetching JASPAR 2024 (Insects) and matching filters..."):
                jaspar_motifs = fetch_jaspar_motifs()
                
                if not jaspar_motifs:
                    st.error("Failed to load JASPAR motifs.")
                else:
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, idx in enumerate(active_indices):
                        # Convert filter to probability map (PPM)
                        filter_ppm = filter_to_ppm(weights[idx])
                        
                        best_matches = []
                        # Search entire JASPAR database
                        for motif_name, jaspar_ppm in jaspar_motifs.items():
                            corr = match_motif(filter_ppm, jaspar_ppm)
                            best_matches.append((motif_name, corr))
                            
                        # Sort by highest correlation
                        best_matches.sort(key=lambda x: x[1], reverse=True)
                        top_matches = best_matches[:top_n_matches]
                        
                        match_strings = [f"{name} (r={corr:.2f})" for name, corr in top_matches]
                        results.append({
                            "Filter ID": f"Filter_{idx}",
                            "Norm": round(norms[idx], 3),
                            "Top 1 Motif": match_strings[0],
                            "Top 2 Motif": match_strings[1] if top_n_matches > 1 else "-",
                            "Top 3 Motif": match_strings[2] if top_n_matches > 2 else "-",
                        })
                        
                        progress_bar.progress((i + 1) / num_active)
                        
                    results_df = pd.DataFrame(results).sort_values(by="Norm", ascending=False)
                    st.success(f"Analysed {num_active} filters against {len(jaspar_motifs)} JASPAR motifs.")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results as TSV
                    tsv = results_df.to_csv(index=False, sep='\t')
                    st.download_button(
                        label="Download Results (TSV)",
                        data=tsv,
                        file_name="filter_jaspar_matches.tsv",
                        mime="text/tab-separated-values"
                    )

else:
    st.info("Upload a Config (YAML) and Model Weights (.pth) in the sidebar to begin filter analysis.")
