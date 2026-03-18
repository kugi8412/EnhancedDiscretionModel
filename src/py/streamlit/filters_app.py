#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import yaml
import requests
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# Importy z Twojego projektu (registry.py musi być w dostępnej ścieżce)
from models.registry import build_model

# --- KONFIGURACJA STREAMLIT ---
st.set_page_config(page_title="CNN Filters & Motif Analysis", layout="wide")
st.title("🧬 Analiza Filtrów Konwolucyjnych i Motywów (JASPAR)")

# --- FUNKCJE POMOCNICZE ---

@st.cache_resource
def load_dynamic_model(config_dict, weights_bytes):
    """Ładuje model na podstawie YAML i wag."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config_dict).to(device)
    model.load_state_dict(torch.load(io.BytesIO(weights_bytes), map_location=device, weights_only=False))
    model.eval()
    return model

@st.cache_data
def fetch_jaspar_motifs():
    """Pobiera i parsuje bazę JASPAR CORE Insects (zawiera D. melanogaster)."""
    url = "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_insects_non-redundant_pfms_jaspar.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text
    except Exception as e:
        st.error(f"Błąd pobierania bazy JASPAR: {e}")
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
                # Konwersja z Count Matrix (PFM) na Probability Matrix (PPM)
                pfm = np.array(current_matrix)
                ppm = pfm / np.maximum(np.sum(pfm, axis=0), 1e-6)
                motifs[current_name] = ppm
                
    return motifs

def filter_to_ppm(weight_matrix):
    """Konwertuje wagi filtra konwolucyjnego (4, L) na prawdopodobieństwa (PPM) używając Softmaxa."""
    # Odejmujemy max dla stabilności numerycznej
    w = weight_matrix - np.max(weight_matrix, axis=0, keepdims=True)
    exp_w = np.exp(w)
    return exp_w / np.sum(exp_w, axis=0)

def match_motif(filter_ppm, jaspar_ppm):
    """Oblicza korelację Pearsona za pomocą ruchomego okna, dopasowując mniejszą macierz do większej."""
    l1 = filter_ppm.shape[1]
    l2 = jaspar_ppm.shape[1]
    
    # Padujemy obie macierze wartością tła (0.25 dla A,C,T,G), aby umożliwić przesuwanie
    padded_len = l1 + l2
    p1 = np.full((4, padded_len), 0.25)
    p2 = np.full((4, padded_len), 0.25)
    
    # Filtr wstawiamy na środek
    p1[:, l2//2 : l2//2 + l1] = filter_ppm
    
    best_corr = -1.0
    # Przesuwamy motyw JASPAR wzdłuż pada
    for offset in range(padded_len - l2 + 1):
        p2_window = np.full((4, padded_len), 0.25)
        p2_window[:, offset : offset + l2] = jaspar_ppm
        
        corr, _ = pearsonr(p1.flatten(), p2_window.flatten())
        if corr > best_corr:
            best_corr = corr
            
    return best_corr

# --- SIDEBAR: KONFIGURACJA ---
st.sidebar.header("📂 1. Wczytaj model")
config_file = st.sidebar.file_uploader("Wgraj Config YAML", type=["yaml", "yml"])
weights_file = st.sidebar.file_uploader("Wgraj Wagi (.pth)", type=["pth"])

st.sidebar.header("⚙️ 2. Ustawienia Analizy")
frobenius_threshold = st.sidebar.number_input("Próg Normy Frobeniusa (odcięcie martwych filtrów)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
top_n_matches = st.sidebar.slider("Liczba pokazywanych dopasowań JASPAR", 1, 5, 3)

# --- LOGIKA GŁÓWNA ---
if config_file and weights_file:
    # 1. Inicjalizacja Modelu
    try:
        config = yaml.safe_load(config_file)
        model = load_dynamic_model(config, weights_file.getvalue())
        st.sidebar.success(f"Model załadowany!")
    except Exception as e:
        st.error(f"Błąd inicjalizacji: {e}")
        st.stop()

    # 2. Szukanie pierwszej warstwy konwolucyjnej
    first_conv = None
    for module in model.modules():
        # Szukamy Conv1d lub Twojego własnego RCConv1d
        if isinstance(module, nn.Conv1d) or type(module).__name__ == "RCConv1d":
            first_conv = module
            break

    if first_conv is None:
        st.error("Nie znaleziono warstwy konwolucyjnej (nn.Conv1d) w modelu.")
        st.stop()

    # Wagi: [out_channels, in_channels (4), kernel_size]
    weights = first_conv.weight.data.cpu().numpy()
    num_filters = weights.shape[0]
    kernel_size = weights.shape[2]
    
    # 3. Obliczanie normy Frobeniusa i filtrowanie "żywych" filtrów
    norms = np.linalg.norm(weights, ord='fro', axis=(1, 2))
    active_indices = np.where(norms >= frobenius_threshold)[0]
    active_weights = weights[active_indices]
    num_active = len(active_indices)
    
    st.markdown(f"**Podsumowanie Pierwszej Warstwy:** Znaleziono **{num_filters}** filtrów o rozmiarze {kernel_size} bp. "
                f"Po zastosowaniu progu normy Frobeniusa (>= {frobenius_threshold}) pozostało **{num_active}** aktywnych filtrów.")

    if num_active < 2:
        st.warning("Zbyt mało aktywnych filtrów do wykonania PCA. Zmniejsz próg normy Frobeniusa.")
        st.stop()

    # --- ZAKŁADKI ---
    tab1, tab2 = st.tabs(["PCA Filtrów (2D)", "Dopasowania do Bazy JASPAR"])

    with tab1:
        st.subheader("PCA Aktywnych Filtrów Konwolucyjnych")
        
        # Spłaszczenie do wektorów (N, 4 * kernel_size)
        flattened_weights = active_weights.reshape(num_active, -1)
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(flattened_weights)
        explained_variance = pca.explained_variance_ratio_ * 100
        
        # Rysowanie Wykresu PCA
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=norms[active_indices], cmap='viridis', s=60, alpha=0.8, edgecolors='k')
        
        # Adnotacje z numerem filtra
        for i, idx in enumerate(active_indices):
            ax.annotate(f"F{idx}", (pca_result[i, 0], pca_result[i, 1]), fontsize=8, alpha=0.7, xytext=(3, 3), textcoords='offset points')
            
        ax.set_xlabel(f"Główna Składowa 1 (Tłumaczy {explained_variance[0]:.2f}% wariancji)")
        ax.set_ylabel(f"Główna Składowa 2 (Tłumaczy {explained_variance[1]:.2f}% wariancji)")
        ax.set_title("Przestrzeń Filtrów Pierwszej Warstwy (PCA)")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Norma Frobeniusa (Aktywność filtra)')
        
        st.pyplot(fig)

    with tab2:
        st.subheader("Dopasowywanie Filtrów do JASPAR (Drosophila Melanogaster)")
        
        if st.button("Uruchom Skanowanie Motywów (Może to zająć chwilę)"):
            with st.spinner("Pobieranie bazy JASPAR 2024 (Owady) i dopasowywanie filtrów..."):
                jaspar_motifs = fetch_jaspar_motifs()
                
                if not jaspar_motifs:
                    st.error("Nie udało się załadować motywów JASPAR.")
                else:
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, idx in enumerate(active_indices):
                        # Konwersja filtra na mapę prawdopodobieństw (PPM)
                        filter_ppm = filter_to_ppm(weights[idx])
                        
                        best_matches = []
                        # Szukanie w całej bazie JASPAR
                        for motif_name, jaspar_ppm in jaspar_motifs.items():
                            corr = match_motif(filter_ppm, jaspar_ppm)
                            best_matches.append((motif_name, corr))
                            
                        # Sortowanie po najwyższej korelacji
                        best_matches.sort(key=lambda x: x[1], reverse=True)
                        top_matches = best_matches[:top_n_matches]
                        
                        match_strings = [f"{name} (r={corr:.2f})" for name, corr in top_matches]
                        results.append({
                            "ID Filtra": f"Filter_{idx}",
                            "Norma": round(norms[idx], 3),
                            "Top 1 Motyw": match_strings[0],
                            "Top 2 Motyw": match_strings[1] if top_n_matches > 1 else "-",
                            "Top 3 Motyw": match_strings[2] if top_n_matches > 2 else "-"
                        })
                        
                        progress_bar.progress((i + 1) / num_active)
                        
                    results_df = pd.DataFrame(results).sort_values(by="Norma", ascending=False)
                    st.success(f"Przeanalizowano {num_active} filtrów z {len(jaspar_motifs)} motywami JASPAR.")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Pobieranie wyników jako TSV
                    tsv = results_df.to_csv(index=False, sep='\t')
                    st.download_button(
                        label="📥 Pobierz Wyniki (TSV)",
                        data=tsv,
                        file_name="filter_jaspar_matches.tsv",
                        mime="text/tab-separated-values"
                    )

else:
    st.info("👈 Wgraj plik Config (YAML) oraz Wagi Modelu (.pth) w panelu bocznym, aby rozpocząć analizę filtrów.")
