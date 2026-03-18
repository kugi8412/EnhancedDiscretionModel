#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import streamlit as st
from Bio import SeqIO

# Importy z Twojego projektu
from models.registry import build_model
from utils import get_reverse_complement, one_hot_encode_dna

# --- KONFIGURACJA STREAMLIT ---
st.set_page_config(page_title="DNA Model Results & Evaluation", layout="wide")
st.title("📊 Analiza Wyników Modelu DNA")

# --- FUNKCJE POMOCNICZE ---
@st.cache_resource
def load_dynamic_model(config_dict, weights_bytes):
    """Ładuje model na podstawie pliku YAML i wag."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config_dict).to(device)
    # Używamy weights_only=True zgodnie z dobrymi praktykami bezpieczeństwa PyTorch
    model.load_state_dict(torch.load(io.BytesIO(weights_bytes), map_location=device, weights_only=True))
    model.eval()
    return model, device

@st.cache_data
def get_predictions(_model, sequences, _device, batch_size=512):
    """Generuje predykcje wsadowo (batching), aby uniknąć braku pamięci (OOM)."""
    all_dev, all_hk = [], []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        encoded = one_hot_encode_dna(batch_seqs)
        tensor = torch.tensor(encoded, dtype=torch.float32).to(_device)
        
        with torch.no_grad():
            out = _model(tensor)
            # Obsługa modeli zwracających krotkę (dev, hk) lub tensor [B, 2]
            if isinstance(out, (list, tuple)):
                out_dev, out_hk = out[0], out[1]
            else:
                out_dev, out_hk = out[:, 0], out[:, 1]
                
        all_dev.extend(out_dev.cpu().numpy().flatten())
        all_hk.extend(out_hk.cpu().numpy().flatten())
    return np.array(all_dev), np.array(all_hk)

def calculate_gc_content(sequences):
    """Oblicza zawartość GC w procentach dla listy sekwencji."""
    gc_contents = []
    for seq in sequences:
        gc = (seq.count('G') + seq.count('C')) / len(seq) * 100
        gc_contents.append(gc)
    return np.array(gc_contents)

def adjust_axes(ax):
    """Usuwa górną i prawą ramkę wykresu dla czytelności."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- SIDEBAR: WGRYWANIE PLIKÓW ---
st.sidebar.header("📂 1. Pliki Modelu")
config_file = st.sidebar.file_uploader("Wgraj Config YAML", type=["yaml", "yml"])
weights_file = st.sidebar.file_uploader("Wgraj Wagi (.pth)", type=["pth"])

st.sidebar.header("📋 2. Dane Testowe")
fasta_file = st.sidebar.file_uploader("Wgraj Sekwencje (.fasta)", type=["fa", "fasta"])
targets_file = st.sidebar.file_uploader("Wgraj Wyniki (np. Sequences_activity.txt)", type=["txt", "csv", "tsv"])

st.sidebar.header("⚙️ 3. Opcje Wizualizacji (Tab 2)")
color_option = st.sidebar.selectbox("Cieniowanie wykresu (Fwd vs RC):", ["Brak", "Zawartość GC (%)", "Błąd MSE (względem referencji)"])

# --- LOGIKA GŁÓWNA ---
if config_file and weights_file and fasta_file and targets_file:
    # 1. Ładowanie modelu
    try:
        config = yaml.safe_load(config_file)
        model, device = load_dynamic_model(config, weights_file.getvalue())
        st.sidebar.success(f"Model {config['model'].get('name', 'N/A')} załadowany!")
    except Exception as e:
        st.error(f"Błąd ładowania modelu: {e}")
        st.stop()

    # 2. Parsowanie danych wejściowych
    fasta_content = fasta_file.getvalue().decode("utf-8")
    records = list(SeqIO.parse(io.StringIO(fasta_content), "fasta"))
    sequences = [str(r.seq).upper() for r in records]
    
    try:
        df_targets = pd.read_csv(io.StringIO(targets_file.getvalue().decode("utf-8")), sep='\t')
        true_dev = df_targets['Dev_log2_enrichment'].values
        true_hk = df_targets['Hk_log2_enrichment'].values
    except Exception as e:
        st.error(f"Błąd przetwarzania pliku z wynikami. Upewnij się, że posiada kolumny 'Dev_log2_enrichment' oraz 'Hk_log2_enrichment'. Szczegóły: {e}")
        st.stop()

    if len(sequences) != len(df_targets):
        st.warning(f"Uwaga: Liczba sekwencji w FASTA ({len(sequences)}) różni się od liczby wierszy w pliku z wynikami ({len(df_targets)}). Ocinam do najkrótszej wartości.")
        min_len = min(len(sequences), len(df_targets))
        sequences = sequences[:min_len]
        true_dev = true_dev[:min_len]
        true_hk = true_hk[:min_len]

    # --- OBLICZENIA PREDYKCJI ---
    with st.spinner("Obliczanie predykcji dla nici Forward..."):
        pred_dev_fwd, pred_hk_fwd = get_predictions(model, sequences, device)
        
    with st.spinner("Obliczanie predykcji dla nici Reverse Complement..."):
        sequences_rc = [get_reverse_complement(seq) for seq in sequences]
        pred_dev_rc, pred_hk_rc = get_predictions(model, sequences_rc, device)

    # Przygotowanie danych do cieniowania (Coloring)
    gc_content = calculate_gc_content(sequences)
    mse_dev = (pred_dev_fwd - true_dev) ** 2
    mse_hk = (pred_hk_fwd - true_hk) ** 2

    if color_option == "Zawartość GC (%)":
        c_dev, c_hk = gc_content, gc_content
        cmap = 'viridis'
        cbar_label = "Zawartość GC (%)"
    elif color_option == "Błąd MSE (względem referencji)":
        c_dev, c_hk = mse_dev, mse_hk
        cmap = 'inferno_r' # Im mniejszy błąd, tym jaśniej
        cbar_label = "MSE (Fwd vs Referencja)"
    else:
        c_dev, c_hk = None, None
        cmap = None
        cbar_label = ""

    # --- WIZUALIZACJA ---
    tab1, tab2 = st.tabs(["Obserwacje vs Predykcje (Hexbin)", "Porównanie Nici: Forward vs Reverse Complement"])

    with tab1:
        st.subheader("Porównanie wartości obserwowanych z przewidywaniami modelu")
        st.write("Wykres z użyciem tzw. logarytmicznego Hexbin, przydatny przy gęstych zbiorach danych (np. DeepSTARR Test Set).")
        
        fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
        
        # Dev Plot
        hb_dev = axes1[0].hexbin(true_dev, pred_dev_fwd, gridsize=50, cmap='Blues', bins='log')
        adjust_axes(axes1[0])
        axes1[0].set_xlabel('Observed fold change [log2]')
        axes1[0].set_ylabel('Predicted fold change [log2]')
        pcc_dev, _ = pearsonr(true_dev, pred_dev_fwd)
        axes1[0].set_title(f'Developmental (PCC = {pcc_dev:.3f})')
        fig1.colorbar(hb_dev, ax=axes1[0], label='Log10(Count)')

        # Hk Plot
        hb_hk = axes1[1].hexbin(true_hk, pred_hk_fwd, gridsize=50, cmap='Oranges', bins='log')
        adjust_axes(axes1[1])
        axes1[1].set_xlabel('Observed fold change [log2]')
        axes1[1].set_ylabel('Predicted fold change [log2]')
        pcc_hk, _ = pearsonr(true_hk, pred_hk_fwd)
        axes1[1].set_title(f'Housekeeping (PCC = {pcc_hk:.3f})')
        fig1.colorbar(hb_hk, ax=axes1[1], label='Log10(Count)')

        plt.tight_layout()
        st.pyplot(fig1)

    with tab2:
        st.subheader("Wpływ odwrócenia nici (Reverse Complement) na predykcje")
        st.write(f"Kolorowanie punktów na podstawie: **{color_option}**.")

        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

        # Dev Fwd vs RC
        pcc_dev_rc, _ = pearsonr(pred_dev_fwd, pred_dev_rc)
        if c_dev is not None:
            sc_dev = axes2[0].scatter(pred_dev_fwd, pred_dev_rc, c=c_dev, cmap=cmap, alpha=0.6, s=10)
            fig2.colorbar(sc_dev, ax=axes2[0], label=cbar_label)
        else:
            axes2[0].scatter(pred_dev_fwd, pred_dev_rc, alpha=0.3, s=10, color='dodgerblue')
            
        axes2[0].plot([pred_dev_fwd.min(), pred_dev_fwd.max()], [pred_dev_fwd.min(), pred_dev_fwd.max()], 'k--', lw=1)
        adjust_axes(axes2[0])
        axes2[0].set_xlabel('Predictions (Forward Strand)')
        axes2[0].set_ylabel('Predictions (Reverse Complement)')
        axes2[0].set_title(f'Developmental (Fwd vs RC PCC = {pcc_dev_rc:.3f})')

        # Hk Fwd vs RC
        pcc_hk_rc, _ = pearsonr(pred_hk_fwd, pred_hk_rc)
        if c_hk is not None:
            sc_hk = axes2[1].scatter(pred_hk_fwd, pred_hk_rc, c=c_hk, cmap=cmap, alpha=0.6, s=10)
            fig2.colorbar(sc_hk, ax=axes2[1], label=cbar_label)
        else:
            axes2[1].scatter(pred_hk_fwd, pred_hk_rc, alpha=0.3, s=10, color='coral')
            
        axes2[1].plot([pred_hk_fwd.min(), pred_hk_fwd.max()], [pred_hk_fwd.min(), pred_hk_fwd.max()], 'k--', lw=1)
        adjust_axes(axes2[1])
        axes2[1].set_xlabel('Predictions (Forward Strand)')
        axes2[1].set_ylabel('Predictions (Reverse Complement)')
        axes2[1].set_title(f'Housekeeping (Fwd vs RC PCC = {pcc_hk_rc:.3f})')

        plt.tight_layout()
        st.pyplot(fig2)

else:
    st.info("👈 Wgraj model, wagi, plik FASTA oraz plik z referencyjnymi wynikami (txt/tsv) w panelu bocznym, aby wygenerować wykresy.")
