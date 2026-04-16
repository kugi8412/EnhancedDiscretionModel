#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import io
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import streamlit as st
from Bio import SeqIO

# --- HACK DLA IMPORTÓW (pozwala uruchamiać z folderu apps/) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Importy z Twojego projektu
from models.registry import build_model
from utils import get_reverse_complement, one_hot_encode_dna


# --- KONFIGURACJA STREAMLIT ---
st.set_page_config(page_title="DNA Model Results & Evaluation", layout="wide")
st.title("📊 Analiza Wyników Modelu DNA")


# --- FUNKCJE POMOCNICZE ---
@st.cache_resource
def load_dynamic_model(config_dict, weights_bytes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config_dict).to(device)
    model.load_state_dict(torch.load(io.BytesIO(weights_bytes), map_location=device, weights_only=True))
    model.eval()
    return model, device

@st.cache_data
def get_predictions(_model, sequences, _device, batch_size=512):
    all_dev, all_hk = [], []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        encoded = one_hot_encode_dna(batch_seqs)
        tensor = torch.tensor(encoded, dtype=torch.float32).to(_device)
        
        with torch.no_grad():
            out = _model(tensor)
            if isinstance(out, (list, tuple)):
                out_dev, out_hk = out[0], out[1]
            else:
                out_dev, out_hk = out[:, 0], out[:, 1]
                
        all_dev.extend(out_dev.cpu().numpy().flatten())
        all_hk.extend(out_hk.cpu().numpy().flatten())
    return np.array(all_dev), np.array(all_hk)

def calculate_gc_content(sequences):
    gc_contents = []
    for seq in sequences:
        gc = (seq.count('G') + seq.count('C')) / len(seq) * 100
        gc_contents.append(gc)
    return np.array(gc_contents)

def adjust_axes(ax):
    """Usuwa górną i prawą ramkę dla czytelności (jak w evaluate.py)."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf


# --- SIDEBAR: INTERFEJS UŻYTKOWNIKA ---
st.sidebar.header("📂 1. Pliki Modelu")
config_file = st.sidebar.file_uploader("Wgraj Config YAML", type=["yaml", "yml"])
weights_file = st.sidebar.file_uploader("Wgraj Wagi (.pth)", type=["pth"])

st.sidebar.header("📋 2. Dane Testowe")
fasta_file = st.sidebar.file_uploader("Wgraj Sekwencje (.fasta)", type=["fa", "fasta"])
targets_file = st.sidebar.file_uploader("Wgraj Wyniki (np. Sequences_activity.txt)", type=["txt", "csv", "tsv"])

st.sidebar.header("⚙️ 3. Opcje Wizualizacji (Zakładka 2)")
plot_type_tab2 = st.sidebar.radio("Sposób cieniowania:", ["Scatter (Punkty)", "Hexbin (Siatka uśredniona)"])
color_option = st.sidebar.selectbox("Zmienna nakładana na wykres:", ["Brak (Oryginalny wykres gęstości z evaluate.py)", "Zawartość GC (%)", "Błąd MSE (względem referencji)"])


# --- GŁÓWNA LOGIKA APLIKACJI ---
if config_file and weights_file and fasta_file and targets_file:
    
    try:
        config = yaml.safe_load(config_file)
        model, device = load_dynamic_model(config, weights_file.getvalue())
        model_name = config.get('model', {}).get('name', 'DNA_Model')
        st.sidebar.success(f"Model {model_name} załadowany pomyślnie!")
    except Exception as e:
        st.error(f"Błąd ładowania modelu. Szczegóły: {e}")
        st.stop()

    fasta_content = fasta_file.getvalue().decode("utf-8")
    records = list(SeqIO.parse(io.StringIO(fasta_content), "fasta"))
    sequences = [str(r.seq).upper() for r in records]
    
    try:
        df_targets = pd.read_csv(io.StringIO(targets_file.getvalue().decode("utf-8")), sep='\t')
        true_dev = df_targets['Dev_log2_enrichment'].values
        true_hk = df_targets['Hk_log2_enrichment'].values
    except Exception as e:
        st.error(f"Błąd pliku referencyjnego. Szczegóły: {e}")
        st.stop()

    if len(sequences) != len(df_targets):
        st.warning(f"Zrównuję długości plików. FASTA: {len(sequences)}, Wyniki: {len(df_targets)}.")
        min_len = min(len(sequences), len(df_targets))
        sequences, true_dev, true_hk = sequences[:min_len], true_dev[:min_len], true_hk[:min_len]

    with st.spinner("Obliczanie predykcji dla nici Forward..."):
        pred_dev_fwd, pred_hk_fwd = get_predictions(model, sequences, device)
        
    with st.spinner("Obliczanie predykcji dla nici Reverse Complement..."):
        sequences_rc = [get_reverse_complement(seq) for seq in sequences]
        pred_dev_rc, pred_hk_rc = get_predictions(model, sequences_rc, device)

    gc_content = calculate_gc_content(sequences)
    mse_dev = (pred_dev_fwd - true_dev) ** 2
    mse_hk = (pred_hk_fwd - true_hk) ** 2


    # --- RYSOWANIE WYKRESÓW (ZAKŁADKI) ---
    tab1, tab2 = st.tabs(["Obserwacje vs Predykcje (DeepSTARR Style)", "Forward vs Reverse Complement"])

    # ====== ZAKŁADKA 1: PREDYKCJE VS LAB (ORYGINAŁ Z EVALUATE.PY) ======
    with tab1:
        st.subheader("Porównanie wartości obserwowanych z przewidywaniami modelu")
        
        fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
        
        # IDEALNA KOPIA evaluate.py (bez dodawania gridsize i cmap)
        hb_dev = axes1[0].hexbin(true_dev, pred_dev_fwd, bins='log')
        hb_hk = axes1[1].hexbin(true_hk, pred_hk_fwd, bins='log')

        adjust_axes(axes1[0])
        adjust_axes(axes1[1])
        
        fig1.supxlabel('Observed fold change [log2]', fontsize=10)
        axes1[0].set_ylabel('Predicted fold change [log2]', fontsize=10)

        pcc_dev = pearsonr(true_dev, pred_dev_fwd)[0]
        pcc_hk = pearsonr(true_hk, pred_hk_fwd)[0]

        fig1.suptitle(f'{model_name} predictions vs observed', fontsize=14)
        plt.subplots_adjust(wspace=0.4) 

        axes1[0].set_title(f'Developmental (PCC = {pcc_dev:.3f})', fontsize=10)
        axes1[1].set_title(f'Housekeeping (PCC = {pcc_hk:.3f})', fontsize=10)
        
        fig1.colorbar(hb_dev, ax=axes1[0], label='Log10(Count)')
        fig1.colorbar(hb_hk, ax=axes1[1], label='Log10(Count)')

        st.pyplot(fig1)
        
        st.download_button(
            label="💾 Pobierz Wykres (PNG)",
            data=fig_to_bytes(fig1),
            file_name=f"{model_name}_predictions_vs_observed.png",
            mime="image/png"
        )

    # ====== ZAKŁADKA 2: STRAND BIAS (FWD VS RC) ======
    with tab2:
        st.subheader("Wpływ odwrócenia nici (Reverse Complement) na predykcje")
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        pcc_dev_rc = pearsonr(pred_dev_fwd, pred_dev_rc)[0]
        pcc_hk_rc = pearsonr(pred_hk_fwd, pred_hk_rc)[0]

        # WARIANT 1: BRAK CIENIOWANIA -> Wymuszamy oryginalny wykres z evaluate.py
        if color_option.startswith("Brak"):
            hb_dev2 = axes2[0].hexbin(pred_dev_fwd, pred_dev_rc, bins='log')
            hb_hk2 = axes2[1].hexbin(pred_hk_fwd, pred_hk_rc, bins='log')
            
            fig2.colorbar(hb_dev2, ax=axes2[0], label='Log10(Count)')
            fig2.colorbar(hb_hk2, ax=axes2[1], label='Log10(Count)')

        # WARIANT 2: ZASTOSOWANO CIENIOWANIE ZMIENNĄ (GC LUB MSE)
        else:
            if "GC" in color_option:
                c_dev, c_hk, cmap, cbar_label = gc_content, gc_content, 'viridis', "Średnie GC (%)"
            else:
                c_dev, c_hk, cmap, cbar_label = mse_dev, mse_hk, 'inferno_r', "Błąd MSE do Ref."

            if "Hexbin" in plot_type_tab2:
                # Cieniowany Hexbin
                hb_dev2 = axes2[0].hexbin(pred_dev_fwd, pred_dev_rc, C=c_dev, reduce_C_function=np.mean, gridsize=50, cmap=cmap)
                hb_hk2 = axes2[1].hexbin(pred_hk_fwd, pred_hk_rc, C=c_hk, reduce_C_function=np.mean, gridsize=50, cmap=cmap)
                fig2.colorbar(hb_dev2, ax=axes2[0], label=cbar_label)
                fig2.colorbar(hb_hk2, ax=axes2[1], label=cbar_label)
            else:
                # Cieniowany Scatter
                sc_dev = axes2[0].scatter(pred_dev_fwd, pred_dev_rc, c=c_dev, cmap=cmap, alpha=0.6, s=10)
                sc_hk = axes2[1].scatter(pred_hk_fwd, pred_hk_rc, c=c_hk, cmap=cmap, alpha=0.6, s=10)
                fig2.colorbar(sc_dev, ax=axes2[0], label=cbar_label.replace("Średnie ", ""))
                fig2.colorbar(sc_hk, ax=axes2[1], label=cbar_label.replace("Średni ", ""))

        # Linia referencyjna y=x na obu wykresach
        axes2[0].plot([pred_dev_fwd.min(), pred_dev_fwd.max()], [pred_dev_fwd.min(), pred_dev_fwd.max()], 'k--', lw=1)
        axes2[1].plot([pred_hk_fwd.min(), pred_hk_fwd.max()], [pred_hk_fwd.min(), pred_hk_fwd.max()], 'k--', lw=1)

        adjust_axes(axes2[0])
        adjust_axes(axes2[1])
        
        fig2.supxlabel('Predictions (Forward Strand) [log2]', fontsize=10)
        axes2[0].set_ylabel('Predictions (RC Strand) [log2]', fontsize=10)
        
        fig2.suptitle(f'{model_name} Forward vs RC Strand Predictions', fontsize=14)
        plt.subplots_adjust(wspace=0.4) 

        axes2[0].set_title(f'Developmental (RC PCC = {pcc_dev_rc:.3f})', fontsize=10)
        axes2[1].set_title(f'Housekeeping (RC PCC = {pcc_hk_rc:.3f})', fontsize=10)

        st.pyplot(fig2)
        
        st.download_button(
            label="💾 Pobierz Wykres (PNG)",
            data=fig_to_bytes(fig2),
            file_name=f"{model_name}_fwd_vs_rc.png",
            mime="image/png"
        )

else:
    st.info("👈 Wgraj config (YAML), wagi (.pth), plik FASTA i plik z wynikami referencyjnymi (txt/tsv) w panelu bocznym, aby rozpocząć analizę.")
