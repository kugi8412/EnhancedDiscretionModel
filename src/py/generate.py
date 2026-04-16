#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importy z Twojego projektu
from utils import load_config
from models.registry import build_model

def adjust_axes(ax):
    """Usuwa ramki wykresu dla czystego wyglądu."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def onehot_to_string(onehot_tensor):
    """Zamienia tensor [B, 4, L] na listę stringów DNA (A, C, G, T)."""
    idx_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    indices = torch.argmax(onehot_tensor, dim=1).cpu().numpy()
    sequences = []
    for seq_idx in indices:
        seq_str = "".join([idx_to_char[i] for i in seq_idx])
        sequences.append(seq_str)
    return sequences

def generate_random_dna(num_samples, seq_len=249, device='cuda'):
    """Generuje czysto losowe DNA (Baseline) z rozkładu jednostajnego."""
    indices = torch.randint(0, 4, (num_samples, seq_len)).to(device)
    onehot = F.one_hot(indices, num_classes=4).transpose(1, 2).float()
    return onehot

def generate_from_vqvae(model, num_samples, dev_target, hk_target, seq_len=249, device='cuda'):
    """Generuje sekwencje od zera używając słownika VQ i warunkowania FiLM."""
    model.eval()
    
    with torch.no_grad():
        # 1. Dynamiczne wyliczanie wymiaru ukrytego (Latent Shape)
        # Używamy dedykowanej metody encode_strand
        dummy_x = torch.zeros(1, 4, seq_len).to(device)
        z_e = model.encode_strand(dummy_x)
        latent_len = z_e.shape[2]
        
        # 2. Sampling "z wyobraźni": losujemy indeksy ze słownika Codebook
        num_embeddings = model.vq_layer.num_embeddings
        random_indices = torch.randint(0, num_embeddings, (num_samples, latent_len)).to(device)
        
        # 3. Wyciągamy wektory (pojęcia) ze słownika
        # model.vq_layer.embed ma kształt [num_embeddings, vq_dim]
        z_q = model.vq_layer.embed[random_indices].transpose(1, 2) # Kształt: [B, vq_dim, L_latent]
        
        # 4. Modulacja FiLM (Zmuszamy wyobraźnię do spełnienia warunku)
        y_dev = torch.full((num_samples,), dev_target, dtype=torch.float32).to(device)
        y_hk = torch.full((num_samples,), hk_target, dtype=torch.float32).to(device)
        c = torch.stack([y_dev, y_hk], dim=1)
        
        # Przejście przez warstwę FiLM Generatora
        film_params = model.film_generator(c)
        gamma = film_params[:, :z_q.size(1)].unsqueeze(2)
        beta = film_params[:, z_q.size(1):].unsqueeze(2)
        
        z_cond = (1.0 + gamma) * z_q + beta
        
        # 5. Dekodowanie z powrotem do DNA (odtworzenie logiki z forward)
        d = model.decoder_cond_proj(z_cond)
        d = d.permute(0, 2, 1)
        d, _ = model.decoder_gru(d)
        d = d.permute(0, 2, 1)
        
        d = model.decoder_blocks(d)
        x_logits_8ch = model.decoder_out(d)
        
        if x_logits_8ch.size(2) != seq_len:
            x_logits_8ch = F.interpolate(x_logits_8ch, size=seq_len, mode='linear', align_corners=False)
        
        # 6. Bierzemy tylko nić Forward (kanały 0-3) i robimy twardy One-Hot
        x_fwd_logits = x_logits_8ch[:, 0:4, :]
        seq_indices = torch.argmax(x_fwd_logits, dim=1)
        seq_onehot = F.one_hot(seq_indices, num_classes=4).transpose(1, 2).float()
        
    return seq_onehot


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Inicjalizacja generacji na urządzeniu: {device}")
    
    # --- ŁADOWANIE MODELI ---
    config = load_config(args.config)
    
    print("[INFO] Budowanie Generatora (cVQ-VAE)...")
    vqvae = build_model(config).to(device)
    vqvae.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    vqvae.eval()
    
    print("[INFO] Budowanie Wyroczni (LegNet)...")
    oracle_cfg = config.get('oracle', None)
    if not oracle_cfg or not oracle_cfg.get('apply', False):
        raise ValueError("Plik YAML musi zawierać poprawną sekcję 'oracle' do oceny wygenerowanych sekwencji!")
    
    oracle_config = load_config(oracle_cfg['config_path'])
    oracle = build_model(oracle_config).to(device)
    oracle.load_state_dict(torch.load(oracle_cfg['weights_path'], map_location=device, weights_only=True))
    oracle.eval()

    # --- DEFINIOWANIE GRUP GENERACJI ---
    groups = {
        "Random_Baseline": {"type": "random", "color": "grey", "marker": "x"},
        "High_Dev_High_HK": {"type": "vqvae", "dev": args.high_dev, "hk": args.high_hk, "color": "red", "marker": "o"},
        "Low_Dev_Low_HK":   {"type": "vqvae", "dev": args.low_dev, "hk": args.low_hk, "color": "blue", "marker": "o"},
        "High_Dev_Low_HK":  {"type": "vqvae", "dev": args.high_dev, "hk": args.low_hk, "color": "green", "marker": "o"},
        "Low_Dev_High_HK":  {"type": "vqvae", "dev": args.low_dev, "hk": args.high_hk, "color": "purple", "marker": "o"}
    }
    
    all_results = []
    fasta_records = []

    # --- GENERACJA I OCENA ---
    for group_name, params in groups.items():
        print(f"\n[INFO] Generowanie grupy: {group_name}...")
        
        if params["type"] == "random":
            seqs_onehot = generate_random_dna(args.n, device=device)
        else:
            seqs_onehot = generate_from_vqvae(vqvae, args.n, params["dev"], params["hk"], device=device)
            
        # Ocena przez LegNet
        with torch.no_grad():
            pred_dev, pred_hk = oracle(seqs_onehot)
            # Obsługa jeśli Oracle zwraca [B, 2] lub krotkę
            if isinstance(pred_dev, (list, tuple)):
                p_dev, p_hk = pred_dev[0], pred_dev[1]
            elif pred_dev.dim() > 1 and pred_dev.shape[1] == 2:
                p_hk = pred_dev[:, 1]
                p_dev = pred_dev[:, 0]
            else:
                p_dev, p_hk = pred_dev, pred_hk
                
        p_dev = p_dev.cpu().numpy().flatten()
        p_hk = p_hk.cpu().numpy().flatten()
        
        # Zapis do FASTA
        seqs_str = onehot_to_string(seqs_onehot)
        for i, seq in enumerate(seqs_str):
            fasta_records.append(f">{group_name}_{i+1} | Pred_Dev:{p_dev[i]:.2f} | Pred_HK:{p_hk[i]:.2f}\n{seq}")
        
        # Zapis do ramki danych (do wykresu)
        df_group = pd.DataFrame({
            "Group": group_name,
            "Pred_Dev": p_dev,
            "Pred_HK": p_hk,
            "Color": params["color"],
            "Marker": params["marker"]
        })
        all_results.append(df_group)

    df_all = pd.concat(all_results, ignore_index=True)

    # --- ZAPIS FASTA ---
    out_dir = "outputs/generated"
    os.makedirs(out_dir, exist_ok=True)
    fasta_path = os.path.join(out_dir, f"Generated_Sequences_n{args.n}.fasta")
    with open(fasta_path, "w") as f:
        f.write("\n".join(fasta_records))
    print(f"\n[SUCCESS] Zapisano {len(fasta_records)} sekwencji do {fasta_path}")

    # --- RYSOWANIE WYKRESU WYNIKÓW ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for group_name in df_all["Group"].unique():
        subset = df_all[df_all["Group"] == group_name]
        color = subset["Color"].iloc[0]
        marker = subset["Marker"].iloc[0]
        ax.scatter(subset["Pred_HK"], subset["Pred_Dev"], 
                   c=color, marker=marker, label=group_name, alpha=0.7, edgecolors='w', s=60)

    # Rysowanie linii celów (Targets)
    ax.axhline(args.high_dev, color='gray', linestyle='--', alpha=0.4, label='Target High Dev')
    ax.axhline(args.low_dev, color='gray', linestyle=':', alpha=0.4, label='Target Low Dev')
    ax.axvline(args.high_hk, color='gray', linestyle='--', alpha=0.4, label='Target High HK')
    ax.axvline(args.low_hk, color='gray', linestyle=':', alpha=0.4, label='Target Low HK')

    adjust_axes(ax)
    ax.set_xlabel('LegNet Predicted HK Fold Change [log2]', fontsize=12)
    ax.set_ylabel('LegNet Predicted Dev Fold Change [log2]', fontsize=12)
    ax.set_title(f'Inverse Design: Synthetic Enhancers Generated by cVQ-VAE', fontsize=14, pad=15)
    
    # Legenda poza wykresem
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"Generation_Scatter_n{args.n}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Wykres predykcji zapisano w {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic DNA using trained cVQ-VAE.")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to VQ-VAE YAML config")
    parser.add_argument('-w', '--weights', type=str, required=True, help="Path to VQ-VAE .pth weights")
    parser.add_argument('-n', type=int, default=100, help="Number of sequences per group (default: 100)")
    
    # Parametry celów (Wysokie / Niskie)
    parser.add_argument('--high_dev', type=float, default=8.0, help="Target for High Dev (default: 3.0)")
    parser.add_argument('--low_dev', type=float, default=-2.0, help="Target for Low Dev (default: -2.0)")
    parser.add_argument('--high_hk', type=float, default=8.0, help="Target for High HK (default: 3.0)")
    parser.add_argument('--low_hk', type=float, default=-2.0, help="Target for Low HK (default: -2.0)")
    
    args = parser.parse_args()
    main(args)
