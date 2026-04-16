#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from utils import load_config
from models.registry import build_model

def generate_from_vqvae(model, num_samples, dev_target, hk_target, seq_len=249, device='cuda'):
    """Generuje sekwencje z losowego szumu (od zera) używając cVQ-VAE."""
    model.eval()
    with torch.no_grad():
        # Używamy dummy_x tylko po to, żeby odpytać Enkoder o długość przestrzeni ukrytej (latent_len)
        dummy_x = torch.zeros(1, 4, seq_len).to(device)
        z_e = model.encode_strand(dummy_x)
        latent_len = z_e.shape[2]
        
        # Losujemy indeksy ze słownika
        num_embeddings = model.vq_layer.num_embeddings
        random_indices = torch.randint(0, num_embeddings, (num_samples, latent_len)).to(device)
        z_q = model.vq_layer.embed[random_indices].transpose(1, 2)
        
        # Tworzymy wektory warunkowe (FiLM) dla docelowej ekspresji
        y_dev = torch.full((num_samples,), dev_target, dtype=torch.float32).to(device)
        y_hk = torch.full((num_samples,), hk_target, dtype=torch.float32).to(device)
        c = torch.stack([y_dev, y_hk], dim=1)
        
        film_params = model.film_generator(c)
        gamma = film_params[:, :z_q.size(1)].unsqueeze(2)
        beta = film_params[:, z_q.size(1):].unsqueeze(2)
        
        z_cond = (1.0 + gamma) * z_q + beta
        
        # Dekodowanie
        d = model.decoder_cond_proj(z_cond)
        d = d.permute(0, 2, 1)
        d, _ = model.decoder_gru(d)
        d = d.permute(0, 2, 1)
        d = model.decoder_blocks(d)
        x_logits = model.decoder_out(d)
        
        if x_logits.size(2) != seq_len:
            x_logits = F.interpolate(x_logits, size=seq_len, mode='linear', align_corners=False)
            
        x_fwd = x_logits[:, 0:4, :]
        seq_indices = torch.argmax(x_fwd, dim=1)
        
    return F.one_hot(seq_indices, num_classes=4).transpose(1, 2).float()

def adjust_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def onehot_to_string(onehot_tensor):
    idx_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    indices = torch.argmax(onehot_tensor, dim=1).cpu().numpy()
    sequences = ["".join([idx_to_char[i] for i in seq]) for seq in indices]
    return sequences

def evaluate_oracle_batch(oracle, sequences_idx, batch_size=2048, device='cuda'):
    """Ewaluuje dużą paczkę sekwencji (indeksów) za pomocą Wyroczni."""
    p_dev_list, p_hk_list = [], []
    with torch.no_grad():
        for i in range(0, len(sequences_idx), batch_size):
            batch_idx = sequences_idx[i:i+batch_size].to(device)
            batch_oh = F.one_hot(batch_idx, num_classes=4).transpose(1, 2).float()
            
            p_dev, p_hk = oracle(batch_oh)
            if isinstance(p_dev, (list, tuple)):
                p_dev, p_hk = p_dev[0], p_dev[1]
            elif p_dev.dim() > 1 and p_dev.shape[1] == 2:
                p_hk, p_dev = p_dev[:, 1], p_dev[:, 0]
                
            p_dev_list.append(p_dev.cpu().numpy().flatten())
            p_hk_list.append(p_hk.cpu().numpy().flatten())
            
    return np.concatenate(p_dev_list), np.concatenate(p_hk_list)

def generate_all_neighbors(seq_idx):
    """Generuje wszystkie 747 możliwych mutacji 1-punktowych dla sekwencji 249bp."""
    seq_len = len(seq_idx)
    neighbors = []
    for pos in range(seq_len):
        current_nuc = seq_idx[pos].item()
        for new_nuc in range(4):
            if new_nuc != current_nuc:
                neighbor = seq_idx.clone()
                neighbor[pos] = new_nuc
                neighbors.append(neighbor)
    return torch.stack(neighbors)

def advanced_beam_search(orig_seq_idx, oracle, target_dev, target_hk, beam_width=5, max_steps=20, prune_tol=0.05, device='cuda'):
    """
    KROK 1 i 2: Zachłanny Beam Search z testowaniem WSZYSTKICH 747 mutacji.
    KROK 4: Kryterium stopu (brak poprawy przez 2 kroki).
    """
    # Ewaluacja startowa
    start_dev, start_hk = evaluate_oracle_batch(oracle, orig_seq_idx.unsqueeze(0), device=device)
    start_dist = (start_dev[0] - target_dev)**2 + (start_hk[0] - target_hk)**2
    
    beam = [{'seq': orig_seq_idx.clone(), 'dist': start_dist}]
    
    best_overall_seq = orig_seq_idx.clone()
    best_overall_dist = start_dist
    
    stall_count = 0
    steps_taken = 0
    
    for step in range(max_steps):
        all_candidates = []
        seen_hashes = set()
        
        # Dla każdej sekwencji w Beam, generujemy i oceniamy wszystkich sąsiadów
        for state in beam:
            neighbors = generate_all_neighbors(state['seq'])
            p_dev, p_hk = evaluate_oracle_batch(oracle, neighbors, device=device)
            dists = (p_dev - target_dev)**2 + (p_hk - target_hk)**2
            
            for i, neighbor in enumerate(neighbors):
                seq_tuple = tuple(neighbor.tolist())
                if seq_tuple not in seen_hashes:
                    seen_hashes.add(seq_tuple)
                    all_candidates.append({'seq': neighbor, 'dist': dists[i]})
        
        # Sortujemy i wybieramy top K (Beam Width)
        all_candidates.sort(key=lambda x: x['dist'])
        beam = all_candidates[:beam_width]
        
        current_best_dist = beam[0]['dist']
        steps_taken += 1
        
        # Sprawdzenie kryterium stopu (Pruning Criterion)
        if (best_overall_dist - current_best_dist) < prune_tol:
            stall_count += 1
        else:
            stall_count = 0
            
        if current_best_dist < best_overall_dist:
            best_overall_dist = current_best_dist
            best_overall_seq = beam[0]['seq'].clone()
            
        # Zatrzymujemy jeśli 2 razy z rzędu nie było znaczącej poprawy (KROK 4)
        if stall_count >= 2:
            break
            
        # Idealne trafienie
        if current_best_dist < 0.01:
            break

    return best_overall_seq, best_overall_dist, steps_taken

def prune_and_refine(best_seq, orig_seq, oracle, target_dev, target_hk, prune_tol, device):
    """
    KROK 3: Pruning (Cofanie mutacji jeśli MSE nie rośnie bardzo).
    KROK 5: Refinement (Próba innej litery na zmutowanej pozycji).
    """
    current_seq = best_seq.clone()
    
    # Aktualna ocena
    c_dev, c_hk = evaluate_oracle_batch(oracle, current_seq.unsqueeze(0), device=device)
    current_dist = (c_dev[0] - target_dev)**2 + (c_hk[0] - target_hk)**2
    
    # Identyfikacja zmutowanych pozycji
    mutated_positions = torch.where(orig_seq != current_seq)[0].tolist()
    
    # ---------------------------------------------------------
    # FAZA 3: PRUNING (Cofanie)
    # ---------------------------------------------------------
    for pos in mutated_positions.copy():
        test_seq = current_seq.clone()
        test_seq[pos] = orig_seq[pos] # Przywracamy oryginał
        
        t_dev, t_hk = evaluate_oracle_batch(oracle, test_seq.unsqueeze(0), device=device)
        test_dist = (t_dev[0] - target_dev)**2 + (t_hk[0] - target_hk)**2
        
        # Jeśli powrót pogarsza dystans o MNIEJ niż tolerancja -> cofamy mutację!
        if test_dist <= current_dist + prune_tol:
            current_seq = test_seq
            current_dist = test_dist
            mutated_positions.remove(pos) # Usunięto z listy mutacji

    # ---------------------------------------------------------
    # FAZA 5: REFINEMENT (Doskonalenie)
    # ---------------------------------------------------------
    for pos in mutated_positions:
        best_local_dist = current_dist
        best_local_nuc = current_seq[pos].item()
        
        for nuc in range(4):
            if nuc != current_seq[pos].item() and nuc != orig_seq[pos].item():
                test_seq = current_seq.clone()
                test_seq[pos] = nuc
                
                t_dev, t_hk = evaluate_oracle_batch(oracle, test_seq.unsqueeze(0), device=device)
                test_dist = (t_dev[0] - target_dev)**2 + (t_hk[0] - target_hk)**2
                
                # Jeśli inna litera daje LEPSZY dystans bezwzględnie -> przyjmujemy
                if test_dist < best_local_dist:
                    best_local_dist = test_dist
                    best_local_nuc = nuc
                    
        current_seq[pos] = best_local_nuc
        current_dist = best_local_dist

    # Ostateczne podliczenie mutacji
    final_mutations = (orig_seq != current_seq).sum().item()
    return current_seq, current_dist, final_mutations


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Inicjalizacja Advanced Greedy Evolution na: {device}")
    
    config = load_config(args.config)
    oracle_cfg = config.get('oracle', None)
    oracle_config = load_config(oracle_cfg['config_path'])
    oracle = build_model(oracle_config).to(device)
    oracle.load_state_dict(torch.load(oracle_cfg['weights_path'], map_location=device, weights_only=True))
    oracle.eval()

    # KROK 1: Losowanie 100 sekwencji o startowych ekspresjach (Używamy generatora VQ-VAE jak w Twoim skrypcie)
    print("\n[INFO] 1. Generowanie sekwencji startowych z modelu generatywnego...")
    vqvae = build_model(config).to(device)
    vqvae.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    vqvae.eval()
    
    # Import Twojej funkcji z poprzedniego skryptu lub po prostu wklej jej ciało tutaj
    # Dla uproszczenia, losujemy sekwencje One-Hot, a następnie bierzemy indeksy
    # Generowanie sekwencji One-Hot bezpośrednio z wewnętrznej funkcji
    orig_onehot = generate_from_vqvae(vqvae, args.n, args.start_dev, args.start_hk, device=device)
    orig_indices = torch.argmax(orig_onehot, dim=1) # [N, 249]
    
    p_dev_start, p_hk_start = evaluate_oracle_batch(oracle, orig_indices, device=device)

    mod_indices_list = []
    final_muts_list = []
    steps_list = []
    
    print(f"\n[INFO] Rozpoczynam optymalizację Beam Search (Beam={args.beam}, MaxSteps={args.max_steps})...")
    
    for i in range(args.n):
        orig_seq = orig_indices[i]
        
        # KROK 2 & 4: Beam Search z wczesnym zatrzymaniem
        best_beam_seq, _, steps = advanced_beam_search(
            orig_seq, oracle, args.end_dev, args.end_hk, 
            beam_width=args.beam, max_steps=args.max_steps, prune_tol=args.tol, device=device
        )
        
        # KROK 3 & 5: Pruning i Refinement
        final_seq, _, final_muts = prune_and_refine(
            best_beam_seq, orig_seq, oracle, args.end_dev, args.end_hk, 
            prune_tol=args.tol, device=device
        )
        
        mod_indices_list.append(final_seq)
        final_muts_list.append(final_muts)
        steps_list.append(steps)
        
        if (i+1) % 10 == 0:
            print(f" -> Zoptymalizowano {i+1}/{args.n} sekwencji (Śr. kroków: {np.mean(steps_list):.1f})")

    mod_indices = torch.stack(mod_indices_list)
    p_dev_mod, p_hk_mod = evaluate_oracle_batch(oracle, mod_indices, device=device)

    # --- ZAPIS DO FASTA ---
    fasta_records = []
    orig_strs = onehot_to_string(F.one_hot(orig_indices, num_classes=4).transpose(1, 2).float())
    mod_strs = onehot_to_string(F.one_hot(mod_indices, num_classes=4).transpose(1, 2).float())
    
    for i in range(args.n):
        fasta_records.append(f">Start_Seq_{i+1} | Dev:{p_dev_start[i]:.2f} | HK:{p_hk_start[i]:.2f}")
        fasta_records.append(orig_strs[i])
        fasta_records.append(f">Optimized_Seq_{i+1} | Dev:{p_dev_mod[i]:.2f} | HK:{p_hk_mod[i]:.2f} | Steps:{steps_list[i]} | Final_Muts:{final_muts_list[i]}/249")
        fasta_records.append(mod_strs[i])

    out_dir = "outputs/generated"
    os.makedirs(out_dir, exist_ok=True)
    fasta_path = os.path.join(out_dir, f"Advanced_Greedy_Evolution_n{args.n}.fasta")
    with open(fasta_path, "w") as f:
        f.write("\n".join(fasta_records))
    print(f"\n[SUCCESS] Zapisano FASTA: {fasta_path}")

    # --- WYKRES Z KRYTERIAMI KROKÓW ---
    fig, ax = plt.subplots(figsize=(11, 8))
    
    for i in range(args.n):
        ax.annotate("", xy=(p_hk_mod[i], p_dev_mod[i]), xycoords='data',
                    xytext=(p_hk_start[i], p_dev_start[i]), textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="purple", alpha=0.4, lw=1.5))

    ax.scatter(p_hk_start, p_dev_start, color='mediumorchid', marker='o', s=50, label='Start Sequences', edgecolors='w')
    ax.scatter(p_hk_mod, p_dev_mod, color='indigo', marker='X', s=80, label='Optimized (Greedy + Refined)', edgecolors='w')
    
    ax.plot(args.start_hk, args.start_dev, marker='*', color='gray', markersize=15, markeredgecolor='k', label='START Target')
    ax.plot(args.end_hk, args.end_dev, marker='*', color='limegreen', markersize=15, markeredgecolor='k', label='END Target')

    # Obliczanie statystyk do tytułu (Zgodnie z wymaganiem nr 4)
    avg_steps = np.mean(steps_list)
    min_steps = np.min(steps_list)
    max_steps = np.max(steps_list)
    avg_muts = np.mean(final_muts_list)
    
    adjust_axes(ax)
    ax.set_xlabel('LegNet Predicted HK Fold Change [log2]', fontsize=12)
    ax.set_ylabel('LegNet Predicted Dev Fold Change [log2]', fontsize=12)
    
    title = (f'Advanced Greedy Optimization (Beam={args.beam}, Tol={args.tol})\n'
             f'Steps taken -> Avg: {avg_steps:.1f} | Min: {min_steps} | Max: {max_steps}\n'
             f'Avg Final Mutations: {avg_muts:.1f} / 249')
    ax.set_title(title, fontsize=13, pad=15)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"Advanced_Greedy_Evolution_Plot_n{args.n}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Wykres zapisano w: {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced Greedy Optimization with Pruning & Refinement.")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to config")
    parser.add_argument('-w', '--weights', type=str, required=True, help="Path to VQVAE weights (for generating starts)")
    parser.add_argument('-n', type=int, default=100, help="Number of sequences")
    
    parser.add_argument('--start_dev', type=float, default=-2.0, help="Start Dev")
    parser.add_argument('--start_hk', type=float, default=-2.0, help="Start HK")
    parser.add_argument('--end_dev', type=float, default=6.0, help="Target Dev")
    parser.add_argument('--end_hk', type=float, default=6.0, help="Target HK")
    
    # Parametry optymalizacji
    parser.add_argument('--beam', type=int, default=5, help="Szerokość strumienia (Beam Width)")
    parser.add_argument('--max_steps', type=int, default=20, help="Maksymalna liczba kroków iteracji")
    parser.add_argument('--tol', type=float, default=0.05, help="Tolerancja spadku MSE (dla Early Stopping i Pruningu)")
    
    args = parser.parse_args()
    main(args)
