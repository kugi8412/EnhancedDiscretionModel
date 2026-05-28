#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Advanced greedy sequence optimization with beam search, pruning, and refinement."""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from utils import load_config
from models.registry import build_model


def generate_from_vqvae(model, num_samples, dev_target, hk_target,
                        seq_len=249, device='cuda'):
    """Generate sequences de novo from random codebook indices via cVQ-VAE."""
    model.eval()
    with torch.no_grad():
        # Probe encoder for latent spatial dimension
        dummy_x = torch.zeros(1, 4, seq_len).to(device)
        z_e = model.encode_strand(dummy_x)
        latent_len = z_e.shape[2]
        
        # Sample random codebook indices
        num_embeddings = model.vq_layer.num_embeddings
        random_indices = torch.randint(0, num_embeddings, (num_samples, latent_len)).to(device)
        z_q = model.vq_layer.embed[random_indices].transpose(1, 2)
        
        # FiLM conditioning for target expression
        y_dev = torch.full((num_samples,), dev_target, dtype=torch.float32).to(device)
        y_hk = torch.full((num_samples,), hk_target, dtype=torch.float32).to(device)
        c = torch.stack([y_dev, y_hk], dim=1)
        
        film_params = model.film_generator(c)
        gamma = film_params[:, :z_q.size(1)].unsqueeze(2)
        beta = film_params[:, z_q.size(1):].unsqueeze(2)
        
        z_cond = (1.0 + gamma) * z_q + beta
        
        # Decode
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
    """Remove top and right spines for cleaner plots."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def onehot_to_string(onehot_tensor):
    """Convert one-hot tensor [B, 4, L] to list of DNA strings."""
    idx_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    indices = torch.argmax(onehot_tensor, dim=1).cpu().numpy()
    return ["".join(idx_to_char[i] for i in seq) for seq in indices]


def evaluate_oracle_batch(oracle, sequences_idx, batch_size=2048, device='cuda'):
    """Evaluate a batch of sequences (as integer indices) with the Oracle model."""
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
    """Generate all 3*(L) single-point mutations for a sequence of length L."""
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

def advanced_beam_search(orig_seq_idx, oracle, target_dev, target_hk,
                         beam_width=5, max_steps=20, prune_tol=0.05,
                         device='cuda'):
    """Greedy beam search testing all single-point mutations per step.

    Includes early stopping when no significant improvement is observed
    for two consecutive steps.
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
        
        # For each beam candidate, generate and evaluate all neighbours
        for state in beam:
            neighbors = generate_all_neighbors(state['seq'])
            p_dev, p_hk = evaluate_oracle_batch(oracle, neighbors, device=device)
            dists = (p_dev - target_dev)**2 + (p_hk - target_hk)**2
            
            for i, neighbor in enumerate(neighbors):
                seq_tuple = tuple(neighbor.tolist())
                if seq_tuple not in seen_hashes:
                    seen_hashes.add(seq_tuple)
                    all_candidates.append({'seq': neighbor, 'dist': dists[i]})
        
        # Sort and keep top K (beam width)
        all_candidates.sort(key=lambda x: x['dist'])
        beam = all_candidates[:beam_width]
        
        current_best_dist = beam[0]['dist']
        steps_taken += 1
        
        # Check early stopping: 2 consecutive stalls (Step 4)
        if (best_overall_dist - current_best_dist) < prune_tol:
            stall_count += 1
        else:
            stall_count = 0
            
        if current_best_dist < best_overall_dist:
            best_overall_dist = current_best_dist
            best_overall_seq = beam[0]['seq'].clone()
            
        # Stop if 2 consecutive steps showed no significant improvement
        if stall_count >= 2:
            break
            
        # Perfect hit
        if current_best_dist < 0.01:
            break

    return best_overall_seq, best_overall_dist, steps_taken

def prune_and_refine(best_seq, orig_seq, oracle, target_dev, target_hk,
                     prune_tol, device):
    """Prune unnecessary mutations and refine remaining ones.

    Phase 1 (Pruning): Revert mutations whose removal does not significantly
    increase the distance to the target.
    Phase 2 (Refinement): Try alternative nucleotides at each mutated position.
    """
    current_seq = best_seq.clone()
    
    # Aktualna ocena
    c_dev, c_hk = evaluate_oracle_batch(oracle, current_seq.unsqueeze(0), device=device)
    current_dist = (c_dev[0] - target_dev)**2 + (c_hk[0] - target_hk)**2
    
    # Identify mutated positions
    mutated_positions = torch.where(orig_seq != current_seq)[0].tolist()
    
    # Phase 1: Pruning - revert mutations that are not essential
    for pos in mutated_positions.copy():
        test_seq = current_seq.clone()
        test_seq[pos] = orig_seq[pos]  # Restore original nucleotide
        
        t_dev, t_hk = evaluate_oracle_batch(oracle, test_seq.unsqueeze(0), device=device)
        test_dist = (t_dev[0] - target_dev)**2 + (t_hk[0] - target_hk)**2
        
        # If reverting worsens distance by less than tolerance, accept revert
        if test_dist <= current_dist + prune_tol:
            current_seq = test_seq
            current_dist = test_dist
            mutated_positions.remove(pos)

    # Phase 2: Refinement - try alternative bases at mutated positions
    for pos in mutated_positions:
        best_local_dist = current_dist
        best_local_nuc = current_seq[pos].item()
        
        for nuc in range(4):
            if nuc != current_seq[pos].item() and nuc != orig_seq[pos].item():
                test_seq = current_seq.clone()
                test_seq[pos] = nuc
                
                t_dev, t_hk = evaluate_oracle_batch(oracle, test_seq.unsqueeze(0), device=device)
                test_dist = (t_dev[0] - target_dev)**2 + (t_hk[0] - target_hk)**2
                
                # Accept if this alternative gives a strictly better distance
                if test_dist < best_local_dist:
                    best_local_dist = test_dist
                    best_local_nuc = nuc
                    
        current_seq[pos] = best_local_nuc
        current_dist = best_local_dist

    # Final mutation count
    final_mutations = (orig_seq != current_seq).sum().item()
    return current_seq, current_dist, final_mutations


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Initialising Advanced Greedy Evolution on: {device}")
    
    config = load_config(args.config)
    oracle_cfg = config.get('oracle', None)
    oracle_config = load_config(oracle_cfg['config_path'])
    oracle = build_model(oracle_config).to(device)
    oracle.load_state_dict(torch.load(oracle_cfg['weights_path'], map_location=device, weights_only=True))
    oracle.eval()

    # Step 1: Generate starting sequences from the generative model
    print("\n[INFO] 1. Generating starting sequences from generative model...")
    vqvae = build_model(config).to(device)
    vqvae.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    vqvae.eval()
    
    # Generate starting sequences
    # For simplicity, generate One-Hot from VQ-VAE then take argmax indices
    orig_onehot = generate_from_vqvae(vqvae, args.n, args.start_dev, args.start_hk, device=device)
    orig_indices = torch.argmax(orig_onehot, dim=1) # [N, 249]
    
    p_dev_start, p_hk_start = evaluate_oracle_batch(oracle, orig_indices, device=device)

    mod_indices_list = []
    final_muts_list = []
    steps_list = []
    
    print(f"\n[INFO] Starting Beam Search optimisation (Beam={args.beam}, MaxSteps={args.max_steps})...")
    
    for i in range(args.n):
        orig_seq = orig_indices[i]
        
        # Step 2 & 4: Beam search with early stopping
        best_beam_seq, _, steps = advanced_beam_search(
            orig_seq, oracle, args.end_dev, args.end_hk, 
            beam_width=args.beam, max_steps=args.max_steps, prune_tol=args.tol, device=device
        )
        
        # Step 3 & 5: Pruning and refinement
        final_seq, _, final_muts = prune_and_refine(
            best_beam_seq, orig_seq, oracle, args.end_dev, args.end_hk, 
            prune_tol=args.tol, device=device
        )
        
        mod_indices_list.append(final_seq)
        final_muts_list.append(final_muts)
        steps_list.append(steps)
        
        if (i+1) % 10 == 0:
            print(f" -> Optimised {i+1}/{args.n} sequences (Avg steps: {np.mean(steps_list):.1f})")

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
    print(f"\n[SUCCESS] Saved FASTA: {fasta_path}")

    # Results scatter plot
    fig, ax = plt.subplots(figsize=(11, 8))
    
    for i in range(args.n):
        ax.annotate("", xy=(p_hk_mod[i], p_dev_mod[i]), xycoords='data',
                    xytext=(p_hk_start[i], p_dev_start[i]), textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="purple", alpha=0.4, lw=1.5))

    ax.scatter(p_hk_start, p_dev_start, color='mediumorchid', marker='o', s=50, label='Start Sequences', edgecolors='w')
    ax.scatter(p_hk_mod, p_dev_mod, color='indigo', marker='X', s=80, label='Optimized (Greedy + Refined)', edgecolors='w')
    
    ax.plot(args.start_hk, args.start_dev, marker='*', color='gray', markersize=15, markeredgecolor='k', label='START Target')
    ax.plot(args.end_hk, args.end_dev, marker='*', color='limegreen', markersize=15, markeredgecolor='k', label='END Target')

    # Statistics for plot title
    avg_steps = np.mean(steps_list)
    min_steps = np.min(steps_list)
    max_steps = np.max(steps_list)
    avg_muts = np.mean(final_muts_list)
    
    adjust_axes(ax)
    ax.set_xlabel('Oracle Predicted HK Fold Change [log2]', fontsize=12)
    ax.set_ylabel('Oracle Predicted Dev Fold Change [log2]', fontsize=12)
    
    title = (f'Advanced Greedy Optimization (Beam={args.beam}, Tol={args.tol})\n'
             f'Steps taken -> Avg: {avg_steps:.1f} | Min: {min_steps} | Max: {max_steps}\n'
             f'Avg Final Mutations: {avg_muts:.1f} / 249')
    ax.set_title(title, fontsize=13, pad=15)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"Advanced_Greedy_Evolution_Plot_n{args.n}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Plot saved to: {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced Greedy Optimization with Pruning & Refinement.")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to config")
    parser.add_argument('-w', '--weights', type=str, required=True, help="Path to VQVAE weights (for generating starts)")
    parser.add_argument('-n', type=int, default=100, help="Number of sequences")
    
    parser.add_argument('--start_dev', type=float, default=-2.0, help="Start Dev")
    parser.add_argument('--start_hk', type=float, default=-2.0, help="Start HK")
    parser.add_argument('--end_dev', type=float, default=6.0, help="Target Dev")
    parser.add_argument('--end_hk', type=float, default=6.0, help="Target HK")
    
    # Optimisation parameters
    parser.add_argument('--beam', type=int, default=5, help="Beam width")
    parser.add_argument('--max_steps', type=int, default=20, help="Maximum number of iteration steps")
    parser.add_argument('--tol', type=float, default=0.05, help="MSE tolerance for early stopping and pruning")
    
    args = parser.parse_args()
    main(args)
