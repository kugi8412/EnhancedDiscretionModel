#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from utils import load_config, prepare_input
from models.registry import build_model


def encode_pre_vq(model, x):
    """Encode input to pre-VQ latent (works with any cVQVAE variant)."""
    if hasattr(model, 'encode_strand'):
        return model.encode_strand(x)
    elif hasattr(model, '_encode'):
        return model._encode(x)
    else:
        raise ValueError("Model has no recognized encoding method (encode_strand or _encode).")


def decode_with_film(model, z_q, y_dev, y_hk, seq_len=249):
    """
    Decode a quantized latent tensor through any cVQVAE variant with FiLM conditioning.
    Supports cVQVAE_MultiTask, cVQVAE_Asymmetric, and HydraDNA_cVQVAE.
    """
    B = z_q.size(0)
    cond = torch.stack([y_dev, y_hk], dim=1)

    # --- cVQVAE_Asymmetric / cVQVAE_MultiTask (has .film as FiLMGenerator) ---
    if hasattr(model, 'film') and hasattr(model.film, 'net'):
        gamma, beta = model.film(cond)
        # Use the model's own _decode which handles GRU presence/absence
        if hasattr(model, '_decode'):
            fwd_logits, _ = model._decode(z_q, gamma, beta, seq_len)
            return fwd_logits
        # Fallback: manual path for cVQVAE_MultiTask
        cq = (1.0 + gamma) * z_q + beta
        d = model.decoder_cond_proj(cq) if hasattr(model, 'decoder_cond_proj') else model.dec_cond_proj(cq)
        if hasattr(model, 'decoder_gru') and model.decoder_gru is not None:
            d = d.permute(0, 2, 1)
            d, _ = model.decoder_gru(d)
            d = d.permute(0, 2, 1)
        d = model.decoder_blocks(d)
        x_logits = model.decoder_out(d)
        if x_logits.size(2) != seq_len:
            x_logits = F.interpolate(x_logits, size=seq_len, mode='linear', align_corners=False)
        return x_logits[:, 0:4, :]

    # --- HydraDNA_cVQVAE (has .film_generator as nn.Sequential) ---
    elif hasattr(model, 'film_generator'):
        film_params = model.film_generator(cond)
        vq_dim = z_q.size(1)
        gamma = film_params[:, :vq_dim].unsqueeze(2)
        beta = film_params[:, vq_dim:].unsqueeze(2)

        z_cond = (1.0 + gamma) * z_q + beta
        d = model.decoder_cond_proj(z_cond)
        d = d.permute(0, 2, 1)
        d, _ = model.decoder_gru(d)
        d = d.permute(0, 2, 1)
        d = model.decoder_blocks(d)
        x_logits = model.decoder_out(d)
        if x_logits.size(2) != seq_len:
            x_logits = F.interpolate(x_logits, size=seq_len, mode='linear', align_corners=False)
        return x_logits[:, 0:4, :]

    else:
        raise ValueError("Model does not have a recognized FiLM conditioning interface.")


# ==========================================
# 1. LATENT SPACE QUADRANT CLUSTERING
# ==========================================
class QuadrantClustering(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.centroids = nn.Parameter(torch.randn(4, latent_dim))
        self.threshold_dev = 0.0
        self.threshold_hk = 0.0

    def get_target_quadrants(self, y_dev, y_hk):
        dev_high = (y_dev > self.threshold_dev).long()
        hk_high = (y_hk > self.threshold_hk).long()
        return dev_high * 2 + hk_high

    def forward(self, z_e, y_dev, y_hk):
        z_e_mean = z_e.mean(dim=2) 
        target_indices = self.get_target_quadrants(y_dev, y_hk).squeeze()
        target_centroids = self.centroids[target_indices]
        clustering_loss = F.mse_loss(z_e_mean, target_centroids)
        return clustering_loss


# ==========================================
# 2. RL AGENT (LATENT NAVIGATOR)
# ==========================================
class LatentNavigator(nn.Module):
    def __init__(self, num_embeddings, latent_len=62): 
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, 64)
        
        self.fc = nn.Sequential(
            nn.Linear(64 + 2, 256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_embeddings) 
        )
        
        # Zero-initialize the final layer weights
        nn.init.zeros_(self.fc[4].weight)
        nn.init.zeros_(self.fc[4].bias)

    def forward(self, current_indices, target_expr):
        B, L = current_indices.shape
        x_emb = self.embedding(current_indices) 
        t_exp = target_expr.unsqueeze(1).expand(B, L, 2) 
        
        x_in = torch.cat([x_emb, t_exp], dim=-1) 
        action_logits = self.fc(x_in) # [B, L, 512]
        
        # --- NO-OP BIAS: prefer keeping current codebook entry ---
        device = action_logits.device
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
        pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        
        action_logits[batch_idx, pos_idx, current_indices] += 10.0
        
        return action_logits


# ==========================================
# 3. MAIN TRAINING LOOP
# ==========================================
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    print(f"[INFO] Initializing RL environment on device: {device}")
    
    vqvae = build_model(config).to(device)
    vqvae.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    
    oracle_cfg = config['oracle']
    oracle = build_model(load_config(oracle_cfg['config_path'])).to(device)
    oracle.load_state_dict(torch.load(oracle_cfg['weights_path'], map_location=device, weights_only=True))
    oracle.eval() 
    
    dummy_x = torch.zeros(1, 4, config['data'].get('seq_len', 249)).to(device)
    z_dummy = encode_pre_vq(vqvae, dummy_x)
    latent_len = z_dummy.shape[2]
    num_embeddings = vqvae.vq_layer.num_embeddings
    
    print(f"[INFO] Latent space -> Latent Len: {latent_len}, Dict Size: {num_embeddings}")
    
    clustering_module = QuadrantClustering(latent_dim=64).to(device)
    navigator = LatentNavigator(num_embeddings, latent_len).to(device)
    
    opt_vq = optim.Adam(list(vqvae.parameters()) + list(clustering_module.parameters()), lr=1e-4)
    opt_rl = optim.Adam(navigator.parameters(), lr=1e-4)
    
    train_loader = prepare_input('Train', config)
    
    reward_baseline = 0.0 

    print("[INFO] Starting training...")
    total_epochs = 100
    # Epoch at which the curriculum schedule reaches its final values
    max_sched_epoch = 80.0 

    for epoch in range(total_epochs):
        
        # --- CURRICULUM LEARNING SCHEDULE ---
        progress = min(1.0, epoch / max_sched_epoch)
        
        # Alpha: decays from 20.0 to 10.0 (agent still cares about the target)
        alpha_mse = 20.0 - (10.0 * progress)     
        
        # Mutation penalty (Beta) ramp-up
        # Starts at 0.5, grows to 2.5 to penalize excessive mutations
        beta_mut = 0.5 + (2.0 * progress)      
        
        for batch_idx, (X_batch, Y_dev, Y_hk) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            Y_dev, Y_hk = Y_dev.to(device), Y_hk.to(device)
            B = X_batch.size(0)
            seq_len = X_batch.size(2)
            
            # ========================================================
            # PHASE 1: VQ-VAE Fine-Tuning + Codebook Clustering
            # ========================================================
            vqvae.train()
            opt_vq.zero_grad()
            
            z_e = encode_pre_vq(vqvae, X_batch)
            z_q, vq_loss, current_indices = vqvae.vq_layer(z_e)
            
            x_recon_logits = decode_with_film(vqvae, z_q, Y_dev, Y_hk, seq_len=seq_len)
            
            recon_loss = F.cross_entropy(x_recon_logits, torch.argmax(X_batch, dim=1))
            cluster_loss = clustering_module(z_e, Y_dev, Y_hk)
            
            loss_vqvae = recon_loss + vq_loss + (0.5 * cluster_loss)
            loss_vqvae.backward()
            opt_vq.step()
            
            orig_seq = torch.argmax(x_recon_logits.detach(), dim=1)
            
            # ========================================================
            # PHASE 2: REINFORCEMENT LEARNING (Actor explores)
            # ========================================================
            navigator.train()
            vqvae.eval() 
            opt_rl.zero_grad()
            
            target_dev = (torch.rand(B).to(device) * 6.0) - 3.0
            target_hk = (torch.rand(B).to(device) * 6.0) - 3.0
            target_expr = torch.stack([target_dev, target_hk], dim=1)
            
            action_logits = navigator(current_indices.detach(), target_expr)
            dist = Categorical(logits=action_logits)
            new_indices = dist.sample() 
            log_probs = dist.log_prob(new_indices).sum(dim=1) 
            
            # ========================================================
            # PHASE 3: ENVIRONMENT EVALUATES AGENT (Reward)
            # ========================================================
            with torch.no_grad():
                z_q_new = vqvae.vq_layer.embed[new_indices].transpose(1, 2).contiguous()
                
                x_logits_new = decode_with_film(vqvae, z_q_new, target_dev, target_hk, seq_len=seq_len)
                x_onehot_new = F.one_hot(torch.argmax(x_logits_new, dim=1), num_classes=4).transpose(1, 2).float()
                
                pred_dev, pred_hk = oracle(x_onehot_new)
                if isinstance(pred_dev, (list, tuple)):
                    pred_dev, pred_hk = pred_dev[0], pred_hk[1]
                elif pred_dev.dim() > 1 and pred_dev.shape[1] == 2:
                    pred_hk, pred_dev = pred_dev[:, 1], pred_dev[:, 0]
                    
                pred_dev, pred_hk = pred_dev.squeeze(), pred_hk.squeeze()
                
                mse_dev = (pred_dev - target_dev)**2
                mse_hk = (pred_hk - target_hk)**2
                mse_total = mse_dev + mse_hk
                
                new_seq = torch.argmax(x_logits_new, dim=1)
                mutations_count = (orig_seq != new_seq).sum(dim=1).float() 
                
                # Use the curriculum-scheduled alpha and beta weights
                reward = - (alpha_mse * mse_total) - (beta_mut * mutations_count)

            # ========================================================
            # PHASE 4: AGENT UPDATE (REINFORCE)
            # ========================================================
            current_mean_reward = reward.mean().item()
            reward_baseline = 0.9 * reward_baseline + 0.1 * current_mean_reward
            advantage = reward - reward_baseline
            
            policy_loss = -(log_probs * advantage.detach()).mean()
            
            policy_loss.backward()
            opt_rl.step()
            
            if batch_idx % 50 == 0:
                print(f"Ep {epoch} | B {batch_idx} | α:{alpha_mse:.1f} β:{beta_mut:.3f} | VQ Loss: {loss_vqvae.item():.3f} | Rew: {current_mean_reward:.1f} | Muts: {mutations_count.mean().item():.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to config (cVQVAE)")
    parser.add_argument('-w', '--weights', type=str, required=True, help="Path to VQVAE weights")
    args = parser.parse_args()
    
    main(args)
