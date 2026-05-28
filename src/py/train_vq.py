#!/usr/bin/env python
"""
Training script for VQ-VAE generative models (LegNet_VQVAE, HydraDNA_cVQVAE, DNA_PixelCNN).
Handles reconstruction loss, VQ commitment loss, oracle guidance, and codebook training.

Usage:
    python train_vq.py -c ../../config/HydraDNA_cVQVAEPlus.yaml
"""
try:
    import comet_ml
except ImportError:
    comet_ml = None

import os
import csv
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

from utils import prepare_input, load_config, set_global_seed
from models.registry import build_model


def calculate_metrics(targets, preds):
    targets_t = torch.tensor(targets, dtype=torch.float32)
    preds_t = torch.tensor(preds, dtype=torch.float32)
    mse = F.mse_loss(preds_t, targets_t).item()
    if np.std(preds) == 0 or np.std(targets) == 0:
        return mse, 0.0, 0.0
    pcc = pearsonr(targets, preds)[0]
    scc = spearmanr(targets, preds)[0]
    return mse, pcc, scc


def plot_vq_history(history, log_dir, seed):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    epochs = history['Epoch']

    axes[0].plot(epochs, history['Tr_Loss'], label='Train')
    axes[0].plot(epochs, history['Val_Loss'], label='Val')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)

    if history.get('Val_Recon_Acc'):
        axes[1].plot(epochs, history['Tr_Recon_Acc'], label='Train', linestyle='--')
        axes[1].plot(epochs, history['Val_Recon_Acc'], label='Val')
        axes[1].set_title('Reconstruction Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)

    axes[2].plot(epochs, history.get('Val_PCC_Dev', [0]*len(epochs)), label='Val Dev')
    axes[2].plot(epochs, history.get('Val_PCC_Hk', [0]*len(epochs)), label='Val Hk')
    axes[2].set_title('Oracle PCC (Val)')
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(epochs, history.get('Oracle_Weight', [0]*len(epochs)))
    axes[3].set_title('Oracle Weight Schedule')
    axes[3].grid(True)

    for ax in axes:
        ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'vq_training_plot_{seed}.png'))
    plt.close()


def compute_vq_loss(model, X_batch, Y_dev_batch, Y_hk_batch, oracle_model,
                    oracle_weight, recon_weight, vq_weight,
                    oracle_bidirectional, criterion):
    """Unified loss computation for all VQ model types."""
    try:
        outputs = model(X_batch, Y_dev_batch, Y_hk_batch)
    except TypeError:
        outputs = model(X_batch)

    recon_acc = None
    pred_dev, pred_hk = None, None

    # --- cVQVAE_MultiTask (5-output: logits_8ch, gumbels, vq_loss, pred_dev, pred_hk) ---
    if isinstance(outputs, tuple) and len(outputs) == 5:
        x_logits_8ch, gumbels, vq_loss, pred_dev, pred_hk = outputs
        x_gumbel_fwd, x_gumbel_rc = gumbels
        x_fwd_logits = x_logits_8ch[:, 0:4, :]
        x_rc_logits  = x_logits_8ch[:, 4:8, :]

        true_seq = torch.argmax(X_batch, dim=1)
        X_rc     = torch.flip(X_batch, dims=[1, 2])
        true_rc  = torch.argmax(X_rc, dim=1)

        loss_recon = (F.cross_entropy(x_fwd_logits, true_seq) +
                      F.cross_entropy(x_rc_logits,  true_rc)) / 2.0

        # Direct multitask head loss
        loss_direct = (criterion(pred_dev.squeeze(), Y_dev_batch) +
                       criterion(pred_hk.squeeze(),  Y_hk_batch))

        # Optional oracle guidance through Gumbel samples
        if oracle_model is not None:
            with torch.no_grad():
                od_fwd, oh_fwd = oracle_model(x_gumbel_fwd)
                if oracle_bidirectional:
                    od_rc, oh_rc = oracle_model(x_gumbel_rc)
                    loss_oracle = (criterion(od_fwd.squeeze(), Y_dev_batch) +
                                   criterion(od_rc.squeeze(),  Y_dev_batch) +
                                   criterion(oh_fwd.squeeze(), Y_hk_batch) +
                                   criterion(oh_rc.squeeze(),  Y_hk_batch)) / 2.0
                else:
                    loss_oracle = (criterion(od_fwd.squeeze(), Y_dev_batch) +
                                   criterion(oh_fwd.squeeze(), Y_hk_batch))
            loss = (oracle_weight * loss_oracle + recon_weight * loss_recon +
                    vq_weight * vq_loss + loss_direct)
        else:
            loss = recon_weight * loss_recon + vq_weight * vq_loss + loss_direct

        with torch.no_grad():
            preds_seq = torch.argmax(x_fwd_logits, dim=1)
            recon_acc = (preds_seq == true_seq).float().mean(dim=1) * 100.0

    # --- cVQ-VAE (3-output: logits_8ch, gumbels, vq_loss) ---
    elif isinstance(outputs, tuple) and len(outputs) == 3 and oracle_model is not None:
        x_logits_8ch, gumbels, vq_loss = outputs
        x_gumbel_fwd, x_gumbel_rc = gumbels

        x_fwd_logits = x_logits_8ch[:, 0:4, :]
        x_rc_logits = x_logits_8ch[:, 4:8, :]

        true_seq = torch.argmax(X_batch, dim=1)
        X_rc = torch.flip(X_batch, dims=[1, 2])
        true_rc = torch.argmax(X_rc, dim=1)

        loss_recon = (F.cross_entropy(x_fwd_logits, true_seq) +
                      F.cross_entropy(x_rc_logits, true_rc)) / 2.0

        pred_dev_fwd, pred_hk_fwd = oracle_model(x_gumbel_fwd)
        if oracle_bidirectional:
            pred_dev_rc, pred_hk_rc = oracle_model(x_gumbel_rc)
            loss_expr = (criterion(pred_dev_fwd.squeeze(), Y_dev_batch) +
                         criterion(pred_dev_rc.squeeze(), Y_dev_batch) +
                         criterion(pred_hk_fwd.squeeze(), Y_hk_batch) +
                         criterion(pred_hk_rc.squeeze(), Y_hk_batch)) / 2.0
            pred_dev = (pred_dev_fwd + pred_dev_rc) / 2.0
            pred_hk = (pred_hk_fwd + pred_hk_rc) / 2.0
        else:
            loss_expr = criterion(pred_dev_fwd.squeeze(), Y_dev_batch) + \
                        criterion(pred_hk_fwd.squeeze(), Y_hk_batch)
            pred_dev = pred_dev_fwd
            pred_hk = pred_hk_fwd

        loss = oracle_weight * loss_expr + recon_weight * loss_recon + vq_weight * vq_loss

        with torch.no_grad():
            preds_seq = torch.argmax(x_fwd_logits, dim=1)
            recon_acc = (preds_seq == true_seq).float().mean(dim=1) * 100.0

    # --- LegNet_VQVAE (4-output: dev, hk, recon, vq_loss) ---
    elif isinstance(outputs, tuple) and len(outputs) == 4:
        pred_dev, pred_hk, x_recon, vq_loss = outputs
        true_seq = torch.argmax(X_batch, dim=1)
        loss_recon = F.cross_entropy(x_recon, true_seq)
        loss_expr = criterion(pred_dev.squeeze(), Y_dev_batch) + \
                    criterion(pred_hk.squeeze(), Y_hk_batch)

        loss = oracle_weight * loss_expr + recon_weight * loss_recon + vq_weight * vq_loss

        with torch.no_grad():
            preds_seq = torch.argmax(x_recon, dim=1)
            recon_acc = (preds_seq == true_seq).float().mean(dim=1) * 100.0

    # --- PixelCNN (returns logits [B, 4, L]) ---
    elif isinstance(outputs, torch.Tensor) and outputs.dim() == 3 and outputs.shape[1] == 4:
        true_seq = torch.argmax(X_batch, dim=1)
        loss = F.cross_entropy(outputs, true_seq)
        vq_loss = torch.tensor(0.0)

        with torch.no_grad():
            preds_seq = torch.argmax(outputs, dim=1)
            recon_acc = (preds_seq == true_seq).float().mean(dim=1) * 100.0

    else:
        raise ValueError(f"Unexpected model output format: {type(outputs)}, len={len(outputs) if isinstance(outputs, tuple) else 'N/A'}")

    return loss, pred_dev, pred_hk, recon_acc


def train_vq(config):
    seed = config.get('seed', 42)
    set_global_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Training VQ model on device: {device}")

    train_cfg = config.get('training', {})
    epochs = train_cfg.get('epochs', 100)
    lr = float(train_cfg.get('lr', 1e-4))
    weight_decay = float(train_cfg.get('weight_decay', 1e-4))
    early_stop_patience = train_cfg.get('early_stop', 15)
    log_dir = train_cfg.get('log_dir', 'train_logs')
    os.makedirs(log_dir, exist_ok=True)

    oracle_warmup_epochs = train_cfg.get('oracle_warmup_epochs', 16)
    oracle_rampup_epochs = train_cfg.get('oracle_rampup_epochs', 4)
    oracle_max_weight = float(train_cfg.get('oracle_max_weight', 0.5))
    recon_weight = float(train_cfg.get('recon_weight', 2.0))
    vq_weight = float(train_cfg.get('vq_weight', 1.0))
    oracle_bidirectional = train_cfg.get('oracle_bidirectional', False)

    # Comet
    comet_cfg = config.get('comet', {})
    experiment = None
    if comet_ml is not None and comet_cfg.get('api_key'):
        experiment = comet_ml.start(
            api_key=comet_cfg['api_key'],
            project_name=comet_cfg['project_name'],
            workspace=comet_cfg['workspace'])
        experiment.set_name(config.get('experiment_name', f"VQ_{seed}"))
        experiment.log_parameters(config)

    # Model
    print(f"[INFO] Building VQ model: {config['model']['name']}")
    model = build_model(config).to(device)

    # Oracle
    oracle_model = None
    oracle_cfg = config.get('oracle', None)
    if oracle_cfg and oracle_cfg.get('apply', False):
        print(f"[INFO] Loading Oracle: {oracle_cfg['config_path']}")
        oracle_config = load_config(oracle_cfg['config_path'])
        oracle_model = build_model(oracle_config).to(device)
        oracle_model.load_state_dict(
            torch.load(oracle_cfg['weights_path'], map_location=device, weights_only=True))
        oracle_model.eval()
        for p in oracle_model.parameters():
            p.requires_grad = False

    # Data
    train_loader = prepare_input(set_name='Train', config=config)
    val_loader = prepare_input(set_name='Val', config=config)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Scheduler
    scheduler_cfg = train_cfg.get('scheduler', {'apply': False})
    scheduler, warmup_epochs = None, 0
    if scheduler_cfg.get('apply', False) and scheduler_cfg.get('type') == 'cosine':
        eta_min = float(scheduler_cfg.get('eta_min', 1e-6))
        warmup_fraction = float(scheduler_cfg.get('warmup_fraction', 0.0))
        warmup_epochs = int(epochs * warmup_fraction)
        if warmup_epochs > 0:
            start_factor = eta_min / lr if lr > 0 else 1e-6
            ws = optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)
            cs = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=eta_min)
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, [ws, cs], milestones=[warmup_epochs])
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    best_val_pcc = -float('inf')
    epochs_no_improve = 0

    history = {'Epoch': [], 'Tr_Loss': [], 'Val_Loss': [],
               'Tr_Recon_Acc': [], 'Val_Recon_Acc': [],
               'Val_PCC_Dev': [], 'Val_PCC_Hk': [], 'Oracle_Weight': []}

    for epoch in range(epochs):
        t0 = time.time()

        # Oracle weight schedule
        if epoch < oracle_warmup_epochs:
            oracle_weight = 0.0
        else:
            ramp = min(1.0, (epoch - oracle_warmup_epochs) / float(max(1, oracle_rampup_epochs)))
            oracle_weight = oracle_max_weight * ramp

        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_recon_accs = []
        train_preds_dev, train_preds_hk = [], []
        train_targs_dev, train_targs_hk = [], []

        for X, Yd, Yh in train_loader:
            X, Yd, Yh = X.to(device), Yd.to(device), Yh.to(device)
            optimizer.zero_grad()

            loss, pd, ph, racc = compute_vq_loss(
                model, X, Yd, Yh, oracle_model,
                oracle_weight, recon_weight, vq_weight,
                oracle_bidirectional, criterion)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            if racc is not None:
                train_recon_accs.extend(racc.cpu().numpy().tolist())
            if pd is not None:
                train_preds_dev.extend(pd.detach().cpu().numpy().flatten())
                train_preds_hk.extend(ph.detach().cpu().numpy().flatten())
            train_targs_dev.extend(Yd.cpu().numpy().flatten())
            train_targs_hk.extend(Yh.cpu().numpy().flatten())

        avg_tr_loss = train_loss / len(train_loader.dataset)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_recon_accs = []
        val_preds_dev, val_preds_hk = [], []
        val_targs_dev, val_targs_hk = [], []

        with torch.no_grad():
            for X, Yd, Yh in val_loader:
                X, Yd, Yh = X.to(device), Yd.to(device), Yh.to(device)

                loss, pd, ph, racc = compute_vq_loss(
                    model, X, Yd, Yh, oracle_model,
                    oracle_weight, recon_weight, vq_weight,
                    oracle_bidirectional, criterion)

                val_loss += loss.item() * X.size(0)
                if racc is not None:
                    val_recon_accs.extend(racc.cpu().numpy().tolist())
                if pd is not None:
                    val_preds_dev.extend(pd.cpu().numpy().flatten())
                    val_preds_hk.extend(ph.cpu().numpy().flatten())
                val_targs_dev.extend(Yd.cpu().numpy().flatten())
                val_targs_hk.extend(Yh.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_loader.dataset)

        val_pcc_dev, val_pcc_hk = 0.0, 0.0
        if len(val_preds_dev) > 0:
            _, val_pcc_dev, _ = calculate_metrics(val_targs_dev, val_preds_dev)
            _, val_pcc_hk, _ = calculate_metrics(val_targs_hk, val_preds_hk)

        if scheduler:
            scheduler.step()

        # Log
        tr_recon = np.mean(train_recon_accs) if train_recon_accs else 0.0
        vl_recon = np.mean(val_recon_accs) if val_recon_accs else 0.0
        history['Epoch'].append(epoch + 1)
        history['Tr_Loss'].append(avg_tr_loss)
        history['Val_Loss'].append(avg_val_loss)
        history['Tr_Recon_Acc'].append(tr_recon)
        history['Val_Recon_Acc'].append(vl_recon)
        history['Val_PCC_Dev'].append(val_pcc_dev)
        history['Val_PCC_Hk'].append(val_pcc_hk)
        history['Oracle_Weight'].append(oracle_weight)

        recon_msg = f" | Recon: Tr={tr_recon:.1f}% Val={vl_recon:.1f}%" if val_recon_accs else ""
        pcc_msg = f" | PCC Dev={val_pcc_dev:.3f}" if val_preds_dev else ""
        print(f"Epoch {epoch+1:03d}/{epochs} | OW={oracle_weight:.3f} | "
              f"Loss: Tr={avg_tr_loss:.4f} Val={avg_val_loss:.4f}{recon_msg}{pcc_msg}")

        if experiment:
            experiment.log_metrics({
                "Loss/Train": avg_tr_loss, "Loss/Val": avg_val_loss,
                "Recon_Acc/Train": tr_recon, "Recon_Acc/Val": vl_recon,
                "PCC_Dev/Val": val_pcc_dev, "PCC_Hk/Val": val_pcc_hk,
                "Oracle_Weight": oracle_weight,
            }, step=epoch + 1)

        # Early stopping on PCC or recon
        metric = val_pcc_dev if val_preds_dev else vl_recon
        if epoch >= oracle_warmup_epochs:
            if metric > best_val_pcc:
                best_val_pcc = metric
                epochs_no_improve = 0
                path = os.path.join(log_dir, f"{config['experiment_name']}_seed{seed}.pth")
                torch.save(model.state_dict(), path)
                if experiment:
                    experiment.log_model(config['experiment_name'], path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"[INFO] Early stopping at epoch {epoch+1}.")
                    break

    plot_vq_history(history, log_dir, seed)
    if experiment:
        experiment.end()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train VQ-VAE / PixelCNN Model")
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train_vq(config)
