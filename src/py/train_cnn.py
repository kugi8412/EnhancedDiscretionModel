#!/usr/bin/env python
"""
Training script for CNN expression prediction models.
Supports: DeepSTARR, LegNet, LegNetV2, ConvNeXt_DNA, SEResNet, BassetNetwork,
           CustomNetwork, ReverseNet_SuperKernel, LegNetPlus.

Usage:
    python train_cnn.py -c ../../config/LegNetPlus.yaml
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
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    preds_tensor = torch.tensor(preds, dtype=torch.float32)
    mse = F.mse_loss(preds_tensor, targets_tensor).item()
    if np.std(preds) == 0 or np.std(targets) == 0:
        pcc, scc = 0.0, 0.0
    else:
        pcc = pearsonr(targets, preds)[0]
        scc = spearmanr(targets, preds)[0]
    return mse, pcc, scc


def plot_training_history(history, log_dir, seed):
    epochs = history['Epoch']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history['Tr_Loss'], label='Train Loss')
    axes[0].plot(epochs, history['Val_Loss'], label='Val Loss')
    axes[0].set_title('Loss (MSE)')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history['Tr_PCC_Dev'], label='Train Dev', linestyle='--')
    axes[1].plot(epochs, history['Val_PCC_Dev'], label='Val Dev')
    axes[1].plot(epochs, history['Tr_PCC_Hk'], label='Train Hk', linestyle='--')
    axes[1].plot(epochs, history['Val_PCC_Hk'], label='Val Hk')
    axes[1].set_title('Pearson Correlation (PCC)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, history['Tr_SCC_Dev'], label='Train Dev', linestyle='--')
    axes[2].plot(epochs, history['Val_SCC_Dev'], label='Val Dev')
    axes[2].plot(epochs, history['Tr_SCC_Hk'], label='Train Hk', linestyle='--')
    axes[2].plot(epochs, history['Val_SCC_Hk'], label='Val Hk')
    axes[2].set_title('Spearman Correlation (SCC)')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(log_dir, f'training_plot_{seed}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Saved training plot to {plot_path}")


def build_scheduler(optimizer, train_cfg, epochs, lr):
    scheduler_cfg = train_cfg.get('scheduler', {'apply': False})
    warmup_epochs = 0

    if not (scheduler_cfg.get('apply', False) and scheduler_cfg.get('type') == 'cosine'):
        return None, 0

    eta_min = float(scheduler_cfg.get('eta_min', 1e-6))
    warmup_fraction = float(scheduler_cfg.get('warmup_fraction', 0.0))
    warmup_epochs = int(epochs * warmup_fraction)

    if warmup_epochs > 0:
        start_factor = eta_min / lr if lr > 0 else 1e-6
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, total_iters=warmup_epochs)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(epochs - warmup_epochs), eta_min=eta_min)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min)

    return scheduler, warmup_epochs


def update_weight_decay(optimizer, epoch, warmup_epochs, epochs, wd_min, wd_max, apply):
    if not apply:
        return optimizer.param_groups[0]['weight_decay']
    if epoch < warmup_epochs:
        wd_fraction = (epoch + 1) / max(1, warmup_epochs)
        current_wd = wd_min + (wd_max - wd_min) * wd_fraction
    else:
        cosine_epoch = epoch - warmup_epochs
        cosine_epochs_total = epochs - warmup_epochs
        current_wd = wd_min + 0.5 * (wd_max - wd_min) * (
            1 + math.cos(math.pi * cosine_epoch / max(1, cosine_epochs_total)))
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = current_wd
    return current_wd


def setup_comet(config):
    comet_cfg = config.get('comet', {})
    if comet_ml is None or not comet_cfg.get('api_key'):
        return None
    experiment = comet_ml.start(
        api_key=comet_cfg['api_key'],
        project_name=comet_cfg['project_name'],
        workspace=comet_cfg['workspace']
    )
    seed = config.get('seed', 42)
    experiment.set_name(config.get('experiment_name', f"Experiment_{seed}"))
    experiment.log_parameters(config)
    return experiment


def train_cnn(config):
    seed = config.get('seed', 42)
    set_global_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Training CNN on device: {device}")

    train_cfg = config.get('training', {})
    epochs = train_cfg.get('epochs', 100)
    lr = float(train_cfg.get('lr', 1e-4))
    weight_decay = float(train_cfg.get('weight_decay', 1e-4))
    early_stop_patience = train_cfg.get('early_stop', 15)
    log_dir = train_cfg.get('log_dir', 'train_logs')
    os.makedirs(log_dir, exist_ok=True)

    experiment = setup_comet(config)

    print(f"[INFO] Building model: {config['model']['name']}")
    model = build_model(config).to(device)

    # Detect single-output ablation mode
    output_head = config['model'].get('output_head', 'both')
    single_output = (output_head != 'both')
    if single_output:
        print(f"[INFO] Single-output ablation mode: '{output_head}' only")

    train_loader = prepare_input(set_name='Train', config=config)
    val_loader = prepare_input(set_name='Val', config=config)

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                            weight_decay=weight_decay)
    criterion = nn.MSELoss()

    scheduler_cfg = train_cfg.get('scheduler', {'apply': False})
    scheduler, warmup_epochs = build_scheduler(optimizer, train_cfg, epochs, lr)

    wd_sched_apply = scheduler_cfg.get('weight_decay_scheduler', False)
    wd_min = float(scheduler_cfg.get('weight_decay_min', 1e-6))
    wd_max = weight_decay

    best_val_pcc = -float('inf')
    epochs_no_improve = 0
    log_file = os.path.join(log_dir, f'training_log_{seed}.csv')

    headers = [
        'Epoch', 'LR', 'Weight_Decay', 'Time(s)',
        'Tr_Loss', 'Tr_MSE_Dev', 'Tr_PCC_Dev', 'Tr_SCC_Dev',
        'Tr_MSE_Hk', 'Tr_PCC_Hk', 'Tr_SCC_Hk',
        'Val_Loss', 'Val_MSE_Dev', 'Val_PCC_Dev', 'Val_SCC_Dev',
        'Val_MSE_Hk', 'Val_PCC_Hk', 'Val_SCC_Hk'
    ]
    with open(log_file, mode='w', newline='') as file:
        csv.writer(file).writerow(headers)
    history = {k: [] for k in headers}

    for epoch in range(epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        current_wd = update_weight_decay(
            optimizer, epoch, warmup_epochs, epochs, wd_min, wd_max, wd_sched_apply)

        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_preds_dev, train_preds_hk = [], []
        train_targs_dev, train_targs_hk = [], []

        for X_batch, Y_dev_batch, Y_hk_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_dev_batch = Y_dev_batch.to(device)
            Y_hk_batch = Y_hk_batch.to(device)
            optimizer.zero_grad()

            if single_output:
                pred = model(X_batch)
                target = Y_dev_batch if output_head == 'dev' else Y_hk_batch
                loss = criterion(pred.squeeze(), target)
                train_loss += loss.item() * X_batch.size(0)
                if output_head == 'dev':
                    train_preds_dev.extend(pred.detach().cpu().numpy().flatten())
                    train_targs_dev.extend(Y_dev_batch.cpu().numpy().flatten())
                else:
                    train_preds_hk.extend(pred.detach().cpu().numpy().flatten())
                    train_targs_hk.extend(Y_hk_batch.cpu().numpy().flatten())
            else:
                pred_dev, pred_hk = model(X_batch)
                loss = criterion(pred_dev.squeeze(), Y_dev_batch) + \
                       criterion(pred_hk.squeeze(), Y_hk_batch)
                train_loss += loss.item() * X_batch.size(0)
                train_preds_dev.extend(pred_dev.detach().cpu().numpy().flatten())
                train_preds_hk.extend(pred_hk.detach().cpu().numpy().flatten())
                train_targs_dev.extend(Y_dev_batch.cpu().numpy().flatten())
                train_targs_hk.extend(Y_hk_batch.cpu().numpy().flatten())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader.dataset)
        if single_output:
            if output_head == 'dev':
                tr_mse_dev, tr_pcc_dev, tr_scc_dev = calculate_metrics(train_targs_dev, train_preds_dev)
                tr_mse_hk, tr_pcc_hk, tr_scc_hk = 0.0, 0.0, 0.0
            else:
                tr_mse_dev, tr_pcc_dev, tr_scc_dev = 0.0, 0.0, 0.0
                tr_mse_hk, tr_pcc_hk, tr_scc_hk = calculate_metrics(train_targs_hk, train_preds_hk)
        else:
            tr_mse_dev, tr_pcc_dev, tr_scc_dev = calculate_metrics(train_targs_dev, train_preds_dev)
            tr_mse_hk, tr_pcc_hk, tr_scc_hk = calculate_metrics(train_targs_hk, train_preds_hk)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_preds_dev, val_preds_hk = [], []
        val_targs_dev, val_targs_hk = [], []

        with torch.no_grad():
            for X_batch, Y_dev_batch, Y_hk_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_dev_batch = Y_dev_batch.to(device)
                Y_hk_batch = Y_hk_batch.to(device)

                if single_output:
                    pred = model(X_batch)
                    target = Y_dev_batch if output_head == 'dev' else Y_hk_batch
                    loss = criterion(pred.squeeze(), target)
                    val_loss += loss.item() * X_batch.size(0)
                    if output_head == 'dev':
                        val_preds_dev.extend(pred.cpu().numpy().flatten())
                        val_targs_dev.extend(Y_dev_batch.cpu().numpy().flatten())
                    else:
                        val_preds_hk.extend(pred.cpu().numpy().flatten())
                        val_targs_hk.extend(Y_hk_batch.cpu().numpy().flatten())
                else:
                    pred_dev, pred_hk = model(X_batch)
                    loss = criterion(pred_dev.squeeze(), Y_dev_batch) + \
                           criterion(pred_hk.squeeze(), Y_hk_batch)
                    val_loss += loss.item() * X_batch.size(0)
                    val_preds_dev.extend(pred_dev.cpu().numpy().flatten())
                    val_preds_hk.extend(pred_hk.cpu().numpy().flatten())
                    val_targs_dev.extend(Y_dev_batch.cpu().numpy().flatten())
                    val_targs_hk.extend(Y_hk_batch.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_loader.dataset)
        if single_output:
            if output_head == 'dev':
                val_mse_dev, val_pcc_dev, val_scc_dev = calculate_metrics(val_targs_dev, val_preds_dev)
                val_mse_hk, val_pcc_hk, val_scc_hk = 0.0, 0.0, 0.0
            else:
                val_mse_dev, val_pcc_dev, val_scc_dev = 0.0, 0.0, 0.0
                val_mse_hk, val_pcc_hk, val_scc_hk = calculate_metrics(val_targs_hk, val_preds_hk)
        else:
            val_mse_dev, val_pcc_dev, val_scc_dev = calculate_metrics(val_targs_dev, val_preds_dev)
            val_mse_hk, val_pcc_hk, val_scc_hk = calculate_metrics(val_targs_hk, val_preds_hk)

        epoch_time = time.time() - epoch_start_time
        if scheduler:
            scheduler.step()

        row_data = [
            epoch + 1, current_lr, current_wd, epoch_time,
            avg_train_loss, tr_mse_dev, tr_pcc_dev, tr_scc_dev,
            tr_mse_hk, tr_pcc_hk, tr_scc_hk,
            avg_val_loss, val_mse_dev, val_pcc_dev, val_scc_dev,
            val_mse_hk, val_pcc_hk, val_scc_hk
        ]
        with open(log_file, mode='a', newline='') as file:
            csv.writer(file).writerow(row_data)
        for key, val in zip(headers, row_data):
            history[key].append(val)

        if experiment:
            experiment.log_metrics({
                "Loss/Train": avg_train_loss, "Loss/Val": avg_val_loss,
                "PCC_Dev/Train": tr_pcc_dev, "PCC_Dev/Val": val_pcc_dev,
                "PCC_Hk/Train": tr_pcc_hk, "PCC_Hk/Val": val_pcc_hk,
                "SCC_Dev/Val": val_scc_dev, "SCC_Hk/Val": val_scc_hk,
                "Learning_Rate": current_lr, "Weight_Decay": current_wd,
            }, step=epoch + 1)

        print(f"Epoch {epoch+1:03d}/{epochs} "
              f"[LR: {current_lr:.2e} | WD: {current_wd:.2e}] | "
              f"Loss: Tr={avg_train_loss:.4f} Val={avg_val_loss:.4f} | "
              f"Dev PCC: Tr={tr_pcc_dev:.3f} Val={val_pcc_dev:.3f}"
              + (f" | Hk PCC: Val={val_pcc_hk:.3f}" if single_output and output_head == 'hk' else ""))

        # Early stopping metric: use the active head's PCC
        if single_output and output_head == 'hk':
            val_pcc_metric = val_pcc_hk
        else:
            val_pcc_metric = val_pcc_dev

        if val_pcc_metric > best_val_pcc:
            best_val_pcc = val_pcc_metric
            epochs_no_improve = 0
            model_path = os.path.join(
                log_dir, f"{config['experiment_name']}_seed{seed}.pth")
            torch.save(model.state_dict(), model_path)
            if experiment:
                experiment.log_model(config['experiment_name'], model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"[INFO] Early stopping after {epochs_no_improve} "
                      f"epochs without PCC improvement.")
                break

    plot_training_history(history, log_dir, seed)
    if experiment:
        experiment.end()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN Expression Model")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()
    config = load_config(args.config)
    train_cnn(config)
