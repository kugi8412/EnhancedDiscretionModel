#!/usr/bin/env python
"""
Fine-tuning script for pretrained models on new/mutated data.

Supports:
  - Loading any registered model with pretrained weights
  - Optionally freezing early layers (partial fine-tuning)
  - Lower learning rate with warmup
  - Fine-tuning on custom data (mutated sequences, new conditions, etc.)
  - Reporting train/val/test metrics

Usage:
    python finetune.py -c ../../config/Finetune.yaml
    python finetune.py -c ../../config/Finetune.yaml --freeze-layers 3
    python finetune.py -c ../../config/Finetune.yaml --freeze-prefix "stem,block"
"""

import os
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr, pearsonr

from utils import prepare_input, load_config, set_global_seed
from models.registry import build_model


def calculate_metrics(targets, preds):
    if np.std(preds) == 0 or np.std(targets) == 0:
        return 0.0, 0.0, 0.0
    mse = float(F.mse_loss(
        torch.tensor(preds, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32)).item())
    pcc = pearsonr(targets, preds)[0]
    scc = spearmanr(targets, preds)[0]
    return mse, pcc, scc


def freeze_by_layer_count(model, n_layers):
    """Freeze the first n_layers parameter groups (by named_parameters order)."""
    # Group by top-level module
    modules_seen = []
    for name, _ in model.named_parameters():
        top_module = name.split('.')[0]
        if top_module not in modules_seen:
            modules_seen.append(top_module)

    freeze_modules = set(modules_seen[:n_layers])
    frozen_count = 0
    for name, param in model.named_parameters():
        top_module = name.split('.')[0]
        if top_module in freeze_modules:
            param.requires_grad = False
            frozen_count += 1

    print(f"[FINETUNE] Frozen {frozen_count} parameters in modules: {list(freeze_modules)}")


def freeze_by_prefix(model, prefixes):
    """Freeze parameters whose names start with any of the given prefixes."""
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad = False
            frozen_count += 1
    print(f"[FINETUNE] Frozen {frozen_count} parameters matching prefixes: {prefixes}")


def evaluate_model(model, loader, criterion, device, model_type='regression'):
    """Evaluate model on a data loader.

    Parameters
    ----------
    model_type : str
        'regression' — model outputs (pred_dev, pred_hk)
        'nucleotide' — model outputs logits (B, 4, L), loss = cross-entropy
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    if model_type == 'regression':
        preds_dev, preds_hk = [], []
        targs_dev, targs_hk = [], []

        with torch.no_grad():
            for batch in loader:
                X = batch[0].to(device)
                Y_dev = batch[1].to(device)
                Y_hk = batch[2].to(device)

                pred_dev, pred_hk = model(X)
                loss = criterion(pred_dev.squeeze(), Y_dev) + \
                       criterion(pred_hk.squeeze(), Y_hk)
                total_loss += loss.item() * X.size(0)
                n_samples += X.size(0)

                preds_dev.extend(pred_dev.cpu().numpy().flatten())
                preds_hk.extend(pred_hk.cpu().numpy().flatten())
                targs_dev.extend(Y_dev.cpu().numpy().flatten())
                targs_hk.extend(Y_hk.cpu().numpy().flatten())

        avg_loss = total_loss / n_samples
        mse_dev, pcc_dev, scc_dev = calculate_metrics(
            np.array(targs_dev), np.array(preds_dev))
        mse_hk, pcc_hk, scc_hk = calculate_metrics(
            np.array(targs_hk), np.array(preds_hk))
        return {
            'loss': avg_loss,
            'Dev': {'MSE': mse_dev, 'PCC': pcc_dev, 'SCC': scc_dev},
            'Hk': {'MSE': mse_hk, 'PCC': pcc_hk, 'SCC': scc_hk},
        }

    elif model_type == 'nucleotide':
        with torch.no_grad():
            for batch in loader:
                X = batch[0].to(device)
                logits = model(X)
                targets = X.argmax(dim=1)
                loss = F.cross_entropy(logits, targets)
                total_loss += loss.item() * X.size(0)
                n_samples += X.size(0)
        return {'loss': total_loss / n_samples}


def finetune(config):
    """Main fine-tuning routine."""
    seed = config.get('seed', 42)
    set_global_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    finetune_cfg = config.get('finetuning', config.get('training', {}))
    epochs = finetune_cfg.get('epochs', 50)
    lr = float(finetune_cfg.get('lr', 1e-5))
    weight_decay = float(finetune_cfg.get('weight_decay', 1e-4))
    early_stop_patience = finetune_cfg.get('early_stop', 15)
    log_dir = finetune_cfg.get('log_dir', 'finetune_logs')
    os.makedirs(log_dir, exist_ok=True)

    pretrained_weights = finetune_cfg.get('pretrained_weights', None)
    freeze_layers = finetune_cfg.get('freeze_layers', 0)
    freeze_prefix = finetune_cfg.get('freeze_prefix', [])
    model_type = finetune_cfg.get('model_type', 'regression')

    print(f"[FINETUNE] Device: {device}")
    print(f"[FINETUNE] Model type: {model_type}")
    print(f"[FINETUNE] LR: {lr}, Epochs: {epochs}")

    # Build model
    model = build_model(config).to(device)

    # Load pretrained weights
    if pretrained_weights and os.path.exists(pretrained_weights):
        state = torch.load(pretrained_weights, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"[FINETUNE] Loaded pretrained weights: {pretrained_weights}")
    else:
        print("[FINETUNE] WARNING: No pretrained weights loaded!")

    # Freeze layers
    if freeze_layers > 0:
        freeze_by_layer_count(model, freeze_layers)
    if freeze_prefix:
        freeze_by_prefix(model, freeze_prefix)

    # Count trainable parameters
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[FINETUNE] Trainable params: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.1f}%)")

    # Data
    train_loader = prepare_input(set_name='Train', config=config)
    val_loader = prepare_input(set_name='Val', config=config)
    test_loader = prepare_input(set_name='Test', config=config)

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # Warmup + cosine schedule
    warmup_epochs = max(1, int(epochs * 0.1))
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

    criterion = nn.MSELoss()

    best_val_metric = float('inf') if model_type == 'nucleotide' else -float('inf')
    no_improve = 0
    model_path = os.path.join(log_dir, f"{config['experiment_name']}_finetuned_seed{seed}.pth")

    log_file = os.path.join(log_dir, f'finetune_log_{seed}.csv')
    if model_type == 'regression':
        headers = ['Epoch', 'LR', 'Train_Loss', 'Val_Loss',
                   'Val_PCC_Dev', 'Val_SCC_Dev', 'Val_PCC_Hk', 'Val_SCC_Hk']
    else:
        headers = ['Epoch', 'LR', 'Train_CE', 'Val_CE']
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(headers)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            X = batch[0].to(device)
            optimizer.zero_grad()

            if model_type == 'regression':
                Y_dev = batch[1].to(device)
                Y_hk = batch[2].to(device)
                pred_dev, pred_hk = model(X)
                loss = criterion(pred_dev.squeeze(), Y_dev) + \
                       criterion(pred_hk.squeeze(), Y_hk)
            else:
                logits = model(X)
                targets = X.argmax(dim=1)
                loss = F.cross_entropy(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            n_train += X.size(0)

        avg_train_loss = train_loss / n_train
        scheduler.step()

        # Validation
        val_result = evaluate_model(model, val_loader, criterion, device, model_type)
        current_lr = optimizer.param_groups[0]['lr']

        if model_type == 'regression':
            val_metric = (val_result['Dev']['PCC'] + val_result['Hk']['PCC']) / 2
            improved = val_metric > best_val_metric
            row = [epoch+1, current_lr, avg_train_loss, val_result['loss'],
                   val_result['Dev']['PCC'], val_result['Dev']['SCC'],
                   val_result['Hk']['PCC'], val_result['Hk']['SCC']]
            print(f"Epoch {epoch+1:03d}/{epochs} | "
                  f"Loss: Tr={avg_train_loss:.4f} Val={val_result['loss']:.4f} | "
                  f"PCC: Dev={val_result['Dev']['PCC']:.4f} Hk={val_result['Hk']['PCC']:.4f}")
        else:
            val_metric = val_result['loss']
            improved = val_metric < best_val_metric
            row = [epoch+1, current_lr, avg_train_loss, val_result['loss']]
            print(f"Epoch {epoch+1:03d}/{epochs} | "
                  f"CE: Tr={avg_train_loss:.4f} Val={val_result['loss']:.4f}")

        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        if improved:
            best_val_metric = val_metric
            no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"[FINETUNE] Early stopping at epoch {epoch+1}")
                break

    # Load best and test
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test_result = evaluate_model(model, test_loader, criterion, device, model_type)

    print(f"\n[FINETUNE] Final Test Results:")
    if model_type == 'regression':
        print(f"  Dev — MSE: {test_result['Dev']['MSE']:.4f}, "
              f"PCC: {test_result['Dev']['PCC']:.4f}, "
              f"SCC: {test_result['Dev']['SCC']:.4f}")
        print(f"  Hk  — MSE: {test_result['Hk']['MSE']:.4f}, "
              f"PCC: {test_result['Hk']['PCC']:.4f}, "
              f"SCC: {test_result['Hk']['SCC']:.4f}")
    else:
        print(f"  CE Loss: {test_result['loss']:.4f}")

    print(f"[FINETUNE] Model saved to: {model_path}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune pretrained model")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument('--freeze-layers', type=int, default=None,
                        help="Number of top-level modules to freeze")
    parser.add_argument('--freeze-prefix', type=str, nargs='+', default=None,
                        help="Parameter name prefixes to freeze")
    args = parser.parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.freeze_layers is not None:
        config.setdefault('finetuning', {})['freeze_layers'] = args.freeze_layers
    if args.freeze_prefix is not None:
        config.setdefault('finetuning', {})['freeze_prefix'] = args.freeze_prefix

    finetune(config)
