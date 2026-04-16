#!/usr/bin/env python
import comet_ml

import os
import csv
import time
import math
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
    axes[0].set_title('Mean Squared Error (Loss)')
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
    print(f"[INFO] Saved training plot to {plot_path}")


def train_model(config):
    seed = config.get('seed', 42)
    set_global_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Initializing training on device: {device}")
    
    train_cfg = config.get('training', {})
    epochs = train_cfg.get('epochs', 100)
    lr = float(train_cfg.get('lr', 1e-4))
    weight_decay = float(train_cfg.get('weight_decay', 1e-4))
    early_stop_patience = train_cfg.get('early_stop', 15)
    log_dir = train_cfg.get('log_dir', 'train_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # --- POBIERANIE PARAMETRÓW STERUJĄCYCH ---
    oracle_warmup_epochs = train_cfg.get('oracle_warmup_epochs', 16)
    oracle_rampup_epochs = train_cfg.get('oracle_rampup_epochs', 4)
    oracle_max_weight    = float(train_cfg.get('oracle_max_weight', 0.5))
    recon_weight         = float(train_cfg.get('recon_weight', 2.0))
    vq_weight            = float(train_cfg.get('vq_weight', 1.0))
    oracle_bidirectional = train_cfg.get('oracle_bidirectional', False) 
    
    comet_cfg = config.get('comet', {})
    experiment = None
    if comet_cfg.get('api_key'):
        experiment = comet_ml.start(
            api_key=comet_cfg['api_key'], 
            project_name=comet_cfg['project_name'], 
            workspace=comet_cfg['workspace']
        )
        experiment.set_name(config.get('experiment_name', f"Experiment_{seed}"))
        experiment.log_parameters(config)

    print(f"[INFO] Building main model: {config['model']['name']}")
    model = build_model(config).to(device)

    oracle_cfg = config.get('oracle', None)
    oracle_model = None
    if oracle_cfg and oracle_cfg.get('apply', False):
        print(f"[INFO] Loading pre-trained Oracle from: {oracle_cfg['config_path']}")
        oracle_model_config = load_config(oracle_cfg['config_path'])
        oracle_model = build_model(oracle_model_config).to(device)
        oracle_model.load_state_dict(torch.load(oracle_cfg['weights_path'], map_location=device, weights_only=True))
        oracle_model.eval() 
        for param in oracle_model.parameters():
            param.requires_grad = False 
        print("[INFO] Oracle loaded and frozen successfully.")
    
    train_loader = prepare_input(set_name='Train', config=config)
    val_loader = prepare_input(set_name='Val', config=config)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # --- LR & WEIGHT DECAY SCHEDULER SETUP ---
    scheduler_cfg = train_cfg.get('scheduler', {'apply': False})
    scheduler = None
    warmup_epochs = 0
    
    wd_sched_apply = scheduler_cfg.get('weight_decay_scheduler', False)
    wd_min = float(scheduler_cfg.get('weight_decay_min', 1e-6))
    wd_max = weight_decay

    if scheduler_cfg.get('apply', False) and scheduler_cfg.get('type') == 'cosine':
        eta_min = float(scheduler_cfg.get('eta_min', 1e-6))
        warmup_fraction = float(scheduler_cfg.get('warmup_fraction', 0.0))
        warmup_epochs = int(epochs * warmup_fraction)
        
        if warmup_epochs > 0:
            start_factor = eta_min / lr if lr > 0 else 1e-6
            warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs - warmup_epochs), eta_min=eta_min)
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    best_val_pcc = -float('inf') 
    epochs_no_improve = 0
    log_file = os.path.join(log_dir, f'training_log_{seed}.csv')
    
    # Dodano kolumnę 'Weight_Decay' do CSV
    headers = [
        'Epoch', 'LR', 'Weight_Decay', 'Time(s)', 
        'Tr_Loss', 'Tr_MSE_Dev', 'Tr_PCC_Dev', 'Tr_SCC_Dev', 'Tr_MSE_Hk', 'Tr_PCC_Hk', 'Tr_SCC_Hk',
        'Val_Loss', 'Val_MSE_Dev', 'Val_PCC_Dev', 'Val_SCC_Dev', 'Val_MSE_Hk', 'Val_PCC_Hk', 'Val_SCC_Hk'
    ]
    with open(log_file, mode='w', newline='') as file:
        csv.writer(file).writerow(headers)
        
    history = {k: [] for k in headers}
    
    # ------------------ MAIN TRAINING LOOP ------------------
    for epoch in range(epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        # --- RĘCZNY UPDATE WEIGHT DECAY ---
        if wd_sched_apply:
            if epoch < warmup_epochs:
                # Liniowy warmup dla Weight Decay
                wd_fraction = (epoch + 1) / max(1, warmup_epochs)
                current_wd = wd_min + (wd_max - wd_min) * wd_fraction
            else:
                # Cosine Annealing dla Weight Decay
                cosine_epoch = epoch - warmup_epochs
                cosine_epochs_total = epochs - warmup_epochs
                current_wd = wd_min + 0.5 * (wd_max - wd_min) * (1 + math.cos(math.pi * cosine_epoch / max(1, cosine_epochs_total)))
                
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = current_wd
        else:
            current_wd = optimizer.param_groups[0]['weight_decay']

        
        if epoch < oracle_warmup_epochs:
            oracle_weight = 0.0
        else:
            ramp_factor = min(1.0, (epoch - oracle_warmup_epochs) / float(max(1, oracle_rampup_epochs)))
            oracle_weight = oracle_max_weight * ramp_factor
        
        model.train()
        train_loss = 0.0
        train_preds_dev, train_preds_hk = [], []
        train_targs_dev, train_targs_hk = [], []
        train_recon_accs = []
        
        for X_batch, Y_dev_batch, Y_hk_batch in train_loader:
            X_batch, Y_dev_batch, Y_hk_batch = X_batch.to(device), Y_dev_batch.to(device), Y_hk_batch.to(device)
            optimizer.zero_grad()
            
            try:
                outputs = model(X_batch, Y_dev_batch, Y_hk_batch)
            except TypeError:
                outputs = model(X_batch)
            
            if len(outputs) == 3 and oracle_model is not None:
                x_logits_8ch, gumbels, vq_loss = outputs
                x_gumbel_fwd, x_gumbel_rc = gumbels
                
                x_fwd_logits = x_logits_8ch[:, 0:4, :]
                x_rc_logits = x_logits_8ch[:, 4:8, :]
                
                true_seq_indices = torch.argmax(X_batch, dim=1)
                X_batch_rc = torch.flip(X_batch, dims=[1, 2])
                true_rc_indices = torch.argmax(X_batch_rc, dim=1)
                
                loss_recon_fwd = F.cross_entropy(x_fwd_logits, true_seq_indices)
                loss_recon_rc = F.cross_entropy(x_rc_logits, true_rc_indices)
                loss_recon = (loss_recon_fwd + loss_recon_rc) / 2.0
                
                pred_dev_fwd, pred_hk_fwd = oracle_model(x_gumbel_fwd)
                
                if oracle_bidirectional:
                    pred_dev_rc, pred_hk_rc = oracle_model(x_gumbel_rc)
                    loss_dev = (criterion(pred_dev_fwd.squeeze(), Y_dev_batch) + criterion(pred_dev_rc.squeeze(), Y_dev_batch)) / 2.0
                    loss_hk = (criterion(pred_hk_fwd.squeeze(), Y_hk_batch) + criterion(pred_hk_rc.squeeze(), Y_hk_batch)) / 2.0
                    pred_dev = (pred_dev_fwd + pred_dev_rc) / 2.0
                    pred_hk = (pred_hk_fwd + pred_hk_rc) / 2.0
                else:
                    loss_dev = criterion(pred_dev_fwd.squeeze(), Y_dev_batch)
                    loss_hk = criterion(pred_hk_fwd.squeeze(), Y_hk_batch)
                    pred_dev = pred_dev_fwd
                    pred_hk = pred_hk_fwd
                
                loss = (oracle_weight * (loss_dev + loss_hk)) + (recon_weight * loss_recon) + (vq_weight * vq_loss)
                
                with torch.no_grad():
                    preds_seq = torch.argmax(x_fwd_logits, dim=1)
                    matches = (preds_seq == true_seq_indices).float()
                    train_recon_accs.extend((matches.mean(dim=1) * 100.0).cpu().numpy().tolist())

            elif len(outputs) == 4:
                pred_dev, pred_hk, x_recon, vq_loss = outputs
                loss_dev = criterion(pred_dev.squeeze(), Y_dev_batch)
                loss_hk = criterion(pred_hk.squeeze(), Y_hk_batch)
                true_seq_indices = torch.argmax(X_batch, dim=1)
                loss_recon = F.cross_entropy(x_recon, true_seq_indices)
                
                loss = (oracle_weight * (loss_dev + loss_hk)) + (recon_weight * loss_recon) + (vq_weight * vq_loss)
                
                with torch.no_grad():
                    preds_seq = torch.argmax(x_recon, dim=1)
                    matches = (preds_seq == true_seq_indices).float()
                    train_recon_accs.extend((matches.mean(dim=1) * 100.0).cpu().numpy().tolist())
            
            else:
                pred_dev, pred_hk = outputs
                loss = criterion(pred_dev.squeeze(), Y_dev_batch) + criterion(pred_hk.squeeze(), Y_hk_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
            if len(outputs) == 3 and oracle_model is not None:
                train_preds_dev.extend(pred_dev.detach().cpu().numpy().flatten())
                train_preds_hk.extend(pred_hk.detach().cpu().numpy().flatten())
            elif len(outputs) == 4 or len(outputs) == 2:
                train_preds_dev.extend(pred_dev.detach().cpu().numpy().flatten())
                train_preds_hk.extend(pred_hk.detach().cpu().numpy().flatten())
            
            train_targs_dev.extend(Y_dev_batch.cpu().numpy().flatten())
            train_targs_hk.extend(Y_hk_batch.cpu().numpy().flatten())
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        tr_mse_dev, tr_pcc_dev, tr_scc_dev = calculate_metrics(train_targs_dev, train_preds_dev)
        tr_mse_hk, tr_pcc_hk, tr_scc_hk = calculate_metrics(train_targs_hk, train_preds_hk)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_preds_dev, val_preds_hk = [], []
        val_targs_dev, val_targs_hk = [], []
        val_recon_accs = []
        
        with torch.no_grad():
            for X_batch, Y_dev_batch, Y_hk_batch in val_loader:
                X_batch, Y_dev_batch, Y_hk_batch = X_batch.to(device), Y_dev_batch.to(device), Y_hk_batch.to(device)
                
                try:
                    outputs = model(X_batch, Y_dev_batch, Y_hk_batch)
                except TypeError:
                    outputs = model(X_batch)
                
                if len(outputs) == 3 and oracle_model is not None:
                    x_logits_8ch, gumbels, vq_loss = outputs
                    x_gumbel_fwd, x_gumbel_rc = gumbels
                    
                    x_fwd_logits = x_logits_8ch[:, 0:4, :]
                    x_rc_logits = x_logits_8ch[:, 4:8, :]
                    
                    true_seq_indices = torch.argmax(X_batch, dim=1)
                    X_batch_rc = torch.flip(X_batch, dims=[1, 2])
                    true_rc_indices = torch.argmax(X_batch_rc, dim=1)
                    
                    loss_recon_fwd = F.cross_entropy(x_fwd_logits, true_seq_indices)
                    loss_recon_rc = F.cross_entropy(x_rc_logits, true_rc_indices)
                    loss_recon = (loss_recon_fwd + loss_recon_rc) / 2.0
                    
                    pred_dev_fwd, pred_hk_fwd = oracle_model(x_gumbel_fwd)
                    
                    if oracle_bidirectional:
                        pred_dev_rc, pred_hk_rc = oracle_model(x_gumbel_rc)
                        loss_dev = (criterion(pred_dev_fwd.squeeze(), Y_dev_batch) + criterion(pred_dev_rc.squeeze(), Y_dev_batch)) / 2.0
                        loss_hk = (criterion(pred_hk_fwd.squeeze(), Y_hk_batch) + criterion(pred_hk_rc.squeeze(), Y_hk_batch)) / 2.0
                        pred_dev = (pred_dev_fwd + pred_dev_rc) / 2.0
                        pred_hk = (pred_hk_fwd + pred_hk_rc) / 2.0
                    else:
                        loss_dev = criterion(pred_dev_fwd.squeeze(), Y_dev_batch)
                        loss_hk = criterion(pred_hk_fwd.squeeze(), Y_hk_batch)
                        pred_dev = pred_dev_fwd
                        pred_hk = pred_hk_fwd
                    
                    loss = (oracle_weight * (loss_dev + loss_hk)) + (recon_weight * loss_recon) + (vq_weight * vq_loss)
                    
                    preds_seq = torch.argmax(x_fwd_logits, dim=1)
                    matches = (preds_seq == true_seq_indices).float()
                    val_recon_accs.extend((matches.mean(dim=1) * 100.0).cpu().numpy().tolist())
                    
                    val_preds_dev.extend(pred_dev.cpu().numpy().flatten())
                    val_preds_hk.extend(pred_hk.cpu().numpy().flatten())

                elif len(outputs) == 4:
                    pred_dev, pred_hk, x_recon, vq_loss = outputs
                    loss_dev = criterion(pred_dev.squeeze(), Y_dev_batch)
                    loss_hk = criterion(pred_hk.squeeze(), Y_hk_batch)
                    true_seq_indices = torch.argmax(X_batch, dim=1)
                    loss_recon = F.cross_entropy(x_recon, true_seq_indices)
                    
                    loss = (oracle_weight * (loss_dev + loss_hk)) + (recon_weight * loss_recon) + (vq_weight * vq_loss)
                    
                    preds_seq = torch.argmax(x_recon, dim=1)
                    matches = (preds_seq == true_seq_indices).float()
                    val_recon_accs.extend((matches.mean(dim=1) * 100.0).cpu().numpy().tolist())
                    
                    val_preds_dev.extend(pred_dev.cpu().numpy().flatten())
                    val_preds_hk.extend(pred_hk.cpu().numpy().flatten())
                else:
                    pred_dev, pred_hk = outputs
                    loss = criterion(pred_dev.squeeze(), Y_dev_batch) + criterion(pred_hk.squeeze(), Y_hk_batch)
                    val_preds_dev.extend(pred_dev.cpu().numpy().flatten())
                    val_preds_hk.extend(pred_hk.cpu().numpy().flatten())

                val_loss += loss.item() * X_batch.size(0)
                val_targs_dev.extend(Y_dev_batch.cpu().numpy().flatten())
                val_targs_hk.extend(Y_hk_batch.cpu().numpy().flatten())
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_mse_dev, val_pcc_dev, val_scc_dev = calculate_metrics(val_targs_dev, val_preds_dev)
        val_mse_hk, val_pcc_hk, val_scc_hk = calculate_metrics(val_targs_hk, val_preds_hk)

        epoch_time = time.time() - epoch_start_time

        if scheduler:
            scheduler.step()

        # Dodano current_wd do row_data
        row_data = [
            epoch + 1, current_lr, current_wd, epoch_time, 
            avg_train_loss, tr_mse_dev, tr_pcc_dev, tr_scc_dev, tr_mse_hk, tr_pcc_hk, tr_scc_hk,
            avg_val_loss, val_mse_dev, val_pcc_dev, val_scc_dev, val_mse_hk, val_pcc_hk, val_scc_hk
        ]
        
        with open(log_file, mode='a', newline='') as file:
            csv.writer(file).writerow(row_data)
            
        for key, val in zip(headers, row_data):
            history[key].append(val)

        recon_msg = ""
        if len(val_recon_accs) > 0:
            v_avg, v_min, v_max = np.mean(val_recon_accs), np.min(val_recon_accs), np.max(val_recon_accs)
            recon_msg = f" | Recon Val: Avg={v_avg:.1f}% (Min={v_min:.1f}%, Max={v_max:.1f}%)"
            
            if experiment:
                experiment.log_metrics({
                    "Recon_Acc_Avg/Train": np.mean(train_recon_accs),
                    "Recon_Acc_Avg/Val": v_avg,
                    "Recon_Acc_Min/Val": v_min,
                    "Recon_Acc_Max/Val": v_max
                }, step=epoch + 1)

        if experiment:
            experiment.log_metrics({
                "Loss/Train": avg_train_loss, "Loss/Val": avg_val_loss,
                "PCC_Dev/Train": tr_pcc_dev, "PCC_Dev/Val": val_pcc_dev,
                "SCC_Dev/Train": tr_scc_dev, "SCC_Dev/Val": val_scc_dev,
                "PCC_Hk/Train": tr_pcc_hk, "PCC_Hk/Val": val_pcc_hk,
                "SCC_Hk/Train": tr_scc_hk, "SCC_Hk/Val": val_scc_hk,
                "Learning_Rate": current_lr,
                "Weight_Decay": current_wd, # LOGOWANIE DO COMETA
                "Oracle_Weight": oracle_weight,
                "Epoch_Duration_sec": epoch_time
            }, step=epoch + 1)

        print(f"Epoch {epoch+1:03d}/{epochs} [LR: {current_lr:.2e} | WD: {current_wd:.2e} | OW: {oracle_weight:.3f}] | "
              f"Loss: Tr={avg_train_loss:.4f} Val={avg_val_loss:.4f} | "
              f"Dev PCC: Tr={tr_pcc_dev:.3f} Val={val_pcc_dev:.3f}{recon_msg}")

        if epoch >= oracle_warmup_epochs:
            if val_pcc_dev > best_val_pcc:
                best_val_pcc = val_pcc_dev
                epochs_no_improve = 0
                model_path = os.path.join(log_dir, f"{config['experiment_name']}_seed{seed}.pth")
                torch.save(model.state_dict(), model_path)
                if experiment: 
                    experiment.log_model(config['experiment_name'],  model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"[INFO] Early stopping triggered after {epochs_no_improve} epochs without PCC improvement.")
                    break
                
    plot_training_history(history, log_dir, seed)
    if experiment:
        experiment.end()
        
    return model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Universal Training Script")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main_config = load_config(args.config)
    trained_model = train_model(main_config)
