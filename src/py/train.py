#!/usr/bin/env 
import comet_ml

import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# Założenie: Posiadasz te funkcje w utils.py i models/registry.py
from utils import prepare_input, load_config, set_global_seed
from models.registry import build_model


def calculate_metrics(targets, preds):
    """
    Calculates evaluation metrics for the entire dataset epoch.
    """
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    preds_tensor = torch.tensor(preds, dtype=torch.float32)
    mse = F.mse_loss(preds_tensor, targets_tensor).item()
    
    # Protect against constant predictions (zero variance) in early epochs
    if np.std(preds) == 0 or np.std(targets) == 0:
        pcc, scc = 0.0, 0.0
    else:
        pcc = pearsonr(targets, preds)[0]
        scc = spearmanr(targets, preds)[0]
        
    return mse, pcc, scc


def plot_training_history(history, log_dir, seed):
    """
    Generates and saves a PNG plot of the training progress.
    """
    epochs = history['Epoch']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss Plot
    axes[0].plot(epochs, history['Tr_Loss'], label='Train Loss')
    axes[0].plot(epochs, history['Val_Loss'], label='Val Loss')
    axes[0].set_title('Mean Squared Error (Loss)')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Pearson Correlation Coefficient (PCC) Plot
    axes[1].plot(epochs, history['Tr_PCC_Dev'], label='Train Dev', linestyle='--')
    axes[1].plot(epochs, history['Val_PCC_Dev'], label='Val Dev')
    axes[1].plot(epochs, history['Tr_PCC_Hk'], label='Train Hk', linestyle='--')
    axes[1].plot(epochs, history['Val_PCC_Hk'], label='Val Hk')
    axes[1].set_title('Pearson Correlation (PCC)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. Spearman Correlation Coefficient (SCC) Plot
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
    """
    Universal training loop controlled entirely by the YAML config.
    """
    # 1. Configuration & Setup
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
    
    # 2. Comet.ml Initialization
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

    # 3. Model Building
    print(f"[INFO] Building model: {config['model']['name']}")
    model = build_model(config).to(device)
    
    # 4. Data Loading
    train_loader = prepare_input(set_name='Train', config=config)
    val_loader = prepare_input(set_name='Val', config=config)
    
    # 5. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # --- SCHEDULER Z LINIOWYM WARMUPEM (LINEAR WARMUP + COSINE DECAY) ---
    scheduler_cfg = train_cfg.get('scheduler', {'apply': False})
    scheduler = None
    
    if scheduler_cfg.get('apply', False) and scheduler_cfg.get('type') == 'cosine':
        eta_min = float(scheduler_cfg.get('eta_min', 1e-6))
        warmup_fraction = float(scheduler_cfg.get('warmup_fraction', 0.0))
        
        warmup_epochs = int(epochs * warmup_fraction)
        
        if warmup_epochs > 0:
            # Liniowy wzrost LR od poziomu (eta_min) do docelowego LR
            start_factor = eta_min / lr if lr > 0 else 1e-6
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=start_factor, total_iters=warmup_epochs
            )
            # Spadek cosinusowy przez pozostałą część epok
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(epochs - warmup_epochs), eta_min=eta_min
            )
            # Złączenie obu schedulerów
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, cosine_scheduler], 
                milestones=[warmup_epochs]
            )
            print(f"[INFO] Scheduler: Linear Warmup ({warmup_epochs} epochs) -> Cosine Decay ({epochs - warmup_epochs} epochs)")
        else:
            # Jeśli warmup_fraction = 0, używamy klasycznego cosinusa
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
            print(f"[INFO] Scheduler: Pure CosineAnnealingLR (T_max={epochs}, eta_min={eta_min})")

    # 6. Local Logging Setup
    best_val_loss = float('inf')
    epochs_no_improve = 0
    log_file = os.path.join(log_dir, f'training_log_{seed}.csv')
    
    headers = [
        'Epoch', 'LR', 'Time(s)', 
        'Tr_Loss', 'Tr_MSE_Dev', 'Tr_PCC_Dev', 'Tr_SCC_Dev', 'Tr_MSE_Hk', 'Tr_PCC_Hk', 'Tr_SCC_Hk',
        'Val_Loss', 'Val_MSE_Dev', 'Val_PCC_Dev', 'Val_SCC_Dev', 'Val_MSE_Hk', 'Val_PCC_Hk', 'Val_SCC_Hk'
    ]
    with open(log_file, mode='w', newline='') as file:
        csv.writer(file).writerow(headers)
        
    history = {k: [] for k in headers}
    
    # ------------------ MAIN TRAINING LOOP ------------------
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Pobieranie aktualnego LR na ten krok do logów
        current_lr = optimizer.param_groups[0]['lr']
        
        # --- TRAIN PHASE ---
        model.train()
        train_loss = 0.0
        train_preds_dev, train_preds_hk = [], []
        train_targs_dev, train_targs_hk = [], []
        
        for X_batch, Y_dev_batch, Y_hk_batch in train_loader:
            X_batch, Y_dev_batch, Y_hk_batch = X_batch.to(device), Y_dev_batch.to(device), Y_hk_batch.to(device)
            
            optimizer.zero_grad()
            pred_dev, pred_hk = model(X_batch)
            loss = criterion(pred_dev.squeeze(), Y_dev_batch) + criterion(pred_hk.squeeze(), Y_hk_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
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
        
        with torch.no_grad():
            for X_batch, Y_dev_batch, Y_hk_batch in val_loader:
                X_batch, Y_dev_batch, Y_hk_batch = X_batch.to(device), Y_dev_batch.to(device), Y_hk_batch.to(device)
                
                pred_dev, pred_hk = model(X_batch)
                loss = criterion(pred_dev.squeeze(), Y_dev_batch) + criterion(pred_hk.squeeze(), Y_hk_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                val_preds_dev.extend(pred_dev.cpu().numpy().flatten())
                val_preds_hk.extend(pred_hk.cpu().numpy().flatten())
                val_targs_dev.extend(Y_dev_batch.cpu().numpy().flatten())
                val_targs_hk.extend(Y_hk_batch.cpu().numpy().flatten())
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_mse_dev, val_pcc_dev, val_scc_dev = calculate_metrics(val_targs_dev, val_preds_dev)
        val_mse_hk, val_pcc_hk, val_scc_hk = calculate_metrics(val_targs_hk, val_preds_hk)

        epoch_time = time.time() - epoch_start_time

        # --- SCHEDULER STEP ---
        if scheduler:
            scheduler.step()

        # --- LOGGING & SAVING ---
        row_data = [
            epoch + 1, current_lr, epoch_time, 
            avg_train_loss, tr_mse_dev, tr_pcc_dev, tr_scc_dev, tr_mse_hk, tr_pcc_hk, tr_scc_hk,
            avg_val_loss, val_mse_dev, val_pcc_dev, val_scc_dev, val_mse_hk, val_pcc_hk, val_scc_hk
        ]
        
        with open(log_file, mode='a', newline='') as file:
            csv.writer(file).writerow(row_data)
            
        for key, val in zip(headers, row_data):
            history[key].append(val)

        if experiment:
            experiment.log_metrics({
                "Loss/Train": avg_train_loss, "Loss/Val": avg_val_loss,
                "PCC_Dev/Train": tr_pcc_dev, "PCC_Dev/Val": val_pcc_dev,
                "SCC_Dev/Train": tr_scc_dev, "SCC_Dev/Val": val_scc_dev,
                "PCC_Hk/Train": tr_pcc_hk, "PCC_Hk/Val": val_pcc_hk,
                "SCC_Hk/Train": tr_scc_hk, "SCC_Hk/Val": val_scc_hk,
                "Learning_Rate": current_lr,
                "Epoch_Duration_sec": epoch_time
            }, step=epoch + 1)

        print(f"Epoch {epoch+1:03d}/{epochs} [LR: {current_lr:.2e}] | "
              f"Loss: Tr={avg_train_loss:.4f} Val={avg_val_loss:.4f} | "
              f"Dev PCC: Tr={tr_pcc_dev:.3f} Val={val_pcc_dev:.3f}")

        # --- EARLY STOPPING & CHECKPOINT ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            model_path = os.path.join(log_dir, f"{config['model']['name']}_best_seed{seed}.pth")
            torch.save(model.state_dict(), model_path)
            if experiment: 
                experiment.log_model(config['model']['name'], model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"[INFO] Early stopping triggered after {epochs_no_improve} epochs.")
                break
                
    plot_training_history(history, log_dir, seed)
    if experiment:
        experiment.end()
        
    return model


if __name__ == '__main__':
    # Usage Example:
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal Training Script")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    # Load configuration
    main_config = load_config(args.config)
    
    # Run training
    trained_model = train_model(main_config)
