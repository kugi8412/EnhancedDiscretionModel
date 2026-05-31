#!/usr/bin/env python
"""
Training script for NucleotideCNN with:
  - Frozen or unfrozen backbone
  - MLP predictor from latent space (Dev/Hk) with train/val/test reporting
  - MC Dropout uncertainty estimation
  - MSE-based uncertainty estimation
  - Active learning on worst (most uncertain) sequences

Usage:
    python train_nucleotide.py -c ../../config/NucleotideCNN.yaml
    python train_nucleotide.py -c ../../config/NucleotideCNN.yaml --train-mlp
    python train_nucleotide.py -c ../../config/NucleotideCNN.yaml --active-learning --al-rounds 3
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
from torch.utils.data import DataLoader, TensorDataset, Subset

from utils import prepare_input, load_config, set_global_seed
from models.registry import build_model


# ==============================================================================
# MLP Latent Predictor
# ==============================================================================

class LatentMLP(nn.Module):
    """MLP that predicts Dev/Hk expression from backbone latent features.

    Parameters
    ----------
    input_dim : int
        Dimensionality of pooled backbone features.
    hidden_dims : list[int]
        Hidden layer sizes.
    dropout : float
        Dropout rate between layers.
    """

    def __init__(self, input_dim, hidden_dims=(256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        self.backbone = nn.Sequential(*layers)
        self.head_dev = nn.Linear(prev_dim, 1)
        self.head_hk = nn.Linear(prev_dim, 1)

    def forward(self, features):
        h = self.backbone(features)
        return self.head_dev(h).squeeze(-1), self.head_hk(h).squeeze(-1)


# ==============================================================================
# Metrics
# ==============================================================================

def calculate_metrics(targets, preds):
    if np.std(preds) == 0 or np.std(targets) == 0:
        return 0.0, 0.0, 0.0
    mse = float(F.mse_loss(
        torch.tensor(preds, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32)).item())
    pcc = pearsonr(targets, preds)[0]
    scc = spearmanr(targets, preds)[0]
    return mse, pcc, scc


# ==============================================================================
# Phase 1: Train NucleotideCNN (cross-entropy on nucleotide identity)
# ==============================================================================

def train_nucleotide_cnn(config):
    """Train the NucleotideCNN model (frozen or unfrozen backbone)."""
    seed = config.get('seed', 42)
    set_global_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Training NucleotideCNN on device: {device}")

    train_cfg = config.get('training', {})
    epochs = train_cfg.get('epochs', 80)
    lr = float(train_cfg.get('lr', 5e-4))
    weight_decay = float(train_cfg.get('weight_decay', 1e-4))
    early_stop_patience = train_cfg.get('early_stop', 15)
    log_dir = train_cfg.get('log_dir', 'train_logs')
    os.makedirs(log_dir, exist_ok=True)

    # Build model (with freeze_backbone from config)
    model = build_model(config).to(device)
    freeze_backbone = config['model'].get('kwargs', {}).get('freeze_backbone', True)
    print(f"[INFO] Backbone frozen: {freeze_backbone}")

    # Data loaders
    train_loader = prepare_input(set_name='Train', config=config)
    val_loader = prepare_input(set_name='Val', config=config)

    # Optimizer: only head if frozen, everything if unfrozen
    if freeze_backbone:
        params = model.head.parameters()
    else:
        params = model.parameters()

    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Scheduler
    scheduler = None
    sched_cfg = train_cfg.get('scheduler', {})
    if sched_cfg.get('apply', False) and sched_cfg.get('type') == 'cosine':
        eta_min = float(sched_cfg.get('eta_min', 1e-6))
        warmup_frac = float(sched_cfg.get('warmup_fraction', 0.0))
        warmup_epochs = int(epochs * warmup_frac)
        if warmup_epochs > 0:
            warmup_sched = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=eta_min / lr, total_iters=warmup_epochs)
            cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=eta_min)
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=eta_min)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    model_path = os.path.join(log_dir, f"{config['experiment_name']}_seed{seed}.pth")

    log_file = os.path.join(log_dir, f'nucleotide_log_{seed}.csv')
    headers = ['Epoch', 'LR', 'Time', 'Train_CE', 'Val_CE', 'Val_Entropy_Mean']
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(headers)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            X_batch = batch[0].to(device)  # (B, 4, L)
            optimizer.zero_grad()

            logits = model(X_batch)  # (B, 4, L)
            targets = X_batch.argmax(dim=1)  # (B, L)
            loss = F.cross_entropy(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            n_train += X_batch.size(0)

        avg_train = train_loss / n_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_entropy_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                X_batch = batch[0].to(device)
                logits = model(X_batch)
                targets = X_batch.argmax(dim=1)
                loss = F.cross_entropy(logits, targets)
                val_loss += loss.item() * X_batch.size(0)

                # Mean entropy
                probs = F.softmax(logits, dim=1)
                entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=1).mean()
                val_entropy_sum += entropy.item() * X_batch.size(0)
                n_val += X_batch.size(0)

        avg_val = val_loss / n_val
        avg_entropy = val_entropy_sum / n_val

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, current_lr, elapsed, avg_train, avg_val, avg_entropy])

        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"CE: Train={avg_train:.4f} Val={avg_val:.4f} | "
              f"Entropy={avg_entropy:.3f} | Time={elapsed:.1f}s")

        if scheduler:
            scheduler.step()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  → Saved best model (val_CE={avg_val:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

    print(f"[INFO] Best validation CE: {best_val_loss:.4f}")
    print(f"[INFO] Model saved to: {model_path}")
    return model, model_path


# ==============================================================================
# Phase 2: Train MLP on latent features for Dev/Hk prediction
# ==============================================================================

def extract_features_and_labels(model, loader, device):
    """Extract pooled backbone features + expression labels from a data loader."""
    model.eval()
    all_feats, all_dev, all_hk = [], [], []

    with torch.no_grad():
        for batch in loader:
            X = batch[0].to(device)
            Y_dev = batch[1]
            Y_hk = batch[2]

            feats = model.get_features(X)  # (B, feat_dim)
            all_feats.append(feats.cpu())
            all_dev.append(Y_dev)
            all_hk.append(Y_hk)

    return (torch.cat(all_feats, dim=0),
            torch.cat(all_dev, dim=0),
            torch.cat(all_hk, dim=0))


def train_latent_mlp(config, model_path=None):
    """Train an MLP predictor from latent features → Dev/Hk expression."""
    seed = config.get('seed', 42)
    set_global_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*60)
    print("[MLP] Training Latent MLP Predictor")
    print("="*60)

    # Load trained NucleotideCNN
    model = build_model(config).to(device)
    if model_path is None:
        train_cfg = config.get('training', {})
        log_dir = train_cfg.get('log_dir', 'train_logs')
        model_path = os.path.join(
            log_dir, f"{config['experiment_name']}_seed{seed}.pth")
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[MLP] Loaded NucleotideCNN from: {model_path}")

    # Extract features from all splits
    train_loader = prepare_input(set_name='Train', config=config)
    val_loader = prepare_input(set_name='Val', config=config)
    test_loader = prepare_input(set_name='Test', config=config)

    print("[MLP] Extracting features...")
    train_feats, train_dev, train_hk = extract_features_and_labels(model, train_loader, device)
    val_feats, val_dev, val_hk = extract_features_and_labels(model, val_loader, device)
    test_feats, test_dev, test_hk = extract_features_and_labels(model, test_loader, device)
    print(f"[MLP] Feature dim: {train_feats.shape[1]}, "
          f"Train={len(train_feats)}, Val={len(val_feats)}, Test={len(test_feats)}")

    # MLP config
    mlp_cfg = config.get('mlp_predictor', {})
    hidden_dims = mlp_cfg.get('hidden_dims', [256, 128])
    mlp_lr = float(mlp_cfg.get('lr', 1e-3))
    mlp_epochs = mlp_cfg.get('epochs', 100)
    mlp_dropout = mlp_cfg.get('dropout', 0.1)
    mlp_patience = mlp_cfg.get('early_stop', 20)
    mlp_batch = mlp_cfg.get('batch_size', 256)

    # Build MLP
    feat_dim = train_feats.shape[1]
    mlp = LatentMLP(feat_dim, hidden_dims=hidden_dims, dropout=mlp_dropout).to(device)
    optimizer = optim.AdamW(mlp.parameters(), lr=mlp_lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # DataLoaders for feature tensors
    train_ds = TensorDataset(train_feats, train_dev, train_hk)
    val_ds = TensorDataset(val_feats, val_dev, val_hk)
    test_ds = TensorDataset(test_feats, test_dev, test_hk)

    train_dl = DataLoader(train_ds, batch_size=mlp_batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=mlp_batch)
    test_dl = DataLoader(test_ds, batch_size=mlp_batch)

    # Training
    best_val_pcc = -float('inf')
    no_improve = 0
    log_dir = config.get('training', {}).get('log_dir', 'train_logs')
    mlp_path = os.path.join(log_dir, f"mlp_predictor_seed{seed}.pth")

    for epoch in range(mlp_epochs):
        mlp.train()
        for feats_b, dev_b, hk_b in train_dl:
            feats_b, dev_b, hk_b = feats_b.to(device), dev_b.to(device), hk_b.to(device)
            optimizer.zero_grad()
            pred_dev, pred_hk = mlp(feats_b)
            loss = criterion(pred_dev, dev_b) + criterion(pred_hk, hk_b)
            loss.backward()
            optimizer.step()

        # Validation
        mlp.eval()
        val_preds_dev, val_preds_hk = [], []
        val_targs_dev, val_targs_hk = [], []
        with torch.no_grad():
            for feats_b, dev_b, hk_b in val_dl:
                feats_b = feats_b.to(device)
                pd, ph = mlp(feats_b)
                val_preds_dev.extend(pd.cpu().numpy())
                val_preds_hk.extend(ph.cpu().numpy())
                val_targs_dev.extend(dev_b.numpy())
                val_targs_hk.extend(hk_b.numpy())

        _, val_pcc_dev, _ = calculate_metrics(
            np.array(val_targs_dev), np.array(val_preds_dev))
        _, val_pcc_hk, _ = calculate_metrics(
            np.array(val_targs_hk), np.array(val_preds_hk))
        avg_pcc = (val_pcc_dev + val_pcc_hk) / 2

        if (epoch + 1) % 10 == 0:
            print(f"  [MLP] Epoch {epoch+1}/{mlp_epochs} | "
                  f"Val PCC: Dev={val_pcc_dev:.4f} Hk={val_pcc_hk:.4f}")

        if avg_pcc > best_val_pcc:
            best_val_pcc = avg_pcc
            no_improve = 0
            torch.save(mlp.state_dict(), mlp_path)
        else:
            no_improve += 1
            if no_improve >= mlp_patience:
                print(f"  [MLP] Early stop at epoch {epoch+1}")
                break

    # Load best and evaluate on test
    mlp.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
    mlp.eval()

    results = {}
    for split_name, dl in [('Train', train_dl), ('Val', val_dl), ('Test', test_dl)]:
        preds_dev, preds_hk = [], []
        targs_dev, targs_hk = [], []
        with torch.no_grad():
            for feats_b, dev_b, hk_b in dl:
                feats_b = feats_b.to(device)
                pd, ph = mlp(feats_b)
                preds_dev.extend(pd.cpu().numpy())
                preds_hk.extend(ph.cpu().numpy())
                targs_dev.extend(dev_b.numpy())
                targs_hk.extend(hk_b.numpy())

        mse_dev, pcc_dev, scc_dev = calculate_metrics(
            np.array(targs_dev), np.array(preds_dev))
        mse_hk, pcc_hk, scc_hk = calculate_metrics(
            np.array(targs_hk), np.array(preds_hk))
        results[split_name] = {
            'Dev': {'MSE': mse_dev, 'PCC': pcc_dev, 'SCC': scc_dev},
            'Hk': {'MSE': mse_hk, 'PCC': pcc_hk, 'SCC': scc_hk},
        }

    print("\n[MLP] Final Results (Latent → Expression):")
    print(f"{'Split':<8} {'Dev_PCC':>8} {'Dev_SCC':>8} {'Hk_PCC':>8} {'Hk_SCC':>8}")
    print("-" * 44)
    for split in ['Train', 'Val', 'Test']:
        r = results[split]
        print(f"{split:<8} {r['Dev']['PCC']:>8.4f} {r['Dev']['SCC']:>8.4f} "
              f"{r['Hk']['PCC']:>8.4f} {r['Hk']['SCC']:>8.4f}")

    print(f"\n[MLP] Model saved to: {mlp_path}")
    return mlp, results


# ==============================================================================
# Phase 3: Uncertainty estimation (MC Dropout + MSE)
# ==============================================================================

def compute_uncertainty(config, model_path=None, method='combined',
                        mc_samples=20, mc_weight=0.5):
    """Compute per-sequence uncertainty scores for all data splits.

    Parameters
    ----------
    method : str
        'mc' — MC Dropout only
        'mse' — MSE only
        'combined' — weighted combination
    """
    seed = config.get('seed', 42)
    set_global_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(config).to(device)
    if model_path is None:
        train_cfg = config.get('training', {})
        log_dir = train_cfg.get('log_dir', 'train_logs')
        model_path = os.path.join(
            log_dir, f"{config['experiment_name']}_seed{seed}.pth")
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print(f"\n[UNCERTAINTY] Method: {method}, MC samples: {mc_samples}")

    results = {}
    for split in ['Train', 'Val', 'Test']:
        loader = prepare_input(set_name=split, config=config)
        all_scores = []

        for batch in loader:
            X = batch[0].to(device)
            with torch.no_grad():
                if method == 'mc':
                    scores = model.mc_uncertainty(X, n_samples=mc_samples)
                elif method == 'mse':
                    _, scores = model.mse_uncertainty(X)
                elif method == 'combined':
                    scores, _, _ = model.combined_uncertainty(
                        X, n_samples=mc_samples, mc_weight=mc_weight)
                else:
                    raise ValueError(f"Unknown method: {method}")
            all_scores.append(scores.cpu())

        all_scores = torch.cat(all_scores, dim=0).numpy()
        results[split] = all_scores
        print(f"  {split}: mean={all_scores.mean():.5f}, "
              f"std={all_scores.std():.5f}, "
              f"max={all_scores.max():.5f}, "
              f"top-10%={np.percentile(all_scores, 90):.5f}")

    return results


# ==============================================================================
# Phase 4: Active Learning on worst sequences
# ==============================================================================

def active_learning(config, model_path=None, n_rounds=3, top_fraction=0.1,
                    uncertainty_method='combined', mc_samples=20, mc_weight=0.5,
                    retrain_epochs=30):
    """Active learning loop: identify most uncertain sequences, retrain on them.

    Strategy:
        1. Compute uncertainty on training set
        2. Select top_fraction most uncertain sequences
        3. Fine-tune model on those sequences (with higher weight)
        4. Repeat for n_rounds

    Parameters
    ----------
    n_rounds : int
        Number of active learning iterations.
    top_fraction : float
        Fraction of most uncertain sequences to focus on each round.
    retrain_epochs : int
        Epochs to fine-tune on uncertain subset each round.
    """
    seed = config.get('seed', 42)
    set_global_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_cfg = config.get('training', {})
    log_dir = train_cfg.get('log_dir', 'train_logs')
    os.makedirs(log_dir, exist_ok=True)

    # Load trained model
    model = build_model(config).to(device)
    if model_path is None:
        model_path = os.path.join(
            log_dir, f"{config['experiment_name']}_seed{seed}.pth")
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"\n[ACTIVE LEARNING] Starting with model: {model_path}")
    print(f"  Rounds: {n_rounds}, Top fraction: {top_fraction}, "
          f"Retrain epochs: {retrain_epochs}")

    # Load full training data as tensors
    train_loader = prepare_input(set_name='Train', config=config)
    val_loader = prepare_input(set_name='Val', config=config)

    # Collect all training data
    all_X, all_dev, all_hk = [], [], []
    for batch in train_loader:
        all_X.append(batch[0])
        all_dev.append(batch[1])
        all_hk.append(batch[2])
    all_X = torch.cat(all_X, dim=0)
    all_dev = torch.cat(all_dev, dim=0)
    all_hk = torch.cat(all_hk, dim=0)
    N = len(all_X)

    for round_idx in range(n_rounds):
        print(f"\n--- Active Learning Round {round_idx + 1}/{n_rounds} ---")

        # 1. Compute uncertainty on full training set
        model.eval()
        uncertainties = []
        batch_size = config.get('data', {}).get('batch_size', 128)

        for i in range(0, N, batch_size):
            X_batch = all_X[i:i+batch_size].to(device)
            with torch.no_grad():
                if uncertainty_method == 'mc':
                    scores = model.mc_uncertainty(X_batch, n_samples=mc_samples)
                elif uncertainty_method == 'mse':
                    _, scores = model.mse_uncertainty(X_batch)
                else:
                    scores, _, _ = model.combined_uncertainty(
                        X_batch, n_samples=mc_samples, mc_weight=mc_weight)
            uncertainties.append(scores.cpu())

        uncertainties = torch.cat(uncertainties, dim=0)

        # 2. Select top uncertain sequences
        n_select = max(1, int(N * top_fraction))
        top_indices = uncertainties.argsort(descending=True)[:n_select]

        print(f"  Selected {n_select} most uncertain sequences "
              f"(uncertainty range: {uncertainties[top_indices[-1]]:.5f} - "
              f"{uncertainties[top_indices[0]]:.5f})")

        # 3. Create focused dataset (uncertain subset + random sample of rest)
        # Mix: 50% uncertain + 50% random from full set for regularization
        n_random = min(n_select, N - n_select)
        remaining_mask = torch.ones(N, dtype=torch.bool)
        remaining_mask[top_indices] = False
        remaining_indices = remaining_mask.nonzero(as_tuple=True)[0]
        random_perm = torch.randperm(len(remaining_indices))[:n_random]
        random_indices = remaining_indices[random_perm]

        focus_indices = torch.cat([top_indices, random_indices])
        focus_X = all_X[focus_indices]
        focus_dev = all_dev[focus_indices]
        focus_hk = all_hk[focus_indices]

        focus_ds = TensorDataset(focus_X, focus_dev, focus_hk)
        focus_dl = DataLoader(focus_ds, batch_size=batch_size, shuffle=True)

        # 4. Fine-tune on focused subset
        freeze_backbone = config['model'].get('kwargs', {}).get('freeze_backbone', True)
        if freeze_backbone:
            params = model.head.parameters()
        else:
            params = model.parameters()

        al_lr = float(train_cfg.get('lr', 5e-4)) * 0.1  # Lower LR for fine-tuning
        optimizer = optim.AdamW(params, lr=al_lr, weight_decay=1e-4)

        model.train()
        for ep in range(retrain_epochs):
            ep_loss = 0.0
            for batch in focus_dl:
                X_b = batch[0].to(device)
                optimizer.zero_grad()
                logits = model(X_b)
                targets = X_b.argmax(dim=1)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ep_loss += loss.item() * X_b.size(0)
            if (ep + 1) % 10 == 0:
                print(f"    Retrain epoch {ep+1}/{retrain_epochs}: "
                      f"CE={ep_loss/len(focus_ds):.4f}")

        # 5. Evaluate on validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                X_b = batch[0].to(device)
                logits = model(X_b)
                targets = X_b.argmax(dim=1)
                val_loss += F.cross_entropy(logits, targets).item() * X_b.size(0)
                n_val += X_b.size(0)
        print(f"  After round {round_idx+1}: Val CE = {val_loss/n_val:.4f}")

    # Save active-learned model
    al_path = os.path.join(log_dir, f"{config['experiment_name']}_AL_seed{seed}.pth")
    torch.save(model.state_dict(), al_path)
    print(f"\n[ACTIVE LEARNING] Final model saved to: {al_path}")
    return model


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NucleotideCNN Training Pipeline")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument('--skip-train', action='store_true',
                        help="Skip NucleotideCNN training (use existing weights)")
    parser.add_argument('--model-path', type=str, default=None,
                        help="Path to pretrained NucleotideCNN weights")

    # MLP predictor
    parser.add_argument('--train-mlp', action='store_true',
                        help="Train MLP predictor from latent features")

    # Uncertainty
    parser.add_argument('--uncertainty', action='store_true',
                        help="Compute uncertainty estimates")
    parser.add_argument('--uncertainty-method', type=str, default='combined',
                        choices=['mc', 'mse', 'combined'],
                        help="Uncertainty method: mc, mse, or combined")
    parser.add_argument('--mc-samples', type=int, default=20,
                        help="Number of MC Dropout samples")
    parser.add_argument('--mc-weight', type=float, default=0.5,
                        help="Weight for MC component in combined uncertainty")

    # Active learning
    parser.add_argument('--active-learning', action='store_true',
                        help="Run active learning on most uncertain sequences")
    parser.add_argument('--al-rounds', type=int, default=3,
                        help="Number of active learning rounds")
    parser.add_argument('--al-fraction', type=float, default=0.1,
                        help="Fraction of most uncertain sequences per round")
    parser.add_argument('--al-epochs', type=int, default=30,
                        help="Retrain epochs per active learning round")

    args = parser.parse_args()
    config = load_config(args.config)

    model_path = args.model_path

    # Phase 1: Train NucleotideCNN
    if not args.skip_train:
        _, model_path = train_nucleotide_cnn(config)

    # Phase 2: Train MLP predictor
    if args.train_mlp:
        train_latent_mlp(config, model_path=model_path)

    # Phase 3: Uncertainty estimation
    if args.uncertainty:
        compute_uncertainty(
            config, model_path=model_path,
            method=args.uncertainty_method,
            mc_samples=args.mc_samples,
            mc_weight=args.mc_weight,
        )

    # Phase 4: Active learning
    if args.active_learning:
        active_learning(
            config, model_path=model_path,
            n_rounds=args.al_rounds,
            top_fraction=args.al_fraction,
            uncertainty_method=args.uncertainty_method,
            mc_samples=args.mc_samples,
            mc_weight=args.mc_weight,
            retrain_epochs=args.al_epochs,
        )
