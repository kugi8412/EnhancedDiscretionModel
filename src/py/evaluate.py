#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model evaluation script with correlation plots and reconstruction metrics."""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from utils import load_config, prepare_input
from models.registry import build_model


def adjust_axes(ax):
    """Remove top and right spines for cleaner plots."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def predicted_vs_observed(true, predicted, title, class_names=('dev', 'hk'), save_path=None):
    """Generate hexbin PCC correlation plots (DeepSTARR style)."""
    df_true = pd.read_csv(true, sep='\t')
    df_pred = pd.read_csv(predicted, sep='\t')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Developmental
    axes[0].hexbin(df_true[f'{class_names[0].capitalize()}_log2_enrichment'], 
                   df_pred[f'Predictions_{class_names[0]}'], bins='log')
    
    # Housekeeping
    axes[1].hexbin(df_true[f'{class_names[1].capitalize()}_log2_enrichment'], 
                   df_pred[f'Predictions_{class_names[1]}'], bins='log')

    adjust_axes(axes[0])
    adjust_axes(axes[1])
    
    fig.supxlabel('Observed fold change [log2]', fontsize=10)
    axes[0].set_ylabel('Predicted fold change [log2]', fontsize=10)

    pcc_dev = pearsonr(df_true[f'{class_names[0].capitalize()}_log2_enrichment'], 
                       df_pred[f'Predictions_{class_names[0]}'])[0]
    pcc_hk = pearsonr(df_true[f'{class_names[1].capitalize()}_log2_enrichment'], 
                      df_pred[f'Predictions_{class_names[1]}'])[0]

    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) 

    if class_names == ('dev', 'hk'):
        axes[0].set_title(f'Developmental (PCC = {pcc_dev:.3f})', fontsize=10)
        axes[1].set_title(f'Housekeeping (PCC = {pcc_hk:.3f})', fontsize=10)
    else:
        axes[0].set_title(f'Primary (PCC = {pcc_dev:.3f})', fontsize=10)
        axes[1].set_title(f'Organoid (PCC = {pcc_hk:.3f})', fontsize=10)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[INFO] Plot saved to: {save_path}")
    plt.close(fig)

def evaluate_model(config_path, weights_path, set_name='Test'):
    """Main evaluation logic supporting cVQ-VAE and LegNet models."""
    config = load_config(config_path)
    
    # Disable augmentation and noise for unbiased evaluation
    if 'data' in config:
        config['data']['augment'] = False
        if 'target_noise' in config['data']:
            config['data']['target_noise']['apply'] = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = config['model']['name']
    
    print(f"[INFO] Evaluating model {model_name} on {set_name} set...")
    
    # Load main model
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Load frozen Oracle (required for cVQ-VAE prediction from Gumbel samples)
    oracle_model = None
    oracle_cfg = config.get('oracle', None)
    if oracle_cfg and oracle_cfg.get('apply', False):
        print(f"[INFO] Loading frozen Oracle for generated sequence evaluation...")
        oracle_model_config = load_config(oracle_cfg['config_path'])
        oracle_model = build_model(oracle_model_config).to(device)
        oracle_model.load_state_dict(torch.load(oracle_cfg['weights_path'], map_location=device))
        oracle_model.eval()

    dataloader = prepare_input(set_name, config, shuffle=False)
    pred_dev_list, pred_hk_list = [], []
    recon_acc_list = []

    print("[INFO] Generating predictions...")
    with torch.no_grad():
        for X_batch, Y_dev_batch, Y_hk_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_dev_batch = Y_dev_batch.to(device)
            Y_hk_batch = Y_hk_batch.to(device)
            
            try:
                # Pass true labels for FiLM conditioning in cVQ-VAE
                outputs = model(X_batch, Y_dev_batch, Y_hk_batch)
            except TypeError:
                outputs = model(X_batch)
            
            if isinstance(outputs, tuple) and len(outputs) == 3 and oracle_model is not None:
                x_logits_8ch, gumbels, _ = outputs
                x_gumbel_fwd, _ = gumbels
                
                # Compute reconstruction accuracy
                x_fwd_logits = x_logits_8ch[:, 0:4, :]
                true_seq_indices = torch.argmax(X_batch, dim=1)
                preds_seq = torch.argmax(x_fwd_logits, dim=1)
                matches = (preds_seq == true_seq_indices).float()
                recon_acc_list.extend((matches.mean(dim=1) * 100.0).cpu().numpy().tolist())
                
                pred_dev, pred_hk = oracle_model(x_gumbel_fwd)
                
            elif isinstance(outputs, tuple) and len(outputs) == 4:
                pred_dev, pred_hk, x_recon, _ = outputs
                # Compute reconstruction accuracy for VQ models
                true_seq_indices = torch.argmax(X_batch, dim=1)
                preds_seq = torch.argmax(x_recon, dim=1)
                matches = (preds_seq == true_seq_indices).float()
                recon_acc_list.extend((matches.mean(dim=1) * 100.0).cpu().numpy().tolist())
                
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                pred_dev, pred_hk = outputs
            else:
                pred_dev = outputs
                
            if not isinstance(pred_dev, (list, tuple)) and pred_dev.dim() > 1 and pred_dev.shape[1] == 2:
                pred_hk = pred_dev[:, 1]
                pred_dev = pred_dev[:, 0]
                
            pred_dev_list.extend(pred_dev.cpu().numpy().flatten())
            pred_hk_list.extend(pred_hk.cpu().numpy().flatten())

    out_dir = os.path.join('outputs', model_name.lower())
    os.makedirs(out_dir, exist_ok=True)
    pred_filename = os.path.join(out_dir, f'Predictions_{set_name}.txt')
    
    df_pred = pd.DataFrame({
        'Predictions_dev': pred_dev_list,
        'Predictions_hk': pred_hk_list
    })
    df_pred.to_csv(pred_filename, sep='\t', index=False)
    print(f"[INFO] Predictions saved to: {pred_filename}")

    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = os.path.join(plots_dir, f'{model_name}_Predictions_{set_name}.png')
    
    dataset_path = config.get('data', {}).get('dataset_path', '../../data/deepSTARR')
    true_filename = f"{dataset_path}/Sequences_activity_{set_name}.txt"
    
    set_to_title = {'Test': 'test', 'Train': 'training', 'Val': 'validation'}
    title = f'{model_name} model predictions on the {set_to_title.get(set_name, set_name.lower())} set'
    
    predicted_vs_observed(true_filename, pred_filename, title, save_path=plot_filename)
    
    df_true = pd.read_csv(true_filename, sep='\t')
    pcc_dev = pearsonr(df_true['Dev_log2_enrichment'], df_pred['Predictions_dev'])[0]
    pcc_hk = pearsonr(df_true['Hk_log2_enrichment'], df_pred['Predictions_hk'])[0]
    
    print("\n===========================================")
    print(f" EVALUATION RESULTS ({set_name}) - {model_name}")
    print("===========================================")
    print(f" PCC Developmental: {pcc_dev:.4f}")
    print(f" PCC Housekeeping:  {pcc_hk:.4f}")
    if len(recon_acc_list) > 0:
        print(f" Reconstruction Accuracy: {np.mean(recon_acc_list):.2f}%")
    print("===========================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model and plot PCC correlations.")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to YAML config file")
    parser.add_argument('-w', '--weights', type=str, required=True, help="Path to .pth model weights")
    parser.add_argument('-s', '--set', type=str, default='Test', help="Dataset to evaluate (Train, Val, Test)")
    
    args = parser.parse_args()
    evaluate_model(args.config, args.weights, args.set)
