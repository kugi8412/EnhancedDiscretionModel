#!/usr/bin/env python
"""
Inspect all named layers/modules of a model built from a YAML config.

Usage:
    python inspect_layers.py --config ../../config/DeepSTARR.yaml
    python inspect_layers.py --config ../../config/LegNetPlus.yaml --weights train_logs/model.pth
    python inspect_layers.py --config ../../config/LegNetOracle.yaml --verbose
"""

import argparse
import torch
from utils import load_config
from models.registry import build_model


def inspect_model(config_path, weights_path=None, verbose=False):
    config = load_config(config_path)
    model = build_model(config)

    if weights_path:
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)

    model_name = config['model']['name']
    print(f"\n{'='*70}")
    print(f"Model: {model_name}  (config: {config_path})")
    print(f"{'='*70}")
    print(f"{'Layer Name':<50} {'Type':<30} {'Output Shape (params)'}")
    print(f"{'-'*50} {'-'*30} {'-'*30}")

    total_params = 0
    for name, module in model.named_modules():
        if name == '':
            continue
        # Count direct parameters (not children's)
        direct_params = sum(p.numel() for p in module.parameters(recurse=False))
        total_params += direct_params

        type_name = module.__class__.__name__

        if verbose or not list(module.children()):
            # Show leaf modules always; containers only in verbose mode
            param_str = f"{direct_params:,}" if direct_params > 0 else ""
            # Try to get shape info from weight
            shape_str = ""
            if hasattr(module, 'weight') and module.weight is not None:
                shape_str = str(list(module.weight.shape))
            elif hasattr(module, 'in_features'):
                shape_str = f"[{module.in_features} -> {module.out_features}]"
            elif hasattr(module, 'in_channels'):
                shape_str = f"[{module.in_channels} -> {module.out_channels}, k={getattr(module, 'kernel_size', '?')}]"

            info = f"{param_str}  {shape_str}" if param_str or shape_str else ""
            print(f"  {name:<48} {type_name:<30} {info}")

    total_all = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*70}")
    print(f"Total parameters:     {total_all:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect model layer names and shapes")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to model YAML config")
    parser.add_argument('--weights', type=str, default=None,
                        help="Optional: path to model weights (.pth)")
    parser.add_argument('--verbose', action='store_true',
                        help="Show all modules including containers")
    args = parser.parse_args()

    inspect_model(args.config, args.weights, args.verbose)
