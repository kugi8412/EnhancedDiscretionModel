#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# sae.sh

python -m ../py/sparse_ae.visualize_usae \
    --usae_checkpoint results/usae/all_models/usae.pth \
    --analysis_dir    results/usae/all_models/analysis/ \
    --indep_sae_dirs \
        results/sae/DeepSTARR \
        results/sae/LegNetPlus \
        results/sae/LegNetOracle \
    --output_dir results/usae/all_models/plots/
