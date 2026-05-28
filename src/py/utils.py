# -*- coding: utf-8 -*-
"""Shared utilities for data loading, configuration, and reproducibility."""

import os
import torch
from torch.utils.data import DataLoader
from Bio import SeqIO
import numpy as np
import pandas as pd
import yaml
import random

from datasets import DNADataset
from models.registry import build_model


GLOBAL_NORMALIZE_TARGETS = False 
GLOBAL_RC_CORRECTION = False


def load_config(config_path):
    """Load a YAML configuration file and return it as a dictionary."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def set_global_seed(seed):
    """Set random seeds across all libraries for reproducibility."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_reverse_complement(seq: str) -> str:
    """Return the reverse-complement of a DNA string."""
    trans = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
    return seq.translate(trans)[::-1]


def load_fasta_with_strand_correction(file_path: str) -> list:
    """Read a FASTA file and return strand-standardised sequences.

    Sequences whose header contains ``_-_`` (minus-strand coordinates)
    are reverse-complemented so that every returned sequence is in the
    5'→3' plus-strand orientation.  This is a no-op for FASTA files
    that were already pre-processed to plus-strand.

    Parameters
    ----------
    file_path : str
        Path to a FASTA file whose headers follow the DeepSTARR
        convention ``>chrom_start_end_±_label``.

    Returns
    -------
    list[str]
        Upper-case, plus-strand DNA sequences.
    """
    global GLOBAL_RC_CORRECTION

    trans = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
    sequences = []
    n_corrected = 0
    for record in SeqIO.parse(file_path, 'fasta'):
        seq = str(record.seq).upper()
        # Match '_-_' anywhere in the record id (e.g. chr2L_100_349_-_peak)
        if '_-_' in record.id and GLOBAL_RC_CORRECTION:
            seq = seq.translate(trans)[::-1]
            n_corrected += 1
        sequences.append(seq)
    if n_corrected:
        print(f'[strand] RC-corrected {n_corrected}/{len(sequences)} '
              f'minus-strand sequences in {os.path.basename(file_path)}')
    return sequences


def one_hot_encode_dna(sequences):
    mapping = {
        'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
        'T': [0, 0, 1, 0], 'G': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }
    mapping.update({k.lower(): v for k, v in mapping.items()})
    encoded = np.array([[mapping.get(base, [0, 0, 0, 0]) for base in seq] for seq in sequences])
    return encoded.transpose(0, 2, 1)


def load_fasta_sequences(file_path: str) -> list:
    """Read raw sequences from a FASTA file without strand correction.

    .. note::
        Use :func:`load_fasta_with_strand_correction` when building training
        data for models that are not RC-equivariant.
    """
    sequences = [str(record.seq).upper() for record in SeqIO.parse(file_path, 'fasta')]
    return sequences


def prepare_input(set_name, config, activity_cols=('Dev_log2_enrichment', 'Hk_log2_enrichment'), shuffle=None):
    """Load sequences and activity labels, returning a ready-to-use DataLoader."""
    
    # Importujemy obie zmienne globalne
    global GLOBAL_NORMALIZE_TARGETS, GLOBAL_RC_CORRECTION
    
    data_cfg = config.get('data', {})
    batch_size = data_cfg.get('batch_size', 128)
    set_dir = data_cfg.get('dataset_path', '../../data/deepSTARR')
    seed = config.get('seed', 42)
    num_workers = data_cfg.get('num_workers', 4)
    pin_memory = torch.cuda.is_available()

    is_train = (set_name.lower() == 'train')
    if shuffle is None:
        shuffle = is_train

    augment = data_cfg.get('augment', False) if is_train else False
    noise_config = data_cfg.get('target_noise', {'apply': False}) if is_train else {'apply': False}

    file_seq = f'{set_dir}/Sequences_{set_name}.fa'
    
    # ==========================================
    # STEROWANIE KOREKTĄ REVERSE COMPLEMENT (RC)
    # ==========================================
    if GLOBAL_RC_CORRECTION:
        sequences = load_fasta_with_strand_correction(file_seq)
        # UWAGA: Wewnątrz Twojej funkcji load_fasta_with_strand_correction 
        # prawdopodobnie jest już print informujący o zamianie nici.
    else:
        sequences = load_fasta_sequences(file_seq)
        print(f'[WARNING] {set_name}: We consider +/- sequences.')
    # ==========================================

    seq_matrix = one_hot_encode_dna(sequences)
    print(f'{set_name} Sequence Matrix Shape: {seq_matrix.shape}')

    X = np.nan_to_num(seq_matrix)

    activity_file = f'{set_dir}/Sequences_activity_{set_name}.txt'
    activity_data = pd.read_table(activity_file)

    Y_first = activity_data[activity_cols[0]].values
    Y_second = activity_data[activity_cols[1]].values

    # ==========================================
    # STEROWANIE NORMALIZACJĄ (Z-SCORE)
    # ==========================================
    if GLOBAL_NORMALIZE_TARGETS:
        if np.std(Y_first) > 0:
            Y_first = (Y_first - np.mean(Y_first)) / np.std(Y_first)
        if np.std(Y_second) > 0:
            Y_second = (Y_second - np.mean(Y_second)) / np.std(Y_second)
    
        print(f'[INFO] {set_name}: Z-score normalization.')
    # ==========================================

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_first_tensor = torch.tensor(Y_first, dtype=torch.float32)
    Y_second_tensor = torch.tensor(Y_second, dtype=torch.float32)

    print(f'Loaded {set_name} data.')

    evoaug_config = data_cfg.get('evoaug', {})

    dataset = DNADataset(
        X_tensor,
        Y_first_tensor,
        Y_second_tensor,
        augment=augment,
        noise_config=noise_config,
        evoaug_config=evoaug_config,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return dataloader


def load_model(model_path, config):
    """Instantiate a model from *config* and load weights from *model_path*."""
    model = build_model(config)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model
