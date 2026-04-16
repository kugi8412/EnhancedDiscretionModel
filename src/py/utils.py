# utils.py
import os
import torch
from torch.utils.data import DataLoader
from Bio import SeqIO
import numpy as np
import pandas as pd
import yaml       # <--- DODANE do ładowania configów
import random     # <--- DODANE do blokowania losowości (seed)

from datasets import DNADataset
from models.registry import build_model


def load_config(config_path):
    """Wczytuje słownik konfiguracyjny z pliku YAML."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def set_global_seed(seed):
    """Blokuje losowość we wszystkich bibliotekach dla powtarzalności wyników."""
    if seed is None:
        return
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Wymusza deterministyczne algorytmy w cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_reverse_complement(seq):
    trans = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
    return seq.translate(trans)[::-1]


def one_hot_encode_dna(sequences):
    mapping = {
        'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
        'T': [0, 0, 1, 0], 'G': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }
    mapping.update({k.lower(): v for k, v in mapping.items()})
    encoded = np.array([[mapping.get(base, [0, 0, 0, 0]) for base in seq] for seq in sequences])
    return encoded.transpose(0, 2, 1)


def load_fasta_sequences(file_path):
    """Reads sequences from a FASTA file and returns a list of sequences."""
    sequences = [str(record.seq).upper() for record in SeqIO.parse(file_path, 'fasta')]
    return sequences


def prepare_input(set_name, config, activity_cols=['Dev_log2_enrichment', 'Hk_log2_enrichment'], shuffle=None):
    """
    Loads sequences and their enhancer activity, converting sequences to one-hot encoding.
    Now configured dynamically via YAML config dictionary.
    """
    # 1. Wyciągnięcie parametrów z configa
    data_cfg = config.get('data', {})
    batch_size = data_cfg.get('batch_size', 128)
    set_dir = data_cfg.get('dataset_path', '../../data/deepSTARR')
    seed = config.get('seed', 42)
    
    # --- OPTYMALIZACJA DATALOADERA ---
    # Domyślnie 4 wątki dla szybszego ładowania, można nadpisać w config.yaml
    num_workers = data_cfg.get('num_workers', 4) 
    # Bezpośredni transfer do GPU
    pin_memory = torch.cuda.is_available()
    
    # 2. Logika dla zbioru Treningowego vs Walidacyjnego/Testowego
    is_train = (set_name.lower() == 'train')
    
    # Domyślnie tasujemy tylko zbiór treningowy
    if shuffle is None:
        shuffle = is_train
        
    # Augmentacja i szum TYLKO dla zbioru treningowego
    augment = data_cfg.get('augment', False) if is_train else False
    noise_config = data_cfg.get('target_noise', {'apply': False}) if is_train else {'apply': False}

    # 3. Ładowanie danych
    file_seq = f'{set_dir}/Sequences_{set_name}.fa'
    sequences = load_fasta_sequences(file_seq)
    
    seq_matrix = one_hot_encode_dna(sequences)
    print(f'{set_name} Sequence Matrix Shape: {seq_matrix.shape}')
    
    X = np.nan_to_num(seq_matrix)
    
    activity_file = f'{set_dir}/Sequences_activity_{set_name}.txt'
    activity_data = pd.read_table(activity_file)
    
    Y_first = activity_data[activity_cols[0]].values
    Y_second = activity_data[activity_cols[1]].values
    
    # Konwersja na tensory
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_first_tensor = torch.tensor(Y_first, dtype=torch.float32)
    Y_second_tensor = torch.tensor(Y_second, dtype=torch.float32)
    
    print(f'Loaded {set_name} data.')

    evoaug_config = data_cfg.get('evoaug', {})
    
    # 4. Inicjalizacja zaktualizowanego DNADataset
    dataset = DNADataset(
        X_tensor, 
        Y_first_tensor, 
        Y_second_tensor,
        augment=augment,
        noise_config=noise_config,
        evoaug_config=evoaug_config, # <--- DODANO TĘ LINIJKĘ
        seed=seed
    )
    
    # 5. Tworzenie zoptymalizowanego DataLoadera
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0) # Utrzymuje workery przy życiu między epokami
    )
    
    return dataloader


def load_model(model_path, config):
    """
    Instantiates the model using the registry and loads weights.
    """
    # Zbudowanie architektury wskazanej w pliku YAML (np. DeepSTARR, ConvNeXt itp.)
    model = build_model(config)
    
    # Ładowanie wag
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    
    return model
