#!/usr/bin/env python
# -*- coding: utf-8 -*-
# split_genome.py

"""
This script is designed to extract specific sequences from a reference genome based
on parameters defined in a YAML configuration file. It supports filtering sequences by
the allowed fraction of unknown nucleotides ('N') and, for accepted sequences,
replaces 'N's with random valid nucleotides using a fixed random seed.
"""

import os
import yaml
import random
import pandas as pd

from tqdm import tqdm
from Bio import SeqIO
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Read YAML file. Returns a dictionary with the configuration parameters."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def extract_sequences(config_path: str = "config.yml"):
    config = load_config(config_path)

    fasta_path = config['genome']['fasta']
    output_path = config['extraction']['output_file']
    chroms_input = config['genome'].get('chromosomes')
    max_n_fraction = config['extraction'].get('max_n_fraction', 0.0)
    seed_value = config['extraction'].get('seed', 42)

    random.seed(seed_value)

    if chroms_input:
        target_chromosomes = str(chroms_input).split()
    else:
        target_chromosomes = []
    
    window_size = config['extraction']['window_size']
    step_size = config['extraction']['step_size']
    extracted_data = []

    valid_nucleotides = ['A', 'C', 'G', 'T']

    if not target_chromosomes:
        print("[WARNING]: Processing ALL Chromosomes")
    
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            chrom = record.id

            if not target_chromosomes or chrom in target_chromosomes:
                print(f"Processing {chrom}...")
                sequence = str(record.seq).upper()
                seq_len = len(sequence)
                
                for start in tqdm(range(0, seq_len - window_size + 1, step_size),
                                  desc=f"{chrom}", leave=False):
                    end = start + window_size
                    fragment = sequence[start:end]
                    n_count = fragment.count('N')
                    n_fraction = n_count / window_size

                    if n_fraction <= max_n_fraction:  
                        if n_count > 0:
                            fragment = "".join(
                                [nuc if nuc != 'N' else random.choice(valid_nucleotides) for nuc in fragment]
                            )
                        
                        seq_id = f"{chrom}_{start}_{end}_+_fragment"
                        extracted_data.append({
                            'id': seq_id,
                            'sequence': fragment
                        })
                        
    except FileNotFoundError:
         print(f"[ERROR]: Not found file {fasta_path}. Check config.yml!")
         return

    print(f"\nIn total saved {len(extracted_data)} sequences!")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"[SUCCESS]: Save results to {output_path}")
    df = pd.DataFrame(extracted_data)
    df.to_csv(output_path, sep='\t', index=False)


if __name__ == "__main__":
    extract_sequences("../../config/config.yml")
