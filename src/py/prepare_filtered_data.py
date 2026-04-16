#!/usr/bin/env python3
"""
Bridge between filtering-seq-pipeline and ML training framework.

Takes the cleaned/split FASTA files from the filtering pipeline and prepares
them into the format expected by train.py (Sequences_{Split}.fa + Sequences_activity_{Split}.txt).

Usage:
    python prepare_filtered_data.py \
        --train_fasta path/to/train.fasta \
        --test_fasta path/to/test.fasta \
        --activity path/to/Sequences_activity_All.txt \
        --output_dir ../../data/deepSTARR_filtered \
        --val_ratio 0.1
"""

import os
import argparse
import random

import pandas as pd
from Bio import SeqIO


def main():
    parser = argparse.ArgumentParser(
        description="Convert filtering pipeline output to ML training format"
    )
    parser.add_argument("--train_fasta", required=True, help="Train FASTA from filtering pipeline")
    parser.add_argument("--test_fasta", required=True, help="Test FASTA from filtering pipeline")
    parser.add_argument("--activity", required=True,
                        help="Full activity file (TSV with ID, Dev_log2_enrichment, Hk_log2_enrichment)")
    parser.add_argument("--output_dir", required=True, help="Output directory for ML-ready files")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of train split to use as validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load activity data keyed by sequence ID
    activity_df = pd.read_table(args.activity)
    # Determine if ID is in first column or a separate column
    if 'ID' in activity_df.columns:
        activity_df = activity_df.set_index('ID')
    elif activity_df.columns[0] not in ('Dev_log2_enrichment', 'Hk_log2_enrichment'):
        activity_df = activity_df.set_index(activity_df.columns[0])

    # Load train/test FASTA records
    train_records = list(SeqIO.parse(args.train_fasta, "fasta"))
    test_records = list(SeqIO.parse(args.test_fasta, "fasta"))

    print(f"Train records: {len(train_records)}")
    print(f"Test records: {len(test_records)}")

    # Split train into train + val
    random.shuffle(train_records)
    n_val = int(len(train_records) * args.val_ratio)
    val_records = train_records[:n_val]
    train_records = train_records[n_val:]

    print(f"After split -> Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    # Write each split
    for split_name, records in [("Train", train_records), ("Val", val_records), ("Test", test_records)]:
        fasta_path = os.path.join(args.output_dir, f"Sequences_{split_name}.fa")
        activity_path = os.path.join(args.output_dir, f"Sequences_activity_{split_name}.txt")

        # Write FASTA
        with open(fasta_path, "w") as f:
            SeqIO.write(records, f, "fasta")

        # Write matching activity file
        rows = []
        for rec in records:
            seq_id = rec.id
            if seq_id in activity_df.index:
                row = activity_df.loc[seq_id]
                rows.append({
                    'Dev_log2_enrichment': row['Dev_log2_enrichment'],
                    'Hk_log2_enrichment': row['Hk_log2_enrichment']
                })
            else:
                print(f"WARNING: No activity data for sequence {seq_id}, using 0.0")
                rows.append({'Dev_log2_enrichment': 0.0, 'Hk_log2_enrichment': 0.0})

        pd.DataFrame(rows).to_csv(activity_path, sep='\t', index=False)
        print(f"Wrote {split_name}: {len(records)} sequences -> {fasta_path}")

    print(f"\nDone. Set dataset_path in your YAML config to: {args.output_dir}")


if __name__ == "__main__":
    main()
