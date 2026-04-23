#!/usr/bin/env python3
"""
Bridge between filtering-seq-pipeline and ML training framework.

Takes the cleaned/split FASTA files produced by the filtering pipeline
(split_datasets_by_clusters.py) and prepares them into the format expected
by train.py: Sequences_{Split}.fa + Sequences_activity_{Split}.txt.

Preferred usage (cluster-safe 3-way split already done upstream):
    python prepare_filtered_data.py \
        --train_fasta path/to/train.fasta \
        --val_fasta   path/to/val.fasta \
        --test_fasta  path/to/test.fasta \
        --activity    path/to/Sequences_activity_All.txt \
        --output_dir  ../../data/deepSTARR_filtered

Legacy usage (val carved randomly from train – not recommended for
  production; use split_datasets_by_clusters.py --val_out instead):
    python prepare_filtered_data.py \
        --train_fasta path/to/train.fasta \
        --test_fasta  path/to/test.fasta \
        --activity    path/to/Sequences_activity_All.txt \
        --output_dir  ../../data/deepSTARR_filtered \
        --val_ratio   0.1

A post-split sanity check is always run to detect exact and
reverse-complement (RC) overlaps across the three output splits.
The script exits with code 1 if leakage is found.
"""

import os
import sys
import argparse
import warnings
from collections import defaultdict

import pandas as pd
from Bio import SeqIO


# ---------------------------------------------------------------------------
# RC helper
# ---------------------------------------------------------------------------

def _rc(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1]


# ---------------------------------------------------------------------------
# Leakage sanity check
# ---------------------------------------------------------------------------

def _check_overlap(records_a: list, records_b: list,
                   name_a: str, name_b: str) -> list:
    """Return (id_a, id_b, kind) tuples for exact and RC overlaps."""
    seq_to_id_b: dict = defaultdict(list)
    for rec in records_b:
        seq_to_id_b[str(rec.seq).upper()].append(rec.id)

    leaks = []
    for rec in records_a:
        seq = str(rec.seq).upper()
        # Exact
        for bid in seq_to_id_b.get(seq, []):
            leaks.append((rec.id, bid, "exact"))
        # RC
        rc_seq = _rc(seq)
        if rc_seq != seq:           # palindromes don't constitute leakage
            for bid in seq_to_id_b.get(rc_seq, []):
                leaks.append((rec.id, bid, "reverse_complement"))
    return leaks


def check_leakage(train_recs: list, val_recs: list, test_recs: list) -> bool:
    """Print a leakage report and return True if any overlap was found."""
    found = False
    for (recs_a, name_a), (recs_b, name_b) in [
        ((train_recs, "train"), (val_recs,  "val")),
        ((train_recs, "train"), (test_recs, "test")),
        ((val_recs,   "val"),   (test_recs, "test")),
    ]:
        leaks = _check_overlap(recs_a, recs_b, name_a, name_b)
        if leaks:
            found = True
            print(f"WARNING: {len(leaks)} overlapping sequences "
                  f"between {name_a} and {name_b}:")
            for aid, bid, kind in leaks[:10]:
                print(f"  [{kind}] {aid} <-> {bid}")
            if len(leaks) > 10:
                print(f"  ... and {len(leaks) - 10} more")
    return found


# ---------------------------------------------------------------------------
# Activity writer
# ---------------------------------------------------------------------------

def _write_split(split_name: str, records: list,
                 activity_df: pd.DataFrame, output_dir: str) -> None:
    fasta_path    = os.path.join(output_dir, f"Sequences_{split_name}.fa")
    activity_path = os.path.join(output_dir, f"Sequences_activity_{split_name}.txt")

    with open(fasta_path, "w") as f:
        SeqIO.write(records, f, "fasta")

    rows = []
    for rec in records:
        seq_id = rec.id
        if seq_id in activity_df.index:
            row = activity_df.loc[seq_id]
            rows.append({
                "Dev_log2_enrichment": row["Dev_log2_enrichment"],
                "Hk_log2_enrichment":  row["Hk_log2_enrichment"],
            })
        else:
            warnings.warn(f"No activity data for sequence {seq_id}, using 0.0")
            rows.append({"Dev_log2_enrichment": 0.0, "Hk_log2_enrichment": 0.0})

    pd.DataFrame(rows).to_csv(activity_path, sep="\t", index=False)
    print(f"Wrote {split_name}: {len(records)} sequences -> {fasta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert filtering pipeline output to ML training format"
    )
    parser.add_argument("--train_fasta", required=True,
                        help="Train FASTA from filtering pipeline")
    parser.add_argument("--val_fasta", default=None,
                        help="Val FASTA (preferred: produced by split_datasets_by_clusters.py). "
                             "If omitted, val is carved randomly from train using --val_ratio.")
    parser.add_argument("--test_fasta", required=True,
                        help="Test FASTA from filtering pipeline")
    parser.add_argument("--activity", required=True,
                        help="Full activity TSV (ID, Dev_log2_enrichment, Hk_log2_enrichment)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for ML-ready files")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="(Legacy) fraction of train to use as val when --val_fasta is absent "
                             "(default: 0.1). This split is NOT cluster-safe.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail_on_leakage", action="store_true",
                        help="Exit with code 1 if exact or RC overlaps are detected")
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load activity data keyed by sequence ID
    activity_df = pd.read_table(args.activity)
    if "ID" in activity_df.columns:
        activity_df = activity_df.set_index("ID")
    elif activity_df.columns[0] not in ("Dev_log2_enrichment", "Hk_log2_enrichment"):
        activity_df = activity_df.set_index(activity_df.columns[0])

    # Load FASTA records
    train_records = list(SeqIO.parse(args.train_fasta, "fasta"))
    test_records  = list(SeqIO.parse(args.test_fasta,  "fasta"))

    if args.val_fasta:
        val_records = list(SeqIO.parse(args.val_fasta, "fasta"))
        print(f"Loaded -> Train: {len(train_records)}, "
              f"Val: {len(val_records)}, Test: {len(test_records)}")
    else:
        warnings.warn(
            "--val_fasta not provided; carving val randomly from train. "
            "This is NOT cluster-safe. Use split_datasets_by_clusters.py "
            "--val_out for fully leakage-free splits.",
            UserWarning,
        )
        random.shuffle(train_records)
        n_val = int(len(train_records) * args.val_ratio)
        val_records   = train_records[:n_val]
        train_records = train_records[n_val:]
        print(f"After split -> Train: {len(train_records)}, "
              f"Val: {len(val_records)}, Test: {len(test_records)}")

    # Sanity check: exact + RC leakage across splits
    print("\nRunning cross-split leakage check (exact + reverse complement)...")
    leakage_found = check_leakage(train_records, val_records, test_records)
    if leakage_found:
        if args.fail_on_leakage:
            print("ERROR: Leakage detected. Aborting.")
            sys.exit(1)
        else:
            print("WARNING: Leakage detected. Consider using check_cross_split_overlaps.py "
                  "and re-running split_datasets_by_clusters.py to produce clean splits.")
    else:
        print("No exact or RC leakage detected.")

    # Write output files
    print()
    for split_name, records in [
        ("Train", train_records),
        ("Val",   val_records),
        ("Test",  test_records),
    ]:
        _write_split(split_name, records, activity_df, args.output_dir)

    print(f"\nDone. Set dataset_path in your YAML config to: {args.output_dir}")


if __name__ == "__main__":
    main()
