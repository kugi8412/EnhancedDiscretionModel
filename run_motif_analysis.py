#!/usr/bin/env python3
"""
Integration bridge: run Drosophila motif analysis on the training dataset.

Copies the standard train/val/test FASTA and activity files from the main
dataset directory into the motifs-pipeline data directory, then runs
analyze_input_motifs.py to detect enriched motifs in high-activity sequences.

Usage
-----
# From src/py/ (the main working directory)
python run_motif_analysis.py \
    --data_dir  ../../data/deepSTARR \
    --motifs_code_dir motifs-pipeline/code \
    --motifs_data_dir motifs-pipeline/data/deepstarr_sequences \
    --result_dir motifs-pipeline/results/motif_analysis/input_enrichment

# With a custom high-expression threshold:
python run_motif_analysis.py \
    --data_dir  ../../data/deepSTARR \
    --high_exp_thresh 7.5
"""

import argparse
import os
import shutil
import subprocess
import sys


SPLITS = ["Train", "Val", "Test"]


def sync_data(src_dir: str, dst_dir: str) -> None:
    """
    Copy Sequences_{split}.fa and Sequences_activity_{split}.txt from src_dir
    to dst_dir for all three splits.  Only copies when the source is newer
    than the destination (mtime-based) to avoid redundant work.
    """
    os.makedirs(dst_dir, exist_ok=True)
    missing = []

    for split in SPLITS:
        for filename in (f"Sequences_{split}.fa",
                         f"Sequences_activity_{split}.txt"):
            src = os.path.join(src_dir, filename)
            dst = os.path.join(dst_dir, filename)

            if not os.path.exists(src):
                missing.append(src)
                continue

            src_mtime = os.path.getmtime(src)
            dst_mtime = os.path.getmtime(dst) if os.path.exists(dst) else 0

            if src_mtime > dst_mtime:
                shutil.copy2(src, dst)
                print(f"  Synced  {filename}")
            else:
                print(f"  Up-to-date  {filename}")

    if missing:
        print("WARNING: The following source files are missing:")
        for p in missing:
            print(f"  {p}")
        print("Run prepare_filtered_data.py first or point --data_dir at the "
              "correct dataset directory.")


def run_analysis(code_dir: str, data_dir: str, result_dir: str,
                 high_exp_thresh: float) -> None:
    """
    Execute analyze_input_motifs.py from within motifs_code_dir so that its
    relative imports (utils/, configs/) resolve correctly.
    """
    script = os.path.join(code_dir, "analyze_input_motifs.py")
    if not os.path.exists(script):
        print(f"ERROR: Analysis script not found: {script}")
        sys.exit(1)

    # Temporarily patch data_dir and high_exp_thresh via env so the script
    # can be kept clean.  The script currently uses hardcoded defaults; provide
    # them as env overrides when non-default values are requested.
    env = os.environ.copy()
    env["MOTIF_DATA_DIR"]      = os.path.abspath(data_dir)
    env["MOTIF_RESULT_DIR"]    = os.path.abspath(result_dir)
    env["MOTIF_HIGH_EXP_THRESH"] = str(high_exp_thresh)

    print(f"\nRunning motif analysis (high_exp_thresh={high_exp_thresh})...")
    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.abspath(code_dir),
        env=env,
    )
    if result.returncode != 0:
        print("ERROR: analyze_input_motifs.py exited with non-zero status.")
        sys.exit(result.returncode)
    print("Motif analysis complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Bridge: sync dataset and run Drosophila motif enrichment analysis"
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join("..", "..", "data", "deepSTARR"),
        help="Source dataset directory containing Sequences_{split}.fa files "
             "(default: ../../data/deepSTARR)",
    )
    parser.add_argument(
        "--motifs_code_dir",
        default=os.path.join("motifs-pipeline", "code"),
        help="Path to motifs-pipeline/code/ (default: motifs-pipeline/code)",
    )
    parser.add_argument(
        "--motifs_data_dir",
        default=os.path.join("motifs-pipeline", "data", "deepstarr_sequences"),
        help="Destination data directory used by the motifs pipeline "
             "(default: motifs-pipeline/data/deepstarr_sequences)",
    )
    parser.add_argument(
        "--result_dir",
        default=os.path.join("motifs-pipeline", "results",
                             "motif_analysis", "input_enrichment"),
        help="Where motif enrichment TSVs will be written",
    )
    parser.add_argument(
        "--high_exp_thresh",
        type=float,
        default=7.0,
        help="log2-enrichment threshold for 'high activity' group (default: 7.0)",
    )
    parser.add_argument(
        "--skip_sync",
        action="store_true",
        help="Skip the data-sync step (use existing files in motifs_data_dir)",
    )
    args = parser.parse_args()

    # Step 1: sync data files
    if not args.skip_sync:
        print(f"Syncing dataset files from {args.data_dir} -> {args.motifs_data_dir}")
        sync_data(args.data_dir, args.motifs_data_dir)
    else:
        print("Skipping data sync (--skip_sync).")

    # Step 2: run enrichment analysis
    run_analysis(
        code_dir=args.motifs_code_dir,
        data_dir=args.motifs_data_dir,
        result_dir=args.result_dir,
        high_exp_thresh=args.high_exp_thresh,
    )


if __name__ == "__main__":
    main()
