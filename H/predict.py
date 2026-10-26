# -*- coding: utf-8 -*-
"""Predict Dev/Hk activity for sequences in a FASTA or plain-text file.

    python predict.py --input seqs.fa --output pred.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepstarr import DeepSTARRModel, read_sequences  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="FASTA or one sequence per line")
    ap.add_argument("--output", required=True, help="output TSV")
    ap.add_argument("--no-tta", action="store_true",
                    help="skip reverse-complement averaging (2x faster, -0.006 PCC)")
    ap.add_argument("--calibrated", action="store_true",
                    help="apply the affine calibration fitted on validation")
    ap.add_argument("--fit-length", action="store_true",
                    help="centre-crop or N-pad to 249 nt instead of failing")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()

    ids, seqs = read_sequences(a.input)
    model = DeepSTARRModel(device=a.device, batch_size=a.batch_size)
    print(model.info(), file=sys.stderr)
    try:
        p = model.predict(seqs, rc_tta=not a.no_tta, calibrated=a.calibrated,
                          fit=a.fit_length, ids=ids)
    except (ValueError, TypeError, FileNotFoundError) as e:
        raise SystemExit(f"\nERROR: {e}")

    with open(a.output, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["ID", "Dev", "Hk"])
        for i, sid in enumerate(ids):
            w.writerow([sid, f"{p['Dev'][i]:.6f}", f"{p['Hk'][i]:.6f}"])
    print(f"wrote {len(ids)} predictions to {a.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
