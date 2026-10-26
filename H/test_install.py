# -*- coding: utf-8 -*-

"""Check that this installation reproduces the author's predictions.
    python test_install.py [--device cpu]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deepstarr import TARGETS, DeepSTARRModel, read_fasta  # noqa: E402

TOLERANCE = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    ids, seqs = read_fasta(HERE / "examples" / "sequences.fa")
    with open(HERE / "examples" / "expected.tsv", encoding="utf-8") as fh:
        ref = {w["ID"]: w for w in csv.DictReader(fh, delimiter="\t")}

    model = DeepSTARRModel(device=a.device)
    print(model.info())

    p = model.predict(seqs)
    ok = True
    for t in TARGETS:
        expected = np.array([float(ref[s][t]) for s in ids])
        diff = float(np.abs(p[t] - expected).max())
        ok &= diff < TOLERANCE
        print(f"[{'OK' if diff < TOLERANCE else 'FAIL'}] {t}: max |pred - reference| "
              f"= {diff:.2e} (tolerance {TOLERANCE:.0e})")

    # eval-mode BatchNorm must not look at batch composition
    alone = {t: np.array([model.predict([s])[t][0] for s in seqs]) for t in TARGETS}
    diff = max(float(np.abs(alone[t] - p[t]).max()) for t in TARGETS)
    ok &= diff < TOLERANCE
    print(f"[{'OK' if diff < TOLERANCE else 'FAIL'}] one at a time vs batched: "
          f"max diff = {diff:.2e}")

    print("\nAll checks passed." if ok else "\nFAILED - do not use these results.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
