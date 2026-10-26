# -*- coding: utf-8 -*-
"""Usage example: prediction, calibration and saturation mutagenesis."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deepstarr import DeepSTARRModel, read_fasta  # noqa: E402

model = DeepSTARRModel()
print(model.info())

ids, seqs = read_fasta(HERE / "examples" / "sequences.fa")
p = model.predict(seqs)
pc = model.predict(seqs, calibrated=True)

print("\nID                                            Dev     Hk  |   cal_Dev cal_Hk")
for i in range(4):
    print(f"{ids[i]:<44} {p['Dev'][i]:6.3f} {p['Hk'][i]:6.3f}  | "
          f"{pc['Dev'][i]:8.3f} {pc['Hk'][i]:6.3f}")

effects = model.mutagenesis(seqs[0])
for t in ("Dev", "Hk"):
    m = effects[t]
    pos, base = np.unravel_index(np.abs(m).argmax(), m.shape)
    print(f"\n{t}: strongest variant: position {pos}, {seqs[0][pos]}->{'ACGT'[base]}, "
          f"{m[pos, base]:+.3f} log2")
    print(f"{t}: 10 most sensitive positions: "
          f"{sorted(np.argsort(np.abs(m).max(axis=1))[::-1][:10].tolist())}")
