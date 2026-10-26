# -*- coding: utf-8 -*-
"""DeepSTARR enhancer activity predictor (HypBaza).

    from deepstarr import DeepSTARRModel
    model = DeepSTARRModel()
    out = model.predict(["ACGT..."])      # {"Dev": array, "Hk": array}
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
MODEL_DIR = ROOT / "model"

SEQ_LEN = 249
TARGETS = ("Dev", "Hk")
_VALID = set("ACGTN")
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _load_src():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import dna
    import models

    for mod in (models, dna):
        origin = Path(mod.__file__).resolve()
        if SRC not in origin.parents:
            raise ImportError(
                f"'{mod.__name__}' loaded from {origin}, not {SRC}. Another "
                f"module with that name was imported first in this process.")
    return models.build_model, dna


@contextlib.contextmanager
def _full_precision(enabled: bool = True):
    """Disable TF32: the quantiser snaps to a grid, so ~1e-3 noise flips codes."""
    if not enabled:
        yield
        return
    prev = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prev


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def read_fasta(path) -> Tuple[List[str], List[str]]:
    ids, seqs, current = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if ids:
                    seqs.append("".join(current))
                ids.append(line[1:].split()[0] if len(line) > 1 else f"seq_{len(ids) + 1}")
                current = []
            else:
                if not ids:
                    raise ValueError(f"{path}: sequence before first '>' header")
                current.append(line)
    if ids:
        seqs.append("".join(current))
    return ids, seqs


def read_sequences(path) -> Tuple[List[str], List[str]]:
    """FASTA, or plain text with one sequence per line."""
    with open(path, encoding="utf-8") as fh:
        first = fh.read(1)
    if first == ">":
        return read_fasta(path)
    with open(path, encoding="utf-8") as fh:
        seqs = [l.strip() for l in fh if l.strip()]
    return [f"seq_{i + 1}" for i in range(len(seqs))], seqs


def fit_length(seq: str) -> str:
    """Centre-crop or pad with N to SEQ_LEN."""
    n = len(seq)
    if n == SEQ_LEN:
        return seq
    if n > SEQ_LEN:
        start = (n - SEQ_LEN) // 2
        return seq[start:start + SEQ_LEN]
    pad = SEQ_LEN - n
    left = pad // 2
    return "N" * left + seq + "N" * (pad - left)


class DeepSTARRModel:
    """Predicts Dev and Hk enhancer activity (log2 enrichment) from 249 nt DNA.

    Parameters
    ----------
    device : "cuda", "cpu" or None (auto).
    batch_size : sequences per forward pass; does not change results.
    full_precision : run in FP32 (default). False uses TF32 on GPU, which is
        the training-time numerics but makes results hardware-dependent to ~1e-2.
    """

    def __init__(self, device: Optional[str] = None, batch_size: int = 256,
                 model_dir=None, full_precision: bool = True):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.full_precision = bool(full_precision)

        build_model, self._dna = _load_src()

        with open(self.model_dir / "config.yaml", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)

        torch.manual_seed(int(self.config.get("seed", 7898)))
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model(self.config)

        state = torch.load(self.model_dir / "weights.pt", map_location="cpu",
                           weights_only=True)
        model.load_state_dict(state.get("model", state), strict=True)
        model.to(self.device).eval()
        model.current_epoch = 10 ** 6
        self.model = model

        self.epoch = state.get("epoch")
        self.val_score = state.get("score")
        self.n_params = sum(p.numel() for p in model.parameters())
        path = self.model_dir / "calibration.json"
        self.calibration = json.loads(path.read_text(encoding="utf-8")) \
            if path.exists() else None

    def encode(self, seqs: Sequence[str]) -> torch.Tensor:
        """One-hot (N, 4, 249); channel order is A, C, T, G."""
        return torch.from_numpy(self._dna.one_hot_encode_dna(list(seqs)))

    def check(self, seqs: Sequence[str], ids=None, fit: bool = False) -> List[str]:
        if isinstance(seqs, str):
            raise TypeError("Pass a list of sequences, not a single string.")
        ids = list(ids) if ids is not None else [f"seq_{i + 1}" for i in range(len(seqs))]
        out, bad_len, bad_chr = [], [], []
        for sid, s in zip(ids, seqs):
            s = s.strip().upper()
            unknown = set(s) - _VALID
            if unknown:
                bad_chr.append(f"{sid} ({''.join(sorted(unknown))})")
            if len(s) != SEQ_LEN:
                if fit:
                    s = fit_length(s)
                else:
                    bad_len.append(f"{sid} ({len(s)} nt)")
            out.append(s)
        if bad_chr:
            raise ValueError("Invalid characters (allowed A/C/G/T/N): "
                             + ", ".join(bad_chr[:10]) + (" ..." if len(bad_chr) > 10 else ""))
        if bad_len:
            raise ValueError(
                f"Model takes {SEQ_LEN} nt sequences only. Wrong length: "
                + ", ".join(bad_len[:10]) + (" ..." if len(bad_len) > 10 else "")
                + ". Use fit=True to centre-crop or pad with N.")
        n_frac = np.array([s.count("N") / SEQ_LEN for s in out])
        if (n_frac > 0.05).any():
            warnings.warn(f"{int((n_frac > 0.05).sum())} sequences have >5% N; "
                          "the model saw none such in training.")
        return out

    @torch.no_grad()
    def _run(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        dev, hk = [], []
        with _full_precision(self.full_precision):
            for i in range(0, x.shape[0], self.batch_size):
                d, h = self.model(x[i:i + self.batch_size].to(self.device))
                dev.append(d.float().reshape(-1).cpu())
                hk.append(h.float().reshape(-1).cpu())
        return (torch.cat(dev).numpy().astype(np.float64),
                torch.cat(hk).numpy().astype(np.float64))

    def predict(self, seqs: Sequence[str], rc_tta: bool = True,
                calibrated: bool = False, fit: bool = False,
                ids=None) -> Dict[str, np.ndarray]:
        """Predicted activity in log2 enrichment.

        rc_tta : average the prediction over both strands (+0.006 PCC).
        calibrated : apply the affine correction fitted on validation; lowers
            MSE, leaves PCC and Spearman unchanged.
        """
        seqs = self.check(seqs, ids, fit)
        if not seqs:
            return {t: np.zeros(0) for t in TARGETS}
        x = self.encode(seqs)
        dev, hk = self._run(x)
        if rc_tta:
            dev_rc, hk_rc = self._run(self._dna.reverse_complement_tensor(x))
            dev, hk = 0.5 * (dev + dev_rc), 0.5 * (hk + hk_rc)
        out = {"Dev": dev, "Hk": hk}
        if calibrated:
            if self.calibration is None:
                raise FileNotFoundError("model/calibration.json is missing.")
            for t in TARGETS:
                c = self.calibration[t]
                out[t] = c["a"] + c["b"] * out[t]
        return out

    def predict_file(self, path, **kwargs) -> Tuple[List[str], Dict[str, np.ndarray]]:
        ids, seqs = read_sequences(path)
        return ids, self.predict(seqs, ids=ids, **kwargs)

    def mutagenesis(self, seq: str, rc_tta: bool = True) -> Dict[str, np.ndarray]:
        """In-silico saturation mutagenesis.

        Returns (249, 4) matrices in A, C, G, T order holding
        predict(variant) - predict(reference); zero at the reference base.
        """
        ref = self.check([seq])[0]
        bases = "ACGT"
        variants, where = [ref], []
        for pos, ref_base in enumerate(ref):
            for j, b in enumerate(bases):
                if b != ref_base:
                    variants.append(ref[:pos] + b + ref[pos + 1:])
                    where.append((pos, j))
        p = self.predict(variants, rc_tta=rc_tta)
        out = {}
        for t in TARGETS:
            m = np.zeros((SEQ_LEN, 4))
            for (pos, j), v in zip(where, p[t][1:]):
                m[pos, j] = v - p[t][0]
            out[t] = m
        return out

    def info(self) -> str:
        return (f"HypBaza | epoch {self.epoch} | val mean PCC {self.val_score:.5f} | "
                f"{self.n_params} params | {self.device} | "
                f"{'FP32' if self.full_precision else 'TF32'} | calibration: "
                f"{'yes' if self.calibration else 'no'}")


__all__ = ["DeepSTARRModel", "read_fasta", "read_sequences", "reverse_complement",
           "fit_length", "SEQ_LEN", "TARGETS"]
