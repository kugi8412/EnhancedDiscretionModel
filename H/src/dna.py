# -*- coding: utf-8 -*-
"""Canonical DNA encoding constants and strand operations.
"""

import numpy as np
import torch

NUCLEOTIDE_ORDER = 'ACTG'
COMPLEMENT_BASE = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

#: Channel permutation implementing complementation; ``[2, 3, 0, 1]`` for 'ACTG'.
COMPLEMENT_PERMUTATION = [NUCLEOTIDE_ORDER.index(COMPLEMENT_BASE[b]) for b in NUCLEOTIDE_ORDER]

ONE_HOT_MAPPING = {
    base: [1 if i == idx else 0 for i in range(len(NUCLEOTIDE_ORDER))]
    for idx, base in enumerate(NUCLEOTIDE_ORDER)
}
ONE_HOT_MAPPING['N'] = [0] * len(NUCLEOTIDE_ORDER)

from shape_spec import SHAPE_FEATURES, SHAPE_RC_ANTISYMMETRIC  # noqa: E402,F401


def shape_rc_signs(n_channels, device=None, dtype=None):
    """``[1, n_channels, 1]`` of +1 / -1 for a reverse-complement flip."""
    signs = [-1.0 if f in SHAPE_RC_ANTISYMMETRIC else 1.0
             for f in SHAPE_FEATURES[:n_channels]]
    return torch.tensor(signs, device=device, dtype=dtype).reshape(1, -1, 1)


def reverse_complement_with_shape(x, n_seq=None):
    n_seq = int(n_seq or len(NUCLEOTIDE_ORDER))
    if x.shape[1] <= n_seq:
        return reverse_complement_tensor(x)
    seq = reverse_complement_tensor(x[:, :n_seq])
    shape = x[:, n_seq:].flip(-1)
    signs = shape_rc_signs(shape.shape[1], device=x.device, dtype=x.dtype)
    return torch.cat([seq, shape * signs], dim=1)


def get_reverse_complement(seq: str) -> str:
    """Return the reverse-complement of a DNA string."""
    trans = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
    return seq.translate(trans)[::-1]


def reverse_complement_tensor(x):
    """Reverse-complement a one-hot encoded tensor of sequences.

    Complements the channel axis according to :data:`NUCLEOTIDE_ORDER` and
    reverses the position axis.  Accepts ``(A, L)`` and ``(B, A, L)`` tensors,
    is differentiable, and is safe for soft (non-binary) encodings produced by
    noise augmentations or Gumbel-Softmax decoders.

    Parameters
    ----------
    x : torch.Tensor
        One-hot (or soft one-hot) sequences with a channel axis of size 4.

    Returns
    -------
    torch.Tensor
        Reverse-complemented sequences, same shape and dtype as *x*.
    """
    n_channels = len(NUCLEOTIDE_ORDER)
    if x.dim() == 2:
        if x.shape[0] != n_channels:
            raise ValueError(
                f"Expected channel axis of size {n_channels}, got shape {tuple(x.shape)}")
        return x[COMPLEMENT_PERMUTATION, :].flip(-1)
    if x.dim() == 3:
        if x.shape[1] != n_channels:
            raise ValueError(
                f"Expected channel axis of size {n_channels}, got shape {tuple(x.shape)}")
        return x[:, COMPLEMENT_PERMUTATION, :].flip(-1)
    raise ValueError(
        f"Expected a 2D (A, L) or 3D (B, A, L) tensor, got shape {tuple(x.shape)}")


def one_hot_encode_dna(sequences):
    mapping = dict(ONE_HOT_MAPPING)
    mapping.update({k.lower(): v for k, v in mapping.items()})
    unknown = [0] * len(NUCLEOTIDE_ORDER)

    lengths = {len(seq) for seq in sequences}
    if len(lengths) > 1:
        raise ValueError(
            f"Sequences have inconsistent lengths {sorted(lengths)}; "
            "one-hot encoding requires a uniform length.")

    encoded = np.array(
        [[mapping.get(base, unknown) for base in seq] for seq in sequences],
        dtype=np.float32)
    return encoded.transpose(0, 2, 1)


__all__ = [
    'NUCLEOTIDE_ORDER', 'COMPLEMENT_BASE', 'COMPLEMENT_PERMUTATION', 'ONE_HOT_MAPPING',
    'get_reverse_complement', 'reverse_complement_tensor', 'one_hot_encode_dna',
]
