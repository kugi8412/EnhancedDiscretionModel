"""Motif / k-mer vocabulary for the graph models.

A :class:`MotifVocabulary` is an ordered list of *named patterns* — each either a
known TF motif written in IUPAC (``WGATAR``) or a plain k-mer (``GGGCCC``).  It
is the node set of every motif graph: node ``v`` of a sequence's graph is
vocabulary entry ``v``.

Two properties matter for the rest of the pipeline:

**Reverse-complement canonicalisation.**
    A transcription factor binds its site on either strand, so ``GGGCCC`` and its
    reverse complement describe the same biological event.  Entries are therefore
    stored canonically (the lexicographically smaller of pattern / RC-pattern)
    and matched on both strands, with the strand recorded as a node feature.
    This is what makes the graph representation approximately RC-invariant, and
    is why the reverse-complement consistency loss has something to hold on to.

**A vocabulary is data, not parameters.**
    Patterns compile to *fixed* convolution kernels, and the networks share one
    MLP across all nodes.  Swapping, adding or pruning entries therefore never
    changes a parameter shape — which is exactly what
    :class:`~models.motif_gnn.GraphSAE` relies on when it resamples motifs
    mid-training.
"""

from __future__ import annotations

import itertools
import os
from typing import Iterable, Sequence

import numpy as np
import torch  # noqa: F401  (kernels are torch tensors)

from dna import NUCLEOTIDE_ORDER, get_reverse_complement

# IUPAC ambiguity codes -> the concrete bases they stand for.
IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "AG", "Y": "CT", "S": "CG", "W": "AT",
    "K": "GT", "M": "AC", "B": "CGT", "D": "AGT",
    "H": "ACT", "V": "ACG", "N": "ACGT",
}

# Complement of each IUPAC code, so an ambiguous motif can be reverse-complemented.
IUPAC_COMPLEMENT = {
    "A": "T", "C": "G", "G": "C", "T": "A",
    "R": "Y", "Y": "R", "S": "S", "W": "W",
    "K": "M", "M": "K", "B": "V", "V": "B",
    "D": "H", "H": "D", "N": "N",
}

#: Canonical Drosophila / DeepSTARR motifs — a sensible default node set.
KNOWN_MOTIFS = {
    "Dref": "TATCGATA",
    "AP-1": "TGASTCA",
    "GATA": "WGATAR",
    "Ebox": "CANNTG",
    "Trl_GAGA": "GAGAGAG",
    "Inr": "TCAGTY",
    "DPE": "RGWYVT",
    "TATA": "TATAWAW",
    "SREBP": "RTCACGTGA",
    "Ohler1": "YGGTCACACTR",
    "Ohler6": "KTYRGTATWTTT",
    "CREB": "TGACGTCA",
    "Zelda": "CAGGTAG",
    "Twist": "CABATG",
    "Dorsal": "GGGWWWWCCM",
    "Snail": "CACCTG",
    "Bicoid": "TAATCC",
    "Hunchback": "TTTTTGGC",
    "Kruppel": "AAGGGTTAA",
    "Caudal": "TTTATGGC",
}


def reverse_complement_pattern(pattern: str) -> str:
    """Reverse-complement an IUPAC pattern (``WGATAR`` -> ``YTATCW``)."""
    try:
        return "".join(IUPAC_COMPLEMENT[b] for b in reversed(pattern.upper()))
    except KeyError as exc:  # pragma: no cover - guarded by validate()
        raise ValueError(f"Not an IUPAC code: {exc.args[0]!r} in {pattern!r}") from exc


def canonical_pattern(pattern: str) -> str:
    """The lexicographically smaller of *pattern* and its reverse complement."""
    pattern = pattern.upper()
    rc = reverse_complement_pattern(pattern)
    return min(pattern, rc)


def is_palindromic(pattern: str) -> bool:
    """True when a pattern equals its own reverse complement (e.g. ``TGACGTCA``)."""
    return pattern.upper() == reverse_complement_pattern(pattern)


class MotifVocabulary:
    """An ordered, named set of motifs forming the graph's node set.

    An entry is either an **IUPAC pattern** (``"WGATAR"``) or a **JASPAR
    position-frequency matrix** (:class:`~motif_graph.jaspar.JasparMotif`).  Both
    compile to a fixed convolution kernel plus a score threshold, so the
    featurizer treats them identically:

    * IUPAC -> a 0/1 kernel; a full match scores exactly the pattern length.
    * PFM   -> a log-odds PWM kernel; the threshold is a *relative* score
      (``pwm_threshold``), the standard MATCH-style cut between the worst
      possible window and the consensus.

    Parameters
    ----------
    entries : sequence
        Each item is ``(name, pattern)`` for IUPAC, ``(name, JasparMotif)``, or a
        bare :class:`JasparMotif` (its ``label`` becomes the name).  Duplicates
        (after RC-canonicalisation) are dropped, keeping the first occurrence.
    canonicalise_rc : bool
        Merge each motif with its reverse complement into a single node.
        Strongly recommended: it halves an all-k-mer vocabulary and makes the
        node set strand-symmetric, which is what lets the reverse-complement
        consistency loss hold.
    pwm_threshold : float
        Relative score threshold for PFM entries (0 = any window, 1 = consensus
        only).  0.8-0.9 is the usual range.
    """

    def __init__(self, entries: Sequence, canonicalise_rc: bool = True,
                 pwm_threshold: float = 0.85):
        self.canonicalise_rc = canonicalise_rc
        self.pwm_threshold = float(pwm_threshold)
        seen: set[str] = set()
        self.names: list[str] = []
        self.patterns: list[str] = []      # IUPAC string, or the PFM's consensus
        self.pwms: list[object | None] = []  # JasparMotif for PFM entries, else None

        for item in entries:
            name, spec = self._unpack(item)
            if spec is None:
                continue
            if isinstance(spec, str):
                pattern = spec.upper().strip()
                if not pattern:
                    continue
                bad = [c for c in pattern if c not in IUPAC]
                if bad:
                    raise ValueError(
                        f"Motif {name!r} = {pattern!r} has non-IUPAC characters {bad}")
                key = canonical_pattern(pattern) if canonicalise_rc else pattern
                pwm = None
            else:
                pattern = spec.consensus()
                # PFMs are keyed by identity, not consensus: two different
                # matrices can share a consensus while scoring very differently.
                key = f"pfm:{getattr(spec, 'matrix_id', name)}"
                pwm = spec

            if key in seen:
                continue
            seen.add(key)
            self.names.append(name)
            self.patterns.append(pattern)
            self.pwms.append(pwm)

        if not self.patterns:
            raise ValueError("MotifVocabulary is empty.")
        self.max_len = max(self.widths)

    @staticmethod
    def _unpack(item):
        """Normalise an entry into ``(name, spec)``."""
        if hasattr(item, "pfm"):                       # bare JasparMotif
            return item.label, item
        name, spec = item
        if hasattr(spec, "pfm"):
            return name, spec
        return name, spec

    # -- basic properties ---------------------------------------------------

    @property
    def widths(self) -> list[int]:
        """Width in bp of every entry."""
        return [p.width if p is not None else len(s)
                for s, p in zip(self.patterns, self.pwms)]

    @property
    def has_pwms(self) -> bool:
        return any(p is not None for p in self.pwms)

    # -- constructors -------------------------------------------------------

    @classmethod
    def known(cls, names: Iterable[str] | None = None, **kwargs) -> "MotifVocabulary":
        """Vocabulary from :data:`KNOWN_MOTIFS` (all of them unless *names* given)."""
        chosen = list(KNOWN_MOTIFS) if names is None else list(names)
        missing = [n for n in chosen if n not in KNOWN_MOTIFS]
        if missing:
            raise KeyError(f"Unknown motif name(s) {missing}. Available: {sorted(KNOWN_MOTIFS)}")
        return cls([(n, KNOWN_MOTIFS[n]) for n in chosen], **kwargs)

    @classmethod
    def kmers(cls, k: int = 6, **kwargs) -> "MotifVocabulary":
        """Vocabulary of every k-mer of length *k* (RC-canonicalised by default).

        ``k=6`` yields 2080 canonical 6-mers, which is far too many nodes for a
        dense graph — use it as the *candidate pool* for
        :class:`~models.motif_gnn.GraphSAE`, which samples a working subset.
        """
        entries = [("".join(p),) * 2 for p in itertools.product(NUCLEOTIDE_ORDER, repeat=k)]
        return cls(entries, **kwargs)

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "MotifVocabulary":
        """Read a vocabulary from a two-column ``name<TAB>pattern`` text file.

        Lines starting with ``#`` are comments.  A single-column line is taken as
        a pattern that also serves as its own name (handy for raw k-mer lists).
        """
        entries = []
        with open(path) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", "\t").split()
                if len(parts) == 1:
                    entries.append((parts[0], parts[0]))
                else:
                    entries.append((parts[0], parts[1]))
        if not entries:
            raise ValueError(f"No motifs found in {path!r}")
        return cls(entries, **kwargs)

    @classmethod
    def from_config(cls, cfg: dict | None) -> "MotifVocabulary":
        """Build from a YAML ``model.kwargs.vocab`` section.

        Recognised keys::

            vocab:
              known: true            # include KNOWN_MOTIFS (or a list of names)
              kmers: 5               # include every k-mer of this length
              file: motifs.txt       # include a name<TAB>pattern file
              extra:                 # inline additions
                MyMotif: GGGWWWCCC
              canonicalise_rc: true  # merge each pattern with its RC (default)

        With no keys at all it falls back to :data:`KNOWN_MOTIFS`.
        """
        cfg = dict(cfg or {})
        canonicalise_rc = bool(cfg.pop("canonicalise_rc", True))
        pwm_threshold = float(cfg.pop("pwm_threshold", 0.85))
        entries: list = []

        # ---- JASPAR 2026 CORE Drosophila melanogaster --------------------
        # jaspar: true            -> non-redundant set (296 profiles)
        # jaspar: {redundant: true, min_information: 8, max_width: 15,
        #          names: [dl, twi], top_n: 96}
        # NB: test against None/False explicitly. ``jaspar: {}`` means "all
        # JASPAR profiles with default options", but an empty dict is falsy, so
        # a truthiness test silently fell through to the 20 built-in motifs.
        jaspar_cfg = cfg.pop("jaspar", None)
        if jaspar_cfg is not None and jaspar_cfg is not False:
            from .jaspar import load_drosophila
            opts = {} if jaspar_cfg is True else dict(jaspar_cfg)
            top_n = int(opts.pop("top_n", 0) or 0)
            motifs = load_drosophila(
                redundant=bool(opts.pop("redundant", False)),
                min_information=float(opts.pop("min_information", 0.0)),
                max_width=opts.pop("max_width", None),
                names=opts.pop("names", None))
            if opts:
                raise ValueError(f"Unknown jaspar option(s): {sorted(opts)}")
            # Keep the most informative profiles when a node budget is given —
            # a dense graph costs O(V^2) memory, so V must stay modest.
            if top_n and top_n < len(motifs):
                motifs = sorted(motifs, key=lambda m: -m.information_content())[:top_n]
            entries += [(m.label, m) for m in motifs]

        known = cfg.pop("known", None)
        if known is True:
            entries += [(n, p) for n, p in KNOWN_MOTIFS.items()]
        elif isinstance(known, (list, tuple)):
            entries += [(n, KNOWN_MOTIFS[n]) for n in known]

        k = cfg.pop("kmers", None)
        if k:
            entries += [("".join(p),) * 2 for p in itertools.product(NUCLEOTIDE_ORDER, repeat=int(k))]

        path = cfg.pop("file", None)
        if path:
            from utils import resolve_path
            resolved = resolve_path(path)
            if not os.path.isfile(resolved):
                raise FileNotFoundError(f"Motif file not found: {path!r} (resolved to {resolved!r})")
            with open(resolved) as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.replace(",", "\t").split()
                    entries.append((parts[0], parts[1]) if len(parts) > 1 else (parts[0], parts[0]))

        extra = cfg.pop("extra", None) or {}
        entries += [(n, p) for n, p in extra.items()]

        if cfg:
            raise ValueError(f"Unknown vocab option(s): {sorted(cfg)}")
        if not entries:
            entries = [(n, p) for n, p in KNOWN_MOTIFS.items()]
        return cls(entries, canonicalise_rc=canonicalise_rc, pwm_threshold=pwm_threshold)

    # -- kernels ------------------------------------------------------------

    def match_kernels(self, dtype=torch.float32, min_match_frac: float = 1.0):
        """Compile the vocabulary into fixed convolution kernels and thresholds.

        Matching becomes a convolution against one-hot DNA:

        * **IUPAC** — encode "this base is accepted here" as 1.  The response
          then equals the pattern length exactly at a full match and less
          everywhere else, so the threshold is ``ceil(min_match_frac * length)``.
        * **PFM** — use the log-odds PWM.  Because the input is one-hot the
          convolution selects exactly one row per position, so the response *is*
          the motif's log-odds score; the threshold is the relative score
          :meth:`~motif_graph.jaspar.JasparMotif.relative_threshold`.

        Padding columns are all-zero and contribute nothing, letting motifs of
        different widths share one kernel width.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``kernels`` ``(2V, 4, max_len)`` — forward motifs first, then their
            reverse complements — ``thresholds`` ``(2V,)`` and ``widths``
            ``(2V,)``.
        """
        v = len(self.patterns)
        n_bases = len(NUCLEOTIDE_ORDER)
        kernels = torch.zeros(2 * v, n_bases, self.max_len, dtype=dtype)
        thresholds = torch.zeros(2 * v, dtype=dtype)
        widths = torch.zeros(2 * v, dtype=dtype)
        base_idx = {b: i for i, b in enumerate(NUCLEOTIDE_ORDER)}

        for i, (pattern, pwm) in enumerate(zip(self.patterns, self.pwms)):
            if pwm is None:
                width = len(pattern)
                cut = float(np.ceil(min_match_frac * width))
                for j, code in enumerate(pattern):
                    for base in IUPAC[code]:
                        kernels[i, base_idx[base], j] = 1.0
                rc = reverse_complement_pattern(pattern)
                for j, code in enumerate(rc):
                    for base in IUPAC[code]:
                        kernels[v + i, base_idx[base], j] = 1.0
            else:
                width = pwm.width
                cut = pwm.relative_threshold(self.pwm_threshold)
                fwd = torch.as_tensor(pwm.pwm(), dtype=dtype)
                rev = torch.as_tensor(pwm.reverse_complement().pwm(), dtype=dtype)
                kernels[i, :, :width] = fwd
                kernels[v + i, :, :width] = rev

            thresholds[i] = thresholds[v + i] = cut
            widths[i] = widths[v + i] = width

        return kernels, thresholds, widths

    def score_ranges(self, dtype=torch.float32):
        """Per-kernel ``(lo, hi)`` score bounds, aligned with :meth:`match_kernels`.

        These convert a raw convolution response into the MATCH-style *relative*
        profile score ``(score - lo) / (hi - lo)``, which lies in ``[0, 1]`` and
        is comparable across motifs.  Raw log-odds are not comparable: a 21-base
        profile spans a far wider range than a 6-base one, so an unnormalised
        occupancy track would let long motifs dominate purely by scale.

        A reverse complement contains the same weights in permuted order, so it
        shares its forward motif's bounds exactly.
        """
        v = len(self.patterns)
        lo = torch.zeros(2 * v, dtype=dtype)
        hi = torch.zeros(2 * v, dtype=dtype)
        for i, (pattern, pwm) in enumerate(zip(self.patterns, self.pwms)):
            if pwm is None:
                # IUPAC kernels answer "is this base accepted", so the response
                # runs from 0 to the pattern length.
                low, high = 0.0, float(len(pattern))
            else:
                low, high = pwm.score_range()
            lo[i] = lo[v + i] = low
            hi[i] = hi[v + i] = high
        return lo, hi

    # -- subsetting ---------------------------------------------------------

    def _entry(self, i: int):
        """``(name, spec)`` for index *i*, preserving PFMs."""
        return (self.names[i], self.pwms[i] if self.pwms[i] is not None else self.patterns[i])

    def subset(self, indices: Sequence[int]) -> "MotifVocabulary":
        """A new vocabulary containing only *indices*, in the order given."""
        return MotifVocabulary([self._entry(i) for i in indices],
                               canonicalise_rc=self.canonicalise_rc,
                               pwm_threshold=self.pwm_threshold)

    def without(self, indices: Iterable[int]) -> "MotifVocabulary":
        """A new vocabulary with *indices* removed (used by validation pruning)."""
        drop = set(int(i) for i in indices)
        keep = [i for i in range(len(self)) if i not in drop]
        if not keep:
            raise ValueError("Refusing to prune every motif from the vocabulary.")
        return self.subset(keep)

    def extended(self, entries: Sequence) -> "MotifVocabulary":
        """A new vocabulary with *entries* appended (duplicates dropped)."""
        return MotifVocabulary(
            [self._entry(i) for i in range(len(self))] + list(entries),
            canonicalise_rc=self.canonicalise_rc,
            pwm_threshold=self.pwm_threshold)

    # -- dunder / misc ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.patterns)

    def __iter__(self):
        return iter(zip(self.names, self.patterns))

    def __repr__(self) -> str:
        head = ", ".join(f"{n}:{p}" for n, p in list(self)[:4])
        more = f", +{len(self) - 4} more" if len(self) > 4 else ""
        kind = f", {sum(p is not None for p in self.pwms)} PWM" if self.has_pwms else ""
        return (f"MotifVocabulary({len(self)} motifs{kind}, max_len={self.max_len}: "
                f"{head}{more})")

    def to_records(self) -> list[dict]:
        """Serialisable description, e.g. for checkpoints and reports."""
        return [{"index": i, "name": n, "pattern": p, "width": w,
                 "kind": "pwm" if pwm is not None else "iupac",
                 "palindromic": is_palindromic(p) if pwm is None else None}
                for i, (n, p, pwm, w) in enumerate(
                    zip(self.names, self.patterns, self.pwms, self.widths))]

    def save(self, path: str) -> str:
        """Write a ``name<TAB>pattern`` file readable by :meth:`from_file`.

        PFM entries are written as their consensus string, which is lossy — the
        file records *which* motifs survived, not their full weight matrices.
        Re-derive the matrices from the JASPAR accession embedded in the name.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as fh:
            fh.write("# name\tpattern\tkind\n")
            for name, pattern, pwm in zip(self.names, self.patterns, self.pwms):
                fh.write(f"{name}\t{pattern}\t{'pwm' if pwm is not None else 'iupac'}\n")
        return path


def sample_vocabulary(pool: MotifVocabulary, size: int, seed: int = 42,
                      include: Sequence[str] | None = None) -> MotifVocabulary:
    """Draw a working vocabulary of *size* entries from a candidate *pool*.

    This is the "sample motifs a priori" step of :class:`GraphSAE`.  Names listed
    in *include* are always kept (use it to pin known TF motifs while the rest of
    the node budget is filled with random k-mers).
    """
    rng = np.random.default_rng(seed)
    forced = [i for i, n in enumerate(pool.names) if include and n in set(include)]
    if size <= len(forced):
        return pool.subset(forced[:size])

    remaining = [i for i in range(len(pool)) if i not in set(forced)]
    n_draw = min(size - len(forced), len(remaining))
    drawn = rng.choice(remaining, size=n_draw, replace=False) if n_draw else []
    return pool.subset(forced + sorted(int(i) for i in drawn))
