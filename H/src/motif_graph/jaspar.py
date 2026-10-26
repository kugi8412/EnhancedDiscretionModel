"""JASPAR position-frequency matrices as motif-graph nodes.

The repository ships the complete JASPAR 2026 CORE *insects* collection, which
is entirely *Drosophila melanogaster* (every profile carries NCBI taxon 7227):

``data/motifs/JASPAR2026_CORE_Drosophila_melanogaster_redundant.jaspar``
    357 profiles — every version/variant, so one TF may appear several times.
``data/motifs/JASPAR2026_CORE_Drosophila_melanogaster_non-redundant.jaspar``
    296 profiles — one representative per TF.  The sensible default node set.

Two details that are easy to get wrong and are handled here explicitly:

**Row order.**
    JASPAR files list rows as A, C, G, T.  This framework one-hot encodes DNA as
    **A, C, T, G** (:data:`dna.NUCLEOTIDE_ORDER`).  Every matrix is permuted on
    load; forgetting this silently swaps G and T in every motif.

**Reverse complement.**
    A PFM is reverse-complemented by reversing the column order *and* swapping
    the complementary rows — not by flipping the row axis (which is only an RC
    under A, C, G, T ordering, and is wrong under this framework's).  See
    :meth:`JasparMotif.reverse_complement`.
"""

from __future__ import annotations

import os
import re
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from dna import NUCLEOTIDE_ORDER

#: Row order used inside JASPAR flat files.
JASPAR_ROW_ORDER = "ACGT"

#: Permutation mapping JASPAR rows (ACGT) onto this framework's order (ACTG).
_JASPAR_TO_FRAMEWORK = [JASPAR_ROW_ORDER.index(b) for b in NUCLEOTIDE_ORDER]

#: Complement permutation within the framework's own row order.
_COMPLEMENT_ROWS = [NUCLEOTIDE_ORDER.index({"A": "T", "C": "G", "G": "C", "T": "A"}[b])
                    for b in NUCLEOTIDE_ORDER]

#: Uniform Drosophila background.  The D. melanogaster genome is ~42% GC, so a
#: uniform prior is a reasonable default; override via ``background=`` when a
#: dataset-specific composition is known.
DEFAULT_BACKGROUND = np.full(4, 0.25, dtype=np.float64)

_HEADER = re.compile(r"^>\s*(\S+)\s*(.*)$")
_ROW = re.compile(r"^\s*([ACGT])?\s*\[?\s*([0-9.eE+\-\s]+?)\s*\]?\s*$")

DATA_SUBDIR = os.path.join("data", "motifs")
DROSOPHILA_NON_REDUNDANT = "JASPAR2026_CORE_Drosophila_melanogaster_non-redundant.jaspar"
DROSOPHILA_REDUNDANT = "JASPAR2026_CORE_Drosophila_melanogaster_redundant.jaspar"


@dataclass
class JasparMotif:
    """One JASPAR profile, stored in this framework's A, C, T, G row order.

    Attributes
    ----------
    matrix_id : str
        JASPAR accession, e.g. ``MA0022.1``.
    name : str
        TF name, e.g. ``dl`` (Dorsal).
    pfm : np.ndarray, shape ``(4, W)``
        Position *frequency* matrix — raw observation counts.
    """

    matrix_id: str
    name: str
    pfm: np.ndarray = field(repr=False)

    def __post_init__(self):
        self.pfm = np.asarray(self.pfm, dtype=np.float64)
        if self.pfm.ndim != 2 or self.pfm.shape[0] != 4:
            raise ValueError(f"{self.matrix_id}: PFM must be (4, W), got {self.pfm.shape}")

    @property
    def width(self) -> int:
        return self.pfm.shape[1]

    @property
    def label(self) -> str:
        """``name.matrix_id`` — unique even when a TF has several profiles."""
        return f"{self.name}.{self.matrix_id}"

    # -- matrix transforms --------------------------------------------------

    def ppm(self, pseudocount: float = 0.25) -> np.ndarray:
        """Position *probability* matrix with a Laplace-style pseudocount."""
        counts = self.pfm + pseudocount
        return counts / counts.sum(axis=0, keepdims=True)

    def pwm(self, background=None, pseudocount: float = 0.25) -> np.ndarray:
        """Log-odds position weight matrix, ``log2(p / background)``.

        This is the matrix that gets convolved with one-hot DNA: because the
        input is one-hot, the convolution sums exactly one row per position, so
        the response is the motif's log-odds score for that window.
        """
        background = DEFAULT_BACKGROUND if background is None else np.asarray(background, dtype=np.float64)
        return np.log2(self.ppm(pseudocount) / background.reshape(4, 1))

    def reverse_complement(self) -> "JasparMotif":
        """Reverse-complement the profile.

        Reverses the column order and permutes rows through the complement map.
        Note this is *not* ``pfm[::-1, ::-1]``: that only happens to be an RC
        when rows are ordered A, C, G, T, and this framework orders them A, C,
        T, G (see :mod:`dna`).
        """
        return JasparMotif(self.matrix_id, self.name, self.pfm[_COMPLEMENT_ROWS][:, ::-1])

    # -- scoring ------------------------------------------------------------

    def score_range(self, background=None, pseudocount: float = 0.25):
        """``(min_score, max_score)`` attainable by this PWM over any window."""
        weights = self.pwm(background, pseudocount)
        return float(weights.min(axis=0).sum()), float(weights.max(axis=0).sum())

    def relative_threshold(self, fraction: float = 0.85, background=None,
                           pseudocount: float = 0.25) -> float:
        """Absolute score corresponding to a *relative* threshold.

        ``fraction`` is the standard MATCH/JASPAR "relative profile score":
        0 puts the cut at the worst possible window, 1 at the consensus.  0.8-0.9
        is the usual operating range; higher means fewer, stronger hits.
        """
        lo, hi = self.score_range(background, pseudocount)
        return lo + float(fraction) * (hi - lo)

    def information_content(self, pseudocount: float = 0.25) -> float:
        """Total information content in bits — a proxy for motif specificity.

        Long, degenerate matrices carry little information and mostly add noise
        as graph nodes; filter on this to keep the node set informative.
        """
        p = self.ppm(pseudocount)
        return float(np.sum(p * np.log2(p * 4.0 + 1e-12)))

    # -- string views -------------------------------------------------------

    def consensus(self, pseudocount: float = 0.25) -> str:
        """Single-letter consensus (most probable base per position)."""
        p = self.ppm(pseudocount)
        return "".join(NUCLEOTIDE_ORDER[i] for i in p.argmax(axis=0))

    def iupac_consensus(self, threshold: float = 0.25, pseudocount: float = 0.25) -> str:
        """Degenerate IUPAC consensus: every base above *threshold* is included.

        Useful for reporting and for feeding a PWM-derived motif into the plain
        IUPAC matching path, but note it discards the quantitative weighting —
        prefer PWM scanning when accuracy matters.
        """
        from .vocab import IUPAC

        reverse = {frozenset(v): k for k, v in IUPAC.items()}
        p = self.ppm(pseudocount)
        letters = []
        for col in p.T:
            bases = {NUCLEOTIDE_ORDER[i] for i, prob in enumerate(col) if prob >= threshold}
            if not bases:
                bases = {NUCLEOTIDE_ORDER[int(col.argmax())]}
            letters.append(reverse.get(frozenset(bases), "N"))
        return "".join(letters)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_jaspar(path: str) -> list[JasparMotif]:
    """Read a JASPAR flat file (``>ID name`` + four count rows) into motifs.

    Handles both the bracketed layout (``A  [ 1 4 2 ]``) and the bare four-line
    numeric layout; ``#`` comment lines are skipped.
    """
    motifs: list[JasparMotif] = []
    header: tuple[str, str] | None = None
    rows: list[list[float]] = []

    def flush():
        if header is None:
            return
        if len(rows) != 4:
            raise ValueError(f"{path}: motif {header[0]} has {len(rows)} rows, expected 4")
        widths = {len(r) for r in rows}
        if len(widths) != 1:
            raise ValueError(f"{path}: motif {header[0]} has ragged rows {sorted(widths)}")
        # JASPAR is A,C,G,T; permute into this framework's A,C,T,G.
        matrix = np.asarray(rows, dtype=np.float64)[_JASPAR_TO_FRAMEWORK]
        motifs.append(JasparMotif(header[0], header[1] or header[0], matrix))

    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith(">"):
                flush()
                match = _HEADER.match(line)
                header = (match.group(1), match.group(2).strip()) if match else (line[1:].strip(), "")
                rows = []
                continue
            match = _ROW.match(line)
            if not match:
                continue
            values = [float(v) for v in match.group(2).split()]
            if values:
                rows.append(values)
    flush()

    if not motifs:
        raise ValueError(f"No JASPAR matrices parsed from {path!r}")
    return motifs


# ---------------------------------------------------------------------------
# Loading / downloading
# ---------------------------------------------------------------------------

def bundled_path(filename: str = DROSOPHILA_NON_REDUNDANT) -> str:
    """Absolute path of a motif file shipped in ``data/motifs/``."""
    from utils import PROJECT_ROOT
    return os.path.join(PROJECT_ROOT, DATA_SUBDIR, filename)


def load_drosophila(redundant: bool = False, min_information: float = 0.0,
                    max_width: int | None = None,
                    names: list[str] | None = None) -> list[JasparMotif]:
    """Load the bundled JASPAR 2026 CORE *D. melanogaster* profiles.

    Parameters
    ----------
    redundant : bool
        ``True`` loads all 357 profiles (several per TF), ``False`` the 296
        non-redundant representatives.  The redundant set deliberately contains
        near-duplicate matrices that hit the same positions — informative only
        if you want the overlap structure between TF variants, otherwise it just
        inflates the node count.
    min_information : float
        Drop profiles whose total information content is below this many bits.
        Long degenerate matrices match almost everywhere and add noise.
    max_width : int, optional
        Drop profiles wider than this.
    names : list of str, optional
        Keep only these TF names (case-insensitive) or matrix IDs.

    Returns
    -------
    list[JasparMotif]
    """
    path = bundled_path(DROSOPHILA_REDUNDANT if redundant else DROSOPHILA_NON_REDUNDANT)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Bundled JASPAR file missing: {path}. Re-download with "
            f"motif_graph.jaspar.download_jaspar().")
    motifs = parse_jaspar(path)

    if names:
        wanted = {n.lower() for n in names}
        motifs = [m for m in motifs
                  if m.name.lower() in wanted or m.matrix_id.lower() in wanted]
    if min_information > 0:
        motifs = [m for m in motifs if m.information_content() >= min_information]
    if max_width:
        motifs = [m for m in motifs if m.width <= max_width]

    if not motifs:
        raise ValueError("All JASPAR motifs were filtered out; relax the filters.")
    return motifs


def download_jaspar(release: int = 2026, collection: str = "CORE",
                    taxonomic_group: str = "insects", redundant: bool = False,
                    dest: str | None = None, timeout: int = 120) -> str:
    """Fetch a JASPAR bulk PFM file and write it to ``data/motifs/``.

    Only needed to refresh the bundled files or to pull a different taxonomic
    group; the Drosophila sets are already shipped with the repository.

    Returns
    -------
    str
        Path of the written file.
    """
    kind = "redundant" if redundant else "non-redundant"
    url = (f"https://jaspar.elixir.no/download/data/{release}/{collection}/"
           f"JASPAR{release}_{collection}_{taxonomic_group}_{kind}_pfms_jaspar.txt")
    if dest is None:
        from utils import PROJECT_ROOT
        dest = os.path.join(PROJECT_ROOT, DATA_SUBDIR,
                            f"JASPAR{release}_{collection}_{taxonomic_group}_{kind}.jaspar")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)

    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# Downloaded from {url}\n")
        fh.write(payload)
    return dest


def summarise(motifs: list[JasparMotif]) -> str:
    """One-line summary of a motif collection, for logs."""
    widths = [m.width for m in motifs]
    info = [m.information_content() for m in motifs]
    tfs = {m.name for m in motifs}
    return (f"{len(motifs)} JASPAR profiles over {len(tfs)} TFs | "
            f"width {min(widths)}-{max(widths)} (median {int(np.median(widths))}) | "
            f"information {min(info):.1f}-{max(info):.1f} bits "
            f"(median {np.median(info):.1f})")
