from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dna import NUCLEOTIDE_ORDER
from .vocab import MotifVocabulary

NODE_FEATURE_NAMES = (
    "present", "count", "position", "spread",
    "strand_bias", "match_quality", "local_purine", "redundancy",
    "left_A", "left_C", "left_T", "left_G",
    "right_A", "right_C", "right_T", "right_G",
)

EDGE_FEATURE_NAMES = (
    "spacer_A", "spacer_C", "spacer_T", "spacer_G",
    "purine_distance", "pyrimidine_distance",
    "signed_distance", "distance",
    "helical_phase", "ry_transitions",
    "order", "same_strand", "overlap", "adjacent", "co_present",
)

NODE_DIM = len(NODE_FEATURE_NAMES)
EDGE_DIM = len(EDGE_FEATURE_NAMES)

#: B-DNA helical periodicity in base pairs.
HELICAL_PERIOD = 10.5


class MotifGraphFeaturizer(nn.Module):
    """One-hot ``[B, 4, L]`` -> ``(node_feats, edge_feats, adjacency)``.

    Parameters
    ----------
    vocab : MotifVocabulary
        Node set.  Swappable at runtime via :meth:`set_vocabulary` — nothing here
        holds parameters, so a resampled vocabulary needs no re-initialisation.
    min_match_frac : float
        Fraction of a pattern's positions that must match for a hit.  ``1.0``
        (default) means exact IUPAC matching; lower values admit near-matches,
        which helps for long motifs on small datasets.
    max_count : int
        Hit counts are normalised by this and clipped, so ``count`` stays in
        ``[0, 1]`` regardless of how repetitive a k-mer is.
    context : int
        Flank, in bp, used for the ``local_purine`` node feature.
    """

    def __init__(self, vocab: MotifVocabulary, min_match_frac: float = 1.0,
                 max_count: int = 8, context: int = 10, adjacent_gap: int = 3,
                 neighbourhood: str = "dense", radius_bp: int = 60, knn_k: int = 16,
                 multi_hits: int = 4, cooccurrence_bins=(0, 15, 40, 90, 250),
                 flank_bp: int = 8):
        super().__init__()
        if neighbourhood not in ("dense", "radius", "knn"):
            raise ValueError(f"neighbourhood must be dense|radius|knn, got {neighbourhood!r}")
        self.min_match_frac = float(min_match_frac)
        self.max_count = int(max_count)
        self.context = int(context)
        self.adjacent_gap = int(adjacent_gap)
        self.neighbourhood = neighbourhood
        # >1 turns on the co-occurrence distance profile: every hit of motif i
        # against every hit of motif j, binned. 1 restores single-representative
        # edges.
        self.multi_hits = int(multi_hits)
        self.cooccurrence_bins = tuple(cooccurrence_bins)
        # Width of the flanking window summarised per node.
        self.flank_bp = int(flank_bp)
        self.radius_bp = int(radius_bp)
        self.knn_k = int(knn_k)
        self.node_dim = NODE_DIM
        self.edge_dim = EDGE_DIM
        self.sparse_edge_dim = EDGE_DIM + (
            len(self.cooccurrence_bins) - 1 if self.multi_hits > 1 else 0)
        self.set_vocabulary(vocab)


    def set_vocabulary(self, vocab: MotifVocabulary):
        """Install a new node set (used by GraphSAE resampling/pruning).

        Thresholds come from the vocabulary itself, because IUPAC patterns and
        PWMs are cut on different scales (match count vs. log-odds score).
        """
        self.vocab = vocab
        kernels, thresholds, widths = vocab.match_kernels(min_match_frac=self.min_match_frac)
        device = getattr(self, "kernels", torch.zeros(1)).device
        self.register_buffer("kernels", kernels.to(device), persistent=False)
        self.register_buffer("thresholds", thresholds.to(device), persistent=False)
        self.register_buffer("widths", widths.to(device), persistent=False)
        # Score attained by a perfect match, used to normalise ``match_quality``
        # onto a common scale across IUPAC and PWM entries.
        best = kernels.clamp(min=0).amax(dim=1).sum(dim=-1).clamp(min=1e-6)
        self.register_buffer("best_scores", best.to(device), persistent=False)
        self.num_motifs = len(vocab)
        self.max_len = vocab.max_len
        return self

    def memory_estimate(self, batch_size: int, dtype_bytes: int = 4) -> float:
        """Megabytes the dense edge tensor needs for one batch.

        The graph is dense, so cost grows as ``V^2``: 296 JASPAR motifs at batch
        128 needs ~500 MB.  Filter the vocabulary (``top_n``, ``min_information``)
        or lower the batch size when this gets large.
        """
        v = self.num_motifs
        return batch_size * v * v * (self.edge_dim + 1) * dtype_bytes / 1e6

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _cumulative(values: torch.Tensor) -> torch.Tensor:
        """Prefix sums with a leading zero: ``out[i] = values[:i].sum()``."""
        return F.pad(values.cumsum(dim=-1), (1, 0))

    @staticmethod
    def _range_sum(cum: torch.Tensor, start: torch.Tensor, end: torch.Tensor):
        """Sum over ``[start, end)`` and the width of that window.

        *start*/*end* are integer tensors of any shape sharing a batch dim with
        *cum* ``[B, L+1]``.  Returns ``(total, width)``; an empty range gives
        ``(0, 0)``.
        """
        b = cum.shape[0]
        limit = cum.shape[1] - 1
        lo = start.clamp(0, limit)
        hi = end.clamp(0, limit)
        # An inverted range means "no window at all" and must yield 0, not the
        # window read backwards. Two overlapping motifs produce spacer_lo >
        # spacer_hi, and min/max-normalising that would silently count the motif
        # body itself as if it were the spacer between them.
        hi = torch.maximum(hi, lo)
        flat = lambda t: t.reshape(b, -1)
        total = torch.gather(cum, 1, flat(hi)) - torch.gather(cum, 1, flat(lo))
        width = (flat(hi) - flat(lo)).to(total.dtype)
        return total.reshape(start.shape), width.reshape(start.shape)

    @classmethod
    def _range_mean(cls, cum: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """Mean of the values summarised by *cum* over ``[start, end)``; 0 if empty."""
        total, width = cls._range_sum(cum, start, end)
        return total / width.clamp(min=1)

    # -- forward ------------------------------------------------------------

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """Featurise a one-hot batch.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, 4, L)``
            One-hot DNA in :data:`dna.NUCLEOTIDE_ORDER` (A, C, T, G).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``node_feats`` ``(B, V, 7)``, ``edge_feats`` ``(B, V, V, 7)`` and
            ``adjacency`` ``(B, V, V)``.
        """
        if x.dim() != 3 or x.shape[1] != len(NUCLEOTIDE_ORDER):
            raise ValueError(f"Expected (B, 4, L) one-hot, got {tuple(x.shape)}")
        b, _, length = x.shape
        v = self.num_motifs
        if length < self.max_len:
            raise ValueError(
                f"Sequence length {length} is shorter than the longest motif "
                f"({self.max_len}); drop long motifs or pad the sequences.")

        x = x.float()
        kernels = self.kernels.to(x.dtype)

        # --- scan both strands ------------------------------------------------
        # All motifs share one kernel width (the longest), so a bare convolution
        # would only offer start positions 0 .. L-max_len. A short motif sitting
        # in the final max_len-1 bases would then be invisible — and, since that
        # dead zone is at the 3' end, it would move to the *other* end under
        # reverse complement, silently breaking RC symmetry.
        #
        # Padding with zero columns restores every start position (padding
        # contributes nothing to the sum), and a per-motif validity mask then
        # removes the positions where a motif would run off the end and be
        # scored against padding rather than DNA.
        pad = self.max_len - 1
        response = F.conv1d(F.pad(x, (0, pad)), kernels)            # [B, 2V, L]
        positions = response.shape[-1]

        starts = torch.arange(positions, device=x.device).view(1, 1, -1)
        valid = starts <= (length - self.widths.view(1, -1, 1))      # [1, 2V, L]
        hits = ((response >= self.thresholds.view(1, -1, 1)) & valid).float()
        response = response.masked_fill(~valid, -1e9)

        fwd, rc = hits[:, :v], hits[:, v:]                          # [B, V, P]
        resp_fwd, resp_rc = response[:, :v], response[:, v:]
        either = torch.clamp(fwd + rc, max=1.0)

        # --- node features ----------------------------------------------------
        n_fwd = fwd.sum(-1)
        n_rc = rc.sum(-1)
        n_hits = either.sum(-1)                                     # [B, V]
        present = (n_hits > 0).float()
        count = (n_hits / self.max_count).clamp(max=1.0)
        strand_bias = (n_fwd - n_rc) / n_hits.clamp(min=1.0)

        # Representative position: the BEST-scoring hit on either strand, not the
        # first. With PWMs the hits differ in quality and the strongest site is
        # the one most likely to be bound.
        #
        # Ties need a *reverse-complement-symmetric* rule. Plain argmax returns
        # the first maximum, but "first" becomes "last" under RC, so a motif with
        # two equally good sites would report a different position on each strand
        # and every distance touching it would change. Preferring the hit nearest
        # the sequence centre is symmetric, because RC maps the centre onto
        # itself and preserves each site's distance to it.
        motif_width = self.widths[:v].view(1, -1, 1)
        best_strand = torch.maximum(resp_fwd, resp_rc)               # [B, V, P]
        max_score = best_strand.amax(dim=-1, keepdim=True)
        is_best = either.bool() & (best_strand >= max_score - 1e-6)

        start_grid = torch.arange(positions, device=x.device, dtype=x.dtype).view(1, 1, -1)
        hit_centre = start_grid + (motif_width - 1.0) / 2.0
        to_middle = (hit_centre - (length - 1) / 2.0).abs()
        ranked = torch.where(is_best, to_middle, torch.full_like(to_middle, float("inf")))
        best_hit = ranked.argmin(dim=-1)                             # [B, V]
        pos_idx = torch.where(present.bool(), best_hit,
                              torch.full_like(best_hit, positions // 2))

        # Per-motif geometry. Distances are measured between motif *centres*, not
        # start positions: under reverse complement a motif's start moves by its
        # own length, so start-to-start distance between two motifs of different
        # lengths is not RC-invariant, while centre-to-centre distance is exactly
        # invariant. That matters because the RC-consistency loss compares a
        # sequence's prediction with its reverse complement's.
        motif_len = self.widths[:v].view(1, -1)                     # [1, V]
        centre = pos_idx.float() + (motif_len - 1.0) / 2.0           # [B, V]
        end_idx = pos_idx.float() + motif_len                        # exclusive

        # Positional spread over all hits, in units of sequence length.
        grid = torch.arange(positions, device=x.device, dtype=x.dtype).view(1, 1, -1)
        denom = n_hits.clamp(min=1.0)
        mean_pos = (either * grid).sum(-1) / denom
        mean_sq = (either * grid.pow(2)).sum(-1) / denom
        spread = (mean_sq - mean_pos.pow(2)).clamp(min=0.0).sqrt() / max(length, 1)

        # How well the best occurrence matched, normalised by the score a perfect
        # match would get — comparable across IUPAC patterns and PWMs. Absent
        # motifs would otherwise report the -1e9 masking sentinel.
        best_resp = torch.maximum(resp_fwd.amax(-1), resp_rc.amax(-1))
        match_quality = (best_resp / self.best_scores[:v].view(1, -1)).clamp(-1.0, 1.0)
        match_quality = match_quality * present

        # Purine content around the representative hit.
        purine = x[:, NUCLEOTIDE_ORDER.index("A"), :] + x[:, NUCLEOTIDE_ORDER.index("G"), :]
        cum_purine = self._cumulative(purine)                       # [B, L+1]
        local_purine = self._range_mean(
            cum_purine,
            (pos_idx.float() - self.context).long(),
            (end_idx + self.context).long())

        # ``redundancy`` is appended once the overlap matrix exists (see below).
        node_parts = [
            present,
            count,
            pos_idx.float() / max(length - 1, 1),
            spread,
            strand_bias,
            match_quality,
            local_purine,
        ]

        # --- edge features ----------------------------------------------------
        ci = centre.unsqueeze(2)                                    # [B, V, 1]
        cj = centre.unsqueeze(1)                                    # [B, 1, V]
        delta = cj - ci                                             # signed, [B,V,V]
        gap = delta.abs()

        distance = gap / max(length, 1)
        helical_phase = torch.cos(2.0 * math.pi * gap / HELICAL_PERIOD)
        order = torch.sign(delta)

        # Spacer = the DNA strictly between the two occurrences: it runs from the
        # END of whichever motif comes first to the START of the other. Using each
        # motif's own length (rather than one global padding width) keeps the
        # window exact for vocabularies that mix short k-mers with long TF motifs.
        starts_i, starts_j = pos_idx.float().unsqueeze(2), pos_idx.float().unsqueeze(1)
        ends_i, ends_j = end_idx.unsqueeze(2), end_idx.unsqueeze(1)
        left_first = (delta >= 0)
        spacer_lo = torch.where(left_first, ends_i, ends_j).long()
        spacer_hi = torch.where(left_first, starts_j, starts_i).long()

        # Separation expressed in purines and pyrimidines. The two counts sum to
        # the base-pair gap, so together they carry the distance *and* what the
        # spacer is made of: 30 bp of 25 purines (a stiff, narrow-grooved tract)
        # is a very different linker from 30 bp of alternating R/Y, even though
        # a plain bp distance cannot tell them apart. Overlapping or adjacent
        # motifs give an empty window and therefore counts of 0.
        purine_total, spacer_width = self._range_sum(cum_purine, spacer_lo, spacer_hi)
        pyrimidine_total = (spacer_width - purine_total).clamp(min=0.0)
        scale = float(max(length, 1))
        purine_distance = purine_total / scale
        pyrimidine_distance = pyrimidine_total / scale

        # Per-base spacer counts, same cumulative-sum trick, one channel each.
        spacer_bases = []
        for channel in range(len(NUCLEOTIDE_ORDER)):
            cum_base = self._cumulative(x[:, channel, :])
            total, _ = self._range_sum(cum_base, spacer_lo, spacer_hi)
            spacer_bases.append(total / scale)

        # R<->Y alternation density of the spacer (DNA flexibility proxy). Switch
        # index p describes the p -> p+1 boundary, so the last index inside a
        # window [lo, hi) is hi-1 and the motif boundary itself is excluded.
        ry_switch = (purine[:, 1:] - purine[:, :-1]).abs()
        cum_switch = self._cumulative(ry_switch)
        ry_transitions = self._range_mean(cum_switch, spacer_lo, (spacer_hi - 1).clamp(min=0))

        strand_sign = torch.sign(strand_bias)
        same_strand = (strand_sign.unsqueeze(2) * strand_sign.unsqueeze(1)).clamp(-1, 1)

        co_present = present.unsqueeze(2) * present.unsqueeze(1)    # [B, V, V]

        # --- signed interval gap --------------------------------------------
        # One continuous channel instead of two: the standard interval gap
        # between [start_i, end_i) and [start_j, end_j).
        #
        #   > 0  bases of spacer between them   (cooperation is possible)
        #   = 0  exactly touching
        #   < 0  bases of OVERLAP               (competition: the two factors
        #                                        cannot occupy the DNA at once)
        #
        # Encoding overlap as negative distance keeps the relationship monotone,
        # so the network sees competition and cooperation on one axis rather
        # than having to combine a non-negative distance with a separate flag.
        shared = torch.minimum(ends_i, ends_j) - torch.maximum(starts_i, starts_j)
        signed_gap = -shared                                    # +spacer / -overlap
        signed_distance = signed_gap / max(length, 1)

        width_i = motif_len.unsqueeze(2)
        width_j = motif_len.unsqueeze(1)
        shorter = torch.minimum(width_i, width_j).clamp(min=1.0)
        overlap = (shared.clamp(min=0.0) / shorter).clamp(max=1.0)

        # Touching or nearly touching without overlapping — composite sites.
        adjacent = ((shared <= 0) & (signed_gap <= self.adjacent_gap)).to(x.dtype)

        edge_feats = torch.stack([
            *[b * co_present for b in spacer_bases],
            purine_distance * co_present,
            pyrimidine_distance * co_present,
            signed_distance * co_present,
            distance * co_present,
            helical_phase * co_present,
            ry_transitions * co_present,
            order * co_present,
            same_strand * co_present,
            overlap * co_present,
            adjacent * co_present,
            co_present,
        ], dim=-1)                                                  # [B, V, V, 10]

        eye = torch.eye(v, device=x.device, dtype=x.dtype).unsqueeze(0)

        # --- redundancy node feature ---------------------------------------
        # How many OTHER motifs land on top of this one's site. With a library
        # like JASPAR this is large wherever several profiles describe the same
        # TF or a shared family motif, and it lets the network discount what
        # would otherwise look like several independent binding events.
        overlaps_other = (overlap > 0).to(x.dtype) * co_present * (1.0 - eye)
        redundancy = overlaps_other.sum(dim=2) / max(v - 1, 1)
        node_parts.append(redundancy)

        # --- flanking base composition ---------------------------------------
        # Fractions of A/C/T/G in the `flank_bp` bases immediately before the
        # motif and immediately after it, from per-channel cumulative sums.
        for side in ("left", "right"):
            if side == "left":
                lo = (pos_idx.float() - self.flank_bp).long()
                hi = pos_idx
            else:
                lo = end_idx.long()
                hi = (end_idx + self.flank_bp).long()
            for channel in range(len(NUCLEOTIDE_ORDER)):
                cum = self._cumulative(x[:, channel, :])
                node_parts.append(self._range_mean(cum, lo, hi) * present)
        node_feats = torch.stack(node_parts, dim=-1)                # [B, V, 8]

        # --- neighbourhood --------------------------------------------------
        # 'dense' connects every pair of motifs present in the sequence. That is
        # a complete graph: on real DeepSTARR enhancers ~137 of 296 JASPAR motifs
        # occur, so each node then averages messages from ~137 neighbours, the
        # median of which sits 61 bp away and over half more than 50 bp away.
        # Enhancer cooperativity is mostly local, so most of those messages are
        # noise being averaged into the signal.
        #
        # 'radius' and 'knn' encode that locality as an explicit prior instead of
        # hoping the distance edge feature teaches the network to ignore distant
        # pairs.
        adjacency = co_present
        if self.neighbourhood == "radius":
            adjacency = adjacency * (gap <= self.radius_bp).to(x.dtype)
        elif self.neighbourhood == "knn":
            k = min(self.knn_k, v)
            # Rank by distance among co-present motifs only.
            ranking = torch.where(co_present > 0, gap, torch.full_like(gap, float("inf")))
            idx = ranking.topk(k, dim=2, largest=False).indices
            keep = torch.zeros_like(adjacency).scatter_(2, idx, 1.0)
            adjacency = adjacency * keep
            # Symmetrise: i is j's neighbour if either selected the other.
            adjacency = torch.clamp(adjacency + adjacency.transpose(1, 2), max=1.0) * co_present

        # A motif that does not occur is an isolated node; self-loops keep every
        # node's own features flowing through the message passing.
        adjacency = torch.clamp(adjacency + eye * present.unsqueeze(2), max=1.0)

        return node_feats, edge_feats, adjacency

    # -- introspection ------------------------------------------------------

    def load_cache(self, parquet_path: str, device=None):
        """Attach a Parquet motif cache written by ``precompute_motifs.py``.

        Motif scanning is the expensive part of featurisation — 592 convolution
        kernels over every sequence, every epoch — and it is *deterministic*
        given (sequence, vocabulary).  Caching the hits turns it into an array
        lookup.  The purine/pyrimidine spacer counts are still computed live
        because they are one cheap cumulative sum, not a convolution.

        The cached vocabulary must match the installed one exactly, otherwise
        motif indices would silently point at different motifs.
        """
        from precompute_motifs import load_parquet

        arrays, meta = load_parquet(parquet_path)
        cached_names = [m["name"] for m in meta["vocabulary"]]
        if cached_names != list(self.vocab.names):
            raise ValueError(
                f"Cache vocabulary does not match the featurizer's: cache has "
                f"{len(cached_names)} motifs starting {cached_names[:3]}, model has "
                f"{len(self.vocab)} starting {list(self.vocab.names)[:3]}. "
                f"Re-run precompute_motifs.py for this vocabulary.")

        device = device or getattr(self, "kernels", torch.zeros(1)).device
        self._cache = {k: torch.as_tensor(v).to(device) for k, v in arrays.items()}
        self._cache_meta = meta
        print(f"[graph] motif cache: {meta['n_sequences']} sequences x "
              f"{meta['n_motifs']} motifs from {parquet_path}")
        return self

    @property
    def has_cache(self) -> bool:
        return getattr(self, "_cache", None) is not None

    @torch.no_grad()
    def hit_positions(self, x: torch.Tensor, top_n: int | None = None):
        """Start positions of each motif's *top_n* best hits, ``(B, V, top_n)``.

        Padded with ``-1`` where a motif has fewer hits.  Needed because 55% of
        detected motifs occur more than once (mean 2.31 hits), so a single
        representative position throws away most of the geometry.
        """
        top_n = int(top_n or self.multi_hits)
        v = self.num_motifs
        pad = self.max_len - 1
        response = F.conv1d(F.pad(x.float(), (0, pad)), self.kernels.to(x.dtype))
        positions = response.shape[-1]
        starts = torch.arange(positions, device=x.device).view(1, 1, -1)
        valid = starts <= (x.shape[-1] - self.widths.view(1, -1, 1))
        hits = (response >= self.thresholds.view(1, -1, 1)) & valid
        either = (hits[:, :v] | hits[:, v:]).to(x.dtype)
        score = torch.where(either.bool(),
                            torch.maximum(response[:, :v], response[:, v:]),
                            torch.full_like(response[:, :v], -1e9))
        best = score.topk(min(top_n, positions), dim=-1)
        return torch.where(best.values > -1e8, best.indices,
                           torch.full_like(best.indices, -1))

    @torch.no_grad()
    def sparse_forward(self, x: torch.Tensor, k: int | None = None):
        """Featurise into a **fixed-degree** graph: each node keeps its *k* nearest motifs.

        Same semantics as a ``torch_geometric`` knn-graph + ``MessagePassing``,
        but without the dependency — PyG's message passing is gather + scatter,
        and with a constant *k* the gather alone suffices, so no scatter is
        needed at all.  (PyG also has no wheels for torch 2.10 / CUDA 13 yet;
        adopting it would mean downgrading a working stack.)

        The win is memory.  Dense message passing costs ``O(B * V^2 * H)``;
        keeping *k* neighbours costs ``O(B * V * k * H)`` — a ``V/k`` reduction,
        which is **148x** at V=296, k=2.  That is what makes the full JASPAR set
        affordable.

        Returns
        -------
        node_feats : ``(B, V, node_dim)``
        edge_feats : ``(B, V, k, edge_dim)``  features of each kept edge
        neighbour_idx : ``(B, V, k)``         index of each neighbour
        neighbour_mask : ``(B, V, k)``        1 where the edge is real
        """
        k = int(k or self.knn_k)
        node_feats, edge_feats, adjacency = self.forward(x)
        v = node_feats.shape[1]
        k = max(1, min(k, v))

        # Rank co-present neighbours by |signed distance|: nearest first, with
        # overlapping motifs (negative gap) ranked closest of all.
        gap = edge_feats[..., EDGE_FEATURE_NAMES.index("signed_distance")].abs()
        eye = torch.eye(v, device=x.device, dtype=torch.bool).unsqueeze(0)
        valid = (adjacency > 0) & ~eye
        ranking = torch.where(valid, gap, torch.full_like(gap, float("inf")))

        nearest = ranking.topk(k, dim=2, largest=False)
        neighbour_idx = nearest.indices                       # [B, V, k]
        neighbour_mask = torch.isfinite(nearest.values).to(x.dtype)

        gathered = torch.gather(
            edge_feats, 2,
            neighbour_idx.unsqueeze(-1).expand(-1, -1, -1, edge_feats.shape[-1]))
        gathered = gathered * neighbour_mask.unsqueeze(-1)

        if self.multi_hits > 1:
            gathered = torch.cat(
                [gathered, self._cooccurrence_profile(x, neighbour_idx, neighbour_mask)],
                dim=-1)
        return node_feats, gathered, neighbour_idx, neighbour_mask

    @torch.no_grad()
    def _cooccurrence_profile(self, x, neighbour_idx, mask):
        """Distance histogram over **all** hit pairs of an edge, ``(B, V, k, bins)``.

        A single representative distance cannot express "these two motifs keep
        turning up ~20 bp apart", yet more than half of detected motifs occur
        several times.  For each kept edge this takes every hit of motif *i*
        against every hit of motif *j* and bins the separations, so repeated
        spacing becomes visible.

        Cheap because the hit list is capped at ``multi_hits``: the cost is
        ``O(B * V * k * multi_hits^2)``, never quadratic in the vocabulary.
        """
        b, v, k = neighbour_idx.shape
        n = self.multi_hits
        length = x.shape[-1]

        pos = self.hit_positions(x, n).float()                      # [B, V, n]
        centre = pos + (self.widths[:v].view(1, -1, 1) - 1.0) / 2.0
        centre = torch.where(pos >= 0, centre, torch.full_like(centre, -1e4))

        src = centre.unsqueeze(2).expand(b, v, k, n)                # [B,V,k,n]
        flat = neighbour_idx.reshape(b, v * k, 1).expand(b, v * k, n)
        dst = torch.gather(centre, 1, flat).reshape(b, v, k, n)

        gap = (src.unsqueeze(-1) - dst.unsqueeze(-2)).abs()         # [B,V,k,n,n]
        real = (src.unsqueeze(-1) > -1e3) & (dst.unsqueeze(-2) > -1e3)

        edges = self.cooccurrence_bins
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            inside = real & (gap >= lo) & (gap < hi)
            out.append(inside.to(x.dtype).flatten(-2).sum(-1) / (n * n))
        profile = torch.stack(out, dim=-1)
        return profile * mask.unsqueeze(-1)

    @torch.no_grad()
    def occurrence_rate(self, x: torch.Tensor) -> torch.Tensor:
        """Fraction of sequences in *x* containing each motif, shape ``(V,)``.

        Used to drop motifs that never fire (dead nodes) before they waste a slot
        in the graph.
        """
        node_feats, _, _ = self.forward(x)
        return node_feats[..., NODE_FEATURE_NAMES.index("present")].mean(dim=0)

    def describe(self) -> str:
        return (f"MotifGraphFeaturizer(V={self.num_motifs}, max_len={self.max_len}, "
                f"node_dim={self.node_dim}, edge_dim={self.edge_dim}, "
                f"min_match_frac={self.min_match_frac})")
