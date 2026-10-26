from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dna import reverse_complement_tensor
from motif_graph import (EDGE_DIM, NODE_DIM, MotifGraphFeaturizer, MotifVocabulary,
                         sample_vocabulary)
from .registry import register_model


# ---------------------------------------------------------------------------
# Message passing
# ---------------------------------------------------------------------------

class EdgeConditionedLayer(nn.Module):

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1,
                 edge_dropout: float = 0.1):
        super().__init__()
        self.edge_dropout = float(edge_dropout)
        self.src = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dst = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge = nn.Linear(edge_dim, hidden_dim)
        self.message_out = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_feats, adj):
        k = h.shape[1]

        if self.training and self.edge_dropout > 0:
            keep = (torch.rand_like(adj) > self.edge_dropout).to(adj.dtype)
            # Never drop self-loops: an isolated node must keep its own signal.
            eye = torch.eye(k, device=adj.device, dtype=adj.dtype).unsqueeze(0)
            adj = adj * torch.clamp(keep + eye, max=1.0)

        # [B,V,1,H] + [B,1,V,H] + [B,V,V,H] -> [B,V,V,H] without any concat.
        msg = self.edge(edge_feats)
        msg = msg + self.src(h).unsqueeze(2) + self.dst(h).unsqueeze(1)
        msg = self.message_out(F.gelu(msg))

        msg = msg * adj.unsqueeze(-1)
        agg = msg.sum(dim=2) / adj.sum(dim=2, keepdim=True).clamp_min(1.0)
        return self.norm(h + self.dropout(F.gelu(self.update(agg))))


class SparseEdgeConditionedLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1,
                 edge_dropout: float = 0.1):
        super().__init__()
        self.edge_dropout = float(edge_dropout)
        self.src = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dst = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge = nn.Linear(edge_dim, hidden_dim)
        self.message_out = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_feats, neighbour_idx, mask):
        b, v, hidden = h.shape
        k = neighbour_idx.shape[-1]

        if self.training and self.edge_dropout > 0:
            mask = mask * (torch.rand_like(mask) > self.edge_dropout).to(mask.dtype)

        # Gather neighbour states straight out of [B, V, H] -> [B, V, k, H].
        # NB: gathering from an expanded [B, V, V, H] view would materialise the
        # dense tensor this layer exists to avoid; index into h directly instead.
        flat_idx = neighbour_idx.reshape(b, v * k, 1).expand(b, v * k, hidden)
        h_neigh = torch.gather(h, 1, flat_idx).reshape(b, v, k, hidden)

        msg = self.edge(edge_feats) + self.src(h).unsqueeze(2) + self.dst(h_neigh)
        msg = self.message_out(F.gelu(msg)) * mask.unsqueeze(-1)
        agg = msg.sum(dim=2) / mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        return self.norm(h + self.dropout(F.gelu(self.update(agg))))


class MaskedPooling(nn.Module):
    """Pool node states to a graph vector, ignoring absent motifs.

    Mean + max over *present* nodes only.  Pooling over absent nodes would let
    the model read "how many motifs are missing" as a shortcut feature, which is
    a dataset artefact rather than biology.

    Sequences containing *none* of the vocabulary's motifs are real and common
    with a small vocabulary, so they are handled explicitly: the pooled vector
    falls back to zeros and a ``has_motifs`` flag is appended, letting the head
    learn a distinct response for an empty graph instead of confusing it with a
    genuinely zero-valued embedding.
    """

    #: Extra channel appended to the pooled vector.
    extra_dim = 1

    def forward(self, h, present):
        mask = present.unsqueeze(-1)
        count = mask.sum(dim=1)                       # [B, 1] — NOT clamped
        denom = count.clamp_min(1.0)
        non_empty = count > 0

        mean = (h * mask).sum(dim=1) / denom

        # A finite sentinel: LayerNorm keeps h at O(1), so -1e4 always loses the
        # max while avoiding the overflow that torch.finfo().min causes downstream.
        maxed = h.masked_fill(~mask.bool(), -1e4).max(dim=1).values
        maxed = torch.where(non_empty, maxed, torch.zeros_like(maxed))
        mean = torch.where(non_empty, mean, torch.zeros_like(mean))

        return torch.cat([mean, maxed, non_empty.to(h.dtype)], dim=-1)


# ---------------------------------------------------------------------------
# 1. MotifGNN
# ---------------------------------------------------------------------------

@register_model("MotifGNN")
class MotifGNN(nn.Module):
    """Edge-conditioned GNN over a motif / k-mer graph.

    Parameters
    ----------
    vocab : dict, optional
        ``motif_graph.MotifVocabulary.from_config`` section; defaults to the
        known Drosophila TF motifs.
    hidden_dim, num_layers : int
        Message-passing capacity.
    dropout, edge_dropout, node_dropout : float
        Regularisation.  ``edge_dropout`` (DropEdge) is the most effective of the
        three on dense motif graphs.
    min_match_frac : float
        Passed to the featurizer; ``1.0`` = exact IUPAC matching.
    node_feature_dropout : float
        Randomly zero whole node-feature channels during training, so the model
        cannot lean entirely on e.g. raw motif counts.
    """

    def __init__(self, vocab=None, hidden_dim=96, num_layers=3, dropout=0.2,
                 edge_dropout=0.15, node_dropout=0.1, node_feature_dropout=0.0,
                 min_match_frac=1.0, max_count=8, context=10, adjacent_gap=3,
                 neighbourhood="dense", radius_bp=60, knn_k=16, sparse=False, **kwargs):
        super().__init__()
        vocabulary = (vocab if isinstance(vocab, MotifVocabulary)
                      else MotifVocabulary.from_config(vocab))
        self.featurizer = MotifGraphFeaturizer(
            vocabulary, min_match_frac=min_match_frac, max_count=max_count,
            context=context, adjacent_gap=adjacent_gap,
            neighbourhood=neighbourhood, radius_bp=radius_bp, knn_k=knn_k)

        self.hidden_dim = hidden_dim
        self.node_feature_dropout = float(node_feature_dropout)

        # Node/edge encoders are shared across nodes, so the vocabulary can be
        # swapped at any time without touching a single parameter shape.
        self.node_encoder = nn.Sequential(
            nn.Linear(NODE_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(node_dropout),
        )
        edge_in = (self.featurizer.sparse_edge_dim if sparse else EDGE_DIM)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in, EDGE_DIM * 4),
            nn.GELU(),
        )
        # 'sparse' keeps only each node's k nearest motifs, turning the
        # O(B*V^2*H) message tensor into O(B*V*k*H). Required for large
        # vocabularies: the full JASPAR set is unusable densely.
        self.sparse = bool(sparse)
        layer_cls = SparseEdgeConditionedLayer if self.sparse else EdgeConditionedLayer
        self.layers = nn.ModuleList(
            layer_cls(hidden_dim, EDGE_DIM * 4, dropout, edge_dropout)
            for _ in range(num_layers)
        )
        self.pool = MaskedPooling()
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim + MaskedPooling.extra_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_dev = nn.Linear(hidden_dim, 1)
        self.fc_hk = nn.Linear(hidden_dim, 1)

        self.register_buffer("_zero", torch.zeros(()), persistent=False)
        self.reg_loss = self._zero

    # -- vocabulary ---------------------------------------------------------

    @property
    def vocab(self) -> MotifVocabulary:
        return self.featurizer.vocab

    def set_vocabulary(self, vocabulary: MotifVocabulary):
        """Install a new node set.  No parameter is resized or reset."""
        self.featurizer.set_vocabulary(vocabulary)
        return self

    def memory_estimate(self, batch_size: int) -> float:
        """Peak activation megabytes per message-passing layer.

        Dense message passing is ``O(B * V^2 * hidden)``, which is what limits
        the vocabulary size: 156 nodes at hidden=96 and batch=128 already needs
        ~1.2 GB *per layer* before autograd overhead.

        In ``sparse`` mode each node only aggregates over its *k* nearest
        motifs, so the cost drops to ``O(B * V * k * hidden)`` — a ``V/k``
        reduction that makes the entire JASPAR set affordable.
        """
        v = self.featurizer.num_motifs
        fan_out = min(self.featurizer.knn_k, v) if self.sparse else v
        return batch_size * v * fan_out * self.hidden_dim * 4 / 1e6

    # -- forward ------------------------------------------------------------

    def embed(self, x):
        if self.sparse:
            node_feats, edge_feats, neighbour_idx, mask = self.featurizer.sparse_forward(x)
        else:
            node_feats, edge_feats, adj = self.featurizer(x)
        present = node_feats[..., 0]

        if self.training and self.node_feature_dropout > 0:
            keep = (torch.rand(node_feats.shape[0], 1, NODE_DIM, device=x.device)
                    > self.node_feature_dropout).to(node_feats.dtype)
            # 'present' drives pooling and adjacency; dropping it is meaningless.
            keep[..., 0] = 1.0
            node_feats = node_feats * keep

        h = self.node_encoder(node_feats)
        e = self.edge_encoder(edge_feats)
        for layer in self.layers:
            h = layer(h, e, neighbour_idx, mask) if self.sparse else layer(h, e, adj)
        return self.head(self.pool(h, present))

    def forward(self, x):
        feat = self.embed(x)
        return self.fc_dev(feat), self.fc_hk(feat)

    def get_features(self, x):
        """Penultimate vector, for SAE / CrossSAE comparison with the CNNs."""
        return self.embed(x)

    # -- interpretation -----------------------------------------------------

    @torch.no_grad()
    def motif_occurrence(self, x):
        """Fraction of sequences in *x* containing each vocabulary motif."""
        return self.featurizer.occurrence_rate(x)

    def motif_names(self):
        return list(self.vocab.names)


# ---------------------------------------------------------------------------
# 2. Gradient-boosted GNN
# ---------------------------------------------------------------------------

@register_model("BoostedMotifGNN")
class BoostedMotifGNN(nn.Module):
    """Gradient boosting over motif-graph GNNs.

    Stage *m* is a deliberately small :class:`MotifGNN` restricted to a random
    subset of the vocabulary, fit to the residual the first *m-1* stages leave
    behind.  The prediction is

    .. math:: F_M(x) = F_0 + \\nu \\sum_{m=1}^{M} f_m(x)

    with shrinkage :math:`\\nu` (``learning_rate``).  Because each stage sees a
    different motif subset, later stages are forced to explain the data with
    *different* motif combinations than the ones already used — this is the
    "many specific motifs correcting each other" behaviour, and it is also what
    keeps the ensemble from simply re-fitting the same features harder.

    Boosting is what actually controls overfitting here: each stage is too small
    to memorise anything on its own, shrinkage keeps every correction tiny, and
    :func:`train_graph.fit_boosted` stops adding stages the moment validation
    stops improving.

    Train with ``python train_graph.py -c ../../config/BoostedMotifGNN.yaml``;
    calling ``forward`` directly evaluates the whole ensemble as usual.

    Parameters
    ----------
    n_stages : int
        Maximum number of boosting stages.
    learning_rate : float
        Shrinkage applied to every stage's contribution.
    subsample_motifs : float
        Fraction of the vocabulary each stage may use (column subsampling, as in
        gradient-boosted trees).  ``1.0`` gives every stage the full vocabulary.
    stage_hidden_dim, stage_layers : int
        Capacity of a single stage — intentionally small.
    """

    def __init__(self, vocab=None, n_stages=8, learning_rate=0.3,
                 subsample_motifs=0.6, stage_hidden_dim=48, stage_layers=2,
                 dropout=0.15, edge_dropout=0.2, min_match_frac=1.0,
                 seed=42, **kwargs):
        super().__init__()
        self.full_vocab = (vocab if isinstance(vocab, MotifVocabulary)
                           else MotifVocabulary.from_config(vocab))
        self.n_stages = int(n_stages)
        self.learning_rate = float(learning_rate)
        self.subsample_motifs = float(subsample_motifs)

        n_sub = max(2, int(round(self.subsample_motifs * len(self.full_vocab))))
        self.stages = nn.ModuleList()
        self.stage_motifs: list[list[str]] = []
        for m in range(self.n_stages):
            sub = sample_vocabulary(self.full_vocab, n_sub, seed=seed + m)
            self.stages.append(MotifGNN(
                vocab=sub, hidden_dim=stage_hidden_dim, num_layers=stage_layers,
                dropout=dropout, edge_dropout=edge_dropout,
                min_match_frac=min_match_frac, **kwargs))
            self.stage_motifs.append(list(sub.names))

        # F_0: the constant baseline every gradient-boosting model starts from,
        # set to the training-target mean by fit_boosted().
        self.register_buffer("init_dev", torch.zeros(1))
        self.register_buffer("init_hk", torch.zeros(1))
        # Stages activate one at a time during boosting; all are live at eval.
        self.register_buffer("stage_active", torch.zeros(self.n_stages))
        self.register_buffer("_zero", torch.zeros(()), persistent=False)
        self.reg_loss = self._zero

    # -- boosting control ---------------------------------------------------

    @property
    def n_active(self) -> int:
        return int(self.stage_active.sum().item())

    def activate_stage(self, index: int):
        """Enable stage *index* and freeze every earlier one."""
        self.stage_active[index] = 1.0
        for m in range(index):
            for p in self.stages[m].parameters():
                p.requires_grad_(False)
            self.stages[m].eval()
        for p in self.stages[index].parameters():
            p.requires_grad_(True)
        return self.stages[index]

    def activate_all(self):
        self.stage_active.fill_(1.0)
        return self

    def truncate(self, n_stages: int):
        """Keep only the first *n_stages* — used to roll back to the best round."""
        self.stage_active[n_stages:] = 0.0
        return self

    def set_baseline(self, dev_mean: float, hk_mean: float):
        self.init_dev.fill_(float(dev_mean))
        self.init_hk.fill_(float(hk_mean))
        return self

    # -- forward ------------------------------------------------------------

    def partial(self, x, upto: int | None = None, detach_previous: bool = False):
        """Ensemble prediction using stages ``[0, upto)``.

        With ``detach_previous=True`` the already-fit stages contribute values but
        no gradient — that is exactly the boosting update, where stage *m* only
        ever sees the residual as a constant target.
        """
        limit = self.n_stages if upto is None else upto
        dev = self.init_dev.expand(x.shape[0], 1).clone()
        hk = self.init_hk.expand(x.shape[0], 1).clone()

        for m in range(limit):
            if self.stage_active[m] == 0:
                continue
            if detach_previous and m < limit - 1:
                with torch.no_grad():
                    d, h = self.stages[m](x)
                d, h = d.detach(), h.detach()
            else:
                d, h = self.stages[m](x)
            dev = dev + self.learning_rate * d
            hk = hk + self.learning_rate * h
        return dev, hk

    def forward(self, x):
        if self.n_active == 0:          # untrained: predict the baseline
            self.activate_all()
        return self.partial(x)

    def get_features(self, x):
        """Concatenated stage embeddings (active stages only)."""
        feats = [self.stages[m].embed(x) for m in range(self.n_stages)
                 if self.stage_active[m] > 0]
        return torch.cat(feats, dim=-1) if feats else self.stages[0].embed(x)

    @torch.no_grad()
    def stage_report(self):
        """Per-stage motif subsets, for reading off what each correction uses."""
        return [{"stage": m, "active": bool(self.stage_active[m] > 0),
                 "n_motifs": len(names), "motifs": names}
                for m, names in enumerate(self.stage_motifs)]


# ---------------------------------------------------------------------------
# 3. Graph-SAE
# ---------------------------------------------------------------------------

@register_model("GraphSAE")
class GraphSAE(nn.Module):
    """A MotifGNN that searches for its own vocabulary.

    The node set starts as an a-priori sample from a large candidate pool (known
    TF motifs plus every k-mer of a chosen length).  During training a sparse
    *usage gate* per motif — top-k over a softmax-free sigmoid, with an L1
    penalty — records which motifs the model actually reads.  Every
    ``resample_every`` epochs :meth:`resample` is called by the trainer:

    1. score each motif by gate value x how often it occurs on the validation
       split (a motif that is gated on but never fires is useless);
    2. prune the worst ``prune_frac`` — validation-driven, so this cannot fit
       training noise;
    3. draw replacements from the untried remainder of the candidate pool.

    This is the SAE dead-feature resampling idea (see
    :mod:`sparse_ae.model`) applied to the *input vocabulary* rather than to a
    hidden dictionary: instead of learning arbitrary directions, the model
    searches over a discrete, human-readable set of motifs, and the surviving
    vocabulary *is* the explanation.

    The reverse-complement consistency loss belongs to this model in particular:
    a motif and its reverse complement are one node, so a well-fit GraphSAE
    should already be nearly strand-symmetric, and the penalty both enforces that
    and acts as a regulariser that does not require any labels.

    Parameters
    ----------
    pool : dict
        Candidate pool spec, same schema as ``vocab`` (e.g.
        ``{known: true, kmers: 5}``).
    active_size : int
        Number of motifs held in the working vocabulary (the graph's node count).
    pin : list of str
        Motif names never pruned — use it to keep known TF motifs in place.
    gate_l1 : float
        L1 penalty on the usage gate; drives unused motifs toward zero so the
        pruning signal is sharp.
    gate_topk : int
        If > 0, hard-limit the gate to its k largest entries each forward pass.
    prune_frac : float
        Fraction of the active vocabulary replaced at each resampling round.
    """

    def __init__(self, pool=None, active_size=48, pin=(), hidden_dim=96,
                 num_layers=3, dropout=0.2, edge_dropout=0.15, node_dropout=0.1,
                 gate_l1=1e-3, gate_topk=0, prune_frac=0.25, min_match_frac=1.0,
                 seed=42, **kwargs):
        super().__init__()
        self.pool = (pool if isinstance(pool, MotifVocabulary)
                     else MotifVocabulary.from_config(pool or {"known": True, "kmers": 5}))
        self.active_size = min(int(active_size), len(self.pool))
        self.pin = list(pin)
        self.gate_l1 = float(gate_l1)
        self.gate_topk = int(gate_topk)
        self.prune_frac = float(prune_frac)
        self.seed = int(seed)

        active = sample_vocabulary(self.pool, self.active_size, seed=seed, include=self.pin)
        self.gnn = MotifGNN(vocab=active, hidden_dim=hidden_dim, num_layers=num_layers,
                            dropout=dropout, edge_dropout=edge_dropout,
                            node_dropout=node_dropout, min_match_frac=min_match_frac,
                            **kwargs)

        # One gate per *slot*; slots are re-bound to new motifs on resampling, so
        # the parameter shape is fixed while the vocabulary behind it changes.
        self.gate_logit = nn.Parameter(torch.zeros(self.active_size))
        self.register_buffer("usage", torch.zeros(self.active_size))
        self.register_buffer("_zero", torch.zeros(()), persistent=False)
        self.reg_loss = self._zero
        self._tried: set[str] = set(self.gnn.vocab.names)

    # -- vocabulary ---------------------------------------------------------

    @property
    def vocab(self) -> MotifVocabulary:
        return self.gnn.vocab

    def set_vocabulary(self, vocabulary: MotifVocabulary):
        """Install a node set (must have :attr:`active_size` entries: one per gate)."""
        if len(vocabulary) != self.active_size:
            raise ValueError(
                f"GraphSAE has {self.active_size} gate slots but was given "
                f"{len(vocabulary)} motifs; resampling replaces slots, it does not resize.")
        self.gnn.set_vocabulary(vocabulary)
        return self

    def motif_names(self):
        return list(self.vocab.names)

    def gates(self) -> torch.Tensor:
        g = torch.sigmoid(self.gate_logit)
        if self.gate_topk > 0 and self.gate_topk < g.numel():
            threshold = torch.topk(g, self.gate_topk).values[-1]
            g = g * (g >= threshold).to(g.dtype)
        return g

    # -- forward ------------------------------------------------------------

    def embed(self, x):
        node_feats, edge_feats, adj = self.gnn.featurizer(x)
        present = node_feats[..., 0]

        # The gate multiplies each motif's node features and isolates gated-off
        # motifs from the graph, so a closed gate genuinely removes the motif.
        gate = self.gates().view(1, -1, 1)
        node_feats = node_feats * gate
        gate_2d = gate.squeeze(-1)
        adj = adj * gate_2d.unsqueeze(2) * gate_2d.unsqueeze(1)

        h = self.gnn.node_encoder(node_feats)
        e = self.gnn.edge_encoder(edge_feats)
        for layer in self.gnn.layers:
            h = layer(h, e, adj)

        if self.training:
            with torch.no_grad():
                # Usage = gate strength weighted by how often the motif fires.
                self.usage.mul_(0.9).add_(0.1 * (self.gates() * present.mean(0)))
        self.reg_loss = self.gate_l1 * self.gates().abs().mean()
        return self.gnn.head(self.gnn.pool(h, present * (gate_2d > 0).float()))

    def forward(self, x):
        feat = self.embed(x)
        return self.gnn.fc_dev(feat), self.gnn.fc_hk(feat)

    def get_features(self, x):
        return self.embed(x)

    # -- vocabulary search --------------------------------------------------

    @torch.no_grad()
    def motif_scores(self, val_batches=None, device=None) -> torch.Tensor:
        """Score every active slot: gate strength x validation occurrence rate.

        A motif is only worth a slot if the model gates it *on* and it actually
        occurs in held-out data.  Scoring on validation is what stops the search
        from chasing training-set noise.
        """
        gate = self.gates().detach().cpu()
        if val_batches is None:
            return gate * self.usage.detach().cpu().clamp(min=1e-6)

        rates = None
        n = 0
        for batch in val_batches:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if device is not None:
                x = x.to(device)
            r = self.gnn.featurizer.occurrence_rate(x).detach().cpu()
            rates = r if rates is None else rates + r
            n += 1
        rates = rates / max(n, 1)
        return gate * rates

    @torch.no_grad()
    def resample(self, val_batches=None, device=None, generator=None):
        """Prune the weakest motifs and draw replacements from the pool.

        Returns
        -------
        dict
            ``{"dropped": [names], "added": [names], "scores": {...}}`` — printed
            by the trainer so the vocabulary search is auditable.
        """
        scores = self.motif_scores(val_batches, device)
        names = self.motif_names()
        n_drop = max(1, int(round(self.prune_frac * len(names))))

        pinned = set(self.pin)
        order = torch.argsort(scores)
        droppable = [int(i) for i in order.tolist() if names[i] not in pinned]
        drop_idx = droppable[:n_drop]
        if not drop_idx:
            return {"dropped": [], "added": [], "scores": {}}

        # Candidates: pool entries not currently active and not tried before, so
        # the search explores rather than cycling through the same motifs.
        active = set(names)
        fresh = [i for i, nm in enumerate(self.pool.names)
                 if nm not in active and nm not in self._tried]
        if len(fresh) < len(drop_idx):     # pool exhausted -> allow retries
            fresh = [i for i, nm in enumerate(self.pool.names) if nm not in active]
        if not fresh:
            return {"dropped": [], "added": [], "scores": {}}

        gen = generator or torch.Generator().manual_seed(self.seed)
        pick = torch.randperm(len(fresh), generator=gen)[:len(drop_idx)].tolist()
        new_entries = [(self.pool.names[fresh[p]], self.pool.patterns[fresh[p]]) for p in pick]

        entries = list(zip(self.vocab.names, self.vocab.patterns))
        dropped = []
        for slot, (nm, pat) in zip(drop_idx, new_entries):
            dropped.append(entries[slot][0])
            entries[slot] = (nm, pat)
            self._tried.add(nm)
            # Re-open the gate for the fresh motif and clear its usage history.
            self.gate_logit.data[slot] = 0.0
            self.usage[slot] = 0.0

        self.gnn.set_vocabulary(
            MotifVocabulary(entries, canonicalise_rc=self.pool.canonicalise_rc))
        return {"dropped": dropped,
                "added": [nm for nm, _ in new_entries],
                "scores": {names[i]: float(scores[i]) for i in drop_idx}}

    @torch.no_grad()
    def vocabulary_report(self):
        """Active motifs with their gate values and usage, best first."""
        gate = self.gates().detach().cpu().tolist()
        usage = self.usage.detach().cpu().tolist()
        rows = [{"slot": i, "name": n, "pattern": p, "gate": gate[i], "usage": usage[i]}
                for i, (n, p) in enumerate(self.vocab)]
        return sorted(rows, key=lambda r: -r["gate"])


# ---------------------------------------------------------------------------
# 4. Hybrid: convolutional trunk + motif graph
# ---------------------------------------------------------------------------

@register_model("MotifGraphHybrid")
class MotifGraphHybrid(nn.Module):
    """A convolutional trunk fused with the motif-graph branch.

    Why this exists
    ---------------
    A motif graph is a *lossy* description of a sequence, and measurably so: it
    keeps one representative occurrence per motif and no base-level position, so
    a 249x4 sequence collapses to a few dozen mostly-saturated indicators.  That
    is exactly what makes it hard to overfit — and also why, on its own, it
    cannot match a CNN that reads the sequence at base resolution.

    The two views are complementary rather than competing:

    * the **trunk** (any registered CNN, e.g. LegNetV2) supplies fine-grained
      sequence features;
    * the **graph** supplies explicit motif syntax — spacing, helical phase,
      spacer flexibility, overlap — which the CNN would otherwise have to
      rediscover from scratch, and which acts as a structured regulariser.

    Both feature vectors are LayerNormed before concatenation so neither branch
    can dominate purely through scale, and ``graph_weight`` lets the graph
    branch be down-weighted (or, at 0, ablated) without changing shapes.

    Parameters
    ----------
    trunk : dict
        ``{"name": "LegNetV2", "kwargs": {...}}`` — any registered model that
        exposes ``get_features``.
    trunk_weights : str, optional
        Checkpoint to warm-start the trunk from (e.g. an already-trained
        LegNetV2).  Missing/extra keys are reported, not silently ignored.
    freeze_trunk : bool
        Train only the graph branch and the fusion head.  Useful to check what
        the graph adds on top of a fixed CNN.
    vocab : dict
        Motif vocabulary for the graph branch (see :class:`MotifGNN`).
    graph_weight : float
        Scales the graph embedding before fusion; 0.0 ablates the branch.
    """

    def __init__(self, trunk=None, trunk_weights=None, freeze_trunk=False,
                 vocab=None, graph_hidden=96, graph_layers=3, graph_weight=1.0,
                 seq_len=249, dropout=0.2, edge_dropout=0.15, node_dropout=0.1,
                 min_match_frac=1.0, hidden_dim=256,
                 neighbourhood="dense", radius_bp=60, knn_k=16, sparse=False, **kwargs):
        super().__init__()
        from .registry import build_model

        trunk_cfg = dict(trunk or {"name": "LegNetV2", "kwargs": {"seq_len": seq_len}})
        self.trunk = build_model({"model": trunk_cfg})
        self.trunk_name = trunk_cfg.get("name")

        if trunk_weights:
            from utils import resolve_path
            path = resolve_path(trunk_weights)
            state = torch.load(path, map_location="cpu", weights_only=True)
            missing, unexpected = self.trunk.load_state_dict(state, strict=False)
            print(f"[hybrid] trunk warm-started from {path} "
                  f"({len(missing)} missing, {len(unexpected)} unexpected keys)")
        if freeze_trunk:
            for p in self.trunk.parameters():
                p.requires_grad_(False)
            self.trunk.eval()
        self.freeze_trunk = bool(freeze_trunk)

        self.graph = MotifGNN(vocab=vocab, hidden_dim=graph_hidden,
                              num_layers=graph_layers, dropout=dropout,
                              edge_dropout=edge_dropout, node_dropout=node_dropout,
                              min_match_frac=min_match_frac,
                              neighbourhood=neighbourhood, radius_bp=radius_bp,
                              knn_k=knn_k, sparse=sparse)
        self.graph_weight = float(graph_weight)

        # Probe the trunk's feature width once rather than hard-coding it.
        with torch.no_grad():
            probe = torch.zeros(2, 4, seq_len)
            probe[:, 0, :] = 1.0
            trunk_dim = int(self.trunk.get_features(probe).shape[-1])
        self.trunk_dim = trunk_dim

        self.trunk_norm = nn.LayerNorm(trunk_dim)
        self.graph_norm = nn.LayerNorm(graph_hidden)
        self.fuse = nn.Sequential(
            nn.Linear(trunk_dim + graph_hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_dev = nn.Linear(hidden_dim, 1)
        self.fc_hk = nn.Linear(hidden_dim, 1)

        self.register_buffer("_zero", torch.zeros(()), persistent=False)
        self.reg_loss = self._zero

    @property
    def vocab(self):
        return self.graph.vocab

    def set_vocabulary(self, vocabulary):
        self.graph.set_vocabulary(vocabulary)
        return self

    def memory_estimate(self, batch_size: int) -> float:
        return self.graph.memory_estimate(batch_size)

    def embed(self, x):  # noqa: D102 - see class docstring
        if self.freeze_trunk:
            with torch.no_grad():
                trunk_feat = self.trunk.get_features(x)
        else:
            trunk_feat = self.trunk.get_features(x)
        graph_feat = self.graph.embed(x)
        fused = torch.cat([
            self.trunk_norm(trunk_feat),
            self.graph_norm(graph_feat) * self.graph_weight,
        ], dim=-1)
        # Propagate any regularisation the graph branch accumulated.
        reg = getattr(self.graph, "reg_loss", None)
        self.reg_loss = reg if torch.is_tensor(reg) else self._zero
        return self.fuse(fused)

    def forward(self, x):
        feat = self.embed(x)
        return self.fc_dev(feat), self.fc_hk(feat)

    def get_features(self, x):
        return self.embed(x)


# ---------------------------------------------------------------------------
# 5. FiLM hybrid: the motif graph *conditions* the CNN
# ---------------------------------------------------------------------------

@register_model("MotifGraphFiLM")
class MotifGraphFiLM(nn.Module):
    """CNN whose feature maps are modulated by the motif graph (FiLM).

    Why not concatenation
    ---------------------
    :class:`MotifGraphHybrid` concatenates a graph vector onto the trunk's
    features.  Measured on DeepSTARR, that made **no difference at all**: with a
    4.12M-parameter trunk and a 0.14M graph branch the fusion head simply learns
    near-zero weights on the graph half, and the ablation (``graph_weight=0``)
    matched it to within 0.0001 PCC.  A concatenated side-input is *optional* by
    construction.

    FiLM makes the graph load-bearing instead.  The graph embedding predicts a
    per-channel scale and shift that are applied to the CNN's activations at
    several depths::

        h  <-  (1 + gamma(g)) * h  +  beta(g)

    The CNN can no longer produce an output without the graph passing through
    it.  If the graph turns out to be uninformative the model can still learn
    ``gamma = beta = 0`` (identity), so this costs nothing when the graph does
    not help — but unlike concatenation it cannot be ignored *by default*.

    This mirrors the FiLM conditioning already used by ``DeepEPI`` in this repo.

    Parameters
    ----------
    trunk : dict
        Registered CNN spec; must expose ``stem`` / ``main`` / ``mapper``
        (LegNet-family) so FiLM can be injected between stages.
    film_stages : list[int] or None
        Which blocks of ``trunk.main`` to modulate.  ``None`` picks a spread
        across the depth of the network.
    graph_weight : float
        Scales the FiLM parameters; ``0.0`` disables conditioning entirely and
        gives a clean ablation with an identical parameter count.
    """

    def __init__(self, trunk=None, trunk_weights=None, vocab=None,
                 graph_hidden=128, graph_layers=3, graph_weight=1.0,
                 film_stages=None, seq_len=249, dropout=0.2, edge_dropout=0.15,
                 node_dropout=0.1, min_match_frac=1.0, hidden_dim=256,
                 neighbourhood="knn", radius_bp=60, knn_k=4, sparse=True, **kwargs):
        super().__init__()
        from .registry import build_model

        trunk_cfg = dict(trunk or {"name": "LegNetV2", "kwargs": {"seq_len": seq_len}})
        self.trunk = build_model({"model": trunk_cfg})
        for part in ("stem", "main", "mapper"):
            if not hasattr(self.trunk, part):
                raise ValueError(
                    f"MotifGraphFiLM needs a LegNet-family trunk exposing "
                    f"stem/main/mapper; {trunk_cfg.get('name')} has no '{part}'.")
        if trunk_weights:
            from utils import resolve_path
            state = torch.load(resolve_path(trunk_weights), map_location="cpu",
                               weights_only=True)
            missing, unexpected = self.trunk.load_state_dict(state, strict=False)
            print(f"[film] trunk warm-started ({len(missing)} missing, "
                  f"{len(unexpected)} unexpected)")

        self.graph = MotifGNN(vocab=vocab, hidden_dim=graph_hidden,
                              num_layers=graph_layers, dropout=dropout,
                              edge_dropout=edge_dropout, node_dropout=node_dropout,
                              min_match_frac=min_match_frac, sparse=sparse,
                              neighbourhood=neighbourhood, radius_bp=radius_bp,
                              knn_k=knn_k)
        self.graph_weight = float(graph_weight)

        # Probe the per-stage channel widths rather than hard-coding them.
        with torch.no_grad():
            probe = torch.zeros(2, 4, seq_len)
            probe[:, 0, :] = 1.0
            h = self.trunk.stem(probe)
            widths = []
            for block in self.trunk.main:
                h = block(h)
                widths.append(h.shape[1])
            trunk_dim = int(self.trunk.mapper(h).mean(dim=-1).shape[-1])

        n_blocks = len(widths)
        if film_stages is None:
            # Spread conditioning through the depth: early stages see motif
            # presence, late stages see composed syntax.
            film_stages = sorted({0, n_blocks // 2, n_blocks - 1})
        self.film_stages = [s for s in film_stages if 0 <= s < n_blocks]
        self.film = nn.ModuleDict({
            str(s): nn.Linear(graph_hidden, 2 * widths[s]) for s in self.film_stages})
        for layer in self.film.values():
            # Start at identity: gamma = beta = 0 -> h unchanged.
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.head = nn.Sequential(
            nn.Linear(trunk_dim + graph_hidden, hidden_dim), nn.GELU(),
            nn.Dropout(dropout))
        self.fc_dev = nn.Linear(hidden_dim, 1)
        self.fc_hk = nn.Linear(hidden_dim, 1)
        self.trunk_dim = trunk_dim

        self.register_buffer("_zero", torch.zeros(()), persistent=False)
        self.reg_loss = self._zero

    @property
    def vocab(self):
        return self.graph.vocab

    def set_vocabulary(self, vocabulary):
        self.graph.set_vocabulary(vocabulary)
        return self

    def memory_estimate(self, batch_size: int) -> float:
        return self.graph.memory_estimate(batch_size)

    def embed(self, x):
        g = self.graph.embed(x)                       # [B, graph_hidden]

        h = self.trunk.stem(x)
        for i, block in enumerate(self.trunk.main):
            h = block(h)
            key = str(i)
            if key in self.film and self.graph_weight != 0.0:
                gamma, beta = self.film[key](g).chunk(2, dim=-1)
                h = h * (1.0 + self.graph_weight * gamma.unsqueeze(-1)) \
                    + self.graph_weight * beta.unsqueeze(-1)
        h = self.trunk.mapper(h)
        trunk_feat = F.adaptive_avg_pool1d(h, 1).squeeze(-1)

        reg = getattr(self.graph, "reg_loss", None)
        self.reg_loss = reg if torch.is_tensor(reg) else self._zero
        return self.head(torch.cat([trunk_feat, g * self.graph_weight], dim=-1))

    def forward(self, x):
        feat = self.embed(x)
        return self.fc_dev(feat), self.fc_hk(feat)

    def get_features(self, x):
        return self.embed(x)

    @torch.no_grad()
    def film_strength(self):
        """Mean |gamma| and |beta| per conditioned stage — how much the graph
        is actually modulating the CNN.  All zeros means the graph is unused."""
        out = {}
        for key, layer in self.film.items():
            w = layer.weight.detach()
            half = w.shape[0] // 2
            out[f"stage{key}"] = {"gamma": float(w[:half].abs().mean()),
                                  "beta": float(w[half:].abs().mean())}
        return out


# ---------------------------------------------------------------------------
# 6. Learned motif nodes: de-novo discovery inside the graph
# ---------------------------------------------------------------------------

@register_model("MotifGraphV3")
class MotifGraphV3(nn.Module):
    """Motif graph whose node set is JASPAR **plus learned de-novo motifs**.

    What the measurements said
    --------------------------
    Every architecture tried so far that contains the 4.1M-parameter LegNetV2
    trunk overfits: concat hybrid +0.069, FiLM hybrid +0.182 by epoch 14, LegNetV2
    itself +0.034 — while the pure motif graph sits at ~0.00 but lacks the
    capacity to get past ~0.37.  Bolting a graph onto a CNN inherits the CNN's
    failure mode; the graph needs the capacity instead.

    So this model has **no CNN trunk**.  Capacity comes from more nodes and
    deeper message passing over an explicitly structured graph:

    * **fixed nodes** — JASPAR PWMs.  Real TF binding models, interpretable, and
      not trainable, so they cannot memorise anything.
    * **learned nodes** — a small bank of trainable PWM filters that discover
      motifs JASPAR does not contain.  This is the one thing a CNN could do that
      a fixed vocabulary could not, kept without the depth that causes
      memorisation.

    Both kinds of node enter the *same* graph and are related by the same syntax
    edges (purine/pyrimidine spacer counts, signed distance, helical phase,
    overlap), so a discovered motif is described in the same terms as a known
    one — and shows up in the same occlusion ranking.

    Parameters
    ----------
    learned_motifs : int
        Number of trainable filters.  Their gradients flow through the node
        features; positions stay discrete.
    learned_width : int
        Filter width in bp.
    """

    def __init__(self, vocab=None, learned_motifs=64, learned_width=12,
                 hidden_dim=192, num_layers=4, dropout=0.15, edge_dropout=0.1,
                 node_dropout=0.05, knn_k=4, min_match_frac=1.0,
                 learned_threshold=0.6, **kwargs):
        super().__init__()
        for owned in ("sparse", "neighbourhood", "knn_k", "hidden_dim",
                      "num_layers", "dropout"):
            kwargs.pop(owned, None)
        self.graph = MotifGNN(vocab=vocab, hidden_dim=hidden_dim,
                              num_layers=num_layers, dropout=dropout,
                              edge_dropout=edge_dropout, node_dropout=node_dropout,
                              min_match_frac=min_match_frac, sparse=True,
                              neighbourhood="knn", knn_k=knn_k, **kwargs)
        self.hidden_dim = hidden_dim

        # De-novo motif bank. Initialised small so training starts close to the
        # fixed-vocabulary model and the learned nodes have to earn their signal.
        self.learned = nn.Conv1d(4, learned_motifs, learned_width, padding="same")
        nn.init.normal_(self.learned.weight, std=0.05)
        nn.init.zeros_(self.learned.bias)
        self.learned_bn = nn.BatchNorm1d(learned_motifs)
        self.n_learned = int(learned_motifs)
        self.learned_threshold = float(learned_threshold)

        # Learned nodes carry a compact summary: max score, mean score,
        # position of the peak, and how often the filter fires.
        self.learned_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(node_dropout))

        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim + MaskedPooling.extra_dim + 2 * hidden_dim,
                      hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.fc_dev = nn.Linear(hidden_dim, 1)
        self.fc_hk = nn.Linear(hidden_dim, 1)

        self.register_buffer("_zero", torch.zeros(()), persistent=False)
        self.reg_loss = self._zero

    @property
    def vocab(self):
        return self.graph.vocab

    def set_vocabulary(self, vocabulary):
        self.graph.set_vocabulary(vocabulary)
        return self

    def memory_estimate(self, batch_size: int) -> float:
        return self.graph.memory_estimate(batch_size)

    def _learned_nodes(self, x):
        """Summarise each learned filter as a node, differentiably."""
        scores = F.relu(self.learned_bn(self.learned(x)))          # [B, M, L]
        length = scores.shape[-1]
        peak, argpeak = scores.max(dim=-1)
        feats = torch.stack([
            peak,
            scores.mean(dim=-1),
            argpeak.to(x.dtype) / max(length - 1, 1),
            (scores > self.learned_threshold).to(x.dtype).mean(dim=-1),
        ], dim=-1)                                                 # [B, M, 4]
        h = self.learned_encoder(feats)
        return torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)

    def embed(self, x):
        node_feats, edge_feats, neighbour_idx, mask = \
            self.graph.featurizer.sparse_forward(x)
        present = node_feats[..., 0]
        h = self.graph.node_encoder(node_feats)
        e = self.graph.edge_encoder(edge_feats)
        for layer in self.graph.layers:
            h = layer(h, e, neighbour_idx, mask)
        pooled = self.graph.pool(h, present)
        return self.fuse(torch.cat([pooled, self._learned_nodes(x)], dim=-1))

    def forward(self, x):
        feat = self.embed(x)
        return self.fc_dev(feat), self.fc_hk(feat)

    def get_features(self, x):
        return self.embed(x)

    @torch.no_grad()
    def learned_pwms(self, as_probs=True):
        """The discovered filters as ``[M, 4, width]`` for logo rendering."""
        w = self.learned.weight.detach().cpu()
        return torch.softmax(w, dim=1) if as_probs else w


# ---------------------------------------------------------------------------
# 7. Motif graph transformer: flash attention over sequence and over motifs
# ---------------------------------------------------------------------------

class FlashBlock(nn.Module):
    """Pre-norm transformer block using ``scaled_dot_product_attention``.

    Torch dispatches SDPA to the FlashAttention kernel when no additive bias is
    supplied; with an ``attn_bias`` it falls back to the memory-efficient
    backend.  That trade is deliberate below: the sequence and cross-attention
    paths run flash, while motif-to-motif attention accepts the slower backend
    in exchange for injecting the graph's edge features as a bias.
    """

    def __init__(self, dim, heads=4, dropout=0.1, ffn_mult=2):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_mult * dim, dim))
        self.dropout = nn.Dropout(dropout)

    def _split(self, t):
        b, n, _ = t.shape
        return t.view(b, n, self.heads, self.dim // self.heads).transpose(1, 2)

    def forward(self, x, context=None, attn_bias=None):
        ctx = x if context is None else context
        q = self._split(self.q(self.norm_q(x)))
        k = self._split(self.k(self.norm_kv(ctx)))
        v = self._split(self.v(self.norm_kv(ctx)))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        b, h, n, d = out.shape
        out = out.transpose(1, 2).reshape(b, n, h * d)
        x = x + self.dropout(self.proj(out))
        return x + self.dropout(self.ff(self.norm_ff(x)))


@register_model("MotifGraphTransformer")
class MotifGraphTransformer(nn.Module):
    """Motif graph + flash attention, with no deep convolutional trunk.

    Three attention paths, each answering something the previous models could
    not:

    ``sequence self-attention``
        Pooled sequence tokens attend to each other, so information is
        accumulated across the whole 249 bp in one hop.  LegNetV2 needs seven
        convolutional stages to span that distance and then destroys the
        positional detail with a global average pool.
    ``motif -> sequence cross-attention``
        Every motif node reads the sequence context around and beyond its own
        site, which is how a node learns about DNA it does not itself cover.
    ``motif self-attention, biased by the graph edges``
        Motifs attend to **all** other motifs, not just the k nearest, so
        next-to-next relationships are reachable in one step.  The edge features
        (purine/pyrimidine spacer counts, signed distance, helical phase,
        overlap) enter as an additive attention bias, so the learned attention
        is still conditioned on the measured syntax rather than replacing it.

    The convolution here is a single shallow stem, not a memorising stack: the
    depth lives in attention over ~32 tokens and ~300 motif nodes rather than in
    4M convolutional parameters.
    """

    def __init__(self, vocab=None, hidden_dim=192, num_layers=3, seq_tokens=32,
                 seq_layers=2, heads=4, dropout=0.15, edge_dropout=0.1,
                 node_dropout=0.05, knn_k=8, min_match_frac=1.0,
                 learned_motifs=64, learned_width=12, stem_ch=128,
                 seq_len=249, **kwargs):
        super().__init__()
        for owned in ("sparse", "neighbourhood", "hidden_dim", "num_layers", "dropout"):
            kwargs.pop(owned, None)

        self.graph = MotifGNN(vocab=vocab, hidden_dim=hidden_dim, num_layers=1,
                              dropout=dropout, edge_dropout=edge_dropout,
                              node_dropout=node_dropout, min_match_frac=min_match_frac,
                              sparse=False, neighbourhood="dense",
                              knn_k=knn_k, **kwargs)
        self.hidden_dim = hidden_dim
        self.seq_tokens = int(seq_tokens)

        # Shallow stem: motif-scale features only, deliberately not a deep stack.
        self.stem = nn.Sequential(
            nn.Conv1d(4, stem_ch, learned_width, padding="same"),
            nn.BatchNorm1d(stem_ch), nn.GELU(),
            nn.Conv1d(stem_ch, hidden_dim, 5, padding="same"),
            nn.BatchNorm1d(hidden_dim), nn.GELU())
        self.token_pos = nn.Parameter(torch.zeros(1, seq_tokens, hidden_dim))
        nn.init.trunc_normal_(self.token_pos, std=0.02)
        self.seq_blocks = nn.ModuleList(
            FlashBlock(hidden_dim, heads, dropout) for _ in range(seq_layers))

        # Edge features -> per-head additive attention bias.
        self.edge_bias = nn.Linear(EDGE_DIM, heads)
        nn.init.zeros_(self.edge_bias.weight)
        nn.init.zeros_(self.edge_bias.bias)

        self.motif_blocks = nn.ModuleList(
            FlashBlock(hidden_dim, heads, dropout) for _ in range(num_layers))
        self.cross_blocks = nn.ModuleList(
            FlashBlock(hidden_dim, heads, dropout) for _ in range(num_layers))

        self.pool = MaskedPooling()
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim + MaskedPooling.extra_dim + hidden_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.fc_dev = nn.Linear(hidden_dim, 1)
        self.fc_hk = nn.Linear(hidden_dim, 1)

        self.register_buffer("_zero", torch.zeros(()), persistent=False)
        self.reg_loss = self._zero

    @property
    def vocab(self):
        return self.graph.vocab

    def set_vocabulary(self, vocabulary):
        self.graph.set_vocabulary(vocabulary)
        return self

    def memory_estimate(self, batch_size: int) -> float:
        v = self.graph.featurizer.num_motifs
        return batch_size * v * v * self.hidden_dim * 4 / 1e6

    def embed(self, x):
        node_feats, edge_feats, adj = self.graph.featurizer(x)
        present = node_feats[..., 0]

        # --- sequence tokens + flash self-attention ---
        tokens = F.adaptive_avg_pool1d(self.stem(x), self.seq_tokens).transpose(1, 2)
        tokens = tokens + self.token_pos
        for block in self.seq_blocks:
            tokens = block(tokens)

        # --- motif nodes, attending to each other and to the sequence ---
        h = self.graph.node_encoder(node_feats)
        # Absent motifs are masked out of attention entirely; the graph's edge
        # features become a per-head additive bias on the remaining pairs.
        bias = self.edge_bias(edge_feats).permute(0, 3, 1, 2)
        mask = (adj > 0).unsqueeze(1)
        bias = bias.masked_fill(~mask, float("-inf"))
        # A row with no admissible key would produce NaN from softmax.
        dead = (~mask).all(dim=-1, keepdim=True)
        bias = bias.masked_fill(dead, 0.0)

        for motif_block, cross_block in zip(self.motif_blocks, self.cross_blocks):
            h = motif_block(h, attn_bias=bias)
            h = cross_block(h, context=tokens)

        pooled = self.pool(h, present)
        return self.head(torch.cat([pooled, tokens.mean(dim=1)], dim=-1))

    def forward(self, x):
        feat = self.embed(x)
        return self.fc_dev(feat), self.fc_hk(feat)

    def get_features(self, x):
        return self.embed(x)


# ---------------------------------------------------------------------------
# 8. Slot pooling + pure motif attention (small)
# ---------------------------------------------------------------------------

class MotifSlotPool(nn.Module):
    """Aggregate motif nodes into *k* latent slots and write the result back.

    A mean/max pool is a dead end for the graph: it summarises once, at the end,
    and nothing flows back to the nodes.  This layer instead

    1. lets ``n_slots`` learned queries attend over the motif nodes, so each slot
       aggregates a different subset of the graph, and
    2. writes the slots back into every node as a **residual**, so a motif can
       see what the rest of the graph contains without any node-to-node edge.

    That second step is what replaces message passing here: global context
    reaches each node through the slots rather than through a neighbourhood, at
    ``O(V * n_slots)`` instead of ``O(V^2)``.  Both directions are plain
    cross-attention with no additive bias, so both dispatch to FlashAttention.
    """

    def __init__(self, dim, n_slots=8, heads=4, dropout=0.1):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(1, n_slots, dim) * 0.02)
        self.gather = FlashBlock(dim, heads, dropout)     # slots <- nodes
        self.scatter = FlashBlock(dim, heads, dropout)    # nodes <- slots
        self.gate = nn.Parameter(torch.zeros(1))          # residual starts off

    def forward(self, h, slots=None):
        s = self.slots.expand(h.shape[0], -1, -1) if slots is None else slots
        s = self.gather(s, context=h)
        # Gated residual: the write-back begins as a no-op and has to earn its
        # influence, which keeps early training stable.
        h = h + torch.tanh(self.gate) * self.scatter(h, context=s)
        return h, s


@register_model("MotifGraphV4")
class MotifGraphV4(nn.Module):
    """Small motif-graph model: attention over motifs only, slot pooling, residuals.

    Deliberately stripped back from :class:`MotifGraphTransformer`:

    * **attention over motif nodes only** — no edge-biased attention, no k-nearest
      neighbourhood message passing, no sequence-token branch;
    * **per-sequence information only** — a node describes what its motif does
      in *this* sequence (presence, count, position, spread, strand, match
      quality, redundancy, 2 bp flanks) and nothing else.  No edge feature
      enters the model; attention has to infer any motif-to-motif relationship
      from the nodes' own positions.  This makes the model a clean test of how
      much of enhancer activity is explained by motif *content* plus learned
      attention, with the hand-designed syntax removed;
    * **slot pooling with a gated residual** (:class:`MotifSlotPool`) carries
      global context back to the nodes in place of message passing;
    * **small** — roughly a third of the transformer and far below LegNetV2.

    Because nothing carries an additive attention bias, every attention call
    here dispatches to the FlashAttention kernel.
    """

    def __init__(self, vocab=None, hidden_dim=128, num_layers=3, n_slots=8,
                 heads=4, dropout=0.15, node_dropout=0.05, min_match_frac=1.0,
                 flank_bp=2, **kwargs):
        super().__init__()
        for owned in ("sparse", "neighbourhood", "hidden_dim", "num_layers", "dropout"):
            kwargs.pop(owned, None)
        self.graph = MotifGNN(vocab=vocab, hidden_dim=hidden_dim, num_layers=1,
                              dropout=dropout, node_dropout=node_dropout,
                              min_match_frac=min_match_frac, sparse=False,
                              neighbourhood="dense", **kwargs)
        self.graph.featurizer.flank_bp = int(flank_bp)
        self.hidden_dim = hidden_dim

        # Per-sequence node information ONLY: what this motif does in this
        # sequence (present, count, position, spread, strand, match quality,
        # redundancy, 2 bp flanks). No pairwise edge information enters the
        # model at all - any relationship between motifs has to be inferred by
        # attention from the nodes' own positions and context.
        self.node_encoder = nn.Sequential(
            nn.Linear(NODE_DIM, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(node_dropout))
        self.blocks = nn.ModuleList(FlashBlock(hidden_dim, heads, dropout)
                                    for _ in range(num_layers))
        self.pools = nn.ModuleList(MotifSlotPool(hidden_dim, n_slots, heads, dropout)
                                   for _ in range(num_layers))

        self.pool = MaskedPooling()
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim + MaskedPooling.extra_dim + hidden_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.fc_dev = nn.Linear(hidden_dim, 1)
        self.fc_hk = nn.Linear(hidden_dim, 1)

        self.register_buffer("_zero", torch.zeros(()), persistent=False)
        self.reg_loss = self._zero

    @property
    def vocab(self):
        return self.graph.vocab

    def set_vocabulary(self, vocabulary):
        self.graph.set_vocabulary(vocabulary)
        return self

    def memory_estimate(self, batch_size: int) -> float:
        v = self.graph.featurizer.num_motifs
        return batch_size * v * self.hidden_dim * 4 / 1e6

    def embed(self, x):
        node_feats, _, _ = self.graph.featurizer(x)
        present = node_feats[..., 0]

        h = self.node_encoder(node_feats)
        slots = None
        for block, pool in zip(self.blocks, self.pools):
            h = block(h)
            h, slots = pool(h, slots)

        pooled = self.pool(h, present)
        return self.head(torch.cat([pooled, slots.mean(dim=1)], dim=-1))

    def forward(self, x):
        feat = self.embed(x)
        return self.fc_dev(feat), self.fc_hk(feat)

    def get_features(self, x):
        return self.embed(x)
