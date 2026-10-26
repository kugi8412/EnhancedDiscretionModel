from __future__ import annotations

import torch
import torch.nn as nn

from motif_graph import EDGE_DIM, NODE_DIM
from .motif_gnn import FlashBlock, MaskedPooling, MotifGNN
from .registry import register_model


class GraphSEBlock(nn.Module):
    """The LegNet EffBlock, translated to nodes.

    Expand the channels, re-weight them from a summary of the whole graph, then
    project back and keep the input alongside the output.  The squeeze is taken
    over *present* nodes only, so a motif is gated by what the sequence actually
    contains rather than by how many vocabulary entries happen to be absent.
    """

    def __init__(self, dim, expand=2, se_reduction=4, dropout=0.1):
        super().__init__()
        inner = dim * expand
        squeezed = max(inner // se_reduction, 8)
        self.expand = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, inner), nn.SiLU())
        self.se = nn.Sequential(
            nn.Linear(inner, squeezed), nn.SiLU(),
            nn.Linear(squeezed, inner), nn.Sigmoid())
        self.project = nn.Linear(inner, dim)
        self.merge = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, present):
        z = self.expand(h)
        mask = present.unsqueeze(-1)
        pooled = (z * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        z = z * self.se(pooled).unsqueeze(1)
        out = self.dropout(self.project(z))
        return self.merge(torch.cat([h, out], dim=-1))


class GraphPoolStage(nn.Module):
    """Progressive node reduction: LegNet max-pooling for graphs.

    Keeps the ``ratio`` highest-scoring nodes and folds the discarded ones into
    the survivors by attention, so a pruned motif is redistributed rather than
    thrown away.  Node count falls geometrically, which is what keeps the deeper
    stages cheap and the model small.
    """

    def __init__(self, dim, ratio=0.5, heads=4, dropout=0.1, min_nodes=8):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.absorb = FlashBlock(dim, heads, dropout)
        self.ratio = float(ratio)
        self.min_nodes = int(min_nodes)

    def forward(self, h, present):
        b, v, dim = h.shape
        keep = max(self.min_nodes, min(v, int(v * self.ratio)))
        # Push absent motifs to the bottom so real occurrences survive.
        score = self.score(h).squeeze(-1) + (present - 1.0) * 1e4
        idx = score.topk(keep, dim=1).indices
        kept = torch.gather(h, 1, idx.unsqueeze(-1).expand(b, keep, dim))
        # Survivors read the full node set once, so dropped motifs still land.
        kept = self.absorb(kept, context=h)
        return kept, torch.gather(present, 1, idx)


class FiLMTaskHead(nn.Module):
    """One shared head, FiLM-conditioned on the task.

    Two independent linear heads share nothing but the trunk.  Here a learned
    task embedding produces a scale and shift, so Dev and Hk read the *same*
    representation through different modulations: all capacity is shared and the
    tasks differ only in how they interrogate it.  Dev and Hk correlate 0.43 in
    this data, so they have both structure to share and structure to separate.

    The embedding initialises to zero, i.e. identity, so training starts with
    the two tasks reading an identical representation.
    """

    def __init__(self, dim, n_tasks=2, hidden=None, dropout=0.1):
        super().__init__()
        hidden = hidden or dim
        self.task = nn.Embedding(n_tasks, 2 * dim)
        nn.init.zeros_(self.task.weight)
        self.body = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.n_tasks = n_tasks

    def forward(self, h):
        outs = []
        for t in range(self.n_tasks):
            idx = torch.full((1,), t, device=h.device, dtype=torch.long)
            gamma, beta = self.task(idx).chunk(2, dim=-1)
            outs.append(self.body(h * (1.0 + gamma) + beta))
        return outs


@register_model("MotifGraphLegNet")
class MotifGraphLegNet(nn.Module):
    def __init__(self, vocab=None, hidden_dim=128, num_stages=3, pool_ratio=0.5,
                 heads=4, expand=2, se_reduction=4, dropout=0.15,
                 node_dropout=0.05, min_match_frac=1.0, flank_bp=2,
                 attn_layers=2, **kwargs):
        super().__init__()
        for owned in ("sparse", "neighbourhood", "hidden_dim", "num_layers", "dropout"):
            kwargs.pop(owned, None)
        self.graph = MotifGNN(vocab=vocab, hidden_dim=hidden_dim, num_layers=1,
                              dropout=dropout, node_dropout=node_dropout,
                              min_match_frac=min_match_frac, sparse=False,
                              neighbourhood="dense", **kwargs)
        self.graph.featurizer.flank_bp = int(flank_bp)
        self.hidden_dim = hidden_dim

        # Node occurrence + a summary of that node relationships, so
        # nucleotide-level distance reaches the model without edge attention.
        self.node_encoder = nn.Sequential(
            nn.Linear(NODE_DIM, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(node_dropout))
        self.n_attn = int(attn_layers)
        self.edge_bias = nn.Linear(EDGE_DIM, heads)
        nn.init.zeros_(self.edge_bias.weight)
        nn.init.zeros_(self.edge_bias.bias)
        self.attn = nn.ModuleList(FlashBlock(hidden_dim, heads, dropout)
                                  for _ in range(self.n_attn))

        self.blocks = nn.ModuleList(
            GraphSEBlock(hidden_dim, expand, se_reduction, dropout)
            for _ in range(num_stages))
        self.pools = nn.ModuleList(
            GraphPoolStage(hidden_dim, pool_ratio, heads, dropout)
            for _ in range(num_stages))

        self.final = MaskedPooling()
        self.mapper = nn.Sequential(
            nn.Linear(2 * hidden_dim + MaskedPooling.extra_dim, hidden_dim), nn.SiLU())
        self.head = FiLMTaskHead(hidden_dim, n_tasks=2, dropout=dropout)

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
        node_feats, edge_feats, adj = self.graph.featurizer(x)
        present = node_feats[..., 0]

        h = self.node_encoder(node_feats)

        # Pairwise edge conditioning: per-head additive bias on the attention
        # logits, masked so absent motifs are unreachable. A row with no
        # admissible key would softmax to NaN, so those fall back to zero bias.
        bias = self.edge_bias(edge_feats).permute(0, 3, 1, 2)
        allowed = (adj > 0).unsqueeze(1)
        bias = bias.masked_fill(~allowed, float("-inf"))
        bias = bias.masked_fill((~allowed).all(dim=-1, keepdim=True), 0.0)
        for block in self.attn:
            h = block(h, attn_bias=bias)

        for block, pool in zip(self.blocks, self.pools):
            h = block(h, present)
            h, present = pool(h, present)

        return self.mapper(self.final(h, present))

    def forward(self, x):
        dev, hk = self.head(self.embed(x))
        return dev, hk

    def get_features(self, x):
        return self.embed(x)
