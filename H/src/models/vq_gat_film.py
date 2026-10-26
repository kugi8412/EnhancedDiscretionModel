from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class ResidualVectorQuantiser(nn.Module):
    """Quantise, but keep what quantisation could not represent.

    A standard VQ layer replaces ``z`` with its nearest codebook entry and
    discards the difference.  Here the difference is kept and concatenated, so
    the layer is *lossless on the forward path* while still producing discrete
    indices.  The codebook becomes an interpretable summary rather than a
    bottleneck, which is what makes it safe on a task where continuous affinity
    demonstrably matters.

    Uses exponential-moving-average codebook updates rather than a codebook
    loss: EMA is markedly less prone to the codebook collapse that makes plain
    VQ-VAE unstable, and it keeps the codebook out of the optimiser.
    """

    def __init__(self, dim, codebook_size=512, decay=0.99, commit_weight=0.25,
                 eps=1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = int(codebook_size)
        self.decay = float(decay)
        self.commit_weight = float(commit_weight)
        self.eps = float(eps)

        embed = torch.randn(self.codebook_size, dim) * (dim ** -0.5)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.ones(self.codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z):
        # z: [B, C, L] -> flatten to [N, C] for nearest-neighbour lookup.
        b, c, length = z.shape
        flat = z.permute(0, 2, 1).reshape(-1, c)

        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embed.t()
                + self.embed.pow(2).sum(1))
        idx = dist.argmin(dim=1)
        quant = self.embed[idx].view(b, length, c).permute(0, 2, 1)

        if self.training:
            self._ema_update(flat, idx)

        # Straight-through: gradients flow to the encoder as if quantisation
        # were the identity.
        commit = F.mse_loss(z, quant.detach())
        quant_st = z + (quant - z).detach()

        residual = z - quant.detach()      # what the codebook could NOT express
        self.loss = self.commit_weight * commit
        self.indices = idx.view(b, length)
        return torch.cat([quant_st, residual], dim=1)

    @torch.no_grad()
    def _ema_update(self, flat, idx):
        onehot = F.one_hot(idx, self.codebook_size).to(flat.dtype)
        counts = onehot.sum(0)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
        n = self.cluster_size.sum()
        weights = ((self.cluster_size + self.eps) / (n + self.codebook_size * self.eps)) * n
        self.embed.copy_(self.embed_avg / weights.unsqueeze(1))

    @torch.no_grad()
    def perplexity(self):
        """Effective codebook usage.  Collapse shows up here as a value far
        below ``codebook_size`` — the number to watch during training."""
        p = self.cluster_size / self.cluster_size.sum()
        return float(torch.exp(-(p * (p + 1e-10).log()).sum()))


class GraphAttentionLayer(nn.Module):
    """GAT over motif nodes, with edge features as an additive logit bias.

    Measured on this data: a per-node *summary* of the edges is worth nothing
    (0.333 Dev, identical to using no edges), while the same features entering
    as a per-head attention bias reached 0.538.  The signal is in which
    specific pair sits at which distance, so the bias formulation is the one
    that survives.  Multi-head, with the standard LeakyReLU-scored attention of
    GAT rather than scaled dot-product, so each head learns an explicit
    compatibility function over node pairs.
    """

    def __init__(self, dim, heads=4, edge_dim=None, dropout=0.1, negative_slope=0.2):
        super().__init__()
        assert dim % heads == 0, "dim must divide evenly across heads"
        self.heads, self.head_dim = heads, dim // heads
        self.proj = nn.Linear(dim, dim)
        self.att_src = nn.Parameter(torch.randn(heads, self.head_dim) * 0.1)
        self.att_dst = nn.Parameter(torch.randn(heads, self.head_dim) * 0.1)
        self.edge_bias = nn.Linear(edge_dim, heads) if edge_dim else None
        if self.edge_bias is not None:
            nn.init.zeros_(self.edge_bias.weight)
            nn.init.zeros_(self.edge_bias.bias)
        self.slope = float(negative_slope)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, mask=None, edge_feats=None):
        b, v, _ = h.shape
        x = self.proj(self.norm(h)).view(b, v, self.heads, self.head_dim)

        # GAT scoring: e_ij = LeakyReLU(a_src . h_i + a_dst . h_j), which is
        # additively separable and so costs O(V) to form, not O(V^2 * d).
        src = (x * self.att_src).sum(-1)                     # [B, V, H]
        dst = (x * self.att_dst).sum(-1)
        logits = F.leaky_relu(src.unsqueeze(2) + dst.unsqueeze(1), self.slope)
        logits = logits.permute(0, 3, 1, 2)                  # [B, H, V, V]

        if edge_feats is not None and self.edge_bias is not None:
            logits = logits + self.edge_bias(edge_feats).permute(0, 3, 1, 2)
        if mask is not None:
            keep = mask.unsqueeze(1).unsqueeze(2)            # [B, 1, 1, V]
            logits = logits.masked_fill(~keep, float("-inf"))
            # A node with no admissible neighbour would softmax over all -inf
            # and produce NaN; give those rows a uniform, harmless attention.
            dead = (~keep).all(dim=-1, keepdim=True)
            logits = logits.masked_fill(dead, 0.0)

        attn = self.drop(logits.softmax(dim=-1))
        out = torch.einsum("bhij,bjhd->bihd", attn, x).reshape(b, v, -1)
        return h + self.drop(out)


class BinnedHead(nn.Module):
    """Predict a distribution over ordinal bins; read out mean and variance.

    The expectation over bin centres is the point prediction used for PCC, so
    the output stays continuous despite the discrete parameterisation.  The
    distribution's spread is the model's uncertainty, obtained for free.
    """

    def __init__(self, dim, n_bins=20, lo=-4.0, hi=6.0, dropout=0.1):
        super().__init__()
        self.n_bins = int(n_bins)
        centres = torch.linspace(lo, hi, self.n_bins)
        self.register_buffer("centres", centres)
        self.register_buffer("width", torch.tensor((hi - lo) / max(self.n_bins - 1, 1)))
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(dim, self.n_bins))

    def forward(self, h):
        logits = self.net(h)
        p = logits.softmax(dim=-1)
        mean = (p * self.centres).sum(-1)
        var = (p * (self.centres - mean.unsqueeze(-1)).pow(2)).sum(-1)
        return logits, mean, var

    def soft_targets(self, y):
        """Gaussian-smoothed one-hot over bins.

        A hard one-hot would throw away the ordering that makes these bins
        meaningful — mistaking bin 3 for bin 4 must cost less than mistaking it
        for bin 19.  Smoothing with a kernel one bin wide encodes that, and
        also softens the label noise the 0.93 replicate ceiling implies.
        """
        d = (y.unsqueeze(-1) - self.centres) / self.width
        return (-0.5 * d.pow(2)).softmax(dim=-1)


@register_model("VQGatFiLM")
class VQGatFiLM(nn.Module):
    """VQ encoder -> LegNet trunk -> GAT + FiLM -> binned dual head.

    ``x -> conv encoder -> residual VQ -> LegNet blocks -> motif GAT (FiLM) ->
    20-bin distributions for Dev and Hk``

    The motif graph is built from the continuous occupancy track, so nodes
    carry graded affinity rather than binary presence.  FiLM lets the graph
    modulate the trunk multiplicatively: a concatenated graph branch was
    measured to be ignorable (ablation matched to 0.0001 PCC), FiLM is not.
    """

    def __init__(self, vocab=None, seq_len=249, enc_ch=128, enc_ks=7,
                 codebook_size=512, vq_dim=128, commit_weight=0.25,
                 ef_block_sizes=(192, 128, 96), pool_sizes=(2, 2, 2),
                 resize_factor=3, ef_ks=7, graph_dim=128, gat_layers=2, heads=4,
                 n_bins=20, bin_lo=-4.0, bin_hi=6.0, hidden_dim=256,
                 dropout=0.2, drop_path=0.1, **kwargs):
        super().__init__()
        from .legnet import LegNetV2
        from .motif_film_v3 import StochasticDepth
        from .motif_occupancy import ContinuousPWMScanner, OccupancyRefiner
        from motif_graph import MotifVocabulary

        # --- continuous motif view (graph nodes) ---
        self._vocab = MotifVocabulary.from_config(vocab)
        self.scanner = ContinuousPWMScanner(self._vocab)
        self.refiner = OccupancyRefiner(len(self._vocab))
        v = len(self._vocab)

        # --- VQ encoder over raw sequence ---
        self.encoder = nn.Sequential(
            nn.Conv1d(4, enc_ch, enc_ks, padding=enc_ks // 2), nn.BatchNorm1d(enc_ch),
            nn.SiLU(), nn.Conv1d(enc_ch, vq_dim, 1))
        self.vq = ResidualVectorQuantiser(vq_dim, codebook_size,
                                          commit_weight=commit_weight)

        # Trunk sees the VQ output (code + residual = 2 * vq_dim) alongside the
        # refined motif occupancy, so it has both a learned and a named view.
        trunk_in = 2 * vq_dim + v
        self.trunk = LegNetV2(
            in_ch=trunk_in, stem_ch=ef_block_sizes[0], stem_ks=ef_ks, ef_ks=ef_ks,
            ef_block_sizes=tuple(ef_block_sizes), pool_sizes=tuple(pool_sizes),
            resize_factor=resize_factor, seq_len=seq_len)

        with torch.no_grad():
            probe = torch.zeros(2, trunk_in, seq_len)
            h = self.trunk.stem(probe)
            widths = []
            for block in self.trunk.main:
                h = block(h)
                widths.append(h.shape[1])
            trunk_dim = int(self.trunk.mapper(h).mean(dim=-1).shape[-1])

        if drop_path > 0:
            n = len(self.trunk.main)
            self.trunk.main = nn.ModuleList([
                StochasticDepth(b, drop_path * (i + 1) / n)
                for i, b in enumerate(self.trunk.main)])

        # --- GAT over motif nodes, 3 descriptors each from the occupancy ---
        self.node_encode = nn.Linear(3, graph_dim)
        self.node_identity = nn.Parameter(torch.randn(v, graph_dim) * 0.02)
        self.gat = nn.ModuleList(
            GraphAttentionLayer(graph_dim, heads, edge_dim=None, dropout=dropout)
            for _ in range(int(gat_layers)))
        self.graph_norm = nn.LayerNorm(graph_dim)

        self.film = nn.ModuleDict({
            str(i): nn.Linear(graph_dim, 2 * w) for i, w in enumerate(widths)})
        for layer in self.film.values():
            nn.init.zeros_(layer.weight)      # identity at init
            nn.init.zeros_(layer.bias)

        self.mapper = nn.Sequential(
            nn.Linear(trunk_dim + graph_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.head_dev = BinnedHead(hidden_dim, n_bins, bin_lo, bin_hi, dropout)
        self.head_hk = BinnedHead(hidden_dim, n_bins, bin_lo, bin_hi, dropout)

        self.register_buffer("_zero", torch.zeros(()), persistent=False)
        self.reg_loss = self._zero

    @property
    def vocab(self):
        return self._vocab

    def set_vocabulary(self, vocabulary):
        self._vocab = vocabulary
        device = next(self.parameters()).device
        self.scanner = ContinuousPWMScanner(vocabulary).to(device)
        return self

    def memory_estimate(self, batch_size):
        return batch_size * len(self._vocab) * 249 * 4 / 1e6

    def _graph(self, occ):
        """Motif nodes from the CONTINUOUS track: peak, total, position."""
        peak = occ.amax(dim=-1)
        total = occ.mean(dim=-1)
        where = occ.argmax(dim=-1).to(occ.dtype) / max(occ.shape[-1] - 1, 1)
        feats = torch.stack([peak, total, where], dim=-1)

        h = self.node_encode(feats) + self.node_identity.unsqueeze(0)
        present = peak > peak.mean(dim=1, keepdim=True)   # relative, not a fixed cut
        for layer in self.gat:
            h = layer(h, mask=present)
        return self.graph_norm(h).mean(dim=1)

    def embed(self, x):
        occ = self.scanner(x)
        g = self._graph(occ)

        z = self.vq(self.encoder(x))
        self.reg_loss = self.vq.loss
        h_in = torch.cat([z, self.refiner(occ)], dim=1)

        h = self.trunk.stem(h_in)
        for i, block in enumerate(self.trunk.main):
            h = block(h)
            key = str(i)
            if key in self.film:
                gamma, beta = self.film[key](g).chunk(2, dim=-1)
                h = h * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        h = self.trunk.mapper(h)
        feat = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        return self.mapper(torch.cat([feat, g], dim=-1))

    def forward(self, x, return_distribution=False):
        feat = self.embed(x)
        ld, md, vd = self.head_dev(feat)
        lh, mh, vh = self.head_hk(feat)
        if return_distribution:
            return {"dev_logits": ld, "hk_logits": lh, "dev": md, "hk": mh,
                    "dev_var": vd, "hk_var": vh}
        # Default signature matches every other model in the repo, so the
        # existing training and evaluation loops work unchanged.
        return md.unsqueeze(-1), mh.unsqueeze(-1)

    def get_features(self, x):
        return self.embed(x)
