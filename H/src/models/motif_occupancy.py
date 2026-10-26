from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from motif_graph import MotifVocabulary
from .motif_legnet import FiLMTaskHead
from .registry import register_model


class ContinuousPWMScanner(nn.Module):
    """Score every motif at every position; return a continuous occupancy track.

    The convolution against one-hot DNA sums exactly one PWM row per position,
    so its response *is* the motif's log-odds score for that window.  Those
    responses are then mapped to the MATCH relative score ``(s - lo) / (hi - lo)``
    so motifs of different widths are comparable.

    Two details that are easy to get wrong and were both bugs earlier in this
    project:

    * **short motifs near the 3' end** — one kernel width is shared by all
      motifs, so the input is right-padded and every motif carries a validity
      mask; without it a 6-base motif would score against zero-padding and
      report a phantom occupancy in the last 15 positions;
    * **strand symmetry** — forward and reverse-complement kernels are scanned
      together and combined with a max, so the track is strand-invariant by
      construction rather than by augmentation.
    """

    def __init__(self, vocab: MotifVocabulary, trainable=False, temperature=0.0,
                 center_align=False):
        super().__init__()
        kernels, thresholds, widths = vocab.match_kernels()
        lo, hi = vocab.score_ranges()

        self.num_motifs = len(vocab)
        self.max_len = int(kernels.shape[-1])
        self.temperature = float(temperature)
        # WYROWNANIE DO SRODKA MOTYWU. Patrz `_na_srodek`.
        self.center_align = bool(center_align)
        _w = widths[:self.num_motifs].long()
        self.register_buffer("_c", (_w - 1) // 2, persistent=False)
        self.register_buffer("_parzyste", (_w % 2 == 0).view(1, -1, 1),
                             persistent=False)

        if trainable:
            # Fine-tuning drifts channels away from their JASPAR identity, which
            # costs the interpretability the whole design is for.  Off by default.
            self.kernels = nn.Parameter(kernels)
        else:
            self.register_buffer("kernels", kernels, persistent=False)
        self.register_buffer("lo", lo.view(1, -1, 1), persistent=False)
        self.register_buffer("span", (hi - lo).clamp_min(1e-6).view(1, -1, 1),
                             persistent=False)
        self.register_buffer("widths", widths, persistent=False)
        self.register_buffer("thresholds", thresholds, persistent=False)

    def forward(self, x):
        b, _, length = x.shape
        pad = self.max_len - 1
        scores = F.conv1d(F.pad(x, (0, pad)), self.kernels)     # [B, 2V, L]

        rel = (scores - self.lo) / self.span                    # MATCH score
        # A window running off the end scored against zero-padding, so mask it.
        pos = torch.arange(length, device=x.device).view(1, 1, -1)
        last_valid = (length - self.widths).view(1, -1, 1)
        rel = torch.where(pos <= last_valid, rel, torch.zeros_like(rel))

        v = self.num_motifs
        occ = torch.maximum(rel[:, :v], rel[:, v:])             # strand-invariant
        if self.center_align:
            occ = self._na_srodek(occ)
        if self.temperature > 0:
            # Optional soft gate: sharpens toward strong sites while staying
            # differentiable and never fully discarding a weak one.
            occ = torch.sigmoid((occ - 0.85) / self.temperature)
        return occ.clamp(0.0, 1.0)

    def _na_srodek(self, occ):
        b, v, length = occ.shape
        # Krok 1: parzyste szerokosci -- maksimum dwoch sasiednich wyrownan.
        lewo = F.pad(occ, (1, 0))[..., :length]
        occ = torch.where(self._parzyste, torch.maximum(occ, lewo), occ)
        # Krok 2: przesuniecie kazdego kanalu w PRAWO o wlasne c_v.
        poz = torch.arange(length, device=occ.device)
        idx = (poz.view(1, -1) - self._c.view(-1, 1)).clamp_min(0)     # [V, L]
        occ = occ.gather(2, idx.view(1, v, length).expand(b, v, length))
        # Pozycje, na ktorych srodek motywu nie moze lezec, sa zerem -- i to
        # nosnik SYMETRYCZNY wzgledem `flip`, w odroznieniu od maski 5'.
        return occ.masked_fill(poz.view(1, 1, -1) < self._c.view(1, -1, 1), 0.0)


class OccupancyRefiner(nn.Module):
    """Give the occupancy track contrast, without discretising it.

    Measured on 128 validation sequences, the raw track is almost entirely
    background: the median position scores 0.46, real sites (>= 0.85) occupy
    **0.43%** of positions, and — the part that actually hurts — every motif
    carries its *own* baseline, anywhere in ``[0.263, 0.594]``.  The MATCH
    normalisation in :class:`ContinuousPWMScanner` equalises each profile's
    *range* but not its *offset*, so a convolution must first learn 357
    different baselines before it can detect anything, while the signal it is
    hunting sits only ~3.4 standard deviations above the noise.

    Two steps fix that.  Per-motif normalisation moves every baseline to zero,
    so one shared detector works across all motifs.  A learnable per-motif soft
    threshold then sharpens what remains: ``softplus(gain * (z - cut))``
    suppresses background toward zero while leaving strong sites large.

    Crucially this is *not* a return to thresholding.  Softplus is smooth and
    unbounded, so a 0.86 match and a 0.99 match remain different numbers and a
    weak site is attenuated rather than deleted — the affinity gradation that
    motivated the continuous track survives intact.  ``cut`` starts at 1.5σ,
    which is below the ~3.4σ where real sites sit.
    """

    def __init__(self, num_motifs, soft_threshold=True, init_cut=1.5, norm="batch"):
        super().__init__()
        self.norm = (nn.InstanceNorm1d(num_motifs, affine=True)
                     if str(norm) == "instance" else nn.BatchNorm1d(num_motifs))
        self.soft_threshold = bool(soft_threshold)
        if self.soft_threshold:
            self.gain = nn.Parameter(torch.ones(1, num_motifs, 1))
            self.cut = nn.Parameter(torch.full((1, num_motifs, 1), float(init_cut)))

    def forward(self, occ):
        z = self.norm(occ)
        if not self.soft_threshold:
            return z
        return F.softplus(self.gain * (z - self.cut))


class DepthwiseMotifStage(nn.Module):
    """Shape each motif's positional profile before any motifs are mixed.

    The stem convolution mixes all input channels at once, so motif identity is
    gone after the very first layer.  A depthwise pass first lets each motif be
    filtered on its own — clustered sites, an isolated strong site, a run of
    weak ones — at the cost of one small kernel per motif instead of a dense
    ``C_in x C_out`` block.  Residual, so it can learn to do nothing.
    """

    def __init__(self, channels, ks=9, dropout=0.0):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, ks, padding=ks // 2, groups=channels)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.act(self.norm(self.dw(x))))


class MotifSummary(nn.Module):
    """Per-motif descriptors pooled over position, for the conditioning branch.

    The positional trunk sees where things are; this branch answers "what does
    this sequence contain, and how strongly", which is what the FiLM parameters
    should depend on.  Deliberately cheap: five scalars per motif.
    """

    N_FEATURES = 5

    def __init__(self, num_motifs, dim=128, heads=4, attn_layers=2, dropout=0.1):
        super().__init__()
        self.num_motifs = num_motifs
        self.identity = nn.Parameter(torch.randn(num_motifs, dim) * 0.02)
        self.encode = nn.Linear(self.N_FEATURES, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=2 * dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, num_layers=int(attn_layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, occ):
        b, v, length = occ.shape
        peak = occ.max(dim=-1)                                  # strongest site
        mean = occ.mean(dim=-1)                                 # total occupancy
        # Soft count of strong sites: differentiable stand-in for "how many
        # hits", with no hard cut anywhere.
        strong = torch.sigmoid((occ - 0.85) * 40.0).sum(dim=-1) / length
        where = peak.indices.to(occ.dtype) / max(length - 1, 1)
        spread = occ.std(dim=-1)

        feats = torch.stack([peak.values, mean, strong, where, spread], dim=-1)
        h = self.encode(feats) + self.identity.unsqueeze(0)
        return self.norm(self.attn(h))


@register_model("MotifOccupancyNet")
class MotifOccupancyNet(nn.Module):
    """One model, one prediction: continuous motif occupancy into a LegNet trunk.

    ``x -> PWM scan -> [B, V, L] continuous occupancy -> conv trunk -> heads``

    The trunk is a LegNetV2 whose input channels are motifs rather than bases,
    so the first layer mixes transcription factors instead of nucleotides.  A
    small set of free filters runs alongside the fixed profiles, because JASPAR
    covers roughly half the fly's transcription factors and anything driven by
    an unmotified factor would otherwise be invisible.

    A motif-summary branch attends over the ``V`` motifs and FiLM-modulates the
    trunk, so "which factors are present" conditions "where things are".  Both
    tasks then read one shared representation through a task embedding.
    """

    def __init__(self, vocab=None, seq_len=249, stem_ch=128, stem_ks=7, ef_ks=7,
                 ef_block_sizes=(128, 96, 64, 64), pool_sizes=(2, 2, 2, 1),
                 resize_factor=3, n_learned=32, trainable_pwm=False,
                 occupancy_temperature=0.0, summary_dim=128, heads=4,
                 attn_layers=2, film=True, hidden_dim=192, dropout=0.2,
                 drop_path=0.1, refine=True, soft_threshold=True, init_cut=1.5,
                 depthwise=True, depthwise_ks=9, **kwargs):
        super().__init__()
        from .legnet import LegNetV2
        from .motif_film_v3 import StochasticDepth

        self.set_vocabulary(MotifVocabulary.from_config(vocab), build=False)
        self.scanner = ContinuousPWMScanner(
            self._vocab, trainable=trainable_pwm, temperature=occupancy_temperature)
        v = len(self._vocab)

        # Free filters for what JASPAR does not cover.  Same width as the
        # profiles so both views are positionally aligned.
        self.n_learned = int(n_learned)
        self.learned = nn.Conv1d(4, self.n_learned, self.scanner.max_len,
                                 padding=self.scanner.max_len - 1) \
            if self.n_learned else None

        # Contrast first: the raw track is 99.6% background with a per-motif
        # baseline, so without this the trunk spends its first layer learning
        # 357 offsets instead of learning biology.
        self.refiner = OccupancyRefiner(v, soft_threshold, init_cut) if refine else None

        in_ch = v + self.n_learned
        self.depthwise = DepthwiseMotifStage(in_ch, depthwise_ks, dropout * 0.5) \
            if depthwise else None
        self.trunk = LegNetV2(
            in_ch=in_ch, stem_ch=stem_ch, stem_ks=stem_ks, ef_ks=ef_ks,
            ef_block_sizes=tuple(ef_block_sizes), pool_sizes=tuple(pool_sizes),
            resize_factor=resize_factor, seq_len=seq_len)

        with torch.no_grad():
            probe = torch.zeros(2, in_ch, seq_len)
            h = self.trunk.stem(probe)
            widths = []
            for block in self.trunk.main:
                h = block(h)
                widths.append(h.shape[1])
            trunk_dim = int(self.trunk.mapper(h).mean(dim=-1).shape[-1])

        self.summary = MotifSummary(v, summary_dim, heads, attn_layers, dropout) \
            if film else None
        if film:
            self.film = nn.ModuleDict({
                str(i): nn.Linear(summary_dim, 2 * w) for i, w in enumerate(widths)})
            for layer in self.film.values():
                nn.init.zeros_(layer.weight)     # identity at initialisation
                nn.init.zeros_(layer.bias)
        else:
            self.film = nn.ModuleDict()

        if drop_path > 0:
            n = len(self.trunk.main)
            self.trunk.main = nn.ModuleList([
                StochasticDepth(b, drop_path * (i + 1) / n)
                for i, b in enumerate(self.trunk.main)])

        extra = summary_dim if film else 0
        self.mapper = nn.Sequential(
            nn.Linear(trunk_dim + extra, hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.head = FiLMTaskHead(hidden_dim, n_tasks=2, dropout=dropout)

        self.register_buffer("_zero", torch.zeros(()), persistent=False)
        self.reg_loss = self._zero

    @property
    def vocab(self):
        return self._vocab

    def set_vocabulary(self, vocabulary, build=True):
        self._vocab = vocabulary
        if build:
            device = next(self.parameters()).device
            self.scanner = ContinuousPWMScanner(
                vocabulary, trainable=isinstance(self.scanner.kernels, nn.Parameter),
                temperature=self.scanner.temperature).to(device)
        return self

    def memory_estimate(self, batch_size: int) -> float:
        v = len(self._vocab)
        return batch_size * 2 * v * 249 * 4 / 1e6

    def occupancy(self, x):
        """The continuous ``[B, V, L]`` track — the model's interpretable input."""
        return self.scanner(x)

    def embed(self, x):
        occ = self.scanner(x)                                   # [B, V, L]

        # The summary branch reads the RAW track: its descriptors are defined
        # in relative-score units (peak height, sites above 0.85), which the
        # refiner's per-motif rescaling would make meaningless.
        g = self.summary(occ).mean(dim=1) if self.summary is not None else None

        h_in = self.refiner(occ) if self.refiner is not None else occ
        if self.learned is not None:
            extra = self.learned(x)[..., :occ.shape[-1]]
            h_in = torch.cat([h_in, torch.sigmoid(extra)], dim=1)
        if self.depthwise is not None:
            h_in = self.depthwise(h_in)

        h = self.trunk.stem(h_in)
        for i, block in enumerate(self.trunk.main):
            h = block(h)
            key = str(i)
            if g is not None and key in self.film:
                gamma, beta = self.film[key](g).chunk(2, dim=-1)
                h = h * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        h = self.trunk.mapper(h)
        feat = F.adaptive_avg_pool1d(h, 1).squeeze(-1)

        if g is not None:
            feat = torch.cat([feat, g], dim=-1)
        return self.mapper(feat)

    def forward(self, x):
        dev, hk = self.head(self.embed(x))
        return dev, hk

    def get_features(self, x):
        return self.embed(x)
