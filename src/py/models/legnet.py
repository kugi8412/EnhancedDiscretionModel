# -*- coding: utf-8 -*-
"""
LegNet family of models for DNA enhancer activity prediction.

Implements the following architectures:

- **LegNetOriginal**: Faithful reproduction of the original LegNet (SeqNN) from
  `Penzar, Nogina et al., Bioinformatics 2023 <https://doi.org/10.1093/bioinformatics/btad457>`_.
  Uses Bilinear tensor-train SE layers, inverted residual blocks, and a
  soft-classification head (18-bin discretisation).  Requires ``tltorch``.

- **LegNet**: Regression-adapted variant that replaces the 18-bin
  soft-classification head with a dual regression head (Dev / Hk).

- **LegNetV2**: Modernised variant with simplified SE, EfficientNet-style
  blocks, residual concatenation, and optional pooling.

References
----------
.. [1] https://github.com/autosome-ru/LegNet
"""

import math
import torch
from torch import nn
from tltorch import TRL
from collections import OrderedDict
import torch.nn.functional as F
from .registry import register_model


# ---------------------------------------------------------------------------
# Shared building blocks (used by LegNetOriginal & LegNet)
# ---------------------------------------------------------------------------

class Bilinear(nn.Module):
    """Low-rank bilinear layer via tensor-train decomposition.

    Introduces pairwise products to model combinatorial effects between
    input features while keeping the parameter count manageable.

    Parameters
    ----------
    n : int
        Number of input features.
    out : int or None
        Number of output features.  Defaults to *n*.
    rank : float
        Fraction of the maximal rank used in the TT decomposition.
    bias : bool
        Whether to include a bias term.
    """

    def __init__(self, n: int, out=None, rank=0.05, bias=False):
        super().__init__()
        if out is None:
            out = (n,)
        self.trl = TRL((n, n), out, bias=bias, rank=rank)
        self.trl.weight = self.trl.weight.normal_(std=0.00075)

    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        return self.trl(x @ x.transpose(-1, -2))


class Concater(nn.Module):
    """Concatenate a module's output with its input along a given dimension.

    Parameters
    ----------
    module : nn.Module
        The transformation whose output is concatenated with the input.
    dim : int
        Concatenation dimension (default ``-1``).
    """

    def __init__(self, module: nn.Module, dim=-1):
        super().__init__()
        self.mod = module
        self.dim = dim

    def forward(self, x):
        return torch.concat((x, self.mod(x)), dim=self.dim)


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer with bilinear gating.

    Combines channel-wise global pooling, a bilinear bottleneck, and
    a sigmoid gate to re-weight feature channels.

    Parameters
    ----------
    inp : int
        Channel count used for determining the bottleneck width.
    oup : int
        Number of input / output channels.
    reduction : int
        Reduction ratio for the bottleneck.
    """

    def __init__(self, inp, oup, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp // reduction)),
            nn.SiLU(),
            nn.Linear(int(inp // reduction), int(inp // reduction)),
            Concater(
                Bilinear(int(inp // reduction),
                         int(inp // reduction // 2),
                         rank=0.5, bias=True)),
            nn.SiLU(),
            nn.Linear(int(inp // reduction) + int(inp // reduction // 2), oup),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


# ---------------------------------------------------------------------------
# LegNetOriginal — faithful reproduction of autosome-ru/LegNet (SeqNN)
# ---------------------------------------------------------------------------

@register_model("LegNetOriginal")
class LegNetOriginal(nn.Module):
    """Original LegNet (SeqNN) architecture from Penzar & Nogina et al.

    This is a direct port of the publicly released model at
    ``https://github.com/autosome-ru/LegNet``.  It uses a
    soft-classification head that discretises expression into 18 bins
    (``final_ch=18``) and returns ``(log_probs, score)`` where *score*
    is the expected bin index.

    Parameters
    ----------
    seqsize : int
        Input sequence length.
    use_single_channel : bool
        If ``True`` the stem expects 6 input channels; otherwise 5.
    block_sizes : list[int]
        Channel widths for each inverted-residual stage.
    ks : int
        Kernel size for all convolutions.
    resize_factor : int
        Channel expansion ratio inside inverted-residual blocks.
    activation : nn.Module class
        Activation constructor (default :class:`nn.SiLU`).
    filter_per_group : int
        Channels per group in the depthwise convolution.
    se_reduction : int
        Reduction factor for the SE bottleneck.
    final_ch : int
        Number of discrete expression bins (default 18).
    bn_momentum : float
        Momentum for all BatchNorm layers.
    """

    __constants__ = ('resize_factor',)

    def __init__(
        self,
        seqsize=249,
        use_single_channel=False,
        block_sizes=(256, 256, 128, 128, 64, 64, 32, 32),
        ks=5,
        resize_factor=4,
        activation=nn.SiLU,
        filter_per_group=2,
        se_reduction=4,
        final_ch=18,
        bn_momentum=0.1,
        **kwargs,
    ):
        super().__init__()
        self.block_sizes = list(block_sizes)
        self.resize_factor = resize_factor
        self.se_reduction = se_reduction
        self.seqsize = seqsize
        self.use_single_channel = use_single_channel
        self.final_ch = final_ch
        self.bn_momentum = bn_momentum

        in_ch = 6 if self.use_single_channel else 5
        seqextblocks = OrderedDict()

        # Stem block
        seqextblocks['blc0'] = nn.Sequential(
            nn.Conv1d(in_ch, block_sizes[0], kernel_size=ks,
                      padding='same', bias=False),
            nn.BatchNorm1d(block_sizes[0], momentum=bn_momentum),
            activation(),
        )

        # Inverted-residual + resize stages
        for ind, (prev_sz, sz) in enumerate(
            zip(block_sizes[:-1], block_sizes[1:])
        ):
            seqextblocks[f'inv_res_blc{ind}'] = nn.Sequential(
                nn.Conv1d(prev_sz, sz * resize_factor, 1,
                          padding='same', bias=False),
                nn.BatchNorm1d(sz * resize_factor, momentum=bn_momentum),
                activation(),
                nn.Conv1d(sz * resize_factor, sz * resize_factor, ks,
                          groups=sz * resize_factor // filter_per_group,
                          padding='same', bias=False),
                nn.BatchNorm1d(sz * resize_factor, momentum=bn_momentum),
                activation(),
                SELayer(prev_sz, sz * resize_factor, reduction=se_reduction),
                nn.Conv1d(sz * resize_factor, prev_sz, 1,
                          padding='same', bias=False),
                nn.BatchNorm1d(prev_sz, momentum=bn_momentum),
                activation(),
            )
            seqextblocks[f'resize_blc{ind}'] = nn.Sequential(
                nn.Conv1d(2 * prev_sz, sz, ks, padding='same', bias=False),
                nn.BatchNorm1d(sz, momentum=bn_momentum),
                activation(),
            )

        self.seqextractor = nn.ModuleDict(seqextblocks)

        self.mapper = nn.Sequential(
            nn.Conv1d(block_sizes[-1], final_ch, 1, padding='same'),
            activation(),
        )

        self.register_buffer(
            'bins',
            torch.arange(0, final_ch, dtype=torch.float32),
        )

    def feature_extractor(self, x):
        """Run the convolutional backbone and return the feature tensor."""
        x = self.seqextractor['blc0'](x)
        for i in range(len(self.block_sizes) - 1):
            x = torch.cat(
                [x, self.seqextractor[f'inv_res_blc{i}'](x)], dim=1)
            x = self.seqextractor[f'resize_blc{i}'](x)
        return x

    def forward(self, x, predict_score=True):
        """Forward pass.

        Returns
        -------
        logprobs : Tensor
            Log-softmax over the *final_ch* bins.
        score : Tensor, optional
            Expected bin index (only when ``predict_score=True``).
        """
        f = self.feature_extractor(x)
        x = self.mapper(f)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        logprobs = F.log_softmax(x, dim=1)
        if predict_score:
            score = (F.softmax(x, dim=1) * self.bins).sum(dim=1)
            return logprobs, score
        return logprobs

    def get_features(self, x):
        """Extract penultimate feature vector (after mapper + GAP)."""
        f = self.feature_extractor(x)
        x = self.mapper(f)
        return F.adaptive_avg_pool1d(x, 1).squeeze(2)


# ---------------------------------------------------------------------------
# LegNet — regression-adapted variant
# ---------------------------------------------------------------------------

@register_model("LegNet")
class LegNet(nn.Module):
    """LegNet with dual regression heads for Dev / Hk prediction.

    Replaces the original 18-bin soft-classification head with global
    average pooling followed by a linear layer that outputs two scalar
    predictions (Developmental and Housekeeping expression).

    Parameters
    ----------
    seq_len : int
        Expected input sequence length (may include padding for augmentation).
    in_channels : int
        Number of input channels (4 for one-hot DNA).
    block_sizes : list[int]
        Channel widths per inverted-residual stage.
    ks : int
        Kernel size.
    resize_factor : int
        Channel expansion ratio.
    activation : nn.Module class
        Activation constructor.
    filter_per_group : int
        Channels per group in depthwise convolution.
    se_reduction : int
        SE bottleneck reduction.
    final_ch : int
        Output channels of the mapper convolution.
    bn_momentum : float
        BatchNorm momentum.
    """

    __constants__ = ('resize_factor',)

    def __init__(
        self,
        seq_len=249,
        in_channels=4,
        block_sizes=(256, 256, 128, 128, 64, 64, 32, 32),
        ks=5,
        resize_factor=4,
        activation=nn.SiLU,
        filter_per_group=2,
        se_reduction=4,
        final_ch=18,
        bn_momentum=0.1,
        **kwargs,
    ):
        super().__init__()
        self.block_sizes = list(block_sizes)
        self.resize_factor = resize_factor
        self.se_reduction = se_reduction
        self.seqsize = seq_len
        self.final_ch = final_ch
        self.bn_momentum = bn_momentum

        seqextblocks = OrderedDict()

        seqextblocks['blc0'] = nn.Sequential(
            nn.Conv1d(in_channels, block_sizes[0], kernel_size=ks,
                      padding='same', bias=False),
            nn.BatchNorm1d(block_sizes[0], momentum=bn_momentum),
            activation(),
        )

        for ind, (prev_sz, sz) in enumerate(
            zip(block_sizes[:-1], block_sizes[1:])
        ):
            seqextblocks[f'inv_res_blc{ind}'] = nn.Sequential(
                nn.Conv1d(prev_sz, sz * resize_factor, 1,
                          padding='same', bias=False),
                nn.BatchNorm1d(sz * resize_factor, momentum=bn_momentum),
                activation(),
                nn.Conv1d(sz * resize_factor, sz * resize_factor, ks,
                          groups=sz * resize_factor // filter_per_group,
                          padding='same', bias=False),
                nn.BatchNorm1d(sz * resize_factor, momentum=bn_momentum),
                activation(),
                SELayer(prev_sz, sz * resize_factor, reduction=se_reduction),
                nn.Conv1d(sz * resize_factor, prev_sz, 1,
                          padding='same', bias=False),
                nn.BatchNorm1d(prev_sz, momentum=bn_momentum),
                activation(),
            )
            seqextblocks[f'resize_blc{ind}'] = nn.Sequential(
                nn.Conv1d(2 * prev_sz, sz, ks, padding='same', bias=False),
                nn.BatchNorm1d(sz, momentum=bn_momentum),
                activation(),
            )

        self.seqextractor = nn.ModuleDict(seqextblocks)

        self.mapper = nn.Sequential(
            nn.Conv1d(block_sizes[-1], final_ch, 1, padding='same'),
            activation(),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(final_ch, 2)

    def feature_extractor(self, x):
        """Run the convolutional backbone."""
        x = self.seqextractor['blc0'](x)
        for i in range(len(self.block_sizes) - 1):
            x = torch.cat(
                [x, self.seqextractor[f'inv_res_blc{i}'](x)], dim=1)
            x = self.seqextractor[f'resize_blc{i}'](x)
        return x

    def forward(self, x):
        """Return ``(pred_dev, pred_hk)`` each of shape ``[B, 1]``."""
        f = self.feature_extractor(x)
        x = self.mapper(f)
        x = self.global_pool(x).squeeze(2)
        out = self.fc_out(x)
        return out[:, 0:1], out[:, 1:2]

    def get_features(self, x):
        """Extract penultimate feature vector for cross-model analysis."""
        f = self.feature_extractor(x)
        x = self.mapper(f)
        return self.global_pool(x).squeeze(2)


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def initialize_weights(m):
    """Kaiming / constant initialisation for Conv1d, BatchNorm1d, Linear."""
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# ---------------------------------------------------------------------------
# Building blocks for LegNetV2
# ---------------------------------------------------------------------------

class SENewLayer(nn.Module):
    """Simplified Squeeze-and-Excitation (without bilinear gating).

    Parameters
    ----------
    inp : int
        Number of input / output channels.
    reduction : int
        Bottleneck reduction ratio.
    """

    def __init__(self, inp, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(inp, int(inp // reduction)),
            nn.SiLU(),
            nn.Linear(int(inp // reduction), inp),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class EffBlock(nn.Module):
    """EfficientNet-style inverted residual block with depthwise SE.

    Parameters
    ----------
    in_ch : int
        Input channels.
    ks : int
        Kernel size for the depthwise convolution.
    resize_factor : int
        Channel expansion ratio.
    activation : nn.Module class
        Activation constructor.
    out_ch : int or None
        Output channels (defaults to *in_ch*).
    se_reduction : int or None
        SE reduction (defaults to *resize_factor*).
    """

    def __init__(self, in_ch, ks, resize_factor, activation,
                 out_ch=None, se_reduction=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = in_ch if out_ch is None else out_ch
        se_red = resize_factor if se_reduction is None else se_reduction
        inner_dim = in_ch * resize_factor

        self.block = nn.Sequential(
            nn.Conv1d(in_ch, inner_dim, 1, padding='same', bias=False),
            nn.BatchNorm1d(inner_dim),
            activation(),
            nn.Conv1d(inner_dim, inner_dim, ks, groups=inner_dim,
                      padding='same', bias=False),
            nn.BatchNorm1d(inner_dim),
            activation(),
            SENewLayer(inner_dim, reduction=se_red),
            nn.Conv1d(inner_dim, in_ch, 1, padding='same', bias=False),
            nn.BatchNorm1d(in_ch),
            activation(),
        )

    def forward(self, x):
        return self.block(x)


class LocalBlock(nn.Module):
    """Single convolution + BN + activation block.

    Parameters
    ----------
    in_ch : int
        Input channels.
    ks : int
        Kernel size.
    activation : nn.Module class
        Activation constructor.
    out_ch : int or None
        Output channels (defaults to *in_ch*).
    """

    def __init__(self, in_ch, ks, activation, out_ch=None):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, ks, padding='same', bias=False),
            nn.BatchNorm1d(out_ch),
            activation(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConcat(nn.Module):
    """Apply a function and concatenate the result with the input."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return torch.concat([self.fn(x, **kwargs), x], dim=1)


class MapperBlock(nn.Module):
    """BatchNorm + 1x1 convolution channel projection."""

    def __init__(self, in_features, out_features, activation=nn.SiLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Conv1d(in_features, out_features, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# LegNetV2 — modernised regression model
# ---------------------------------------------------------------------------

@register_model("LegNetV2")
class LegNetV2(nn.Module):
    """Modernised LegNet variant with simplified SE and residual concatenation.

    Uses :class:`EffBlock` inverted residuals, :class:`ResidualConcat`
    skip-connections, optional per-stage max-pooling, and a dual regression
    head.

    Parameters
    ----------
    in_ch : int
        Input channels (4 for one-hot DNA).
    stem_ch : int
        Number of channels produced by the stem convolution.
    stem_ks : int
        Kernel size of the stem convolution.
    ef_ks : int
        Kernel size in all EffBlocks and LocalBlocks.
    ef_block_sizes : list[int]
        Output channel count for each backbone stage.
    pool_sizes : list[int]
        Max-pool kernel per stage (use 1 for no pooling).
    resize_factor : int
        Channel expansion ratio.
    seq_len : int
        Expected input length (informational, not enforced).
    """

    def __init__(
        self,
        in_ch=4,
        stem_ch=256,
        stem_ks=5,
        ef_ks=5,
        ef_block_sizes=(256, 128, 128, 64, 64, 32, 32),
        pool_sizes=(1, 2, 1, 2, 1, 2, 1),
        resize_factor=4,
        seq_len=249,
        **kwargs,
    ):
        super().__init__()
        assert len(pool_sizes) == len(ef_block_sizes)

        activation = nn.SiLU
        self.stem = LocalBlock(in_ch=in_ch, out_ch=stem_ch,
                               ks=stem_ks, activation=activation)

        blocks = []
        cur_ch = stem_ch
        for pool_sz, out_ch in zip(pool_sizes, ef_block_sizes):
            blc = nn.Sequential(
                ResidualConcat(
                    EffBlock(cur_ch, ef_ks, resize_factor, activation)),
                LocalBlock(cur_ch * 2, ef_ks, activation, out_ch=out_ch),
                nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity(),
            )
            cur_ch = out_ch
            blocks.append(blc)

        self.main = nn.Sequential(*blocks)

        self.mapper = MapperBlock(cur_ch, cur_ch * 2, activation)

        self.head = nn.Sequential(
            nn.Linear(cur_ch * 2, cur_ch * 2),
            nn.BatchNorm1d(cur_ch * 2),
            activation(),
            nn.Linear(cur_ch * 2, 2),
        )

        self.apply(initialize_weights)

    def forward(self, x):
        """Return ``(pred_dev, pred_hk)`` each of shape ``[B, 1]``."""
        x = self.stem(x)
        x = self.main(x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.head(x)
        return x[:, 0:1], x[:, 1:2]

    def get_features(self, x):
        """Extract penultimate feature vector for cross-model analysis."""
        x = self.stem(x)
        x = self.main(x)
        x = self.mapper(x)
        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)

