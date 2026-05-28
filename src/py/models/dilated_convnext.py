import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model


class GRN1d(nn.Module):
    """Global Response Normalization for 1D sequences (channels-last)."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        # x: (B, L, C)
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 block with optional dilation, adapted from ASAP."""

    def __init__(self, channels_in, channels_out, kernel_size,
                 inv_bottleneck_scale=4, dilation_rate=1):
        super().__init__()
        self.res_early = (channels_in == channels_out)
        inner_dim = int(inv_bottleneck_scale * channels_out)

        self.dwconv = nn.Conv1d(
            channels_in, channels_out,
            kernel_size=kernel_size, padding='same',
            dilation=dilation_rate
        )
        self.norm = nn.LayerNorm(channels_out, eps=1e-6)
        self.pwconv1 = nn.Linear(channels_out, inner_dim)
        self.act = nn.GELU()
        self.grn = GRN1d(inner_dim)
        self.pwconv2 = nn.Linear(inner_dim, channels_out)

    def forward(self, x):
        # x: (B, C, L)
        if self.res_early:
            residual = x
            x = self.dwconv(x)
        else:
            x = self.dwconv(x)
            residual = x

        x = x.permute(0, 2, 1)  # (B, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)  # (B, C, L)

        return residual + x


class DilatedConvBlock(nn.Module):
    """Conv -> BatchNorm -> GELU with dilation (from ASAP/Basenji)."""

    def __init__(self, channels_in, channels_out, kernel_size=3,
                 dilation_rate=1, bn_gamma=None):
        super().__init__()
        self.activation = nn.GELU()
        self.conv = nn.Conv1d(
            channels_in, channels_out, kernel_size,
            bias=False, dilation=dilation_rate, padding='same'
        )
        self.bn = nn.BatchNorm1d(channels_out, momentum=0.1)
        if bn_gamma == 'zeros':
            self.bn.weight = nn.Parameter(torch.zeros_like(self.bn.weight))

    def forward(self, x):
        x = self.activation(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class BasenjiCore(nn.Module):
    """Stacked dilated residual blocks with exponentially increasing dilation.

    Adapted from ASAP's BasenjiCoreBlock for short regulatory sequences
    with 3bp resolution output.
    """

    def __init__(self, filters_in, nr_res_blocks=7, rate_mult=1.5,
                 filters1=128, kernel1=3, kernel2=1, dropout=0.3):
        super().__init__()
        self.nr_res_blocks = nr_res_blocks
        self.dropout = nn.Dropout(p=dropout)

        dconv_blocks = []
        conv_blocks = []
        dilation_rate = 1.0

        for _ in range(nr_res_blocks):
            dconv_blocks.append(DilatedConvBlock(
                filters_in, filters1, kernel_size=kernel1,
                dilation_rate=int(np.round(dilation_rate))
            ))
            conv_blocks.append(DilatedConvBlock(
                filters1, filters_in, kernel_size=kernel2,
                bn_gamma='zeros'
            ))
            dilation_rate *= rate_mult

        self.dconv_blocks = nn.ModuleList(dconv_blocks)
        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(self, x):
        for i in range(self.nr_res_blocks):
            residual = x
            x = self.dconv_blocks[i](x)
            x = self.conv_blocks[i](x)
            x = self.dropout(x)
            x = x + residual
        return x


@register_model("DilatedConvNeXt")
class DilatedConvNeXt(nn.Module):
    """Dilated ConvNeXt CNN adapted from ASAP for short regulatory sequences.

    Key adaptations from ASAP's ConvNeXtDCNN:
    - Input: one-hot DNA (batch, 4, seq_len) with short enhancer sequences (~249bp)
    - Resolution: 3bp (instead of 4bp in original ASAP)
    - Output: two scalar predictions (Dev, Hk) instead of per-bin track
    - Uses Global Average Pooling instead of per-position output

    Parameters
    ----------
    seq_len : int
        Input sequence length (default 249).
    filters0 : int
        Channels after initial ConvNeXt stem (default 256).
    filters1 : int
        Inner channels in dilated residual blocks (default 128).
    nr_res_blocks : int
        Number of stacked dilated residual blocks (default 7).
    rate_mult : float
        Dilation rate multiplier per block (default 1.5).
    kernel0 : int
        Initial stem kernel size (default 15).
    kernel1 : int
        Dilated conv kernel size (default 3).
    dropout : float
        Dropout in residual blocks (default 0.3).
    final_dropout : float
        Dropout before output heads (default 0.1).
    dense_dim : int
        Dense layer width before output (default 256).
    """

    def __init__(self, seq_len=249, filters0=256, filters1=128,
                 nr_res_blocks=7, rate_mult=1.5, kernel0=15, kernel1=3,
                 dropout=0.3, final_dropout=0.1, dense_dim=256, **kwargs):
        super().__init__()

        # Stem: ConvNeXt block for initial feature extraction
        self.stem = ConvNeXtV2Block(
            channels_in=4, channels_out=filters0, kernel_size=kernel0
        )
        # Pool to reduce by factor of 3 (3bp resolution)
        self.stem_pool = nn.AvgPool1d(kernel_size=3, stride=3)

        # Core: dilated residual blocks
        self.core = BasenjiCore(
            filters_in=filters0,
            nr_res_blocks=nr_res_blocks,
            rate_mult=rate_mult,
            filters1=filters1,
            kernel1=kernel1,
            dropout=dropout
        )

        # Final conv to expand representation
        self.final_conv = DilatedConvBlock(filters0, filters0 * 2)
        self.final_dropout = nn.Dropout(p=final_dropout)

        # Global aggregation + prediction heads
        self.head = nn.Sequential(
            nn.Linear(filters0 * 2, dense_dim),
            nn.GELU(),
            nn.Dropout(final_dropout),
        )
        self.fc_dev = nn.Linear(dense_dim, 1)
        self.fc_hk = nn.Linear(dense_dim, 1)

    def forward(self, x):
        # x: (batch, 4, seq_len)
        x = self.stem(x)           # (batch, filters0, seq_len)
        x = self.stem_pool(x)      # (batch, filters0, seq_len//3)
        x = self.core(x)           # (batch, filters0, seq_len//3)
        x = self.final_conv(x)     # (batch, filters0*2, seq_len//3)
        x = self.final_dropout(x)

        # Global Average Pooling
        x = x.mean(dim=2)          # (batch, filters0*2)

        x = self.head(x)           # (batch, dense_dim)
        out_dev = self.fc_dev(x)   # (batch, 1)
        out_hk = self.fc_hk(x)    # (batch, 1)

        return out_dev, out_hk

    def get_features(self, x):
        """Extract penultimate feature vector."""
        x = self.stem(x)
        x = self.stem_pool(x)
        x = self.core(x)
        x = self.final_conv(x)
        x = self.final_dropout(x)
        x = x.mean(dim=2)
        x = self.head(x)
        return x
