"""
LegNetPlus: Enhanced LegNet architecture for DNA enhancer prediction.

Improvements over LegNetV2:
1. Multi-scale stem (kernels 3, 7, 15) captures short motifs + longer syntax
2. Stochastic depth (DropPath) for deeper training without overfitting
3. Gated Linear Unit (GLU) in expansion — better gradient flow than SiLU
4. Relative position bias via depthwise conv in SE path
5. Dual-head with shared backbone — explicit Dev/Hk specialization

No extra dependencies — pure PyTorch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model


def _init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DropPath(nn.Module):
    """Stochastic depth — drops entire residual branch during training."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep_prob
        return x * mask / keep_prob


class MultiScaleStem(nn.Module):
    """Multi-scale convolution stem with kernels 3, 7, 15."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        branch_ch = out_ch // 3
        remainder = out_ch - branch_ch * 3

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(branch_ch), nn.SiLU())
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(branch_ch), nn.SiLU())
        self.branch15 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch + remainder, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(branch_ch + remainder), nn.SiLU())

        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.SiLU())

    def forward(self, x):
        return self.fuse(torch.cat([self.branch3(x), self.branch7(x), self.branch15(x)], dim=1))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation with depthwise position encoding."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pos_conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2,
                                  groups=channels, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid), nn.SiLU(),
            nn.Linear(mid, channels), nn.Sigmoid())

    def forward(self, x):
        # Position-aware pooling
        x_pos = self.pos_conv(x)
        y = (x + x_pos).mean(dim=2)  # [B, C]
        y = self.fc(y).unsqueeze(2)
        return x * y


class GLUBlock(nn.Module):
    """Efficient block with Gated Linear Unit and stochastic depth."""
    def __init__(self, in_ch, ks=5, expansion=4, drop_path=0.0):
        super().__init__()
        inner = in_ch * expansion
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, inner * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(inner * 2),
        )
        self.dw_conv = nn.Conv1d(inner, inner, kernel_size=ks, padding=ks // 2,
                                 groups=inner, bias=False)
        self.dw_norm = nn.BatchNorm1d(inner)
        self.se = SEBlock(inner, reduction=expansion)
        self.proj = nn.Sequential(
            nn.Conv1d(inner, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_ch))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        h = self.block(x)
        # GLU: split and gate
        h1, gate = h.chunk(2, dim=1)
        h = h1 * torch.sigmoid(gate)
        h = self.dw_conv(h)
        h = F.silu(self.dw_norm(h))
        h = self.se(h)
        h = self.proj(h)
        return x + self.drop_path(h)


class ResidualConcatBlock(nn.Module):
    """ResidualConcat + LocalBlock + optional pooling (matching LegNetV2 pattern)."""
    def __init__(self, in_ch, out_ch, ks=5, expansion=4, pool_sz=1, drop_path=0.0):
        super().__init__()
        self.glu = GLUBlock(in_ch, ks=ks, expansion=expansion, drop_path=drop_path)
        self.local = nn.Sequential(
            nn.Conv1d(in_ch * 2, out_ch, kernel_size=ks, padding=ks // 2, bias=False),
            nn.BatchNorm1d(out_ch), nn.SiLU())
        self.pool = nn.MaxPool1d(pool_sz) if pool_sz > 1 else nn.Identity()

    def forward(self, x):
        h = self.glu(x)
        x = torch.cat([h, x], dim=1)
        x = self.local(x)
        x = self.pool(x)
        return x


@register_model("LegNetPlus")
class LegNetPlus(nn.Module):
    """
    LegNetPlus: Enhanced DNA enhancer prediction model.

    Combines multi-scale stem, GLU expansion, SE with position encoding,
    stochastic depth, and dual prediction heads.
    """
    def __init__(self, in_ch=4, stem_ch=256, ef_ks=5,
                 block_sizes=[256, 128, 128, 64, 64, 32, 32],
                 pool_sizes=[1, 2, 1, 2, 1, 2, 1],
                 expansion=4, drop_path_rate=0.1,
                 seq_len=249, **kwargs):
        super().__init__()
        assert len(pool_sizes) == len(block_sizes)

        self.stem = MultiScaleStem(in_ch, stem_ch)

        # Linearly increasing drop path rate
        n_blocks = len(block_sizes)
        dp_rates = [drop_path_rate * i / max(n_blocks - 1, 1) for i in range(n_blocks)]

        blocks = []
        cur_ch = stem_ch
        for i, (out_ch, pool_sz) in enumerate(zip(block_sizes, pool_sizes)):
            blocks.append(ResidualConcatBlock(
                cur_ch, out_ch, ks=ef_ks, expansion=expansion,
                pool_sz=pool_sz, drop_path=dp_rates[i]))
            cur_ch = out_ch
        self.backbone = nn.Sequential(*blocks)

        # Mapper
        self.mapper = nn.Sequential(
            nn.BatchNorm1d(cur_ch),
            nn.Conv1d(cur_ch, cur_ch * 2, kernel_size=1))

        feat_dim = cur_ch * 2

        # Dual heads with shared features
        self.head_dev = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.BatchNorm1d(feat_dim),
            nn.SiLU(), nn.Dropout(0.1), nn.Linear(feat_dim, 1))
        self.head_hk = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.BatchNorm1d(feat_dim),
            nn.SiLU(), nn.Dropout(0.1), nn.Linear(feat_dim, 1))

        self.apply(_init_weights)

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.head_dev(x), self.head_hk(x)

    def get_features(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.mapper(x)
        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)
