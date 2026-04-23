# -*- coding: utf-8 -*-
"""
Reusable building-block layers for all CNN models in this framework.

Provides two optional, config-driven training modifications:

MaskedConv1d
    Drop-in replacement for ``nn.Conv1d`` whose kernel is multiplied by a
    differentiable soft-mask before each forward pass.  Every output filter
    owns an independent learnable radius *r* that controls its effective
    receptive field width.  Positions within *r* of the kernel centre receive
    full weight; positions beyond it are smoothly suppressed by a sigmoid gate.

    This is the 1-D generalisation of ``Conv2dSuperKernel`` already used by
    ``BassetNetwork`` in ``regseqnet.py``.

    The mask for filter *f* at position *k* is::

        r_f          = softplus(raw_radius_f)          # always > 0
        distance_k   = |k - kernel_size // 2|
        mask_f_k     = sigmoid( τ · (r_f - distance_k) )

    Large τ → hard binary mask; small τ → smooth, wide transition.

LearnableAttention1d
    Purely-parametric (input-independent) attention for feature maps of shape
    ``[B, C, L]``.  Unlike SE-blocks or self-attention, no input projection is
    performed — the weights are *directly* trained scalars.

    Two orthogonal components, each independently enabled:

    **Channel attention** – a ``[C]`` parameter vector gated through
    ``sigmoid``.  Learns which feature channels are globally important.
    Initialised so the gate starts near 1 (near-identity) to avoid
    disrupting training dynamics at step 0.

    **Position attention** – a ``[1, C, L]`` parameter tensor whose
    ``softmax`` over the position axis is multiplied back by *L* to
    preserve magnitude.  Uniform initialisation → exact identity at step 0.
    When the actual spatial dim differs from the registered ``seq_len``
    (e.g. after augmentation-induced length changes), the weights are
    linearly interpolated.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MaskedConv1d
# ---------------------------------------------------------------------------

class MaskedConv1d(nn.Module):
    """1-D convolution with a per-filter learnable soft kernel mask.

    Parameters
    ----------
    in_channels, out_channels, kernel_size, stride, padding,
    dilation, groups, bias :
        Identical semantics to ``nn.Conv1d``.
    r_init : float
        Initial radius as a fraction of the kernel half-width.
        ``1.0`` = full kernel active (default).
        ``0.5`` = inner half of kernel active.
        ``0.0`` = centre position only.
        Clamped to ``[0.01, 1.0]``.
    tau : float
        Sigmoid sharpness of the mask boundary.  Higher values → harder
        cutoff.  Default ``5.0``.
    r_trainable : bool
        Whether the radius parameters receive gradient updates.
        Set to ``False`` to use a fixed structural prior.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, str] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        r_init: float = 1.0,
        tau: float = 5.0,
        r_trainable: bool = True,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.groups       = groups
        self.tau          = tau

        # Convolution weights — same layout as nn.Conv1d
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Learnable radius (reparameterised through softplus so r > 0 always)
        half = max(kernel_size // 2, 1)
        r_clamped = max(min(r_init, 1.0), 0.01) * half          # absolute radius ∈ (0, half]
        # Softplus inverse: raw = log(exp(r) - 1)  →  softplus(raw) = r
        raw_init = torch.log(
            torch.expm1(torch.tensor(r_clamped, dtype=torch.float32))
        )
        self.raw_radius = nn.Parameter(
            raw_init.expand(out_channels).clone(),
            requires_grad=r_trainable,
        )

        # Static per-position distances from kernel centre  [1, 1, K]
        positions = torch.arange(kernel_size, dtype=torch.float32)
        half_idx  = kernel_size // 2
        distances = (positions - half_idx).abs().view(1, 1, kernel_size)
        self.register_buffer("distances", distances)

    # ------------------------------------------------------------------

    def _mask(self) -> torch.Tensor:
        """Compute per-filter soft mask  shape [out_channels, 1, K]."""
        r = F.softplus(self.raw_radius).view(self.out_channels, 1, 1)
        return torch.sigmoid(self.tau * (r - self.distances))

    def get_effective_radii(self) -> torch.Tensor:
        """Return the current learning radii for all filters (no gradient)."""
        with torch.no_grad():
            return F.softplus(self.raw_radius)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight * self._mask()
        return F.conv1d(
            x, masked_weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"k={self.kernel_size}, padding={self.padding}, "
            f"tau={self.tau}, r_trainable={self.raw_radius.requires_grad}"
        )


# ---------------------------------------------------------------------------
# LearnableAttention1d
# ---------------------------------------------------------------------------

class LearnableAttention1d(nn.Module):
    """Purely-parametric (input-independent) attention for ``[B, C, L]`` maps.

    Parameters
    ----------
    channels : int
        Feature-map channel count C.
    seq_len : int or None
        Spatial length for position attention weight registration.
        Required when ``use_position=True``.  When the forward-time spatial
        dimension differs (e.g. after augmentation or pooling), the weights
        are linearly interpolated.
    use_channel : bool
        Enable the channel attention component.
    use_position : bool
        Enable the position attention component.
    """

    def __init__(
        self,
        channels: int,
        seq_len: Optional[int] = None,
        use_channel: bool = True,
        use_position: bool = False,
    ):
        super().__init__()
        self.use_channel  = use_channel
        self.use_position = use_position

        if use_channel:
            # sigmoid(4.0) ≈ 0.982 ≈ identity at init
            self.channel_weight = nn.Parameter(torch.full((channels,), 4.0))

        if use_position:
            if seq_len is None:
                raise ValueError(
                    "LearnableAttention1d: seq_len is required when use_position=True"
                )
            # All-zero → softmax uniform → ×L = 1.0 at every position (identity)
            self.position_weight = nn.Parameter(
                torch.zeros(1, channels, seq_len)
            )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        if self.use_channel:
            gate = torch.sigmoid(self.channel_weight).view(1, -1, 1)
            x = x * gate

        if self.use_position:
            pos_w = self.position_weight
            L_in  = x.shape[2]
            if pos_w.shape[2] != L_in:
                # Interpolate to match actual spatial size (augmentation, pooling)
                pos_w = F.interpolate(
                    pos_w, size=L_in, mode="linear", align_corners=False
                )
            # softmax over position axis → normalised weights, scale by L to
            # preserve feature-map magnitude relative to the uniform baseline
            pos_w = F.softmax(pos_w, dim=2) * L_in
            x = x * pos_w

        return x

    def extra_repr(self) -> str:
        ch = self.channel_weight.shape[0] if self.use_channel else "–"
        sl = self.position_weight.shape[2] if self.use_position else "–"
        return (
            f"channels={ch}, seq_len={sl}, "
            f"channel={self.use_channel}, position={self.use_position}"
        )
