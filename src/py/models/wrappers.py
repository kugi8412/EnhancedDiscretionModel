# -*- coding: utf-8 -*-
"""
In-place injection utilities for MaskedConv1d and LearnableAttention1d.

inject_masked_conv
    Recursively replaces every ``nn.Conv1d`` inside a model with a
    ``MaskedConv1d`` that learns per-filter receptive-field widths.
    Existing weights are copied so the switch is seamless at any training
    checkpoint.

inject_attention
    Recursively wraps every ``nn.Conv1d`` with a subsequent
    ``LearnableAttention1d`` by replacing it with
    ``nn.Sequential(conv, attention)``.  The attention module is initialised
    to the identity transform, so training dynamics are undisturbed at step 0.

Both utilities are called automatically by ``build_model`` in
``registry.py`` when the corresponding config sections are present:

    model:
      mask_conv:
        enabled: true
        ...
      attention:
        enabled: true
        ...
"""

from typing import Optional

import torch
import torch.nn as nn

from .layers import LearnableAttention1d, MaskedConv1d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scalar(val) -> int:
    """Unwrap a length-1 tuple/list, or return the value as-is."""
    return val[0] if isinstance(val, (tuple, list)) else val


def _replace_child(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Set a child module on *parent* by *name* (works for Sequential too)."""
    # nn.Module.__setattr__ registers nn.Module instances in ._modules, so
    # a plain setattr call is the correct and supported way.
    setattr(parent, name, new_module)


# ---------------------------------------------------------------------------
# inject_masked_conv
# ---------------------------------------------------------------------------

def inject_masked_conv(
    model: nn.Module,
    r: float = 1.0,
    tau: float = 5.0,
    trainable: bool = True,
    skip_depthwise: bool = False,
    skip_pointwise: bool = True,
) -> None:
    """Replace ``nn.Conv1d`` layers inside *model* with :class:`MaskedConv1d`.

    The traversal is recursive so all nested sub-modules are covered.
    Layers that are already :class:`MaskedConv1d` instances are skipped to
    prevent double-wrapping.

    Parameters
    ----------
    model : nn.Module
        Target model, modified **in-place**.
    r : float
        Initial kernel-mask radius as a fraction of the kernel half-width.
        ``1.0`` keeps the full kernel active at init (default).
    tau : float
        Sigmoid sharpness of the mask boundary.
    trainable : bool
        Whether each filter's radius is updated by backprop.
    skip_depthwise : bool
        If ``True``, depthwise convolutions (``groups == in_channels``) are
        left untouched.  Useful when depthwise convs serve as position
        encoders and their receptive field should remain fixed.
    skip_pointwise : bool
        If ``True`` (default), pointwise convolutions (``kernel_size == 1``)
        are skipped — masking a 1-element kernel has no effect.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Conv1d) and not isinstance(module, MaskedConv1d):
            ks = _scalar(module.kernel_size)

            if skip_pointwise and ks == 1:
                inject_masked_conv(
                    module, r=r, tau=tau, trainable=trainable,
                    skip_depthwise=skip_depthwise, skip_pointwise=skip_pointwise,
                )
                continue

            is_dw = (module.groups == module.in_channels and module.in_channels > 1)
            if skip_depthwise and is_dw:
                inject_masked_conv(
                    module, r=r, tau=tau, trainable=trainable,
                    skip_depthwise=skip_depthwise, skip_pointwise=skip_pointwise,
                )
                continue

            # Build padding: pass through 'same'/'valid' strings unchanged;
            # unwrap integer tuples to a plain int.
            padding = module.padding
            if isinstance(padding, (tuple, list)):
                padding = padding[0]

            new_conv = MaskedConv1d(
                in_channels  = module.in_channels,
                out_channels = module.out_channels,
                kernel_size  = ks,
                stride       = _scalar(module.stride),
                padding      = padding,
                dilation     = _scalar(module.dilation),
                groups       = module.groups,
                bias         = module.bias is not None,
                r_init       = r,
                tau          = tau,
                r_trainable  = trainable,
            )

            # Copy learned weights so any prior training is preserved
            with torch.no_grad():
                new_conv.weight.copy_(module.weight)
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias)

            _replace_child(model, name, new_conv)

        else:
            # Recurse into containers (Sequential, ModuleList, custom blocks)
            inject_masked_conv(
                module, r=r, tau=tau, trainable=trainable,
                skip_depthwise=skip_depthwise, skip_pointwise=skip_pointwise,
            )


# ---------------------------------------------------------------------------
# inject_attention
# ---------------------------------------------------------------------------

def inject_attention(
    model: nn.Module,
    type: str = "channel",
    seq_len: Optional[int] = None,
    skip_pointwise: bool = True,
    skip_depthwise: bool = False,
) -> None:
    """Append :class:`LearnableAttention1d` after each ``nn.Conv1d`` layer.

    Each matched ``conv`` attribute is replaced by
    ``nn.Sequential(conv, LearnableAttention1d(...))``.  Because the
    attention module starts at the identity transform, optimisation behaviour
    is unchanged at step 0.

    Parameters
    ----------
    model : nn.Module
        Target model, modified **in-place**.
    type : {'channel', 'position', 'both'}
        Which attention component(s) to insert.
    seq_len : int or None
        Sequence length for the position-attention weight tensor.
        Required when *type* is ``'position'`` or ``'both'``.
        If the actual spatial dimension differs at forward time the weights
        are linearly interpolated automatically.
    skip_pointwise : bool
        If ``True`` (default), pointwise (kernel_size == 1) convolutions are
        skipped — they operate on channels only and position attention on a
        1-element window is meaningless.
    skip_depthwise : bool
        If ``True``, depthwise convolutions are skipped.
    """
    if type not in ("channel", "position", "both"):
        raise ValueError(
            f"inject_attention: type must be 'channel', 'position', or 'both', "
            f"got '{type}'"
        )
    use_channel  = type in ("channel", "both")
    use_position = type in ("position", "both")

    for name, module in list(model.named_children()):
        if isinstance(module, nn.Conv1d):
            ks    = _scalar(module.kernel_size)
            is_dw = (module.groups == module.in_channels and module.in_channels > 1)

            if skip_pointwise and ks == 1:
                inject_attention(
                    module, type=type, seq_len=seq_len,
                    skip_pointwise=skip_pointwise, skip_depthwise=skip_depthwise,
                )
                continue

            if skip_depthwise and is_dw:
                inject_attention(
                    module, type=type, seq_len=seq_len,
                    skip_pointwise=skip_pointwise, skip_depthwise=skip_depthwise,
                )
                continue

            attn = LearnableAttention1d(
                channels     = module.out_channels,
                seq_len      = seq_len,
                use_channel  = use_channel,
                use_position = use_position,
            )
            # Wrap: conv output is fed directly into the attention module
            _replace_child(model, name, nn.Sequential(module, attn))

        else:
            inject_attention(
                module, type=type, seq_len=seq_len,
                skip_pointwise=skip_pointwise, skip_depthwise=skip_depthwise,
            )
