# -*- coding: utf-8 -*-
"""Model registry for dynamic model construction from YAML configuration.

Provides a decorator-based registration system so that any model class
can be instantiated by name at runtime.
"""

import torch.nn as nn

MODELS = {}


def register_model(name: str):
    """Register a model class under the given name.

    Parameters
    ----------
    name : str
        Unique identifier used in YAML config ``model.name``.

    Returns
    -------
    Callable
        Class decorator that inserts the class into :data:`MODELS`.
    """
    def decorator(cls):
        MODELS[name] = cls
        return cls
    return decorator


def build_model(config: dict):
    """Instantiate and return a model from a YAML configuration dictionary.

    Besides basic construction, this function applies two optional in-place
    modifications controlled by dedicated config sections:

    ``model.mask_conv``
        Replaces every ``nn.Conv1d`` with a :class:`~models.layers.MaskedConv1d`
        whose kernel is multiplied by a per-filter learnable soft mask.
        The mask radius *r* controls the effective receptive-field width and
        is trained by backprop (when ``trainable: true``).

        Example YAML section::

            model:
              mask_conv:
                enabled: true
                r: 0.7          # initial radius fraction (0–1, default 1.0)
                tau: 5.0        # mask sharpness (default 5.0)
                trainable: true # train radius by backprop (default true)
                skip_depthwise: false
                skip_pointwise: true

    ``model.attention``
        Appends a :class:`~models.layers.LearnableAttention1d` module after
        every ``nn.Conv1d`` layer.  The attention starts at the identity
        transform and learns per-channel or per-position weights directly
        from the loss (no input projection).

        Example YAML section::

            model:
              attention:
                enabled: true
                type: "channel"   # "channel" | "position" | "both"
                seq_len: 249      # required for position or both
                skip_pointwise: true
                skip_depthwise: false

    If both sections are present and ``enabled: true``, masked conv is
    applied first, then attention is injected.

    Parameters
    ----------
    config : dict
        Parsed YAML config.  Must contain ``model.name`` and optionally
        ``model.kwargs``, ``model.mask_conv``, ``model.attention``.

    Returns
    -------
    torch.nn.Module
        Fully configured model instance.

    Raises
    ------
    ValueError
        If ``model.name`` is not found in the registry.
    """
    model_cfg  = config['model']
    model_name = model_cfg['name']
    model_kwargs = model_cfg.get('kwargs', {})

    if model_name not in MODELS:
        raise ValueError(
            f"Model '{model_name}' is not registered. "
            f"Available models: {list(MODELS.keys())}"
        )

    model = MODELS[model_name](**model_kwargs)

    # ------------------------------------------------------------------ #
    # Optional: Masked Convolution                                         #
    # ------------------------------------------------------------------ #
    mask_cfg = model_cfg.get('mask_conv', {})
    if mask_cfg.get('enabled', False):
        from .wrappers import inject_masked_conv
        inject_masked_conv(
            model,
            r              = float(mask_cfg.get('r', 1.0)),
            tau            = float(mask_cfg.get('tau', 5.0)),
            trainable      = bool(mask_cfg.get('trainable', True)),
            skip_depthwise = bool(mask_cfg.get('skip_depthwise', False)),
            skip_pointwise = bool(mask_cfg.get('skip_pointwise', True)),
        )
        n_masked = sum(
            1 for m in model.modules()
            if m.__class__.__name__ == 'MaskedConv1d'
        )
        print(f"[model] MaskedConv1d applied to {n_masked} Conv1d layer(s) "
              f"(r={mask_cfg.get('r', 1.0)}, tau={mask_cfg.get('tau', 5.0)}, "
              f"trainable={mask_cfg.get('trainable', True)})")

    # ------------------------------------------------------------------ #
    # Optional: Learnable Attention                                        #
    # ------------------------------------------------------------------ #
    attn_cfg = model_cfg.get('attention', {})
    if attn_cfg.get('enabled', False):
        from .wrappers import inject_attention
        inject_attention(
            model,
            type           = str(attn_cfg.get('type', 'channel')),
            seq_len        = attn_cfg.get('seq_len', None),
            skip_pointwise = bool(attn_cfg.get('skip_pointwise', True)),
            skip_depthwise = bool(attn_cfg.get('skip_depthwise', False)),
        )
        n_attn = sum(
            1 for m in model.modules()
            if m.__class__.__name__ == 'LearnableAttention1d'
        )
        print(f"[model] LearnableAttention1d injected after {n_attn} Conv1d layer(s) "
              f"(type='{attn_cfg.get('type', 'channel')}', "
              f"seq_len={attn_cfg.get('seq_len')})")

    # ------------------------------------------------------------------ #
    # Optional: Single-output ablation mode                                #
    # ------------------------------------------------------------------ #
    output_head = model_cfg.get('output_head', 'both')
    if output_head != 'both':
        model = SingleOutputWrapper(model, output_head)
        print(f"[model] Single-output mode: only '{output_head}' head active")

    return model


class SingleOutputWrapper(nn.Module):
    """Wraps a dual-output model to produce only one output.

    Used for ablation studies comparing multitask vs single-task learning.
    The model still computes both outputs internally, but only one is
    returned (and only one gets gradient signal during training).

    Parameters
    ----------
    model : nn.Module
        A model whose forward() returns (out_dev, out_hk).
    output : str
        Which output to keep: 'dev' (index 0) or 'hk' (index 1).
    """

    def __init__(self, model, output='dev'):
        super().__init__()
        self.model = model
        if output not in ('dev', 'hk'):
            raise ValueError(f"output_head must be 'dev', 'hk', or 'both', got: {output}")
        self.output_idx = 0 if output == 'dev' else 1
        self.output_name = output

    def forward(self, x):
        outputs = self.model(x)
        return outputs[self.output_idx]

    def get_features(self, x):
        if hasattr(self.model, 'get_features'):
            return self.model.get_features(x)
        raise AttributeError("Wrapped model has no get_features method")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

