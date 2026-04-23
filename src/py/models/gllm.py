"""Genomic Large Language Model (gLLM) adapters.

Wraps pretrained DNA foundation models so they plug into the same
training / evaluation / SAE pipeline as the CNN models.

Architecture
------------
Each adapter follows the same pattern:
  1. Accept one-hot encoded DNA ``[B, 4, L]``.
  2. Decode to a list of plain DNA strings (argmax along channel dim).
  3. Tokenise and run through the frozen (or LoRA-adapted) backbone.
  4. Pool the last-hidden-state to ``[B, embed_dim]``.
  5. Pass through two independent regression heads → ``(pred_dev, pred_hk)``.

Strand standardisation
----------------------
All callers are expected to pass **plus-strand sequences** only.  The
data pipeline in ``utils.prepare_input`` applies RC correction for
``_-_`` FASTA headers when ``data.strand_correct: true`` (the default).
Neither DNABERT-2 nor Evo2 are RC-equivariant, so feeding minus-strand
sequences without correction would degrade their predictions.

Registered names
----------------
``"DNABert2"``  — DNABERT-2-117M (zhihan1996/DNABERT-2-117M, 117 M params)
``"Evo2"``      — Evo2-1b-base  (arcinstitute/evo2_1b_base, 1.1 B params)

Optional dependencies
---------------------
* ``transformers >= 4.36``   — required for DNABert2
* ``evo2``                   — required for Evo2 (``pip install evo2``)
* ``peft``                   — optional, enables LoRA fine-tuning for both

If a backbone library is not installed the corresponding model class is
still importable but raises ``ImportError`` when instantiated.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASES_IDX = ('A', 'C', 'T', 'G')   # must match one_hot_encode_dna channel order


def _decode_onehot(x: torch.Tensor) -> list:
    """Decode a one-hot batch ``[B, 4, L]`` to a list of DNA strings.

    Uses ``argmax`` along the channel dimension; all-zero positions
    (padding / unknown bases) are decoded as ``'N'``.
    """
    max_vals, indices = x.max(dim=1)          # [B, L]
    seqs = []
    for b in range(x.size(0)):
        chars = []
        for pos in range(x.size(2)):
            if max_vals[b, pos].item() == 0.0:
                chars.append('N')
            else:
                chars.append(_BASES_IDX[indices[b, pos].item()])
        seqs.append(''.join(chars))
    return seqs


def _build_head(embed_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """Shared regression head: Linear → BN → SiLU → Dropout → Linear."""
    return nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


# ---------------------------------------------------------------------------
# DNABERT-2 adapter
# ---------------------------------------------------------------------------

@register_model("DNABert2")
class DNABert2Wrapper(nn.Module):
    """DNABERT-2 fine-tuning adapter for enhancer activity regression.

    Parameters
    ----------
    freeze_backbone : bool
        If *True* (default), the DNABERT-2 backbone weights are frozen and
        only the regression heads are trained.  Set to *False* for full
        fine-tuning.
    lora : bool
        Inject LoRA adapters via ``peft``.  Requires ``pip install peft``.
        Incompatible with ``freeze_backbone=True``.
    lora_r : int
        LoRA rank (default 8).
    lora_alpha : float
        LoRA alpha scaling factor (default 16).
    hidden_dim : int
        Hidden layer width in the regression heads.
    dropout_rate : float
        Dropout probability in regression heads (enables MC Dropout).
    pool : str
        Pooling strategy: ``'cls'`` (default) or ``'mean'``.
    pretrained_name : str
        HuggingFace model ID.
    """

    def __init__(
        self,
        freeze_backbone: bool = True,
        lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        pool: str = 'cls',
        pretrained_name: str = 'zhihan1996/DNABERT-2-117M',
        **kwargs,
    ):
        super().__init__()

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as exc:
            raise ImportError(
                "DNABert2Wrapper requires 'transformers'. "
                "Install with: pip install transformers"
            ) from exc

        logger.info("Loading DNABERT-2 backbone: %s", pretrained_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_name, trust_remote_code=True)
        self.backbone  = AutoModel.from_pretrained(
            pretrained_name, trust_remote_code=True)

        self.pool = pool
        embed_dim = self.backbone.config.hidden_size   # 768 for 117M

        if lora:
            try:
                from peft import get_peft_model, LoraConfig, TaskType
            except ImportError as exc:
                raise ImportError(
                    "LoRA requires 'peft'. Install with: pip install peft"
                ) from exc
            peft_cfg = LoraConfig(
                r=int(lora_r),
                lora_alpha=float(lora_alpha),
                target_modules=["query", "value"],
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, peft_cfg)
            self.backbone.print_trainable_parameters()
        elif freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("DNABERT-2 backbone frozen — training heads only.")

        self.head_dev = _build_head(embed_dim, hidden_dim, dropout_rate)
        self.head_hk  = _build_head(embed_dim, hidden_dim, dropout_rate)

    # ------------------------------------------------------------------

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """One-hot ``[B, 4, L]`` → pooled embedding ``[B, embed_dim]``."""
        seqs = _decode_onehot(x)
        enc  = self.tokenizer(
            seqs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc  = {k: v.to(x.device) for k, v in enc.items()}
        out  = self.backbone(**enc)            # BaseModelOutputWithPooling

        if self.pool == 'cls':
            return out.last_hidden_state[:, 0, :]   # [CLS] token
        else:  # mean pooling over non-padding tokens
            mask = enc['attention_mask'].unsqueeze(-1).float()
            return (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(pred_dev, pred_hk)`` each ``[B, 1]``."""
        emb = self._embed(x)
        return self.head_dev(emb), self.head_hk(emb)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled backbone embedding ``[B, embed_dim]`` for SAE input."""
        with torch.no_grad():
            return self._embed(x)


# ---------------------------------------------------------------------------
# Evo2 adapter
# ---------------------------------------------------------------------------

@register_model("Evo2")
class Evo2Wrapper(nn.Module):
    """Evo2 fine-tuning adapter for enhancer activity regression.

    Uses the Arc Institute ``evo2`` package or the HuggingFace
    ``arcinstitute/evo2_1b_base`` checkpoint.

    Parameters
    ----------
    model_name : str
        Evo2 model size: ``'evo2_1b_base'`` (default) or
        ``'evo2_7b_base'``, ``'evo2_40b_base'``.
    freeze_backbone : bool
        Freeze Evo2 weights and train regression heads only.
    lora : bool
        Inject LoRA adapters (requires ``peft``).
    lora_r : int
    lora_alpha : float
    hidden_dim : int
        Regression head hidden width.
    dropout_rate : float
        Dropout probability (enables MC Dropout).
    pool : str
        ``'mean'`` (default) or ``'last'`` token pooling.
    """

    def __init__(
        self,
        model_name: str = 'evo2_1b_base',
        freeze_backbone: bool = True,
        lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        pool: str = 'mean',
        **kwargs,
    ):
        super().__init__()

        try:
            from evo2 import Evo as _Evo   # Arc Institute evo2 package
            _api = 'evo2_pkg'
        except ImportError:
            try:
                from transformers import AutoModel, AutoTokenizer
                _api = 'hf'
            except ImportError as exc:
                raise ImportError(
                    "Evo2Wrapper requires either the 'evo2' package "
                    "(pip install evo2) or 'transformers' with HuggingFace "
                    "weights (pip install transformers)."
                ) from exc

        self.pool = pool
        self._api  = _api

        if _api == 'evo2_pkg':
            logger.info("Loading Evo2 via evo2 package: %s", model_name)
            _instance        = _Evo(model_name)
            self.backbone    = _instance.model
            self.tokenizer   = _instance.tokenizer
            embed_dim        = self.backbone.config.d_model
        else:
            hf_name = f'arcinstitute/{model_name}'
            logger.info("Loading Evo2 from HuggingFace: %s", hf_name)
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                hf_name, trust_remote_code=True)
            self.backbone  = AutoModel.from_pretrained(
                hf_name, trust_remote_code=True)
            embed_dim = self.backbone.config.hidden_size

        if lora:
            try:
                from peft import get_peft_model, LoraConfig
            except ImportError as exc:
                raise ImportError("LoRA requires 'peft'. pip install peft") from exc
            peft_cfg = LoraConfig(
                r=int(lora_r), lora_alpha=float(lora_alpha),
                target_modules=["q_proj", "v_proj"], bias="none",
            )
            self.backbone = get_peft_model(self.backbone, peft_cfg)
        elif freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("Evo2 backbone frozen — training heads only.")

        self.embed_dim = embed_dim
        self.head_dev  = _build_head(embed_dim, hidden_dim, dropout_rate)
        self.head_hk   = _build_head(embed_dim, hidden_dim, dropout_rate)

    # ------------------------------------------------------------------

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """One-hot ``[B, 4, L]`` → pooled Evo2 embedding ``[B, embed_dim]``."""
        seqs = _decode_onehot(x)

        if self._api == 'evo2_pkg':
            # evo2 package: tokenize each sequence separately
            device = x.device
            ids_list = []
            max_len  = 0
            for seq in seqs:
                ids, _ = self.tokenizer(seq, return_tensors='pt')
                ids_list.append(ids.squeeze(0))
                max_len = max(max_len, ids.size(-1))
            # Pad to same length
            padded = torch.zeros(len(ids_list), max_len, dtype=torch.long, device=device)
            for i, ids in enumerate(ids_list):
                padded[i, :ids.size(0)] = ids.to(device)
            out, _ = self.backbone(padded, return_embeddings=True)
            # out: [B, L, embed_dim]
        else:
            enc = self.tokenizer(
                seqs, return_tensors='pt', padding=True,
                truncation=True, max_length=2048,
            )
            enc = {k: v.to(x.device) for k, v in enc.items()}
            out = self.backbone(**enc).last_hidden_state      # [B, L, D]

        # Pool
        if self.pool == 'last':
            return out[:, -1, :]
        else:
            return out.mean(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(pred_dev, pred_hk)`` each ``[B, 1]``."""
        emb = self._embed(x)
        return self.head_dev(emb), self.head_hk(emb)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled backbone embedding ``[B, embed_dim]`` for SAE input."""
        with torch.no_grad():
            return self._embed(x)
