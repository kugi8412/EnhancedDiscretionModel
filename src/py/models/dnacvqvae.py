# -*- coding: utf-8 -*-
"""Conditional VQ-VAE for DNA enhancer sequence modelling (multitask edition).

Architecture
------------
The model consists of four cooperating subsystems:

**Encoder** (configurable depth — "heavy" by default)
    CNN stem → N × (EffBlock + MaxPool) → Bidirectional GRU → pre-VQ
    projection.  Encoder depth is controlled by ``encoder_depth``; each
    stage halves the spatial dimension.

**EMA Vector Quantizer**
    Differentiable codebook with exponential moving average updates.
    Produces quantized latent codes together with the VQ commitment loss.

**Multitask prediction heads** (new)
    Two lightweight regression heads applied directly to the *quantized*
    latent (no oracle required at inference time).  The oracle is still
    supported as an auxiliary training signal but is no longer mandatory.

**FiLM-conditioned decoder** (configurable depth — "light" by default)
    FiLM modulation → projection → Bidirectional GRU → N × ConvTranspose1d
    → output logits for both strands (forward + RC).  Decoder depth is
    controlled by ``decoder_depth``.

**Sequence rewriter** (new)
    ``rewrite(x, target_dev, target_hk)`` re-encodes the input, then
    decodes with a new target activity vector.  The output is a Gumbel-hard
    one-hot sequence conditioned on the desired expression profile.

Registered names
----------------
- ``"cVQVAE_MultiTask"``   — new multitask model (recommended)
- ``"HydraDNA_cVQVAE"``    — original model kept for backward compatibility

Example YAML
------------
.. code-block:: yaml

    model:
      name: "cVQVAE_MultiTask"
      kwargs:
        in_ch: 4
        stem_ch: 192
        gru_dim: 192
        vq_dim: 128
        num_embeddings: 4096
        encoder_depth: 4       # heavy encoder (default 3)
        decoder_depth: 2       # light decoder (default == encoder_depth)
        commitment_cost: 0.25
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class EMAVectorQuantizer(nn.Module):
    """Vector quantizer with exponential moving average codebook updates.

    Parameters
    ----------
    num_embeddings : int
        Codebook size K.
    embedding_dim : int
        Dimension of each codebook vector.
    commitment_cost : float
        Weight on the encoder commitment loss term (β in VQ-VAE paper).
    decay : float
        EMA decay rate for codebook update.
    epsilon : float
        Laplace smoothing factor to avoid division by zero.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat_inputs = inputs.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                     + torch.sum(self.embed**2, dim=1) 
                     - 2 * torch.matmul(flat_inputs, self.embed.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embed).view(input_shape)

        if self.training:
            cluster_size = torch.sum(encodings, dim=0)
            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            embed_sum = torch.matmul(encodings.t(), flat_inputs)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = torch.sum(self.cluster_size.data)
            self.cluster_size.data = (self.cluster_size.data + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            self.embed.data.copy_(self.embed_avg.data / self.cluster_size.data.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()
        return quantized, loss, encoding_indices.view(input_shape[0], input_shape[1])

class SELayer(nn.Module):
    """Channel-wise Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid), nn.SiLU(),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = x.mean(2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class EffBlock(nn.Module):
    """EfficientNet-style inverted-residual block with depthwise separable conv."""

    def __init__(self, channels: int, kernel_size: int = 5, expand: int = 4):
        super().__init__()
        inner = channels * expand
        self.block = nn.Sequential(
            nn.Conv1d(channels, inner, 1, padding="same", bias=False),
            nn.BatchNorm1d(inner), nn.SiLU(),
            nn.Conv1d(inner, inner, kernel_size, groups=inner, padding="same", bias=False),
            nn.BatchNorm1d(inner), nn.SiLU(),
            SELayer(inner, reduction=expand),
            nn.Conv1d(inner, channels, 1, padding="same", bias=False),
            nn.BatchNorm1d(channels), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FiLMGenerator(nn.Module):
    """Generates FiLM γ and β modulation vectors from a scalar activity pair.

    Parameters
    ----------
    cond_dim : int
        Conditioning input dimension (2 for Dev + Hk).
    out_dim : int
        Output dimension equal to ``vq_dim``.
    hidden_dim : int
        Width of the MLP hidden layer.
    """

    def __init__(self, cond_dim: int = 2, out_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, out_dim * 2),
        )

    def forward(self, cond: torch.Tensor) -> tuple:
        """Return ``(gamma, beta)``, each shape ``(B, out_dim, 1)``."""
        params = self.net(cond.float())
        half   = params.shape[1] // 2
        return params[:, :half].unsqueeze(2), params[:, half:].unsqueeze(2)


# ---------------------------------------------------------------------------
# Encoder / Decoder factory helpers
# ---------------------------------------------------------------------------

def _build_encoder(in_ch, stem_ch, gru_dim, vq_dim, depth, kernel_size=5):
    """Return (cnn_stem, cnn_blocks, encoder_gru, pre_vq_conv)."""
    cnn_stem = nn.Sequential(
        nn.Conv1d(in_ch, stem_ch, kernel_size=7, padding=3, bias=False),
        nn.BatchNorm1d(stem_ch), nn.SiLU(),
    )
    blocks = []
    for _ in range(depth):
        blocks += [EffBlock(stem_ch, kernel_size=kernel_size), nn.MaxPool1d(2)]
    cnn_blocks  = nn.Sequential(*blocks)
    encoder_gru = nn.GRU(stem_ch, gru_dim, batch_first=True, bidirectional=True)
    pre_vq_conv = nn.Conv1d(gru_dim * 2, vq_dim, kernel_size=1)
    return cnn_stem, cnn_blocks, encoder_gru, pre_vq_conv


def _build_decoder(gru_dim, stem_ch, in_ch, vq_dim, depth):
    """Return (cond_proj, decoder_gru, dec_blocks, dec_out)."""
    cond_proj   = nn.Conv1d(vq_dim, gru_dim * 2, kernel_size=1)
    decoder_gru = nn.GRU(gru_dim * 2, gru_dim, batch_first=True, bidirectional=True)
    layers      = []
    in_ch_dec   = gru_dim * 2
    for _ in range(depth):
        layers += [
            nn.ConvTranspose1d(in_ch_dec, stem_ch, kernel_size=4, stride=2, padding=1),
            EffBlock(stem_ch),
        ]
        in_ch_dec = stem_ch
    dec_blocks = nn.Sequential(*layers)
    dec_out    = nn.Conv1d(stem_ch, in_ch * 2, kernel_size=5, padding=2)
    return cond_proj, decoder_gru, dec_blocks, dec_out


# ---------------------------------------------------------------------------
# New multitask model
# ---------------------------------------------------------------------------

@register_model("cVQVAE_MultiTask")
class cVQVAE_MultiTask(nn.Module):
    """Conditional VQ-VAE with multitask regression heads and sequence rewriting.

    Parameters
    ----------
    in_ch : int
        Input channels (4 for one-hot DNA).
    stem_ch : int
        CNN feature-map width.
    gru_dim : int
        GRU hidden size; bidirectional output is ``gru_dim * 2``.
    vq_dim : int
        Codebook vector dimension.
    num_embeddings : int
        Codebook size.
    encoder_depth : int
        CNN downsampling stages (heavy encoder when large).
    decoder_depth : int or None
        CNN upsampling stages.  Defaults to ``encoder_depth``.
        Use a smaller value for a lighter decoder.
    commitment_cost : float
        VQ commitment loss weight β.
    classifier_dim : int
        Hidden size of the regression prediction heads.
    dropout_rate : float
        Dropout probability in heads (enables MC Dropout at inference).
    kernel_size : int
        EffBlock depthwise convolution kernel size.
    film_hidden : int
        Hidden size of the FiLM generator MLP.
    uncond_dropout_rate : float
        Classifier-free guidance: probability of zeroing the activity
        conditioning vector per batch element during training.
    """

    def __init__(
        self,
        in_ch: int = 4,
        stem_ch: int = 192,
        gru_dim: int = 192,
        vq_dim: int = 128,
        num_embeddings: int = 4096,
        encoder_depth: int = 3,
        decoder_depth: int = None,
        commitment_cost: float = 0.25,
        classifier_dim: int = 256,
        dropout_rate: float = 0.2,
        kernel_size: int = 5,
        film_hidden: int = 128,
        uncond_dropout_rate: float = 0.15,
        **kwargs,
    ):
        super().__init__()
        if decoder_depth is None:
            decoder_depth = encoder_depth

        self.vq_dim              = vq_dim
        self.uncond_dropout_rate = uncond_dropout_rate

        # Encoder
        (self.cnn_stem,
         self.cnn_blocks,
         self.encoder_gru,
         self.pre_vq_conv) = _build_encoder(in_ch, stem_ch, gru_dim, vq_dim, encoder_depth, kernel_size)

        # Quantizer
        self.vq_layer = EMAVectorQuantizer(num_embeddings, vq_dim, commitment_cost=commitment_cost)

        # Multitask regression heads
        self.head_dev = nn.Sequential(
            nn.Linear(vq_dim, classifier_dim), nn.BatchNorm1d(classifier_dim),
            nn.SiLU(), nn.Dropout(dropout_rate),
            nn.Linear(classifier_dim, classifier_dim // 2), nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_dim // 2, 1),
        )
        self.head_hk = nn.Sequential(
            nn.Linear(vq_dim, classifier_dim), nn.BatchNorm1d(classifier_dim),
            nn.SiLU(), nn.Dropout(dropout_rate),
            nn.Linear(classifier_dim, classifier_dim // 2), nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_dim // 2, 1),
        )

        # FiLM conditioning
        self.film = FiLMGenerator(cond_dim=2, out_dim=vq_dim, hidden_dim=film_hidden)

        # Decoder
        (self.dec_cond_proj,
         self.decoder_gru,
         self.decoder_blocks,
         self.decoder_out) = _build_decoder(gru_dim, stem_ch, in_ch, vq_dim, decoder_depth)

    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """CNN + GRU encoder to pre-VQ latent ``[B, vq_dim, L']``."""
        h = self.cnn_stem(x)
        h = self.cnn_blocks(h)
        h = h.permute(0, 2, 1)
        h, _ = self.encoder_gru(h)
        h = h.permute(0, 2, 1)
        return self.pre_vq_conv(h)

    def _decode(self, quantized, gamma, beta, target_len):
        """FiLM-modulate the quantized latent and upsample to logits."""
        cq = (1.0 + gamma) * quantized + beta
        d = self.dec_cond_proj(cq)
        d = d.permute(0, 2, 1)
        d, _ = self.decoder_gru(d)
        d = d.permute(0, 2, 1)
        d = self.decoder_blocks(d)
        logits = self.decoder_out(d)
        if logits.size(2) != target_len:
            logits = F.interpolate(logits, size=target_len, mode="linear", align_corners=False)
        return logits[:, :4], logits[:, 4:]

    # ------------------------------------------------------------------

    def forward(self, x, y_dev=None, y_hk=None, tau=1.0):
        """Full forward pass (training).

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, 4, L)``
        y_dev, y_hk : torch.Tensor, shape ``(B,)`` or None
            Ground-truth activity values used for FiLM conditioning.
        tau : float
            Gumbel-Softmax temperature.

        Returns
        -------
        logits_8ch : torch.Tensor, shape ``(B, 8, L)``
        (fwd_gumbel, rc_gumbel) : each shape ``(B, 4, L)``
        vq_loss : scalar tensor
        pred_dev : torch.Tensor, shape ``(B, 1)``
        pred_hk  : torch.Tensor, shape ``(B, 1)``
        """
        L = x.size(2)
        B = x.size(0)

        z               = self._encode(x)
        quantized, vq_loss, _ = self.vq_layer(z)

        q_pooled = F.adaptive_avg_pool1d(quantized, 1).squeeze(-1)
        pred_dev = self.head_dev(q_pooled)
        pred_hk  = self.head_hk(q_pooled)

        if y_dev is not None and y_hk is not None:
            cond = torch.stack([y_dev.float(), y_hk.float()], dim=1)
            if self.training:
                drop = (torch.rand(B, 1, device=x.device) < self.uncond_dropout_rate)
                cond = cond * (~drop).float()
        else:
            cond = torch.zeros(B, 2, device=x.device)

        gamma, beta = self.film(cond)
        fwd_logits, rc_logits = self._decode(quantized, gamma, beta, L)

        fwd_g = F.gumbel_softmax(fwd_logits, tau=tau, hard=True, dim=1)
        rc_g  = F.gumbel_softmax(rc_logits,  tau=tau, hard=True, dim=1)

        return torch.cat([fwd_logits, rc_logits], dim=1), (fwd_g, rc_g), vq_loss, pred_dev, pred_hk

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor):
        """Encode *x* to quantized latent (no gradient).

        Returns
        -------
        quantized : ``[B, vq_dim, L']``
        indices   : ``[B, L']``
        """
        self.eval()
        z = self._encode(x)
        quantized, _, indices = self.vq_layer(z)
        return quantized, indices

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """Multitask-only forward (no decoding, faster inference).

        Returns
        -------
        pred_dev, pred_hk : each ``(B, 1)``
        """
        self.eval()
        z             = self._encode(x)
        quantized, _, _ = self.vq_layer(z)
        q_pooled      = F.adaptive_avg_pool1d(quantized, 1).squeeze(-1)
        return self.head_dev(q_pooled), self.head_hk(q_pooled)

    @torch.no_grad()
    def rewrite(self, x: torch.Tensor, target_dev, target_hk, tau: float = 0.1):
        """Rewrite a sequence towards a target expression profile.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, 4, L)``
        target_dev, target_hk : float or torch.Tensor of shape ``(B,)``
            Desired log2-enrichment values.
        tau : float
            Gumbel temperature (lower = more deterministic output).

        Returns
        -------
        torch.Tensor, shape ``(B, 4, L)``
            Gumbel-hard one-hot rewritten sequences.
        """
        self.eval()
        L = x.size(2)
        B = x.size(0)

        z             = self._encode(x)
        quantized, _, _ = self.vq_layer(z)

        if isinstance(target_dev, (int, float)):
            target_dev = torch.full((B,), float(target_dev), device=x.device)
        if isinstance(target_hk, (int, float)):
            target_hk  = torch.full((B,), float(target_hk),  device=x.device)

        cond        = torch.stack([target_dev.float(), target_hk.float()], dim=1)
        gamma, beta = self.film(cond)
        fwd_logits, _ = self._decode(quantized, gamma, beta, L)
        return F.gumbel_softmax(fwd_logits, tau=tau, hard=True, dim=1)


# ---------------------------------------------------------------------------
# Original model — preserved for checkpoint backward compatibility
# ---------------------------------------------------------------------------

@register_model("HydraDNA_cVQVAE")
class HydraDNA_cVQVAE(nn.Module):
    """Original conditional VQ-VAE with FiLM conditioning.

    Preserved unchanged for backward compatibility with existing checkpoints.
    For new experiments, prefer :class:`cVQVAE_MultiTask`.
    """

    def __init__(self, in_ch=4, stem_ch=128, gru_dim=128, vq_dim=64, num_embeddings=2048, depth=3, **kwargs):
        super().__init__()
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(in_ch, stem_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(stem_ch), nn.SiLU(),
        )
        enc_layers = []
        for _ in range(depth):
            enc_layers += [EffBlock(stem_ch), nn.MaxPool1d(2)]
        self.cnn_blocks  = nn.Sequential(*enc_layers)
        self.encoder_gru = nn.GRU(stem_ch, gru_dim, batch_first=True, bidirectional=True)
        self.pre_vq_conv = nn.Conv1d(gru_dim * 2, vq_dim, kernel_size=1)
        self.vq_layer    = EMAVectorQuantizer(num_embeddings, vq_dim, commitment_cost=0.25)
        self.film_generator = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, vq_dim * 2)
        )
        self.decoder_cond_proj = nn.Conv1d(vq_dim, gru_dim * 2, kernel_size=1)
        self.decoder_gru       = nn.GRU(gru_dim * 2, gru_dim, batch_first=True, bidirectional=True)
        dec_layers  = []
        in_ch_dec   = gru_dim * 2
        for _ in range(depth):
            dec_layers += [
                nn.ConvTranspose1d(in_ch_dec, stem_ch, kernel_size=4, stride=2, padding=1),
                EffBlock(stem_ch),
            ]
            in_ch_dec = stem_ch
        self.decoder_blocks = nn.Sequential(*dec_layers)
        self.decoder_out    = nn.Conv1d(stem_ch, in_ch * 2, kernel_size=5, padding=2)

    def encode_strand(self, x):
        """Encode a sequence to the pre-VQ latent representation."""
        h = self.cnn_stem(x)
        h = self.cnn_blocks(h)
        h = h.permute(0, 2, 1)
        h, _ = self.encoder_gru(h)
        h = h.permute(0, 2, 1)
        return self.pre_vq_conv(h)

    def forward(self, x, y_dev=None, y_hk=None, tau=1.0):
        """FiLM-conditioned encode-quantize-decode pass.

        Returns
        -------
        logits_8ch : ``[B, 8, L]``
        (fwd_gumbel, rc_gumbel) : each ``[B, 4, L]``
        vq_loss : scalar
        """
        L = x.size(2)
        B = x.size(0)

        z = self.encode_strand(x)
        quantized, vq_loss, _ = self.vq_layer(z)

        if y_dev is not None and y_hk is not None:
            cond = torch.cat([
                (y_dev * 10.0).round().div(10.0).view(-1, 1),
                (y_hk  * 10.0).round().div(10.0).view(-1, 1),
            ], dim=1).float()
        else:
            cond = torch.zeros(B, 2, device=x.device)

        if self.training and torch.rand(1).item() < 0.20:
            cond = torch.zeros_like(cond)

        film   = self.film_generator(cond)
        vq_dim = quantized.size(1)
        gamma  = film[:, :vq_dim].unsqueeze(2)
        beta   = film[:, vq_dim:].unsqueeze(2)
        cq     = (1.0 + gamma) * quantized + beta

        d = self.decoder_cond_proj(cq)
        d = d.permute(0, 2, 1)
        d, _ = self.decoder_gru(d)
        d = d.permute(0, 2, 1)
        d = self.decoder_blocks(d)
        logits = self.decoder_out(d)
        if logits.size(2) != L:
            logits = F.interpolate(logits, size=L, mode="linear", align_corners=False)

        fwd_g = F.gumbel_softmax(logits[:, :4], tau=tau, hard=True, dim=1)
        rc_g  = F.gumbel_softmax(logits[:, 4:], tau=tau, hard=True, dim=1)
        return logits, (fwd_g, rc_g), vq_loss

