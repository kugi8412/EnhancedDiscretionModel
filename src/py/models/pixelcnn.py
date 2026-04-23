import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model


class MaskedConv1d(nn.Conv1d):
    """Causal (masked) 1D convolution — each position can only see previous positions."""

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        # Force odd kernel for symmetric padding calculation
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        # Build causal mask: zero out future positions
        mask = torch.ones(out_channels, in_channels, kernel_size)
        center = kernel_size // 2
        mask[:, :, center + 1:] = 0  # block future
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class GatedResBlock(nn.Module):
    """Gated residual block with two masked convolutions."""

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.conv1 = MaskedConv1d(channels, channels * 2, kernel_size)
        self.conv2 = MaskedConv1d(channels, channels, kernel_size)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        h = self.norm1(x)
        h = self.conv1(h)
        # Gated activation: tanh(h1) * sigmoid(h2)
        h1, h2 = h.chunk(2, dim=1)
        h = torch.tanh(h1) * torch.sigmoid(h2)
        h = self.norm2(h)
        h = self.conv2(h)
        return x + h


@register_model("DNA_PixelCNN")
class DNA_PixelCNN(nn.Module):
    """
    PixelCNN adapted for autoregressive DNA sequence modeling.

    Input:  one-hot encoded DNA [B, 4, L]
    Output: logits [B, 4, L] (categorical distribution over A, C, G, T at each position)

    Trained with cross-entropy loss for next-nucleotide prediction.
    Can be used for unconditional sequence generation.
    """

    def __init__(self, in_ch=4, hidden_ch=128, n_layers=8, kernel_size=5, **kwargs):
        super().__init__()

        # Input projection
        self.input_conv = MaskedConv1d(in_ch, hidden_ch, kernel_size=7)
        self.input_norm = nn.BatchNorm1d(hidden_ch)

        # Gated residual stack
        self.res_blocks = nn.ModuleList([
            GatedResBlock(hidden_ch, kernel_size=kernel_size)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_ch, in_ch, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, 4, L] one-hot encoded DNA
        Returns:
            logits: [B, 4, L] per-position nucleotide logits
        """
        h = self.input_conv(x)
        h = self.input_norm(h)

        for block in self.res_blocks:
            h = block(h)

        logits = self.output_conv(h)
        return logits

    @torch.no_grad()
    def generate(self, num_samples, seq_len=249, device='cpu', temperature=1.0):
        """Autoregressively generate DNA sequences."""
        self.eval()
        x = torch.zeros(num_samples, 4, seq_len, device=device)

        for pos in range(seq_len):
            logits = self.forward(x)
            logits_pos = logits[:, :, pos] / temperature
            probs = F.softmax(logits_pos, dim=1)
            sampled = torch.multinomial(probs, 1).squeeze(1)  # [B]
            x[:, :, pos] = F.one_hot(sampled, num_classes=4).float()

        return x

    def get_features(self, x):
        """Extract feature representation (last hidden state before output)."""
        h = self.input_conv(x)
        h = self.input_norm(h)
        for block in self.res_blocks:
            h = block(h)
        return h.mean(dim=2)  # Global average pooling -> [B, hidden_ch]


@register_model("DNA_PixelCNN_Conditioned")
class DNA_PixelCNN_Conditioned(nn.Module):
    """PixelCNN conditioned on a cVQVAE quantized latent tensor.

    Parameters
    ----------
    in_ch : int
        Input channels (4 for one-hot DNA).
    hidden_ch : int
        Width of each residual block.
    n_layers : int
        Number of GatedResBlocks.
    kernel_size : int
        Masked convolution kernel size (must be odd).
    vq_dim : int
        Dimension of the conditioning VQ latent vectors.
    """

    def __init__(self, in_ch=4, hidden_ch=128, n_layers=8, kernel_size=5, vq_dim=64, **kwargs):
        super().__init__()

        self.input_conv = MaskedConv1d(in_ch, hidden_ch, kernel_size=7)
        self.input_norm = nn.BatchNorm1d(hidden_ch)

        # Project conditioned latent to hidden width and add it to every position
        self.cond_proj = nn.Sequential(
            nn.Conv1d(vq_dim, hidden_ch, kernel_size=1),
            nn.SiLU(),
        )

        self.res_blocks = nn.ModuleList([
            GatedResBlock(hidden_ch, kernel_size=kernel_size)
            for _ in range(n_layers)
        ])

        self.output_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_ch, in_ch, kernel_size=1),
        )

    def forward(self, x, latent=None):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, 4, L)``
            Input one-hot DNA.
        latent : torch.Tensor or None, shape ``(B, vq_dim, L')``
            Quantized VQ latent from cVQVAE.  Interpolated to sequence length
            when provided.

        Returns
        -------
        logits : torch.Tensor, shape ``(B, 4, L)``
        """
        h = self.input_conv(x)
        h = self.input_norm(h)

        if latent is not None:
            cond = self.cond_proj(latent)
            if cond.size(2) != h.size(2):
                cond = F.interpolate(cond, size=h.size(2), mode="nearest")
            h = h + cond

        for block in self.res_blocks:
            h = block(h)

        return self.output_conv(h)

    @torch.no_grad()
    def generate(self, num_samples, seq_len=249, latent=None, device="cpu", temperature=1.0):
        """Autoregressively generate sequences, optionally conditioned on *latent*."""
        self.eval()
        x = torch.zeros(num_samples, 4, seq_len, device=device)

        for pos in range(seq_len):
            logits = self.forward(x, latent=latent)
            logits_pos = logits[:, :, pos] / temperature
            probs   = F.softmax(logits_pos, dim=1)
            sampled = torch.multinomial(probs, 1).squeeze(1)
            x[:, :, pos] = F.one_hot(sampled, num_classes=4).float()

        return x

    def get_features(self, x, latent=None):
        """Return pooled hidden representation ``[B, hidden_ch]`` for SAE input."""
        h = self.input_conv(x)
        h = self.input_norm(h)
        if latent is not None:
            cond = self.cond_proj(latent)
            if cond.size(2) != h.size(2):
                cond = F.interpolate(cond, size=h.size(2), mode="nearest")
            h = h + cond
        for block in self.res_blocks:
            h = block(h)
        return h.mean(dim=2)
