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


# ===========================================================================
# NucleotideCNN — Per-position nucleotide probability predictor
# ===========================================================================

class BackboneFeatureExtractor(nn.Module):
    """Extracts per-position features from a backbone model via hooks.

    Captures activations from intermediate layers and interpolates them to
    full sequence length, producing a rich per-position feature map.

    Parameters
    ----------
    backbone : nn.Module
        A pretrained model (e.g. DeepSTARR, LegNet, DilatedConvNeXt).
    layer_names : list[str] or None
        Which layers to tap. If None, auto-selects Conv1d/BatchNorm layers.
    max_layers : int
        Maximum number of layers to capture (to limit memory).
    freeze : bool
        If True, backbone parameters are frozen (default). If False,
        backbone is trainable (fine-tuning mode).
    """

    def __init__(self, backbone, layer_names=None, max_layers=8, freeze=True):
        super().__init__()
        self.backbone = backbone
        self._hooks = []
        self._activations = {}
        self.frozen = freeze

        # Freeze/unfreeze backbone
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

        # Select layers to capture
        if layer_names is None:
            layer_names = self._auto_select(backbone, max_layers)
        self.layer_names = layer_names

        # Register hooks
        name_to_mod = dict(backbone.named_modules())
        for name in self.layer_names:
            if name in name_to_mod:
                hook = name_to_mod[name].register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    @staticmethod
    def _auto_select(model, max_layers):
        """Auto-select layers that output 3D tensors (B, C, L)."""
        candidates = []
        for name, module in model.named_modules():
            if name == '':
                continue
            if isinstance(module, (nn.Conv1d, nn.BatchNorm1d)):
                candidates.append(name)
        # Subsample evenly
        if len(candidates) > max_layers:
            step = max(1, len(candidates) // max_layers)
            candidates = candidates[::step][:max_layers]
        return candidates

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self._activations[name] = output
        return hook_fn

    @property
    def output_channels(self):
        """Total channels from all captured layers (computed after first forward)."""
        return self._total_channels

    def forward(self, x):
        """Extract and concatenate per-position features from backbone.

        Returns
        -------
        features : Tensor of shape (B, total_channels, L)
            Per-position features interpolated to input length.
        """
        self._activations.clear()
        L = x.size(2)

        def _run_backbone():
            try:
                self.backbone(x)
            except Exception:
                pass  # Some models may error after hooks capture what we need

        if self.frozen:
            with torch.no_grad():
                _run_backbone()
        else:
            _run_backbone()

        # Collect and interpolate all activations to full length L
        feature_maps = []
        for name in self.layer_names:
            if name in self._activations:
                act = self._activations[name]
                if act.dim() == 3:  # (B, C, L')
                    if act.size(2) != L:
                        act = F.interpolate(act, size=L, mode='linear', align_corners=False)
                    if self.frozen:
                        feature_maps.append(act.detach())
                    else:
                        feature_maps.append(act)
                elif act.dim() == 2:  # (B, C) — expand to (B, C, L)
                    if self.frozen:
                        feature_maps.append(act.detach().unsqueeze(2).expand(-1, -1, L))
                    else:
                        feature_maps.append(act.unsqueeze(2).expand(-1, -1, L))

        if not feature_maps:
            # Fallback: use raw input
            feature_maps = [x]

        out = torch.cat(feature_maps, dim=1)  # (B, total_ch, L)
        self._total_channels = out.size(1)
        return out


class NucleotideHead(nn.Module):
    """Per-position nucleotide probability prediction head.

    Uses masked (causal) convolutions so that prediction at position i
    depends only on features at positions <= i, enabling proper
    autoregressive uncertainty estimation.

    Parameters
    ----------
    in_channels : int
        Input feature channels from backbone extractor.
    hidden_ch : int
        Hidden width of residual blocks.
    n_layers : int
        Number of gated residual blocks.
    kernel_size : int
        Causal convolution kernel.
    dropout : float
        Dropout rate.
    """

    def __init__(self, in_channels, hidden_ch=128, n_layers=6,
                 kernel_size=5, dropout=0.1):
        super().__init__()

        # Projection from backbone features to hidden dim
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_ch, kernel_size=1),
            nn.BatchNorm1d(hidden_ch),
            nn.SiLU(),
        )

        # Causal gated residual blocks
        self.blocks = nn.ModuleList([
            GatedResBlock(hidden_ch, kernel_size=kernel_size)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Output: 4 logits per position (A, C, G, T probabilities)
        self.output_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_ch, 4, kernel_size=1),
        )

    def forward(self, features):
        """
        Parameters
        ----------
        features : Tensor (B, in_channels, L)
            Per-position features from frozen backbone.

        Returns
        -------
        logits : Tensor (B, 4, L)
            Per-position nucleotide logits.
        """
        h = self.input_proj(features)
        for block in self.blocks:
            h = block(h)
        h = self.dropout(h)
        return self.output_conv(h)


@register_model("NucleotideCNN")
class NucleotideCNN(nn.Module):
    """Per-position nucleotide probability predictor with configurable backbone.

    Uses a pretrained model as a feature extractor (frozen or trainable) and
    trains a causal PixelCNN-like head on top to predict p(nucleotide | position).
    This provides:
    - Natural per-position uncertainty (entropy of predicted distribution)
    - MC Dropout uncertainty (variance across stochastic forward passes)
    - MSE-based uncertainty estimation
    - Single nucleotide variant effect (compare p before/after mutation)
    - Sequence quality score (average log-likelihood)

    Architecture:
        Input DNA → [Backbone (frozen/unfrozen)] → per-position features
        → [Trainable NucleotideHead] → p(A,C,G,T) at each position

    Training:
        Cross-entropy loss: -log p(true_nt | context)

    Parameters
    ----------
    backbone_config : str
        Path to YAML config for the backbone model.
    backbone_weights : str
        Path to trained weights for the backbone model.
    hidden_ch : int
        NucleotideHead hidden width.
    n_layers : int
        NucleotideHead depth (gated residual blocks).
    kernel_size : int
        Causal convolution kernel size.
    dropout : float
        Dropout in the head.
    layer_names : list[str] or None
        Backbone layers to extract features from.
    max_layers : int
        Max backbone layers to capture.
    freeze_backbone : bool
        If True, backbone is frozen (default). If False, backbone is fine-tuned.
    mc_samples : int
        Default number of MC Dropout forward passes for uncertainty estimation.
    """

    def __init__(self, backbone_config=None, backbone_weights=None,
                 hidden_ch=128, n_layers=6, kernel_size=5, dropout=0.1,
                 layer_names=None, max_layers=8, seq_len=249,
                 freeze_backbone=True, mc_samples=20, **kwargs):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_ch = hidden_ch
        self.freeze_backbone = freeze_backbone
        self.mc_samples = mc_samples
        self._backbone_config = backbone_config
        self._backbone_weights = backbone_weights

        # Build backbone
        if backbone_config is not None and backbone_weights is not None:
            from utils import load_config
            from .registry import build_model as _build_model
            bcfg = load_config(backbone_config)
            backbone = _build_model(bcfg)
            state = torch.load(backbone_weights, map_location='cpu', weights_only=True)
            backbone.load_state_dict(state)
            if freeze_backbone:
                backbone.eval()
        else:
            backbone = nn.Identity()

        self.extractor = BackboneFeatureExtractor(
            backbone, layer_names=layer_names, max_layers=max_layers,
            freeze=freeze_backbone,
        )

        # Run a dummy forward to determine total channels
        with torch.no_grad():
            dummy = torch.randn(1, 4, seq_len)
            dummy_feat = self.extractor(dummy)
            feat_channels = dummy_feat.size(1)

        # Trainable per-position head
        self.head = NucleotideHead(
            in_channels=feat_channels,
            hidden_ch=hidden_ch,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def train(self, mode=True):
        """Override train() to keep frozen backbone in eval mode."""
        super().train(mode)
        if self.freeze_backbone:
            self.extractor.backbone.eval()
        return self

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : Tensor (B, 4, L)
            One-hot encoded input DNA.

        Returns
        -------
        logits : Tensor (B, 4, L)
            Per-position nucleotide logits (apply softmax for probabilities).
        """
        features = self.extractor(x)  # (B, feat_ch, L)
        logits = self.head(features)   # (B, 4, L)
        return logits

    def get_probabilities(self, x):
        """Get per-position nucleotide probabilities.

        Returns
        -------
        probs : Tensor (B, 4, L) — probabilities summing to 1 per position
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_uncertainty(self, x):
        """Get per-position entropy (uncertainty in bits).

        Returns
        -------
        entropy : Tensor (B, L)
            Shannon entropy at each position. Max = 2 bits (uniform).
            Low entropy = model is confident about the nucleotide.
            High entropy = position is uncertain/variable.
        """
        probs = self.get_probabilities(x)
        log_probs = torch.log2(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=1)  # (B, L)
        return entropy

    def get_snv_effect(self, x, pos, alt_nt):
        """Compute single nucleotide variant effect score.

        Compares log-likelihood of the reference vs alternative nucleotide
        at a given position.

        Parameters
        ----------
        x : Tensor (B, 4, L) — reference sequence
        pos : int — position to mutate (0-indexed)
        alt_nt : int — alternative nucleotide index (0=A, 1=C, 2=G, 3=T)

        Returns
        -------
        delta_ll : Tensor (B,) — log-likelihood difference (alt - ref)
        """
        probs = self.get_probabilities(x)  # (B, 4, L)
        ref_nt = x[:, :, pos].argmax(dim=1)  # (B,) reference nucleotide
        ref_ll = torch.log(probs[range(x.size(0)), ref_nt, pos] + 1e-10)
        alt_ll = torch.log(probs[:, alt_nt, pos] + 1e-10)
        return alt_ll - ref_ll

    def sequence_log_likelihood(self, x):
        """Compute total log-likelihood of a sequence under the model.

        Returns
        -------
        ll : Tensor (B,) — log p(sequence) = sum of log p(nt_i | context)
        """
        probs = self.get_probabilities(x)  # (B, 4, L)
        true_nt = x.argmax(dim=1)  # (B, L)
        # Gather prob of true nucleotide at each position
        true_probs = probs.gather(1, true_nt.unsqueeze(1)).squeeze(1)  # (B, L)
        return torch.log(true_probs + 1e-10).sum(dim=1)  # (B,)

    def get_features(self, x):
        """Extract pooled features for SAE analysis or MLP predictor."""
        features = self.extractor(x)
        return features.mean(dim=2)  # (B, feat_ch)

    def get_position_features(self, x):
        """Extract per-position features (before head) for latent analysis."""
        return self.extractor(x)  # (B, feat_ch, L)

    # ------------------------------------------------------------------
    # MC Dropout uncertainty estimation
    # ------------------------------------------------------------------

    def mc_predict(self, x, n_samples=None):
        """Monte Carlo Dropout: run n_samples forward passes with dropout active.

        Parameters
        ----------
        x : Tensor (B, 4, L)
        n_samples : int or None
            Number of stochastic passes. Uses self.mc_samples if None.

        Returns
        -------
        mean_probs : Tensor (B, 4, L) — mean predicted probabilities
        variance : Tensor (B, 4, L) — per-nucleotide variance across samples
        entropy : Tensor (B, L) — mean entropy across samples
        """
        if n_samples is None:
            n_samples = self.mc_samples

        was_training = self.training
        # Enable dropout but keep backbone frozen if needed
        self.head.train()
        if self.freeze_backbone:
            self.extractor.backbone.eval()

        all_probs = []
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            for _ in range(n_samples):
                features = self.extractor(x)
                logits = self.head(features)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)

        # Restore training state
        self.train(was_training)

        stacked = torch.stack(all_probs, dim=0)  # (S, B, 4, L)
        mean_probs = stacked.mean(dim=0)          # (B, 4, L)
        variance = stacked.var(dim=0)              # (B, 4, L)

        # Mean entropy
        log_probs = torch.log2(mean_probs + 1e-10)
        entropy = -(mean_probs * log_probs).sum(dim=1)  # (B, L)

        return mean_probs, variance, entropy

    def mc_uncertainty(self, x, n_samples=None, reduction='mean'):
        """Compute scalar uncertainty per sequence via MC Dropout.

        Parameters
        ----------
        x : Tensor (B, 4, L)
        n_samples : int
        reduction : str
            'mean' — mean variance across positions and nucleotides
            'max' — max variance across positions
            'sum' — total variance

        Returns
        -------
        uncertainty : Tensor (B,) — scalar uncertainty per sequence
        """
        _, variance, _ = self.mc_predict(x, n_samples)
        # variance is (B, 4, L), reduce to (B,)
        if reduction == 'mean':
            return variance.mean(dim=(1, 2))
        elif reduction == 'max':
            return variance.sum(dim=1).max(dim=1).values
        elif reduction == 'sum':
            return variance.sum(dim=(1, 2))
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    # ------------------------------------------------------------------
    # MSE-based uncertainty estimation
    # ------------------------------------------------------------------

    def mse_uncertainty(self, x):
        """MSE-based uncertainty: how much the model's prediction deviates
        from the actual one-hot input.

        This measures the model's "surprise" at each position — high MSE
        means the model didn't expect the observed nucleotide.

        Parameters
        ----------
        x : Tensor (B, 4, L) — one-hot encoded input

        Returns
        -------
        mse_per_pos : Tensor (B, L) — MSE at each position
        mse_total : Tensor (B,) — total MSE per sequence
        """
        probs = self.get_probabilities(x)  # (B, 4, L)
        mse_per_pos = ((probs - x) ** 2).mean(dim=1)  # (B, L)
        mse_total = mse_per_pos.mean(dim=1)  # (B,)
        return mse_per_pos, mse_total

    def combined_uncertainty(self, x, n_samples=None, mc_weight=0.5):
        """Combined uncertainty score: weighted mix of MC variance + MSE.

        Parameters
        ----------
        x : Tensor (B, 4, L)
        n_samples : int
        mc_weight : float
            Weight for MC dropout component (1-mc_weight for MSE).

        Returns
        -------
        score : Tensor (B,) — combined uncertainty per sequence
        mc_unc : Tensor (B,) — MC dropout component
        mse_unc : Tensor (B,) — MSE component
        """
        mc_unc = self.mc_uncertainty(x, n_samples, reduction='mean')
        _, mse_unc = self.mse_uncertainty(x)
        score = mc_weight * mc_unc + (1 - mc_weight) * mse_unc
        return score, mc_unc, mse_unc

