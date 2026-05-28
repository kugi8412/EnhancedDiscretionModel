import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .registry import register_model


class TransformerBlock(nn.Module):
    """Transformer encoder block adapted from DeepEPI.

    Multi-head self-attention + feed-forward network with residual
    connections and layer normalization.
    """

    def __init__(self, d_model, num_heads=8, dff=256, dropout=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x


class AttentionPool(nn.Module):
    """Learnable attention pooling (from DeepEPI's AttLayer).

    Computes attention-weighted sum over sequence positions to produce
    a fixed-size representation.
    """

    def __init__(self, d_model, attention_dim=128):
        super().__init__()
        self.W = nn.Linear(d_model, attention_dim, bias=True)
        self.u = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        scores = torch.tanh(self.W(x))          # (batch, seq_len, attention_dim)
        scores = self.u(scores).squeeze(-1)     # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)     # (batch, seq_len)
        # Weighted sum
        out = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, d_model)
        return out


# ============================================================================
# DeepSTARR promoters used in the STARR-seq assay (Drosophila melanogaster):
#
# 1. DSCP (Drosophila Synthetic Core Promoter) — used for Developmental enhancers
#    ~120bp synthetic promoter containing Inr + MTE + DPE motifs
#
# 2. RpS12 (Ribosomal Protein S12 promoter) — used for Housekeeping enhancers
#    ~120bp housekeeping gene promoter
#
# In DeepSTARR STARR-seq, ALL enhancer candidates are tested against BOTH
# promoters simultaneously, yielding Dev and Hk activity scores.
# Since the promoters are constant across all samples, we encode them as
# learnable fixed embeddings that the model can specialize per output.
# ============================================================================

# DSCP sequence (Drosophila Synthetic Core Promoter, ~120bp)
DSCP_SEQUENCE = (
    "GAGCTTCTTGTTCTTCTTGCAGATATCAGAAATGAACAGCTTGAATCGCGACCGTGTG"
    "ATTACAGACACACACACAGCGCATATAAATGTCAGTATCTTGTCAGCGATCGGCGGATC"
)

# RpS12 promoter sequence (Drosophila housekeeping, ~120bp)
RPS12_SEQUENCE = (
    "CAAGCAAGCAAAGTAAACAGAAACAAACAATCAAACAAAGAAATAAATTGGCAGACCCA"
    "GCGAGCGAGCGACTTCTTCGTCGTTCGTCGTCCTCTTCTTCACAGCTTCTTCAACTCG"
)


def _encode_dna(seq):
    """One-hot encode a DNA sequence string to tensor (4, L)."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = torch.zeros(4, len(seq))
    for i, nt in enumerate(seq.upper()):
        if nt in mapping:
            encoded[mapping[nt], i] = 1.0
    return encoded


@register_model("DeepEPI")
class DeepEPI(nn.Module):
    """DeepEPI-inspired enhancer-promoter interaction model for DeepSTARR.

    Key insight: since there are only 2 fixed promoters (DSCP, RpS12) that
    never change across samples, we encode the enhancer ONCE with a shared
    CNN+Transformer encoder, then use learnable promoter embeddings to
    condition the output via FiLM modulation (Feature-wise Linear Modulation).

    Architecture:
    1. Shared enhancer encoder: CNN (3bp resolution) → Transformer → AttPool
    2. Two learnable promoter embeddings (one per promoter/output)
    3. FiLM: promoter embedding generates (gamma, beta) to scale/shift
       the shared enhancer representation differently for Dev vs Hk
    4. Separate output heads

    This is more efficient than running the enhancer twice and avoids the
    redundant CNN on constant promoter sequences.

    Parameters
    ----------
    seq_len : int
        Enhancer input sequence length (default 249).
    conv_filters : int
        Number of CNN filters / transformer d_model (default 128).
    conv_kernel : int
        CNN kernel size in bp (default 15).
    pool_size : int
        CNN pooling kernel — 3bp resolution (default 3).
    num_heads : int
        Number of attention heads (default 8).
    dff : int
        Feed-forward hidden dimension in transformer (default 256).
    dropout : float
        Dropout rate (default 0.3).
    attention_dim : int
        Attention pooling hidden dimension (default 128).
    dense_dim : int
        Dense layer width before output (default 256).
    promoter_dim : int
        Learnable promoter embedding dimension (default 64).
    """

    def __init__(self, seq_len=249, conv_filters=128, conv_kernel=15,
                 pool_size=3, num_heads=8, dff=256, dropout=0.3,
                 attention_dim=128, dense_dim=256, promoter_dim=64,
                 **kwargs):
        super().__init__()

        d_model = conv_filters

        # --- Shared enhancer encoder (processed ONCE per sample) ---
        self.enhancer_conv = nn.Sequential(
            nn.Conv1d(4, d_model, kernel_size=conv_kernel, padding='same'),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=pool_size, stride=pool_size),  # 3bp resolution
        )

        enh_pooled_len = seq_len // pool_size

        # Positional embedding for enhancer positions
        self.pos_embedding = nn.Parameter(
            torch.randn(1, enh_pooled_len, d_model) * 0.02
        )

        # Transformer on enhancer features
        self.transformer = TransformerBlock(
            d_model=d_model, num_heads=num_heads, dff=dff, dropout=dropout
        )

        # Attention pooling: sequence -> single vector
        self.attention_pool = AttentionPool(d_model, attention_dim)

        # --- Learnable promoter embeddings (only 2 promoters!) ---
        # Instead of encoding fixed DNA through CNN every forward pass,
        # learn the optimal promoter representation from data directly.
        # Initialized from one-hot encoded DSCP/RpS12 projected down.
        self.promoter_dev = nn.Parameter(torch.randn(promoter_dim) * 0.02)
        self.promoter_hk = nn.Parameter(torch.randn(promoter_dim) * 0.02)

        # --- FiLM conditioning: promoter modulates enhancer features ---
        # Each promoter generates (gamma, beta) to scale/shift the shared
        # enhancer representation, specializing it for Dev or Hk context.
        self.film_dev = nn.Linear(promoter_dim, d_model * 2)  # gamma + beta
        self.film_hk = nn.Linear(promoter_dim, d_model * 2)

        # --- Output heads ---
        self.head_dev = nn.Sequential(
            nn.Linear(d_model, dense_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 1),
        )
        self.head_hk = nn.Sequential(
            nn.Linear(d_model, dense_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 1),
        )

    def _encode_enhancer(self, x):
        """Shared enhancer encoding: CNN → Transformer → AttPool."""
        # x: (B, 4, seq_len)
        feat = self.enhancer_conv(x)                # (B, d_model, L_pooled)
        feat = feat.permute(0, 2, 1)                # (B, L_pooled, d_model)
        feat = feat + self.pos_embedding[:, :feat.size(1), :]
        feat = self.transformer(feat)               # (B, L_pooled, d_model)
        feat = self.attention_pool(feat)            # (B, d_model)
        return feat

    def _film_modulate(self, feat, film_params):
        """Apply FiLM: gamma * feat + beta (feature-wise affine transform)."""
        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, d_model)
        return (1 + gamma) * feat + beta

    def forward(self, x):
        # x: (batch, 4, seq_len) — enhancer sequence
        batch_size = x.size(0)

        # Encode enhancer ONCE (shared representation)
        enh_feat = self._encode_enhancer(x)  # (B, d_model)

        # Generate FiLM parameters from promoter embeddings
        film_dev = self.film_dev(self.promoter_dev).unsqueeze(0).expand(batch_size, -1)
        film_hk = self.film_hk(self.promoter_hk).unsqueeze(0).expand(batch_size, -1)

        # Modulate shared features differently for each promoter context
        feat_dev = self._film_modulate(enh_feat, film_dev)
        feat_hk = self._film_modulate(enh_feat, film_hk)

        # Predict
        out_dev = self.head_dev(feat_dev)  # (B, 1)
        out_hk = self.head_hk(feat_hk)    # (B, 1)

        return out_dev, out_hk

    def get_features(self, x):
        """Extract shared enhancer features."""
        return self._encode_enhancer(x)
