import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .registry import register_model

def initialize_weights(m):
    """Modern weight initialization for ConvNeXt (truncated normal)."""
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        # Truncated normal is better for deep networks
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class LayerNorm1d(nn.Module):
    """
    Channel-wise LayerNorm for 1D data [Batch, Channels, Length].
    Unlike Transformers, ConvNeXt normalizes across channels rather than positions.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # x shape: [B, C, L]
        # Normalize along the channel dimension C
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(1) * x + self.bias.unsqueeze(1)
        return x

class ConvNeXtBlock(nn.Module):
    """
    Core ConvNeXt building block: large-kernel depthwise conv + inverted bottleneck.
    """
    def __init__(self, dim, expansion=4, kernel_size=11, dropout=0.1):
        super().__init__()
        # 1. Depthwise conv: large kernel (e.g. 11), groups=dim -> "attention-like"
        # Processes spatial information within each channel.
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, 
                                padding=kernel_size//2, groups=dim)
        
        # 2. LayerNorm1d instead of BatchNorm (modern approach)
        self.norm = LayerNorm1d(dim)
        
        # 3. Inverted bottleneck (1x1 convs acting as linear layers)
        # Processes cross-channel information.
        inner_dim = dim * expansion
        self.pwconv1 = nn.Conv1d(dim, inner_dim, kernel_size=1)  # Pointwise conv (expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(inner_dim, dim, kernel_size=1)  # Pointwise conv (reduction)
        
        # 4. DropPath (stochastic depth) — using standard Dropout here for simplicity
        self.drop = nn.Dropout(dropout)
        
        # 5. Layer Scale omitted for simplicity

    def forward(self, x):
        # Save identity for residual connection
        identity = x
        
        # Processing path
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.drop(out)
        
        # Residual connection: identity + f(x)
        return identity + out

@register_model("ConvNeXt_DNA")
class ConvNeXt_DNA(nn.Module):
    def __init__(self, in_ch=4, stem_ch=96, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 kernel_size=11, dropout=0.1, **kwargs):
        super().__init__()
        
        # 1. STEM: initial convolution (LegNet-inspired to avoid aggressive downsampling on 250bp DNA)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, stem_ch, kernel_size=7, padding=3, bias=False),
            LayerNorm1d(stem_ch),
            nn.GELU()
        )

        # 2. CONVOLUTION BLOCKS (Stages with modern techniques)
        self.stages = nn.ModuleList()
        current_dim = stem_ch
        
        # Build successive ConvNeXt stages
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], kernel_size=kernel_size, dropout=dropout) for _ in range(depths[i])]
            )
            
            # Add downsampling layer between stages (except the last)
            if i < len(depths) - 1:
                downsample = nn.Sequential(
                    LayerNorm1d(dims[i]),
                    # Strided 1x1 conv to change channel count and reduce sequence length
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2) 
                )
            else:
                downsample = nn.Identity()
            
            # Handle channel change from stem_ch to dims[0] if they differ.
            if i == 0 and stem_ch != dims[0]:
                self.stages.append(nn.Sequential(
                    nn.Conv1d(stem_ch, dims[0], kernel_size=1),  # Channel alignment
                    stage,
                    downsample
                ))
            else:
                self.stages.append(nn.Sequential(stage, downsample))

        # 3. GLOBAL AGGREGATION — final normalization before the head
        self.norm = LayerNorm1d(dims[-1]) 
        # Output head: GAP + Linear
        self.head = nn.Sequential(
            nn.Linear(dims[-1], 2)  # Dev, Hk
        )
        
        # Apply weight initialization
        self.apply(initialize_weights)

    def forward(self, x):
        # x: [Batch, 4, L]
        
        # --- 1. REVERSE COMPLEMENT (RC) ---
        # Assumes one-hot encoding: A, C, G, T.
        # Flipping channels (dim=1) swaps A<->T and C<->G.
        x_rc = torch.flip(x, dims=[1, 2])

        # --- 2. STEM & STAGES (Forward strand) ---
        feat_fwd = self.stem(x)
        for stage in self.stages:
            feat_fwd = stage(feat_fwd)
            
        # --- 3. STEM & STAGES (RC strand — shared weights) ---
        feat_rc = self.stem(x_rc)
        for stage in self.stages:
            feat_rc = stage(feat_rc)

        # --- 4. GLOBAL AGGREGATION ---
        feat_fwd = self.norm(feat_fwd)
        feat_rc = self.norm(feat_rc)
        
        # Global Average Pooling 1D: [B, C, L] -> [B, C]
        out_fwd = feat_fwd.mean(dim=2) 
        out_rc = feat_rc.mean(dim=2)   
        
        # --- 5. SYMMETRIC FUSION (RC equivariance) ---
        # Addition instead of cat() enforces strand-direction invariance.
        out_combined = out_fwd + out_rc 
        
        # --- 6. PREDICTION ---
        out = self.head(out_combined)
        
        return out[:, 0:1], out[:, 1:2]

    def get_features(self, x):
        """Extract penultimate feature vector for cross-model comparison."""
        x_rc = torch.flip(x, dims=[1, 2])
        feat_fwd = self.stem(x)
        for stage in self.stages:
            feat_fwd = stage(feat_fwd)
        feat_rc = self.stem(x_rc)
        for stage in self.stages:
            feat_rc = stage(feat_rc)
        feat_fwd = self.norm(feat_fwd)
        feat_rc = self.norm(feat_rc)
        out_fwd = feat_fwd.mean(dim=2)
        out_rc = feat_rc.mean(dim=2)
        return out_fwd + out_rc
