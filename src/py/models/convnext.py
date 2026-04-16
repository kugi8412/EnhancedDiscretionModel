import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .registry import register_model


def initialize_weights(m):
    """Nowoczesna inicjalizacja wag dla ConvNeXt."""
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class LayerNorm1d(nn.Module):
    """
    LayerNorm dostosowany do danych 1D [Batch, Channels, Length].
    ConvNeXt (w przeciwieństwie do Transformerów) woli normalizować kanały, a nie pozycję.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # x kształt: [B, C, L]
        # Normalizacja wzdłuż wymiaru C
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(1) * x + self.bias.unsqueeze(1)
        return x

class ConvNeXtBlock(nn.Module):

    def __init__(self, dim, expansion=4, kernel_size=11, dropout=0.1):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, 
                                padding=kernel_size//2, groups=dim)

        self.norm = LayerNorm1d(dim)

        inner_dim = dim * expansion
        self.pwconv1 = nn.Conv1d(dim, inner_dim, kernel_size=1) # Punktowa konwolucja (expansion)
        self.act = nn.GELU() # Nowoczesna aktywacja
        self.pwconv2 = nn.Conv1d(inner_dim, dim, kernel_size=1) # Punktowa konwolucja (reduction)
        
        # 4. DropPath (Stochastic Depth) - dla głębokich sieci, tu użyjemy standardowego Dropout
        self.drop = nn.Dropout(dropout)
        
        # 5. Layer Scale (opcjonalnie dla stabilności bardzo głębokich sieci, tu pominiemy dla prostoty)

    def forward(self, x):
        # Zachowujemy identity do połączenia rezydualnego
        identity = x
        
        # Ścieżka przetwarzania
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.drop(out)
        
        # Połączenie rezydualne: identity + f(x)
        return identity + out

@register_model("ConvNeXt_DNA")
class ConvNeXt_DNA(nn.Module):
    def __init__(self, in_ch=4, stem_ch=96, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 kernel_size=11, dropout=0.1, **kwargs):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, stem_ch, kernel_size=7, padding=3, bias=False),
            LayerNorm1d(stem_ch),
            nn.GELU()
        )

        self.stages = nn.ModuleList()
        current_dim = stem_ch
        
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], kernel_size=kernel_size, dropout=dropout) for _ in range(depths[i])]
            )
            
            if i < len(depths) - 1:
                downsample = nn.Sequential(
                    LayerNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2) 
                )
            else:
                downsample = nn.Identity()

            if i == 0 and stem_ch != dims[0]:
                self.stages.append(nn.Sequential(
                    nn.Conv1d(stem_ch, dims[0], kernel_size=1),
                    stage,
                    downsample
                ))
            else:
                self.stages.append(nn.Sequential(stage, downsample))

        self.norm = LayerNorm1d(dims[-1]) 
        self.head = nn.Sequential(
            nn.Linear(dims[-1], 2) # Dev, Hk
        )
        
        self.apply(initialize_weights)

    def forward(self, x):
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
        out_combined = out_fwd + out_rc 

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
