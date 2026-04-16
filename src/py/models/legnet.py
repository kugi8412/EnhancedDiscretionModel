import math
import torch
from torch import nn
from tltorch import TRL
from collections import OrderedDict
import torch.nn.functional as F 
from .registry import register_model

class Bilinear(nn.Module):
    def __init__(self, n: int, out=None, rank=0.05, bias=False):        
        super().__init__()
        if out is None:
            out = (n, )
        self.trl = TRL((n, n), out, bias=bias, rank=rank)
        self.trl.weight = self.trl.weight.normal_(std=0.00075)
    
    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        return self.trl(x @ x.transpose(-1, -2))

class Concater(nn.Module):
    def __init__(self, module: nn.Module, dim=-1):        
        super().__init__()
        self.mod = module
        self.dim = dim
    
    def forward(self, x):
        return torch.concat((x, self.mod(x)), dim=self.dim)

class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, int(inp // reduction)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction), int(inp // reduction)),
                Concater(Bilinear(int(inp // reduction), int(inp // reduction // 2), rank=0.5, bias=True)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction) +  int(inp // reduction // 2), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y
    
@register_model("LegNet")
class LegNet(nn.Module):
    __constants__ = ('resize_factor')
    
    def __init__(self, 
                 seq_len=249, # Zostawione dla spójności API z configiem
                 in_channels=4, # Nasze kodowanie One-Hot
                 block_sizes=[256, 256, 128, 128, 64, 64, 32, 32], 
                 ks=5, 
                 resize_factor=4, 
                 activation=nn.SiLU,
                 filter_per_group=2,
                 se_reduction=4,
                 final_ch=18,
                 bn_momentum=0.1,
                 **kwargs):        
        super().__init__()
        self.block_sizes = block_sizes
        self.resize_factor = resize_factor
        self.se_reduction = se_reduction
        self.seqsize = seq_len
        self.final_ch = final_ch
        self.bn_momentum = bn_momentum
        seqextblocks = OrderedDict()
        
        block = nn.Sequential(
                       nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=block_sizes[0],
                            kernel_size=ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(block_sizes[0], momentum=self.bn_momentum),
                       activation()
        )
        seqextblocks['blc0'] = block
        
        for ind, (prev_sz, sz) in enumerate(zip(block_sizes[:-1], block_sizes[1:])):
            block = nn.Sequential(
                       nn.Conv1d(
                            in_channels=prev_sz,
                            out_channels=sz * self.resize_factor,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz * self.resize_factor, momentum=self.bn_momentum),
                       activation(), 
                       
                       nn.Conv1d(
                            in_channels=sz * self.resize_factor,
                            out_channels=sz * self.resize_factor,
                            kernel_size=ks,
                            groups=sz * self.resize_factor // filter_per_group,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz * self.resize_factor, momentum=self.bn_momentum),
                       activation(), 
                       SELayer(prev_sz, sz * self.resize_factor, reduction=self.se_reduction),
                       nn.Conv1d(
                            in_channels=sz * self.resize_factor,
                            out_channels=prev_sz,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(prev_sz, momentum=self.bn_momentum),
                       activation(), 
            )
            seqextblocks[f'inv_res_blc{ind}'] = block
            
            block = nn.Sequential(
                       nn.Conv1d(
                            in_channels=2 * prev_sz,
                            out_channels=sz,
                            kernel_size=ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz, momentum=self.bn_momentum),
                       activation(),
            )
            seqextblocks[f'resize_blc{ind}'] = block

        self.seqextractor = nn.ModuleDict(seqextblocks)

        self.mapper = nn.Sequential(
                       nn.Conv1d(
                            in_channels=block_sizes[-1],
                            out_channels=self.final_ch,
                            kernel_size=1,
                            padding='same',
                       ),
                       activation()
        )
        
        # --- NOWA GŁOWICA DO REGRESJI (Zamiast 18 dyskretnych binów) ---
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(self.final_ch, 2)
        
    def feature_extractor(self, x):
        x = self.seqextractor['blc0'](x)
        
        for i in range(len(self.block_sizes) - 1):
            x = torch.cat([x, self.seqextractor[f'inv_res_blc{i}'](x)], dim=1)
            x = self.seqextractor[f'resize_blc{i}'](x)
        return x 

    def forward(self, x):    
        f = self.feature_extractor(x)
        x = self.mapper(f)
        
        # [Batch, final_ch, L] -> [Batch, final_ch]
        x = self.global_pool(x).squeeze(2)
        
        # Output [Batch, 2]
        out = self.fc_out(x)
        
        # Zgodność z API Twojego train.py
        return out[:, 0:1], out[:, 1:2]

    def get_features(self, x):
        """Extract penultimate feature vector for cross-model comparison."""
        f = self.feature_extractor(x)
        x = self.mapper(f)
        x = self.global_pool(x).squeeze(2)
        return x


def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

class SENewLayer(nn.Module):
    def __init__(self, inp, reduction=4):
        super(SENewLayer, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(inp, int(inp // reduction)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction), inp),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y

class EffBlock(nn.Module):
    def __init__(self, in_ch, ks, resize_factor, activation, out_ch=None, se_reduction=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.resize_factor = resize_factor
        self.se_reduction = resize_factor if se_reduction is None else se_reduction
        self.ks = ks
        self.inner_dim = self.in_ch * self.resize_factor
        
        self.block = nn.Sequential(
                       nn.Conv1d(
                            in_channels=self.in_ch,
                            out_channels=self.inner_dim,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.inner_dim),
                       activation(),
                       
                       nn.Conv1d(
                            in_channels=self.inner_dim,
                            out_channels=self.inner_dim,
                            kernel_size=ks,
                            groups=self.inner_dim,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.inner_dim),
                       activation(),
                       SENewLayer(self.inner_dim, reduction=self.se_reduction),
                       nn.Conv1d(
                            in_channels=self.inner_dim,
                            out_channels=self.in_ch,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.in_ch),
                       activation(),
        )
    
    def forward(self, x):
        return self.block(x)
    
class LocalBlock(nn.Module):
    def __init__(self, in_ch, ks, activation, out_ch=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.ks = ks
        
        self.block = nn.Sequential(
                       nn.Conv1d(
                            in_channels=self.in_ch,
                            out_channels=self.out_ch,
                            kernel_size=self.ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.out_ch),
                       activation()
        )
        
    def forward(self, x):
        return self.block(x)


class ResidualConcat(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return torch.concat([self.fn(x, **kwargs), x], dim=1)


class MapperBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.SiLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1),
        )
        
    def forward(self, x):
        return self.block(x) 


@register_model("LegNetV2")
class LegNetV2(nn.Module):
    def __init__(self, 
                 in_ch=4,
                 stem_ch=256,
                 stem_ks=5, 
                 ef_ks=5,
                 ef_block_sizes=[256, 128, 128, 64, 64, 32, 32],
                 pool_sizes=[1, 2, 1, 2, 1, 2, 1],
                 resize_factor=4,
                 seq_len=249,
                 **kwargs):
        super().__init__()
        assert len(pool_sizes) == len(ef_block_sizes)
        
        activation = nn.SiLU
        self.in_ch = in_ch
        self.stem = LocalBlock(in_ch=in_ch, 
                               out_ch=stem_ch,
                               ks=stem_ks,
                               activation=activation)
        
        blocks = []
        in_ch = stem_ch
        out_ch = stem_ch
        for pool_sz, out_ch in zip(pool_sizes, ef_block_sizes):
            blc = nn.Sequential(
                ResidualConcat(
                    EffBlock(
                        in_ch=in_ch, 
                        out_ch=in_ch,
                        ks=ef_ks,
                        resize_factor=resize_factor,
                        activation=activation)
                ),
                LocalBlock(in_ch=in_ch * 2,
                           out_ch=out_ch,
                           ks=ef_ks,
                           activation=activation),
                nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity()
            )
            in_ch = out_ch
            blocks.append(blc)
            
        self.main = nn.Sequential(*blocks)
        
        self.mapper = MapperBlock(in_features=out_ch, 
                                  out_features=out_ch * 2,
                                  activation=activation)
                                  
        self.head = nn.Sequential(nn.Linear(out_ch * 2, out_ch * 2),
                                   nn.BatchNorm1d(out_ch * 2),
                                   activation(),
                                   nn.Linear(out_ch * 2, 2))
            
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.stem(x)
        x = self.main(x)
        x = self.mapper(x)
        
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        x = self.head(x)
        
        return x[:, 0:1], x[:, 1:2]

    def get_features(self, x):
        """Extract penultimate feature vector for cross-model comparison."""
        x = self.stem(x)
        x = self.main(x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        return x

