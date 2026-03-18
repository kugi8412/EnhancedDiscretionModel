import torch
import torch.nn as nn
from .registry import register_model

class SEBlock(nn.Module):
    def __init__(self, in_channels, r=8):
        super(SEBlock, self).__init__()
        # Squeeze
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Excitation
        self.fc1 = nn.Linear(in_channels, in_channels // r)
        self.fc2 = nn.Linear(in_channels // r, in_channels)
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        # Squeeze: [Batch, Channels, Length] -> [Batch, Channels]
        branch = self.pool(x).view(b, c)
        
        # Excitation
        branch = self.fc1(branch)
        branch = self.gelu(branch)
        branch = self.dropout(self.fc2(branch))
        branch = self.sigmoid(branch).view(b, c, 1) # Przywracamy 3 wymiar do mnożenia
        
        # Scale
        return x * branch


class BottleNeck(nn.Module):
    def __init__(self, in_channels, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        out_channels = filter_num * 4

        # Redukcja kanałów (1x1)
        self.conv1 = nn.Conv1d(in_channels, filter_num, kernel_size=1, stride=1, padding=0, bias=False)
        # Konwolucja przestrzenna (3x3)
        self.conv2 = nn.Conv1d(filter_num, filter_num, kernel_size=3, stride=stride, padding=1, bias=False)
        # Ekspansja kanałów (1x1)
        self.conv3 = nn.Conv1d(filter_num, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.se = SEBlock(out_channels)

        # Skrót (Shortcut) / Downsample - w Kerasie był aplikowany zawsze
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.gelu(out)
        
        out = self.conv2(out)
        out = self.gelu(out)
        
        out = self.conv3(out)
        out = self.se(out)
        
        # Połączenie rezydualne
        out = self.gelu(identity + out)
        return out


@register_model("SEResNet")
class SEResNet(nn.Module):
    def __init__(self, block_num=[2, 2, 2, 2], in_channels=4, **kwargs):
        super(SEResNet, self).__init__()
        
        # Uwaga: L2 Regularizers z Keras (l2=1e-4) w PyTorch 
        # aplikuje się globalnie w optymalizatorze AdamW poprzez parametr `weight_decay`.

        # Stem (Początek sieci)
        self.pre1 = nn.Conv1d(in_channels, 512, kernel_size=7, stride=1, padding=3, bias=False)
        self.pre2 = nn.BatchNorm1d(512)
        # pre3 to operacja torch.exp() aplikowana w metodzie forward
        self.pre4 = nn.AvgPool1d(kernel_size=5, stride=2, padding=0)

        # ResNet Blocks
        current_channels = 512
        self.layer1, current_channels = self._make_res_block(current_channels, 128, block_num[0], stride=1)
        self.layer2, current_channels = self._make_res_block(current_channels, 256, block_num[1], stride=1)
        self.layer3, current_channels = self._make_res_block(current_channels, 512, block_num[2], stride=2)
        self.layer4, current_channels = self._make_res_block(current_channels, 512, block_num[3], stride=2)

        # Obliczanie ostatecznego rozmiaru po spłaszczeniu (dla sekwencji startowej 249 bp)
        # 249 -> AvgPool(5, 2) = 122 
        # Layer 1 & 2 nie zmniejszają wymiaru (stride=1) = 122
        # Layer 3 (stride=2) = 61
        # Layer 4 (stride=2) = 31
        self.flattened_size = current_channels * 31

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2) # Wyjście to 2 wartości (Dev, Hk)
        
        self.gelu = nn.GELU()

    def _make_res_block(self, in_channels, filter_num, blocks, stride):
        layers = []
        # Pierwszy blok obsługuje zmianę rozdzielczości (stride) i dopasowanie kanałów
        layers.append(BottleNeck(in_channels, filter_num, stride=stride))
        out_channels = filter_num * 4

        # Kolejne bloki utrzymują tę samą rozdzielczość
        for _ in range(1, blocks):
            layers.append(BottleNeck(out_channels, filter_num, stride=1))

        return nn.Sequential(*layers), out_channels

    def forward(self, x):
        # x: [Batch, 4, 249]
        x = self.pre1(x)
        x = self.pre2(x)
        x = torch.exp(x)  # Keras: Activation('exponential')
        x = self.pre4(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.flatten(x)
        
        x = self.dropout1(x)
        x = self.gelu(self.fc1(x))
        x = self.dropout2(x)
        x = self.gelu(self.fc2(x))
        
        out = self.fc3(x)
        
        # Zwracamy dwie osobne zmienne dla kompatybilności z DeepSTARR
        return out[:, 0:1], out[:, 1:2]
