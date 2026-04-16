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
        branch = self.sigmoid(branch).view(b, c, 1)
        
        # Scale
        return x * branch


class BottleNeck(nn.Module):
    def __init__(self, in_channels, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        out_channels = filter_num * 4
        self.conv1 = nn.Conv1d(in_channels, filter_num, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv1d(filter_num, filter_num, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv1d(filter_num, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.se = SEBlock(out_channels)

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
        
        # Residual connection
        out = self.gelu(identity + out)
        return out


@register_model("SEResNet")
class SEResNet(nn.Module):
    def __init__(self, block_num=[2, 2, 2, 2], in_channels=4, seq_len=249, **kwargs):
        super(SEResNet, self).__init__()

        self.pre1 = nn.Conv1d(in_channels, 512, kernel_size=7, stride=1, padding=3, bias=False)
        self.pre2 = nn.BatchNorm1d(512)
        self.pre4 = nn.AvgPool1d(kernel_size=5, stride=2, padding=0)

        # ResNet Blocks
        current_channels = 512
        self.layer1, current_channels = self._make_res_block(current_channels, 128, block_num[0], stride=1)
        self.layer2, current_channels = self._make_res_block(current_channels, 256, block_num[1], stride=1)
        self.layer3, current_channels = self._make_res_block(current_channels, 512, block_num[2], stride=2)
        self.layer4, current_channels = self._make_res_block(current_channels, 512, block_num[3], stride=2)

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, seq_len)
            x = self.pre1(dummy_input)
            x = self.pre2(x)
            x = self.pre4(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            dummy_output = self.layer4(x)

            self.flattened_size = dummy_output.view(1, -1).size(1) 

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)
        
        self.gelu = nn.GELU()

    def _make_res_block(self, in_channels, filter_num, blocks, stride):
        layers = []
        layers.append(BottleNeck(in_channels, filter_num, stride=stride))
        out_channels = filter_num * 4

        for _ in range(1, blocks):
            layers.append(BottleNeck(out_channels, filter_num, stride=1))

        return nn.Sequential(*layers), out_channels

    def forward(self, x):
        # x: [Batch, 4, seq_len]
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
        
        return out[:, 0:1], out[:, 1:2]

    def get_features(self, x):
        """Extract penultimate feature vector for cross-model comparison."""
        x = self.pre1(x)
        x = self.pre2(x)
        x = torch.exp(x)
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
        return x
