import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .registry import register_model


class Conv2dSuperKernel(nn.Module):
    """
    SuperKernel mechanizm przystosowany dla 2D Convolutions używanych w modelu Basset.
    Maska ucina wagi tylko wzdłuż osi szerokości (sequence length), ignorując oś nukleotydów (wysokość).
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 0), tau=5.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.tau = tau
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        kernel_width = kernel_size[1]
        init_val = float(kernel_width // 2)
        safe_init = torch.log(torch.exp(torch.tensor(init_val)) - 1.0) if init_val > 0 else torch.tensor(0.0)
        self.raw_filter_radius = nn.Parameter(torch.full((out_channels, 1, 1, 1), fill_value=safe_init.item()))

        # Precompute distance buffer for the kernel width dimension
        positions = torch.arange(kernel_width).float()
        center = kernel_width // 2

        distances = torch.abs(positions - center).view(1, 1, 1, kernel_width)
        self.register_buffer('distances', distances)

        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        safe_radius = F.softplus(self.raw_filter_radius)
        mask = torch.sigmoid(self.tau * (safe_radius - self.distances))
        masked_weight = self.weight * mask

        return F.conv2d(x, masked_weight, self.bias, padding=self.padding)


@register_model("BassetNetwork")
class BassetNetwork(torch.nn.Module):
    def __init__(self, seq_len=249, dropout=0.3, pooling_widths=[3, 4, 4], 
                 num_channels=[300, 200, 200], kernel_widths=[19, 11, 7], 
                 num_units=[1000, 2], **kwargs):
        super(BassetNetwork, self).__init__()
        paddings = [int((w-1)/2) for w in kernel_widths]

        self.layer1 = nn.Sequential(
            Conv2dSuperKernel(1, num_channels[0], kernel_size=(4, kernel_widths[0]), padding=(0, paddings[0])),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[0]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[0])

        self.layer2 = nn.Sequential(
            Conv2dSuperKernel(num_channels[0], num_channels[1], kernel_size=(1, kernel_widths[1]), padding=(0, paddings[1])),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[1]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[1])

        self.layer3 = nn.Sequential(
            Conv2dSuperKernel(num_channels[1], num_channels[2], kernel_size=(1, kernel_widths[2]), padding=(0, paddings[2])),
            nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[2]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[2])

        self.fc_input = 1 * seq_len * num_channels[-1]
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.fc_input, out_features=num_units[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout))

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=num_units[0], out_features=num_units[1]))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, self.fc_input)  # reshape
        x = self.fc1(x)
        x = self.fc2(x)

        return x[:, 0:1], x[:, 1:2]

    def get_features(self, x):
        """Extract penultimate feature vector for cross-model comparison."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, self.fc_input)
        x = self.fc1(x)
        return x


@register_model("CustomNetwork")
class CustomNetwork(torch.nn.Module):
    def __init__(self, seq_len=249, num_channels=[300, 200, 200], kernel_widths=[19, 11, 7], 
                 pooling_widths=[3, 4, 4], num_units=[2000, 2], dropout=0.5, **kwargs):
        super(CustomNetwork, self).__init__()
        paddings = [int((w-1)/2) for w in kernel_widths]
        self.seq_len = seq_len
        self.dropout = dropout

        conv_modules = []
        num_channels_list = [1] + num_channels
        for num, (input_channels, output_channels, kernel, padding, pooling) in \
                enumerate(zip(num_channels_list[:-1], num_channels_list[1:], kernel_widths, paddings, pooling_widths)):
            k = 4 if num == 0 else 1
            conv_modules += [
                Conv2dSuperKernel(input_channels, output_channels, kernel_size=(k, kernel), padding=(0, padding)),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, pooling), ceil_mode=True)
            ]
            seq_len = math.ceil(seq_len / pooling)
        self.conv_layers = nn.Sequential(*conv_modules)

        fc_modules = []
        self.fc_input = 1 * seq_len * num_channels_list[-1]
        num_units_list = [self.fc_input] + num_units
        for i, (input_units, output_units) in enumerate(zip(num_units_list[:-1], num_units_list[1:])):
            fc_modules.append(nn.Linear(in_features=input_units, out_features=output_units))
            if i < len(num_units_list) - 2:
                fc_modules.append(nn.ReLU())
                fc_modules.append(nn.Dropout(p=self.dropout))
                
        self.fc_layers = nn.Sequential(*fc_modules)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_input)  # reshape
        x = self.fc_layers(x)
        
        return x[:, 0:1], x[:, 1:2]

    def get_features(self, x):
        """Extract penultimate feature vector for cross-model comparison."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_input)

        # Run through all FC layers except the final Linear
        for layer in self.fc_layers[:-1]:
            x = layer(x)
        return x
