import torch
import torch.nn as nn
import math
from .registry import register_model


@register_model("BassetNetwork")
class BassetNetwork(torch.nn.Module):
    def __init__(self, seq_len=249, dropout=0.3, pooling_widths=[3, 4, 4], 
                 num_channels=[300, 200, 200], kernel_widths=[19, 11, 7], 
                 num_units=[1000, 2], **kwargs):
        super(BassetNetwork, self).__init__()
        paddings = [int((w-1)/2) for w in kernel_widths]

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, num_channels[0], kernel_size=(4, kernel_widths[0]), padding=(0, paddings[0])),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[0]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[0])

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_channels[0], num_channels[1], kernel_size=(1, kernel_widths[1]), padding=(0, paddings[1])),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[1]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[1])

        self.layer3 = nn.Sequential(
            nn.Conv2d(num_channels[1], num_channels[2], kernel_size=(1, kernel_widths[2]), padding=(0, paddings[2])),
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
        # Wejście DeepSTARR: [B, 4, L]. Basset wymaga: [B, 1, 4, L]
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, self.fc_input)  # reshape
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Zwracamy dwie osobne wartości ciągłe (Log2 enrichment) dla Dev i Hk
        return x[:, 0:1], x[:, 1:2]


@register_model("CustomNetwork")
class CustomNetwork(torch.nn.Module):
    def __init__(self, seq_len=249, num_channels=[300, 200, 200], kernel_widths=[19, 11, 7], 
                 pooling_widths=[3, 4, 4], num_units=[2000, 2], dropout=0.5, **kwargs):
        super(CustomNetwork, self).__init__()
        paddings = [int((w-1)/2) for w in kernel_widths]
        self.seq_len = seq_len
        self.dropout = dropout

        conv_modules = []
        num_channels = [1] + num_channels
        for num, (input_channels, output_channels, kernel, padding, pooling) in \
                enumerate(zip(num_channels[:-1], num_channels[1:], kernel_widths, paddings, pooling_widths)):
            k = 4 if num == 0 else 1
            conv_modules += [
                nn.Conv2d(input_channels, output_channels, kernel_size=(k, kernel), padding=(0, padding)),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, pooling), ceil_mode=True)
            ]
            seq_len = math.ceil(seq_len / pooling)
        self.conv_layers = nn.Sequential(*conv_modules)

        fc_modules = []
        self.fc_input = 1 * seq_len * num_channels[-1]
        num_units = [self.fc_input] + num_units
        for i, (input_units, output_units) in enumerate(zip(num_units[:-1], num_units[1:])):
            fc_modules.append(nn.Linear(in_features=input_units, out_features=output_units))
            if i < len(num_units) - 2: # Dodajemy ReLU/Dropout tylko między ukrytymi warstwami
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


@register_model("TestNetwork")
class TestNetwork(torch.nn.Module):
    def __init__(self, seq_len=249, dropout=0.3, pooling_widths=[3, 4, 4], 
                 num_channels=[30, 20, 20], kernel_widths=[5, 3, 3], 
                 num_units=[100, 2], **kwargs):
        super(TestNetwork, self).__init__()
        paddings = [int((w-1)/2) for w in kernel_widths]

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, num_channels[0], kernel_size=(4, kernel_widths[0]), padding=(0, paddings[0])),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[0]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[0])

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_channels[0], num_channels[1], kernel_size=(1, kernel_widths[1]), padding=(0, paddings[1])),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[1]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[1])

        self.layer3 = nn.Sequential(
            nn.Conv2d(num_channels[1], num_channels[2], kernel_size=(1, kernel_widths[2]), padding=(0, paddings[2])),
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


@register_model("DropoutNetwork")
class DropoutNetwork(torch.nn.Module):
    def __init__(self, seq_len=249, num_channels=[300, 200, 200], kernel_widths=[19, 11, 7], 
                 pooling_widths=[3, 4, 4], num_units=[2000, 2], 
                 dropout_conv=0.2, dropout_fc=0.5, **kwargs):
        super(DropoutNetwork, self).__init__()
        paddings = [int((w-1)/2) for w in kernel_widths]
        self.seq_len = seq_len
        self.dropout_conv = dropout_conv
        self.dropout_fc = dropout_fc

        conv_modules = []
        num_channels = [1] + num_channels
        for num, (input_channels, output_channels, kernel, padding, pooling) in \
                enumerate(zip(num_channels[:-1], num_channels[1:], kernel_widths, paddings, pooling_widths)):
            k = 4 if num == 0 else 1
            conv_modules += [
                nn.Conv2d(input_channels, output_channels, kernel_size=(k, kernel), padding=(0, padding)),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Dropout2d(p=self.dropout_conv),
                nn.MaxPool2d(kernel_size=(1, pooling), ceil_mode=True)
            ]
            seq_len = math.ceil(seq_len / pooling)
        self.conv_layers = nn.Sequential(*conv_modules)

        fc_modules = []
        self.fc_input = 1 * seq_len * num_channels[-1]
        num_units = [self.fc_input] + num_units
        for i, (input_units, output_units) in enumerate(zip(num_units[:-1], num_units[1:])):
            fc_modules.append(nn.Linear(in_features=input_units, out_features=output_units))
            if i < len(num_units) - 2:
                fc_modules.append(nn.ReLU())
                fc_modules.append(nn.Dropout(p=self.dropout_fc))
                
        self.fc_layers = nn.Sequential(*fc_modules)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_input)  # reshape
        x = self.fc_layers(x)
        
        return x[:, 0:1], x[:, 1:2]
