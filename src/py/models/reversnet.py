import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model

class RCBatchNorm1d(nn.BatchNorm1d):
    """
    RC-invariant BatchNorm.
    Forces FWD and RC channels to share exactly the same mean, variance, gamma, and beta.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        assert num_features % 2 == 0, "Number of features must be even for RC BatchNorm"
        super().__init__(num_features // 2, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        B, C, L = x.shape
        half = C // 2
        
        x_fwd = x[:, :half, :]
        x_rc = x[:, half:, :]
        
        # Concatenate along the batch dimension (B * 2)
        x_stacked = torch.cat([x_fwd, x_rc], dim=0)
        out_stacked = super().forward(x_stacked)
        
        out_fwd = out_stacked[:B, :, :]
        out_rc = out_stacked[B:, :, :]
        
        return torch.cat([out_fwd, out_rc], dim=1)


class RCMaxPool1d(nn.Module):
    """
    Reverse-Complement Max Pooling.
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.maxpool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        half = x.shape[1] // 2
        
        x_fwd = self.maxpool(x[:, :half, :])
        
        x_rc_flipped = x[:, half:, :].flip(dims=[2])
        x_rc_pooled = self.maxpool(x_rc_flipped)
        x_rc = x_rc_pooled.flip(dims=[2])
        
        return torch.cat([x_fwd, x_rc], dim=1)


class RCConv1dSuperKernel(nn.Module):
    """
    Reverse-Complement 1D Convolution with SuperKernel mechanism.
    Allows each filter to dynamically learn its effective spatial size.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', is_first_layer=False, tau=5.0):
        super().__init__()
        assert out_channels % 2 == 0
        self.is_first_layer = is_first_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.kernel_size = kernel_size
        self.tau = tau  

        # Base weights
        self.weight = nn.Parameter(torch.Tensor(out_channels // 2, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels // 2))

        # SUPERKERNEL: learnable radius per filter
        # Initialized so that after softplus it equals half the kernel size
        # Inverse softplus for initial value: log(exp(val) - 1)
        init_val = float(kernel_size // 2)
        safe_init = torch.log(torch.exp(torch.tensor(init_val)) - 1.0)
        self.raw_filter_radius = nn.Parameter(torch.full((out_channels // 2, 1, 1), fill_value=safe_init.item()))
        
        # Compute static distances from the filter center
        positions = torch.arange(kernel_size).float()
        center = kernel_size // 2
        distances = torch.abs(positions - center).view(1, 1, kernel_size)
        self.register_buffer('distances', distances)

        # Weight initialization
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # Safety: softplus ensures radius is always > 0
        safe_radius = F.softplus(self.raw_filter_radius)
        
        # Generate SuperKernel mask
        mask = torch.sigmoid(self.tau * (safe_radius - self.distances))
        
        # Apply mask to weights
        masked_weight = self.weight * mask

        # Reverse-complement logic on the MASKED weights
        if self.is_first_layer:
            rev_idx = [2, 3, 0, 1]
        else:
            half = self.in_channels // 2
            rev_idx = list(range(half, self.in_channels)) + list(range(0, half))
            
        weight_rc = masked_weight.flip(dims=[2])[:, rev_idx, :]
        weight_combined = torch.cat([masked_weight, weight_rc], dim=0)
        
        bias_combined = torch.cat([self.bias, self.bias], dim=0)
        
        return F.conv1d(x, weight_combined, bias_combined, padding=self.padding)


@register_model("ReverseNet_SuperKernel")
class ReverseNet_SuperKernel(nn.Module):
    """
    ReverseNet (DeepSTARR-style) RC-Sharing model extended with SuperKernels.
    """
    def __init__(self, num_filters=256, num_filters2=60, num_filters3=60, num_filters4=120,
                 kernel_size1=15, kernel_size2=5, kernel_size3=3, kernel_size4=3, 
                 dense_neurons1=256, dense_neurons2=256, dropout_prob=0.4, pad='same',
                 permute_before_flatten=False, seq_len=249, **kwargs):
        super().__init__()
        
        self.conv1 = RCConv1dSuperKernel(in_channels=4, out_channels=num_filters,
                                         kernel_size=kernel_size1, padding=pad, is_first_layer=True)
        self.bn1 = RCBatchNorm1d(num_filters)
        self.pool1 = RCMaxPool1d(2)

        self.conv2 = RCConv1dSuperKernel(num_filters, num_filters2, kernel_size2, padding=pad)
        self.bn2 = RCBatchNorm1d(num_filters2)
        self.pool2 = RCMaxPool1d(2)
        
        self.conv3 = RCConv1dSuperKernel(num_filters2, num_filters3, kernel_size3, padding=pad)
        self.bn3 = RCBatchNorm1d(num_filters3)
        self.pool3 = RCMaxPool1d(2)
        
        self.conv4 = RCConv1dSuperKernel(num_filters3, num_filters4, kernel_size4, padding=pad)
        self.bn4 = RCBatchNorm1d(num_filters4)
        self.pool4 = RCMaxPool1d(2)
        
        # Dynamic flattened size computation for FC layer
        flattened_size = (num_filters4 // 2) * (seq_len // (2**4))
        self.fc1 = nn.Linear(flattened_size, dense_neurons1)
        self.bn_fc1 = nn.BatchNorm1d(dense_neurons1, eps=1e-3, momentum=0.01)
        
        self.fc2 = nn.Linear(dense_neurons1, dense_neurons2)
        self.bn_fc2 = nn.BatchNorm1d(dense_neurons2, eps=1e-3, momentum=0.01)
        
        # Separate output heads for Dev and Hk
        self.fc_dev = nn.Linear(dense_neurons2, 1)
        self.fc_hk = nn.Linear(dense_neurons2, 1)
        
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.permute_before_flatten = permute_before_flatten
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Symmetric strand fusion
        half = x.shape[1] // 2
        x_fwd = x[:, :half, :]
        x_rc = x[:, half:, :].flip(dims=[2]) 
        x_sym = x_fwd + x_rc
        
        if self.permute_before_flatten:
            x_sym = x_sym.permute(0, 2, 1)  
            
        x_sym = x_sym.reshape(x_sym.shape[0], -1)
        x_out = self.dropout1(F.relu(self.bn_fc1(self.fc1(x_sym))))
        x_out = self.dropout2(F.relu(self.bn_fc2(self.fc2(x_out))))
        
        # Return two separate predictions for the training loop
        return self.fc_dev(x_out), self.fc_hk(x_out)

    def get_features(self, x):
        """Extract penultimate feature vector for cross-model comparison."""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        half = x.shape[1] // 2
        x_fwd = x[:, :half, :]
        x_rc = x[:, half:, :].flip(dims=[2])
        x_sym = x_fwd + x_rc
        if self.permute_before_flatten:
            x_sym = x_sym.permute(0, 2, 1)
        x_sym = x_sym.reshape(x_sym.shape[0], -1)
        x_out = self.dropout1(F.relu(self.bn_fc1(self.fc1(x_sym))))
        x_out = self.dropout2(F.relu(self.bn_fc2(self.fc2(x_out))))
        return x_out