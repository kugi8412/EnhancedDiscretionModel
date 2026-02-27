import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSTARR(nn.Module):
    def __init__(self, params, permute_before_flatten=False):
        super(DeepSTARR, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=params['num_filters'],
                               kernel_size=params['kernel_size1'], padding=params['pad'])
        self.bn1 = nn.BatchNorm1d(params['num_filters'], eps=1e-3, momentum=0.01)
  
        self.conv2 = nn.Conv1d(in_channels=params['num_filters'], out_channels=params['num_filters2'],
                               kernel_size=params['kernel_size2'], padding=params['pad'])
        self.bn2 = nn.BatchNorm1d(params['num_filters2'], eps=1e-3, momentum=0.01)
        
        self.conv3 = nn.Conv1d(in_channels=params['num_filters2'], out_channels=params['num_filters3'],
                               kernel_size=params['kernel_size3'], padding=params['pad'])
        self.bn3 = nn.BatchNorm1d(params['num_filters3'], eps=1e-3, momentum=0.01)
        
        self.conv4 = nn.Conv1d(in_channels=params['num_filters3'], out_channels=params['num_filters4'],
                               kernel_size=params['kernel_size4'], padding=params['pad'])
        self.bn4 = nn.BatchNorm1d(params['num_filters4'], eps=1e-3, momentum=0.01)
        
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        
        self.fc1 = nn.Linear(120 * (249 // (2**4)), params['dense_neurons1'])
        self.bn_fc1 = nn.BatchNorm1d(params['dense_neurons1'], eps=1e-3, momentum=0.01)
        
        self.fc2 = nn.Linear(params['dense_neurons1'], params['dense_neurons2'])
        self.bn_fc2 = nn.BatchNorm1d(params['dense_neurons2'], eps=1e-3, momentum=0.01)
        
        self.fc_dev = nn.Linear(params['dense_neurons2'], 1)
        self.fc_hk = nn.Linear(params['dense_neurons2'], 1)
        
        self.dropout1 = nn.Dropout(params['dropout_prob'])
        self.dropout2 = nn.Dropout(params['dropout_prob'])
        
        self.permute_before_flatten = permute_before_flatten

    def forward_one_strand(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        if self.permute_before_flatten:
            x = x.permute(0, 2, 1)  
        x = x.reshape(x.shape[0], -1)

        x = self.dropout1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn_fc2(self.fc2(x))))
        
        out_dev = self.fc_dev(x)
        out_hk = self.fc_hk(x)
        
        return out_dev, out_hk

    def forward(self, x):
        return self.forward_one_strand(x)

    def get_reverse_complement_tensor(self, x):
        return x.flip(dims=[2])[:, [2, 3, 0, 1], :]


import torch
import torch.nn as nn
import torch.nn.functional as F

class RCBatchNorm1d(nn.BatchNorm1d):
    """
    PyTorch translation of RevCompConv1DBatchNorm from Keras (Shrikumar et al.).
    Forces FWD and RC channels to share exactly the same mean, variance, gamma, and beta.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        assert num_features % 2 == 0, "Number of features must be even for RC BatchNorm"
        # Inicjalizujemy wagi tylko dla połowy kanałów
        super().__init__(num_features // 2, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        B, C, L = x.shape
        half = C // 2
        
        # Rozdziel tensor na filtry FWD i RC
        x_fwd = x[:, :half, :]
        x_rc = x[:, half:, :]
        
        # Konkatenacja wzdłuż wymiaru Batch (B * 2).
        # Zmuszamy BatchNorm do potraktowania nici komplementarnej jako kolejnej 
        # sekwencji w paczce. Wyliczy to idealnie wspólną średnią i wariancję.
        x_stacked = torch.cat([x_fwd, x_rc], dim=0)
        
        out_stacked = super().forward(x_stacked)
        
        # Rozdziel i złóż z powrotem wzdłuż wymiaru Kanałów
        out_fwd = out_stacked[:B, :, :]
        out_rc = out_stacked[B:, :, :]
        
        return torch.cat([out_fwd, out_rc], dim=1)


class RCMaxPool1d(nn.Module):
    """
    Reverse-Complement Max Pooling.
    Guarantees that sliding windows for RC channels perfectly mirror the FWD channels,
    even for odd sequence lengths (like 249 bp).
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.maxpool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        half = x.shape[1] // 2
        
        # FWD: normalny MaxPool z lewej do prawej
        x_fwd = self.maxpool(x[:, :half, :])
        
        # RC: odwracamy -> pool -> odwracamy
        # Dzięki temu wymuszamy poolowanie "od prawej do lewej"
        x_rc_flipped = x[:, half:, :].flip(dims=[2])
        x_rc_pooled = self.maxpool(x_rc_flipped)
        x_rc = x_rc_pooled.flip(dims=[2])
        
        return torch.cat([x_fwd, x_rc], dim=1)


class RCConv1d(nn.Module):
    """
    Reverse-Complement 1D Convolution.
    Compatible with both the first input layer (One-Hot) and deep hidden layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', is_first_layer=False):
        super().__init__()
        assert out_channels % 2 == 0
        self.is_first_layer = is_first_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        # We train only half of the filters, the other half are generated by flipping and reordering
        self.weight = nn.Parameter(torch.Tensor(out_channels // 2, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels // 2))
        
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.is_first_layer:
            rev_idx = [2, 3, 0, 1]
        else:
            # 'i' -> 'i + in_channels/2'
            half = self.in_channels // 2
            rev_idx = list(range(half, self.in_channels)) + list(range(0, half))
            
        # Mirror weights for RC channels
        weight_rc = self.weight.flip(dims=[2])[:, rev_idx, :]
        weight_combined = torch.cat([self.weight, weight_rc], dim=0)
        
        # Clone bias and reverse for RC channels
        bias_combined = torch.cat([self.bias, self.bias], dim=0)
        
        return F.conv1d(x, weight_combined, bias_combined, padding=self.padding)


class DeepSTARR_RC_Sharing(nn.Module):
    """
    DeepSTARR architecture implemented with 100% true Reverse-Complement Isometry.
    """
    def __init__(self, params, permute_before_flatten=False):
        super().__init__()
        
        # Input layer must be RCConv1d to ensure perfect RC isometry from the start
        self.conv1 = RCConv1d(in_channels=4, out_channels=params['num_filters'],
                              kernel_size=params['kernel_size1'], padding=params['pad'], is_first_layer=True)
        self.bn1 = RCBatchNorm1d(params['num_filters'])
        self.pool1 = RCMaxPool1d(2)
  
        self.conv2 = RCConv1d(params['num_filters'], params['num_filters2'], params['kernel_size2'], padding=params['pad'])
        self.bn2 = RCBatchNorm1d(params['num_filters2'])
        self.pool2 = RCMaxPool1d(2)
        
        self.conv3 = RCConv1d(params['num_filters2'], params['num_filters3'], params['kernel_size3'], padding=params['pad'])
        self.bn3 = RCBatchNorm1d(params['num_filters3'])
        self.pool3 = RCMaxPool1d(2)
        
        self.conv4 = RCConv1d(params['num_filters3'], params['num_filters4'], params['kernel_size4'], padding=params['pad'])
        self.bn4 = RCBatchNorm1d(params['num_filters4'])
        self.pool4 = RCMaxPool1d(2)
        
        # Half of the filters are redundant due to RC sharing
        self.fc1 = nn.Linear((params['num_filters4'] // 2) * (249 // (2**4)), params['dense_neurons1'])
        self.bn_fc1 = nn.BatchNorm1d(params['dense_neurons1'], eps=1e-3, momentum=0.01)
        
        self.fc2 = nn.Linear(params['dense_neurons1'], params['dense_neurons2'])
        self.bn_fc2 = nn.BatchNorm1d(params['dense_neurons2'], eps=1e-3, momentum=0.01)
        
        self.fc_dev = nn.Linear(params['dense_neurons2'], 1)
        self.fc_hk = nn.Linear(params['dense_neurons2'], 1)
        
        self.dropout1 = nn.Dropout(params['dropout_prob'])
        self.dropout2 = nn.Dropout(params['dropout_prob'])
        self.permute_before_flatten = permute_before_flatten
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Symetric fusion
        half = x.shape[1] // 2
        x_fwd = x[:, :half, :]
        x_rc = x[:, half:, :].flip(dims=[2]) 
        x_sym = x_fwd + x_rc
        
        if self.permute_before_flatten:
            x_sym = x_sym.permute(0, 2, 1)  
            
        x_sym = x_sym.reshape(x_sym.shape[0], -1)
        x_out = self.dropout1(F.relu(self.bn_fc1(self.fc1(x_sym))))
        x_out = self.dropout2(F.relu(self.bn_fc2(self.fc2(x_out))))
        
        return self.fc_dev(x_out), self.fc_hk(x_out)


class DeepSTARR_Siamese(DeepSTARR):
    def __init__(self, params, permute_before_flatten=False):
        super().__init__(params, permute_before_flatten)

    def forward(self, x):
        dev_fwd, hk_fwd = self.forward_one_strand(x)
        
        x_rc = self.get_reverse_complement_tensor(x)
        dev_rc, hk_rc = self.forward_one_strand(x_rc)
        
        out_dev = (dev_fwd + dev_rc) / 2.0
        out_hk = (hk_fwd + hk_rc) / 2.0
        
        return out_dev, out_hk


class DeepSTARR_2D_Fusion(DeepSTARR):
    def __init__(self, params, permute_before_flatten=False):
        super().__init__(params, permute_before_flatten)
        
        pad_w = params['kernel_size1'] // 2 if params['pad'] == 'same' else 0
        
        self.conv1_2d = nn.Conv2d(
            in_channels=4, 
            out_channels=params['num_filters'],
            kernel_size=(2, params['kernel_size1']), 
            padding=(0, pad_w)
        )
        del self.conv1 

    def forward(self, x):
        x_rc = self.get_reverse_complement_tensor(x)
        x_2d = torch.stack([x, x_rc], dim=2)
        
        x = self.conv1_2d(x_2d).squeeze(2)
        
        x = self.pool1(F.relu(self.bn1(x)))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        if self.permute_before_flatten:
            x = x.permute(0, 2, 1)  
            
        x = x.reshape(x.shape[0], -1)

        x = self.dropout1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn_fc2(self.fc2(x))))
        
        out_dev = self.fc_dev(x)
        out_hk = self.fc_hk(x)
        
        return out_dev, out_hk
