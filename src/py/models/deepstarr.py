import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model

@register_model("DeepSTARR")
class DeepSTARR(nn.Module):
    # Dodajemy parametr seq_len (domyślnie 249 dla wstecznej kompatybilności)
    def __init__(self, num_filters=256, num_filters2=60, num_filters3=60, num_filters4=120,
                 kernel_size1=7, kernel_size2=3, kernel_size3=5, kernel_size4=3, 
                 dense_neurons1=256, dense_neurons2=256, dropout_prob=0.4, pad='same',
                 permute_before_flatten=False, seq_len=249, **kwargs): 
        super(DeepSTARR, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=num_filters, kernel_size=kernel_size1, padding=pad)
        self.bn1 = nn.BatchNorm1d(num_filters, eps=1e-3, momentum=0.01)
  
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters2, kernel_size=kernel_size2, padding=pad)
        self.bn2 = nn.BatchNorm1d(num_filters2, eps=1e-3, momentum=0.01)
        
        self.conv3 = nn.Conv1d(in_channels=num_filters2, out_channels=num_filters3, kernel_size=kernel_size3, padding=pad)
        self.bn3 = nn.BatchNorm1d(num_filters3, eps=1e-3, momentum=0.01)
        
        self.conv4 = nn.Conv1d(in_channels=num_filters3, out_channels=num_filters4, kernel_size=kernel_size4, padding=pad)
        self.bn4 = nn.BatchNorm1d(num_filters4, eps=1e-3, momentum=0.01)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Tuta robimy dynamiczne wyliczenie rozmiaru dla warstwy Linear!
        flattened_size = num_filters4 * (seq_len // (2**4))
        
        self.fc1 = nn.Linear(flattened_size, dense_neurons1)
        self.bn_fc1 = nn.BatchNorm1d(dense_neurons1, eps=1e-3, momentum=0.01)
        
        self.fc2 = nn.Linear(dense_neurons1, dense_neurons2)
        self.bn_fc2 = nn.BatchNorm1d(dense_neurons2, eps=1e-3, momentum=0.01)
        
        self.fc_dev = nn.Linear(dense_neurons2, 1)
        self.fc_hk = nn.Linear(dense_neurons2, 1)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.permute_before_flatten = permute_before_flatten
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        if self.permute_before_flatten:
            x = x.permute(0, 2, 1)
            
        x = x.reshape(x.shape[0], -1)

        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        
        out_dev = self.fc_dev(x)
        out_hk = self.fc_hk(x)
        
        return out_dev, out_hk


@register_model("DeepSTARR_Siamese")
class DeepSTARR_Siamese(DeepSTARR):
    # Podklasy używają elastycznego **kwargs, które po prostu podają dalej (do DeepSTARR)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        dev_fwd, hk_fwd = self.forward_one_strand(x)
        
        x_rc = self.get_reverse_complement_tensor(x)
        dev_rc, hk_rc = self.forward_one_strand(x_rc)
        
        out_dev = (dev_fwd + dev_rc) / 2.0
        out_hk = (hk_fwd + hk_rc) / 2.0
        
        return out_dev, out_hk


@register_model("DeepSTARR_2D_Fusion")
class DeepSTARR_2D_Fusion(DeepSTARR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Pobieramy potrzebne parametry, by dynamicznie zbudować warstwy 2D
        num_filters = kwargs.get('num_filters', 256)
        num_filters2 = kwargs.get('num_filters2', 60)
        num_filters3 = kwargs.get('num_filters3', 60)
        num_filters4 = kwargs.get('num_filters4', 120)
        kernel_size1 = kwargs.get('kernel_size1', 7)
        kernel_size2 = kwargs.get('kernel_size2', 3)
        kernel_size3 = kwargs.get('kernel_size3', 5)
        kernel_size4 = kwargs.get('kernel_size4', 3)
        dense_neurons1 = kwargs.get('dense_neurons1', 256)
        
        # 1. Zastępujemy wszystkie filtry 1D filtrami 2D. 
        # padding='same' utrzymuje wymiar H=2 (obie nici) przez wszystkie warstwy.
        self.conv1_2d = nn.Conv2d(4, num_filters, kernel_size=(2, kernel_size1), padding='same')
        self.bn1_2d = nn.BatchNorm2d(num_filters, eps=1e-3, momentum=0.01)
        
        self.conv2_2d = nn.Conv2d(num_filters, num_filters2, kernel_size=(2, kernel_size2), padding='same')
        self.bn2_2d = nn.BatchNorm2d(num_filters2, eps=1e-3, momentum=0.01)
        
        self.conv3_2d = nn.Conv2d(num_filters2, num_filters3, kernel_size=(2, kernel_size3), padding='same')
        self.bn3_2d = nn.BatchNorm2d(num_filters3, eps=1e-3, momentum=0.01)
        
        self.conv4_2d = nn.Conv2d(num_filters3, num_filters4, kernel_size=(2, kernel_size4), padding='same')
        self.bn4_2d = nn.BatchNorm2d(num_filters4, eps=1e-3, momentum=0.01)
        
        # 2. Pooling działa tylko w osi X (skraca sekwencję), ale nie rusza osi Y (nici).
        self.pool1_2d = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool2_2d = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool3_2d = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool4_2d = nn.MaxPool2d(kernel_size=(1, 2))
        
        # Kasujemy stare warstwy 1D odziedziczone z bazowego DeepSTARR (oszczędność VRAM GPU)
        del self.conv1, self.bn1, self.pool1
        del self.conv2, self.bn2, self.pool2
        del self.conv3, self.bn3, self.pool3
        del self.conv4, self.bn4, self.pool4
        del self.fc1
        
        # 3. Nowa warstwa w pełni połączona musi przyjąć 2x więcej danych na wejściu (bo obie nici dotrwały do końca)
        # Wymiar to: num_filters4 * 2 (nici) * (249 // 16) (długość sekwencji po poolingach)
        self.fc1_2d = nn.Linear(num_filters4 * 2 * (249 // (2**4)), dense_neurons1)

    def forward(self, x):
        x_rc = self.get_reverse_complement_tensor(x)
        # Tworzymy tensor 2D: [Batch, Channels=4, Nici=2, Sekwencja=249]
        x = torch.stack([x, x_rc], dim=2)
        
        # Przejście przez pełną ścieżkę 2D
        x = self.pool1_2d(F.relu(self.bn1_2d(self.conv1_2d(x))))
        x = self.pool2_2d(F.relu(self.bn2_2d(self.conv2_2d(x))))
        x = self.pool3_2d(F.relu(self.bn3_2d(self.conv3_2d(x))))
        x = self.pool4_2d(F.relu(self.bn4_2d(self.conv4_2d(x))))
        
        if self.permute_before_flatten:
            # Dopasowanie jeśli model ma emulować wymiary wprost z Kerasa
            x = x.permute(0, 2, 3, 1)  
            
        # Spłaszczenie do wektora 1D [Batch, features]
        x = x.reshape(x.shape[0], -1)

        # Głowica gęsta, teraz korzystająca z fc1_2d
        x = self.dropout1(F.relu(self.bn_fc1(self.fc1_2d(x))))
        x = self.dropout2(F.relu(self.bn_fc2(self.fc2(x))))
        
        out_dev = self.fc_dev(x)
        out_hk = self.fc_hk(x)
        
        return out_dev, out_hk
