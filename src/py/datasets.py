import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

# Importujemy augmentacje bezpośrednio z Twojego pliku augment.py
from augment import (
    RandomDeletion, RandomInsertion, RandomTranslocation, 
    RandomInversion, RandomMutation, RandomRC, RandomNoise,
    NormalizeNoise
)


class DNADataset(Dataset):
    def __init__(self, X, Y_first, Y_second, augment=False, noise_config=None, evoaug_config=None, seed=None):
        self.X = X
        self.Y_first = Y_first
        self.Y_second = Y_second
        self.augment = augment
        
        # Konfiguracja szumu na wartościach docelowych (Target Noise)
        self.noise_apply = noise_config.get('apply', False) if noise_config else False
        self.base_std = noise_config.get('std', 0.05) if noise_config else 0.0

        # EvoAug Config (Bezpośrednio w Dataset)
        self.aug_list = []
        self.max_augs = 0
        self.insert_max = 0
        self.hard_aug = False
        self.distribution = 'uniform'
        self.exp_scale = 0.5  # Parametr skali dla rozkładu wykładniczego

        if self.augment and evoaug_config:
            self.max_augs = evoaug_config.get('max_augs_per_seq', 2)
            self.hard_aug = evoaug_config.get('hard_aug', False)
            self.distribution = evoaug_config.get('distribution', 'uniform')
            self.exp_scale = evoaug_config.get('exp_scale', 1.0) # Reguluje jak rzadkie są mutacje
            
            self._build_aug_list(evoaug_config.get('augmentations', []))

    def _build_aug_list(self, aug_configs):
        """Mapuje nazwy z YAML na instancje z augment.py i zlicza insert_max."""
        aug_dict = {
            'RandomDeletion': RandomDeletion,
            'RandomInsertion': RandomInsertion,
            'RandomTranslocation': RandomTranslocation,
            'RandomInversion': RandomInversion,
            'RandomMutation': RandomMutation,
            'RandomRC': RandomRC,
            'RandomNoise': RandomNoise,
            'NormalizeNoise': NormalizeNoise
        }
        
        for ac in aug_configs:
            name = ac['name']
            kwargs = ac.get('kwargs', {})
            if name in aug_dict:
                aug_instance = aug_dict[name](**kwargs)
                self.aug_list.append(aug_instance)
                
                # Zapisujemy insert_max, aby wyrównywać długości DNA (padding)
                if hasattr(aug_instance, 'insert_max'):
                    self.insert_max = max(self.insert_max, aug_instance.insert_max)

    def _pad_end(self, x):
        """Dodaje losowe DNA na końcu sekwencji (wyrównanie wymiarów)."""
        N, A, L = x.shape
        a = torch.eye(A)
        p = torch.tensor([1/A for _ in range(A)])
        # Losujemy nukleotydy do paddingu
        padding = torch.stack([a[p.multinomial(self.insert_max, replacement=True)].transpose(0,1) for _ in range(N)]).to(x.device)
        x_padded = torch.cat([x, padding], dim=2)
        return x_padded

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Pobieramy kopię tensorów
        x_tensor = self.X[idx].clone().detach() # Wymiar: (4, L)
        y1 = self.Y_first[idx].clone().detach()
        y2 = self.Y_second[idx].clone().detach()

        num_applied = 0

        # --- NAKŁADANIE AUGMENTACJI (DNA) ---
        if self.augment and self.max_augs > 0 and len(self.aug_list) > 0:
            
            # 1. Decyzja ile augmentacji nałożyć
            if self.hard_aug:
                num_applied = self.max_augs
            else:
                if self.distribution == 'exponential':
                    # Rozkład wykładniczy: faworyzuje 0, rzadsze 1, bardzo rzadkie 2 itd.
                    sampled_val = int(np.random.exponential(scale=self.exp_scale))
                    num_applied = min(sampled_val, self.max_augs)
                else:
                    # Rozkład jednostajny: równe szanse na 0, 1, ..., max_augs
                    num_applied = np.random.randint(0, self.max_augs + 1)
            
            # 2. Aplikowanie wylosowanych mutacji
            if num_applied > 0:
                # Wybieramy unikalne indeksy augmentacji z listy
                aug_indices = np.random.choice(len(self.aug_list), num_applied, replace=False)
                
                # Klasy z augment.py operują na wymiarze (N, A, L), więc dodajemy sztuczny batch (N=1)
                x_tensor = x_tensor.unsqueeze(0) 
                insert_status = True
                
                for a_idx in aug_indices:
                    aug_func = self.aug_list[a_idx]
                    x_tensor = aug_func(x_tensor)
                    
                    if hasattr(aug_func, 'insert_max'):
                        insert_status = False
                
                # Uzupełnianie paddingu, jeśli sekwencja nie dostała RandomInsertion
                if insert_status and self.insert_max > 0:
                    x_tensor = self._pad_end(x_tensor)
                    
                x_tensor = x_tensor.squeeze(0)

        elif self.insert_max > 0:
            # Wyrównanie długości dla sekwencji, która wylosowała 0 mutacji
            x_tensor = self._pad_end(x_tensor.unsqueeze(0)).squeeze(0)

        # --- NAKŁADANIE SZUMU NA TARGET (Wartości wyjściowe) ---
        if self.noise_apply:
            # Skalowanie: im więcej mutacji na sekwencji, tym większa niepewność (szum) na wartościach Y
            multiplier = num_applied / self.max_augs if self.max_augs > 0 else 0.0
            current_std = self.base_std + (self.base_std * multiplier)
            
            if current_std > 0:
                y1 += torch.normal(mean=0.0, std=current_std, size=y1.size())
                y2 += torch.normal(mean=0.0, std=current_std, size=y2.size())

        return x_tensor, y1, y2
