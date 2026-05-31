# -*- coding: utf-8 -*-
"""Dataset class for DNA enhancer sequences with EvoAug augmentation."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from augment import (
    RandomDeletion, RandomInsertion, RandomTranslocation,
    RandomInversion, RandomMutation, RandomRC, RandomNoise, NormalizeNoise,
)


class DNADataset(Dataset):
    """One-hot encoded DNA dataset with optional augmentation and label noise.

    Parameters
    ----------
    X : torch.Tensor
        One-hot encoded sequences, shape ``(N, 4, L)``.
    Y_first, Y_second : torch.Tensor
        Scalar expression labels (Dev / Hk).
    augment : bool
        Whether to apply augmentations at sampling time.
    noise_config : dict or None
        Label noise settings (``apply``, ``std``).
    evoaug_config : dict or None
        Augmentation pipeline settings (see YAML ``data.evoaug``).
    seed : int or None
        Not used directly (global seed is set elsewhere).
    """

    def __init__(self, X, Y_first, Y_second, augment=False,
                 noise_config=None, evoaug_config=None, seed=None):
        self.X = X
        self.Y_first = Y_first
        self.Y_second = Y_second
        self.augment = augment

        self.noise_apply = noise_config.get('apply', False) if noise_config else False
        self.base_std = noise_config.get('std', 0.05) if noise_config else 0.0

        self.aug_list = []
        self.max_augs = 0
        self.insert_max = 0
        self.hard_aug = False
        self.distribution = 'poisson'
        self.exp_scale = 1.0
        self.poisson_lambda = 0.5

        # Always read evoaug config to determine insert_max (needed for padding even at validation)
        if evoaug_config:
            self.max_augs = evoaug_config.get('max_augs_per_seq', 2)
            self.hard_aug = evoaug_config.get('hard_aug', False)
            self.distribution = evoaug_config.get('distribution', 'poisson')
            self.exp_scale = evoaug_config.get('exp_scale', 1.0)
            self.poisson_lambda = evoaug_config.get('poisson_lambda', 0.5)
            self._build_aug_list(evoaug_config.get('augmentations', []))

    def _build_aug_list(self, aug_configs):
        aug_dict = {
            'RandomDeletion': RandomDeletion,
            'RandomInsertion': RandomInsertion,
            'RandomTranslocation': RandomTranslocation,
            'RandomInversion': RandomInversion,
            'RandomMutation': RandomMutation,
            'RandomRC': RandomRC,
            'RandomNoise': RandomNoise,
            'NormalizeNoise': NormalizeNoise,
        }
        
        for ac in aug_configs:
            name = ac['name']
            kwargs = ac.get('kwargs', {})
            if name in aug_dict:
                aug_instance = aug_dict[name](**kwargs)
                self.aug_list.append(aug_instance)
                
                if hasattr(aug_instance, 'insert_max'):
                    self.insert_max = max(self.insert_max, aug_instance.insert_max)

    def _pad_end(self, x):
        N, A, L = x.shape
        a = torch.eye(A)
        p = torch.tensor([1/A for _ in range(A)])
        padding = torch.stack([a[p.multinomial(self.insert_max, replacement=True)].transpose(0,1) for _ in range(N)]).to(x.device)
        return torch.cat([x, padding], dim=2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_tensor = self.X[idx].clone().detach()
        y1 = self.Y_first[idx].clone().detach()
        y2 = self.Y_second[idx].clone().detach()

        num_applied = 0
        inserted = False  # Track whether an insertion augmentation was applied

        if self.augment and self.max_augs > 0 and len(self.aug_list) > 0:
            if self.hard_aug:
                num_applied = self.max_augs
            else:
                if self.distribution == 'poisson':
                    sampled_val = np.random.poisson(lam=self.poisson_lambda)
                    num_applied = min(sampled_val, self.max_augs)
                elif self.distribution == 'exponential':
                    sampled_val = int(np.random.exponential(scale=self.exp_scale))
                    num_applied = min(sampled_val, self.max_augs)
                else:
                    num_applied = np.random.randint(0, self.max_augs + 1)
            
            if num_applied > 0:
                aug_indices = np.random.choice(len(self.aug_list), num_applied, replace=False)
                x_tensor = x_tensor.unsqueeze(0) 
                applied_noise = False
                
                for a_idx in aug_indices:
                    aug_func = self.aug_list[a_idx]
                    x_tensor = aug_func(x_tensor)
                    
                    if isinstance(aug_func, (RandomNoise, NormalizeNoise)):
                        applied_noise = True
                    if hasattr(aug_func, 'insert_max'):
                        inserted = True
                
                x_tensor = x_tensor.squeeze(0)
                
                if applied_noise:
                    x_tensor = F.softmax(x_tensor, dim=0)

        # Pad sequences to uniform length when insertion augmentation is configured
        # but this particular sample did not receive an insertion (or we are in validation).
        if self.insert_max > 0 and not inserted:
            x_tensor = self._pad_end(x_tensor.unsqueeze(0)).squeeze(0)

        if self.noise_apply:
            multiplier = num_applied / self.max_augs if self.max_augs > 0 else 0.0
            current_std = self.base_std + (self.base_std * multiplier)
            
            if current_std > 0:
                y1 += torch.normal(mean=0.0, std=current_std, size=y1.size())
                y2 += torch.normal(mean=0.0, std=current_std, size=y2.size())

        return x_tensor, y1, y2
