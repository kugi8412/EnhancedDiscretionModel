# -*- coding: utf-8 -*-
"""Finite Scalar Quantisation — HYPPE's replacement for the VQ-VAE codebook.

FSQ (Mentzer et al., 2023) drops the codebook entirely.  Each latent channel is
squashed to a bounded range and rounded to one of ``L_i`` levels; the implicit
codebook is the Cartesian product grid, of size ``prod(L_i)``.

Why this is the right trade for promoter design, concretely:

* **Dead codes cannot happen.**  HypVQNet needs EMA updates, a Frechet-mean
  codebook step, a dead-code restart heuristic and a commitment weight, and the
  parent framework still measured perplexities of 120.8/128, 214.4/256 and
  341.1/512 — a quarter to a third of every codebook idle.  An FSQ grid point
  exists whether or not anything landed on it, so utilisation is a property of
  the encoder, not a failure mode of the quantiser.
* **No commitment loss.**  That removes the one term in HypVQNet that has to be
  written as the smooth surrogate ``-<x,y>_L - 1`` to avoid NaN, and removes
  the tuning of its weight against reconstruction.
* **The code index is a small integer vector.**  Diffusion over the latent grid
  needs a latent it can perturb continuously and snap back; FSQ's rounding *is*
  that snap, and it is one line.

The cost is that the quantiser no longer learns where the codes should be.  For
a hierarchy that is an advantage: the *geometry* is imposed by the hyperbolic
head (see ``hyppe.py``), and a learned codebook drifting under EMA would fight
it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def round_ste(x):
    """Round with a straight-through gradient."""
    return x + (torch.round(x) - x).detach()


class FSQ(nn.Module):
    """Finite scalar quantiser over ``len(levels)`` channels.

    ``levels`` gives the number of quantisation bins per channel; the implicit
    codebook size is their product.  Output is in ``[-1, 1]`` per channel.
    """

    def __init__(self, levels=(8, 8, 5, 5, 5), eps=1e-3, autoscale=0.0,
                 autoscale_momentum=0.01):
        super().__init__()
        levels = tuple(int(l) for l in levels)
        if any(l < 2 for l in levels):
            raise ValueError("every FSQ level count must be >= 2")
        self.levels = levels
        self.eps = float(eps)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32),
                             persistent=False)
        basis = torch.cumprod(torch.tensor((1,) + levels[:-1], dtype=torch.long), dim=0)
        self.register_buffer("_basis", basis, persistent=False)
        self.register_buffer("_usage", torch.zeros(self.codebook_size), persistent=False)

        # SAMOISTNE SKALOWANIE WEJSCIA. `bound` liczy tanh(z)*half, gdzie dla 8
        # poziomow half = 3,4965. Zeby koszyki byly uzyte rownomiernie, tanh(z)
        # musi rozlozyc sie po (-1, 1), a to wymaga std(z) rzedu 1,5-2. Przy
        # std(z) = 0,3 mamy tanh(z) ~ z, zakres (-1,05; 1,05), i rundowanie
        # trafia w 3 koszyki z 8.
        #
        # Zmierzone na L0 biegu cbp (9600 sekwencji, wagi z epoki 1): kanaly
        # DOTYKAJA 7-8 koszykow z 8, ale rozklad jest spiczasty -- kanal 0 ma
        # 70,8% masy w jednym koszyku, kanal 4 ma 73,4%. Entropia na kanal
        # wychodzi 1,04-1,99 bitu z dostepnych 3, a laczna 7,96 z 14. I to
        # dotyczy poziomu, ktory NOSI CALA PREDYKCJE (wklad +0,0542 wobec
        # +0,0012 dla L2 przy SE 0,00299).
        #
        # Skalowanie jest SAMOISTNE: dzielimy przez biezace odchylenie (EMA pod
        # `no_grad`, per kanal) i mnozymy do celu. Zaden nowy hiperparametr poza
        # samym celem, i zadne zalozenie o rozkladzie -- jesli enkoder zmieni
        # skale w trakcie treningu, normalizacja podazy za nim.
        #
        # `autoscale` = docelowe odchylenie wejscia do `tanh`. 0.0 wylacza.
        self.autoscale = float(autoscale)
        self.autoscale_momentum = float(autoscale_momentum)
        if self.autoscale > 0.0:
            self.register_buffer("_in_std", torch.ones(len(levels)))
            self.register_buffer("_in_seen", torch.zeros(()))

    @property
    def dim(self):
        return len(self.levels)

    @property
    def codebook_size(self):
        size = 1
        for l in self.levels:
            size *= l
        return size

    def bound(self, z):
        """Squash to the open interval covering ``levels`` integers, centred.

        The ``offset``/``shift`` correction handles even level counts: without
        it an even grid has no bin centred on zero and ``tanh``'s most linear
        region straddles two bins, which biases the quantiser at initialisation.
        """
        levels = self._levels.to(z.dtype)
        half = (levels - 1) * (1 - self.eps) / 2
        offset = torch.where(levels % 2 == 0, torch.full_like(levels, 0.5),
                             torch.zeros_like(levels))
        shift = torch.atanh(offset / half)
        return torch.tanh(z + shift) * half - offset

    def _skaluj(self, z):
        """Normalizacja wejscia do docelowego odchylenia, per kanal.

        Statystyka liczona pod `no_grad` i aktualizowana wykladniczo, wiec w
        ewaluacji jest deterministyczna i nie zalezy od skladu batcha -- to
        wazne, bo BatchNorm w gardle sprawial wczesniej, ze kod FSQ nie byl
        funkcja sekwencji (70,4% slotow L0 zmienialo sie ze skladem batcha).
        """
        if self.autoscale <= 0.0:
            return z
        if self.training:
            with torch.no_grad():
                osie = tuple(range(z.dim() - 1))
                s = z.detach().std(dim=osie).clamp_min(1e-4)
                if float(self._in_seen) < 1.0:
                    self._in_std.copy_(s)
                else:
                    m = self.autoscale_momentum
                    self._in_std.mul_(1.0 - m).add_(s, alpha=m)
                self._in_seen += 1.0
        return z * (self.autoscale / self._in_std.to(z.dtype).clamp_min(1e-4))

    def quantise(self, z):
        """``z`` last-dim ``dim`` -> quantised values in ``[-1, 1]``."""
        q = round_ste(self.bound(self._skaluj(z)))
        half_width = (self._levels.to(z.dtype) // 2).clamp_min(1)
        return q / half_width

    def codes_to_indices(self, q):
        """Quantised values in ``[-1, 1]`` -> a single integer per position."""
        half_width = (self._levels.to(q.dtype) // 2).clamp_min(1)
        digits = (q * half_width).round().long() + (self._levels.long() // 2)
        digits = digits.clamp(torch.zeros_like(self._levels.long()), self._levels.long() - 1)
        return (digits * self._basis).sum(dim=-1)

    def indices_to_codes(self, idx):
        """Inverse of :meth:`codes_to_indices`."""
        levels = self._levels.long()
        digits = (idx.unsqueeze(-1) // self._basis) % levels
        half_width = (self._levels // 2).clamp_min(1)
        return (digits.to(self._levels.dtype) - (levels // 2).to(self._levels.dtype)) / half_width

    def forward(self, z):
        """``z`` of shape ``[B, C, L]`` with ``C == dim``.

        Returns ``(quantised [B, C, L], indices [B, L])``.
        """
        zt = z.transpose(1, 2)
        q = self.quantise(zt)
        idx = self.codes_to_indices(q.detach())
        # `_force_track` exists because the diagnostic was unmeasurable: the
        # buffer is `persistent=False` so a checkpoint restores it as zeros, and
        # tracking was gated on `self.training` while every analysis runs under
        # `eval()`. `fsq_grid_used` therefore read [0.0, 0.0, 0.0] -- which looks
        # like total codebook collapse and is in fact no measurement at all.
        if self.training or getattr(self, "_force_track", False):
            self._track(idx)
        return q.transpose(1, 2), idx

    @torch.no_grad()
    def _track(self, idx, decay=0.99):
        counts = torch.bincount(idx.reshape(-1), minlength=self.codebook_size).to(self._usage.dtype)
        self._usage.mul_(decay).add_(counts, alpha=1 - decay)

    @torch.no_grad()
    def utilisation(self):
        """``(perplexity, fraction of grid points ever used)``.

        FSQ cannot collapse the way a learned codebook can, but the *encoder*
        can still park every sequence in one corner of the grid — this is the
        measurement that catches it.
        """
        total = self._usage.sum()
        if float(total) <= 0:
            return 0.0, 0.0
        p = self._usage / total
        perplexity = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
        return perplexity, float((self._usage > 0).float().mean())


class ResidualFSQ(nn.Module):
    """A stack of FSQ levels, each quantising what the previous ones missed.

    Ordered **coarse -> fine**, matching HypVQNet's top-down VQ: the first level
    sees a clean signal and states the most global thing about the promoter,
    later levels only encode residual.  Bottom-up ordering was measured in the
    parent framework to invert the hierarchy outright, leaving the coarse
    codebooks quantising the high-frequency residual — i.e. noise.
    """

    def __init__(self, dim, levels_per_stage=((7, 5, 5), (8, 5, 5, 5), (8, 8, 5, 5, 5))):
        super().__init__()
        self.stages = nn.ModuleList(FSQ(levels) for levels in levels_per_stage)
        self.project_in = nn.ModuleList(nn.Conv1d(dim, s.dim, 1) for s in self.stages)
        self.project_out = nn.ModuleList(nn.Conv1d(s.dim, dim, 1) for s in self.stages)
        self.dim = int(dim)

    def __len__(self):
        return len(self.stages)

    @property
    def codebook_sizes(self):
        return [s.codebook_size for s in self.stages]

    def forward(self, z):
        """``z`` ``[B, C, L]`` -> ``(sum of stage outputs, per-stage codes, indices)``."""
        residual = z
        outs, codes, indices = [], [], []
        for stage, proj_in, proj_out in zip(self.stages, self.project_in, self.project_out):
            q, idx = stage(proj_in(residual))
            out = proj_out(q)
            outs.append(out)
            codes.append(q)
            indices.append(idx)
            residual = residual - out
        return sum(outs), codes, indices

    def decode(self, codes):
        return sum(proj(c) for proj, c in zip(self.project_out, codes))

    @torch.no_grad()
    def utilisation(self):
        return [s.utilisation() for s in self.stages]

    def measure(self, enabled=True, reset=True):
        """Count grid usage outside training, for analysis under ``eval()``.

        Off by default so an evaluation pass cannot silently mix the validation
        distribution into a statistic read as training behaviour. ``reset``
        clears the EMA first, so the number describes THIS pass rather than
        whatever the decay still carries from training.
        """
        for s in self.stages:
            s._force_track = bool(enabled)
            if reset:
                s._usage.zero_()
        return self


__all__ = ["FSQ", "ResidualFSQ", "round_ste"]
