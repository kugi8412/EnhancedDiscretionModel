import math

import torch
import torch.nn as nn


class AtencjaMotywowGlobalna(nn.Module):
    """Zapytania z najgrubszej mapy, klucze i wartosci z pelnego obłozenia.

    `dim_mapy` -- szerokosc mapy idacej do L0 (zapytania i wyjscie),
    `dim_occ`  -- szerokosc obłozenia JASPAR (klucze, wartosci),
    `dim_g`    -- szerokosc wektora grafu; 0 wylacza token globalny.
    """

    def __init__(self, dim_mapy, dim_occ, dim_g=0, glowic=4, dim_glowy=32,
                 dropout=0.0):
        super().__init__()
        self.h, self.dk = int(glowic), int(dim_glowy)
        w = self.h * self.dk
        c = int(dim_mapy)
        self.q = nn.Conv1d(c, w, 1, bias=False)
        self.k = nn.Conv1d(int(dim_occ), w, 1, bias=False)
        self.v = nn.Conv1d(int(dim_occ), w, 1, bias=False)
        self.o = nn.Conv1d(w, c, 1, bias=False)
        self.dim_g = int(dim_g)
        if self.dim_g:
            self.kg = nn.Linear(self.dim_g, w, bias=False)
            self.vg = nn.Linear(self.dim_g, w, bias=False)
        else:
            self.kg = self.vg = None
        self.drop = nn.Dropout(dropout)
        # SCALENIE STRUKTURALNE: mapa i atencja razem, potem powrot do `c`.
        # Mapa nie ma drogi do L0 omijajacej ten splot.
        self.scal = nn.Conv1d(2 * c, c, 1)
        self._c = c
        self._diag = {}

    def forward(self, h, occ, g=None):
        """`h`: [B, C, W] najgrubsza mapa. `occ`: [B, Mo, L]. `g`: [B, Cg]."""
        B, _, W = h.shape
        q = self.q(h).view(B, self.h, self.dk, W).transpose(-1, -2)
        k = self.k(occ)
        v = self.v(occ)
        L = k.shape[-1]
        k = k.view(B, self.h, self.dk, L).transpose(-1, -2)
        v = v.view(B, self.h, self.dk, L).transpose(-1, -2)
        if self.kg is not None and g is not None:
            k = torch.cat([k, self.kg(g).view(B, self.h, 1, self.dk)], dim=-2)
            v = torch.cat([v, self.vg(g).view(B, self.h, 1, self.dk)], dim=-2)
        w = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(self.dk), -1)
        y = (self.drop(w) @ v).transpose(-1, -2).reshape(B, -1, W)
        y = self.o(y)
        # DOPASOWANIE RMS PER PROBKA. Forma `cat([h, y])` zaklada, ze `y`
        # jest porownywalne z `h`; przy BatchNormie w srodku to zalozenie
        # jest FALSZYWE (zmierzone w tym repo: modul kontekstu dawal RMS
        # 11,68 przy wejsciu 0,92, dominacja 29,79). Per PROBKA, bo
        # `encode_codes` musi zostac niezalezne od skladu batcha -- inaczej
        # umowa U2 dyfuzji pada.
        rh = h.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        ry = y.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        y = y * (rh / ry)
        wy = self.scal(torch.cat([h, y], dim=1))
        with torch.no_grad():
            _e = -(w * (w + 1e-12).log2()).sum(-1).mean()
            _wm = self.scal.weight[:, :self._c].norm()
            _wa = self.scal.weight[:, self._c:].norm()
            self._diag = {
                "mot_l0_H": round(float(_e), 3),
                "mot_l0_H_max": round(math.log2(w.shape[-1]), 3),
                "mot_l0_udzial": round(float(_wa / _wm.clamp_min(1e-9)), 4),
                "mot_l0_dom": round(float(
                    y.pow(2).mean().sqrt() / rh.mean().clamp_min(1e-9)), 4),
            }
        return wy

    def raport(self):
        return dict(self._diag)

    def extra_repr(self):
        return "glowic=%d, dim_glowy=%d, token_globalny=%s" % (
            self.h, self.dk, bool(self.dim_g))
