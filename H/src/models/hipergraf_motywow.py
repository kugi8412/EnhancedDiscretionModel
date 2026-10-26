# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperbolic import expmap0, logmap0, lorentz_centroid


class HipergrafWspolwystepowania(nn.Module):
    """Splot hipergrafowy: motyw -> hiperkrawedz -> motyw.

    Hiperkrawedz `e` to okno sekwencji; jej wezly to motywy obecne w tym
    oknie. Macierz incydencji `H[b, m, e]` jest MIEKKA (obłozenie), nie
    binarna -- prog zamienilby ciagly sygnal na decyzje, a w tym repo
    zmierzone jest, ze progi na tych danych sie nie przenosza.

    Postac (HGNN, Feng i in.):  X' = D_v^-1 H W D_e^-1 H^T X Theta
    """

    def __init__(self, n_motywow, dim, okien=15, hiper=False, rundy=2,
                 topk=8):
        super().__init__()
        self.M = int(n_motywow)
        self.okien = int(okien)
        self.hiper = bool(hiper)
        self.rundy = int(rundy)
        self.topk = int(topk)
        self.wezel = nn.Linear(dim, dim, bias=False)
        self.kraw = nn.Linear(dim, dim, bias=False)
        self.waga_kraw = nn.Parameter(torch.ones(self.okien))
        self.norma = nn.LayerNorm(dim)
        self._diag = {}

    def _incydencja(self, occ):
        """`occ` [B, M, L] -> `H` [B, M, E]: maksimum w oknie, potem TOP-K.

        MAKSIMUM, nie srednia: o obecnosci motywu w oknie decyduje to, czy
        ISTNIEJE mocne miejsce wiazania, a nie srednie powinowactwo --
        zmierzone 0,329 wobec 0,004 dla tych samych profili.

        TOP-K, NIE PROG. Zmierzone na 256 loci walidacji (obłozenie
        log-szans: sr -17,88, sd 12,28, tylko 1,63 % dodatnich; po maksimum
        w oknie sr -6,40, sd 7,54):

            prog 0 (relu)  ->  52,0 motywow na krawedz
            prog 2         ->  23,7
            prog 4         ->   8,1
            prog 6         ->   2,4   <- przestaje byc hipergrafem
            top-k = 8      ->   8     (prog efektywny 3,47; 32 % masy)

        Prog jest zly nie z estetyki: rozklad log-szans zalezy od
        SZEROKOSCI i informacyjnosci profilu, a profile maja 7-21 bp, wiec
        jedna liczba nie moze byc wlasciwa dla wszystkich 296. Top-k daje
        krawedzie rozmiaru `k` niezaleznie od skali profilu, a `k = 8` to
        skala enhanceosomu.

        Wagi zostaja MIEKKIE (wartosc obłozenia), nie binarne -- binaryzacja
        zamienilaby ciagly sygnal na decyzje, a progi na tych danych sa
        zmierzone jako nieprzenoszace sie.
        """
        B, M, L = occ.shape
        kr = max(L // self.okien, 1)
        o = occ[..., :kr * self.okien].reshape(B, M, self.okien, kr).amax(-1)
        if self.topk <= 0 or self.topk >= M:
            return F.relu(o)
        v, idx = o.topk(self.topk, dim=1)               # [B, k, E]
        H = torch.zeros_like(o)
        H.scatter_(1, idx, F.relu(v))
        return H

    def forward(self, X, occ):
        """`X` [B, M, d] cechy motywow, `occ` [B, M, L] obłozenie."""
        H = self._incydencja(occ)
        w = F.softplus(self.waga_kraw).view(1, 1, -1)
        Hw = H * w
        dv = Hw.sum(-1, keepdim=True).clamp_min(1e-6)    # stopien wezla
        de = Hw.sum(1, keepdim=True).clamp_min(1e-6)     # stopien krawedzi
        for _ in range(self.rundy):
            # WEZEL -> HIPERKRAWEDZ: agregacja po motywach nalezacych do e
            E = torch.einsum("bme,bmd->bed", Hw, X) / de.transpose(1, 2)
            E = self.kraw(E)
            if self.hiper:
                # AGREGACJA GEODEZYJNA. Punkty na hiperboloidzie, centroid
                # Lorentza; porownywana z euklidesowa przez `hg_rozn`.
                _p = expmap0(E)
                _c = lorentz_centroid(_p, dim=-2)
                E_h = logmap0(_c).unsqueeze(-2).expand_as(E)
                with torch.no_grad():
                    # ROZDZIELENIE KATA I PROMIENIA -- bez tego `hg_rozn`
                    # jest nieinterpretowalne. Zapisany DOWOD z odczytu kodu
                    # mowi, ze KIERUNEK centroidu Lorentza jest DOKLADNIE
                    # znormalizowana euklidesowa srednia wazona; cala roznica
                    # moze wiec siedziec w PROMIENIU, bo `logmap0` skaluje
                    # przez `asinh(|v|)/|v|`, czyli kompresuje nieliniowo.
                    # Kompresja radialna to nie krzywizna -- to samo zrobilby
                    # LayerNorm. Pierwsza wersja tego falsyfikatora dala
                    # 0,4253 na prawdziwym obłozeniu i wygladalo to jak
                    # "krzywizna wreszcie dziala".
                    _e = E.mean(-2, keepdim=True).expand_as(E)
                    _a = F.normalize(E_h.reshape(-1, E_h.shape[-1]), dim=-1)
                    _b = F.normalize(_e.reshape(-1, _e.shape[-1]), dim=-1)
                    self._diag["hg_kat_cos"] = round(float(
                        (_a * _b).sum(-1).mean()), 6)
                    self._diag["hg_prom_iloraz"] = round(float(
                        E_h.norm(dim=-1).mean()
                        / _e.norm(dim=-1).mean().clamp_min(1e-9)), 4)
                    self._diag["hg_rozn"] = round(float(
                        (E_h - _e).norm() / _e.norm().clamp_min(1e-9)), 6)
                E = E + E_h - E.mean(-2, keepdim=True)
            # HIPERKRAWEDZ -> WEZEL
            X2 = torch.einsum("bme,bed->bmd", Hw, E) / dv
            X = self.norma(X + self.wezel(X2))
        with torch.no_grad():
            self._diag.update({
                "hg_stopien_w": round(float(dv.mean()), 4),
                "hg_stopien_e": round(float(de.mean()), 4),
                # ILE motywow na hiperkrawedz -- to jest "rozmiar
                # enhanceosomu" widziany przez modul. 1-2 znaczy, ze
                # hipergraf zdegenerowal sie do zwyklego grafu.
                "hg_na_kraw": round(float((H > 0.1).float().sum(1).mean()), 2),
                "hg_waga_kraw": round(float(F.softplus(
                    self.waga_kraw).mean()), 4),
            })
        return X

    def raport(self):
        return dict(self._diag)


class AtencjaPodprzestrzenMotyw(nn.Module):
    """Podprzestrzen predykcyjna ZADAJE PYTANIE motywom. Osobno Dev i Hk.

    `dim_pod` -- szerokosc wektora podprzestrzeni (rzut latentu),
    `dim_mot` -- szerokosc reprezentacji motywu po hipergrafie.
    """

    def __init__(self, dim_pod, dim_mot, dim_wy=32, glowic=4, dim_glowy=32):
        super().__init__()
        self.h, self.dk = int(glowic), int(dim_glowy)
        w = self.h * self.dk
        self.q = nn.Linear(int(dim_pod), w, bias=False)
        self.k = nn.Linear(int(dim_mot), w, bias=False)
        self.v = nn.Linear(int(dim_mot), w, bias=False)
        self.wy = nn.Linear(w, int(dim_wy), bias=False)
        self._diag = {}

    def forward(self, pod, X):
        """`pod` [B, dim_pod] jedna podprzestrzen, `X` [B, M, dim_mot]."""
        B = pod.shape[0]
        q = self.q(pod).view(B, self.h, 1, self.dk)
        k = self.k(X).view(B, -1, self.h, self.dk).transpose(1, 2)
        v = self.v(X).view(B, -1, self.h, self.dk).transpose(1, 2)
        a = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(self.dk), -1)
        o = (a @ v).reshape(B, -1)
        with torch.no_grad():
            _e = -(a * (a + 1e-12).log2()).sum(-1).mean()
            self._diag = {
                # Sufit = log2(M), czyli atencja JEDNOSTAJNA: podprzestrzen
                # pyta o wszystkie motywy jednakowo, wiec dostaje ich
                # srednia i mechanizm nie wybiera niczego.
                "pm_H": round(float(_e), 3),
                "pm_H_max": round(math.log2(X.shape[1]), 3),
                "pm_szczyt": round(float(a.amax(-1).mean()), 4),
            }
        return self.wy(o)

    def raport(self):
        return dict(self._diag)
