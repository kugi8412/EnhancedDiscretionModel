# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F


class ReprezentacjaPar(nn.Module):
    """Mapa okien [B, C, W] -> tensor par [B, W, W, c].

    `Z_ij` powstaje z iloczynu zewnetrznego rzutow okien plus osadzenie
    ODLEGLOSCI |i-j|. Osadzenie odleglosci jest tu konieczne, nie ozdobne:
    zmierzone w tym repo, ze niosa **tylko pary blizkie** (0-33 bp), wiec
    modul musi miec czym odroznic blisko od daleka, zamiast uczyc sie tego
    z samej tresci.
    """

    def __init__(self, dim_we, dim_par=32, max_okien=32):
        super().__init__()
        self.dim_par = int(dim_par)
        self.lewy = nn.Conv1d(int(dim_we), self.dim_par, 1, bias=False)
        self.prawy = nn.Conv1d(int(dim_we), self.dim_par, 1, bias=False)
        # |i-j| ma W wartosci; tabela z zapasem, zeby W dalo sie zmienic
        self.odl = nn.Embedding(int(max_okien), self.dim_par)
        nn.init.normal_(self.odl.weight, std=0.02)
        self.norma = nn.LayerNorm(self.dim_par)

    def forward(self, h):
        B, _, W = h.shape
        a = self.lewy(h).permute(0, 2, 1)             # [B, W, c]
        b = self.prawy(h).permute(0, 2, 1)
        Z = a.unsqueeze(2) * b.unsqueeze(1)           # [B, W, W, c] RANGI 1
        d = (torch.arange(W, device=h.device).unsqueeze(1)
             - torch.arange(W, device=h.device).unsqueeze(0)).abs()
        Z = Z + self.odl(d).unsqueeze(0)
        return self.norma(Z)


class AktualizacjaTrojkatna(nn.Module):
    """Triangle multiplicative update z AF2, wychodzaca albo wchodzaca.

    wychodzaca:  out_ij = sum_k a_ik * b_jk
    wchodzaca:   out_ij = sum_k a_ki * b_kj

    To jest cala tresc "domkniecia trojkata": nowa relacja miedzy `i` i `j`
    jest liczona przez WSZYSTKIE mozliwe trzecie okna `k`.
    """

    def __init__(self, dim_par, ukryty=None, wychodzaca=True, gain_wy=0.1):
        super().__init__()
        c = int(dim_par)
        u = int(ukryty or c)
        self.wychodzaca = bool(wychodzaca)
        self.norma_we = nn.LayerNorm(c)
        self.proj_a = nn.Linear(c, u, bias=False)
        self.proj_b = nn.Linear(c, u, bias=False)
        self.br_a = nn.Linear(c, u)
        self.br_b = nn.Linear(c, u)
        self.norma_sr = nn.LayerNorm(u)
        self.proj_wy = nn.Linear(u, c, bias=False)
        self.br_wy = nn.Linear(c, c)
        # NIEZEROWA, mala. AF2 daje tu zero; zero nie przepuszcza gradientu
        # w gore i gasi wlasne zrodlo -- blad popelniony w tym repo 4x.
        nn.init.xavier_uniform_(self.proj_wy.weight, gain=float(gain_wy))
        self._diag = {}

    def forward(self, Z):
        z = self.norma_we(Z)
        a = torch.sigmoid(self.br_a(z)) * self.proj_a(z)   # [B, W, W, u]
        b = torch.sigmoid(self.br_b(z)) * self.proj_b(z)
        if self.wychodzaca:
            sr = torch.einsum("biku,bjku->biju", a, b)
        else:
            sr = torch.einsum("bkiu,bkju->biju", a, b)
        out = torch.sigmoid(self.br_wy(z)) * self.proj_wy(self.norma_sr(sr))
        with torch.no_grad():
            self._diag = {
                "wklad": round(float(out.pow(2).mean().sqrt()
                                     / Z.pow(2).mean().sqrt().clamp_min(1e-9)), 4),
            }
        return Z + out


class AtencjaTrojkatna(nn.Module):
    """Triangle attention wokol wezla poczatkowego.

    Dla pary (i, j) atencja idzie po `k`, a bias pochodzi z pary (j, k) --
    czyli waga trzeciego okna zalezy od tego, jak `k` odnosi sie do `j`.
    Pamiec to [B, h, W, W, W]; przy W = 15 to 3375 komorek na glowe, wiec
    koszt jest zaniedbywalny i nie potrzeba porcjowania.
    """

    def __init__(self, dim_par, glowic=4, dim_glowy=16, gain_wy=0.1):
        super().__init__()
        c = int(dim_par)
        self.h, self.dk = int(glowic), int(dim_glowy)
        w = self.h * self.dk
        self.norma = nn.LayerNorm(c)
        self.q = nn.Linear(c, w, bias=False)
        self.k = nn.Linear(c, w, bias=False)
        self.v = nn.Linear(c, w, bias=False)
        self.bias = nn.Linear(c, self.h, bias=False)
        self.brama = nn.Linear(c, w)
        self.wy = nn.Linear(w, c, bias=False)
        nn.init.xavier_uniform_(self.wy.weight, gain=float(gain_wy))
        self._diag = {}

    def forward(self, Z):
        B, W, _, _ = Z.shape
        z = self.norma(Z)
        q = self.q(z).view(B, W, W, self.h, self.dk)
        k = self.k(z).view(B, W, W, self.h, self.dk)
        v = self.v(z).view(B, W, W, self.h, self.dk)
        bi = self.bias(z).permute(0, 3, 1, 2)                # [B, h, j, k]
        # logit(i,j,k) = <q_ij, k_ik> / sqrt(dk) + bias_jk
        lg = torch.einsum("bijhd,bikhd->bhijk", q, k) / math.sqrt(self.dk)
        lg = lg + bi.unsqueeze(2)
        w = torch.softmax(lg, dim=-1)
        o = torch.einsum("bhijk,bikhd->bijhd", w, v).reshape(B, W, W, -1)
        # BRAMA dziala na wyjsciu glowic (szerokosc `w`), PRZED rzutowaniem
        # do `c` -- tak jak w AF2. Odwrotna kolejnosc nie zlozy sie wymiarowo.
        o = self.wy(torch.sigmoid(self.brama(z)) * o)
        with torch.no_grad():
            e = -(w * (w + 1e-12).log2()).sum(-1).mean()
            self._diag = {"troj_H": round(float(e), 3),
                          "troj_H_max": round(math.log2(W), 3)}
        return Z + o



def _przesun_wiersze(t, s):
    """`t[:, i]` -> `t[:, i+s]`, brzegi wypelnione zerem."""
    if s == 0:
        return t
    z = torch.zeros_like(t)
    if s > 0:
        z[:, :-s] = t[:, s:]
    else:
        z[:, -s:] = t[:, :s]
    return z


def _transponuj_pasmo(Z, R):
    """Transpozycja we wspolrzednych pasmowych: `Zt[i, o] = Z[i+o, -o]`.

    Potrzebna, zeby aktualizacja WCHODZACA nie wymagala O^2 przesuniec:
    `sum_k a_ki b_kj` to `sum_k (a^T)_ik (b^T)_jk`, czyli ten sam jadro
    wychodzace na transponowanych czynnikach.
    """
    O = 2 * R + 1
    wy = torch.zeros_like(Z)
    for oi in range(O):
        o = oi - R
        wy[:, :, oi] = _przesun_wiersze(Z[:, :, O - 1 - oi], o)
    return wy


class ReprezentacjaParPasmowa(nn.Module):
    """Mapa zasad [B, C, L] -> pary w pasmie [B, L, 2R+1, c].

    `Z[i, o]` opisuje pare zasad `(i, i+o)` dla `|o| <= R`.
    """

    def __init__(self, dim_we, dim_par=32, pasmo=16):
        super().__init__()
        self.R = int(pasmo)
        self.dim_par = int(dim_par)
        self.lewy = nn.Conv1d(int(dim_we), self.dim_par, 1, bias=False)
        self.prawy = nn.Conv1d(int(dim_we), self.dim_par, 1, bias=False)
        self.odl = nn.Embedding(2 * self.R + 1, self.dim_par)
        nn.init.normal_(self.odl.weight, std=0.02)
        self.norma = nn.LayerNorm(self.dim_par)

    def forward(self, h):
        R, O = self.R, 2 * self.R + 1
        a = self.lewy(h).permute(0, 2, 1)               # [B, L, c]
        b = self.prawy(h).permute(0, 2, 1)
        # b[i+o] dla kazdego o -- 33 przesuniecia wierszy, nie L x L
        bs = torch.stack([_przesun_wiersze(b, oi - R) for oi in range(O)], 2)
        Z = a.unsqueeze(2) * bs                         # [B, L, O, c]
        Z = Z + self.odl.weight.view(1, 1, O, -1)
        return self.norma(Z)


class AktualizacjaTrojkatnaPasmowa(nn.Module):
    """Triangle multiplicative update ograniczona do pasma.

    wychodzaca:  out[i, o] = sum_p a[i, p] * b[i+o, p-o]
    (bo `k = i+p`, a para `(j, k)` ma przesuniecie `k-j = p-o`)

    Sumowanie po `p` biegnie tylko tam, gdzie OBA wpisy leza w pasmie,
    czyli `p in [max(-R, o-R), min(R, o+R)]` -- dla `o = 0` to pelne 2R+1
    trzecich zasad, dla `|o| = R` tylko jedna. To jest geometrycznie
    poprawne: im dalsze pary, tym mniej wspolnych trzecich zasad.
    """

    def __init__(self, dim_par, pasmo, ukryty=None, wychodzaca=True,
                 gain_wy=0.1):
        super().__init__()
        c = int(dim_par)
        u = int(ukryty or c)
        self.R = int(pasmo)
        self.wychodzaca = bool(wychodzaca)
        self.norma_we = nn.LayerNorm(c)
        self.proj_a = nn.Linear(c, u, bias=False)
        self.proj_b = nn.Linear(c, u, bias=False)
        self.br_a = nn.Linear(c, u)
        self.br_b = nn.Linear(c, u)
        self.norma_sr = nn.LayerNorm(u)
        self.proj_wy = nn.Linear(u, c, bias=False)
        self.br_wy = nn.Linear(c, c)
        nn.init.xavier_uniform_(self.proj_wy.weight, gain=float(gain_wy))
        self._diag = {}

    def forward(self, Z):
        R, O = self.R, 2 * self.R + 1
        z = self.norma_we(Z)
        a = torch.sigmoid(self.br_a(z)) * self.proj_a(z)
        b = torch.sigmoid(self.br_b(z)) * self.proj_b(z)
        if not self.wychodzaca:
            a, b = _transponuj_pasmo(a, R), _transponuj_pasmo(b, R)
        czesci = []
        for oi in range(O):
            o = oi - R
            p0, p1 = max(-R, o - R), min(R, o + R)
            ps = slice(p0 + R, p1 + R + 1)
            qs = slice(p0 - o + R, p1 - o + R + 1)
            # PRZESUNIECIE NA WYCINKU, nie na calym tensorze. Wersja
            # `_przesun_wiersze(b, o)[:, :, qs]` alokowala PELNY tensor
            # [B, L, O, u] w kazdej z 2R+1 iteracji i trzymala go na tasmie
            # autograda: zmierzone 18,05 GiB szczytu przy batchu 96, czyli
            # +7,5 GiB nad baza za 40 802 parametry. Przesuwanie wycinka
            # zbija to do sumy szerokosci wycinkow.
            bs = _przesun_wiersze(b[:, :, qs], o)
            czesci.append((a[:, :, ps] * bs).sum(2))
        sr = torch.stack(czesci, 2)                     # [B, L, O, u]
        out = torch.sigmoid(self.br_wy(z)) * self.proj_wy(self.norma_sr(sr))
        with torch.no_grad():
            self._diag = {"wklad": round(float(
                out.pow(2).mean().sqrt()
                / Z.pow(2).mean().sqrt().clamp_min(1e-9)), 4)}
        return Z + out


class AtencjaZBiasemPar(nn.Module):

    def __init__(self, dim_we, dim_par, pasmo, glowic=4, dim_glowy=24,
                 gain_wy=0.1, udzial=0.0):
        super().__init__()
        self.udzial = float(udzial)
        self.R = int(pasmo)
        self.h, self.dk = int(glowic), int(dim_glowy)
        w = self.h * self.dk
        self.norma = nn.LayerNorm(int(dim_we))
        self.q = nn.Linear(int(dim_we), w, bias=False)
        self.k = nn.Linear(int(dim_we), w, bias=False)
        self.v = nn.Linear(int(dim_we), w, bias=False)
        self.z_bias = nn.Linear(int(dim_par), self.h, bias=False)
        self.poza = nn.Parameter(torch.zeros(self.h))
        self.wy = nn.Linear(w, int(dim_we), bias=False)
        nn.init.xavier_uniform_(self.wy.weight, gain=float(gain_wy))
        self._diag = {}

    def _rozwin_bias(self, bz, L):
        """[B, L, 2R+1, h] w pasmie -> [B, h, L, L] pelny, poza pasmem stala."""
        B, _, O, H = bz.shape
        R = self.R
        pel = self.poza.view(1, H, 1, 1).expand(B, H, L, L).clone()
        idx_i = torch.arange(L, device=bz.device)
        for oi in range(O):
            o = oi - R
            i0, i1 = max(0, -o), min(L, L - o)
            if i1 <= i0:
                continue
            ii = idx_i[i0:i1]
            pel[:, :, ii, ii + o] = bz[:, i0:i1, oi, :].permute(0, 2, 1)
        return pel

    def forward(self, h1, Z):
        """`h1`: [B, C, L] strumien sekwencji. `Z`: [B, L, 2R+1, c] pary."""
        B, C, L = h1.shape
        x = self.norma(h1.permute(0, 2, 1))                  # [B, L, C]
        q = self.q(x).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.dk).transpose(1, 2)
        bz = self.z_bias(Z)                                  # [B, L, O, h]
        bias = self._rozwin_bias(bz, L)                      # [B, h, L, L]
        lg = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk) + bias
        w = torch.softmax(lg, dim=-1)
        o = (w @ v).transpose(1, 2).reshape(B, L, -1)
        o = self.wy(o).permute(0, 2, 1)
        with torch.no_grad():
            # SUROWA dominacja, liczona PRZED tlumieniem. Po tlumieniu iloraz
            # rowna sie DOKLADNIE `udzial`, czyli obserwabla zamienilaby sie
            # w odczyt pokretla -- ta sama tautologia co `attn_H` celujace
            # w `cel_entropii`. Surowa mowi, ile modul CHCE wniesc, wiec
            # falsyfikator zyje takze przy wlaczonym tlumiku: jesli rosnie
            # mimo tlumienia, modul nadal probuje nadpisywac mape.
            _dom_sur = float(o.pow(2).mean().sqrt()
                             / h1.pow(2).mean().sqrt().clamp_min(1e-9))
        if self.udzial > 0.0:
            # RMS PER PROBKA (dim=(1,2), keepdim) -- nie po batchu.
            _rh = h1.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
            _ro = o.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-9)
            o = o * (self.udzial * _rh / _ro)
        with torch.no_grad():
            e = -(w * (w + 1e-12).log2()).sum(-1).mean()
            self._diag = {
                "seq_H": round(float(e), 3),
                "seq_H_max": round(math.log2(L), 3),
                # UDZIAL BIASU: jak mocno tensor par przestawia atencje
                # wobec samej tresci sekwencji. Zero = pary nie wplywaja
                # na czytanie sekwencji i ten kierunek jest martwy.
                "seq_bias_rms": round(float(bias.pow(2).mean().sqrt()), 4),
                "seq_poza": round(float(self.poza.abs().mean()), 4),
                # `seq_dom` -- SUROWA, sprzed tlumienia: falsyfikator.
                # `seq_dom_ef` -- po tlumieniu: tyle realnie wchodzi do mapy.
                # Przy `udzial = 0` sa identyczne i log jest jak dotad.
                "seq_dom": round(_dom_sur, 4),
                "seq_dom_ef": round(float(
                    o.pow(2).mean().sqrt()
                    / h1.pow(2).mean().sqrt().clamp_min(1e-9)), 4),
            }
        return h1 + o

    def raport(self):
        return dict(self._diag)



class AtencjaTrojkatnaKrojona(nn.Module):
    """Triangle attention AF2 wokol wezla poczatkowego, krojona po `i`.

    Dla pary (i, j) atencja idzie po `k`, a bias pochodzi z pary (j, k) --
    czyli waga trzeciego wezla zalezy od tego, jak `k` odnosi sie do `j`.
    Nigdy nie materializuje [L, L, L].
    """

    def __init__(self, dim_par, glowic=4, dim_glowy=16, kawalek=64,
                 gain_wy=0.1):
        super().__init__()
        c = int(dim_par)
        self.h, self.dk = int(glowic), int(dim_glowy)
        self.kawalek = int(kawalek)
        w = self.h * self.dk
        self.norma = nn.LayerNorm(c)
        self.q = nn.Linear(c, w, bias=False)
        self.k = nn.Linear(c, w, bias=False)
        self.v = nn.Linear(c, w, bias=False)
        self.bias = nn.Linear(c, self.h, bias=False)
        self.brama = nn.Linear(c, w)
        self.wy = nn.Linear(w, c, bias=False)
        nn.init.xavier_uniform_(self.wy.weight, gain=float(gain_wy))
        self._diag = {}

    def forward(self, Z):
        """`Z`: [B, L, L, c] PELNY tensor par (nie pasmowy)."""
        B, L, _, _ = Z.shape
        z = self.norma(Z)
        q = self.q(z).view(B, L, L, self.h, self.dk)
        k = self.k(z).view(B, L, L, self.h, self.dk)
        v = self.v(z).view(B, L, L, self.h, self.dk)
        bi = self.bias(z).permute(0, 3, 1, 2)                # [B, h, j, k]
        czesci, ent = [], 0.0
        for a in range(0, L, self.kawalek):
            b = min(a + self.kawalek, L)
            lg = torch.einsum("bijhd,bikhd->bhijk",
                              q[:, a:b], k[:, a:b]) / math.sqrt(self.dk)
            lg = lg + bi.unsqueeze(2)
            w = torch.softmax(lg, dim=-1)
            o = torch.einsum("bhijk,bikhd->bijhd", w, v[:, a:b])
            czesci.append(o.reshape(B, b - a, L, -1))
            with torch.no_grad():
                ent += float(-(w * (w + 1e-12).log2()).sum(-1).mean()) * (b - a)
        o = torch.cat(czesci, dim=1)
        o = self.wy(torch.sigmoid(self.brama(z)) * o)
        with torch.no_grad():
            self._diag = {"tk_H": round(ent / max(L, 1), 3),
                          "tk_H_max": round(math.log2(L), 3),
                          "tk_dom": round(float(
                              o.pow(2).mean().sqrt()
                              / Z.pow(2).mean().sqrt().clamp_min(1e-9)), 4)}
        return Z + o

    def raport(self):
        return dict(self._diag)


class BlokTrojkatnyPelny(nn.Module):
    """Pelny tensor par: aktualizacje multiplikatywne + ATENCJA trojkatna.

    Na tym poziomie nie ma pasma, wiec `AktualizacjaTrojkatna` (wersja
    kwadratowa) dziala wprost -- suma po `k` biegnie po WSZYSTKICH
    pozycjach, czyli domkniecie jest globalne.
    """

    def __init__(self, dim_par, glowic=4, dim_glowy=16, kawalek=64):
        super().__init__()
        self.wych = AktualizacjaTrojkatna(dim_par, wychodzaca=True)
        self.wchodz = AktualizacjaTrojkatna(dim_par, wychodzaca=False)
        self.aten = AtencjaTrojkatnaKrojona(dim_par, glowic, dim_glowy,
                                            kawalek)
        self.mlp = nn.Sequential(nn.LayerNorm(dim_par),
                                 nn.Linear(dim_par, 2 * dim_par), nn.SiLU(),
                                 nn.Linear(2 * dim_par, dim_par))
        nn.init.xavier_uniform_(self.mlp[3].weight, gain=0.1)
        nn.init.zeros_(self.mlp[3].bias)

    def forward(self, Z):
        Z = self.wych(Z)
        Z = self.wchodz(Z)
        Z = self.aten(Z)
        return Z + self.mlp(Z)

    def raport(self):
        r = {"tp_wkl_wych": self.wych._diag.get("wklad"),
             "tp_wkl_wchodz": self.wchodz._diag.get("wklad")}
        r.update(self.aten.raport())
        return r


class BlokTrojkatnyPasmowy(nn.Module):
    """Wychodzaca + wchodzaca + MLP. BEZ atencji -- patrz arytmetyka wyzej."""

    def __init__(self, dim_par, pasmo):
        super().__init__()
        self.wych = AktualizacjaTrojkatnaPasmowa(dim_par, pasmo, wychodzaca=True)
        self.wchodz = AktualizacjaTrojkatnaPasmowa(dim_par, pasmo,
                                                   wychodzaca=False)
        self.mlp = nn.Sequential(nn.LayerNorm(dim_par),
                                 nn.Linear(dim_par, 2 * dim_par), nn.SiLU(),
                                 nn.Linear(2 * dim_par, dim_par))
        nn.init.xavier_uniform_(self.mlp[3].weight, gain=0.1)
        nn.init.zeros_(self.mlp[3].bias)

    def forward(self, Z):
        Z = self.wych(Z)
        Z = self.wchodz(Z)
        return Z + self.mlp(Z)

    def raport(self):
        return {"trojp_wkl_wych": self.wych._diag.get("wklad"),
                "trojp_wkl_wchodz": self.wchodz._diag.get("wklad")}


class TorTrojkatnyPasmowy(nn.Module):
    """Nukleotydowy tor 3D: pary zasad w pasmie -> trojkaty -> cechy na zasade.

    Zwraca `(cechy [B, dim_wy, L], pred_aux [B, 2])`. Wpiecie przez
    KONKATENACJE, nie przez skalarna brame -- uzasadnienie w naglowku pliku.
    """

    def __init__(self, dim_we, dim_par=24, pasmo=16, blokow=2, dim_wy=24,
                 seq_atencja=True, glowic=4, dim_glowy=24, checkpoint=True,
                 gruby_L=0, gruby_bloki=2, gruby_par=16, gruby_kawalek=64,
                 gruby_wy=16, brama_pozycyjna=False,
                 udzial_seq=0.0):
        super().__init__()
        self.R = int(pasmo)
        self.checkpoint = bool(checkpoint)
        # ---- POZIOM GRUBY: pelny tensor par + ATENCJA trojkatna AF2 ----
        # `gruby_L = 62` to poziom L1 drabiny, wiec pary i hierarchia
        # latentu siedza w tej samej siatce. Zmierzone: 2,08 GiB i 0,017 s
        # na warstwe przy batchu 96 -- +35 % epoki za PELNE, globalne
        # domkniecie trojkata na granulacji 4 bp.
        self.gruby_L = int(gruby_L or 0)
        if self.gruby_L > 0:
            self.gruby_pary = ReprezentacjaPar(dim_we, gruby_par,
                                               max_okien=self.gruby_L + 1)
            self.gruby = nn.ModuleList([
                BlokTrojkatnyPelny(gruby_par, glowic, 16, gruby_kawalek)
                for _ in range(int(gruby_bloki))])
            self.gruby_odczyt = nn.Conv1d(2 * gruby_par, int(gruby_wy), 1)
            self.gruby_norma = nn.BatchNorm1d(int(gruby_wy))
        else:
            self.gruby_pary = self.gruby = None
        self.pary = ReprezentacjaParPasmowa(dim_we, dim_par, pasmo)
        self.bloki = nn.ModuleList([BlokTrojkatnyPasmowy(dim_par, pasmo)
                                    for _ in range(int(blokow))])
        # DROGA POWROTNA 2D -> 1D, po KAZDYM bloku par. Bez niej tensor par
        # jest slepa uliczka: liczy sie i jest odczytywany poolingiem, ale
        # nie wplywa na to, JAK siec czyta sekwencje.
        self.seq_aten = nn.ModuleList([
            AtencjaZBiasemPar(dim_we, dim_par, pasmo, glowic,
                              dim_glowy, udzial=udzial_seq)
            for _ in range(int(blokow))]) if seq_atencja else None
        self.odczyt = nn.Conv1d(2 * dim_par, int(dim_wy), 1)
        self.norma_wy = nn.BatchNorm1d(int(dim_wy))
        self.brama_poz = None
        if brama_pozycyjna:
            self.brama_poz = nn.Conv1d(2 * dim_par, int(dim_we), 1)
            nn.init.zeros_(self.brama_poz.weight)
            nn.init.zeros_(self.brama_poz.bias)
        self.aux = nn.Sequential(nn.Linear(2 * dim_par, 64), nn.SiLU(),
                                 nn.Linear(64, 2))
        self._diag = {}

    def _blok(self, b, Z):
        if self.checkpoint and self.training and Z.requires_grad:
            return torch.utils.checkpoint.checkpoint(b, Z, use_reentrant=False)
        return b(Z)

    def forward(self, h):
        h1 = h
        L = h.shape[-1]
        gr = None
        if self.gruby is not None:
            hg = F.adaptive_avg_pool1d(h1, self.gruby_L)
            Zg = self.gruby_pary(hg)
            for b in self.gruby:
                Zg = (torch.utils.checkpoint.checkpoint(b, Zg,
                                                        use_reentrant=False)
                      if self.checkpoint and self.training and Zg.requires_grad
                      else b(Zg))
            stg = torch.cat([Zg.mean(dim=2), Zg.amax(dim=2)], dim=-1)
            gr = self.gruby_norma(self.gruby_odczyt(stg.permute(0, 2, 1)))
            gr = F.interpolate(gr, size=L, mode="linear", align_corners=False)
        Z = self.pary(h1)
        for i, b in enumerate(self.bloki):
            Z = self._blok(b, Z)
            if self.seq_aten is not None:
                # SEKWENCJA <- PARY: struktura przestawia atencje po
                # pozycjach, a nie tylko dokleja sie jako cecha.
                h1 = self.seq_aten[i](h1, Z)
        st = torch.cat([Z.mean(dim=2), Z.amax(dim=2)], dim=-1)   # [B, L, 2c]
        _stp = st.permute(0, 2, 1)
        if self.brama_poz is not None:
            # RANGA L, nie 1: modulacja zmienia sie wzdluz sekwencji.
            _g = torch.tanh(self.brama_poz(_stp))
            h1 = h1 * (1.0 + _g)
            with torch.no_grad():
                self._brama_amp = round(float(_g.abs().mean()), 5)
                # ROZKLAD WARIANCJI: ile modulacji zmienia sie MIEDZY
                # POZYCJAMI, a ile jest stala na sekwencje. Dla SE ta druga
                # czesc to 83-93 %; jesli tu wyjdzie podobnie, modul
                # zdegenerowal sie do bramy kanalowej i przewagi nie ma.
                _sp = _g.std(dim=-1).pow(2).mean()
                _sk = _g.mean(dim=-1).std(dim=0).pow(2).mean()
                self._brama_poz_udzial = round(float(
                    _sp / (_sp + _sk).clamp_min(1e-12)), 4)
        wy = self.norma_wy(self.odczyt(_stp))
        if gr is not None:
            wy = torch.cat([wy, gr], dim=1)
        aux = self.aux(st.mean(dim=1))
        with torch.no_grad():
            d = {"trojp_pasmo": 2 * self.R + 1,
                 "bp_amp": getattr(self, "_brama_amp", None),
                 "bp_udzial_poz": getattr(self, "_brama_poz_udzial", None),
                 "trojp_rms": round(float(Z.pow(2).mean().sqrt()), 4),
                 # ILE atencja par zmienila strumien sekwencji. Zero =
                 # droga powrotna martwa i architektura jest znowu
                 # jednokierunkowa, mimo ze modul istnieje.
                 "seq_zmiana": round(float(
                     (h1 - h).pow(2).mean().sqrt()
                     / h.pow(2).mean().sqrt().clamp_min(1e-9)), 4)}
            if self.gruby is not None:
                d["tk_L"] = self.gruby_L
                for i, b in enumerate(self.gruby):
                    for k, v in b.raport().items():
                        d["%s_g%d" % (k, i)] = v
            for i, b in enumerate(self.bloki):
                for k, v in b.raport().items():
                    d["%s_b%d" % (k, i)] = v
            if self.seq_aten is not None:
                for i, a in enumerate(self.seq_aten):
                    for k, v in a.raport().items():
                        d["%s_b%d" % (k, i)] = v
            self._diag = d
        return F.silu(wy), aux, h1

    def raport(self):
        return dict(self._diag)


class BlokTrojkatny(nn.Module):
    """Jeden blok: wychodzaca + wchodzaca + atencja + MLP na parach."""

    def __init__(self, dim_par, glowic=4, dim_glowy=16, atencja=True):
        super().__init__()
        self.wych = AktualizacjaTrojkatna(dim_par, wychodzaca=True)
        self.wchodz = AktualizacjaTrojkatna(dim_par, wychodzaca=False)
        self.aten = (AtencjaTrojkatna(dim_par, glowic, dim_glowy)
                     if atencja else None)
        self.mlp = nn.Sequential(nn.LayerNorm(dim_par),
                                 nn.Linear(dim_par, 2 * dim_par), nn.SiLU(),
                                 nn.Linear(2 * dim_par, dim_par))
        nn.init.xavier_uniform_(self.mlp[3].weight, gain=0.1)
        nn.init.zeros_(self.mlp[3].bias)

    def forward(self, Z):
        Z = self.wych(Z)
        Z = self.wchodz(Z)
        if self.aten is not None:
            Z = self.aten(Z)
        return Z + self.mlp(Z)

    def raport(self):
        r = {"troj_wkl_wych": self.wych._diag.get("wklad"),
             "troj_wkl_wchodz": self.wchodz._diag.get("wklad")}
        if self.aten is not None:
            r.update(self.aten._diag)
        return r


class TorTrojkatny(nn.Module):
    """Caly tor: mapa okien -> pary -> trojkaty -> odczyt per okno + aux.

    `forward` zwraca `(cechy_okien, pred_aux)`:
      * `cechy_okien` [B, dim_wy, W] -- do KONKATENACJI z mapa trzonu,
        celowo nie za brama (patrz naglowek);
      * `pred_aux` [B, 2] -- predykcja Dev/Hk z SAMEJ reprezentacji parowej,
        pod strate pomocnicza. Galaz z wlasnym celem nie da sie wyzerowac
        bez kary, i to jest jedyny mechanizm, ktory w tym repo utrzymal
        galaz boczna zywa.
    """

    def __init__(self, dim_we, dim_par=32, blokow=2, dim_wy=32,
                 glowic=4, dim_glowy=16, atencja=True, max_okien=32):
        super().__init__()
        self.pary = ReprezentacjaPar(dim_we, dim_par, max_okien)
        self.bloki = nn.ModuleList([
            BlokTrojkatny(dim_par, glowic, dim_glowy, atencja)
            for _ in range(int(blokow))])
        # ODCZYT: para (i, j) -> okno i. Srednia i maks po `j`, bo srednia
        # jest agregatem, a maks znajduje NAJMOCNIEJSZY kontakt -- i to
        # druga z tych statystyk jest tu istotna, inaczej jeden silny styk
        # rozmywa sie po 15 oknach.
        self.odczyt = nn.Conv1d(2 * dim_par, int(dim_wy), 1)
        self.norma_wy = nn.BatchNorm1d(int(dim_wy))
        self.aux = nn.Sequential(nn.Linear(2 * dim_par, 64), nn.SiLU(),
                                 nn.Linear(64, 2))
        self._diag = {}

    def forward(self, h):
        Z = self.pary(h)
        Z0 = Z
        for b in self.bloki:
            Z = b(Z)
        st = torch.cat([Z.mean(dim=2), Z.amax(dim=2)], dim=-1)  # [B, W, 2c]
        wy = self.norma_wy(self.odczyt(st.permute(0, 2, 1)))
        aux = self.aux(st.mean(dim=1))
        with torch.no_grad():
            self._diag = self._falsyfikatory(Z0, Z)
            for i, b in enumerate(self.bloki):
                for k, v in b.raport().items():
                    self._diag["%s_b%d" % (k, i)] = v
        return F.silu(wy), aux

    @staticmethod
    def _ranga_ef(Z):
        """Efektywna ranga (exp entropii widma) PER KANAL, potem srednia.

        POPRAWKA DO WLASNEGO MIERNIKA. Pierwsza wersja liczyla range
        `Z.mean(-1)`, czyli sredniej po kanalach -- a srednia po kanalach
        iloczynu zewnetrznego `a_ic b_jc` to macierz Grama rangi do `c`,
        nie rangi 1. Zmierzone: czysty iloczyn zewnetrzny dawal **10,648**
        tam, gdzie z konstrukcji ma byc 1,000. Falsyfikator mierzyl wiec
        mieszanie kanalow, nie strukture par, i nie mogl wykryc, czy
        trojkaty cokolwiek dodaly. Ranga liczona jest teraz na KAZDYM
        kanale osobno -- tam iloczyn zewnetrzny JEST rangi 1.
        """
        B, W, _, c = Z.shape
        M = Z.permute(0, 3, 1, 2).reshape(B * c, W, W).float()
        s = torch.linalg.svdvals(M)
        p = s / s.sum(-1, keepdim=True).clamp_min(1e-12)
        h = -(p * (p + 1e-12).log()).sum(-1)
        return round(float(h.exp().mean()), 3)

    def _falsyfikatory(self, Z0, Z):
        M = Z.mean(-1)
        M = M - M.mean(dim=(1, 2), keepdim=True)
        M2 = M @ M
        a = M.reshape(M.shape[0], -1)
        b = M2.reshape(M2.shape[0], -1)
        a = a - a.mean(-1, keepdim=True)
        b = b - b.mean(-1, keepdim=True)
        dom = ((a * b).sum(-1)
               / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12))
        # KONTROLA: to samo dla Z przetasowanego po parach. Bez niej
        # `domkniecie` nie ma odniesienia -- kor(M, M^2) jest dodatnie
        # dla prawie kazdej macierzy.
        idx = torch.randperm(a.shape[-1], device=a.device)
        ak = a[:, idx]
        domk = ((ak * b).sum(-1)
                / (ak.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12))
        return {
            "troj_ranga": self._ranga_ef(Z),
            "troj_ranga0": self._ranga_ef(Z0),
            "troj_domkniecie": round(float(dom.mean()), 4),
            "troj_domk_kontrola": round(float(domk.mean()), 4),
            "troj_asym": round(float(
                (Z - Z.transpose(1, 2)).pow(2).mean().sqrt()
                / Z.pow(2).mean().sqrt().clamp_min(1e-9)), 4),
        }

    def raport(self):
        return dict(self._diag)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, W = 8, 96, 15
    tor = TorTrojkatny(C, dim_par=32, blokow=2, dim_wy=32)
    h = torch.randn(B, C, W)
    wy, aux = tor(h)
    print("[trojkat] wy %s | aux %s | param %d"
          % (tuple(wy.shape), tuple(aux.shape),
             sum(p.numel() for p in tor.parameters())))
    for k, v in sorted(tor.raport().items()):
        print("   %-22s %s" % (k, v))

    print("\nTEST 1 -- czy modul jest naprawde TRZECIEGO rzedu")
    # POPRAWKA DO WLASNEGO TESTU. Pierwsza wersja perturbowala `Z[2,3]`
    # i zadala zmiany w `(0,1)` -- i dostala DOKLADNIE 0. To bylo poprawne
    # zachowanie modulu, a bledny test: `out_ij = sum_k a_ik b_jk` czyta
    # WIERSZ i oraz WIERSZ j, czyli wpisy `Z_ik` i `Z_jk`. Wpis `Z_23` nie
    # lezy w zadnym z nich. Wlasciwa droga trojkata dla pary (0,1) idzie
    # przez TRZECIE okno k, wiec perturbowac trzeba `Z[1,k]`.
    par = ReprezentacjaPar(C, 32)
    akt = AktualizacjaTrojkatna(32, wychodzaca=True)
    with torch.no_grad():
        Z = par(h)
        o1 = akt(Z)
        Zp = Z.clone()
        Zp[:, 1, 2] += 5.0                     # para (j=1, k=2)
        d_troj = float((akt(Zp) - o1)[:, 0, 1].abs().max())
        Zq = Z.clone()
        Zq[:, 2, 3] += 5.0                     # para (2,3): poza wierszami 0 i 1
        d_poza = float((akt(Zq) - o1)[:, 0, 1].abs().max())
    print("   perturbacja Z[1,2] -> zmiana w (0,1): %.6e  (musi byc > 0)"
          % d_troj)
    print("   perturbacja Z[2,3] -> zmiana w (0,1): %.6e  (MUSI byc 0 --"
          % d_poza)
    print("      AF2 czyta wiersze i oraz j, nie caly tensor)")
    print("   werdykt: %s"
          % ("TRZECI RZAD potwierdzony" if d_troj > 0 and d_poza == 0.0
             else "NIEZGODNE z semantyka AF2 -- sprawdzic einsum"))

    print("\nTEST 2 -- czy trojkaty podnosza range ponad iloczyn zewnetrzny")
    # POPRAWKA: `troj_ranga0` mierzy `Z` PO dodaniu osadzenia odleglosci,
    # ktore samo jest pelnorangowe -- wiec 11,87 nie bylo ranga iloczynu
    # zewnetrznego i ten falsyfikator nie mial rozdzielczosci. Wlasciwe
    # odniesienie to CZYSTY iloczyn zewnetrzny, ktory MUSI miec range 1.
    with torch.no_grad():
        a = par.lewy(h).permute(0, 2, 1)
        b = par.prawy(h).permute(0, 2, 1)
        Zsur = a.unsqueeze(2) * b.unsqueeze(1)
        r_sur = TorTrojkatny._ranga_ef(Zsur)
    r = tor.raport()
    print("   ranga czystego iloczynu zewnetrznego  %s  (musi byc ~1)" % r_sur)
    print("   ranga Z0 (iloczyn + osadzenie odl.)   %s" % r["troj_ranga0"])
    print("   ranga Z  (po dwoch blokach)           %s" % r["troj_ranga"])

    print("\nTEST 3 -- gradient dochodzi do wszystkich czesci")
    wy, aux = tor(h)
    (wy.sum() + aux.sum()).backward()
    bez = [n for n, p in tor.named_parameters()
           if p.grad is None or float(p.grad.abs().max()) == 0.0]
    print("   parametry BEZ gradientu: %d %s" % (len(bez), bez[:4]))

    print("\nTEST 4 -- domkniecie odstaje od kontroli?")
    print("   troj_domkniecie %s wobec kontroli %s"
          % (r["troj_domkniecie"], r["troj_domk_kontrola"]))

    print("\nTEST 5 -- wersja NUKLEOTYDOWA pasmowa (pary ZASAD)")
    L, Rp = 249, 16
    torp = TorTrojkatnyPasmowy(C, dim_par=24, pasmo=Rp, blokow=2, dim_wy=24)
    hn = torch.randn(4, C, L)
    wyp, auxp, h1p = torp(hn)
    print("   wy %s | aux %s | strumien 1D %s | param %d | pasmo %d nt"
          % (tuple(wyp.shape), tuple(auxp.shape), tuple(h1p.shape),
             sum(p.numel() for p in torp.parameters()), 2 * Rp + 1))
    print("   TEST 5b -- droga powrotna 2D->1D zmienila strumien?")
    print("      seq_zmiana %s  (0 = kierunek martwy)"
          % torp.raport().get("seq_zmiana"))
    for k, v in sorted(torp.raport().items()):
        print("      %-22s %s" % (k, v))

    print("\nTEST 6 -- poprawnosc transpozycji pasmowej")
    # Zt[i,o] musi rownac sie Z[i+o,-o]. Sprawdzone wprost na losowym Z,
    # bo od tego zalezy, czy aktualizacja WCHODZACA liczy to, co twierdzi.
    Zt_we = torch.randn(2, L, 2 * Rp + 1, 3)
    Zt = _transponuj_pasmo(Zt_we, Rp)
    bl, ok = 0.0, 0
    for i in range(0, L, 37):
        for o in range(-Rp, Rp + 1):
            if 0 <= i + o < L:
                bl = max(bl, float((Zt[:, i, o + Rp]
                                    - Zt_we[:, i + o, -o + Rp]).abs().max()))
                ok += 1
    print("   max blad %.3e na %d sprawdzonych wpisach -> %s"
          % (bl, ok, "OK" if bl == 0.0 else "ZLE"))

    print("\nTEST 7 -- trzeci rzad w wersji pasmowej")
    parp = ReprezentacjaParPasmowa(C, 24, Rp)
    aktp = AktualizacjaTrojkatnaPasmowa(24, Rp, wychodzaca=True)
    with torch.no_grad():
        Zb = parp(hn)
        b1 = aktp(Zb)
        Zb2 = Zb.clone()
        Zb2[:, 100 + 1, 2 + Rp] += 5.0        # para (101, 103): j=101, k=103
        d1 = float((aktp(Zb2) - b1)[:, 100, 1 + Rp].abs().max())
        Zb3 = Zb.clone()
        Zb3[:, 140, 0 + Rp] += 5.0            # para poza pasmem pary (100,101)
        d2 = float((aktp(Zb3) - b1)[:, 100, 1 + Rp].abs().max())
    print("   Z[101,+2] -> zmiana w (100,101): %.6e  (musi byc > 0)" % d1)
    print("   Z[140, 0] -> zmiana w (100,101): %.6e  (musi byc 0)" % d2)
    print("   werdykt: %s"
          % ("TRZECI RZAD lokalny potwierdzony" if d1 > 0 and d2 == 0.0
             else "SPRAWDZIC indeksowanie pasma"))
