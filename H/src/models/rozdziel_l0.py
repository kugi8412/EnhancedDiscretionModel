# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from .hyperbolic import logmap0


def _kor(a, b):
    """Korelacja Pearsona po batchu, [N] x [N] -> skalar."""
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12)


class RozdzielenieL0(nn.Module):
    """Dwa uczone kierunki styczne w origo; rzut punktow L0 na kazdy z nich.

    `dim` -- wymiar przestrzeni STYCZNEJ, czyli tyle, ile zwraca `logmap0`
    (= `vq_dim`, a punkt ma `vq_dim + 1` wspolrzednych ambientnych).
    """

    def __init__(self, dim, lambda_ort=1.0):
        super().__init__()
        self.u_dev = nn.Parameter(torch.randn(int(dim)) * (1.0 / dim ** 0.5))
        self.u_hk = nn.Parameter(torch.randn(int(dim)) * (1.0 / dim ** 0.5))
        self.lambda_ort = float(lambda_ort)
        self._diag = {}

    def rzuty(self, punkty_l0):
        """`punkty_l0`: [B, L0, dim+1] na hiperboloidzie -> (p_dev, p_hk) [B].

        `logmap0` daje wektor styczny w origo; skladowa czasowa jest tam
        zerowa z konstrukcji, wiec brany jest wylacznie blok przestrzenny.
        Agregacja SREDNIA po pozycjach L0 -- rozdzielenie ma byc wlasnoscia
        sekwencji, a nie jednej pozycji.
        """
        v = logmap0(punkty_l0.double())                     # [B, L0, dim]
        if v.shape[-1] != self.u_dev.shape[0]:
            raise RuntimeError(
                "RozdzielenieL0: wektor styczny ma %d wymiarow, kierunki %d. "
                "Buduj z `vq_dim - 1`, nie `vq_dim`."
                % (v.shape[-1], self.u_dev.shape[0]))
        ud = self.u_dev.double() / self.u_dev.double().norm().clamp_min(1e-9)
        uh = self.u_hk.double() / self.u_hk.double().norm().clamp_min(1e-9)
        p_dev = (v * ud).sum(-1).mean(-1)
        p_hk = (v * uh).sum(-1).mean(-1)
        return p_dev.float(), p_hk.float()

    def strata(self, punkty_l0, y_dev, y_hk):
        """Strata rozdzielenia. Zwraca skalar i wypelnia `raport()`."""
        p_dev, p_hk = self.rzuty(punkty_l0)
        s_dd, s_hh = _kor(p_dev, y_dev), _kor(p_hk, y_hk)
        s_dh, s_hd = _kor(p_dev, y_hk), _kor(p_hk, y_dev)
        ud = self.u_dev / self.u_dev.norm().clamp_min(1e-9)
        uh = self.u_hk / self.u_hk.norm().clamp_min(1e-9)
        ort = (ud * uh).sum().pow(2)
        L = (s_dh.abs() + s_hd.abs() - s_dd.abs() - s_hh.abs()
             + self.lambda_ort * ort)
        with torch.no_grad():
            self._diag = {
                # SYGNAL: ile kierunek Dev wie o Dev. Zero = sektor nie
                # niesie nic i mechanizm jest tylko kosztem.
                "rozdz_dev": round(float(s_dd), 4),
                "rozdz_hk": round(float(s_hh), 4),
                # PRZECIEK: ile kierunek Dev wie o Hk. To jest wielkosc,
                # ktora ma SPADAC -- i jedyna, ktora mowi, czy cokolwiek
                # zostalo ROZDZIELONE, a nie tylko dopasowane.
                "rozdz_przeciek_dh": round(float(s_dh), 4),
                "rozdz_przeciek_hd": round(float(s_hd), 4),
                # KONTROLA STRUKTURALNA: sektory musza byc rozlaczne.
                # Bez tego oba kierunki zbiegaja do jednej osi "aktywnosci"
                # i oba sygnaly wygladaja swietnie przy zerowym rozdzieleniu.
                "rozdz_cos_uu": round(float((ud * uh).sum()), 4),
                "rozdz_strata": round(float(L), 5),
            }
        return L

    def raport(self):
        return dict(self._diag)


@torch.no_grad()
def sonda_rozdzielenia(model, X, y):
    """FALSYFIKATOR: czy sektory realnie rozdzielaja, wobec KONTROLI LOSOWEJ.

    Zwraca slownik z korelacjami dla kierunkow UCZONYCH i dla pary
    kierunkow LOSOWYCH o tej samej normie. Bez kontroli losowej same
    korelacje nic nie mowia: dwa dowolne kierunki w przestrzeni stycznej
    beda skorelowane z aktywnoscia, bo aktywnosc jest funkcja sekwencji,
    a punkty L0 sa jej reprezentacja.
    """
    model.encode_codes(X)
    p = model.punkty_poziomow()[0]
    r = model.rozdziel
    pd, ph = r.rzuty(p)
    dim = r.u_dev.shape[0]
    g = torch.Generator(device="cpu").manual_seed(0)
    ud = torch.randn(dim, generator=g).to(p.device)
    uh = torch.randn(dim, generator=g).to(p.device)
    v = logmap0(p.double())[..., 1:]
    ld = (v * (ud.double() / ud.double().norm())).sum(-1).mean(-1).float()
    lh = (v * (uh.double() / uh.double().norm())).sum(-1).mean(-1).float()
    return {
        "uczone_dev_Dev": round(float(_kor(pd, y[:, 0])), 4),
        "uczone_dev_Hk": round(float(_kor(pd, y[:, 1])), 4),
        "uczone_hk_Hk": round(float(_kor(ph, y[:, 1])), 4),
        "uczone_hk_Dev": round(float(_kor(ph, y[:, 0])), 4),
        "losowe_1_Dev": round(float(_kor(ld, y[:, 0])), 4),
        "losowe_1_Hk": round(float(_kor(ld, y[:, 1])), 4),
        "losowe_2_Dev": round(float(_kor(lh, y[:, 0])), 4),
        "losowe_2_Hk": round(float(_kor(lh, y[:, 1])), 4),
    }
