from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kanon_rc import kanonizuj
from .rozdziel_l0 import RozdzielenieL0
from .hipergraf_motywow import (AtencjaPodprzestrzenMotyw,
                                HipergrafWspolwystepowania)
from .atencja_motywow import AtencjaMotywowGlobalna
from .trojkat import TorTrojkatnyPasmowy
from .genomic_blocks import MultiScaleDilatedBlock, RCConv1d
from .hyperbolic import logmap0, lorentz_centroid, HyperbolicVQ
from .registry import register_model
from .vq_gat_film import GraphAttentionLayer


def _dlugosci_z_kodow(kody, seq_len):
    """Dlugosci map poziomow, OD NAJGRUBSZEJ, z jednego zrodla prawdy.

    `HyperbolicCodeStack` liczy `dlugosci_poziomow` swiadomie dla obu
    przypadkow: bez `rc_pelne` wzorem splotu, a z `rc_pelne` z `_rc_ksztalt`,
    bo tam arytmetyka dopelnienia jest inna (okna musza przechodzic na siebie
    pod odbiciem). Przy stridach (2,2,4) i 249 bp daje to (15, 62, 124) bez RC
    i (17, ..., ...) z RC.

    Kazde miejsce, ktore przelicza te liczby wlasnym wzorem, jest kandydatem
    na cichy blad ksztaltu -- i jednym takim byla tablica biasu `AtencjaKodow`,
    ktora przy `hyp_heads: true` + `rc_pelne: true` wywalala pierwszy batch.
    Fallback jest tu tylko dla wariantow BEZ drabiny stozkowej, gdzie ten
    atrybut nie powstaje.
    """
    _dl = getattr(kody, "dlugosci_poziomow", None)
    if _dl:
        return tuple(int(v) for v in _dl)
    _L, _out = int(seq_len), []
    for _s in kody.strides:
        _L = (_L + 2 * (_s // 2) - 2 * _s) // _s + 1
        _out.append(_L)
    return tuple(_out[::-1])


class HyperbolicCodeStack(nn.Module):
    """Multi-level hyperbolic VQ over a downsampling encoder.

    The codebooks are held in float64.  That is not caution: autoguard_v2
    measured float32 Lorentz parameters reaching NaN within a few hundred steps
    at diameter 15, because ``<x,y>_L = x0*y0 - x.y`` subtracts two large,
    nearly equal quantities.  The surrounding network stays float32; only the
    manifold arithmetic is promoted.
    """

    def __init__(self, dim=64, strides=(2, 2, 4), codes=(512, 256, 128),
                 commit_weight=0.25, rc_stem=True, level_radii=(0.25, 0.6, 1.2),
                 recon_bypass=True, trunk_bypass=None,
                 frechet=False, learnable=False,
                 codebook_weight=1.0, pelny_zasieg=False, enc_dropout=0.0,
                 in_ch=4, skip_enc=False, enc_na_powloce=False,
                 rc_stem_split=False, rc_pelne=False, seq_len_enc=249,
                 brama_glu=False, brama_glu_z_se=False, brama_glu_ranga=0,
                 pasmo_r=0.0, przes_powloki=False,
                 promien_specyf=0.0,
                 kanal_promienia=False,
                 bloki_poziomu=0, blok_jadro=3, blok_dropout=0.1,
                 udzial_kontekst=0.0, kontekst_typ="splot",
                 kontekst_bloki=1, kontekst_stan=32,
                 agregacje=None,
                 drabina=False, fsq=False, drab_dlugosc=1.0,
                 drab_krok_promienia=0.5, fsq_bity=9, fsq_entropia=0.0,
                 fsq_wzmocnienie=15.0,
                 stem_rezydualny=False, udzial_motywow=0.3, motyw_brama=0.1,
                 waga_stozka=0.0, apertura_stozka=1.0,
                 bloki_przed=0, blok_przed_jadro=7,
                 mot_atencja_l0=0, mot_l0_occ=0, mot_l0_g=0,
                 mot_l0_glowic=4, mot_l0_dim_glowy=32, kwargs_wieza=None):
        super().__init__()
        self.strides = tuple(int(s) for s in strides)
        self.n_levels = len(self.strides)
        # `rc_stem` MILCZY, gdy in_ch != 4. Przy `motywy_do_latentu: true`
        # in_ch = 12, wiec warunek nizej jest FALSZYWY i `rc_stem: true`
        # w configu nie robi nic. `rc_stem_split` naprawia to czesciowo:
        # nukleotydy przez `RCConv1d`, kanaly dodatkowe swobodna sciezka.
        # ============ EKWIWARIANCJA RC PRZEZ CALY ENKODER ============
        # ZMIERZONE, czemu to jest tu, a nie w stracie: model daje dwie rozne
        # liczby dla tej samej czasteczki -- sr|f(x)-f(rc x)|/sd(f) = 0,1780
        # (Dev) i 0,1245 (Hk) na pelnej walidacji -- MIMO ZE dane treningowe
        # zawieraja oba pasma i mimo czlonu `rc_consistency` o wadze 0,1.
        # Luka jest SZUMEM: usrednienie poprawia blad dla 52,2 % sekwencji,
        # czyli rzut moneta. Sufit z samego usredniania (= RC-TTA) to
        # 0,74626 -> 0,75153, zysk +0,00527 -- i to jest DOLNA granica tego,
        # co daje struktura, bo ta dodatkowo zwalnia pojemnosc idaca dzis na
        # dwie kopie kazdego filtra (`SplotParowy` ma DOKLADNIE polowe wag).
        self.rc_pelne = bool(rc_pelne)
        if self.rc_pelne and stem_rezydualny:
            raise ValueError(
                "stem_rezydualny + rc_pelne: `StemRC` JUZ realizuje forme "
                "addytywna (`out = conv(nuk) + conv(mot)`) z wiazaniem RC "
                "i zerowym initem toru motywowego. Dwa stemy naraz nie "
                "powstana, a flaga bylaby cichym no-opem.")
        if self.rc_pelne:
            from .rc_ekwi import StemRC, SplotParowy, NormaParowa
            if dim % 2:
                raise ValueError("rc_pelne wymaga parzystego vq_dim")
            from .rc_ekwi import dobierz_padding
            _stem0 = StemRC(dim, max(in_ch - 4, 0), 5)
            self.stem = nn.Sequential(_stem0, NormaParowa(dim), nn.SiLU())
            # JADRO I DOPELNIENIE DOBIERANE DO DLUGOSCI, nie stale. Warunek
            # `(L + 2p - k) % s == 0` jest KONIECZNY dla ekwiwariancji; przy
            # L = 249 i stalym (k=2s, p=s//2) nie zachodzi i blad wychodzi
            # 1,22 zamiast 4e-07 -- czyli wyglada na zepsuta konstrukcje,
            # a jest sama arytmetyka indeksow. Dla DeepSTARR daje to
            # 249 -> 125 -> 63 -> 17.
            self._rc_ksztalt = []
            _L = int(seq_len_enc)
            _downs, _skips = [], []
            for _s in self.strides:
                _k, _p, _L2 = dobierz_padding(_L, _s)
                _ks, _ps, _Ls = dobierz_padding(_L, _s, k=_s + 1)
                if _Ls != _L2:
                    raise ValueError(
                        "skrot i splot glowny daja rozne dlugosci (%d vs %d) "
                        "na poziomie o kroku %d -- dodawanie ich zlamaloby "
                        "ekwiwariancje" % (_Ls, _L2, _s))
                self._rc_ksztalt.append((_L, _k, _p, _L2))
                _downs.append(nn.Sequential(
                    SplotParowy(dim, dim, _k, stride=_s, padding=_p),
                    NormaParowa(dim), nn.SiLU()))
                _skips.append(SplotParowy(dim, dim, _ks, stride=_s, padding=_ps))
                _L = _L2
            self.downs = nn.ModuleList(_downs)
            self._rc_skips = nn.ModuleList(_skips)
        else:
            if stem_rezydualny and in_ch > 4:
                # TOR MOTYWOWY REZYDUALNIE, Z BRAMA. Konkatenacja plus jeden
                # splot jest algebraicznie ta sama suma, ale bez bramy
                # proporcja miedzy torami stoi na losowej inicjalizacji.
                # ZMIERZONE na HypMaks (296 profili, motyw_ch 32): wklad
                # motywow ma RMS 0,1443 wobec 0,0927 dla nukleotydow, czyli
                # iloraz 1,557 -- tor motywowy DOMINUJE tor sekwencyjny,
                # zanim czegokolwiek sie nauczy, i to w miejscu, ktore stoi
                # przed wszystkimi `downs` i przed kwantyzatorem.
                from .stem_motyw import StemMotyw
                _stem0 = StemMotyw(dim, in_nuk=4, in_mot=in_ch - 4, jadro=5,
                                   udzial=float(udzial_motywow),
                                   brama0=float(motyw_brama))
            elif rc_stem and in_ch == 4:
                _stem0 = RCConv1d(dim // 2, 5)
            elif rc_stem_split and in_ch > 4:
                from .genomic_blocks import RCStemSplit
                _stem0 = RCStemSplit(dim, in_ch - 4, 5)
            else:
                _stem0 = nn.Conv1d(in_ch, dim, 5, padding=2)
            self.stem = nn.Sequential(_stem0, nn.BatchNorm1d(dim), nn.SiLU())
            self.downs = nn.ModuleList(
                nn.Sequential(nn.Conv1d(dim, dim, 2 * s, stride=s, padding=s // 2),
                              nn.BatchNorm1d(dim), nn.SiLU())
                for s in self.strides)
        # ============ BLOKI PRZED PIERWSZYM `down` ============
        # UMIEJSCOWIENIE JEST ZMIERZONE, nie dobrane. Zapisana naprawa
        # wygłodzonego enkodera brzmi: "6 blokow residualnych kernel 7 MIEDZY
        # `stem` i `down`" i dala epoke 1 = 0,70010 wobec 0,48545. Uzasadnienie
        # jest fazowe: "po stride 2 motyw 6-15 bp traci faze; kernel 7 daje po
        # dwoch warstwach pole widzenia 13 bp". `bloki_poziomu` siedza PO
        # kazdym `down`, wiec pracuja na sygnale, ktoremu faza juz zostala
        # odebrana -- to inne miejsce i inny mechanizm.
        #
        # Po co w ogole: przy `trzon_tylko_z_latentu` enkoder jest JEDYNA droga
        # do predykcji. Zmierzone w HypFinal: enkoder 212 768 = 6,6 % modelu,
        # a bieg wyszedl -0,036 w epoce 2 wobec bazy. Zapisana regula mowi
        # wprost: "jesli sciezka latentna ma kilka procent modelu, zadna
        # geometria ani cel pomocniczy tego nie nadrobi -- i wnioski
        # o geometrii sa wtedy NIEWIARYGODNE, bo mierza wyglodzony kanal".
        # ============ ATENCJA MOTYWOW DO L0 (model billboardu) =============
        # Na NAJGRUBSZEJ mapie, czyli tam, gdzie juz jest zapisane, ze
        # "tylko najgrubsza mapa staje sie globalna". Zapytanie z okna,
        # klucze z calego obłozenia 249 bp -- bo zmierzone: pary BEZ
        # odleglosci bija pary w powlokach o +0,01932 (billboard).
        # Wynik wchodzi do mapy, ktora jest KWANTYZOWANA, wiec mechanizm
        # zmienia czym JEST kod L0 -- tym rozni sie od `KontekstKrzyzowy`,
        # ktory nigdy nie dotykal latentu i dlatego jego smierc sie tu
        # nie przenosi.
        kwargs_wieza = dict(kwargs_wieza or {})
        # ============ WIEZA TROJKATNA NA WCZESNEJ MAPIE =====================
        # `dim` kanalow na PELNYCH 249 pozycjach, przed stridem i przed
        # kwantyzacja -- patrz naglowek `wpnij_wieze_enkoder`. Wynik
        # PODMIENIA `h`, wiec trafia do tego, co jest kwantyzowane.
        self.wieza = None
        if int(kwargs_wieza.get("troj_pasmo", 0) or 0) > 0:
            from .trojkat import TorTrojkatnyPasmowy as _TTP
            self.wieza = _TTP(
                dim,
                dim_par=int(kwargs_wieza.get("troj_kanaly", 24)),
                pasmo=int(kwargs_wieza.get("troj_pasmo", 16)),
                blokow=int(kwargs_wieza.get("troj_bloki", 2)),
                dim_wy=int(kwargs_wieza.get("troj_wy", 24)),
                seq_atencja=bool(kwargs_wieza.get("troj_seq_atencja", True)),
                glowic=int(kwargs_wieza.get("troj_glowic", 4)),
                dim_glowy=int(kwargs_wieza.get("troj_dim_glowy", 24)),
                checkpoint=bool(kwargs_wieza.get("troj_checkpoint", True)),
                gruby_L=int(kwargs_wieza.get("troj_gruby_L", 0) or 0),
                gruby_bloki=int(kwargs_wieza.get("troj_gruby_bloki", 2)),
                gruby_par=int(kwargs_wieza.get("troj_gruby_par", 16)),
                gruby_kawalek=int(kwargs_wieza.get("troj_gruby_kawalek", 64)),
                gruby_wy=int(kwargs_wieza.get("troj_gruby_wy", 16)),
                brama_pozycyjna=bool(
                    kwargs_wieza.get("troj_brama_pozycyjna", False)),
                udzial_seq=float(
                    kwargs_wieza.get("troj_udzial_seq", 0.0) or 0.0))
            _ww = (int(kwargs_wieza.get("troj_wy", 24))
                   + (int(kwargs_wieza.get("troj_gruby_wy", 16))
                      if int(kwargs_wieza.get("troj_gruby_L", 0) or 0) else 0))
            # RZUT 1x1 z powrotem do `dim` i DODANIE, zeby szerokosci
            # w dol nie zmieniac. Nie brama: `h` jest podmieniane strumieniem
            # po atencji, wiec stanu "wylaczone" nie ma.
            self.wieza_rzut = nn.Conv1d(_ww, dim, 1)
        self.mot_l0 = None
        if int(mot_atencja_l0) > 0 and int(mot_l0_occ) > 0:
            self.mot_l0 = AtencjaMotywowGlobalna(
                dim, int(mot_l0_occ), dim_g=int(mot_l0_g),
                glowic=int(mot_l0_glowic),
                dim_glowy=int(mot_l0_dim_glowy))
        self._occ_kv = None
        self._g_kv = None
        self.bloki_przed = None
        # BRAMA GLU ZAMIAST SE -- jedna flaga na CALY model, bo eksperyment
        # ma mierzyc RODZAJ bramkowania, a nie mieszanke dwoch rodzajow.
        _glu = bool(brama_glu)
        self._brama_glu = _glu
        # WSPOLISTNIENIE, a nie mieszanka. Bramy sa ORTOGONALNE: GLU jest
        # per POZYCJA (splot 1x1 z wejscia bloku, zasieg lokalny), SE jest
        # per KANAL ze statystyki po WSZYSTKICH pozycjach, czyli rangi 1
        # po pozycji. Wykluczenie ich bylo decyzja METODYCZNA tamtego
        # eksperymentu ("bieg ma mierzyc RODZAJ bramy"), nie ograniczeniem
        # architektury -- i kosztowalo SE, czyli mechanizm z jedynym
        # dodatnim pomiarem o UJEMNEJ kontroli pojemnosci (netto +0,0976
        # na L0, +0,0565 na L1). Domyslnie False, wiec bez zmian.
        _glu_se = bool(brama_glu_z_se)
        _glu_r = int(brama_glu_ranga or 0)
        self._brama_glu_z_se = _glu_se and _glu
        self._brama_glu_ranga = _glu_r
        if int(bloki_przed) > 0:
            from .genomic_blocks import MultiScaleDilatedBlock
            self.bloki_przed = nn.ModuleList(
                MultiScaleDilatedBlock(dim, int(blok_przed_jadro),
                                       (1, 2, 4), dropout=blok_dropout,
                                       brama_glu=_glu,
                                       brama_glu_z_se=_glu_se,
                                       brama_glu_ranga=_glu_r)
                for _ in range(int(bloki_przed)))

        # ============ BLOKI NA ROZDZIELCZOSCI POZIOMU ============
        # ZMIERZONE na HypEncSkala_best: rzad efektywny KIERUNKOW wejscia
        # kwantyzatora, z 64 dostepnych wymiarow --
        #     stem 3,85 | down[0] (=L2) 12,24 | down[1] (=L1) 2,89
        #     down[2] 1,95 | po `kontekst` (=L0) 1,03
        # Czyli L0 koduje SKALAR (stozek 4,5 stopnia, 1,6 % wymiarow), a L1
        # niesie 2,89 wymiaru i uzywa 67 z 256 kodow. `down[1]` sam niszczy
        # 76 % rzedu (12,24 -> 2,89).
        #
        # PRZYCZYNA STRUKTURALNA: enkoder nie ma ANI JEDNEJ warstwy
        # przetwarzajacej na rozdzielczosci poziomu. Kazdy `feats[i]` to
        # wyjscie JEDNEGO splotu ze stridem plus skip 1x1 -- poziom nie ma
        # czym zbudowac reprezentacji, dostaje surowy wynik zwezenia.
        #
        # `bez_se=True` jest KONIECZNE: bramka SE to `h * se(h.mean(dim=-1))`,
        # czyli sciskanie po WSZYSTKICH pozycjach, wiec choc jeden blok z SE
        # przed kwantyzatorem daje globalne mieszanie na kazdym poziomie --
        # zabronione przed kwantyzatorem, bo psuje faktoryzacje prioru dyfuzji.
        #
        # Skalar wyjscia startuje MALO, ale NIE zerem (0,1): zerowy glodzi obie
        # strony gradientu, a duzy wstrzykuje szum, zanim blok sie czegokolwiek
        # nauczy -- oba bledy popelnione w tym repo. Nazwa zawiera `_skala`,
        # wiec trafia na liste zwolnionych z weight decay.
        # ============ ROZNE AGREGACJE DO ROZNYCH POZIOMOW LATENTU ============
        # Kazdy `down[i]` to dzis SPLOT ZE STRIDEM, czyli uczona SUMA WAZONA po
        # oknie. Naturalna operacja splotu jest usrednianie, a zmierzone na tych
        # danych (sonda liniowa out-of-sample, oblozenie 20 profili PWM, skala
        # izolowana przez usrednianie po oknach):
        #
        #   statystyka |  8 bp (L2) | 14 bp (L1) | 32 bp | 249 bp (L0)
        #   max        |     0,186  |     0,224  | 0,267 |    0,329
        #   std        |     0,188  |     0,205  | 0,224 |    0,223
        #   zakres     |     0,211  |     0,242  | 0,274 |    0,329
        #   mean       |     0,004  |     0,005  | 0,002 |    0,004
        #   argmax     |     0,047  |     0,071  | 0,064 |    0,067
        #
        # Trzy odczyty. (1) MAKSIMUM rosnie monotonicznie z oknem, wiec jego
        # najlepsza skala jest CALA sekwencja -- czyli L0. (2) STD jest plaskie
        # w skali i przy 8 bp REMISUJE z maksimum (0,188 wobec 0,186), a mierzy
        # co INNEGO: kontrast, nie wysokosc piku -- wiec nalezy do L2, gdzie
        # maksimum jest najslabsze. (3) SREDNIA nie niesie nic przy ZADNYM
        # oknie (0,002-0,005), wiec nie zasluguje na zaden poziom.
        #
        # Domyslne przypisanie `agregacje = ("std", "zakres", "max")` idzie
        # w kolejnosci `downs`, czyli L2 / L1 / L0 (bo `by_level = feats[::-1]`).
        #
        # CZEMU TO MA ZNACZENIE DLA HIERARCHII: dzis reszta `feats[k+1] -
        # up(q_k)` jest roznica TEJ SAMEJ statystyki w dwoch skalach, wiec jest
        # redundantna -- i model ja WYLACZYL (`skala_reszty` 4e-05). Przy
        # roznych statystykach na roznych poziomach reszta ma co niesc.
        # FALSYFIKATOR: `skala_reszty` musi przestac zapadac do zera.
        #
        # Galaz statystyki wchodzi PRZEZ UCZONA BRAME (init 0,1, nie 0 -- zero
        # gloduje obie strony gradientu), a laczy sie splotem 1x1: 8 256 wag
        # na poziom, 24 768 razem.
        # ===== KANAL PROMIENIA =====
        # `HyperbolicVQ.forward` zwraca `out[...,1:] / radius_scale`, czyli
        # SAMA CZESC PRZESTRZENNA -- promien jest dzielony przez siebie
        # i nigdzie dalej nie dociera. Dekoder, trzon i glowice sa slepe na
        # jedyna wspolrzedna, ktorej przestrzen euklidesowa NIE MA.
        # Dopoki promien byl przypiety (rozrzut enkodera 5,3e-17), nie bylo
        # czego czytac -- zmierzone: sfera i hiperboloida dawaly wtedy
        # IDENTYCZNE kody (zgodnosc 1,0000 po splaszczeniu kodebooka).
        # Przy `promien_specyf` promien niesie specyficznosc sygnalu
        # (w biegu: r_sd_w 0,04133, spec_a 0,5236, spec_nasyc 0,3676).
        # UKLAD KANALOW: [q0..qK, r0..rK, (slot)] -- promienie PRZED slotem,
        # zeby `tylko_kody` zostalo cieciem prefiksu, a slot OSTATNIMI `dim`
        # kanalami, inaczej cala arytmetyka `recon_bypass` przestaje grac.
        # Promien jako (r - promien_cel)/sinh(promien_cel): wysrodkowany,
        # rzedu jednosci, stala per poziom, ZERO zaleznosci od skladu batcha.
        self.kanal_promienia = bool(kanal_promienia)
        self.agregacje = tuple(agregacje) if agregacje else None
        if self.agregacje is not None:
            if len(self.agregacje) != len(self.strides):
                raise ValueError("agregacje: oczekiwano %d wpisow, dostano %d"
                                 % (len(self.strides), len(self.agregacje)))
            if self.rc_pelne:
                # `torch.cat([h, stat])` daje uklad [h_a, h_b, s_a, s_b], a
                # konwencja parowania wymaga [pierwsza polowa | druga polowa],
                # czyli [h_a, s_a | h_b, s_b]. Przeplot robi `_przeplot_par`;
                # bez niego splot parowy wiazalby ZLE kanaly i ekwiwariancja
                # padlaby cicho.
                from .rc_ekwi import SplotParowy
                self.agr_lacz = nn.ModuleList(SplotParowy(2 * dim, dim, 1)
                                              for _ in self.strides)
            else:
                self.agr_lacz = nn.ModuleList(nn.Conv1d(2 * dim, dim, 1)
                                              for _ in self.strides)
            self.agr_brama_log = nn.Parameter(
                torch.full((len(self.strides),), float(math.log(0.1))))
        else:
            self.agr_lacz = None
            self.agr_brama_log = None
        self.bloki_poziomu = int(bloki_poziomu)
        if self.bloki_poziomu > 0:
            from .genomic_blocks import MultiScaleDilatedBlock
            self.bloki = nn.ModuleList(nn.ModuleList(
                MultiScaleDilatedBlock(dim, kernel_size=int(blok_jadro),
                                       dilations=(1,), expand=2,
                                       dropout=float(blok_dropout), bez_se=True)
                for _ in range(self.bloki_poziomu))
                for _ in self.strides)
            self.blok_skala_log = nn.Parameter(
                torch.full((len(self.strides),), float(torch.log(torch.tensor(0.1)))))
        else:
            self.bloki = None
            self.blok_skala_log = None
        # PELNE POLE RECEPCYJNE NA NAJGRUBSZEJ MAPIE.
        #
        # Zmierzone: stem k=5 -> (k=4,s=2) -> (k=4,s=2) -> (k=8,s=4) daje RF
        # 5 -> 8 -> 14 -> 42, czyli poziom "najbardziej globalny" widzi 42 z 249
        # zasad -- 17% sekwencji. Nazwa "most global" byla wiec deklaracja.
        #
        # Splot k=15 na mapie o 15 pozycjach POKRYWA JA CALA, wiec L0 widzi
        # wszystkie 249 zasad. Umieszczony PO petli `downs`, nie w srodku: gdyby
        # byl w srodku, kontekst przeszedlby przez kolejne `down` i zglobalizowal
        # takze L1 i L2, a to niszczy faktoryzacje prioru dyfuzji (lancuch
        # residualny `cur = nxt - interpolate(q.detach())` odcina GRADIENT, nie
        # WARTOSC). Tutaj globalny staje sie WYLACZNIE poziom 0.
        #
        # Brama startuje na 0,02, NIE na zerze: zerowa brama nie przepuszcza
        # gradientu w gore i gasi wlasne zrodlo -- blad popelniony tu czterokrotnie.
        # ============ BLAD, ZDIAGNOZOWANY CO DO PRZYCZYNY ============
        # ZMIERZONE na HypEncSkala_best (najgrubsza mapa, 15 pozycji):
        #     mapa PRZED kontekst    rzad_ucentr 8,46  wspolna 0,862  RMS  0,9159
        #     samo wyjscie kontekst  rzad_ucentr 2,75  wspolna 0,990  RMS 11,6766
        #     wklad z brama          rzad_ucentr 2,68  wspolna 0,996  RMS 27,2844
        #     mapa PO kontekst       rzad_ucentr 3,48  wspolna 0,996  RMS 27,9024
        #     DOMINACJA ||brama*kontekst|| / ||mapa|| = 29,79
        #
        # PRZYCZYNA MECHANICZNA: `kontekst` zawiera `nn.BatchNorm1d`, ktory
        # RESETUJE skale wyjscia niezaleznie od wejscia. Modul dostaje mape
        # o RMS 0,92 i zwraca sygnal o RMS 11,68 -- 12,7x wiecej JESZCZE PRZED
        # brama. Postac rezydualna `h + brama * f(h)` zaklada, ze `f(h)` jest
        # porownywalne z `h`; przy BatchNormie w srodku to zalozenie jest
        # FALSZYWE. Brama, startujac z 0,02, urosla do sredniej 1,2829 (64x)
        # i wklad doszedl do 30x mapy.
        #
        # SKUTEK: L0 nie kwantyzuje zgrubionej mapy sekwencji, tylko wyjscie
        # modulu wygladzajacego o rzedzie 2,75 i skladowej wspolnej 0,990.
        # Stad rzad L0 3,53 (a nie 8,46 mapy), wspolna 0,996, sonda liniowa
        # 0,0197 i 126 kodow stloczonych w stozku 4,5 stopnia.
        #
        # To jest TRZECI raz w tej sesji, gdy mala brama przed NIEOGRANICZONA
        # galezia konczy sie zagluszeniem sygnalu (poprzednio: glowica o sd 21,3
        # przy celu 2,15 oraz pasmo promienia z `tanh` nasyconym na 0,9996).
        #
        # NAPRAWA: wklad normalizowany do RMS mapy PER PROBKA, potem brama
        # ograniczona `tanh` i staly ulamek `udzial_kontekst`. Per PROBKA, nie
        # per batch -- to warunek konieczny, bo `encode_codes(x)` musi zostac
        # funkcja wylacznie `x`, inaczej API dyfuzji przestaje byc wykonalne.
        # Przy `udzial_kontekst: 0.0` sciezka jest BIT W BIT jak dotad.
        self.udzial_kontekst = float(udzial_kontekst)
        self.kontekst_typ = str(kontekst_typ)
        self.pelny_zasieg = bool(pelny_zasieg)
        if self.pelny_zasieg and self.kontekst_typ == "mamba":
            # SKAN DWUKIERUNKOWY jako modul globalny. Uzasadnienie liczbowe:
            # splot glebinowy k=15 jest WYGLADZANIEM i jego wyjscie ma rzad
            # ucentrowany 2,75 przy skladowej wspolnej 0,990 -- daje zasieg,
            # ale niskiego rzedu. Skan dwukierunkowy jest globalny i nie jest
            # wygladzaniem, wiec moze dac zasieg BEZ splaszczenia.
            # ZASTRZEZENIE, ktore musi byc w logu: SSM jako TRZON zmierzono
            # w tym repo na -0,022 Dev. Tu jest w innej roli (modul kontekstu
            # w enkoderze, jedna warstwa), ale to nie jest to samo co dowod.
            from .ssm import MambaStack
            self.kontekst = MambaStack(dim, n_blocks=int(kontekst_bloki),
                                       d_state=int(kontekst_stan), n_heads=4,
                                       expand=2, dropout=0.0, bidirectional=True)
        elif self.pelny_zasieg and self.rc_pelne:
            # WERSJA PAROWANA. Bez niej bieg RC musialby roznic sie od
            # `HypEncSkala` DWIEMA rzeczami naraz -- symetria PLUS zdjety
            # modul zasiegu -- czyli nie mierzylby symetrii, tylko ich sume.
            # `KontekstRC` ma ten sam uklad warstw i POLOWE wag; blad
            # ekwiwariancji 4,0e-16 wobec 1,4209 dla wersji swobodnej.
            from .rc_ekwi import KontekstRC
            self.kontekst = KontekstRC(dim, jadro=15)
        elif self.pelny_zasieg:
            self.kontekst = nn.Sequential(
                nn.Conv1d(dim, dim, 15, padding=7, groups=dim),
                nn.BatchNorm1d(dim), nn.SiLU(),
                nn.Conv1d(dim, dim, 1))
        else:
            self.kontekst = None
        if self.rc_pelne and self.pelny_zasieg and self.kontekst_typ == "mamba":
            # SKAN DWUKIERUNKOWY nie ma wersji parowanej i nie udajemy, ze ma:
            # stan rekurencyjny wymagalby zwiazania macierzy przejscia obu
            # kierunkow, co jest inna konstrukcja niz sprzezenie jadra.
            raise ValueError(
                "rc_pelne + kontekst_typ: mamba -- MambaStack nie jest "
                "parowany. Uzyj `kontekst_typ: splot` (jest `KontekstRC`) "
                "albo `pelny_zasieg: false`.")
        if self.rc_pelne and getattr(self, "bloki", None) is not None:
            # STRAZNIK NA FAKTYCZNYCH ATRYBUTACH, nie na kwargach. `kontekst_typ`
            # ma domyslna wartosc "splot", wiec warunek na kwargu zapalalby sie
            # ZAWSZE -- to ta sama rodzina bledow co `rc_stem`, ktory byl cichym
            # no-opem przez warunek na `in_ch`.
            raise ValueError(
                "rc_pelne wyklucza `bloki_poziomu`: MultiScaleDilatedBlock "
                "nie jest parowany, wiec zlamalby ekwiwariancje CICHO -- blad "
                "wyszedlby jako 1e-2 wygladajace na numeryke. Zmierzona "
                "wartosc tej galezi jest zerowa albo ujemna (HypEncAgr -0,003 "
                "wobec HypEncSkala mimo bram otwartych 10x), a RC ma zmierzone "
                "+0,00527. `kontekst` juz NIE jest wykluczony -- ma wersje "
                "parowana `KontekstRC` (blad 4,0e-16).")
        if self.kontekst is not None:
            # Przy naprawie brama idzie przez `tanh`, wiec init 0,5 daje
            # tanh(0,5) = 0,462 i wklad 0,462 * udzial. Model ujawnil, ze CHCE
            # duzego kontekstu (urosl 64x) -- naprawa nie zabiera mu tego,
            # tylko ogranicza do ulamka normy mapy zamiast 30-krotnosci.
            _b0 = 0.5 if self.udzial_kontekst > 0.0 else 0.02
            # BRAMA TEZ MUSI BYC PAROWANA. Jest PER KANAL, wiec przy `rc_pelne`
            # kanal `c` i jego para `c + dim/2` dostawaly rozne wartosci i
            # symetria ginela na MNOZENIU, nie na module -- czyli w miejscu,
            # ktore wyglada na niewinne. Trzymamy polowe i sklejamy w `forward`.
            _n = dim // 2 if self.rc_pelne else dim
            self.brama_kontekst = nn.Parameter(torch.full((1, _n, 1), _b0))
        else:
            self.brama_kontekst = None
        self.drop = nn.Dropout(float(enc_dropout)) if enc_dropout else None
        # SKIP CONNECTIONS w enkoderze: residuum wokol kazdego `down`, przez
        # splot 1x1 o tym samym stridzie. Bez nich sciezka od stemu do
        # najgrubszej mapy to trzy sploty z BatchNormem i SiLU pod rzad, wiec
        # gradient do stemu przechodzi przez trzy nieliniowosci -- a stem jest
        # RC-wiazany i to on niesie symetrie niciowa.
        # Skip NIE lamie reguly o mieszaniu przed kwantyzatorem: jest LOKALNY,
        # ma to samo pole recepcyjne co splot, ktory omija.
        if skip_enc and self.rc_pelne:
            # Skrot 1x1 tez musi byc parowany I spelniac ten sam warunek
            # parzystosci, inaczej symetria ginie na skrocie, a nie na
            # sciezce glownej -- i wyglada to na blad numeryczny, ktorym
            # nie jest. Jadra dobrano juz wyzej razem z `downs`.
            self.skip = self._rc_skips
        elif skip_enc:
            self.skip = nn.ModuleList(
                nn.Conv1d(dim, dim, 1, stride=s) for s in self.strides)
        else:
            self.skip = None

        # LEVEL 0 IS THE MOST GLOBAL.  Codebooks are indexed by abstraction, not
        # by spatial resolution: level 0 quantises the coarsest feature map and
        # is initialised NEAREST THE ORIGIN, deeper levels progressively further
        # out.  That placement is the hypothesis made structural — in hyperbolic
        # space the volume available at radius r grows like sinh^{n-1}(r), so a
        # shell near the origin holds few, general codes while an outer shell
        # holds exponentially many specific ones.  Regulatory grammar, if it is
        # hierarchical, should occupy exactly that arrangement.
        #
        # Whether it *stays* arranged that way is the measurement, not the
        # assumption: nothing constrains the radii during training, so
        # :meth:`hierarchy_report` reports what the data actually chose.
        radii = tuple(level_radii)[:self.n_levels]
        self.level_radii = radii
        # codes[] is given finest-first for continuity with the Euclidean stack;
        # reverse it so index 0 is the most global level.
        codes_by_level = tuple(codes)[::-1]
        # ======= ILE VQ REALNIE POWSTAJE (odtwarzalnosc strumienia RNG) =====
        # Przy drabinie stozkowej poziomy 1.. NIE sa kwantyzowane kodebookiem
        # na rozmaitosci -- robia to `KwantyzatorFSQ` / `KwantyzatorStyczny`,
        # a `self.vqs` jest zaraz nizej SKRACANE do jednego wpisu (`del
        # self.vqs; self.vqs = [_vq0]`). Zbudowanie tamtych `HyperbolicVQ`
        # tylko po to, zeby je porzucic, jest niewidoczne w liczbie parametrow
        # I WIDOCZNE w wyniku: `HyperbolicVQ.__init__` losuje kodebook
        # (`torch.randn(codebook_size, dim)`), wiec kazdy porzucony poziom
        # PRZESUWA strumien RNG dla wszystkiego, co powstaje pozniej --
        # w tym dla kwantyzatorow drabiny, trzonu i glowic. Ziarno 7898
        # dawalo wiec inny model przy `codes: [512, 256, 256, 128]` niz przy
        # `codes: [128]`, mimo ze OBA buduja identyczna architekture.
        #
        # ZMIERZONE (2026-09-10, `HyppeDeepSTARR`): jeden porzucony
        # `HyperbolicVQ(512)` przy tym samym ziarnie przesuwa nastepne
        # `torch.randn(1)` z -0,447071 na -0,560266, a trzy porzucone poziomy
        # (256+256+512) zjadaly 65 536 liczb. PO poprawce podpis md5
        # WSZYSTKICH parametrow modelu jest IDENTYCZNY dla `codes:
        # [512,256,256,128]` i `codes: [128]` (63a28b5b5081f4e9), przy tej
        # samej liczbie parametrow. Podpis samego kodebooka L0 byl zgodny
        # zawsze -- bo L0 powstaje PIERWSZY -- i dlatego sprawdzanie go
        # NIE JEST testem tej usterki.
        #
        # Poziom 0 jest budowany zawsze i PIERWSZY, wiec jego wpis jest bit
        # w bit ten sam co wczesniej. To jest warunek, przy ktorym ta poprawka
        # nie zmienia zadnego biegu bez drabiny.
        _n_vq = 1 if drabina else self.n_levels
        if drabina and len(codes_by_level) < 1:
            raise ValueError("codes musi miec co najmniej jeden wpis")
        if drabina and not fsq and len(codes_by_level) < self.n_levels:
            # `KwantyzatorStyczny` czyta `codes_by_level[_k]` dla k = 1..n-1,
            # wiec tam skrocona lista jest TWARDYM bledem, a nie oszczednoscia.
            raise ValueError(
                "drabina bez FSQ potrzebuje %d wpisow `codes` (kazdy poziom "
                "styczny ma wlasny kodebook), dostalem %d"
                % (self.n_levels, len(codes_by_level)))
        self.vqs = nn.ModuleList(
            HyperbolicVQ(dim, c, commit_weight=commit_weight, radius_scale=r,
                         frechet=frechet, learnable=learnable,
                         codebook_weight=codebook_weight,
                         enc_na_powloce=enc_na_powloce,
                         pasmo_r=pasmo_r,
                         przes_powloki=przes_powloki,
                         promien_specyf=promien_specyf).double()
            for c, r in zip(codes_by_level[:_n_vq], radii[:_n_vq]))
        # STRAZ NA PORZADEK POWLOK, gdy powyzsza petla nie zbudowala
        # wszystkich poziomow. Falsyfikator ponizej (`zakresy przesuniecia
        # powlok zachodza`) potrzebuje `promien_cel` KAZDEGO poziomu, a ten
        # jest liczony z wylosowanego kodebooka -- przy jednym VQ petla
        # `range(len(_pr) - 1)` jest pusta i falsyfikator CICHO znika.
        # Zamiast tego sprawdzamy to, co da sie sprawdzic bez modulow:
        # zadeklarowane `level_radii` musza rosnac, bo cala hipoteza tego
        # modelu mowi, ze powloka blizej origa trzyma kody OGOLNE.
        if _n_vq < self.n_levels:
            if any(b <= a for a, b in zip(radii, radii[1:])):
                raise ValueError(
                    "level_radii musi byc SCISLE rosnace (powloka blizej "
                    "origa = kody ogolniejsze), jest %s" % (radii,))
            if przes_powloki:
                # Przy jednym VQ margines `przes_max` liczy sie z listy
                # sasiadow, ktora jest PUSTA -> 0,0, czyli mechanizm umiera
                # po cichu. To ta sama klasa awarii co cztery flagi z
                # "wylaczanie podsystemow ma ciche fallbacki".
                raise ValueError(
                    "przes_powloki + drabina stozkowa: margines przesuniecia "
                    "powloki jest liczony z odstepow miedzy powlokami "
                    "SASIEDNICH `HyperbolicVQ`, a przy drabinie istnieje "
                    "tylko poziom 0 -- margines wyszedlby 0,0 i flaga bylaby "
                    "cichym no-opem")
        # MARGINES PRZESUNIECIA POWLOKI, liczony z FAKTYCZNYCH odstepow.
        #
        # Kazdy poziom moze przesunac swoja powloke o +-`przes_max`, zeby stale
        # zapotrzebowanie enkodera nie zjadalo pasma per sekwencja (zmierzone na
        # HypEncZ ep1: `tanh` nasycony w 99,9 % na L0 i 87,2 % na L2). Zeby
        # jednak poziomy nie mogly sie ZAMIENIC MIEJSCAMI, suma marginesow dwoch
        # sasiadow musi byc mniejsza od odstepu miedzy ich powlokami.
        #
        # Stala wpisana recznie tego nie gwarantuje: 0,35 przy bazach 1,4393 /
        # 2,2723 / 2,9539 dawalo zakresy [1,92; 2,62] i [2,60; 3,30], czyli
        # ZACHODZACE. Odstepy zaleza od losowej inicjalizacji kodebooka, wiec
        # margines musi byc z nich POLICZONY, a nie zgadniety.
        #
        # 0,40 odstepu do najblizszego sasiada daje sume 0,80 odstepu, czyli
        # zapas 20 %. Przy zmierzonych odstepach 0,8330 i 0,6816 wychodzi
        # 0,333 / 0,273 / 0,273, a zapotrzebowanie (0,15 i 0,13) zajmuje
        # polowe budzetu -- `tanh` zostaje w zakresie liniowym.
        with torch.no_grad():
            _pr = [float(v.promien_cel) for v in self.vqs]
        for _i, _v in enumerate(self.vqs):
            if not getattr(_v, "przes_powloki", False):
                _v.przes_max = 0.0
                continue
            _sas = [abs(_pr[_j] - _pr[_i]) for _j in (_i - 1, _i + 1)
                    if 0 <= _j < len(_pr)]
            _v.przes_max = 0.40 * min(_sas) if _sas else 0.0
        # Falsyfikator strukturalny: zakresy MUSZA byc rozlaczne. Blad tutaj
        # jest lepszy niz bieg, w ktorym hierarchia radialna cicho sie odwraca.
        for _i in range(len(_pr) - 1):
            _g = (_pr[_i] + self.vqs[_i].przes_max,
                  _pr[_i + 1] - self.vqs[_i + 1].przes_max)
            if _g[0] >= _g[1]:
                raise ValueError(
                    "zakresy przesuniecia powlok L%d i L%d zachodza (%.4f >= %.4f)"
                    % (_i, _i + 1, _g[0], _g[1]))
        # SKALA RESZTY -- jeden uczony skalar na PRZEJSCIE miedzy poziomami.
        # Parametryzacja logarytmiczna, zeby skala nie mogla przejsc przez zero
        # ani zmienic znaku (odejmowanie stalo by sie dodawaniem).
        # Inicjalizacja ZMIERZONYMI ilorazami ||mapa|| / ||up(q)|| z
        # HypEncI_best: 0,0265 (L0->L1) i 0,0542 (L1->L2). Start z 1,0 byl by
        # startem z awarii, a nie neutralnym punktem.
        _init = [0.0265, 0.0542]
        self.skala_reszty_log = nn.Parameter(torch.tensor(
            [math.log(_init[i] if i < len(_init) else _init[-1])
             for i in range(max(self.n_levels - 1, 1))], dtype=torch.float32))
        self._kor_reszty = []
        self._udzial_sekw = []
        self.dim = dim
        self._last_steps = None
        # `recon_bypass=False` ZAMYKA ciagly kanal obejsciowy: wejscie dekodera
        # staje sie w pelni skwantowane. Tozsamosc z[L2] + z[RES] == feats[0]
        # zachodzi z dokladnoscia 2,6e-07, wiec dekoder ma bezstratna kopie
        # enkodera i kody sa mu zbedne -- rekonstrukcja z samych kodow 0,29 przy
        # przypadku 0,25, a z samego residuum 0,94.
        self.recon_bypass = bool(recon_bypass)
        # `trunk_bypass=None` -> stare zachowanie, czyli to samo co dekoder.
        self.trunk_bypass = (bool(recon_bypass) if trunk_bypass is None
                             else bool(trunk_bypass))
        # Slot istnieje w `z`, gdy chce go KTORAKOLWIEK strona.
        self.slot_obecny = self.recon_bypass or self.trunk_bypass

        # ================== DRABINA STOZKOWA (opcjonalna) ==================
        # ZMIENIA TO, CO SIE KWANTYZUJE: poziom 0 kwantyzuje POZYCJE (punkt na
        # hiperboloidzie), poziomy dalsze USZCZEGOLOWIENIE (wektor STYCZNY),
        # a zlozenie jest geodezyjne:
        #     p_0     = embed_0[idx_0]
        #     p_{k+1} = expmap(p_k, PT(origo -> p_k, v_{k+1}))
        #
        # PO CO, W TRZECH ZMIERZONYCH PUNKTACH (`HypEncAgr_best`):
        #  1. Drabina REZYDUALNA nie jest hierarchia ZAWIERANIA:
        #     cos(kierunek dziecka, kierunek JEGO rodzica) = 0,0981 wobec
        #     0,0981 dla PAROWANIA LOSOWEGO. Dziecko kwantyzuje RESZTE
        #     rodzica, a reszta jest korekta, nie uszczegolowieniem. Tutaj
        #     `p_{k+1}` lezy w odleglosci geodezyjnej dokladnie `||v_{k+1}||`
        #     od `p_k`, wiec zawieranie jest z KONSTRUKCJI.
        #  2. Przy przypietym promieniu przypisanie kodu jest KOSINUSOWE
        #     (zgodnosc ze sfera 1,0000 / 0,9995 / 0,9901). Powod: kazdy
        #     poziom kwantyzuje wobec WLASNEJ powloki, wiec czlon
        #     `cosh(r_x)cosh(rho_k)` jest wspolny dla kandydatow i wypada
        #     z argmaxu. Krzywizna nie ma gdzie zadzialac. Tutaj dziala przez
        #     APERTURE: krok o dlugosci `d` z promienia `r` obejmuje kat
        #     ~`d/sinh(r)`, czyli ten sam kod znaczy 4,8x mniej na L2 niz na
        #     L0. Euklides tego nie ma, bo nie ma wyroznionego srodka.
        #  3. Rekonstrukcja NIE DOCIERALA DO KODEBOOKA: straight-through daje
        #     `points + (quant - points).detach()`, wiec kodebook widzial
        #     wylacznie commitment ciagnacy go DO POWLOKI (rozrzut promieni
        #     kodow zapadl do sd 0,0002 / 0,0023). Tutaj `p_1` i `p_2` sa
        #     funkcjami `embed_0[idx_0]` przez `expmap`, wiec gradient straty
        #     rekonstrukcyjnej plynie DO WPISOW kodebooka -- i to jest
        #     pierwszy raz, gdy `RiemannianAdam` ma co optymalizowac.
        #
        # FALSYFIKATOR TEJ KONSTRUKCJI (`raport()`): `cos_rodzic` musi
        # ODSTAWAC od `cos_losowy`. Jesli zostana rowne, zawieranie NIE
        # POWSTALO i konstrukcja jest ozdoba -- niezaleznie od tego, co
        # pokaze PCC.
        self.drabina = None
        if drabina:
            if self.n_levels < 2:
                raise ValueError("drabina stozkowa wymaga >= 2 poziomow")
            if self.kanal_promienia:
                # `promien_enk` powstaje w `HyperbolicVQ._na_powloke`; poziomy
                # styczne go nie maja, wiec `z` byloby o `n_levels - 1`
                # kanalow za waskie i wywalilo sie dopiero w dekoderze.
                raise ValueError(
                    "drabina + kanal_promienia: poziomy styczne nie licza "
                    "`promien_enk`, wiec kanal promienia istnialby tylko dla "
                    "L0 i `out_channels()` klamalby o szerokosci `z`")
            from .drabina_stozkowa import DrabinaStozkowa, KwantyzatorStyczny
            # Poziomy 1.. buduja sie PO calym `self.vqs`, wiec strumien RNG
            # dla poziomu 0 i dla wszystkiego, co powstalo wczesniej, jest
            # bit w bit taki sam jak w biegu bazowym. To jest warunek
            # porownywalnosci ablacji, nie kosmetyka.
            # APERTURA PER POZIOM. Jeden skalar dla wszystkich poziomow byl
            # zalozeniem, ktore pomiar odrzucil: `sonda_zawierania.py` na
            # `HypStozek_best` (ep19) daje rozrzut rodzicow 0,2657 (L0->L1)
            # i 0,5649 (L1->L2), czyli DWUKROTNIE rozny. Jedna apertura nie
            # moze wiec byc mniejsza od rozrzutu na obu poziomach naraz, a to
            # jest warunek, zeby zawieranie ograniczalo POZYCJE.
            _apert = drab_dlugosc
            if not isinstance(_apert, (list, tuple)):
                _apert = [float(_apert)] * (self.n_levels - 1)
            if len(_apert) != self.n_levels - 1:
                raise ValueError(
                    "drab_dlugosc jako lista musi miec %d wpisow (po jednym na "
                    "PRZEJSCIE), dostalem %d" % (self.n_levels - 1, len(_apert)))
            # WZMOCNIENIE PER POZIOM. Skalar jest rozgloszony, zeby stare
            # configi zachowaly sie bit w bit; lista musi miec dlugosc
            # `n_levels - 1` i rozjazd jest TWARDYM bledem, nie cichym
            # obcieciem -- inaczej brakujacy wpis dostalby domyslne 15,0
            # i poziom niosłby ~0,7 zadeklarowanych bitow.
            _wzm = (list(fsq_wzmocnienie)
                    if isinstance(fsq_wzmocnienie, (list, tuple))
                    else [float(fsq_wzmocnienie)] * (self.n_levels - 1))
            if len(_wzm) != self.n_levels - 1:
                raise ValueError(
                    "fsq_wzmocnienie: %d wartosci na %d poziomow stycznych"
                    % (len(_wzm), self.n_levels - 1))
            _kroki = []
            for _k in range(1, self.n_levels):
                if fsq:
                    from .fsq_styczny import (KwantyzatorFSQ,
                                              poziomy_dla_bitow)
                    # `fsq_bity` MOZE BYC LISTA per poziom styczny. Jednym
                    # skalarem to pokretlo z KONIECZNOSCI sciskalo L2 -- jedyny
                    # poziom, ktorego sciskac nie wolno. ZMIERZONE przypinaniem
                    # kierunkow post-hoc: L1 toleruje d=6->2 przy recon
                    # 0,999416, a L2 przy d=6->5 spada do 0,952054, czyli PONIZEJ
                    # progu 0,99. Wklady do rekonstrukcji: L2 ~0,70, L1 ~0,003.
                    # `[3, 9]` daje L1 (3,3) 3,1699 b/poz i L2 bez zmian:
                    # budzet 1480,7 b wobec 1873,8 (-21,0 %), -260 parametrow.
                    # Wzorzec listy per poziom maja juz `drab_dlugosc`
                    # i `fsq_wzmocnienie`, wiec to nie nowa konwencja.
                    _fb = (fsq_bity if isinstance(fsq_bity, (list, tuple))
                           else [fsq_bity])
                    _fb_k = int(_fb[min(_k - 1, len(_fb) - 1)])
                    _kroki.append(KwantyzatorFSQ(
                        dim, poziomy=poziomy_dla_bitow(_fb_k),
                        dlugosc=float(_apert[_k - 1]),
                        rc_pelne=self.rc_pelne,
                        waga_entropii=float(fsq_entropia),
                        wzmocnienie=float(_wzm[_k - 1]),
                        krok_promienia=float(drab_krok_promienia)))
                else:
                    _kroki.append(KwantyzatorStyczny(
                        dim, codes_by_level[_k],
                        dlugosc=float(_apert[_k - 1]),
                        commit_weight=commit_weight,
                        krok_promienia=float(drab_krok_promienia)))
            _vq0 = self.vqs[0]
            # `self.vqs` przestaje byc `nn.ModuleList`, zeby poziom 0 NIE byl
            # zarejestrowany DWA razy (`codes.vqs.0.*` i `codes.drabina.vq0.*`).
            # Podwojna rejestracja nie psuje uczenia, ale duplikuje kodebook
            # w `state_dict` -- a `ModelEMA` klasyfikuje wpisy po nazwie
            # i dtype, wiec dwa imiona tego samego bufora float64 to dokladnie
            # ta klasa usterki, ktora raz juz zdjela kodebook z rozmaitosci.
            del self.vqs
            self.drabina = DrabinaStozkowa(
                _vq0, _kroki,
                waga_stozka=float(waga_stozka),
                apertura=float(apertura_stozka))
            self.vqs = [_vq0]
            # NORMALIZACJA MAP PER POZIOM. Istniejaca umowa jest taka, ze
            # kanal ma RMS ~1: `HyperbolicVQ` zwraca `out[...,1:]/radius_scale`,
            # a `radius_scale` to DOKLADNIE `sinh(promien_cel)/sqrt(dim)`
            # (0,25 = sinh(1,4436)/8 przy dim 64, 0,6 = sinh(1,9436)/8).
            # Trzymamy te sama regule dla punktow drabiny, liczac PLANOWANY
            # promien poziomu jako `r_{k-1} + krok_promienia`.
            #
            # To jest STALA, nie statystyka batcha -- warunek konieczny, bo
            # cokolwiek zaleznego od skladu batcha w drodze do dekodera
            # rozjezdza `decode_codes` z przebiegiem wprzod.
            _r = float(_vq0.promien_cel)
            _sk = [float(_vq0.radius_scale)]
            for _ in range(1, self.n_levels):
                _r = _r + float(drab_krok_promienia)
                _sk.append(math.sinh(_r) / math.sqrt(float(dim)))
            self.register_buffer("norm_poz",
                                 torch.tensor(_sk, dtype=torch.float32))
            # DLUGOSCI MAP liczone Z KONFIGURACJI, nie zgadywane -- `z_indeksow`
            # potrzebuje ich przy dekodowaniu z samych indeksow, gdzie zadnej
            # mapy nie ma. Asercja wobec faktycznych dlugosci stoi w `forward`.
            if self.rc_pelne:
                _dl = [k[3] for k in self._rc_ksztalt][::-1]
            else:
                _dl, _L = [], int(seq_len_enc)
                for _s in self.strides:
                    _L = (_L + 2 * (_s // 2) - 2 * _s) // _s + 1
                    _dl.append(_L)
                _dl = _dl[::-1]
            self.dlugosci_poziomow = tuple(_dl)
            # `skala_reszty_log` OPISUJE LANCUCH REZYDUALNY, KTOREGO TU NIE MA.
            # Zostawiony bylby parametrem BEZ GRADIENTU (zmierzone: jedyny taki
            # w calym modelu), a `train_graph` bierze `model.codes.parameters()`
            # jako ZBIOR WSPOLNY dla targu Nasha -- czyli wpuscilby do targu
            # wiersz, ktorego zadne zadanie nie dotyka. Do tego log drukowalby
            # `skala_reszty` jako stala i czytalo by sie to jak zywy mechanizm,
            # ktory model "wylaczyl", a nie jak martwy wpis.
            del self.skala_reszty_log

    def out_channels(self):
        # PELNA szerokosc `z`. Uklad kanalow, w tej kolejnosci:
        #     [ q0 .. q_{L-1} | r0 .. r_{L-1} | slot ]
        #       dim*L kanalow    L kanalow       dim kanalow
        # Kazdy widok jest wiec CIECIEM PREFIKSU, a nie maska.
        return (self.dim * (self.n_levels + (1 if self.slot_obecny else 0))
                + self.kanaly_promienia())

    def kanaly_promienia(self):
        """Ile kanalow zajmuje promien enkodera: jeden na poziom albo zero."""
        return self.n_levels if self.kanal_promienia else 0

    def kanaly_kodow(self):
        """Szerokosc widoku "tylko kody" -- pierwsze `dim * n_levels` kanalow.

        BEZ promienia, i to nie jest przeoczenie. Ten widok ma jedna umowe:
        `decode_codes` musi umiec go ODTWORZYC Z SAMYCH INDEKSOW. Promien
        enkodera jest liczony PRZED kwantyzacja i z indeksow nie wynika, wiec
        dopisanie go tutaj zamienialoby rekonstrukcje "z samych kodow" w
        rekonstrukcje z kodow i ciagle podpowiedzi -- ta sama usterka, ktora
        `recon_bypass` juz raz w tym repo unewaznila (recon 0,9984 wygladajace
        na wlasnosc kodow, a bedace wlasnoscia kanalu ciaglego).
        """
        return self.dim * self.n_levels

    def kanaly_dekodera(self):
        """Ile kanalow widzi DEKODER. Bez slotu = dyfuzja wykonalna.

        Przy `recon_bypass: false` dekoder dostaje DOKLADNIE `tylko_kody(z)`,
        czyli tyle, ile produkuje `decode_codes`. Promien i slot zostaja poza
        -- oba sa wielkosciami enkodera, ktorych przy generowaniu nie ma.
        """
        if not self.recon_bypass:
            return self.kanaly_kodow()
        return self.out_channels()

    def kanaly_trzonu(self):
        """Ile kanalow widzi TRZON przez `code_proj`.

        Tu promien wchodzi, i tylko tu. Trzon NIE GENERUJE sekwencji, wiec
        wolno mu czytac ciagle wielkosci enkodera -- to jest dokladnie ten sam
        argument, ktory rozdzielil `recon_bypass` od `trunk_bypass` i ktory
        zmierzono jako lepszy od obu skrajnosci (+0,0097 / +0,0066 w ep1/ep2).
        """
        if not self.trunk_bypass:
            return self.kanaly_kodow() + self.kanaly_promienia()
        return self.out_channels()

    def tylko_kody(self, z):
        """Widok `z` bez promienia i bez slotu -- czyste kody, prefiks."""
        return z[:, :self.kanaly_kodow()]

    def widok_trzonu(self, z):
        """Widok dla trzonu: kody + promien, ze slotem tylko przy `trunk_bypass`.

        Osobna metoda, bo `tylko_kody` ma teraz WEZSZA umowe (odtwarzalnosc
        z indeksow). Sklejenie obu w jedna dawaloby albo promien dekoderowi,
        albo nic trzonowi.
        """
        return z if self.trunk_bypass else z[:, :self.kanaly_trzonu()]

    def _skaluj(self, t, size):
        """Skalowanie SYMETRYCZNE NA ODBICIE, gdy `rc_pelne`.

        `mode="nearest"` uzywa `floor`, wiec pod odwroceniem indeks zrodlowy
        NIE przechodzi na swoje odbicie i ekwiwariancja pada -- ZMIERZONE:
        L0 (bez reszty) 6,4e-07, a L1 i L2 (przez reszte, czyli przez to
        skalowanie) 2,1e-01 i 4,4e-01 w tym samym przebiegu.

        `mode="linear", align_corners=False` liczy `src = (i+0.5)*skala - 0.5`,
        dla ktorego zachodzi DOKLADNIE
            src(L-1-i) = L_zr - 1 - src(i),
        wiec skalowanie komutuje z odbiciem. To nie jest wygladzanie dla
        estetyki, tylko warunek konieczny.
        """
        if t.shape[-1] == size:
            return t
        if self.rc_pelne:
            return F.interpolate(t, size=size, mode="linear",
                                 align_corners=False)
        return F.interpolate(t, size=size, mode="nearest")


    def forward(self, x):
        if self.rc_pelne:
            # KODEBOOK DOMKNIETY NA ZAMIANE POLOWEK, PRZED kwantyzacja.
            # Ekwiwariancja enkodera daje h(rc x) = P(h(x)), ale INDEKS jest
            # ekwiwariantny dopiero wtedy, gdy kodebook jest zamkniety na P.
            # Bez tego symetria ginie dokladnie tam, gdzie powstaje latent,
            # a dyfuzja dostaje dwa rozne kody na te sama czasteczke.
            # Domykamy PRZED uzyciem, nie po: inaczej krok kwantyzuje jeszcze
            # niesparowanym kodebookiem. Koszt: kopia K/2 x (D+1) liczb.
            from .rc_ekwi import sparuj_kodebook
            for _v in self.vqs:
                sparuj_kodebook(_v.embed)
            if self.drabina is not None:
                # Poziomy styczne maja WLASNE domkniecie: ich kodebook trzyma
                # wektory, nie punkty, wiec rzut na hiperboloide bylby bledem
                # typu. `KwantyzatorFSQ.sparuj` jest jawnym no-opem (parowanie
                # siedzi w stalej ramie) -- metoda istnieje, zeby nie robic
                # tego przez `hasattr`, ktory pominalby prawdziwe domkniecie
                # po cichu.
                for _kr in self.drabina.kroki:
                    _kr.sparuj()
        h = self.stem(x)
        if self.bloki_przed is not None:
            # PELNA ROZDZIELCZOSC (249), przed jakimkolwiek stride -- to jest
            # cala rzecz w tym umiejscowieniu.
            for _b in self.bloki_przed:
                h = _b(h)
        if self.wieza is not None:
            # PELNE 249, PRZED stridem i PRZED kwantyzacja. Strumien 1D
            # wraca po atencji z biasem par; cechy par wchodza rzutem 1x1.
            _tc, self._wieza_aux, _h1 = self.wieza(h)
            h = _h1 + self.wieza_rzut(_tc)
        # WCZESNA MAPA zapamietana dla celu JEPA: pelne 249 pozycji,
        # PRZED stridem i PRZED kwantyzacja. To jest obiekt, ktory JEPA
        # ma odtwarzac -- reprezentacja, nie zasady.
        self._mapa_wczesna = h
        feats = []
        for i, down in enumerate(self.downs):
            _h_in = h
            h = down(h)
            if self.skip is not None:
                _sk = self.skip[i](_h_in)
                if _sk.shape[-1] != h.shape[-1]:
                    _sk = self._skaluj(_sk, h.shape[-1])
                h = h + _sk
            if self.drop is not None:
                h = self.drop(h)
            if self.agregacje is not None:
                # Statystyka liczona w TYM SAMYM oknie co splot ze stridem,
                # zeby galaz opisywala to samo miejsce sekwencji.
                _s = self.strides[i]
                if self.rc_pelne:
                    # OKNO O TEJ SAMEJ ARYTMETYCE CO SPLOT GLOWNY.
                    # `unfold(-1, 2s, s)` bez dopelnienia daje kafelkowanie,
                    # ktore pod odwroceniem przechodzi na siebie tylko przy
                    # `L % s == 0`. Dla DeepSTARR L = 249 jest NIEPARZYSTE,
                    # wiec warunek nie zachodzi i galaz statystyki lamala
                    # ekwiwariancje, ktora reszta enkodera juz miala --
                    # ZMIERZONE: bez tej galezi blad L0 spadal z 7,6e-02
                    # do 6,4e-07. Te same (k, p, s), ktore dobrano dla
                    # `downs`, daja okna symetryczne z konstrukcji.
                    _L0, _k, _pd, _ = self._rc_ksztalt[i]
                    _hp = F.pad(_h_in, (_pd, _pd))
                    _p = (_hp.unfold(-1, _k, _s) if _hp.shape[-1] >= _k
                          else _hp.unsqueeze(-1))
                else:
                    _k = 2 * _s
                    _p = (_h_in.unfold(-1, _k, _s)
                          if _h_in.shape[-1] >= _k else _h_in.unsqueeze(-1))
                _j = self.agregacje[i]
                if _j == "max":
                    _st = _p.amax(-1)
                elif _j == "std":
                    _st = _p.std(-1)
                elif _j == "zakres":
                    _st = _p.amax(-1) - _p.amin(-1)
                elif _j == "mean":
                    _st = _p.mean(-1)
                else:
                    raise ValueError("nieznana agregacja %r" % (_j,))
                if _st.shape[-1] != h.shape[-1]:
                    _st = self._skaluj(_st, h.shape[-1])
                _g = torch.exp(self.agr_brama_log[i]).to(h.dtype)
                if self.rc_pelne:
                    _d = h.shape[1] // 2
                    _we = torch.cat([h[:, :_d], _st[:, :_d],
                                     h[:, _d:], _st[:, _d:]], dim=1)
                else:
                    _we = torch.cat([h, _st], dim=1)
                h = h + _g * self.agr_lacz[i](_we)
            if self.bloki is not None:
                # Rezydualnie i ze skalarem: przy skali 0 model wraca do bazy
                # bit w bit, wiec zmiana nie moze zaszkodzic strukturalnie.
                _y = h
                for blk in self.bloki[i]:
                    _y = blk(_y)
                h = h + torch.exp(self.blok_skala_log[i]).to(h.dtype) * (_y - h)
            feats.append(h)
        # ATENCJA MOTYWOW -- na tej samej mapie i z tego samego powodu.
        # PRZED `kontekst`, zeby modul globalny widzial mape juz wzbogacona
        # o kombinacje czynnikow, a nie odwrotnie.
        if self.mot_l0 is not None and self._occ_kv is not None:
            feats[-1] = self.mot_l0(feats[-1], self._occ_kv, self._g_kv)
        # Kontekst PO petli -- tylko najgrubsza mapa staje sie globalna.
        if self.kontekst is not None:
            _h = feats[-1]
            _f = self.kontekst(_h)
            if self.udzial_kontekst > 0.0:
                # Normalizacja PER PROBKA: funkcja wylacznie tej sekwencji,
                # wiec `encode_codes` zostaje niezalezne od skladu batcha.
                _rh = _h.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
                _rf = _f.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
                _f = _f * (_rh / _rf)
                _w = self.udzial_kontekst * torch.tanh(self.brama_kontekst)
            else:
                _w = self.brama_kontekst
            if self.rc_pelne:
                _w = torch.cat([_w, _w], dim=1)
            feats[-1] = _h + _w * _f
            with torch.no_grad():
                # FALSYFIKATOR: dominacja modulu globalnego nad mapa.
                # Zmierzone przed naprawa: 29,79. PROG: musi spasc ponizej 0,5,
                # inaczej `kontekst` nadal NADPISUJE mape zamiast ja wzbogacac.
                self._dom_kontekst = float(
                    (_w * _f).pow(2).mean().sqrt() / _h.pow(2).mean().sqrt().clamp_min(1e-9))
                self._brama_kontekst = float(self.brama_kontekst.mean())

        # Level 0 = coarsest feature map = the most global statement about the
        # sequence.  Each subsequent level quantises only the RESIDUAL its
        # predecessor could not express, so refinements are nested by
        # construction and the hypothesis "each step is smaller than the last"
        # becomes a quantity that can be checked rather than asserted.
        with torch.no_grad():
            # Rzad efektywny KIERUNKOW per poziom -- falsyfikator `bloki_poziomu`.
            #
            # LICZONE CO `rzad_ef_co` KROKOW, NIE CO KROK. Zmierzone na
            # dendrite (batch 96, ablacja naprzemienna wobec zywego biegu):
            # ten blok to **2 167,7 ms z 3 095,6 ms kroku, czyli 70 %** --
            # przy 4 190 krokach na epoke **151 z 216 minut**. Wartosc idzie
            # do `band_report()`, czyli do JEDNEJ linii w logu na epoke.
            #
            # Dlaczego az tyle, skoro macierz ma 96x96: to `float64` na
            # karcie konsumenckiej (FP64 = 1/64 FP32) ORAZ fallback
            # cuSOLVERa. Macierz Grama kierunkow ZNORMALIZOWANYCH jest
            # z konstrukcji bliska singularnej, wiec szybka sciezka nie
            # zbiega (`failed to converge` w logu) i wchodzi metoda
            # dokladniejsza, a przez to duzo wolniejsza.
            #
            # PULAPKA, W KTORA SAM WPADLEM: benchmark na macierzy
            # z LOSOWYCH kierunkow daje 30 ms i mowi "to nic nie kosztuje".
            # Losowa macierz jest dobrze uwarunkowana, wiec fallback sie
            # NIE URUCHAMIA -- mierzy sie wtedy inna sciezke kodu niz ta,
            # ktora leci w biegu. Roznica wyszla **24x**.
            self._rzad_krok = getattr(self, "_rzad_krok", 0) + 1
            _co = int(getattr(self, "rzad_ef_co", 200) or 0)
            if not (_co <= 0 or self._rzad_krok % _co == 1
                    or getattr(self, "_rzad_ef", None) is None):
                pass                     # zostaje ostatnia policzona wartosc
            else:
                self._rzad_ef = []
                for _f in feats[::-1]:
                    _u = _f.detach().float().transpose(1, 2).reshape(-1, _f.shape[1])
                    _u = _u / _u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    if _u.shape[0] > 2048:
                        _u = _u[torch.randperm(_u.shape[0], device=_u.device)[:2048]]
                    _c = (_u.t() @ _u).double() / _u.shape[0]
                    # DIAGNOSTYKA NIE MOZE ZABIC BIEGU. `eigvalsh` (syevd)
                    # wywalilo trening po bramce, ktora przeszla:
                    # `linalg.eigh: The algorithm failed to converge because
                    # the input matrix is ill-conditioned or has too many
                    # repeated eigenvalues (error code: 46)`. Macierz Grama
                    # kierunkow ZNORMALIZOWANYCH jest z konstrukcji bliska
                    # singularnej, a po wpieciu wiezy do enkodera przestala
                    # sie miescic w tolerancji cuSOLVERa.
                    # Trzy poprawki: grzbiet na przekatnej, `svdvals` (gesvd,
                    # znacznie odporniejszy, a dla symetrycznej PSD wartosci
                    # singularne ROWNAJA sie wlasnym), i przechwycenie wyjatku
                    # z jawnym `nan` -- brak liczby jest widoczny w logu,
                    # a ubity bieg to strata epok.
                    _c = _c + 1e-10 * torch.eye(_c.shape[0], dtype=_c.dtype,
                                                device=_c.device)
                    try:
                        _ev = torch.linalg.svdvals(_c).clamp_min(0)
                    except Exception:
                        try:
                            _ev = torch.linalg.svdvals(_c.cpu()).clamp_min(0)
                        except Exception:
                            self._rzad_ef.append(float("nan"))
                            continue
                    _p = _ev / _ev.sum().clamp_min(1e-30)
                    _p = _p[_p > 1e-12]
                    self._rzad_ef.append(float(torch.exp(-(_p * _p.log()).sum())))
        # MAPY WSZYSTKICH POZIOMOW dla JEPA HIERARCHICZNEJ. `feats` idzie
        # od najdrobniejszej; `by_level` odwraca. Trzymamy oba konce, bo
        # cel JEPA ma byc liczony NA KAZDEJ SKALI osobno.
        self._mapy_poziomow = list(feats)
        by_level = feats[::-1]                       # coarsest first
        quants, indices, steps = [], [], []
        self._kor_reszty = [float("nan")] * max(self.n_levels - 1, 0)
        self._udzial_sekw = [float("nan")] * max(self.n_levels - 1, 0)
        # `last_indices` wystawione dla `HypVQNet.encode_codes`: bez tego nie ma
        # jak dostac indeksow bez powtarzania calej petli kwantyzacji na zewnatrz.
        self.last_indices = indices
        loss = None
        if self.drabina is not None:
            # ZLOZENIE GEODEZYJNE zamiast lancucha rezydualnego. `mapy` ida
            # od NAJGRUBSZEJ, tak samo jak `by_level`, wiec kolejnosc poziomow
            # jest ta sama co w petli ponizej i wszystko, co czyta `indices`,
            # dziala bez zmiany.
            # MAPY W float32. `DrabinaStozkowa.forward` podnosi do float64
            # tylko poziom 0, bo tylko tam kodebook lezy na hiperboloidzie
            # (arytmetyka Minkowskiego odejmuje dwie bliskie duze liczby).
            # Kwantyzatory styczne dobieraja dtype same: `KwantyzatorStyczny`
            # rzutuje na dtype kodebooka, a `KwantyzatorFSQ` liczy w float32,
            # bo jego rama jest float32 -- podanie double zabijalo go
            # na `skala_we`.
            _pkt, _idx = self.drabina(list(by_level))
            indices.extend(_idx)
            loss = self.drabina.loss
            self._punkty = _pkt
            if tuple(int(p.shape[1]) for p in _pkt) != self.dlugosci_poziomow:
                # Asercja, nie ufnosc: `dlugosci_poziomow` policzone w
                # `__init__` sa JEDYNYM zrodlem dlugosci dla `decode_codes`,
                # gdzie zadnej mapy nie ma. Rozjazd tutaj oznacza, ze dyfuzja
                # dekodowalaby na innej siatce niz trening -- awaria cicha,
                # bo wynik nadal wyglada jak sekwencja.
                raise RuntimeError(
                    "dlugosci poziomow %r != policzone w __init__ %r"
                    % (tuple(int(p.shape[1]) for p in _pkt),
                       self.dlugosci_poziomow))
            # Punkt -> mapa kanalow, ta sama umowa co `HyperbolicVQ.forward`:
            # czesc PRZESTRZENNA podzielona przez `sinh(r)/sqrt(dim)`, czyli
            # RMS ~1 na kanal. Wspolrzedna czasowa NIE wchodzi -- jest funkcja
            # czesci przestrzennej (`x0 = sqrt(1 + ||x_s||^2)`), wiec byloby to
            # dopisanie kanalu o zerowej informacji warunkowej.
            quants = [(p[..., 1:].transpose(1, 2).float()
                       / self.norm_poz[k].to(torch.float32))
                      for k, p in enumerate(_pkt)]
        else:
            cur = by_level[0]
            for k in range(self.n_levels):
                q, idx = self.vqs[k](cur.double())
                q = q.float()
                quants.append(q)
                indices.append(idx)
                l = self.vqs[k].loss.float()
                loss = l if loss is None else loss + l
                # Signal remaining at this level, i.e. what its predecessors could
                # not explain.  Measured on `cur` rather than on `q`: the quantised
                # output is scaled by each level's own radius_scale, so its norm
                # reports the initialisation rather than the refinement.  Under the
                # hierarchy hypothesis this residual DECREASES with depth -- each
                # level has strictly less left to say than the one before.
                if k + 1 < self.n_levels:
                    nxt = by_level[k + 1]
                    # `level_grad` decyduje, czy poziomy WSPOLPRACUJA gradientowo.
                    # Domyslnie `q.detach()`: wartosc plynie w dol, gradient nie --
                    # kazdy poziom uczy sie wylacznie ze swojej wlasnej straty, a
                    # hierarchia jest utrzymana WYLACZNIE strukturalnie.
                    # Przy `True` strata poziomu k+1 moze przesuwac kodebook
                    # poziomu k. RYZYKO, ktore trzeba miec w logu: hierarchia
                    # top-down byla zmierzona jako 47/38/40% wyjasnionej wariancji
                    # wobec 46/14/12% dla bottom-up, a gradient plynacy w gore jest
                    # dokladnie kanalem, ktorym drabina moze sie odwrocic.
                    # Falsyfikator: `res_frac` musi MALEC z poziomem.
                    _q = q if getattr(self, "level_grad", False) else q.detach()
                    up = self._skaluj(_q, nxt.shape[-1])
                    # ============ WYROWNANIE SKALI PRZED ODJECIEM ============
                    # ZMIERZONE na HypEncI_best (haki wprost na vq.forward):
                    #   L0 -> L1  ||cur|| 1,0006  ||up(q)|| 0,9904  ||mapa|| 0,0263
                    #   L1 -> L2  ||cur|| 1,0031  ||up(q)|| 0,9986  ||mapa|| 0,0541
                    #   kor(cur, -up(q)) = 0,9997 i 0,9985
                    # Czyli 97,4 % i 94,6 % reszty to WLASNY KOD poprzedniego
                    # poziomu ze znakiem minus, a nie sekwencja. Drabina
                    # kwantyzowala sama siebie.
                    #
                    # PRZYCZYNA JEST SKALOWA. Przy `enc_na_powloce` `_na_powloke`
                    # ustawia norme czesci przestrzennej na `sinh(r_k)` NIEZALEZNIE
                    # od wejscia, a `forward` zwraca `out[...,1:] / radius_scale`.
                    # Norma wyjscia jest wiec STALA sinh(r_k)/radius_scale_k, rowna
                    # 1,99/0,25 ~ 4,83/0,6 ~ 9,64/1,2 ~ 8 na wektor (RMS ~1,0 przy
                    # dim 64), podczas gdy mapa po BatchNorm+SiLU ma RMS 0,026-0,054.
                    # Odejmowany kod byl 20-40x wiekszy od mapy.
                    #
                    # `skala_reszty` to JEDEN uczony skalar na przejscie,
                    # inicjowany zmierzonym ilorazem. NIE jest statystyka batcha --
                    # to warunek konieczny, bo cokolwiek zaleznego od skladu batcha
                    # w sciezce sterujacej kwantyzacja lamie niezmiennik
                    # `decode_codes` (przypisanie kodu przestaje byc funkcja samej
                    # sekwencji) i zabija dyfuzje.
                    up = up * self.skala_reszty[k].to(up.dtype)
                    cur = nxt - up
                    with torch.no_grad():
                        # FALSYFIKATOR, mierzy DOKLADNIE to, co ta zmiana naprawia.
                        # `kor_reszty` blisko 1 znaczy, ze reszta nadal jest
                        # zanegowanym poprzednim kodem. `udzial_sekw` to
                        # ||mapa|| / ||cur||: przed naprawa 0,026 i 0,054.
                        _c = cur.detach().float().reshape(-1)
                        _u = (-up).detach().float().reshape(-1)
                        _c = _c - _c.mean(); _u = _u - _u.mean()
                        self._kor_reszty[k] = float(
                            (_c * _u).sum() / (_c.norm() * _u.norm()).clamp_min(1e-9))
                        self._udzial_sekw[k] = float(
                            nxt.detach().float().pow(2).mean().sqrt()
                            / cur.detach().float().pow(2).mean().sqrt().clamp_min(1e-9))

        self.loss = loss
        self.indices = indices

        # Align everything to the FINEST resolution for the decoder; coarse
        # levels are upsampled, never the reverse, so nothing positional is lost.
        target = feats[0].shape[-1]
        aligned = [self._skaluj(q, target) for q in quants]
        # Contribution magnitude per level, measured AFTER alignment so every
        # level is compared in one common frame (finest resolution, and each
        # level's radius_scale already divided out inside HyperbolicVQ).
        # Earlier attempts measured the quantised output's raw RMS -- dominated
        # by radius_scale -- and then variance explained against `by_level[k]`,
        # which is invalid for k>0 because those levels quantise RESIDUALS, not
        # feature maps.  Both reported numbers that had nothing to do with the
        # hypothesis.  This is the quantity the hypothesis is actually about:
        # how much each successive level ADDS to the reconstruction.
        self._last_steps = [float(a.detach().float().pow(2).mean().sqrt()) for a in aligned]
        if self.kanal_promienia:
            # PROMIEN ENKODERA, jeden kanal na poziom.
            #
            # ZRODLEM JEST `promien_enk`, NIE `radius(st_points)`. Straight-
            # through zwraca `points + (quant - points).detach()`, wiec
            # WARTOSCIA `st_points` jest wpis kodebooka, a promien kodebooka
            # jest przypinany co epoke (rozrzut 0,0048 / 0,0087 / 0,0260).
            # Co wiecej, `HyperbolicVQ.forward` zwraca `out[..., 1:] /
            # radius_scale` BEZ normalizacji, wiec NORMA tego wycinka JUZ JEST
            # promieniem kodu z dokladnoscia do stalej per poziom -- kanal z
            # `st_points` bylby kopia czegos, co dekoder ma od zawsze.
            # `promien_enk` to promien PRZED kwantyzacja, czyli jedyna
            # wspolrzedna hiperboliczna, ktorej za kwantyzatorem nie ma.
            for _v in self.vqs:
                _p = getattr(_v, "promien_enk", None)
                if _p is None:
                    raise RuntimeError(
                        "kanal_promienia wlaczony, a poziom nie zapisal "
                        "`promien_enk` -- `z` mialoby za malo kanalow")
                # [B, L, 1] -> [B, 1, L]
                _r = _p.detach().float().transpose(1, 2)
                # Skala poziomu, nie surowe radiany: powloki stoja na 1,44 /
                # 2,27 / 2,95, wiec bez centrowania trzy kanaly roznilyby sie
                # glownie STALA i `code_proj` musialby ja odjac sam.
                _pc = float(_v.promien_cel)
                _r = (_r - _pc) / max(math.sinh(_pc), 1e-6)
                # `_skaluj`, nie `nearest` na sztywno: bez `rc_pelne` jest to
                # ta sama operacja bit w bit, z `rc_pelne` jedyna, ktora
                # komutuje z odbiciem.
                _r = self._skaluj(_r, target)
                aligned.append(_r)
                self._promien_kanal = float(_r.std())
        if self.slot_obecny:
            # CIAGLE OBEJSCIE. Ta jedna linia sprawia, ze
            # z[L2] + z[RES] == feats[0] z dokladnoscia 2,6e-07. Kto go WIDZI,
            # decyduja `recon_bypass` (dekoder) i `trunk_bypass` (trzon) --
            # tutaj tylko powstaje.
            # `aligned[-1]` przestalo byc q2 po dopisaniu kanalow promienia,
            # wiec slot musi brac OSTATNI KOD, a nie ostatni element listy.
            _ost = aligned[self.n_levels - 1]
            aligned.append(feats[0] - _ost.detach())
        return torch.cat(aligned, dim=1)

    @property
    def skala_reszty(self):
        """Dodatnia skala reszty per przejscie; `exp` nie przechodzi przez zero.

        `None` przy drabinie stozkowej: tam nie ma lancucha rezydualnego, wiec
        nie ma czego skalowac, a parametr zostal usuniety, zeby nie trafic do
        zbioru wspolnego Nasha bez gradientu.
        """
        _p = self._parameters.get("skala_reszty_log", None)
        return None if _p is None else torch.exp(_p)

    def perplexities(self):
        out = [vq.perplexity() for vq in self.vqs]
        if self.drabina is not None:
            # Kroki styczne raportuja pod wlasnymi nazwami (`perplexity`
            # w `KwantyzatorStyczny`, `perpl` w `KwantyzatorFSQ`). Brak
            # liczby jest tu grozniejszy niz zla nazwa: warunkiem NIEZBEDNYM
            # alarmu zapadu jest spadek wykorzystania kodow, wiec poziom bez
            # zadnej liczby jest poziomem, na ktorym alarm jest slepy.
            for _kr in self.drabina.kroki:
                _r = _kr.raport()
                out.append(_r.get("perplexity", _r.get("perpl", float("nan"))))
        return out

    def radii(self):
        # TYLKO poziom 0 ma kodebook NA ROZMAITOSCI. Kodebooki styczne leza
        # w przestrzeni stycznej w origo i "promien kodu" nie jest tam
        # zdefiniowany -- zwracanie ich normy pod ta nazwa mieszaloby dwie
        # rozne wielkosci w jednej kolumnie logu. Promienie ZLOZONYCH punktow
        # sa w `raport()` drabiny, pod nazwa `promien`.
        return [vq.radii() for vq in self.vqs]

    def raport_drabiny(self):
        """Falsyfikatory drabiny stozkowej albo `None`, gdy jej nie ma."""
        return None if self.drabina is None else self.drabina.raport()


@register_model("HypVQNet")
class HypVQNet(nn.Module):
    """Occupancy trunk for activity, hyperbolic codebook for reconstruction."""

    def __init__(self, vocab=None, seq_len=249, vq_dim=64, strides=(2, 2, 4),
                 codes=(512, 256, 128), commit_weight=0.25, rc_stem=True,
                 level_radii=(0.25, 0.6, 1.2),
                 recon_weight=0.15, occ_channels=128, n_learned=32,
                 trunk_dim=224, n_blocks=5, kernel_size=5, dilations=(1, 2, 4, 8),
                 graph_dim=128, gat_layers=2, heads=4, hidden_dim=256,
                 dropout=0.2, task_weights=(1.0, 1.0),
                 rc_filters=False, occ_center=False, **kwargs):
        super().__init__()
        # ======= `seq_len` JAKO ATRYBUT, nie tylko kwarg =====================
        # `decode_codes` konczy sie interpolacja do `int(length or getattr(
        # self, "seq_len", 249))`. Atrybut NIE BYL ustawiany, wiec `getattr`
        # zwracalo domyslne 249 -- ZAWSZE, niezaleznie od configu. Dla
        # DeepSTARR to zgodne przypadkiem (config podaje 249), ale kazdy inny
        # `seq_len` dawal dyfuzje interpolowana do zlej dlugosci, przy
        # poprawnie wygladajacych logitach i bez ani jednego ostrzezenia.
        # Pulapka utajona -- dzisiaj bez skutku, i dokladnie dlatego nie
        # zostawia sladu, po ktorym dalo by sie ja znalezc po awarii.
        self.seq_len = int(seq_len)
        from motif_graph import MotifVocabulary
        from .motif_occupancy import ContinuousPWMScanner, OccupancyRefiner

        self._vocab = MotifVocabulary.from_config(vocab)
        v = len(self._vocab)
        # WYROWNANIE DO SRODKA MOTYWU. Pelne uzasadnienie i pomiary w
        # `ContinuousPWMScanner._na_srodek`. W skrocie: przy wyrownaniu do 5'
        # kazdy kanal przesuwa sie pod RC o wlasne w-1 (zmierzone rozjechanie
        # 8,75e-01, docstring `RCConv1d`), wiec dla oblozenia NIE ISTNIEJE
        # poprawny operator RC -- i to jest jedyna przyczyna, dla ktorej brama
        # boczna drabiny musiala byc PO enkoderze. Po wyrownaniu do srodka
        # operatorem jest sam `flip`, zmierzone max|occ(rc x) - flip(occ x)| =
        # 0,0 DOKLADNIE na wszystkich 20 kanalach.
        # Flagi PRZED skanerem i `motyw_proj`, bo one ich uzywaja.
        self.motywy_do_latentu = bool(kwargs.get("motywy_do_latentu", False))
        self._motyw_ch = int(kwargs.get("motyw_ch", 8))
        # USTALANE TUTAJ, przed `self.codes`: te dwie flagi decyduja o SZEROKOSCI
        # wejscia enkodera, a `HyperbolicCodeStack` dostaje ja jako `in_ch`.
        # Ustawienie ich pozniej dawalo `StemMotyw zbudowany na 4+32 kanalow,
        # dostal 276` -- czyli rozjazd `_codes_input` z `_encoder_in_channels`,
        # dokladnie ten, o ktorym ostrzega docstring tej pary.
        self.kontekst_do_latentu = bool(kwargs.get("kontekst_do_latentu", False))
        self._graf_ch = int(kwargs.get("graf_do_latentu_ch", 16))
        # `occ_channels` jest NAZWANYM argumentem `__init__`, wiec NIE MA go
        # w `kwargs` -- `kwargs.get("occ_channels", 128)` zwracalo domyslne 128
        # przy configu z 256 i dawalo rozjazd szerokosci (148 wobec 276),
        # ktory wychodzil dopiero na pierwszym batchu w `StemMotyw`.
        self._occ_ch_lat = int(occ_channels)
        # ============ STRAZ: rc_pelne + motywy_do_latentu WYMAGA occ_center ==
        # ZMIERZONE, i to jest awaria, ktora WYGLADA jak numeryka. Bez
        # `occ_center` skaner raportuje na 5' koncu motywu, wiec kazdy kanal
        # oblozenia przesuwa sie pod RC o WLASNE w-1 i poprawny operator RC dla
        # oblozenia NIE ISTNIEJE. Sciezka motywowa `StemRC` jest zainicjowana
        # ZEREM, wiec przy starcie nie wnosi nic i blad ekwiwariancji wynosi
        # 2,4e-07 -- czyli wyglada na poprawny. Po TRZECH krokach optymalizatora
        # `w_mot` przestaje byc zerem i blad skacze do 4,7e-03 (L0) i 4,2e-02
        # (L2), a zgodnosc indeksow z para spada 1,0000 -> 0,9802 / 0,9800.
        #
        # Bez tej strazy bieg wygladalby na ekwiwariantny w kazdym tescie
        # robionym na SWIEZYM modelu i nie byl nim ani przez jedna epoke.
        # Ta sama klasa cichej awarii co `side_exact` w rodzinie widmowej.
        if (self.motywy_do_latentu and bool(kwargs.get("rc_pelne", False))
                and not bool(occ_center)):
            raise ValueError(
                "rc_pelne + motywy_do_latentu wymaga occ_center: true. Przy "
                "wyrownaniu do 5' konca kazdy kanal oblozenia przesuwa sie pod "
                "RC o wlasne w-1, wiec tor motywowy stemu NIE jest "
                "ekwiwariantny. Wyglada poprawnie tylko dlatego, ze `w_mot` "
                "startuje ZEREM: blad 2,4e-07 na starcie i 4,7e-03 po trzech "
                "krokach. Albo occ_center: true, albo motywy_do_latentu: false.")
        self.scanner = ContinuousPWMScanner(
            self._vocab, center_align=bool(occ_center))
        # NORMALIZACJA OBLOZENIA JAKO KLUCZ, nie jako stala. `OccupancyRefiner`
        # ma parametr `norm` od poczatku, ale nikt go tu nie przekazywal --
        # czyli wybor byl zaszyty, a jego wlasny komentarz zapisuje, ze przy
        # 296 profilach JASPAR wersja ze statystyka batcha ROZJECHALA SIE TRZY
        # RAZY: strata treningowa i czlon regularyzacji spadaly plynnie, a
        # walidacja eksplodowala z 2,50 do 6,25 MIEDZY EPOKA 8 I 15. To jest
        # podpis running statistics rozjezdzajacych sie z batchowymi, a wiecej
        # kanalow to wiecej miejsca na to.
        # Domyslna zostaje "batch", zeby kazdy istniejacy config i checkpoint
        # budowal sie BIT W BIT. `occ_norm: instance` to ta poprawka.
        self.refiner = OccupancyRefiner(
            v, norm=str(kwargs.get("occ_norm", "batch")))
        # Waski rzut oblozenia do latentu. Waski celowo: to nie ma byc drugi
        # trzon, tylko wskazowka, KTORE pozycje sa motywowe.
        #
        # ======= NIE POWSTAJE PRZY `kontekst_do_latentu` =====================
        # Jedynym wolajacym jest `_codes_input`, i to w galezi ZA warunkiem
        # `if not self.kontekst_do_latentu`. Przy `kontekst_do_latentu: true`
        # enkoder dostaje PELNE `_h_occ` (128 kanalow) plus rzut grafu, wiec
        # ta warstwa powstaje i nie jest wolana ANI RAZU -- 2 376 wag
        # (296 profili JASPAR x 8 + bias) pod weight decay, bez gradientu.
        #
        # SPRZEZENIE, KTOREGO TA POPRAWKA NIE RUSZA (i to jest jej warunek):
        # `_szerokosc_wejscia_enkodera()` przy `kontekst_do_latentu` zwraca
        # `4 + occ_channels + graf_ch` i flagi `motywy_do_latentu` NIE CZYTA.
        # Dlatego zdejmowana jest WARSTWA, a nie flaga: gdyby zdjac flage,
        # `in_ch` spadloby z 148 na 4, warunek `rc_stem and in_ch == 4`
        # w `genomic_blocks.py:562` zrobilby sie PRAWDZIWY i stem zmienilby
        # sie na `RCConv1d` -- czyli jedna zmiana w configu przestawilaby
        # model w dwoch miejscach. ZMIERZONE po tej poprawce: in_ch = 148
        # bez zmian, `RCConv1d w stemie: False` bez zmian.
        self.motyw_proj = (
            nn.Conv1d(v, self._motyw_ch, 1)
            if (self.motywy_do_latentu and not self.kontekst_do_latentu)
            else None)
        self.task_weights = tuple(float(t) for t in task_weights)
        self.recon_weight = float(recon_weight)

        # --- DWA DETACHE, DWA NIEZALEZNE POKRETLA ---
        # `latent_grad` -- czy strata AKTYWNOSCI dochodzi do enkodera kodow.
        #   Zmierzone przy 'none': ||d Dev/d enkoder|| = ||d Hk/d enkoder|| =
        #   0.000000e+00 (dokladne zero, 16 tensorow / 66 912 wag). Przy tej
        #   wartosci zbior parametrow WSPOLNYCH miedzy rekonstrukcja a Dev/Hk
        #   jest pusty, wiec KAZDA metoda wielozadaniowa (Nash, CAGrad, PCGrad)
        #   degeneruje sie do normalizatora norm. Zdjecie tego detachu jest
        #   warunkiem koniecznym Nasha, nie jego alternatywa.
        # `level_grad` -- czy poziomy wspolpracuja gradientowo (patrz lancuch
        #   residualny w HyperbolicCodeStack).
        self.latent_grad = str(kwargs.get("latent_grad", "none"))
        if self.latent_grad not in ("none", "codes", "all"):
            raise ValueError("latent_grad: 'none' | 'codes' | 'all', dostano %r"
                             % (self.latent_grad,))
        self._z_last = None
        self._realnosc = None
        # Wymuszenie hierarchii radialnej. Domyslnie WYLACZONE, zeby biegi
        # sprzed poprawki byly odtwarzalne bit w bit.
        self.renorm_shells = bool(kwargs.get("renorm_shells", False))
        self._hyp_heads_cfg = dict(
            wlacz=bool(kwargs.get("hyp_heads", False)),
            n_proto=tuple(kwargs.get("hyp_n_proto", (32, 48, 64))),
            hidden=int(kwargs.get("hyp_hidden", 128)),
            tau_skala=float(kwargs.get("hyp_tau_skala", 0.02)),
            brama=float(kwargs.get("hyp_gate_init", 0.02)),
            attn=bool(kwargs.get("attn_codes", False)),
            # ============ PODGALAZ `realnosc` -- DOMYSLNIE NIE POWSTAJE ======
            # Cel tej glowicy (BCE prawdziwa-vs-przetasowana) liczy
            # `train_graph.py` i WYLACZNIE przy `training.negatywy_frac > 0`.
            # Model tego klucza NIE WIDZI (jest w bloku `training`, a
            # `registry.build_model` przekazuje tylko `model.kwargs`), wiec
            # nie da sie tego wywnioskowac -- trzeba zadeklarowac.
            #
            # ZMIERZONE na `HyppeDeepSTARR` (pomiar, nie oszacowanie):
            # `glowice=("dev","hk","realnosc")` przy `geom_per_cel: true` daje
            # trzecia KOMPLETNA kopie proj/attn/odczyt/mlp -- 61 836 wag, z tego
            # 14 560 to prototypy na hiperboloidzie, czyli 24,6 % wszystkiego,
            # co widzi `RiemannianAdam` (14 560 z 59 280). Bez negatywow ta
            # kopia nie ma ZADNEJ drogi do zadnej straty: jedynym konsumentem
            # `o["realnosc"]` jest `self._realnosc`, ktore czyta wylacznie blok
            # `negatywy_frac` w `train_graph`. To jest ten sam wzorzec, ktory
            # w tym repo zdjal `lat_brama` do DWOCH wpisow zamiast trzech
            # ("trzeci wpis bylby parametrem bez gradientu i pojawilby sie
            # w kazdym raporcie jako stala, udajac zmierzony zerowy wklad").
            #
            # Domyslna `False` jest wiec DOMYSLNA POPRAWNA, nie zgodna:
            # config, ktory chce negatywow, musi powiedziec to modelowi, a
            # `train_graph` ma na odwrotna pomylke (negatywy bez glowicy)
            # twarda asercje z 2026-09-09.
            realnosc=bool(kwargs.get("glowica_realnosci", False)))

        # --- reconstruction path: hyperbolic codes + decoder ---
        self.codes = HyperbolicCodeStack(
            vq_dim, strides, codes, commit_weight, rc_stem, tuple(level_radii),
            recon_bypass=bool(kwargs.get("recon_bypass", True)),
            trunk_bypass=kwargs.get("trunk_bypass", None),
            frechet=bool(kwargs.get("vq_frechet", False)),
            learnable=bool(kwargs.get("vq_learnable", False)),
            enc_na_powloce=bool(kwargs.get("enc_na_powloce", False)),
            pasmo_r=float(kwargs.get("pasmo_r", 0.0) or 0.0),
            przes_powloki=bool(kwargs.get("przes_powloki", False)),
            promien_specyf=float(kwargs.get("promien_specyf", 0.0) or 0.0),
            kanal_promienia=bool(kwargs.get("kanal_promienia", False)),
            bloki_poziomu=int(kwargs.get("bloki_poziomu", 0) or 0),
            udzial_kontekst=float(kwargs.get("udzial_kontekst", 0.0) or 0.0),
            kontekst_typ=str(kwargs.get("kontekst_typ", "splot")),
            kontekst_bloki=int(kwargs.get("kontekst_bloki", 1)),
            kontekst_stan=int(kwargs.get("kontekst_stan", 32)),
            agregacje=kwargs.get("agregacje", None),
            blok_jadro=int(kwargs.get("blok_jadro", 3)),
            blok_dropout=float(kwargs.get("blok_dropout", 0.1)),
            codebook_weight=float(kwargs.get("codebook_weight", 1.0)),
            pelny_zasieg=bool(kwargs.get("pelny_zasieg", False)),
            enc_dropout=float(kwargs.get("enc_dropout", 0.0)),
            in_ch=self._szerokosc_wejscia_enkodera(),
            skip_enc=bool(kwargs.get("skip_enc", False)),
            rc_stem_split=bool(kwargs.get("rc_stem_split", False)),
            rc_pelne=bool(kwargs.get("rc_pelne", False)),
            drabina=bool(kwargs.get("drabina_stozkowa", False)),
            fsq=bool(kwargs.get("fsq_styczny", False)),
            drab_dlugosc=kwargs.get("drab_dlugosc", 1.0),
            drab_krok_promienia=float(kwargs.get("drab_krok_promienia", 0.5)),
            # BEZ int() -- `fsq_bity` moze byc lista per poziom styczny.
            # Twarde `int()` tutaj bylo jedyna przyczyna, dla ktorej lista
            # nie dawala sie przekazac configiem.
            fsq_bity=(list(kwargs["fsq_bity"])
                      if isinstance(kwargs.get("fsq_bity", 9), (list, tuple))
                      else int(kwargs.get("fsq_bity", 9))),
            fsq_entropia=float(kwargs.get("fsq_entropia", 0.0)),
            # BEZ `float()`: lista musi przejsc. Rzutowanie zjadloby ja
            # wyjatkiem dopiero w petli, a przy jednoelementowej liscie
            # przeszloby po cichu na wszystkie poziomy.
            fsq_wzmocnienie=kwargs.get("fsq_wzmocnienie", 15.0),
            waga_stozka=float(kwargs.get("waga_stozka", 0.0)),
            apertura_stozka=float(kwargs.get("apertura_stozka", 1.0)),
            bloki_przed=int(kwargs.get("bloki_przed", 0) or 0),
            mot_atencja_l0=int(kwargs.get("mot_atencja_l0", 0) or 0),
            mot_l0_occ=(int(occ_channels)
                        if int(kwargs.get("mot_atencja_l0", 0) or 0) else 0),
            # `graph_dim` (128), NIE `_graf_ch` (16). `_graf_ch` to
            # szerokosc RZUTU grafu do latentu (`graf_proj`), a nie wektora
            # `g`, ktory idzie w `_ctx`. Pomylka dala `mat1 and mat2 shapes
            # cannot be multiplied (96x128 and 16x128)` i jest tej samej
            # klasy co zapisane `occ_channels` -- nazwa podobna, obiekt inny.
            # `KontekstKrzyzowy` w tym samym pliku bierze `dim_g=graph_dim`.
            mot_l0_g=(int(graph_dim)
                      if int(kwargs.get("mot_atencja_l0", 0) or 0) else 0),
            mot_l0_glowic=int(kwargs.get("mot_l0_glowic", 4)),
            mot_l0_dim_glowy=int(kwargs.get("mot_l0_dim_glowy", 32)),
            kwargs_wieza=(kwargs
                          if bool(kwargs.get("wieza_w_enkoderze", False))
                          else None),
            blok_przed_jadro=int(kwargs.get("blok_przed_jadro", 7)),
            stem_rezydualny=bool(kwargs.get("stem_rezydualny", False)),
            udzial_motywow=float(kwargs.get("udzial_motywow", 0.3)),
            motyw_brama=float(kwargs.get("motyw_brama", 0.1)),
            seq_len_enc=int(kwargs.get("seq_len", 249)),
        brama_glu=bool(kwargs.get("brama_glu", False)),
            brama_glu_z_se=bool(kwargs.get("brama_glu_z_se", False)),
            brama_glu_ranga=int(kwargs.get("brama_glu_ranga", 0) or 0))
        # Wspolpraca gradientowa miedzy poziomami. Ustawiane PO konstrukcji,
        # zeby nie zmieniac sygnatury `HyperbolicCodeStack` i zeby potomkowie
        # podmieniajacy `self.codes` dostali ja tak samo.
        self.codes.level_grad = bool(kwargs.get("level_grad", False))
        # KROK NADPROBKOWANIA MUSI ROWNAC SIE ZWEZENIU ENKODERA.
        #
        # `up = strides[0]` bylo poprawne dla `HyperbolicCodeStack`, ktory zweza
        # sygnal wlasnie tym stridem. Ale potomkowie podmieniaja `self.codes` na
        # `NestedSpectralLadder`, ktora ma WLASNY `in_stride` -- i `decoder`
        # zbudowany tutaj zostaje. Przy `in_stride: 1` latent siedzi na 248
        # pozycjach, dekoder podnosil go na 496, a `F.interpolate` w
        # `_parent_pooled` scinal z powrotem na 249.
        #
        # Zmierzone na HypSOTA2: wejscie dekodera (4, 480, 248) -> wyjscie surowe
        # 496 -> po interpolacji 249, czyli krotnosc 1,99 w gore i 2x w dol w
        # jednym przebiegu. Jedynym skutkiem takiej pary jest ROZMYCIE, a siedzi
        # ono dokladnie w sciezce, ktora ma wymuszac zachowanie fazy:
        # `occ_from_recon` liczy oblozenie ze `softmax(recon_logits)`, wiec skaner
        # PWM widzial rekonstrukcje po filtrze dolnoprzepustowym. Cala zmiana
        # `in_stride: 2 -> 1`, zrobiona zeby zachowac faze motywu 6-15 bp, byla
        # tutaj cofana.
        #
        # `in_stride` czytamy z kwargs, a NIE z `self.codes` -- w tym momencie
        # `self.codes` to jeszcze `HyperbolicCodeStack`, bo drabina powstaje
        # dopiero w `HypSpectralNet.__init__`, po `super().__init__()`.
        up = int(kwargs.get("in_stride", strides[0]))
        # Dekoder TYLKO przy niezerowej wadze rekonstrukcji. Bez tego warunku
        # bieg predykcyjny (`recon_weight: 0`) nosilby ~200 tys. parametrow bez
        # gradientu, ciagnietych weight decayem -- czyli nie byl by kontrola,
        # a modelem z martwym balastem. Patrz naglowek patcha.
        # ============ DEKODER RC-EKWIWARIANTNY (opcjonalny) ============
        # `dekoder(P(z)) == rc(dekoder(z))` DOKLADNIE -- zmierzone 1,1e-15
        # wobec 1,5152 dla ukladu swobodnego ponizej.
        #
        # PO CO, skoro gradient predykcji przez dekoder jest DOKLADNIE 0
        # (budzet gradientu: trzon 63,7 %, most 12,7 %, enkoder 10,8 %,
        # dekoder 4,5 %): dla DYFUZJI. Dzis `decode_codes(P(kody))` i
        # `rc(decode_codes(kody))` to dwie ROZNE sekwencje, wiec generator
        # przypisuje dwie rozne wiarygodnosci tej samej czasteczce. Priur,
        # ktory sam sobie przeczy o pasmo, jest gorszy niz priur wiazany --
        # a kody LOSOWE dekoduja sie GORZEJ niz tasowanie dinukleotydowe
        # (TVD-3 0,1317 wobec 0,0392), wiec priur jest tu jedynym, co mamy.
        #
        # KOLEJNOSC MA ZNACZENIE: bez `rc_pelne` latent NIE jest ekwiwariantny,
        # wiec ekwiwariantny dekoder nie ma czego wykorzystac. Straznik ponizej
        # nie pozwala wlaczyc samego dekodera i wygladac na naprawionego.
        _dek_rc = bool(kwargs.get("dekoder_rc", False))
        if _dek_rc and not self.recon_weight:
            raise ValueError(
                "dekoder_rc: true przy recon_weight = 0 -- dekoder w ogole "
                "nie powstaje, wiec flaga bylaby cichym no-opem")
        if _dek_rc and not self.codes.rc_pelne:
            raise ValueError(
                "dekoder_rc wymaga rc_pelne: bez ekwiwariantnego enkodera "
                "latent nie spelnia z(rc x) = P(z(x)), a wtedy wiazany dekoder "
                "wiaze dwie NIEZWIAZANE wielkosci -- test przechodzi na "
                "sztucznych danych i nic nie znaczy na prawdziwych")
        if _dek_rc:
            from .dekoder_rc import DekoderRC
            _bl = [vq_dim] * self.codes.n_levels
            if self.codes.recon_bypass:
                if self.codes.kanaly_promienia():
                    raise ValueError(
                        "dekoder_rc + kanal_promienia + recon_bypass: kanaly "
                        "promienia nie maja parowania RC (jeden na poziom)")
                _bl.append(vq_dim)          # slot ciagly, tez parowany
            if sum(_bl) != self.codes.kanaly_dekodera():
                raise ValueError("bloki %r nie sumuja sie do kanaly_dekodera()"
                                 "=%d" % (_bl, self.codes.kanaly_dekodera()))
            self.decoder = DekoderRC(
                _bl, up=up, szer=int(kwargs.get("dekoder_szer", 212)),
                tryb_up=str(kwargs.get("dekoder_up", "shuffle")))
        else:
            # ============ NORMALIZACJA DEKODERA ============
            # `dekoder_norma: grupowa` jest WARUNKIEM KONIECZNYM maskowania,
            # nie ozdoba. Przy `maska_kodow` dekoder dostaje CZTERY WIDOKI
            # sklejone po batchu, wiec BatchNorm liczy statystyke MIESZANKI
            # czterech rozkladow i wpisuje ja do `running_*`, ktore ida do
            # `eval`. ZMIERZONE (dwa modele o identycznych wagach -- 0,000e+00
            # na 185 wspolnych nazwach -- te same 15 batchy, tylko wprzod,
            # bez optymalizatora, wiec rozni je WYLACZNIE `running_*`):
            #     running_mean: rozjazd wzgledny  39,10 %
            #     running_var : rozjazd wzgledny  19,82 %
            #     logity w eval: roznica  27,41 % skali
            #     zgodnosc argmax (tej samej zasady): 0,8791
            # Czyli maskowanie z BatchNormem zmienia 12 % przewidywanych zasad
            # w `eval` i uderza wprost w falsyfikator `rek_pelny >= 0,99`,
            # a metryka `recon` w logu przestaje byc porownywalna z rodzina.
            # GroupNorm nie ma statystyk biezacych: `train` i `eval` to ta sama
            # funkcja, widoki nie mieszaja sie ze soba, a widok pelny jest
            # normalizowany dokladnie tak samo jak bez maski.
            # KOSZT: Delta0 parametrow. GroupNorm(8, 128) ma `weight` i `bias`
            # o tej samej szerokosci co BatchNorm1d(128) affine (256 liczb),
            # a `running_mean`/`running_var`/`num_batches_tracked` to BUFORY,
            # nie parametry. Ubocznie: dwa z czternastu BatchNormow modelu
            # znikaja, wiec zaleznosc od skladu batcha maleje.
            _dn = str(kwargs.get("dekoder_norma", "batch")).lower()
            if _dn not in ("batch", "grupowa"):
                raise ValueError(
                    "dekoder_norma musi byc 'batch' albo 'grupowa', jest %r" % (_dn,))
            if bool(kwargs.get("maska_kodow", False)) and _dn == "batch":
                raise ValueError(
                    "maska_kodow: true przy dekoder_norma: batch. Dekoder "
                    "dostaje wtedy cztery widoki sklejone po batchu, a jego "
                    "BatchNorm wpisuje do `running_*` statystyke MIESZANKI -- "
                    "zmierzone: rozjazd 39,1 %, zgodnosc argmax w eval 0,8791, "
                    "czyli 12 % zasad zmienionych i falsyfikator `rek_pelny` "
                    "zawalony z powodu normalizacji, nie maskowania. "
                    "Ustaw dekoder_norma: grupowa (Delta0 parametrow).")
            _nrm = ((lambda c: nn.GroupNorm(8, c)) if _dn == "grupowa"
                    else (lambda c: nn.BatchNorm1d(c)))
            # ============ AGREGACJA TRESCIOWA PO POZYCJACH KODU ============
            # Dekoder ma dwie drogi do odleglych pozycji i obie sa zle:
            # splot jest lokalny (50 % masy gradientu w 3 pozycjach), a
            # normalizacja daje zaleznosc AGREGATOWA (99 % masy w 119 ze 124,
            # ale to srednia i wariancja, nie tresc). Zmierzone skutki na
            # `HyppeDeepSTARR3_best.pt` (ep2): dokladnosc w zamaskowanym oknie
            # spada z 0,9583 (1 bp od krawedzi) do ~0,33 od 5 bp w glab,
            # a rozklad przewidywany w srodku ma entropie 1,9389 z 2,0 przy
            # entropii skladu 1,9660 -- model wypisuje tam ROZKLAD BRZEGOWY.
            # Maskowanie daje wiec gradient prawie wylacznie na brzegach
            # dziury, czyli nie robi tego, po co istnieje.
            # Modul stoi PRZED pierwszym splotem, wiec jego wejsciem jest
            # `tylko_kody(z)` -- funkcja samych indeksow, przez co
            # `decode_codes` (a z nim dyfuzja) zostaje nietkniete.
            _agr = []
            if bool(kwargs.get("dek_agregacja", False)):
                if self.codes.recon_bypass:
                    raise ValueError(
                        "dek_agregacja przy recon_bypass: true -- wejsciem "
                        "dekodera jest wtedy `z` ze slotem ciaglym, wiec "
                        "agregacja mieszalaby kanal, ktorego przy generowaniu "
                        "nie ma, i zlamalaby umowe `decode_codes`")
                from .agregacja_kodow import AgregacjaKodow
                _agr = [AgregacjaKodow(
                    self.codes.kanaly_dekodera(),
                    dlugosc=int(_dlugosci_z_kodow(self.codes, int(seq_len))[-1]),
                    glowic=int(kwargs.get("dek_agr_glowic", 4)),
                    dim_glowy=int(kwargs.get("dek_agr_dim", 16)),
                    brama0=float(kwargs.get("dek_agr_brama", 0.1)))]
            self.decoder = None if not self.recon_weight else nn.Sequential(
                *_agr,
                nn.Conv1d(self.codes.kanaly_dekodera(), 128, 3, padding=1),
                _nrm(128), nn.SiLU(),
                # Jadro == krok, wiec filtry sie NIE nakladaja: ta warstwa jest
                # matematycznie rownowazna 1D PixelShuffle i artefaktow
                # szachownicy tu nie ma. Sprawdzone, bo bylo podejrzewane.
                nn.ConvTranspose1d(128, 128, up, stride=up),
                _nrm(128), nn.SiLU(),
                nn.Conv1d(128, 4, 3, padding=1))

        # ============ MASKOWANIE KODOW W LATENCIE ============
        # WIDOKI DEKODERA, nie zmiana `z`. Trzon i glowica dostaja latent
        # nietkniety, wiec sciezka predykcyjna ma w treningu i w `eval`
        # DOKLADNIE ten sam rozklad wejscia i metryki zostaja porownywalne
        # z cala rodzina. Enkoder i tak dostaje gradient z zamaskowanej
        # rekonstrukcji -- przez pozycje NIEzamaskowane, bo to one musza
        # wystarczyc.
        #
        # Cztery pomiary, ktore wyznaczyly `maska_szerokosc` (urwisko
        # przecieku miedzy 18 i 34 bp) oraz przekreslily maskowanie
        # jednopoziomowe (poziomy odtwarzaja sie wzajemnie w 95 %), sa
        # w naglowku `models/maska_latentu.py`.
        _mk_wl = bool(kwargs.get("maska_kodow", False))
        self.maska = None
        self._maska_raport = None
        if _mk_wl:
            if self.decoder is None:
                raise ValueError(
                    "maska_kodow: true przy recon_weight = 0 -- dekoder nie "
                    "powstaje, wiec nie ma czego maskowac i mechanizm bylby "
                    "cichym no-opem (to ta sama rodzina awarii co `se_stat` "
                    "bramkowane `se_stat_bloki: 0`)")
            if self.codes.recon_bypass:
                raise ValueError(
                    "maska_kodow: true przy recon_bypass: true -- dekoder "
                    "widzi wtedy slot ciagly `feats[0] - q_ost`, czyli "
                    "BEZSTRATNA kopie enkodera sprzed kwantyzacji "
                    "(tozsamosc zmierzona na 2,6e-07), i odtworzy zamaskowane "
                    "okno z niej. Ten dokladny blad raz w tym repo dal "
                    "recon 0,9984, ktore bylo wlasnoscia slotu, a z samych "
                    "kodow wychodzilo 0,29")
            from .maska_latentu import MaskaLatentu
            self.maska = MaskaLatentu(
                dim=self.codes.dim, n_levels=self.codes.n_levels,
                szerokosc=int(kwargs.get("maska_szerokosc", 17)),
                n_blokow=int(kwargs.get("maska_blokow", 1)),
                wagi=kwargs.get("maska_wagi", None),
                prefiksy=bool(kwargs.get("maska_prefiksy", True)),
                ziarno=int(kwargs.get("maska_ziarno", 7898)),
                # Widok PODMIANY: korekcja bledu kodu, nie interpolacja dziury.
                # To operacja, ktora dyfuzja realnie wykonuje -- i jedyna,
                # w ktorej agregacja tresciowa dala zmierzona poprawe
                # (wypuklosc krzywej odpornosci -0,0239 -> +0,0156 przy 20 %),
                # podczas gdy wypelnianie ciaglego okna 34 bp nie drgnelo
                # (0,3550 -> 0,3401). Domyslnie 0, czyli widoku nie ma.
                podmiana_p=float(kwargs.get("maska_podmiana_p", 0.0)),
                stopnie=bool(kwargs.get("maska_stopnie", False)))

        # --- prediction path: continuous occupancy at FULL resolution ---
        # SWOBODNE FILTRY: niciowo niezmiennicze przy `rc_filters`. Musza byc,
        # jesli caly tor ma byc dokladny -- `_codes_input` wklada ich wyjscie do
        # tego samego wektora, co oblozenie PWM, wiec jeden nieekwiwariantny
        # skladnik psuje sume. Jadro 15 jest nieparzyste, czyli wyjscie
        # `padding=7` jest juz wyrownane do SRODKA okna.
        # BANK WIELOSKALOWY, opcjonalny. `n_learned_jadra` to lista szerokosci;
        # `n_learned` jest wtedy LACZNA liczba filtrow i musi sie dzielic bez
        # reszty. Domyslnie None -> zachowanie BIT W BIT jak dotad.
        _jad = kwargs.get("n_learned_jadra", None)
        _n_uczonych = int(n_learned or 0)
        # STRAZ: bank wieloskalowy istnieje TYLKO w wersji RC-dokladnej
        # (`RCMaxConv1d` na kazda szerokosc). Bez `rc_filters` poprzednia
        # wersja tego kodu CICHO spadala na jednoskalowy `nn.Conv1d(4, n, 15)`
        # -- czyli klucz `n_learned_jadra` bylby ustawiony i bezczynny.
        # To dokladnie wzorzec, ktory w tym repo zabil kilkanascie mechanizmow.
        if _jad and _n_uczonych and not bool(rc_filters):
            raise ValueError(
                "n_learned_jadra wymaga `rc_filters: true` -- bank wieloskalowy "
                "istnieje tylko w wersji niciowo dokladnej. Bez tego klucz "
                "bylby bezczynny, a filtry zostalyby jednoskalowe (15).")
        if _jad and _n_uczonych and bool(rc_filters):
            from .genomic_blocks import RCMaxBankWieloskalowy
            _jad = [int(k) for k in _jad]
            if _n_uczonych % len(_jad) != 0:
                raise ValueError(
                    "n_learned (%d) musi dzielic sie przez liczbe skal (%d) -- "
                    "inaczej bank ma inna liczbe kanalow niz zadana i "
                    "`occ_proj` cicho przyjmie zla szerokosc"
                    % (_n_uczonych, len(_jad)))
            self.learned = RCMaxBankWieloskalowy(_n_uczonych // len(_jad), _jad)
            _kan_uczonych = self.learned.out_channels
        elif _n_uczonych and bool(rc_filters):
            from .genomic_blocks import RCMaxConv1d
            self.learned = RCMaxConv1d(_n_uczonych, 15)
            _kan_uczonych = _n_uczonych
        elif _n_uczonych:
            self.learned = nn.Conv1d(4, _n_uczonych, 15, padding=7)
            _kan_uczonych = _n_uczonych
        else:
            self.learned = None
            _kan_uczonych = 0
        self.occ_proj = nn.Conv1d(v + _kan_uczonych, occ_channels, 1)
        # The code stack is a second view of the same sequence; upsampled to
        # full resolution it costs little and lets reconstruction structure
        # inform prediction without the two objectives sharing an encoder.
        # ============ KONTEKST DO LATENTU (opcjonalny) ============
        # ZMIERZONE, po co: latent widzi 4 (one-hot) + `motyw_ch` kanalow,
        # a trzon startuje od 256 (h_occ) + 64 (z_up). Przy 296 profilach
        # JASPAR i `motyw_ch: 32` latent dostaje **9,8 %** kanalow oblozenia
        # (32 z 328) i **ZERO** wektora grafu, ktory idzie do trzonu przez FiLM
        # w kazdym bloku. Kontekst globalny ma wiec tylko przez wlasny
        # najgrubszy poziom -- 15 komorek, 16,6 bp na komorke.
        #
        # Przy `kontekst_do_latentu` enkoder dostaje PELNE `h_occ` (te same
        # kanaly, od ktorych startuje trzon) plus waski rzut wektora grafu
        # rozgloszony po pozycjach. Rzut jest waski celowo: `g` jest STALE
        # wzdluz sekwencji, wiec 128 kanalow stalej to 128 kanalow tej samej
        # liczby.
        self.graf_proj = (nn.Conv1d(graph_dim, self._graf_ch, 1)
                          if self.kontekst_do_latentu and self._graf_ch else None)
        # ============ TRZON CZYTA WYLACZNIE LATENT (opcjonalny) ============
        # CEL PIERWSZY projektu: aktywnosc ma byc funkcja KODOW, zeby edycja
        # SNP i dyfuzja mialy sens. Dzis trzon startuje od
        # `cat([h_occ, z_up])`, wiec oblozenie omija kody CALKOWICIE -- i to
        # jest droga, ktora w ablacji rekordu wazyla +0,28005 (oblozenie)
        # i +0,61092 (filtry swobodne).
        #
        # Przy `trzon_tylko_z_latentu` trzon dostaje WYLACZNIE `z_up`, wiec
        # cala jego pojemnosc (1,94 M) staje sie GLOWICA NA KODACH, a nie
        # obejsciem. Zmierzony wcześniej koszt takiego zawezenia to
        # 0,02-0,036 PCC -- ale mierzono go, gdy latent widzial 9,8 %
        # oblozenia i zero grafu, czyli BEZ punktu powyzej.
        self.trzon_tylko_z_latentu = bool(
            kwargs.get("trzon_tylko_z_latentu", False))
        if self.trzon_tylko_z_latentu and self.codes.trunk_bypass:
            # Slot ciagly `feats[0] - aligned[-1]` niosl by kontekst
            # NIEKWANTYZOWANY prosto do trzonu, wiec "aktywnosc jest funkcja
            # kodow" byloby nieprawda przy poprawnie wygladajacym configu.
            # Ta sama awaria co `recon_bypass`, ktory raz uniewaznil recon
            # 0,9984 nalezace do kanalu ciaglego, nie do kodow.
            raise ValueError(
                "trzon_tylko_z_latentu wymaga trunk_bypass: false -- inaczej "
                "slot ciagly jest 64-kanalowa autostrada omijajaca kody")
        _cp = int(kwargs.get("code_proj_ch", 64))
        self.code_proj = nn.Conv1d(self.codes.kanaly_trzonu(), _cp, 1)
        self._cp_ch = _cp
        # ================= lat_center =================
        # ZMIERZONE na HypEncI_best (768 sekwencji walidacji, wyjscie code_proj):
        #   norma skladowej WSPOLNEJ (srednia po batchu)  53,3107
        #   norma skladowej PER SEKWENCJA                   3,5597
        #   iloraz                                          0,0668
        # Czyli 93,7 % tego, co latent wysyla do trzonu, to STALA identyczna
        # dla wszystkich sekwencji.
        #
        # Widac to wprost w ablacji (split testowy, RC-TTA):
        #   tasowanie calego latentu po batchu:  Dev -0,0046, Hk -0,0055
        #   zerowanie latentu:                   Dev -0,2409, Hk -0,4037
        # Tasowanie podmienia te 6,7 % i nie robi nic; zerowanie usuwa stala,
        # ktorej trzon sie nauczyl, i psuje wszystko. Droga latent -> predykcja
        # niesie wiec praktycznie ZERO tresci per sekwencja -- mimo ze wagi
        # stemu na te kanaly maja 58 % normy wag na oblozenie, wiec trzon
        # wcale ich nie ignoruje.
        #
        # `BatchNorm1d(affine=False)` usuwa skladowa wspolna i wyrownuje skale
        # per kanal. ZERO parametrow. To jest poprawka juz w tym repo zmierzona:
        # w HypCompactNet ten sam iloraz wynosil 0,0869, a po `lat_center`
        # wzrosl do 1,0701 -- DWUNASTOKROTNIE wiecej tresci per sekwencja.
        #
        # DLACZEGO NIE LayerNorm: normalizuje kazda probke po cechach i NIE
        # rusza kierunku WSPOLNEGO miedzy probkami -- czyli zostawia dokladnie
        # to, co tu przeszkadza. Zapisane w tym repo jako blad juz popelniony.
        self.lat_center = (nn.BatchNorm1d(_cp, affine=False)
                           if bool(kwargs.get("lat_center", False)) else None)

        # ============ KONTEKST KRZYZOWY (opcjonalny) ============
        # Zapytania z mapy trzonu (czyli Z KODOW przy
        # `trzon_tylko_z_latentu`), klucze i wartosci z `h_occ` plus token
        # globalny z wektora grafu. To jest miejsce spotkania hierarchii DNA
        # (drabina) z hierarchia motywow (skaner + graf) -- i jedyna droga
        # kontekstu do predykcji, ktora NIE dotyka dekodera, wiec umowa
        # dyfuzji zostaje nietknieta.
        _kb = int(kwargs.get("krzyz_bloki", 0) or 0)
        if _kb > 0:
            from .kontekst_krzyzowy import KontekstKrzyzowy
            self.krzyz = nn.ModuleDict({
                str(i): KontekstKrzyzowy(
                    trunk_dim, occ_channels, dim_g=graph_dim,
                    heads=int(kwargs.get("krzyz_heads", 4)),
                    dim_head=int(kwargs.get("krzyz_dim_head", 32)),
                    dropout=dropout,
                    brama0=float(kwargs.get("krzyz_brama", 0.05)))
                for i in range(int(n_blocks) - _kb, int(n_blocks))})
        else:
            self.krzyz = None
        _we_trzonu = _cp if self.trzon_tylko_z_latentu else (occ_channels + _cp)
        # ============ TRZON CZYTA ZDEKODOWANA SEKWENCJE ====================
        #
        # PROBLEM, KTORY TO ROZWIAZUJE -- ZMIERZONY, NIE PRZYPUSZCZONY.
        #
        # `trzon_tylko_z_latentu` odbiera trzonowi oblozenie i zostawia mu
        # 64 kanaly `code_proj(z)`. Cena tego zawezenia jest w tym repo
        # zmierzona i wynosi 0,02-0,036 PCC, a bieg HyppeDeepSTARR4 traci
        # wobec HypStozek dokladnie 0,018 -- czyli CALA luka miesci sie
        # w tej jednej pozycji. Rownoczesnie ablacja rekordu mowi, ze
        # trzon czytajacy SEKWENCJE ma dwie najgrubsze drogi w calym
        # modelu: filtry swobodne +0,61092 i oblozenie +0,28005.
        #
        # Ale te dwie drogi byly odciete NIE DLATEGO, ze czytaly sekwencje,
        # tylko dlatego, ze czytaly sekwencje WEJSCIOWA -- czyli omijaly
        # kwantyzator i lamaly warunek "aktywnosc jest funkcja kodow".
        # `x_hat = softmax(dekoder(kody)/tau)` NIE omija kwantyzatora:
        # jest funkcja WYLACZNIE kodow, wiec warunek zostaje spelniony
        # co do litery, a trzon dostaje sekwencje w formacie, dla ktorego
        # zostal zaprojektowany.
        #
        # Informacji to nie dodaje ANI BITU -- `rek_pelny` = 0,9996, wiec
        # `x_hat` jest bijekcja kodow. Zmienia sie wylacznie FORMAT: trzon
        # przestaje odczytywac zasady z 192 kanalow kodu (zmierzone: kody
        # wiedza o motywach mniej niz 4-mery -- L0 8,3 %, L2 37,2 % sufitu
        # skladu 4-merowego, bo latent ZAPISUJE, a nie PARSUJE) i dostaje
        # je z powrotem jako 4 kanaly one-hot.
        #
        # SKUTEK UBOCZNY, KTORY JEST GLOWNYM POWODEM: gradient predykcji
        # plynie teraz przez DEKODER do kodow. Rekonstrukcja i predykcja
        # przestaja byc konkurentami w Nashu i staja sie zalezne -- lepszy
        # dekoder to lepsza predykcja. Dotad dekoder dostawal gradient
        # wylacznie z CE rekonstrukcji.
        #
        # UMOWA DYFUZJI ZOSTAJE. grad_kody(Dev) jest pelny, tylko biegnie
        # teraz droga kody -> dekoder -> softmax -> trzon -> Dev, czyli
        # przez te sama funkcje, ktorej uzywa generowanie.
        self.trzon_z_rekonstrukcji = bool(
            kwargs.get("trzon_z_rekonstrukcji", False))
        # TEMPERATURA -- JEDYNE MIEJSCE, W KTORYM TO MOZE SIE CICHO ZADUSIC.
        # Pochodna softmaxu to `diag(p) - p p^T`, wiec przy `p ~ (1,0,0,0)`
        # cala macierz jest ~0. Entropia dekodera na tym biegu wynosi
        # 0,0084 bitu, czyli softmax jest NASYCONY -- bez temperatury
        # gradient do dekodera gasnie DOKLADNIE wtedy, gdy rekonstrukcja
        # jest dobra, a dobra jest od epoki 1 (0,9996). Wartosc dla trzonu
        # bylaby wtedy w porzadku, ale sprzezenie zwrotne bylo by ozdoba.
        # Falsyfikator w logu: `rt_ent` musi byc wyraznie nad 0,01 bitu.
        self.recon_tau = float(kwargs.get("recon_tau", 1.0))
        if self.recon_tau <= 0.0:
            raise ValueError("recon_tau musi byc dodatnie, jest %r"
                             % (self.recon_tau,))
        # RAMPA. Przy starcie dekoder jest losowy, wiec przez pierwsze kroki
        # trzon dostawalby szum w miejscu sekwencji. Mieszamy z nicia
        # KANONICZNA (nie z podana -- `x_hat` przewiduje nic kanoniczna,
        # wiec mieszanka z podana laczylaby dwa rozne uklady wspolrzednych
        # i psula niezmienniczosc RC). Przy `w = 1` mieszanka jest
        # `x_hat` DOKLADNIE i aktywnosc jest funkcja wylacznie kodow.
        self.rek_trzon_warmup = int(kwargs.get("rek_trzon_warmup", 0))
        self.rek_trzon_rampup = int(kwargs.get("rek_trzon_rampup", 0))
        # MSE na predykcji SAMEGO trzonu, przed dodaniem galezi poziomowej.
        # Domyslnie 0.0, czyli zachowanie bit w bit jak dotad.
        self.trunk_aux_waga = float(kwargs.get("trunk_aux_waga", 0.0) or 0.0)
        if self.trunk_aux_waga < 0.0:
            raise ValueError("trunk_aux_waga musi byc >= 0, jest %r"
                             % (self.trunk_aux_waga,))
        if self.trzon_z_rekonstrukcji:
            if self.recon_weight <= 0.0:
                raise ValueError(
                    "trzon_z_rekonstrukcji wymaga recon_weight > 0 -- bez celu "
                    "rekonstrukcji dekoder nie powstaje i `recon_logits` nie "
                    "istnieja, wiec trzon dostalby kanaly bez zrodla")
            if self.codes.recon_bypass:
                # Przy otwartym kanale ciaglym `x_hat` jest funkcja SLOTU,
                # czyli bezstratnej kopii enkodera sprzed kwantyzacji --
                # a nie kodow. To ta sama awaria, ktora raz uniewaznila
                # recon 0,9984 w tym repo.
                raise ValueError(
                    "trzon_z_rekonstrukcji wymaga recon_bypass: false -- "
                    "inaczej `x_hat` jest funkcja slotu ciaglego, nie kodow, "
                    "i warunek 'aktywnosc jest funkcja kodow' jest falszywy "
                    "przy poprawnie wygladajacym configu")
            _we_trzonu = _we_trzonu + 4
        # ====== OBLOZENIE LICZONE Z `x_hat` -- 128 KANALOW DO TRZONU ======
        #
        # NAJWIEKSZA NIEWYPROBOWANA DZWIGNIA PCC W TYM MODELU.
        #
        # Trzon dostaje dzis 4 kanaly `x_hat` i 64 kanaly kodow. ODMAWIA MU
        # SIE 128 kanalow `h_occ`, w ktorych siedza 296 profili JASPAR ORAZ
        # 32 filtry uczone DE NOVO (`occ_proj(refiner(occ) + sigmoid(learned)))`.
        # Ablacja rekordu: filtry swobodne +0,61092, oblozenie +0,28005 --
        # dwie najgrubsze drogi w calym modelu. Zmierzone dzis na zbudowanym
        # HD5: `learned` to 1 952 param., `occ_proj` 42 112, a ZERO z tego
        # nie dochodzi do predykcji, bo `_h_occ` idzie WYLACZNIE do enkodera.
        #
        # Liczone z `x_hat` sa funkcja WYLACZNIE kodow, wiec umowa dyfuzji
        # zostaje. Sprawdzone pomiarem na HD5 (`sonda_tylko_kody.py`):
        # przypiecie stanu kodowego daje luke predykcji 0,000000e+00 przy
        # kontroli pozytywnej 3,700727e-02.
        #
        # CYKL I JAK GO ROZWIAZUJEMY. `x_hat` wymaga kodow, kody wymagaja
        # oblozenia. Wiec ENKODER zostaje przy oblozeniu z `x` (jego wyjscie
        # jest kwantyzowane, wiec umowy to nie lamie), a TRZON dostaje
        # oblozenie z `x_hat`. To znaczy DWA przebiegi skanera na krok.
        #
        # DLACZEGO TO WYMAGA `norma_typ: grupowa`. Drugi przebieg przez
        # `refiner` z BatchNormem przepisalby `running_*` dwukrotnie na krok
        # -- w tym repo "kazdy dodatkowy forward psuje EMA" to zapisana
        # awaria (recon 0,79 -> 0,62, i winilem wtedy inna flage). Straz
        # ponizej sprawdza TYP normalizacji w `refiner`, a nie flage configu,
        # bo flaga moze byc ustawiona i nie dojsc (piaty taki przypadek
        # w tym projekcie).
        self.occ_z_rekonstrukcji = bool(
            kwargs.get("occ_z_rekonstrukcji", False))
        if self.occ_z_rekonstrukcji:
            if not self.trzon_z_rekonstrukcji:
                raise ValueError(
                    "occ_z_rekonstrukcji wymaga trzon_z_rekonstrukcji: true "
                    "-- zrodlem oblozenia jest ten sam zmieszany tensor "
                    "`_zr`, ktory tamten mechanizm buduje, wiec bez niego "
                    "nie ma z czego liczyc skanu")
            # STRAZ TYPU NORMALIZACJI JEST LENIWA -- patrz `_straz_normy`.
            # Nie moze stac tutaj: `podmien_normalizacje` dziala PO
            # konstrukcji (w `registry.build_model`), wiec w tym momencie
            # `refiner` ma jeszcze BatchNorm i straz zapalalaby sie ZAWSZE,
            # takze przy poprawnym configu. Pierwsza wersja tego kodu miala
            # ten blad; sprawdzenie musi siedziec ZA miejscem, w ktorym typ
            # sie ustala -- to ta sama regula, ktora zlamalem dzis w
            # `sonda_tylko_kody`.
            self._straz_normy_zrobiona = False
            _we_trzonu = _we_trzonu + int(occ_channels)
        # DROPOUT NA NOWYCH KANALACH -- regularyzacja CELOWANA.
        #
        # Zmierzone w HyppeDeepSTARR8: te 128 kanalow podnosza PCC (+0,023
        # wobec HD5 w epoce 5), ale `gap` rosnie LINIOWO o ~0,0085 na epoke
        # (+0,005 / +0,016 / +0,025 / +0,033 / +0,042 w epokach 3-7).
        # Ekstrapolacja na epoke 10 daje ~+0,068, czyli WIECEJ niz szczytowy
        # gap calej rodziny odniesienia (+0,054 w epoce 19). To jest zapisany
        # mechanizm tego repo: kazdy wariant z wieksza pojemnoscia sie
        # przeuczal (gap +0,05..+0,18).
        #
        # DLACZEGO TU, A NIE GLOBALNY weight decay albo mniejsze
        # `occ_channels`: pojemnosc doszla w JEDNYM miejscu i tam ma byc
        # regularyzowana. Podniesienie globalnego decayu (dzis 0,08) dotknie
        # takze kodebooka -- a w tym repo jest zmierzone, ze weight decay
        # zdejmuje punkty z hiperboloidy i zabija bramy galezi. Zmniejszenie
        # `occ_channels` zmienia RONWIEZ wejscie enkodera (`_h_occ` idzie tam
        # przy `kontekst_do_latentu: true`), wiec bieg mierzylby dwie rzeczy.
        # `nn.Dropout1d` zeruje CALE kanaly, czyli wymusza, zeby trzon nie
        # opieral sie na pojedynczym profilu JASPAR -- to jest dokladnie ta
        # forma przeuczenia, ktorej sie tu spodziewamy.
        #
        # ZERO nowych parametrow. Domyslnie 0,0, wiec HD8 buduje sie bit
        # w bit jak dotad.
        _od = float(kwargs.get("occ_dropout", 0.0) or 0.0)
        if not 0.0 <= _od < 1.0:
            raise ValueError("occ_dropout musi byc w [0, 1), jest %r" % (_od,))
        if _od > 0.0 and not self.occ_z_rekonstrukcji:
            raise ValueError(
                "occ_dropout > 0 bez occ_z_rekonstrukcji -- nie ma kanalow, "
                "ktore mialby regularyzowac (to ta sama rodzina co "
                "`negatywy_frac` przy martwej glowicy: koszt bez efektu)")
        self.occ_dropout = nn.Dropout1d(_od) if _od > 0.0 else None
        # ===================== TOR TROJKATNY (struktura 3D) ================
        # Reprezentacja PAR ZASAD w pasmie +/-R plus domkniecie trojkata
        # (triangle multiplicative update z AF2). Zmierzone sonda grzbietowa
        # na 32k/8k PRZED biegiem: trzeci rzad daje `T - B2 = +0,04379` przy
        # progu +0,009, kontrola pojemnosci `T - Ttas = +0,04735`, a Dev
        # **+0,02244** tam, gdzie pary dawaly -0,00052. To jedyny kierunek
        # architektoniczny otwarty po zamknieciu szesciu innych.
        #
        # PASMO, nie pelne 249x249: pary 33-249 bp sa zmierzone jako
        # REDUNDANTNE wobec 0-33 bp (`C - B = -0,00308`, kontrola tasowania
        # 0,2 SE), wiec pelny tensor wydalby 98 % pamieci na powloke bez
        # tresci. R = 16 to dokladnie ta powloka. Atencja trojkatna zostaje
        # WYLACZONA na tej rozdzielczosci: [B, h, 249, 249, 249] to 11,9 TiB.
        #
        # WEJSCIE DO TRZONU IDZIE KONKATENACJA, NIE BRAMA. W tej samej
        # rodzinie, w tym samym biegu, dwie galezie za skalarna brama umarly
        # (`krzyz_brama` 0,05 -> 0,0003; `para_brama` -> -0,0001), a osobno
        # jest zmierzone, ze podmiana TRESCI kosztuje zero, a caly koszt to
        # skok SKALI. Kanalu skonkatenowanego nie da sie wylaczyc bez
        # zabicia wejscia, od ktorego stem trzonu zalezy.
        # ============ KANONIZACJA NICI (punkt 2) ==========================
        # Dokladna niezmienniczosc RC latentu przez wybor orientacji, nie
        # przez wiazanie wag -- ekwiwariancja PERMUTUJE indeksy L0 i nigdy
        # nie da z nich rownosci. Szczegoly i cena (1 bit) w `kanon_rc.py`.
        #
        # TRZY SKUTKI, KTORE TRZEBA MIEC NA PISMIE:
        #  * `RandomRC` staje sie NO-OPEM, a jest jedyna dozwolona
        #    augmentacja (mutacje punktowe i indele sa zakazane, bo model
        #    ma przewidywac efekt wariantu). Bieg traci wiec augmentacje
        #    w zamian za niezmienniczosc DOKLADNA zamiast uczonej.
        #  * RC-TTA tez staje sie no-opem; zapisany zysk +0,00527 zostaje
        #    wchloniety, wiec nie wolno go doliczac drugi raz.
        #  * `recon` liczy sie wobec nici KANONICZNEJ. Liczony wobec podanej
        #    spadlby do ~0,5 na polowie probek i wygladalby jak awaria
        #    dekodera, a bylby wlasnie brakiem tego jednego bitu.
        # KALIBRACJA NIEPEWNOSCI NA ODCIETYCH CECHACH. Patrz naglowek
        # `sonda_przewidywalnosc_bledu.py`: sufit 1-D to 0,139, glowica
        # osiaga 0,030, a jej klasa funkcji to rozwiazanie 1-D zawiera.
        # FiLM I WEJSCIE GRAFU DO GLOWICY -- dwie drogi omijajace
        # kwantyzator. Zmierzone: po odcieciu `code_proj` predykcja nadal
        # zalezy od sekwencji (sd 0,3226 wobec 0,3130), a
        # ||d(dev)/d(g)|| = 1,52. Domyslnie WLACZONE, zeby stare configi
        # zachowaly sie bit w bit.
        self.film_wl = bool(kwargs.get("film_wl", True))
        self.glowica_graf = bool(kwargs.get("glowica_graf", True))
        self.kal_wariancji = bool(kwargs.get("kal_wariancji", False))
        # WAGA CZLONU KALIBRACYJNEGO -- jedyna realna dzwignia wariancji.
        #
        # ZMIERZONE na HyppeDeepSTARR8_best.pt (ep9), pelna walidacja po
        # deduplikacji (20 285 unikalnych loci), wagi EMA, tryb eval:
        #   glowica          r(lv, log e^2) = +0,1526 (Dev) / +0,1527 (Hk)
        #   sufit tej klasy  (ridge na `feat`, tren -> val)  0,1401 / 0,1572
        #   realizacja sufitu                  109 % / 97 %
        # Czyli KORELACJA jest przy sufcie i wiecej wejsc nie pomoze --
        # sprawdzone: wielomian predykcji dal 0,1366 / 0,1286, a 512
        # losowych cech ReLU 0,1371 / 0,1591, wszystko w szumie.
        #
        # OTWARTE JEST NATOMIAST ROZPROSZENIE: nachylenie OLS **1,898**,
        # `sd(lv)` **0,1843** wobec wymaganych **0,3091**. Glowica trafia
        # w KIERUNEK niepewnosci i myli sie w SKALI -- a to wlasnie znaczy
        # "nieskalibrowana wariancja". Czlon kalibracyjny jest czystym NLL
        # (beta = 0) na `_lv_kal`, z cechami I bledem ODCIETYMI, wiec jest
        # jedynym miejscem, ktore moze to naprawic, i nie ma drogi do
        # sredniej ani do trzonu.
        #
        # 0,1 -> 0,5. Gradient tego czlonu dochodzi WYLACZNIE do
        # `fc_dev_lv`/`fc_hk_lv` (dwa tensory po 256 wag), wiec wzmocnienie
        # nie przestawia rownowagi Nasha w sposob, ktory dotknalby PCC.
        self.kal_waga = float(kwargs.get("kal_waga", 0.1) or 0.0)
        # STRAZ NA UTAJONA PULAPKE. Caly blok heteroskedastyczny w
        # `train_graph` (beta-NLL, czlon kalibracyjny, akumulatory `lv_r`)
        # siedzi wewnatrz galezi `elif nash is not None:`. Przy `pcgrad`
        # albo bez Nasha glowica nie dostalaby gradientu przez CALY bieg,
        # a log wygladalby IDENTYCZNIE jak w rozgrzewce (`lv_sd = 0.0`,
        # `pre_lv = (0.0, 0.0)`, brak klucza `lv_r`). Zadna asercja tego
        # nie pilnowala, wiec bieg moglby zaraportowac "model przewiduje
        # wariancje" przy glowicy, ktora nigdy nie zobaczyla gradientu.
        if bool(kwargs.get("heteroskedastyczna", False)):
            self._wymaga_nasha_dla_lv = True
        self._lv_kal = None
        self.kanon_rc = bool(kwargs.get("kanon_rc", False))
        self._orient = None
        # ============ ROZDZIELENIE L0 Dev/Hk (punkt 4) ====================
        # Po KIERUNKACH stycznych, nie po polowie kodow: L0 niesie 4,585
        # z 5,0 bitow (91,7 % osiagalnej) i kwantyzuje POZYCJE, ktorej
        # potrzebuje rekonstrukcja. Podzial 2 x 16 wyrzucilby jedno i drugie.
        self.rozdziel = None
        self.rozdziel_waga = float(kwargs.get("rozdziel_waga", 0.0) or 0.0)
        if self.rozdziel_waga > 0:
            # `vq_dim`, NIE `vq_dim - 1`. Punkt ma `vq_dim + 1` wspolrzednych
            # ambientnych (zmierzone: embed [32, 97] przy vq_dim 96), a
            # `logmap0` zwraca `vq_dim` wspolrzednych stycznych.
            # `vq_dim` z ARGUMENTU, nie z `kwargs`. `vq_dim` jest NAZWANYM
            # parametrem tego `__init__` (linia 1201), wiec do `**kwargs`
            # NIE TRAFIA NIGDY i `kwargs.get("vq_dim", 96)` zwracalo zawsze
            # 96. Dzialalo tylko przez zbieg okolicznosci -- w rodzinie
            # HypFinal `vq_dim` JEST 96. Na HypStozku (64) ten modul
            # przechodzil budowe, forward i OBIE bramki, a padal w pierwszym
            # kroku treningu, czyli po zajeciu slotu GPU.
            self.rozdziel = RozdzielenieL0(
                int(vq_dim),
                lambda_ort=float(kwargs.get("rozdziel_ort", 1.0)))
        self.trojkat = None
        self.troj_waga = float(kwargs.get("troj_waga", 0.1) or 0.0)
        self._troj_aux = None
        self._troj_strata = None
        # WIEZA W TRZONIE jest teraz OPCJONALNA. Przy `wieza_w_enkoderze`
        # ta sama wieza stoi PRZED kwantyzatorem i czyta wczesna mape (96
        # kanalow, 249 pozycji), wiec jej wynik wchodzi do latentu.
        # Trzymanie drugiej kopii w trzonie kosztuje ~132 tys. parametrow
        # i liczy pary na tresci PO kwantyzacji, czyli na obiekcie
        # ubozszym. Domyslnie WLACZONA, zeby stare configi nie zmienily
        # zachowania.
        _tp = (int(kwargs.get("troj_pasmo", 0) or 0)
               if bool(kwargs.get("troj_w_trzonie", True)) else 0)
        if _tp > 0:
            _tw = int(kwargs.get("troj_wy", 24))
            self.trojkat = TorTrojkatnyPasmowy(
                _we_trzonu, dim_par=int(kwargs.get("troj_kanaly", 24)),
                pasmo=_tp, blokow=int(kwargs.get("troj_bloki", 2)),
                dim_wy=_tw,
                seq_atencja=bool(kwargs.get("troj_seq_atencja", True)),
                glowic=int(kwargs.get("troj_glowic", 4)),
                dim_glowy=int(kwargs.get("troj_dim_glowy", 24)),
                checkpoint=bool(kwargs.get("troj_checkpoint", True)),
                gruby_L=int(kwargs.get("troj_gruby_L", 0) or 0),
                gruby_bloki=int(kwargs.get("troj_gruby_bloki", 2)),
                gruby_par=int(kwargs.get("troj_gruby_par", 16)),
                gruby_kawalek=int(kwargs.get("troj_gruby_kawalek", 64)),
                gruby_wy=int(kwargs.get("troj_gruby_wy", 16)),
                udzial_seq=float(
                    kwargs.get("troj_udzial_seq", 0.0) or 0.0))
            _gw = (int(kwargs.get("troj_gruby_wy", 16))
                   if int(kwargs.get("troj_gruby_L", 0) or 0) else 0)
            _we_trzonu = _we_trzonu + _tw + _gw
        self.stem = nn.Sequential(
            nn.Conv1d(_we_trzonu, trunk_dim, 1),
            nn.BatchNorm1d(trunk_dim), nn.SiLU())
        # `para_bloki` = ile OSTATNICH blokow dostaje sprzezenie parowe zamiast
        # bramki SE. Ostatnich, bo tam reprezentacja jest juz motywowa, a nie
        # surowo nukleotydowa -- para motywow jest tym, czego brakuje, a nie
        # para pojedynczych zasad (zmierzone: par o |i-j| > 202 jest 3,5 %,
        # wiec zasiegu nie brakuje; brakuje SPRZEZENIA).
        _nb = int(n_blocks)
        _pb = int(kwargs.get("para_bloki", 0)) if isinstance(kwargs, dict) else 0
        self.blocks = nn.ModuleList(
            MultiScaleDilatedBlock(
                trunk_dim, kernel_size, tuple(dilations), dropout=dropout,
                brama_glu=bool(kwargs.get("brama_glu", False))
                if isinstance(kwargs, dict) else False,
                brama_glu_z_se=bool(kwargs.get("brama_glu_z_se", False))
                if isinstance(kwargs, dict) else False,
                brama_glu_ranga=int(kwargs.get("brama_glu_ranga", 0) or 0)
                if isinstance(kwargs, dict) else 0,
                para_atencja=(i >= _nb - _pb and _pb > 0),
                para_zamiast_se=bool(kwargs.get("para_zamiast_se", True))
                if isinstance(kwargs, dict) else True,
                para_heads=int(kwargs.get("para_heads", 4)) if isinstance(kwargs, dict) else 4,
                para_dim_head=int(kwargs.get("para_dim_head", 32)) if isinstance(kwargs, dict) else 32,
                bez_se=bool(kwargs.get("gram_zamiast_se", False))
                if isinstance(kwargs, dict) else False,
                ekstrema=bool(kwargs.get("ekstrema", False))
                if isinstance(kwargs, dict) else False,
                ekstrema_ranga=int(kwargs.get("ekstrema_ranga", 74))
                if isinstance(kwargs, dict) else 74,
                # STATYSTYKA SQUEEZE tylko na `se_stat_bloki` PIERWSZYCH
                # blokach. Zysk jest skupiony na wczesnych: surowy zysk
                # `mean+sd` nad `mean` to +0,0672 / +0,0285 / +0,0135 /
                # -0,0036 / +0,0025 na L0..L4, wiec dokladanie go na L3-L4
                # kupuje 50 176 wag za nic.
                se_stat=(str(kwargs.get("se_stat", "mean"))
                         if i < int(kwargs.get("se_stat_bloki", 0) or 0)
                         else "mean"))
            for i in range(_nb))

        self.node_encode = nn.Linear(3, graph_dim)
        self.node_identity = nn.Parameter(torch.randn(v, graph_dim) * 0.02)
        # ================= edge_bias =================
        # `GraphAttentionLayer` ma w pelni napisana, ZEROINICJALIZOWANA sciezke
        # `edge_bias = nn.Linear(edge_dim, heads)` (vq_gat_film.py:128-130),
        # a tutaj byla wolana BEZ `edge_dim`, wiec nigdy nie powstawala, i
        # `_graph` wolal warstwy bez `edge_feats`.
        #
        # Bez niej logit atencji to `LeakyReLU(a_src . h_i + a_dst . h_j)`,
        # czyli ADDYTYWNIE ROZDZIELNY: zaden zestaw wag nie wyrazi "wezel i
        # RAZEM z wezlem j", bo funkcja rozklada sie na sume czlonow
        # jednowezlowych. To strukturalna niezdolnosc do wyrazenia PARY
        # w podsystemie o 288 960 wagach -- w modelu, ktorego zmierzonym
        # brakiem jest wlasnie PARA, a nie zasieg (par o |i-j| > 202 bp
        # jest 3,5 %).
        #
        # Koszt: `Linear(4, heads)` na warstwe. Start BIT W BIT identyczny,
        # bo `zeros_` -- wiec przy `edge_dim: 0` (domyslnie) model jest
        # nieodrozninalny od poprzedniego, a przy 4 rozni sie dopiero tym,
        # czego sie nauczy.
        # ================= POLE SALIENTNOSCI =================
        # JEDYNY mechanizm w tym modelu, ktory moze powiedziec
        # "ten element jest wazny W TYM MIEJSCU".
        #
        # ZMIERZONE, dlaczego jest potrzebny. Kazda droga w tej rodzinie jest
        # albo SKALAREM doklejonym do predykcji, albo modulacja RANGI 1
        # STALA PO POZYCJI:
        #   dev = dev + _og["dev"].unsqueeze(-1)              # HypGram: skalar
        #   dev = dev + hyp_gate[0]*o["dev"].unsqueeze(-1)    # glowice: skalar
        #   h = h*(1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)   # FiLM: [B,C]
        #                                                     rozglaszane po L
        #   SE: bramka kanalowa, permutacyjnie niezmiennicza po pozycjach
        # Razem 793 520 wag modulacji, z ktorych ZADNA nie zalezy od pozycji.
        #
        # A najwiekszy niewykorzystany efekt w torze to interakcja
        # MOTYW x SEKWENCJA: 57 % waznosci motywu, wobec 43 % efektu glownego
        # (pomiar P4, 30 tys. sekwencji, 40 motywow JASPAR). Ten sam motyw
        # znaczy tu, a nie tam -- i nic w modelu nie moze tego wyrazic.
        #
        # Zmierzone konsekwencje: trzy modele o roznych architekturach myla sie
        # w tych samych miejscach w 93 % (Dev), a w najgorszym kwintylu --
        # gdzie siedzi 68 % calego bledu -- w **97,4 %**. Zespol nasycil sie
        # na 0,77001 przy sufcie z wiarygodnosci ~0,877.
        #
        # KONSTRUKCJA. Pole powstaje z DRABINY (czyli z latentu), wiec laczy
        # latent z predykcja droga, ktorej dotad nie bylo -- modulacja zamiast
        # konkatenacji kanalow:
        #     s(p) = MLP(punkt drabiny w pozycji p)   in R^M
        #     h    = h * (1 + U^T s(p))               U in R^{M x C}
        # Ranga M w kanale, PELNA ranga w pozycji. Liczba parametrow NIE ROSNIE
        # ani z L, ani z liczba pozycji -- MLP i U sa DZIELONE miedzy wszystkimi
        # pozycjami i sekwencjami, co jest wymogiem sciany informacyjnej
        # (804 592 liczby nadzoru wobec 1 333 300 mozliwych trojek).
        #
        # `zeros_` na U: start BIT W BIT jak model bez pola, wiec galaz wchodzi
        # dopiero, gdy sie czegos nauczy -- i wtedy `pole_amp` w logu to pokaze.
        self.pole_M = int(kwargs.get("pole_salientnosci", 0) or 0)
        if self.pole_M > 0:
            self.pole_mlp = nn.Sequential(
                nn.Linear(self.codes.dim + 1, 32), nn.SiLU(),
                nn.Linear(32, self.pole_M))
            self.pole_U = nn.Linear(self.pole_M, trunk_dim, bias=False)
            # NIE `zeros_`. Modulacja to `mod = U @ s`, wiec
            #   dL/dU  proporcjonalne do s      i      dL/ds  proporcjonalne do U.
            # Przy U = 0 ZEROWE SA OBIE, nie tylko jedna: `pole_mlp`
            # i `pole_skala` nie dostaja gradientu wcale i mechanizm rusza
            # dopiero, gdy U wypelznie z zera samo -- a nie ma czym, bo jego
            # wlasny gradient tez zalezy od s, ktore jest wtedy nieuczone.
            # Sprawdzone przed startem: przy `zeros_` piec z szesciu parametrow
            # pola bylo na liscie "bez gradientu".
            #
            # To jest zapisana w tym repo pulapka "wygrzewac ZRODLO, nie SKALE"
            # (skok bramki 0,02 -> 1,0 kosztowal -0,44 PCC). Maly niezerowy init
            # daje modulacje rzedu 3 % na starcie -- prawie tozsamosc, ale
            # z zywym gradientem w calej galezi.
            nn.init.normal_(self.pole_U.weight, std=0.01)
            self.pole_skala = nn.Parameter(torch.zeros(self.codes.n_levels))
        else:
            self.pole_mlp = self.pole_U = None
            self.pole_skala = None
        self.edge_dim = int(kwargs.get("edge_dim", 0) or 0)
        self.gat = nn.ModuleList(
            GraphAttentionLayer(graph_dim, heads,
                                edge_dim=(self.edge_dim if self.edge_dim else None),
                                dropout=dropout)
            for _ in range(int(gat_layers)))
        # ============ HIPERGRAF WSPOLWYSTEPOWANIA ==========================
        # Hiperkrawedz = zbior TOP-K motywow o najwyzszym obłozeniu w danym
        # oknie, bez informacji o ich wzajemnej odleglosci. Zmierzone na 256
        # loci: top-k = 8 daje dokladnie 8 motywow na krawedz (prog efektywny
        # 3,47, stopien wezla 2,47), a prog staly nie moze byc wlasciwy dla
        # 296 profili o szerokosci 7-21 bp.
        #
        # DLACZEGO NIE HIPERBOLICZNIE. Agregacja geodezyjna wobec euklidesowej
        # na PRAWDZIWYM obłozeniu: `hg_kat_cos` = **0,9963**, czyli kierunek
        # identyczny, a `hg_prom_iloraz` = **0,5723**. Cala roznica (0,4326)
        # to kompresja PROMIENIA z `asinh`, nie krzywizna -- to samo zrobilby
        # LayerNorm. Potwierdza zapisany dowod, ze kierunek centroidu Lorentza
        # jest dokladnie znormalizowana euklidesowa srednia wazona. Domyslnie
        # `hg_hiper: false`; flaga zostaje, bo falsyfikator jest tanszy od
        # przekonania.
        self.hipergraf = None
        if bool(kwargs.get("hipergraf", False)):
            self.hipergraf = HipergrafWspolwystepowania(
                len(self._vocab), graph_dim,
                okien=int(kwargs.get("hg_okien", 15)),
                hiper=bool(kwargs.get("hg_hiper", False)),
                rundy=int(kwargs.get("hg_rundy", 2)),
                topk=int(kwargs.get("hg_topk", 8)))
        # ============ ATENCJA PODPRZESTRZEN <-> MOTYW ======================
        # Kazda podprzestrzen predykcyjna (Dev, Hk) ZADAJE PYTANIE motywom.
        # Osobno na cel, bo rankingi waznosci motywow dla Dev i Hk koreluja
        # na rho = -0,04. Bez skalarnej bramy -- wyjscie konkatenowane.
        self.pm_wy = int(kwargs.get("pm_wy", 0) or 0)
        if self.pm_wy > 0:
            # `vq_dim` z ARGUMENTU, nie z `kwargs` -- ta sama usterka co
            # w `RozdzielenieL0` wyzej. Przy vq_dim 64 `pm_wy` padalo na
            # forwardzie: "mat1 and mat2 shapes cannot be multiplied
            # (8x64 and 96x128)".
            _vd = int(vq_dim)
            self.pm_dev = AtencjaPodprzestrzenMotyw(_vd, graph_dim,
                                                    dim_wy=self.pm_wy)
            self.pm_hk = AtencjaPodprzestrzenMotyw(_vd, graph_dim,
                                                   dim_wy=self.pm_wy)
        else:
            self.pm_dev = self.pm_hk = None
        self.graph_norm = nn.LayerNorm(graph_dim)
        self.film = None if not self.film_wl else nn.ModuleDict({
            str(i): nn.Linear(graph_dim, 2 * trunk_dim) for i in range(int(n_blocks))})
        # `film_init_std > 0` odblokowuje galaz motywowa NA STARCIE. Gradient
        # wstecz do grafu to W^T @ dL/dgamma, wiec przy W = 0 graf, GAT i skaner
        # dostaja DOKLADNIE ZERO, dopoki sama bramka nie ruszy. To bootstrap na
        # epoki -- w drugiej rodzinie zmierzono przy zeros_ udzial galezi 0,007%.
        # Rekord szczytowal w epoce 26, wiec tam sie odbudowalo; przy budzecie
        # 10 epok nie ma na to czasu. Domyslnie 0,0, czyli STARE zachowanie,
        # wiec zmiana jest przypisywalna i odwracalna jedna liczba w configu.
        _fis = float(kwargs.get("film_init_std", 0.0))
        for layer in (self.film.values() if self.film is not None else ()):
            if _fis > 0.0:
                nn.init.normal_(layer.weight, std=_fis)
            else:
                nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Pooled continuous archetype -- Euclidean, so an SAE can read it.
        # POOLING PER CEL. `pooled = cat([mean, amax, g])` sprowadza 249 pozycji
        # do sredniej i maksimum na kanal -- IDENTYCZNIE dla Dev i Hk. To jest
        # realne waskie gardlo informacyjne i jedyne miejsce, gdzie moglaby sie
        # wyrazic zmierzona roznica miedzy celami: sklad jest gruboskalowy
        # i napedza Dev, uklad motywow jest drobnoskalowy i napedza Hk, a ich
        # rankingi waznosci motywow koreluja na rho = -0,04.
        #
        # Zastepujemy SAMA SREDNIA miekka pula z atencja po pozycjach, osobna
        # per cel; `amax` zostaje wspolne. Wagi logitow startuja z ZERA, wiec
        # softmax jest jednostajny i pula jest DOKLADNIE srednia -- model
        # zaczyna tozsamo z obecnym i moze sie od niego dopiero oddalic.
        # To jest wymog tego projektu: nie startowac w miejscu, ktorego nie
        # umiemy odtworzyc.
        # POOLING WIELOSKALOWY. Model pooluje dzis na JEDNEJ skali -- calej
        # sekwencji -- a to jest skala, ktora w zmierzonym przemiataniu niesie
        # NAJMNIEJ. Regresja grzbietowa na czestosciach mono i dinukleotydow
        # w oknach o dlugosci W, dokladane od grubej do drobnej, z kontrola
        # pojemnosci, zyski NETTO:
        #     W=249 (1 okno)  -- baza
        #     W= 83 (3 okna)  **+0,01990**   <- najwiekszy skok w calym sweepie
        #     W= 41 (7 okien)   +0,00512
        #     W= 21 (12 okien)  +0,00483
        #     W= 11 (23 okna)   -0,00382
        # Zastepujemy `h.mean(-1)` srednimi po `pool_skale` segmentach. Nic nie
        # ginie: srednia globalna jest ich srednia, wiec model dostaje NADZBIOR
        # tego, co mial. Przy `pool_skale = 1` zachowanie jest bit w bit stare.
        _psk = int(kwargs.get("pool_skale", 1)) if isinstance(kwargs, dict) else 1
        self.pool_skale = max(1, _psk)
        _ppc = bool(kwargs.get("pool_per_cel", False)) if isinstance(kwargs, dict) else False
        if _ppc:
            self.pool_q = nn.ModuleDict({
                g: nn.Conv1d(trunk_dim, 1, 1, bias=False) for g in ("dev", "hk")})
            for m_ in self.pool_q.values():
                nn.init.zeros_(m_.weight)
        else:
            self.pool_q = None
        self.archetype = nn.Sequential(
            nn.Linear((self.pool_skale + 1) * trunk_dim
                      + (graph_dim if self.glowica_graf else 0),
                      hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        # GLOWICA HETEROSCEDASTYCZNA: obok `mu` przewidujemy `log sigma^2`.
        # POWOD, ZMIERZONY W ZRODLE: zgodnosc replik UMI-STARR-seq to
        # **PCC 0,73 (Dev) i 0,76 (Hk)** (Almeida i in., cytat dokladny), czyli
        # ~27 % wariancji Dev to szum pomiaru. MSE traktuje kazda etykiete jako
        # rownie wiarygodna, wiec model wydaje pojemnosc na dopasowanie do
        # szumu -- a to jest dokladnie mechanizm przeuczenia, ktore widzimy
        # (gap +0,133). Wzbogacenie to logarytm ILORAZU ZLICZEN, wiec fragmenty
        # o niskim pokryciu maja z definicji szumniejsze etykiety, a pokrycie
        # koreluje ze skladem sekwencji -- czyli jest wyuczalne z sekwencji.
        # PREDYKCJA POZOSTAJE SUROWA AKTYWNOSCIA: zwracamy `mu`, `logvar` idzie
        # wylacznie do straty.
        self.heteroskedastyczna = (bool(kwargs.get("heteroskedastyczna", False))
                                   if isinstance(kwargs, dict) else False)
        # `odczyt_latentu` przejmuje glowice WARIANCJI, bo jego wlasna jest
        # NIEZMIENNICZA na RC, a `fc_*_lv` czyta ceche trzonu, ktora nie jest.
        # Dwie glowice wariancji naraz nie powstaja: druga bylaby parametrami
        # bez zadania, ciagnietymi weight decayem -- ta sama usterka, ktora
        # `decoder is None if not recon_weight` juz raz zamknela.
        self._lv_z_latentu = (self.heteroskedastyczna
                              and bool(kwargs.get("odczyt_latentu", False)))
        if self.heteroskedastyczna and not self._lv_z_latentu:
            # TA SAMA szerokosc co glowice sredniej. `feat` jest juz
            # skonkatenowane z kontekstem podprzestrzen<->motyw, gdy on
            # istnieje, wiec glowice wariancji musza to widziec -- inaczej
            # `mat1 and mat2 shapes cannot be multiplied (96x288 and 256x1)`,
            # co wlasnie wyszlo. `pm_wy` jest ustawione wyzej, wiec czytamy
            # je tutaj, a nie liczymy drugi raz.
            _pmw = self.pm_wy if getattr(self, "pm_dev", None) is not None else 0
            # ============ WARIANCJA Z POZIOMU AKTYWNOSCI ============
            #
            # ZMIERZONE, GDZIE SIEDZI SYGNAL. Sufit przewidywalnosci kwadratu
            # bledu z cech sekwencji to R^2 = 0,0495 out-of-sample (podzial
            # PO SEKWENCJACH, kontrola tasowania -0,0003). Rozbicie:
            #   sama predykcja        0,0453 (Dev) / 0,0339 (Hk)
            #   cechy trzonu dokladaja      +0,0040 / +0,0159
            #   latent                0,0428  -- MNIEJ niz cechy
            #   wszystko razem        0,0489  -- NIE wiecej niz cechy same
            # Czyli ~90 % przewidywalnej czesci bledu to zaleznosc od POZIOMU
            # AKTYWNOSCI, nie od tresci sekwencji.
            #
            # A glowica realizuje z tego 0,0024 (`lv_r` ~ 0,049, R^2 = r^2),
            # czyli **4,8 % wlasnego sufitu**. Wycofuje wiec swoj wczesniejszy
            # wniosek "kierunek zamkniety arytmetyka": arytmetyka zamyka
            # SUFIT, a nie mowi, ze glowica go osiaga. Rezerwa jest 20-krotna.
            #
            # DLACZEGO WARSTWA LINIOWA NIE WYSTARCZA. Zaleznosc wariancji od
            # sredniej jest typowo U-KSZTALTNA (wieksza na obu skrajach
            # aktywnosci), a `nn.Linear(feat)` moze wyrazic najwyzej
            # `a*dev + b`, bo `dev = fc_dev(feat)` jest liniowe w `feat`.
            # `|dev|` ani `dev^2` sa dla niej NIEOSIAGALNE. Dokladamy je
            # jawnie.
            #
            # `.detach()` JEST WARUNKIEM KONIECZNYM, nie ostroznoscia:
            # inaczej cel wariancji zyskuje droge gradientu na SREDNIA, a
            # predykcje sa w tej rodzinie ZMIERZONE jako scisniete do
            # sredniej (nachylenie regresji 0,8414 Dev / 0,8887 Hk,
            # sd(pred)/sd(cel) 1,331/1,530 i 1,524/1,703, czyli 13 % i 11 %
            # za wasko). Przy odcieciu glowica wariancji NIE MOZE tego
            # pogorszyc -- jest wkladem, ktory z konstrukcji nie kosztuje PCC.
            #
            # Skala 1/2: `dev` ma sd ~1,5, wiec `dev^2` doszloby do ~9 i przy
            # `weight_decay: 0.08` te trzy kolumny mialyby inny rezim niz
            # `feat`. Dzielenie przez 2 jest stala, nie statystyka batcha --
            # statystyka wprowadzilaby zaleznosc od skladu batcha.
            self.lv_z_sredniej = bool(kwargs.get("lv_z_sredniej", False))
            _lvs = 3 if self.lv_z_sredniej else 0
            self.fc_dev_lv = nn.Linear(hidden_dim + _pmw + _lvs, 1)
            self.fc_hk_lv = nn.Linear(hidden_dim + _pmw + _lvs, 1)
            for _m in (self.fc_dev_lv, self.fc_hk_lv):
                nn.init.zeros_(_m.weight); nn.init.zeros_(_m.bias)
        else:
            self.fc_dev_lv = self.fc_hk_lv = None
            self.lv_z_sredniej = False
        # GLOWICE POSZERZONE o kontekst z atencji podprzestrzen<->motyw.
        # Konkatenacja, nie brama: cztery galezie za skalarna brama umarly
        # w tym repo, a zmierzone jest, ze zabija je gradient.
        _pm = self.pm_wy if self.pm_dev is not None else 0
        self.fc_dev = nn.Linear(hidden_dim + _pm, 1)

        # KOD GLOBALNY — jedna etykieta grupy na sekwencje.
        # Zmierzony sufit (sonda_grupowanie.py): 32 grupy maja 97 % tego,
        # co wyjasnia caly ciagly model (eta^2 0,4954 z 0,510 dla Dev,
        # 0,5924 z 0,607 dla Hk), a dzisiejszy latent — 0,0050, czyli sto
        # razy mniej, bo drabina uszczegolawia POZYCJE, nie SEKWENCJE.
        # Wpiety ADDYTYWNIE i ZERO-INIT: przy starcie model jest bit w bit
        # baza. Waskie gardlo byloby blede — predykcja z samej grupy ma
        # sufit sqrt(0,4954) = 0,704, czyli PONIZEJ dzisiejszych 0,714.
        _kg = int(kwargs.get("kod_globalny_K", 0) or 0)
        self.kod_glob = None
        if _kg > 0:
            from .kod_globalny import KodGlobalny
            self.kod_glob = KodGlobalny(
                hidden_dim + _pm, K=_kg,
                waga=float(kwargs.get("kod_globalny_commit", 0.25)),
                martwe_po=int(kwargs.get("kod_globalny_martwe_po", 20)))
        self._kg_strata = None
        self.fc_hk = nn.Linear(hidden_dim + _pm, 1)

        # --- GLOWICE HIPERBOLICZNE (opcjonalne, addytywne) ---
        # Konstruowane NA KONCU __init__, zeby strumien RNG dla wszystkiego
        # powyzej byl bit w bit taki sam jak w bazie. Przy `hyp_heads: false`
        # nie powstaje ani jeden parametr i model jest tozsamy z rekordzista.
        cfgh = self._hyp_heads_cfg
        if cfgh["wlacz"]:
            from .hyp_heads import HypHeads
            # DLUGOSCI Z JEDNEGO ZRODLA PRAWDY, nie z przeliczonego wzoru.
            #
            # Poprzednia wersja liczyla je tutaj wzorem
            # `(L + 2*(s//2) - 2*s)//s + 1`, ktory NIE ZNA `rc_pelne`.
            # A `rc_pelne` zmienia arytmetyke dopelnienia (okna musza byc
            # symetryczne na odbicie, patrz `_rc_ksztalt` i `dobierz_padding`),
            # wiec przy nim poziom 0 ma **17** komorek, nie 15. Tablica biasu
            # `AtencjaKodow` ma rozmiar L, wiec bledna liczba dawala
            # `RuntimeError: The size of tensor a (17) must match the size of
            # tensor b (15)` przy PIERWSZYM batchu -- ale tylko przy
            # `hyp_heads: true` RAZEM z `rc_pelne: true`, czyli w kombinacji,
            # ktorej nikt w tym repo nigdy nie zbudowal.
            # `HyperbolicCodeStack.dlugosci_poziomow` jest liczone swiadomie
            # dla oba przypadkow i jest tym samym zrodlem, z ktorego korzysta
            # `decode_codes` -- czyli zgodnosc z dyfuzja jest darmowa.
            _dl = _dlugosci_z_kodow(self.codes, int(seq_len))
            # GLOWICE BUDOWANE Z DEKLARACJI, nie ze stalej krotki. Patrz
            # `_hyp_heads_cfg["realnosc"]` -- podgalaz realnosci powstaje
            # WYLACZNIE przy `glowica_realnosci: true`, bo tylko wtedy istnieje
            # cel, ktory ja karmi (`training.negatywy_frac > 0`).
            _gl = ("dev", "hk") + (("realnosc",) if cfgh["realnosc"] else ())
            self.hyp_heads = HypHeads(
                self.codes.level_radii, dim=self.codes.dim,
                n_proto=cfgh["n_proto"], hidden=cfgh["hidden"],
                tau_skala=cfgh["tau_skala"],
                attn_codes=cfgh["attn"], dlugosci=tuple(_dl),
                glowice=_gl,
                geom_per_cel=bool(kwargs.get("geom_per_cel", False)))
            # TYLE WPISOW BRAMY, ILE GLOWIC. Nadwyzkowy wpis nie jest tani
            # inaczej niz arytmetycznie: jest parametrem BEZ GRADIENTU, ktory
            # `_hyp_brama` drukuje w kazdej epoce jako stala rowna
            # inicjalizacji -- i taka linia czyta sie jak zmierzony zerowy
            # wklad zywej galezi, a nie jak martwy wpis. Dokladnie ten argument
            # zdjal trzeci wpis `lat_brama` w galezi `odczyt_latentu`.
            self.hyp_gate = nn.Parameter(
                torch.full((len(_gl),), float(cfgh["brama"])))
            if cfgh["realnosc"]:
                # OBSERWABLA, nie komentarz. Model nie widzi `negatywy_frac`
                # (inny blok configu), wiec nie moze tego sprawdzic -- ale moze
                # powiedziec, czego wymaga, i to jest jedyna linia w logu,
                # z ktorej da sie ten stan odczytac.
                print("[model] glowica `realnosc` ZBUDOWANA: %d wag. Bez "
                      "`training.negatywy_frac > 0` nie ma ona ZADNEJ drogi "
                      "do straty -- `_realnosc` czyta tylko tamten blok."
                      % self._wagi_glowicy("realnosc"), flush=True)
        elif bool(kwargs.get("odczyt_latentu", False)):
            # (`hyp_aux_waga` ustawiane ponizej, wspolnie dla wszystkich
            #  wariantow glowic -- patrz koniec tego bloku.)
            # ODCZYT NIEZMIENNICZY NA RC. Trzecia proba tej galezi i pierwsza,
            # w ktorej wszystkie trzy zapisane przyczyny smierci sa zablokowane
            # STRUKTURALNIE, a nie zalatane:
            #
            #   P2a rozcienczenie centroidem po pozycjach (1/124 na kod L2)
            #       -- nie ma tu ani jednej sredniej przed odczytem; droga kodu
            #       do wyjscia jest pozycyjna, a redukcja zawiera `amax`,
            #       ktorego wzmocnienie wzgledem jednej pozycji wynosi 1;
            #   P2b `nn.LayerNorm` kasujacy skladowa WSPOLNA miedzy sekwencjami
            #       (sd 4,5e-02 -> 5,2e-06) -- modul asertuje w `__init__`, ze
            #       nie ma ANI JEDNEGO LayerNorma w swoim drzewie;
            #   P2c weight decay na malej bramie (0,02 -> 1,42e-04) -- brama
            #       nazywa sie `lat_brama` (wzorzec "brama"), a warstwa
            #       koncowa `skala_wy` (wzorzec "skala_"), wiec obie sa poza
            #       decayem; `GlowicaKodow` miala naprawione tylko P2a i P2b.
            #
            # NIEZMIENNICZOSC JEST TOZSAMOSCIA ALGEBRAICZNA, nie wynikiem
            # uczenia: zmierzone 1,5e-06 ... 2,9e-06 wobec kontroli negatywnej
            # 0,989-1,656, czyli separacja 5,3e+05.
            from .odczyt_z_latentu import OdczytZLatentu
            self.hyp_heads = OdczytZLatentu(
                dim=self.codes.dim, n_levels=self.codes.n_levels,
                kanaly=int(kwargs.get("ol_kanaly", 64)),
                jadro_par=int(kwargs.get("ol_jadro_par", 5)),
                n_par=int(kwargs.get("ol_n_par", 1)),
                szerokosc=int(kwargs.get("ol_szerokosc", 96)),
                jadro=int(kwargs.get("ol_jadro", 7)),
                n_blokow=int(kwargs.get("ol_bloki", 2)),
                ukryte=int(kwargs.get("ol_ukryte", 128)),
                dropout=float(kwargs.get("ol_dropout", 0.1)),
                okna=int(kwargs.get("ol_okna", 4)),
                inv_dim=int(kwargs.get("ol_inv_dim", 32)))
            # DWA wpisy, nie trzy: ten modul nie ma glowicy `realnosc`.
            # Trzeci wpis bylby parametrem bez gradientu i pojawilby sie
            # w kazdym raporcie jako stala, udajac zmierzony zerowy wklad.
            self.lat_brama = nn.Parameter(
                torch.full((2,), float(kwargs.get("ol_brama", 0.02))))
            self.hyp_gate = None
        elif bool(kwargs.get("glowica_kodow", False)):
            # POZYCYJNY ODCZYT Z KODOW zamiast martwych glowic hiperbolicznych.
            #
            # ZMIERZONE na HypEncI: `HypHeads` zwracaly STALA -- sd wyjscia
            # 5,93e-09 przy sd(Dev) = 1,61, a brama zeszla do 1,42e-04. Kosztuja
            # 81 108 parametrow (3,45 % modelu) za wklad +0,00000 (zerowanie
            # `hyp_gate` zmienia predykcje o 4,2e-08). Dwie przyczyny, obie
            # zmierzone i obie naprawione w `GlowicaKodow`: agregacja centroidem
            # rozciencza zmiane jednego kodu L2 do 1/124, a `nn.LayerNorm`
            # w odczycie kasuje skladowa WSPOLNA miedzy sekwencjami
            # (sd 4,5e-02 -> 5,2e-06) -- ta sama choroba, ktora `lat_center`
            # naprawil w `code_proj` (tresc per sekwencja 0,0668 -> 2,1167).
            #
            # Interfejs jest identyczny (`forward` + `prefiksy`), wiec strata
            # hierarchiczna `hier_weight` dziala bez zmian w `train_graph.py`.
            from .glowica_kodow import GlowicaKodow
            # To samo zrodlo prawdy co wyzej; `[::-1]`, bo ta galaz indeksuje
            # `_dl[0]` jako NAJDROBNIEJSZA mape, a `dlugosci_poziomow` jest
            # uporzadkowane od najgrubszej.
            _dl = list(_dlugosci_z_kodow(self.codes, int(seq_len)))[::-1]
            self.hyp_heads = GlowicaKodow(
                dim=self.codes.dim, n_levels=self.codes.n_levels,
                kanaly=int(kwargs.get("gk_kanaly", 64)),
                szerokosc=int(kwargs.get("gk_szerokosc", 128)),
                jadro=int(kwargs.get("gk_jadro", 7)),
                n_blokow=int(kwargs.get("gk_bloki", 3)),
                ukryte=int(kwargs.get("gk_ukryte", 128)),
                dropout=float(kwargs.get("gk_dropout", 0.1)),
                dlugosc=int(_dl[0]))          # najdrobniejsza mapa
            # Brama startuje na 0,02, NIE na zerze: zerowa brama nie przepuszcza
            # gradientu w gore i gasi wlasne zrodlo -- blad popelniony tu
            # czterokrotnie (`film_init_std`, wycofane).
            self.hyp_gate = nn.Parameter(
                torch.full((3,), float(kwargs.get("gk_brama", 0.02))))
        else:
            self.hyp_heads = None
            self.hyp_gate = None
        # CEL WLASNY GLOWIC KODOWYCH. Pelne uzasadnienie przy uzyciu
        # w `forward`. Domyslnie 0,0, wiec zaden istniejacy config nie
        # zmienia zachowania -- to jest warunek testu regresji na czterech
        # starych configach w `smoke_maska`.
        self.hyp_aux_waga = float(kwargs.get("hyp_aux_waga", 0.0) or 0.0)
        # "trzon" = cel z predykcji trzonu (zmierzone jako REMIS, patrz
        # `forward`); "etykiety" = cel z prawdziwych Dev/Hk, liczony
        # w `train_graph`, bo w `forward` etykiet nie ma.
        self.hyp_aux_cel = str(kwargs.get("hyp_aux_cel", "trzon"))
        if self.hyp_aux_cel not in ("trzon", "etykiety"):
            raise ValueError("hyp_aux_cel musi byc 'trzon' albo 'etykiety', "
                             "jest %r" % (self.hyp_aux_cel,))
        if self.hyp_aux_waga > 0.0 and self.hyp_heads is None:
            raise ValueError(
                "hyp_aux_waga > 0 bez glowic kodowych -- czlon nie mialby na "
                "czym stanac. Wlaczyc `hyp_heads: true` albo zdjac wage. "
                "(To ta sama rodzina co `negatywy_frac` przy martwej glowicy "
                "`realnosc`: koszt bez efektu, niewidoczny w logu.)")
        if not hasattr(self, "lat_brama"):
            self.lat_brama = None

        # HypGram: K slotow z drabiny + wyczerpujace pary miedzy nimi.
        # Wchodzi ADDYTYWNIE, tak jak `hyp_heads` -- trzon jest poziomem zerowym
        # predykcji, a gramatyka par go POPRAWIA. Przy zerowym wkladzie bieg
        # wraca do bazy sam z siebie.
        _ns = int(kwargs.get("gram_slotow", 0)) if isinstance(kwargs, dict) else 0
        if _ns > 0:
            from .hyp_gram import HypGram
            # Dlugosci map liczone TU, nie zapozyczone z galezi `hyp_heads` --
            # tamto `_dl` istnieje tylko przy `hyp_heads: true`, wiec HypGram
            # bez glowic dostawalby NameError. Cichy blad tego rodzaju juz raz
            # kosztowal w tej sesji caly bieg.
            _Lg, _dlg = int(seq_len), []
            for _s in self.codes.strides:
                _Lg = (_Lg + int(_s) - 1) // int(_s)
                _dlg.append(_Lg)
            _dlg = _dlg[::-1]                     # najgrubszy pierwszy
            self.gram = HypGram(
                dim=self.codes.dim, n_slotow=_ns,
                dlugosci=tuple(_dlg), dlugosc_bp=int(seq_len),
                glowice=("dev", "hk"),
                rundy=int(kwargs.get("gram_rundy", 2)),
                # `gram_wersja: 3` wlacza pakiet poprawek z 2026-09-06
                # (rownowazenie poziomow, temperatura na softmaxach po parach,
                # maska addytywna, cechy grupy rzedu 3). Domyslnie 2, zeby
                # HypGramPelny i HypGram2 pozostaly odtwarzalne bit w bit.
                wersja=int(kwargs.get("gram_wersja", 2)))
        else:
            self.gram = None

        # ============ GLOWICE POZIOMOWE (LevelResidualDecoder) ==============
        # PO CO. `LevelResidualDecoder` (`level_decoder.py:73`) ma DOKLADNIE
        # te mechanizmy, o ktore chodzi w tym kierunku: predykcja per poziom
        # i REZYDUALNIE, atencja poziom->trzon (zapytanie z podsumowania
        # poziomu), atencja poziom->poziom z biasem geodezyjnym (`cross_mix`),
        # blizniacze glowice Dev/Hk rozdzielone az do PROTOTYPOW, gleboki
        # nadzor na sumach czesciowych (`deep_supervision`), MI z celem per
        # poziom (`mi_z_celem`) i temperatura per powloka z prawa sinh^2(r).
        #
        # A byl instancjonowany WYLACZNIE w `HypCompactNet`
        # (`hyp_compact_net.py:486`) i ANI RAZU w tej klasie -- czyli w
        # rodzinie, ktora buduje wszystkie 108 configow, w tym HD5-HD20 i
        # config kanoniczny. Zmierzone skutki tego rozjazdu:
        #   * w HypVQNet podmiana `x_hat` na PRAWDZIWE `x` zmienia PCC o
        #     0,00005 (`sonda_deficytu.py`), czyli trzon robi konwencjonalna
        #     predykcje z sekwencji, a latent jest wylacznie nosnikiem
        #     rekonstrukcji;
        #   * w rodzinie, gdzie `level_dec` byl wpiety, jego wyzerowanie
        #     kosztowalo -0,58815, a tasowanie po batchu -0,39380 -- czyli
        #     niosl WIEKSZOSC predykcji.
        #
        # DOMYSLNIE WYLACZONE i to jest warunek zaporowy: przy
        # `glowice_poziomowe: false` nie powstaje ani jeden parametr i nie
        # rejestruje sie ani jeden hak, wiec kazdy istniejacy config buduje
        # sie BIT W BIT (regresja `smoke_maska`: HypStozek 2 277 210,
        # HypEncNapr 2 268 112, HypFinal2 2 575 263,
        # HyppeDeepSTARR_pomiarowy 2 293 450).
        #
        # Konstruowane NA KONCU `__init__`, po `hyp_heads` i `gram`, z tego
        # samego powodu co one: strumien RNG dla wszystkiego powyzej zostaje
        # nietkniety.
        self.glowice_poziomowe = bool(kwargs.get("glowice_poziomowe", False))
        self.level_dec = None
        self._mapa_trzonu = None
        self._pred_trzonu = None
        self._pred_poziomow = None
        self._gp_wklad = None
        self._gp_r_trzon = None
        if self.glowice_poziomowe:
            from .level_decoder import LevelResidualDecoder
            # WYMIAR ROZMAITOSCI NIE JEST WOLNYM POKRETLEM. `geo_features`
            # liczy `minkowski_dot(point, P)`, a `point` ma `codes.dim + 1`
            # wspolrzednych (patrz `punkty_poziomow`), wiec prototypy musza
            # miec dokladnie tyle samo. `gp_hyp_dim: 0` znaczy "wez
            # `codes.dim`"; kazda inna wartosc jest bledem configu, ktory
            # inaczej padlby dopiero na pierwszym batchu w mnozeniu macierzy.
            _gp_hd = int(kwargs.get("gp_hyp_dim", 0) or 0) or int(self.codes.dim)
            if _gp_hd != int(self.codes.dim):
                raise ValueError(
                    "gp_hyp_dim=%d wobec codes.dim=%d. Prototypy dekodera "
                    "poziomow leza na TEJ SAMEJ hiperboloidzie co kody, wiec "
                    "wymiar nie jest wolny -- `minkowski_dot` wymaga zgodnosci "
                    "ostatniego wymiaru. Zostawic gp_hyp_dim: 0."
                    % (_gp_hd, int(self.codes.dim)))
            _gp_zap = int(kwargs.get("gp_zapytania", 0) or 0)
            # KANALY WARTOSCI ATENCJI GEODEZYJNEJ.
            #
            # `HypCompactNet` podaje `tuple(len(st.levels) for st in stages)`,
            # czyli liczbe cyfr siatki FSQ per poziom. Tu tego zrobic NIE
            # MOZNA z dwoch niezaleznych powodow, oba sprawdzone czytaniem
            # kodu: `codes.drabina.kroki` ma `n_levels - 1` wpisow (poziom 0
            # to `vq_pozycji` z kodebookiem NA ROZMAITOSCI, nie krok
            # styczny), wiec `fsq_ch[n-1]` w petli `for k, r in
            # enumerate(self.radii)` wyszloby poza zakres; a HypVQNet nie
            # przechowuje ZADNEGO tensora cyfr FSQ per slot -- `codes_q` /
            # `codes_hyp` istnieja tylko w stosie widmowym rodziny Compact.
            #
            # Wartoscia jest wiec CZESC PRZESTRZENNA PUNKTU, czyli sam wpis
            # kodebooka (`punkty_poziomow()[k][..., 1:]`): dostepna na KAZDYM
            # poziomie, w tym na L0, i bedaca dokladnie tym, co kwantyzator
            # przypisal. Szerokosc to `codes.dim` na kazdym poziomie.
            _gp_fsq = tuple(int(self.codes.dim) for _ in range(self.codes.n_levels))
            self.level_dec = LevelResidualDecoder(
                n_levels=int(self.codes.n_levels),
                code_dim=int(self.codes.dim),
                trunk_dim=int(trunk_dim),
                level_radii=self._promienie_poziomow(),
                hyp_dim=_gp_hd,
                n_proto=int(kwargs.get("gp_protos", 12)),
                hidden=int(kwargs.get("gp_hidden", 128)),
                attn_dim=int(kwargs.get("gp_attn", 64)),
                ramp_epochs=int(kwargs.get("gp_ramp", 3)),
                aux_weight=float(kwargs.get("gp_aux", 0.25)),
                tau=float(kwargs.get("gp_tau", 0.5)),
                cross_level=bool(kwargs.get("gp_cross", True)),
                cross_tau=float(kwargs.get("gp_cross_tau", 1.0)),
                hyp_queries=_gp_zap,
                hq_dim=int(kwargs.get("gp_hq_dim", 32)),
                fsq_ch=_gp_fsq,
                twin_heads=bool(kwargs.get("gp_twin", True)))
            # SKALA GEODEZYJNA WOLNA OD PROMIENIA -- warunek waznosci
            # KAZDEJ liczby `gp_geo_*`.
            #
            # `geo_features` dzieli kwadrat odleglosci przez `taus[k]`
            # z prawa temperatury `tau_k ~ sinh^2(r_k)`, ktore zaklada, ze
            # oba punkty leza na powloce r_k. My podajemy CENTROID LORENTZA
            # po slotach, bo dekoder chce jednego punktu na poziom -- a
            # centroid dla rozbieznych kierunkow siedzi blizej origa.
            # Zmierzone po wpieciu: sinh(rho)/sinh(r) = 0,648-0,732 wobec
            # wymaganych ~0,99, i `gp_geo_mi_bits` 0,076-0,132 wobec
            # zapisanego poziomu szumu 0,341, czyli odczyt byl OZDOBA.
            # Docstring `geo_features` opisuje ten przypadek wprost jako
            # realny defekt "dla innego configu" -- i tym configiem jest ten.
            #
            # Skala jest na BUFORZE EMA (patrz `level_decoder.geo_skala`),
            # nie na sredniej batcha, bo srednia batcha uzalezniloby
            # predykcje od jego skladu i zlamala umowe dyfuzji.
            self.level_dec.geo_scale_free = True
            if not _gp_zap:
                # USTERKA W `renormalise_gauge`, ZNALEZIONA PRZY WPINANIU.
                # Ta metoda iteruje po `(self.hq_proto, self.protos, ...)`
                # BEZWARUNKOWO, a `hq_proto` powstaje tylko przy
                # `hyp_queries > 0`. `train_graph.py:1531` wola ja po KAZDYM
                # kroku optymalizatora dla kazdego modelu z `level_dec`, wiec
                # `gp_zapytania: 0` wywalaloby bieg na AttributeError w
                # pierwszym kroku. Nigdy nie wyszlo, bo wszystkie trzy configi
                # rodziny Compact ustawiaja `hyp_queries: 4`.
                #
                # `None` zamiast braku atrybutu: petla ma jawne `if lst is
                # None: continue`, wiec prototypy `protos`/`protos_hk` sa
                # DALEJ reprojektowane. Wyciecie calego wolania byloby gorsze
                # -- to wlasnie one dryfuja do origa (zmierzone 1142x).
                self.level_dec.hq_proto = None
            # MAPA TRZONU. Tak samo jak w `HypCompactNet` (linia 657): hak na
            # wyjsciu OSTATNIEGO bloku, bo `h` po petli blokow jest zmienna
            # LOKALNA w `embed`, a przepisywanie `embed` dla jednego tensora
            # zamienia jeden problem w piec. Hak nie tworzy parametrow.
            self.blocks[-1].register_forward_hook(self._przechwyc_trzon)

        self.register_buffer("_zero", torch.zeros(()), persistent=False)
        self.reg_loss = self._zero

    @torch.no_grad()
    def band_report(self, *args, **kwargs):
        """Diagnostyka kodebooka. Bez niej bieg tej klasy jest nieobserwowalny.

        `train_graph.py` szuka metody raportu pod jedna z pieciu nazw i drukuje
        wynik jako linie `lat:`. HypVQNet nie mial zadnej, wiec nie produkowal
        ANI JEDNEJ liczby o kodebooku -- a warunkiem niezbednym alarmu zapadu
        jest spadek `bits_used_frac`. Alarm bylby wiec slepy na modelu
        priorytetowym, a jego cisza nierozroznialna od zdrowia.

        Kazda liczba pochodzi Z TEJ SAMEJ warstwy VQ: perplexity, rozmiar
        kodebooka i obsadzenie sa czytane per warstwa, a nie z dwoch osobnych
        list, ktore moglyby sie rozjechac kolejnoscia.

        `*args` przyjmowane, bo `train_graph` probuje najpierw wolac raport z
        argumentem `_occ_hits` i dopiero po `TypeError` bez argumentu.
        """
        out = {}
        st = getattr(self, "codes", None)
        if st is None:
            return out

        warstwy = [m for m in st.modules()
                   if hasattr(m, "cluster_size") and hasattr(m, "perplexity")]
        if not warstwy:
            return out

        perp, rozm, uzyte, frakcje = [], [], [], []
        for w in warstwy:
            cs = w.cluster_size.detach().float()
            r = int(cs.numel())
            try:
                p = float(w.perplexity())
            except Exception:
                p = float("nan")
            # Prog WZGLEDNY: kod zywy, gdy jego masa EMA przekracza 1% masy
            # sredniej. Prog bezwzgledny nie przenosi sie miedzy poziomami o
            # roznych rozmiarach kodebooka.
            prog = 0.01 * float(cs.mean())
            perp.append(p)
            rozm.append(r)
            uzyte.append(round(float((cs > prog).float().mean()), 4))
            if p == p and r > 1:
                frakcje.append(math.log2(max(p, 1.0)) / math.log2(r))

        out["perplexity"] = [round(p, 2) for p in perp]
        out["code_ent_bits"] = [round(math.log2(max(p, 1.0)), 3)
                                if p == p else None for p in perp]
        out["code_ent_max"] = [round(math.log2(r), 2) for r in rozm]
        out["code_used_frac"] = uzyte
        # UZYCIE EMPIRYCZNE, z biezacego batcha i PER KOMORKA. `code_used_frac`
        # powyzej czyta EMA z historia i progiem wzglednym, wiec pokazywal
        # ~1,0 przy 16 z 128 kodow realnie wybieranych. Ta petla dodaje liczby,
        # ktorych tamten wskaznik nie umie pokazac -- alarm: `emp_H_poz < 2,0`
        # znaczy, ze poziom jest tablica pozycyjna, a nie kodem tresci.
        for _w in warstwy:
            _ue = getattr(_w, "uzycie_empiryczne", None)
            if not callable(_ue):
                continue
            try:
                _r = _ue()
            except Exception:
                _r = {}
            for _k, _v in (_r or {}).items():
                out.setdefault(_k, []).append(_v)
        if frakcje:
            # Udzial WYKORZYSTANEJ entropii w osiagalnej, sredni po poziomach.
            # Bez wazenia liczba slotow, bo alarm zapadu uzywa kryterium
            # WZGLEDNEGO (spadek wobec wlasnego maksimum biegu), a to jest
            # niewrazliwe na wybor wagi. Wazenie slotami dawalo mianownik
            # policzony z domyslu [1,1,1] i liczbe 24 zamiast 1741 -- lepiej
            # nie raportowac wcale niz raportowac zle.
            out["bits_used_frac"] = round(sum(frakcje) / len(frakcje), 4)

        # `radii()` zwraca WEKTOR promieni na kod dla kazdego poziomu, nie
        # skalar -- pierwsza wersja robila `float()` na tensorze i dostawala
        # ValueError, cicho przechwycony. Srednia i rozrzut mowia wiecej:
        # docstring `HyperbolicVQ.radii` stawia warunek, ze hierarchia ma sie
        # ROZWARSTWIAC (kody ogolne blisko origo, szczegolowe daleko), a plaski
        # rozklad promieni znaczy, ze geometria hiperboliczna nie jest uzywana
        # i model dzialalby tak samo w przestrzeni euklidesowej.
        # `recon_kody` mowi, ile dekoder bierze Z KODOW, a nie z obejscia.
        # Roznica `recon - recon_kody` to bezposrednia miara zaleznosci od
        # kanalu ciaglego: 0,00 = obejscie zbedne, 0,45 = obejscie robi wszystko.
        _rk = getattr(self, "_recon_kody", None)
        if _rk is not None:
            out["recon_kody"] = round(float(_rk), 4)

        # FALSYFIKATORY DRABINY STOZKOWEJ. Wchodza z przedrostkiem `dr_`, zeby
        # nie zderzyly sie z kluczami kodebooka poziomu 0 (`uzyte`, `perplexity`
        # wystepuja w obu). Bez tego bloku bieg drabiny nie produkowalby ANI
        # JEDNEJ liczby o zawieraniu -- a `cos_rodzic` wobec `cos_losowy` jest
        # jedynym falsyfikatorem calej konstrukcji.
        if getattr(self, "krzyz", None) is not None:
            # Bez tych liczb bieg z atencja krzyzowa nie produkuje ZADNEGO
            # falsyfikatora tego mechanizmu -- ta sama slepota, ktora HypPara
            # miala na `para_H` (ani jednej linii w 29 epokach).
            _kh, _kd, _kbr = [], [], []
            for _m in self.krzyz.values():
                _r = _m.raport()
                if _r:
                    _kh.append(_r["krzyz_H"]); _kd.append(_r["krzyz_dom"])
                    _kbr.append(_r["krzyz_brama"])
            if _kh:
                out["krzyz_H"] = _kh
                out["krzyz_H_max"] = round(
                    list(self.krzyz.values())[0].raport()["krzyz_H_max"], 3)
                out["krzyz_dom"] = _kd
                out["krzyz_brama"] = _kbr
        if getattr(st, "wieza", None) is not None:
            for _k, _v in st.wieza.raport().items():
                out["w_" + _k] = _v
        if getattr(self, "_jepa_war", None) is not None:
            out["jepa_war"] = self._jepa_war
            out["jepa_skal"] = getattr(self, "_jepa_skal", None)
        if self.decoder is not None:
            for _w in self.decoder:
                _r = getattr(_w, "raport", None)
                if callable(_r):
                    out.update(_r() or {})
        if getattr(self, "_hyp_wklad", None) is not None:
            # `hyp_wklad` = RMS(wklad glowic) / RMS(wyjscia).
            #
            # FALSYFIKATOR JEST NA WZROST, NIE NA PROG. Przy bramie startowej
            # 0,02 wartosc na inicjalizacji to zmierzone (0,006, 0,013), wiec
            # prog bezwzgledny "ponizej 0,01 = martwe" oglaszalby porazke
            # w epoce 1 z konstrukcji -- ten sam blad, ktory w tym repo
            # uniewaznil test `fsq_entropia` (`any(v > 0)` zapalalo sie
            # dokladnie wtedy, gdy czlon dziala) i kontrole na zerowo
            # inicjalizowanej `fc_dev_lv`.
            # ALARM: `hyp_wklad` NIE ROSNIE wobec epoki 1 albo `hyp_brama`
            # spada -- wtedy cala galaz (atencja kodow, prototypy, odczyt
            # geodezyjny) jest ozdoba NIEZALEZNIE od tego, co pokazuje
            # `attn_H`. Zmierzona awaria tej rodziny: brama 0,02 -> 1,42e-04
            # i glowice zwracajace stala o sd 5,93e-09, przy braku
            # jakiegokolwiek klucza w logu, ktory by to pokazal.
            out["hyp_wklad"] = self._hyp_wklad
            out["hyp_brama"] = getattr(self, "_hyp_brama", None)
            # ZRODLO, NIE TYLKO WKLAD. `hyp_wklad` jest iloczynem bramy i
            # zrodla, wiec zero w nim nie rozroznia "brama zamknieta" od
            # "glowica zwraca stala". Udokumentowana smierc tej galezi to
            # sd 5,93e-09 przy sd(Dev) = 1,61 -- czyli drugi przypadek.
            if getattr(self, "_hyp_zrodlo", None) is not None:
                out["hyp_zrodlo"] = self._hyp_zrodlo
            if getattr(self, "_hyp_aux", None) is not None:
                out["hyp_aux"] = self._hyp_aux
            # KORELACJA RESZT galezi i trzonu (nie predykcji -- pierwsza
            # wersja liczyla predykcje i dala 0,9898 w epoce 1, bo dwa
            # predyktory tej samej etykiety sa skorelowane Z DEFINICJI).
            # Blisko 1,0 = te same bledy, czyli suma nie moze pomoc.
            # Wyraznie ponizej = bledy rozlaczne, jest z czego zyskac.
            if getattr(self, "_hyp_kor", None) is not None:
                out["hyp_kor"] = self._hyp_kor
        if getattr(self, "_rt_raport", None):
            # FALSYFIKATORY KANALU "TRZON CZYTA DEKODER":
            #   `rt_w`   -- ile `x_hat` realnie poszlo do trzonu. Musi dojsc
            #               do 1,0000; ponizej tego predykcja NIE jest
            #               funkcja wylacznie kodow i `_full.pt` nie powstaje.
            #   `rt_ent` -- entropia `softmax(logity/tau)` w bitach. Przy
            #               nasyceniu (~0) pochodna softmaxu jest zerem i
            #               sprzezenie predykcja -> dekoder jest OZDOBA,
            #               mimo ze wartosc dla trzonu wyglada dobrze.
            #               Prog: > 0,01 bitu. Regulator: `recon_tau`.
            #   `rt_max` -- srednie max p. Blisko 1,0 przy niskim `rt_ent`
            #               to ten sam alarm widziany od drugiej strony.
            out.update(self._rt_raport)
        if getattr(self, "_maska_raport_tren", None) or getattr(self, "_maska_raport", None):
            # FALSYFIKATORY MASKOWANIA, wszystkie w jednej linii logu:
            #   `rek_pelny`   -- nie moze spasc; regresja rekonstrukcji
            #                    z samych kodow (dzis 1,0000).
            #   `rek_kolumna` -- dokladnosc NA POZYCJACH ZAMASKOWANYCH.
            #                    MUSI przekroczyc podloge skladu 0,2934,
            #                    inaczej model odtwarza sklad, nie gramatyke.
            #   `rek_prefiks_0` -- dekodowanie z SAMEGO L0. Dzis 0,2775,
            #                    czyli PONIZEJ podlogi 0,2934: hierarchia nie
            #                    ma wlasnosci prefiksu. To jest liczba, ktora
            #                    ten mechanizm ma podniesc.
            #   `mask_frakcja` -- REALNA frakcja zamaskowana (bloki moga sie
            #                    nalozyc), `mask_norma` -- czy wektor MASK nie
            #                    zapadl do zera pod weight decay.
            out.update(getattr(self, "_maska_raport_tren", None)
                       or self._maska_raport)
        if getattr(self, "hipergraf", None) is not None:
            out.update(self.hipergraf.raport())
        for _nm, _m in (("dev", getattr(self, "pm_dev", None)),
                        ("hk", getattr(self, "pm_hk", None))):
            if _m is not None:
                for _k, _v in _m.raport().items():
                    out["%s_%s" % (_k, _nm)] = _v
        if getattr(self, "rozdziel", None) is not None:
            out.update(self.rozdziel.raport())
        if self.kanon_rc and self._orient is not None:
            # Frakcja probek, na ktorych wzieto nic odwrotna. Powinna byc
            # ~0,5; wartosc skrajna znaczy, ze kryterium leksykograficzne
            # jest zdegenerowane (np. przez kolumny `N`).
            out["kanon_frakcja"] = round(float(self._orient.float().mean()), 4)
        if getattr(st, "mot_l0", None) is not None:
            # Bez tych liczb bieg z atencja motywowa do L0 nie produkuje
            # ZADNEGO falsyfikatora tego mechanizmu -- ta sama slepota,
            # ktora HypPara miala na `para_H` przez 29 epok.
            out.update(st.mot_l0.raport())
        if getattr(self, "trojkat", None) is not None:
            # Bez tych liczb bieg z torem trojkatnym nie produkuje ZADNEGO
            # falsyfikatora -- dokladnie ta slepota, ktora HypPara miala na
            # `para_H` przez 29 epok.
            out.update(self.trojkat.raport())
            if self._troj_strata is not None:
                out["troj_strata"] = round(float(self._troj_strata), 5)
        _sm = getattr(st.stem[0], "raport", None)
        if _sm is not None:
            # Bez tego bloku `dom_motywow` nie trafia do logu, a cisza
            # falsyfikatora jest nieodroznialna od zdrowia -- ten sam defekt,
            # ktory `band_report` juz raz w tym repo zamknal.
            out.update(_sm())
        _rd = st.raport_drabiny() if hasattr(st, "raport_drabiny") else None
        if _rd:
            for _k, _v in _rd.items():
                out["dr_" + _k] = _v

        if hasattr(st, "radii"):
            try:
                sr, roz = [], []
                for t in st.radii():
                    t = t.detach().float().reshape(-1)
                    sr.append(round(float(t.mean()), 4))
                    roz.append(round(float(t.std()), 4))
                out["radius_mean"] = sr
                out["radius_std"] = roz
            except Exception as _e:
                out["radii_blad"] = type(_e).__name__

        # Promien WYJSCIA ENKODERA wobec przypietego promienia kodebooka.
        # Rozjazd tych dwoch liczb jest bezposrednia miara tego, ile `commit`
        # musi nadrabiac -- a `commit` nie ma sufitu.
        _er, _cm, _ns, _rsd, _pow = [], [], [], [], []
        _spn, _spa = [], []
        for _v in getattr(self.codes, "vqs", []):
            _e2 = getattr(_v, "_enc_radius", None)
            if _e2 is not None:
                _er.append(round(_e2[0], 4))
            _c2 = getattr(_v, "_commit_raw", None)
            if _c2 is not None:
                _cm.append(round(_c2, 4))
            # FALSYFIKATOR NASYCENIA PASMA. `r_nasyc` to sredni |tanh| bramki
            # radialnej. Wartosc blisko 1 znaczy, ze pasmo zostalo zjedzone
            # przez STALE przesuniecie i nie niesie tresci per sekwencja --
            # zmierzone na HypEncZ ep1: 0,999 na L0 i 0,872 na L2, przy
            # koszcie -4,6 sigma. PROG: powyzej 0,8 mechanizm jest martwy.
            # `r_sd_w` musi byc > 0, inaczej promien jest stala mimo pasma.
            _sn = getattr(_v, "_spec_nasyc", None)
            if _sn is not None:
                _spn.append(round(_sn, 4))
                _spa.append(round(getattr(_v, "_spec_a", 0.0), 4))
            _n2 = getattr(_v, "_r_nasyc", None)
            if _n2 is not None:
                _ns.append(round(_n2, 4))
            _s2 = getattr(_v, "_r_sd_wewn", None)
            if _s2 is not None:
                _rsd.append(round(_s2, 5))
            # Promien powloki PO uczonym przesunieciu. Musi zachowac
            # uporzadkowanie L0 < L1 < L2, inaczej hierarchia radialna padla.
            try:
                _pow.append(round(float(_v.promien_efektywny()), 4))
            except Exception:
                pass
        if _er:
            out["enc_radius"] = _er
        if _cm:
            out["commit_raw"] = _cm
        if _ns:
            out["r_nasyc"] = _ns
        if _spn:
            out["spec_nasyc"] = _spn
            out["spec_a"] = _spa
        if _rsd:
            out["r_sd_w"] = _rsd
        if _pow:
            out["powloka"] = _pow
        # ===== FALSYFIKATOR LANCUCHA REZYDUALNEGO =====
        # `kor_reszty`  -- korelacja reszty z ZANEGOWANYM kodem poprzedniego
        #                  poziomu. Przed naprawa: 0,9997 i 0,9985, czyli
        #                  drabina kwantyzowala sama siebie. PROG: musi spasc
        #                  wyraznie ponizej 0,9. Wartosc blisko 1 znaczy, ze
        #                  naprawa skalowa nie zadziala i poziomy L1/L2 nadal
        #                  opisuja kod L0, a nie sekwencje.
        # `udzial_sekw` -- ||mapa|| / ||reszta||, czyli ile reszty pochodzi
        #                  z SEKWENCJI. Przed naprawa: 0,026 i 0,054.
        #                  PROG: musi wzrosnac powyzej 0,3.
        # `skala_reszty`-- sama skala; start 0,0265 / 0,0542. Powrot do 1,0
        #                  oznacza powrot do awarii.
        # ===== RZAD EFEKTYWNY WEJSCIA KWANTYZATORA =====
        # GLOWNY falsyfikator tego biegu. Zmierzone na HypEncSkala_best:
        # 1,03 (L0) / 2,89 (L1) / 12,24 (L2) z 64 wymiarow, czyli enkoder
        # koduje na L0 SKALAR. Rzad liczony na KIERUNKACH, bo kwantyzacja jest
        # katowa: exp(entropia wartosci wlasnych macierzy korelacji).
        # PROG: musi WZROSNAC na kazdym poziomie. Brak wzrostu = bloki nie
        # wzbogacaja reprezentacji i mechanizm jest ozdoba.
        _ek = [b.se.raport() for b in getattr(self, "blocks", [])
               if getattr(b, "ekstrema", False) and hasattr(getattr(b, "se", None), "raport")]
        if _ek:
            # FALSYFIKATOR SAMOFALSYFIKUJACY: `beta` mowi, ktory punkt osi
            # srednia<->maksimum wybral model. Powrot do ~0 = wolal srednia,
            # czyli teza o maksimum upada. `udzial_miedzy` ma wzrosnac powyzej
            # 0,30 wobec zmierzonych 0,066-0,169 dla SE.
            out["ekstrema_beta"] = [r["beta"] for r in _ek]
            out["ekstrema_udzial"] = [r["udzial_miedzy"] for r in _ek]
        _ag = getattr(self.codes, "agr_brama_log", None)
        if _ag is not None:
            out["agr_brama"] = [round(float(v), 4) for v in _ag.detach().exp()]
            out["agregacje"] = list(self.codes.agregacje)
        _dk = getattr(self.codes, "_dom_kontekst", None)
        if _dk is not None:
            out["dom_kontekst"] = round(_dk, 4)
            out["brama_kontekst"] = round(getattr(self.codes, "_brama_kontekst", 0.0), 4)
        _re = getattr(self.codes, "_rzad_ef", None)
        if _re:
            out["rzad_ef"] = [round(v, 2) for v in _re]
        _bs = getattr(self.codes, "blok_skala_log", None)
        if _bs is not None:
            out["blok_skala"] = [round(float(v), 4) for v in _bs.detach().exp()]
        _kr = getattr(self.codes, "_kor_reszty", None)
        if _kr:
            out["kor_reszty"] = [round(v, 4) for v in _kr]
        _us = getattr(self.codes, "_udzial_sekw", None)
        if _us:
            out["udzial_sekw"] = [round(v, 4) for v in _us]
        _sr = getattr(self.codes, "skala_reszty", None)
        if _sr is not None:
            out["skala_reszty"] = [round(float(v), 5) for v in _sr.detach()]
        # Falsyfikator hierarchii, ktory ISTNIAL w hierarchy_report(), ale nie
        # byl drukowany w ZADNYM logu -- wiec przez caly projekt nikt nie mogl
        # zobaczyc, ze wklad poziomow ROSNIE zamiast malec (0,9905 / 0,9986 /
        # 1,0029 na HypEncI_best).
        _ls = getattr(self.codes, "_last_steps", None)
        if _ls:
            out["wklad"] = [round(v, 4) for v in _ls]
            out["wklad_maleje"] = all(a > b for a, b in zip(_ls, _ls[1:]))

        # ENTROPIA ATENCJI PAROWEJ. Bez tej liczby bieg z `para_bloki` nie
        # produkuje ZADNEGO falsyfikatora tego mechanizmu, a jego cisza jest
        # nieodroznialna od zdrowia -- ten sam defekt, przez ktory alarm zapadu
        # kodebooka byl slepy, dopoki nie powstal `band_report`.
        # Przy suficie log2(L) atencja jest JEDNOSTAJNA, czyli robi dokladnie to
        # samo, co srednia globalna, ktora zastapila -- i wtedy wymiana bramki
        # SE na sprzezenie parowe jest zerem, a nie porazka pomyslu.
        _pe, _ps = [], None
        for _b in getattr(self, "blocks", []):
            _pa = getattr(_b, "para", None)
            _e = getattr(_pa, "_ostatnia_entropia", None) if _pa is not None else None
            if _e is not None:
                # `_e[0]` jest TENSOREM (patrz ParaAtencja.forward: trzymamy
                # tensor, zeby nie synchronizowac GPU na kazdym kroku), wiec
                # konwersja jest tutaj — raz na epoke.
                _pe.append(round(float(_e[0]), 3)); _ps = round(_e[1], 3)
        if _pe:
            out["para_H"] = _pe
            out["para_H_max"] = _ps
            # `para_brama` ZNIKLO: skalarnej bramy nie ma, bo zero bylo
            # stanem pochlaniajacym (zmierzone: 0,05 -> -0,0004 w trzech
            # epokach, gradient do galezi ~0). Falsyfikatorem jest teraz
            # WKLAD galezi do strumienia rezydualnego: start dokladnie 0
            # (zerowa `o`), wiec kazda wartosc > 0 znaczy, ze galaz rosnie,
            # a stagnacja na zerze znaczy, ze mechanizm jest martwy.
            out["para_wkl"] = [
                round(float(b.para._wklad), 5)
                for b in self.blocks
                if getattr(b, "para", None) is not None
                and getattr(b.para, "_wklad", None) is not None]
            # `para_wkl` jest po normalizacji, wiec z konstrukcji ~`skala` —
            # to STRAZ, ze ograniczenie trzyma (eksplozja do 1,399 byla
            # zmierzona). O tym, czy galaz czegokolwiek sie uczy, mowi
            # `para_sygnal`: RMS przed normalizacja wzgledem wejscia.
            out["para_sygnal"] = [
                round(float(b.para._sygnal), 5)
                for b in self.blocks
                if getattr(b, "para", None) is not None
                and getattr(b.para, "_sygnal", None) is not None]
        # POLE SALIENTNOSCI -- falsyfikator glowny nowego mechanizmu.
        # `pole_amp` to amplituda modulacji wzgledem samej mapy trzonu.
        # Start jest DOKLADNIE 0 (zeros_ na `pole_U`), wiec kazda wartosc
        # rozna od zera znaczy, ze model modulacji CHCE. Zero po kilku
        # epokach wyklucza hipoteze o interakcji motyw x pozycja -- i to
        # jest wynik, nie porazka pomiaru.
        # ===== GLOWICE POZIOMOWE =====
        # Bez tego bloku bieg z `glowice_poziomowe` nie produkowalby ANI
        # JEDNEJ liczby o nowej drodze -- a ten projekt ma siedem cicho
        # martwych mechanizmow znalezionych dokladnie z tego powodu.
        # Wszystko z przedrostkiem `gp_`, bo `level_decoder.report()` uzywa
        # nazw (`geo_ent`, `lvl_std`, `cross_bias_std`), z ktorych czesc
        # zderzylaby sie z kluczami kodebooka.
        #
        # CO CZYTAC, w kolejnosci wazkosci:
        #   `gp_wklad`         -- RMS wkladu galezi / RMS wyjscia, per
        #                         zadanie. Alarm na BRAK WZROSTU wobec epoki
        #                         1 (rampa `gp_ramp` tlumi start).
        #   `gp_geo_rho_frac`  -- sinh(rho)/sinh(r) dla punktu poziomu.
        #                         Punkty ida z centroidu Lorentza po slotach,
        #                         wiec ta liczba MUSI byc sprawdzona przed
        #                         czytaniem czegokolwiek `gp_geo_*`: przy
        #                         0,04 softmax jest plaski Z ARYTMETYKI i
        #                         odczyt geodezyjny nie mierzy niczego.
        #   `gp_geo_mi_y_dev`  -- I(przypisanie; kwantyl aktywnosci) w bitach,
        #   `gp_geo_mi_y_hk`      per poziom. To jedyna liczba, ktora mowi, czy
        #                         geometria wie cos o CELU; `gp_geo_mi_bits`
        #                         mierzy tylko zroznicowanie miedzy
        #                         sekwencjami i ma zmierzony poziom szumu
        #                         (L0 0,341 bitu). Liczona na ostatnim batchu
        #                         TRENINGOWYM -- patrz `mi_z_celem`.
        #   `gp_lvl_p_std`     -- odchylenie surowego wyjscia glowicy per
        #                         poziom. Blisko zera = poziom zamilkl.
        #   `gp_cross_bias_std`-- rozrzut biasu geodezyjnego w atencji
        #                         poziom->poziom. Zero = geometria nie
        #                         rozroznia poziomow i atencja jest zwykla
        #                         euklidesowa.
        #   `gp_promienie`     -- promienie powlok PODANE dekoderowi. Musza
        #                         zgadzac sie z `dr_promien` / `powloka`;
        #                         rozjazd znaczy, ze `taus` sa policzone dla
        #                         innej geometrii niz punkty (patrz
        #                         `_promienie_poziomow`).
        _gp = getattr(self, "level_dec", None)
        if _gp is not None:
            for _k, _v in (_gp.report() or {}).items():
                out["gp_" + _k] = _v
            out["gp_promienie"] = [round(float(r), 4) for r in _gp.radii]
            out["gp_tau"] = [round(float(t), 5) for t in _gp.taus]
            out["gp_ramp"] = round(float(_gp.ramp()), 4)
            if getattr(self, "_gp_wklad", None) is not None:
                out["gp_wklad"] = self._gp_wklad
            # `gp_r_trzon` -- korelacja odczytu SAMEGO trzonu z celem,
            # liczona w `train_graph` (tu nie ma etykiet). To JEDYNA liczba,
            # ktora rozdziela "galaz jest uzyteczna" od "trzon zamarl":
            # `gp_wklad` to iloraz RMS galezi do wyjscia, wiec rosnie
            # w OBU tych przypadkach i dwie przeciwne diagnozy daja te sama
            # liczbe. Prog: spadek ponizej 0,5 znaczy, ze trzon przestaje
            # przewidywac sam, a zmierzony zapad w rodzinie, w ktorej
            # `level_dec` dotad zyl, siegal 0,193.
            _rt = getattr(self, "_gp_r_trzon", None)
            if _rt is not None:
                out["gp_r_trzon"] = _rt
            if getattr(self, "_gp_nadzor", None) is not None:
                # Wartosc czlonu glebokiego nadzoru. `None` znaczy
                # `training.gp_nadzor: 0.0`, czyli nadzoru NIE MA -- i wtedy
                # poziomy dostaja gradient wylacznie przez sume, co jest
                # dokladnie tym stanem, ktory ten modul mial naprawic.
                out["gp_nadzor"] = self._gp_nadzor
        _ld = getattr(self, "_lat_dev", None)
        if _ld is not None:
            out["lat_dev"] = round(_ld, 4)
        _pa = getattr(self, "_pole_amp", None)
        if _pa is not None:
            out["pole_amp"] = round(_pa, 6)
            out["pole_skala"] = [round(v, 4) for v in getattr(self, "_pole_skala", [])]
        return out

    @torch.no_grad()
    def ustaw_orientacje(self, orient):
        """Narzuca nic kanoniczna (albo zdejmuje narzucenie przy ``None``).

        Do OCENY WARIANTOW. Bez tego `wybierz_orientacje` decyduje osobno dla
        kazdego wariantu, a poniewaz decyduje o niej 1-3 pierwsze zasady,
        1,26 % podstawien PRZEWRACA nic i kody zmieniaja sie hurtowo -- efekt
        wariantu jest wtedy nieciagly. Pelne liczby w `_codes_input`.
        """
        if orient is None:
            self._orient_wymuszony = None
            return
        if not self.kanon_rc:
            raise ValueError(
                "ustaw_orientacje przy kanon_rc: false -- nie ma czego "
                "narzucac, bo kanonizacja nie biegnie i model nie jest "
                "niezmienniczy na nic")
        o = orient if torch.is_tensor(orient) else torch.as_tensor(orient)
        self._orient_wymuszony = o.detach().reshape(-1).bool()

    def encode_codes(self, x):
        """Sekwencja -> lista INDEKSOW kodow, jeden tensor na poziom.

        Zwraca to, na czym ma operowac dyfuzja: [B, L_k] liczb calkowitych na
        kazdy z `n_levels` poziomow, od NAJGRUBSZEGO do najdrobniejszego (taka
        sama kolejnosc jak `indices` w `forward` drabiny).
        """
        self.codes(self._codes_input(x))
        return [i.detach().clone() for i in self.codes.last_indices]

    @torch.no_grad()
    def _z_i_punkty_z_indeksow(self, indices):
        """Indeksy -> `(z, punkty)`. WSPOLNA sciezka dyfuzji i predykcji.

        Wydzielone z `decode_codes`, zeby `predykcja_z_kodow` NIE miala
        drugiej kopii tej arytmetyki. Plik sam ostrzega w `decode_codes`,
        ze rozejscie sie drogi dyfuzji z przebiegiem wprzod jest awaria
        CICHA: sekwencja wyglada sensownie i nie odpowiada rekonstrukcji
        z treningu. Dwie kopie tego samego skalowania (`norm_poz`,
        `radius_scale`, `_skaluj`) byly by wprost zaproszeniem do takiego
        rozejscia -- wiec jest jedna.

        `punkty` to lista punktow drabiny na hiperboloidzie albo `None`,
        gdy drabiny nie ma. Nalezy do STANU KODOWEGO: `punkty_poziomow()`,
        `hyp_heads` i galaz `pm_dev` czytaja `codes._punkty`, wiec bez
        odtworzenia tej wielkosci predykcja z kodow byla by liczona przy
        punktach z POPRZEDNIEGO przebiegu -- to ta sama dziura, ktora
        w sondzie latent-only zamykalo pinowanie `_punkty` obok `z`
        (trzy iteracje, trzy dziury w pomiarze).
        """
        if len(indices) != self.codes.n_levels:
            raise ValueError("oczekiwano %d poziomow, dostalem %d"
                             % (self.codes.n_levels, len(indices)))
        # ======= STRAZ NA DLUGOSC KAZDEGO POZIOMU ===========================
        # `DrabinaStozkowa.z_indeksow` uzywa `dlugosci[k + 1]` dla k = 0..n-2,
        # czyli NIE UZYWA `dlugosci[0]` ANI RAZU. Zla dlugosc L0 nie jest wiec
        # bledem -- jest bazą, ktora zostaje CICHO zinterpolowana do dlugosci
        # poziomu 1, i `decode_codes` zwraca poprawnie wygladajaca, ZLA
        # sekwencje. Dla dyfuzji to najgorszy mozliwy tryb awarii: kandydat
        # jest oceniany, wynik jest liczba, nic nie pada.
        #
        # Bramkowane WYLACZNIE przez `dlugosci_poziomow`, czyli to samo
        # zrodlo prawdy, z ktorego korzystaja `z_indeksow` i `_dlugosci_z_kodow`
        # (przy `rc_pelne` liczone z `_rc_ksztalt`, bo tam arytmetyka
        # dopelnienia jest inna). Wariant bez drabiny tego atrybutu nie ma
        # i tam straz milczy -- przeliczanie dlugosci wlasnym wzorem byloby
        # dokladnie tym bledem, ktory ten plik juz raz zapisal jako zrodlo
        # cichego rozjazdu ksztaltu.
        _dl_ref = getattr(self.codes, "dlugosci_poziomow", None)
        if _dl_ref:
            _dl_ma = tuple(int(i.shape[1]) for i in indices)
            if _dl_ma != tuple(int(v) for v in _dl_ref):
                raise ValueError(
                    "dlugosci poziomow indeksow %s != %s (`dlugosci_poziomow`). "
                    "Poziom 0 jest szczegolnie grozny: `z_indeksow` czyta "
                    "`dlugosci[k+1]`, wiec zla dlugosc L0 zostaje CICHO "
                    "zinterpolowana i `decode_codes` zwraca sensownie "
                    "wygladajaca, ZLA sekwencje."
                    % (_dl_ma, tuple(int(v) for v in _dl_ref)))

        _pkt = None
        if getattr(self.codes, "drabina", None) is not None:
            # SCIEZKA DRABINY. `z_indeksow` odtwarza dokladnie to, co robi
            # `forward`: ta sama metoda `_krok` dla obu, wiec nie ma jak sie
            # rozejsc. Rozejscie tych dwoch drog jest awaria, ktora zabija
            # dyfuzje PO CICHU -- sekwencja wyglada sensownie i nie odpowiada
            # przebiegowi wprzod.
            _pkt = self.codes.drabina.z_indeksow(
                indices, dlugosci=self.codes.dlugosci_poziomow)
            czesci = [(p[..., 1:].transpose(1, 2).float()
                       / self.codes.norm_poz[k].to(torch.float32))
                      for k, p in enumerate(_pkt)]
        else:
            czesci = []
            for k, idx in enumerate(indices):
                vq = self.codes.vqs[k]
                e = vq.embed[idx.reshape(-1)].reshape(*idx.shape, -1)
                q = (e[..., 1:] / vq.radius_scale).transpose(1, 2).float()
                czesci.append(q)
        target = max(q.shape[-1] for q in czesci)
        # SKALOWANIE PRZEZ `self.codes._skaluj`, NIE przez wlasne `nearest`.
        # Ta petla byla JEDYNYM miejscem, w ktorym droga dyfuzji rozjezdzala
        # sie z przebiegiem wprzod: `forward` przy `rc_pelne` uzywa `linear`
        # (bo `nearest` liczy `floor` i nie komutuje z odbiciem -- zmierzone
        # 2,1e-01 i 4,4e-01 na L1/L2 wobec 6,4e-07 na L0), a tutaj stalo
        # `nearest` na sztywno. Skutek byl cichy: `decode_codes` zwracalo
        # sensownie wygladajaca sekwencje, ktora NIE odpowiadala rekonstrukcji
        # z treningu, i falsyfikator F3 (`decode_codes(P(idx))` wobec
        # `rc(decode_codes(idx))`) nie mial prawa przejsc niezaleznie od tego,
        # jak dobry jest dekoder.
        czesci = [self.codes._skaluj(q, target) for q in czesci]
        z = torch.cat(czesci, dim=1)
        return z, _pkt

    def decode_codes(self, indices, length=None):
        """Indeksy kodow -> logity zasad. To jest wejscie dyfuzji do sekwencji.

        DZIALA TYLKO przy `recon_bypass: false`. Przy otwartym kanale wejscie
        dekodera zawiera slot `feats[0] - aligned[-1]`, czyli wielkosc, ktorej
        przy GENEROWANIU nie ma i ktorej z kodow wyliczyc sie nie da -- wtedy
        metoda podnosi wyjatek zamiast po cichu zwrocic bzdure.

        Odwraca dokladnie to, co robi `HyperbolicVQ.forward`:
        wpis kodebooka [K, D+1] -> (embed[idx][..., 1:] / radius_scale).T,
        czyli [B, C, L_k]. Pominiecie dzielenia przez `radius_scale` (0,25 / 0,6
        / 1,2) dawaloby kody przeskalowane i cicho zla sekwencje przy poprawnie
        wygladajacym kodzie.
        """
        if self.decoder is None:
            raise RuntimeError(
                "dekoder nie istnieje: `recon_weight` musi byc > 0")
        if getattr(self.codes, "recon_bypass", True):
            raise RuntimeError(
                "dekodowanie z samych kodow wymaga `recon_bypass: false` -- "
                "przy otwartym kanale wejscie dekodera zawiera slot ciagly "
                "`feats[0] - aligned[-1]`, ktorego przy generowaniu nie ma")
        z, _ = self._z_i_punkty_z_indeksow(indices)
        logits = self.decoder(z)
        # `seq_len` JEST od 2026-09-10 zapisywane jako atrybut w `__init__`.
        # `getattr` z domyslna zostaje wylacznie dla checkpointow i podklas
        # sprzed tej poprawki -- wczesniej atrybutu nie bylo i ta linia
        # ZAWSZE interpolowala do 249, niezaleznie od `seq_len` w configu.
        n = int(length or getattr(self, "seq_len", 249))
        if logits.shape[-1] != n:
            logits = F.interpolate(logits, size=n, mode="linear",
                                   align_corners=False)
        return logits

    @torch.no_grad()
    def decode_to_onehot(self, indices, length=None):
        """Jak `decode_codes`, ale zwraca one-hot w kolejnosci A, C, T, G.

        Kolejnosc kanalow to `dna.NUCLEOTIDE_ORDER`, czyli A, C, T, G -- NIE
        A, C, G, T. Zwrocony tensor mozna podac wprost na wejscie predyktora,
        i to jest cala petla dyfuzji: rusz kod, zdekoduj, przewiduj.
        """
        lg = self.decode_codes(indices, length=length)
        oh = torch.zeros_like(lg)
        oh.scatter_(1, lg.argmax(1, keepdim=True), 1.0)
        return oh

    @torch.no_grad()
    def predykcja_z_kodow(self, indices, length=None):
        """Indeksy kodow -> `(dev, hk)`. To jest funkcja celu dyfuzji.

        DOTAD ISTNIALA TYLKO JAKO KONTRAKT WERYFIKOWANY HAKIEM.
        `sonda_tylko_kody2.py` mierzyla, ze predykcja jest funkcja
        wylacznie kodow (`max|pred(A|k=A) - pred(B|k=A)| = 0,000000e+00`),
        ale zeby z tego SKORZYSTAC, trzeba bylo powtorzyc ten hak
        u siebie. Dyfuzja sterowana potrzebuje oceniac KANDYDATA
        w przestrzeni kodow, nie w przestrzeni sekwencji, wiec brak tej
        metody znaczyl, ze umowa jest sprawdzona i nieuzywalna.

        JAK TO DZIALA, i dlaczego nie wystarczy `model(decode_to_onehot(idx))`.
        Enkoder policzylby z podanej sekwencji WLASNE kody, ktore dla
        kodow POCHODZACYCH z sekwencji sa te same (zmierzone:
        `encode(decode(idx)) == idx` daje 1,0000 na trzech poziomach), ale
        dla kodow WYGENEROWANYCH -- czyli dla jedynego przypadku, ktory
        w dyfuzji wystepuje -- byc nimi nie musza. Punkt staly nie jest
        tozsamoscia. Dlatego stan kodowy jest PRZYPINANY hakiem, a nie
        odtwarzany przez przebieg.

        Przypinane jest jedno i drugie: wyjscie kwantyzatora (`z`) ORAZ
        `indices` i `_punkty`. Pominiecie `_punkty` bylo w tym repo
        czterokrotna dziura w pomiarze -- `punkty_poziomow()`,
        `hyp_heads` i galaz `pm_dev` czytaja je bezposrednio, wiec bez
        przypiecia predykcja liczylaby sie przy punktach z poprzedniego
        przebiegu i wynik zalezalby od tego, co model widzial wczesniej.

        Wejsciem przebiegu jest `decode_to_onehot(indices)`. Przy
        `trzon_tylko_z_latentu` zadna droga do predykcji nie czyta
        sekwencji podanej (zmierzone: 0,000000e+00), wiec ten argument
        jest formalnoscia -- ale formalnoscia SPOJNA, a nie dowolnym
        tensorem, wiec gdyby kiedys jakas droga wejscie czytala, czytalaby
        sekwencje odpowiadajaca kodom.

        Zwraca `(dev, hk)` o ksztalcie `[B]`, w skali etykiet treningowych.
        """
        if self.decoder is None:
            raise RuntimeError(
                "dekoder nie istnieje: `recon_weight` musi byc > 0")
        if getattr(self.codes, "recon_bypass", True):
            raise RuntimeError(
                "predykcja z samych kodow wymaga `recon_bypass: false` -- "
                "przy otwartym kanale wejscie dekodera zawiera slot ciagly, "
                "ktorego przy generowaniu nie ma")
        if not getattr(self, "trzon_tylko_z_latentu", False):
            # STRAZ, NIE OSTRZEZENIE. Bez `trzon_tylko_z_latentu` trzon
            # czyta oblozenie i graf liczone z PODANEJ sekwencji, wiec
            # zwrocona liczba nie byla by funkcja kodow -- a metoda
            # nazywa sie `predykcja_z_kodow` i taka umowe oglasza.
            # Zmierzone na `HypFinal2`: przeciek 30 % (tasowanie latentu
            # -0,426), czyli to nie jest ryzyko teoretyczne.
            raise RuntimeError(
                "predykcja_z_kodow wymaga `trzon_tylko_z_latentu: true` -- "
                "inaczej do predykcji dochodzi oblozenie i graf z podanej "
                "sekwencji i wynik NIE jest funkcja kodow")
        z, pkt = self._z_i_punkty_z_indeksow(indices)
        x_hat = self.decode_to_onehot(indices, length=length)

        _idx = [i.clone() for i in indices]

        def _hak(mod, we, wy):
            mod.indices = [i.clone() for i in _idx]
            if pkt is not None:
                mod._punkty = [p.clone() for p in pkt]
            return z

        uchwyt = self.codes.register_forward_hook(_hak)
        try:
            dev, hk = self(x_hat)
        finally:
            uchwyt.remove()
        return dev.reshape(-1), hk.reshape(-1)

    def manifold_parameters(self):
        """Tensory LEZACE na hiperboloidzie, ktore maja gradient.

        Metody dotad NIE BYLO w tej klasie, co mialo dwa skutki. Po pierwsze
        galaz riemannowska w `train_graph.py` byla po cichu nieosiagalna: warunek
        `hasattr(model, "manifold_parameters")` nie przechodzil, wiec bieg
        dostawal plaski AdamW i nikt tego nie widzial w logu. Po drugie
        narzedzia, ktore wolaly ja bez `hasattr`, wywalaly sie na tym modelu, a
        dzialaly na innych.

        Przy `vq_learnable: false` kodebooki sa BUFORAMI i lista jest pusta --
        i to jest uczciwa odpowiedz, nie brak. Wtedy geometria jest aktualizowana
        wylacznie riemannowska EMA (`vq_frechet`), a nie optymalizatorem.
        """
        out = []
        st = getattr(self, "codes", None)
        if st is not None:
            for m in st.modules():
                e = getattr(m, "embed", None)
                if isinstance(e, nn.Parameter) and e.requires_grad:
                    out.append(e)
        # PROTOTYPY GLOWIC. Bez tego wpisu sa optymalizowane plaskim AdamW,
        # czyli chodza po PROSTYCH w ambiencie zamiast po geodezyjnych -- a to
        # jest dokladnie ta roznica, dla ktorej caly modul istnieje. Defekt byl
        # cichy: smoke pokazywal "rozmaitosc 0 tensorow" przy wlaczonych
        # glowicach i nic sie nie wywalalo.
        gh = getattr(self, "hyp_heads", None)
        if gh is not None:
            out.extend(p for p in gh.manifold_parameters() if p.requires_grad)
        # PROTOTYPY DEKODERA POZIOMOW -- WSZYSTKIE TRZY LISTY.
        #
        # To jest awaria ZAPISANA w `hyp_compact_net.py:628-650` i naprawiona
        # tam raz: dodajac blizniacze glowice rozszerzono `protos`, a
        # `protos_hk` NIE -- wiec Hk bylo optymalizowane EUKLIDESOWO z
        # weight_decay 0,08, a Dev riemannowsko z 0,0. Dwa zadania dostawaly
        # rozna geometrie i to nie z zamyslu. `gp_twin` jest domyslnie TRUE,
        # wiec pominiecie `protos_hk` tutaj odtworzyloby dokladnie ten stan.
        #
        # `hq_proto` z tego samego powodu: zapytania atencji geodezyjnej leza
        # na powlokach. To one dryfuja do origa najmocniej (zmierzone ||sp||
        # 1142x ponizej inicjalizacji), wiec bez kroku geodezyjnego i bez
        # `renormalise_gauge` sa losowo obracane o 69 stopni na krok.
        #
        # `shell()` / `hq_shell()` reprojektuja te tensory w przodzie, wiec
        # BEZ tego wpisu wszystko wyglada poprawnie i nic nie pada -- to jest
        # dokladnie ta klasa cichej awarii, ktora `smoke` pokazywal jako
        # "rozmaitosc 0 tensorow" przy wlaczonym mechanizmie.
        ld = getattr(self, "level_dec", None)
        if ld is not None:
            for _lst in (getattr(ld, "protos", None),
                         getattr(ld, "protos_hk", None),
                         getattr(ld, "hq_proto", None)):
                if _lst is None:
                    continue
                out.extend(p for p in _lst if p.requires_grad)
        return out

    def _wagi_glowicy(self, g):
        """Ile wag wazy JEDNA glowica `HypHeads` (`dev` / `hk` / `realnosc`).

        Liczone przez cztery `nn.ModuleDict`, ktore przy `geom_per_cel: true`
        sa per cel (`proj`, `attn`, `odczyt`, `mlp`). Przy `geom_per_cel:
        false` tylko `mlp` jest per cel -- i wtedy ta liczba jest MNIEJSZA
        z prawdziwego powodu, a nie z przeoczenia: reszta jest wspolna.
        """
        gh = getattr(self, "hyp_heads", None)
        if gh is None:
            return 0
        n = 0
        for pole in ("proj", "attn", "odczyt", "mlp"):
            sub = getattr(gh, pole, None)
            if isinstance(sub, nn.ModuleDict) and g in sub:
                n += sum(p.numel() for p in sub[g].parameters())
        return n

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Dopasowuje `hyp_gate` z checkpointow SPRZED zdjecia `realnosc`.

        ZMIERZONA KONIECZNOSC, nie ostroznosc. `load_state_dict(strict=False)`
        toleruje klucze NADWYZKOWE i BRAKUJACE, ale NIE toleruje ROZJAZDU
        KSZTALTU: przy trzywpisowym `hyp_gate` w checkpoincie i dwuwpisowym
        w modelu rzuca `RuntimeError: size mismatch for hyp_gate`. Wszystkie
        sondy w tym repo wolaja `strict=False`, wiec bez tego zaczepu KAZDY
        checkpoint rodziny `hyp_heads` (w tym `HyppeDeepSTARR_best.pt` z biegu
        pomiarowego) przestalby sie wczytywac -- a to jest cena, ktorej
        zdjecie martwej podgalezi nie jest warte.

        Bierzemy PREFIKS: wpisy 0 i 1 to `dev` i `hk` w kazdej wersji tego
        pliku, a wpis 2 (`realnosc`) jest tym, ktory znika. Nadwyzka jest
        odrzucana, niedobor uzupelniany biezaca inicjalizacja.
        """
        _k = prefix + "hyp_gate"
        _g = getattr(self, "hyp_gate", None)
        if (_g is not None and _k in state_dict
                and torch.is_tensor(state_dict[_k])
                and state_dict[_k].shape != _g.shape
                and state_dict[_k].dim() == 1 and _g.dim() == 1):
            _w = state_dict[_k]
            _nowy = _g.detach().clone()
            _n = min(int(_w.numel()), int(_nowy.numel()))
            _nowy[:_n] = _w[:_n].to(dtype=_nowy.dtype, device=_nowy.device)
            state_dict = dict(state_dict)
            state_dict[_k] = _nowy
            print("[model] hyp_gate z checkpointu ma %d wpisow, model ma %d "
                  "-- wzialem prefiks %d (wpisy dev/hk; trzeci wpis to "
                  "zdjeta glowica `realnosc`)."
                  % (int(_w.numel()), int(_nowy.numel()), _n), flush=True)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def raport_kluczy_bez_efektu(self, kwargs):
        """Klucze CZYTANE przez model, ktore i tak nic nie robia.

        Trzecia i najpodstepniejsza kategoria rozjazdu config-kod. Walidator
        w `registry.build_model` lapie klucze NIEPRZYJETE i NIECZYTANE; ten
        raport lapie te, ktore sa odczytane -- wiec wszystko milczy -- a
        mechanizm i tak nie dziala, bo inna flaga omija galaz.
        Wszystkie cztery przypadki nizej sa ZMIERZONE, nie hipotetyczne.

        Nie rzuca wyjatkiem: rzucenie uniewaznilo by wczytywanie istniejacych
        checkpointow, ktore te klucze maja w swoim configu. `build_model`
        drukuje ten raport, wiec kazdy przyszly log go zawiera i da sie po
        nim grepowac.
        """
        bez = []

        # 1. Cel EMA Frecheta jest za `not self.learnable` (hyperbolic.py:1112),
        #    wiec przy uczonym kodebooku galaz jest nieosiagalna.
        if kwargs.get("vq_frechet") and kwargs.get("vq_learnable"):
            bez.append(("vq_frechet",
                        "galaz EMA jest za `not self.learnable`, a "
                        "vq_learnable=True -- geodezyjny cel EMA nieosiagalny"))

        # 2. Argumenty glowicy, ktora nie powstaje.
        if not kwargs.get("hyp_heads", False):
            for k in ("hyp_n_proto", "hyp_hidden", "hyp_tau_skala",
                      "hyp_gate_init", "attn_codes"):
                if k in kwargs:
                    bez.append((k, "argument HypHeads, a hyp_heads=False -- "
                                   "glowica nie powstaje"))

        # 3. Statystyka squeeze jest podawana tylko `se_stat_bloki` PIERWSZYM
        #    blokom, a domyslna to 0. Zmierzone: `se_stat: mean_sd` bez tej
        #    drugiej flagi daje delte parametrow DOKLADNIE 0.
        if (str(kwargs.get("se_stat", "mean")) != "mean"
                and int(kwargs.get("se_stat_bloki", 0) or 0) <= 0):
            bez.append(("se_stat",
                        "se_stat_bloki=0, wiec statystyka nie dochodzi do "
                        "ZADNEGO bloku -- delta parametrow bedzie 0"))

        # 4. Wiazanie RC w stemie powstaje tylko przy `in_ch == 4`
        #    (`elif rc_stem and in_ch == 4`). Przy `motywy_do_latentu` kanalow
        #    jest 4 + motyw_ch, wiec stem spada na zwykly Conv1d. Skutek
        #    uboczny jest osobno grozny: `dom_motywow` i `brama_motywow` nie
        #    pojawiaja sie w logu ANI RAZ, wiec kanal motywowy do latentu
        #    jest NIEOBSERWOWALNY.
        # WARUNEK NA OBECNOSCI KLUCZA, nie na jego wartosci efektywnej.
        # `kwargs.get("rc_stem", True)` bylo prawda takze wtedy, gdy
        # config tego klucza NIE MA -- a raport ma mowic o tym, co w
        # configu NAPISANO. Skutek byl zaporowy: bramka zadala pustej
        # listy martwych kluczy, wiec config bez `rc_stem` oblewal ja
        # ZAWSZE, i to za brak wpisu.
        if "rc_stem" in kwargs and kwargs.get("rc_stem", True):
            _st = getattr(getattr(self, "codes", None), "stem", None)
            _s0 = _st[0] if (_st is not None and len(_st) > 0) else None
            if _s0 is not None and type(_s0).__name__ == "Conv1d":
                bez.append(("rc_stem",
                            "stem to zwykly Conv1d (in_ch != 4, bo "
                            "motywy_do_latentu dokłada kanaly) -- wiazanie RC "
                            "nie powstaje, a dom_motywow/brama_motywow NIE "
                            "DRUKUJA SIE w logu"))

        # 5. Argumenty dekodera poziomow, ktory nie powstaje. Ta sama rodzina
        #    co punkt 2 -- i ten sam powod, dla ktorego punkt 2 istnieje:
        #    `gp_twin: true` przy `glowice_poziomowe: false` czyta sie w
        #    configu jak wlaczony rozdzial Dev/Hk, a nie powstaje ani jeden
        #    parametr.
        if not kwargs.get("glowice_poziomowe", False):
            for k in ("gp_hidden", "gp_protos", "gp_attn", "gp_ramp",
                      "gp_aux", "gp_tau", "gp_cross", "gp_cross_tau",
                      "gp_twin", "gp_hyp_dim", "gp_zapytania", "gp_hq_dim"):
                if k in kwargs:
                    bez.append((k, "argument LevelResidualDecoder, a "
                                   "glowice_poziomowe=False -- dekoder "
                                   "poziomow nie powstaje"))

        # 6. GLOWICE POZIOMOWE PRZY `latent_grad: none`.
        #    Nie jest to klucz calkowicie bez efektu -- galaz przewiduje i
        #    jej wlasne wagi sie ucza -- ale POLOWA mechanizmu jest odcieta
        #    strukturalnie: `_podsumowania_poziomow()` czyta `_z_last`, czyli
        #    tensor po `latent_grad`, a przy 'none' jest on w calosci
        #    `detach()`owany (zmierzone: ||d Dev/d enkoder|| = 0,000000e+00 na
        #    16 tensorach / 66 912 wagach). Wtedy predykcja per poziom NIE
        #    KSZTALTUJE latentu i teza "predykcja idzie z latentu" jest
        #    zredukowana do "predykcja czyta latent".
        #    Raportujemy, a nie rzucamy: to jest legalna ablacja (ile daje sam
        #    ODCZYT), tylko musi byc widoczna w logu, a nie domyslana.
        if (kwargs.get("glowice_poziomowe", False)
                and str(kwargs.get("latent_grad", "none")) == "none"):
            bez.append(("glowice_poziomowe",
                        "latent_grad: none -- podsumowania poziomow sa "
                        "odciete, wiec gleboki nadzor nie dochodzi do "
                        "enkodera; galaz tylko CZYTA latent, nie kształtuje go"))

        # 7. KANAL MOTYWOWY DO LATENTU PRZY `kontekst_do_latentu: true`.
        #    `_codes_input` ma dwie rozlaczne galezie i ta z `motyw_proj`
        #    siedzi ZA `if not self.kontekst_do_latentu`. Przy wlaczonym
        #    kontekscie enkoder dostaje pelne `_h_occ`, wiec waski rzut nie
        #    jest wolany ANI RAZU. Warstwa juz nie powstaje (patrz
        #    `self.motyw_proj`), ale klucze zostaja w configu i czytaja sie
        #    jak wlaczony mechanizm.
        if (kwargs.get("motywy_do_latentu", False)
                and kwargs.get("kontekst_do_latentu", False)):
            for k in ("motywy_do_latentu", "motyw_ch"):
                if k in kwargs:
                    bez.append((k, "kontekst_do_latentu przechwytuje obie "
                                   "drogi wejscia enkodera -- `motyw_proj` "
                                   "nie jest wolane i NIE POWSTAJE"))

        # 8. WPISY `codes` PONAD POZIOM 0 PRZY DRABINIE Z FSQ.
        #    Poziomy styczne kwantyzuje `KwantyzatorFSQ` (alfabet z
        #    `fsq_bity`), wiec `codes[:-1]` nie tworzy zadnego kodebooka.
        #    Do 2026-09-10 tworzylo `HyperbolicVQ`, ktore bylo natychmiast
        #    usuwane -- i to PRZESUWALO strumien RNG. Dzis nie powstaja,
        #    ale wpisy nadal sugeruja rozmiary alfabetow, ktorych nie ma.
        if (kwargs.get("drabina_stozkowa", False)
                and kwargs.get("fsq_styczny", False)
                and len(tuple(kwargs.get("codes", ()) or ())) > 1):
            bez.append(("codes",
                        "drabina + fsq_styczny: tylko OSTATNI wpis (kodebook "
                        "L0 na rozmaitosci) cokolwiek buduje; alfabety "
                        "poziomow stycznych sa w `fsq_bity`"))
        return bez

    def euclidean_parameters(self):
        """Wszystko poza rozmaitoscia. Wymagane przez galaz riemannowska."""
        skip = {id(p) for p in self.manifold_parameters()}
        return [p for p in self.parameters()
                if p.requires_grad and id(p) not in skip]

    @property
    def vocab(self):
        return self._vocab

    def set_vocabulary(self, vocabulary):
        from .motif_occupancy import ContinuousPWMScanner
        self._vocab = vocabulary
        self.scanner = ContinuousPWMScanner(vocabulary).to(next(self.parameters()).device)
        return self

    def memory_estimate(self, batch_size):
        return batch_size * len(self._vocab) * 249 * 4 / 1e6

    def _graph(self, occ):
        length = occ.shape[-1]
        peak = occ.amax(dim=-1)
        pos = occ.argmax(dim=-1).to(occ.dtype) / max(length - 1, 1)
        h = self.node_encode(torch.stack([peak, occ.mean(dim=-1), pos], dim=-1))
        h = h + self.node_identity.unsqueeze(0)
        present = peak > peak.mean(dim=1, keepdim=True)
        # CECHY KRAWEDZI -- wszystkie RC-NIEZMIENNICZE.
        #
        # `pos` wezla przechodzi pod odwrotnym komplementem w `1 - pos`, wiec
        # SAM jest niesymetryczny nicia. Ponizsze cztery sa niezmiennicze:
        #   |pos_i - pos_j|      odleglosc      -> |(1-p_i) - (1-p_j)| = to samo
        #   peak_i * peak_j      wspolobecnosc  -> bez pozycji
        #   |peak_i - peak_j|    kontrast       -> bez pozycji
        #   1 - |pos_i + pos_j - 1|  symetria wzgledem srodka -> niezmiennicza
        # Ostatnia niesie polozenie pary wzgledem SRODKA fragmentu, ktorego
        # sama odleglosc nie ma, a ktore ma sens dla rdzenia promotora.
        _kr = None
        if getattr(self, "edge_dim", 0):
            _d = (pos.unsqueeze(2) - pos.unsqueeze(1)).abs()
            _pk = peak.unsqueeze(2) * peak.unsqueeze(1)
            _kt = (peak.unsqueeze(2) - peak.unsqueeze(1)).abs()
            _sr = 1.0 - (pos.unsqueeze(2) + pos.unsqueeze(1) - 1.0).abs()
            _kr = torch.stack([_d, _pk, _kt, _sr], dim=-1)
        for layer in self.gat:
            h = layer(h, mask=present, edge_feats=_kr)
        if self.hipergraf is not None:
            # PO grafie parowym, na jego wyjsciu. Graf parowy koduje
            # `order`/`overlap`/`adjacent`, czyli odleglosc -- zmierzona
            # jako nienoszaca; hipergraf dodaje to, co niesie:
            # wspolwystepowanie zbiorow.
            h = self.hipergraf(h, occ)
        # WEZLY zachowane dla atencji podprzestrzen<->motyw; `g` to nadal
        # srednia po wezlach, wiec stara droga jest bit w bit ta sama.
        self._wezly_mot = h
        return self.graph_norm(h).mean(dim=1)

    def _code_features(self, z, length):
        """How the code stack enters the prediction trunk.

        A hook, so a subclass can change the *route* the codes take without
        touching anything else in ``embed`` — the rest of this model measured
        well and should stay byte-identical when it is being used as a baseline.
        The default is the behaviour that produced the record.

        ``z.detach()``: reconstruction shapes the codes, prediction only reads
        them. Without it the activity loss pulls the codebook away from
        decodability — the interference that held joint reconstruction to 69.6%
        against 99.84% once the paths were separated.
        """
        # Trzon widzi slot tylko przy `trunk_bypass`; inaczej przycinamy.
        _zt = self.codes.widok_trzonu(z)
        tryb = getattr(self, "latent_grad", "none")
        if tryb == "none":
            # WARTOSC DOMYSLNA -- bit w bit zachowanie rekordzisty.
            _zt = _zt.detach()
        elif tryb == "codes":
            # Gradient aktywnosci wchodzi do enkodera WYLACZNIE przez
            # kwantyzator (straight-through). Slot ciagly zostaje odciety, wiec
            # aktywnosc nie dostaje 64-kanalowej autostrady omijajacej kody.
            nk = self.codes.kanaly_kodow()
            if self.codes.trunk_bypass and _zt.shape[1] > nk:
                _zt = torch.cat([_zt[:, :nk], _zt[:, nk:].detach()], dim=1)
        elif tryb != "all":
            raise ValueError("latent_grad musi byc 'none', 'codes' albo 'all', "
                             "dostano %r" % (tryb,))
        # Sonda `dz_dev`/`dz_hk` potrzebuje uchwytu na wejscie tej drogi.
        self._z_last = _zt
        _cf = self.code_proj(_zt)
        if self.lat_center is not None:
            _cf = self.lat_center(_cf)
            with torch.no_grad():
                # FALSYFIKATOR: iloraz tresci per-sekwencja do wspolnej.
                # Przed poprawka 0,0668. Ponizej ~0,5 znaczy, ze centrowanie
                # nie zadzialalo; przy dzialajacym BatchNormie skladowa wspolna
                # jest z definicji zerowa, wiec liczba ma byc DUZA.
                _sr = _cf.mean(0, keepdim=True)
                _od = (_cf - _sr).norm() / max(_cf.shape[0], 1) ** 0.5
                self._lat_dev = float(_od / _sr.norm().clamp_min(1e-9))
        return self.codes._skaluj(_cf, length)

    def _wejscie_lv(self, feat, pred):
        """Wejscie glowicy wariancji. JEDNO ZRODLO PRAWDY dla obu miejsc.

        Glowica jest wolana DWA razy: raz na czlon kalibracyjny
        (`_lv_kal`, cechy odciete) i raz na wyjscie uzywane przez NLL.
        Rozjazd tych dwoch wejsc znaczylby, ze czlon kalibracyjny
        optymalizuje INNA FUNKCJE niz ta, ktora idzie do straty -- a wagi
        sa wspolne, wiec uczylby glowice pod cel, ktorego nikt nie czyta.
        To jest w tym kodzie zapisane jako blad juz raz popelniony (przy
        pominietym `2*tanh(x/2)` w wersji kalibracyjnej), wiec sklejenie
        siedzi w jednym miejscu i nie da sie ich rozjechac.

        `pred` MUSI przyjsc odciete. Uzasadnienie przy konstrukcji
        `fc_dev_lv`: bez odciecia cel wariancji zyskuje droge gradientu na
        srednia, ktora w tej rodzinie jest zmierzona jako scisnieta
        (nachylenie 0,8414).
        """
        if not self.lv_z_sredniej:
            return feat
        _p = pred.detach().reshape(feat.shape[0], 1) / 2.0
        return torch.cat([feat, _p, _p * _p, _p.abs()], dim=-1)

    def _straz_normy(self):
        """Czy `refiner` da sie policzyc DWA razy na krok. Leniwa, raz.

        Sprawdza TYP modulu, nie flage configu. Roznica jest zmierzona:
        `model.norma_typ` moze byc ustawione i nie dojsc (piaty przypadek
        tego wzorca w tym projekcie -- klucze ladowaly w bloku `training`,
        gdzie YAML je przyjmowal, a kod ignorowal, i trzy biegi szly na
        wartosciach domyslnych).
        """
        if self._straz_normy_zrobiona:
            return
        self._straz_normy_zrobiona = True
        _bn = [n for n, m in self.refiner.named_modules()
               if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
        if _bn:
            raise RuntimeError(
                "occ_z_rekonstrukcji przy BatchNormie w `refiner` (%r). "
                "Oblozenie liczy sie DWA razy na krok, wiec `running_*` "
                "zostalyby przepisane dwukrotnie -- w tym repo 'kazdy "
                "dodatkowy forward psuje EMA' to zapisana awaria (recon "
                "0,79 -> 0,62). Ustaw `model.norma_typ: grupowa`; podmiana "
                "jest po konstrukcji i ma ZERO nowych wag." % (_bn,))

    def rek_trzon_ramp(self):
        """Waga `x_hat` w kanalach sekwencyjnych trzonu. 0 = nic kanoniczna,
        1 = wylacznie dekoder kodow (stan docelowy pod dyfuzje).

        HARMONOGRAM, NIE STAN. To, co realnie poszlo do trzonu, mowi
        `self._occ_w` ustawiane w `forward` -- i to ONO bramkuje `_full.pt`.
        Rozroznienie nie jest kosmetyczne: przy `recon_weight: 0` dekodera
        nie ma, wiec harmonogram moglby wskazywac 1,0 dla mechanizmu,
        ktory nigdy sie nie wykonal. Ta sama rodzina awarii co siedem cicho
        martwych mechanizmow z audytu `sonda_zywotnosci`.
        """
        if not self.trzon_z_rekonstrukcji:
            return 0.0
        n = int(self.rek_trzon_rampup)
        if n <= 0:
            return 1.0
        ep = int(getattr(self, "current_epoch", 10 ** 6))
        st = int(self.rek_trzon_warmup)
        if ep < st:
            return 0.0
        return min(1.0, float(ep - st + 1) / float(n))

    def embed(self, x):
        # KANONIZACJA NIE STOI TUTAJ. Siedzi w `_codes_input`, ktore jest
        # jedynym wspolnym punktem `embed` i `encode_codes`; tutaj tylko
        # PRZEJMUJEMY jej wynik, zeby cel rekonstrukcji byl ta sama nicia,
        # ktora dostal enkoder. Liczenie jej po raz drugi bylo poprawne
        # wynikiem (idempotencja), ale zerowalo `_orient`.
        self._x_kan = None
        # ZERUJEMY ZRODLA DLA `krzyz` NA WEJSCIU. Sa ustawiane w bloku
        # `trzon_z_rekonstrukcji` nizej; bez zerowania wartosc z POPRZEDNIEGO
        # batcha przezylaby przebieg, w ktorym ten blok sie nie wykonal, i
        # `krzyz` dostalby klucze z innych sekwencji. To ta sama rodzina
        # awarii co cztery przypadki nadpisywanych atrybutow w `train_graph`
        # -- z tym, ze tam bylo widac rozmiar, a tu batch by sie zgadzal.
        self._h_occ_rek = None
        self._g_rek = None
        # KOLEJNOSC ODWROCONA: `_codes_input` liczy oblozenie, `h_occ` i graf,
        # wiec wolamy je PIERWSZE i czytamy z `self._ctx`. Poprzednia wersja
        # liczyla `occ` i `g` tutaj, a `_codes_input` liczylo `occ` DRUGI raz --
        # przy `refiner` z BatchNormem to przepisywalo `running_*` dwukrotnie
        # na kazdy krok treningu.
        _we = self._codes_input(x)
        z = self.codes(_we)
        occ = self._ctx["occ"]
        g = self._ctx["g"]
        reg = self.codes.loss
        if self.recon_weight:
            # Dekoder widzi slot tylko przy `recon_bypass`; inaczej przycinamy,
            # zeby rekonstrukcja byla wylacznie z kodow i dyfuzja miala sens.
            _we_dek = (z if self.codes.recon_bypass
                       else self.codes.tylko_kody(z))
            self._maska_raport = None
            if self.maska is not None and self.training:
                # JEDNO wywolanie dekodera na widokach sklejonych po wymiarze
                # batcha. Osobne wywolania to OSOBNE aktualizacje `running_*`
                # w dwoch BatchNormach dekodera -- zmierzone wykonaniem:
                # `num_batches_tracked` rosnie o 1 na kazde wywolanie, wiec
                # cztery wywolania to cztery aktualizacje na krok, i to
                # statystykami czterech ROZNYCH rozkladow. Ta sama rodzina
                # awarii co "kazdy dodatkowy forward psuje EMA kodebooka".
                _widoki, _nazwy_w, _maski_w = self.maska.widoki(_we_dek)
                logits = self.decoder(_widoki)
            else:
                # `eval` (i bieg bez maski) idzie JEDNYM widokiem, czyli
                # bit w bit jak cala dotychczasowa rodzina.
                _nazwy_w, _maski_w = ("pelny",), (None,)
                logits = self.decoder(_we_dek)
            # REKONSTRUKCJA Z SAMYCH KODOW, gdy dekoder widzi tez slot ciagly.
            # Bez tej liczby log nie ma ZADNEGO wskaznika na "kody nie niosa
            # rekonstrukcji": zmierzone na `vqbypass` ep6, `bits_used_frac`
            # 0,9485 (najwyzej w biegu) przy rekonstrukcji z kodow 0,5405.
            # Entropia kodebooka i jego przydatnosc to dwie rozne rzeczy.
            # Tylko w `eval` i pod `no_grad`, wiec trening nie placi.
            self._recon_kody = None
            if self.codes.recon_bypass and not self.training:
                with torch.no_grad():
                    # ZEROWANIE slotu, nie odciecie: przy otwartym kanale
                    # `decoder[0]` oczekuje pelnych 256 kanalow, wiec podanie
                    # 192 rzuca ksztaltem. Zerowanie zachowuje szerokosc i
                    # odbiera dekoderowi dokladnie te informacje, ktorej przy
                    # generowaniu nie ma.
                    _zk = z.clone()
                    _zk[:, self.codes.kanaly_kodow():] = 0.0
                    _lk = self.decoder(_zk)
                    if _lk.shape[-1] != x.shape[-1]:
                        _lk = F.interpolate(_lk, size=x.shape[-1],
                                            mode="linear", align_corners=False)
                    self._recon_kody = float(
                        (_lk.argmax(1) == x.argmax(1)).float().mean())
            if logits.shape[-1] != x.shape[-1]:
                # Interpolacja liniowa, i to jest POPRAWNY wybor -- zmierzone.
                # Podejrzewalem, ze domykanie 248 -> 249 narzuca sufit
                # rekonstrukcji (rzekomo 0,9888 przy zmierzonych 0,9874) i ze
                # rozmywa srodek sekwencji. To bylo bledne: tamten pomiar liczyl
                # przejscie 249 -> 248 -> 249, czyli sam najpierw psul sygnal
                # interpolacja w dol. Przy 248 SWOBODNYCH wyjsciach dekodera,
                # optymalizowanych pod cel, osiagalna dokladnosc to 1,0000 --
                # rowniez w srodku sekwencji. Sufitu NIE MA.
                # Dopelnienie ogona (`F.pad(..., "replicate")`) dawalo 0,9969, bo
                # wymusza ostatnia pozycje rowna przedostatniej. Czyli gorzej.
                logits = F.interpolate(logits, size=x.shape[-1], mode="linear",
                                       align_corners=False)
            # CEL = NIC KANONICZNA. Liczony wobec podanej spadlby do ~0,5
            # na polowie probek i wygladalby jak awaria dekodera, a bylby
            # brakiem jednego bitu orientacji z budzetu 1844.
            _cel_x = self._x_kan if self._x_kan is not None else x
            _cel_i = _cel_x.argmax(1)
            if len(_nazwy_w) == 1:
                self.recon_logits = logits
                _term = self.recon_weight * F.cross_entropy(logits, _cel_i)
            else:
                _cz = logits.chunk(len(_nazwy_w), dim=0)
                # `recon_logits` MUSI byc widokiem PELNYM. Czytaja je trzy
                # miejsca: `evaluate` (metryka `recon` w logu),
                # `reconstruction_accuracy` i `occ_from_recon`. Podanie im
                # tensora sklejonego z widokow bylo by awaria cicha -- metryka
                # policzylaby sie na K-krotnym batchu i wyszla jako srednia
                # po widokach, czyli liczba nieporownywalna z rodzina.
                # Widok pelny jest pierwszy z konstrukcji `MaskaLatentu`.
                self.recon_logits = _cz[0]
                _sk, _rap = None, {}
                for _i, _n in enumerate(_nazwy_w):
                    _w = self.maska.wagi[_n]
                    _m = _maski_w[_i]
                    if _m is None:
                        _ce = F.cross_entropy(_cz[_i], _cel_i)
                        _tr = float((_cz[_i].argmax(1) == _cel_i).float().mean())
                    else:
                        # Maska powstaje na siatce KODOW (124 pozycje), a cel
                        # ma 249 zasad -- `nearest` trzyma ja binarna.
                        _mb = F.interpolate(_m, size=_cel_i.shape[-1],
                                            mode="nearest").squeeze(1)
                        _msum = _mb.sum().clamp_min(1.0)
                        # Strata widoku kolumnowego liczona TYLKO na pozycjach
                        # ZAMASKOWANYCH. Pozycje widoczne obejmuje widok pelny,
                        # a ich dolozenie tutaj rozcienczyloby gradient czescia
                        # zadania, ktora jest trywialna: zmierzone R^2 0,95
                        # odtworzenia kodow z sasiedztwa (poza proba).
                        # Maskowany jest OPERAND, nie wynik -- `torch.where`
                        # nie chroni gradientu i to jest zapisana awaria.
                        _cep = F.cross_entropy(_cz[_i], _cel_i, reduction="none")
                        _ce = (_cep * _mb).sum() / _msum
                        with torch.no_grad():
                            _tr = float(
                                ((_cz[_i].argmax(1) == _cel_i).float() * _mb).sum()
                                / _msum)
                    _sk = _w * _ce if _sk is None else _sk + _w * _ce
                    _rap["rek_" + _n] = round(_tr, 4)
                _rap.update(self.maska.raport())
                self._maska_raport = _rap
                # DRUGI, NIEWYCIERANY EGZEMPLARZ. `band_report` jest wolany
                # PO walidacji, czyli po przebiegu w `eval`, a tam maska nie
                # dziala i `_maska_raport` jest ustawiany na None -- wiec
                # `rek_prefiks_0`, GLOWNA liczba tego biegu, nie docierala do
                # logu ani raz. Ta sama rodzina awarii co siedem cicho
                # martwych mechanizmow z audytu `sonda_zywotnosci`: obserwabla
                # istnieje, jest liczona i zostaje WYTARTA przed odczytem.
                self._maska_raport_tren = dict(_rap)
                # JEDEN czlon, wiec JEDEN gracz Nasha. Wagi widokow sumuja sie
                # do 1 (asercja w `MaskaLatentu.__init__`), wiec skala tego
                # czlonu jest ta sama co bez maski i `nash_min_waga`/`max_waga`
                # nie trzeba przestawiac.
                _term = self.recon_weight * _sk
            # Do OBU, nie przeniesienie: `_reg_recon` jest w train_graph
            # tylko CZYTANE, wiec przeniesienie wylaczyloby trening
            # rekonstrukcji calkowicie. Brak tego atrybutu byl przyczyna,
            # dla ktorej `pcgrad` na calej rodzinie VQ byl cicho martwy.
            self._reg_recon = _term
            reg = reg + _term
        self.reg_loss = reg

        h_occ = self._ctx["h_occ"]
        z_up = self._code_features(z, h_occ.shape[-1])
        # CEL PIERWSZY: przy `trzon_tylko_z_latentu` aktywnosc jest funkcja
        # KODOW, bo oblozenie nie ma juz drogi omijajacej kwantyzator.
        _we_tr = (z_up if self.trzon_tylko_z_latentu
                  else torch.cat([h_occ, z_up], dim=1))
        # ---- ZDEKODOWANA SEKWENCJA JAKO KANALY TRZONU ----
        # Pelne uzasadnienie przy konstrukcji `_we_trzonu` w `__init__`.
        # Tutaj tylko trzy rzeczy warte pilnowania:
        #  1. `recon_logits` to widok PELNY (`_cz[0]`), juz zinterpolowany
        #     do `x.shape[-1]`. Widoki maskowane maja K-krotny batch, wiec
        #     podanie ich tutaj rozjechaloby wymiar batcha wobec `z_up`.
        #  2. Mieszamy z `_x_kan`, nie z `x`: dekoder przewiduje nic
        #     KANONICZNA, a mieszanka dwoch ukladow wspolrzednych zlamalaby
        #     niezmienniczosc RC w polowie probek.
        #  3. Gradientu NIE odcinamy. Odciecie zostawiloby wartosc dla
        #     trzonu, ale zabraloby jedyna nowa rzecz w tym biegu --
        #     sprzezenie predykcja -> dekoder.
        # `_occ_w` ustawiamy TYLKO gdy mechanizm dziala. Bezwarunkowe
        # zerowanie zmienialoby raport `_full.pt` w czterech starych
        # configach, ktore tego mechanizmu nie maja -- `train_graph` czyta
        # `getattr(model, "_occ_w", 1.0)`, wiec brak atrybutu znaczy tam
        # "mechanizmu nie ma, warunek nie dotyczy", a jawne 0,0 znaczyloby
        # "mechanizm jest i nie zadzialal". To dwie rozne rzeczy.
        if self.trzon_z_rekonstrukcji:
            _lg = getattr(self, "recon_logits", None)
            if _lg is None:
                raise RuntimeError(
                    "trzon_z_rekonstrukcji: brak `recon_logits` w `forward`. "
                    "Dekoder powstaje tylko przy recon_weight > 0, a straz "
                    "w `__init__` tego pilnuje -- wiec to znaczy, ze `embed` "
                    "nie zostalo wywolane przed trzonem.")
            _w = float(self.rek_trzon_ramp())
            self._occ_w = _w
            _xh = (_lg / self.recon_tau).softmax(dim=1)
            _kan = self._x_kan if self._x_kan is not None else x
            _zr = _xh if _w >= 1.0 else (1.0 - _w) * _kan[:, :_xh.shape[1]] + _w * _xh
            if _zr.shape[-1] != _we_tr.shape[-1]:
                _zr = F.interpolate(_zr, size=_we_tr.shape[-1], mode="linear",
                                    align_corners=False)
            with torch.no_grad():
                _p = _xh.clamp_min(1e-9)
                self._rt_raport = {
                    "rt_w": round(_w, 4),
                    "rt_ent": round(float(-(_p * _p.log2()).sum(1).mean()), 4),
                    "rt_max": round(float(_xh.max(1).values.mean()), 4)}
            # ---- OBLOZENIE Z TEGO SAMEGO ZRODLA ----
            # `_zr` jest zmieszanym tensorem sekwencji (nic kanoniczna ->
            # `x_hat` wraz z rampa). Skan liczymy z NIEGO, nie z `_xh`
            # osobno: jedna rampa steruje wtedy OBIEMA drogami i `_occ_w`
            # opisuje stan calego mechanizmu, a nie jego polowy. Gdyby
            # kanaly sekwencyjne i oblozenie szly z roznych zrodel, log
            # pokazywalby jedna liczbe dla dwoch rzeczy.
            if self.occ_z_rekonstrukcji:
                self._straz_normy()
                _o2 = self._gate_occupancy(self.scanner(_zr))
                _h2 = self.refiner(_o2)
                if self.learned is not None:
                    _h2 = torch.cat([_h2, torch.sigmoid(self.learned(_zr))],
                                    dim=1)
                _h2 = self.occ_proj(_h2)
                # ---- WEKTOR GRAFU Z `x_hat`, DLA `krzyz` ----
                #
                # PO CO: `KontekstKrzyzowy` (`krzyz_bloki`) to zapytania
                # z mapy trzonu, a klucze i wartosci z OBLOZENIA plus token
                # globalny z wektora grafu. W `HypFinal2` czyta on
                # `_ctx["h_occ"]` i `_ctx["g"]`, czyli oblozenie liczone
                # z PODANEJ sekwencji -- i to jest CALY jego przeciek:
                # zmierzone `sonda_tylko_kody2.py` = 7,348e-02, czyli
                # **30,0 % kontroli pozytywnej**, przy 0,000000e+00 dla HD8.
                #
                # A `HypFinal2` osiaga 0,75695 w epoce 21 i WCIAZ ROSNIE,
                # czyli ten modul jest dokladnie tym, czego naszej rodzinie
                # brakuje do poziomu rekordu. Liczony z `x_hat` jest funkcja
                # WYLACZNIE kodow, wiec umowa dyfuzji zostaje -- ten sam
                # ruch, ktory dla trzonu dal +0,0233 w epoce 5 (HD5 -> HD8),
                # najwiekszy dodatni krok w calej rodzinie.
                #
                # DRUGIE WYWOLANIE `_graph` JEST BEZPIECZNE i to jest
                # SPRAWDZONE, nie zalozone: cala droga grafu uzywa wylacznie
                # LayerNormow (`graph_norm`, `gat.*.norm`, `archetype.3`),
                # zero statystyk biezacych. Gdyby byl tam BatchNorm, drugie
                # wywolanie przepisalo by `running_*` -- ta sama pulapka,
                # ktora w `refiner` wymusila `norma_typ: grupowa`.
                self._g_rek = self._graph(_o2)
                if self.occ_dropout is not None:
                    # PO `occ_proj`, nie przed: zerujemy kanaly, ktore
                    # realnie wchodza do trzonu, a nie surowe profile.
                    # `Dropout1d` jest bezczynny w `eval`, wiec umowa
                    # dyfuzji i `sonda_tylko_kody` sa nietkniete.
                    _h2 = self.occ_dropout(_h2)
                if _h2.shape[-1] != _we_tr.shape[-1]:
                    _h2 = F.interpolate(_h2, size=_we_tr.shape[-1],
                                        mode="linear", align_corners=False)
                with torch.no_grad():
                    # FALSYFIKATOR: czy oblozenie z `x_hat` w ogole rozni sie
                    # od tego z prawdziwej sekwencji. Zero znaczyloby, ze
                    # dekoder jest tozsamoscia na skanie PWM i mechanizm nie
                    # ma czego wnosic; wartosc rzedu normy `h_occ` znaczy, ze
                    # to DWA rozne tensory i porownanie ma sens.
                    _ho = self._ctx["h_occ"]
                    self._rt_raport["rt_occ_d"] = round(
                        float((_h2 - _ho).norm() / _ho.norm().clamp_min(1e-9)), 4)
                # ZAPAMIETANE DLA `krzyz`. Sklejenie do `_we_tr` idzie
                # osobno; `krzyz` potrzebuje `_h2` w pelnej rozdzielczosci
                # jako KLUCZE/WARTOSCI, a nie jako kanaly wejscia trzonu.
                self._h_occ_rek = _h2
                _we_tr = torch.cat([_h2, _we_tr], dim=1)
            _we_tr = torch.cat([_zr, _we_tr], dim=1)
            # ===== PRZEKIEROWANIE WEKTORA GRAFU NA `x_hat` =====
            #
            # TO JEST JEDNA LINIA, KTORA ZAMYKA PRAWDZIWY PRZECIEK -- i tego
            # dowiedzialem sie POMIAREM, po tym jak dwa razy celowalem w zle
            # miejsce.
            #
            # `g` jest zmienna LOKALNA czytana przez TRZY drogi do predykcji:
            #   * FiLM na kazdym bloku trzonu (`self.film[str(i)](g)`),
            #   * `krzyz` jako token globalny,
            #   * glowica (`archetype(cat([..., g]))` przy `glowica_graf`).
            # Wszystkie trzy siedza ZA tym miejscem, wiec jedno podstawienie
            # przekierowuje komplet.
            #
            # DLACZEGO TO WLASNIE TU JEST PRZECIEK, a nie w `krzyz`:
            # `film_wl` i `glowica_graf` maja domyslne **True**, a `HypFinal2`
            # ich NIE USTAWIA -- buduje wiec FiLM (123 840 param.) na kazdym
            # bloku i podaje `g` wprost do glowicy. Zmierzone na jego logu:
            # `krzyz_brama` i `para_brama` sa ~1e-4 przez WSZYSTKIE 21 epok,
            # czyli te dwa moduly sa tam MARTWE, a bieg mimo to osiaga
            # 0,75695. Wniosek: jego 30 % przecieku (zmierzone
            # `sonda_tylko_kody2.py`) idzie FiLM-em i glowica grafu, nie
            # atencja krzyzowa. Moja poprzednia atrybucja byla falszywa.
            #
            # `_g_rek = self._graph(_o2)` jest policzone z `x_hat`, wiec jest
            # funkcja WYLACZNIE kodow -- ten sam ruch, ktory dla trzonu dal
            # +0,0233 (HD5 -> HD8), najwiekszy dodatni krok w rodzinie.
            if getattr(self, "_g_rek", None) is not None:
                g = self._g_rek
        if self.trojkat is not None:
            # Zapytania trojkatne ida z TEGO SAMEGO wejscia, ktore dostaje
            # trzon -- czyli z oblozenia i kodow razem, w pelnej
            # rozdzielczosci nukleotydowej (`occ_center` trzyma 249 pozycji).
            # Wieza zwraca TRZY rzeczy: odczyt par, predykcje pomocnicza
            # oraz STRUMIEN SEKWENCJI PO ATENCJI Z BIASEM PAR. Ten trzeci
            # PODMIENIA wejscie trzonu -- struktura ma przestawiac czytanie
            # sekwencji, nie tylko doklejac sie jako kanaly. Bez tego tensor
            # par jest slepa uliczka (kierunek 1D -> 2D bez powrotu).
            _tc, self._troj_aux, _we_tr = self.trojkat(_we_tr)
            _we_tr = torch.cat([_we_tr, _tc], dim=1)
        h = self.stem(_we_tr)
        # Hook, like `_trunk_input`: a subclass may modulate the trunk here.
        # After the stem, so a modulator sees `trunk_dim` channels rather than
        # the raw concatenation, and before the blocks, so it conditions all of
        # them rather than decorating the last.
        h = self._trunk_modulate(h)

        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if self.film is not None:
                gamma, beta = self.film[str(i)](g).chunk(2, dim=-1)
                h = h * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
            if self.krzyz is not None and str(i) in self.krzyz:
                # REZYDUALNIE. Przy bramie 0,05 i dopasowaniu RMS start jest
                # bliski tozsamosci, wiec bieg nie moze zaszkodzic
                # strukturalnie -- a przy zerowym wkladzie wraca do bazy sam.
                # ZRODLO KLUCZY DLA `krzyz` -- funkcja KODOW, gdy tylko
                # `occ_z_rekonstrukcji` jest wlaczone. Pelne uzasadnienie
                # przy liczeniu `_g_rek` wyzej. Bez tego przekierowania ten
                # modul jest jedyna droga w calym modelu, ktora czyta
                # `_ctx`, czyli oblozenie z PODANEJ sekwencji -- zmierzone
                # na `HypFinal2` jako 30,0 % kontroli pozytywnej.
                #
                # `getattr` z domyslna, a nie bezwarunkowe czytanie: przy
                # `occ_z_rekonstrukcji: false` te atrybuty nie powstaja,
                # a cztery stare configi z `krzyz_bloki` musza budowac sie
                # bit w bit jak dotad.
                _kv = getattr(self, "_h_occ_rek", None)
                _gt = getattr(self, "_g_rek", None)
                h = h + self.krzyz[str(i)](
                    h,
                    h_occ if _kv is None else _kv,
                    g if _gt is None else _gt)

        def _segmenty(hh, wagi=None):
            """Srednie po `pool_skale` rownych segmentach; `wagi` opcjonalnie
            wazy pozycje WEWNATRZ segmentu (pula per cel)."""
            S = self.pool_skale
            if S == 1:
                if wagi is None:
                    return [hh.mean(-1)]
                return [(hh * wagi).sum(-1)]
            L = hh.shape[-1]
            gr = [round(i * L / S) for i in range(S + 1)]
            out = []
            for i in range(S):
                a_, b_ = gr[i], max(gr[i + 1], gr[i] + 1)
                kawalek = hh[..., a_:b_]
                if wagi is None:
                    out.append(kawalek.mean(-1))
                else:
                    w_ = wagi[..., a_:b_]
                    w_ = w_ / w_.sum(-1, keepdim=True).clamp_min(1e-12)
                    out.append((kawalek * w_).sum(-1))
            return out

        if self.pool_q is not None:
            self._feat_per = {}
            for _g in ("dev", "hk"):
                a = torch.softmax(self.pool_q[_g](h), dim=-1)          # [B,1,L]
                self._feat_per[_g] = self.archetype(
                    torch.cat(_segmenty(h, a) + [h.amax(-1)]
                              + ([g] if self.glowica_graf else []), dim=-1))
            return self._feat_per["dev"]
        self._feat_per = None
        # WEKTOR GRAFU tylko przy `glowica_graf`. To druga z dwoch drog
        # omijajacych kwantyzator; przy `false` predykcja idzie wylacznie
        # przez latent i wieze trojkatna.
        return self.archetype(torch.cat(
            _segmenty(h) + [h.amax(-1)]
            + ([g] if self.glowica_graf else []), dim=-1))

    def _trunk_modulate(self, h):
        """Pole salientnosci: modulacja trzonu ZALEZNA OD POZYCJI.

        Bez `pole_salientnosci` -- tozsamosc, czyli stare zachowanie bit w bit.

        Z polem: kazdy poziom drabiny daje wlasne pole, ktore jest nadprobkowane
        do dlugosci mapy trzonu i sumowane z wolna waga per poziom. Poziom
        NAJDROBNIEJSZY (L2, 124 komorki, ~2 bp) moze wiec modulowac lokalnie,
        a najgrubszy (L0, 15 komorek, 16,6 bp) globalnie -- i model sam wybiera
        proporcje. To jest ta sama zasada, ktora w gramatyce przeniosla mase
        uwagi z [1,00; 0,00; 0,00] na rozlozona: skale per poziom jako WOLNY
        parametr, a nie ustalona z gory.

        `nearest` przy nadprobkowaniu, nie `linear`: komorka kodu jest
        JEDNOSTKA DYSKRETNA, a interpolacja liniowa rozmazalaby granice miedzy
        sasiednimi kodami, czyli dokladnie to, czego drabina ma nie robic.
        """
        if self.pole_U is None:
            return h
        pk = self.punkty_poziomow()
        pole = None
        for i, p in enumerate(pk):
            # `p`: [B, L_i, D+1] na hiperboloidzie -> [B, L_i, M]
            si = self.pole_mlp(p.float())
            si = self.codes._skaluj(si.transpose(1, 2), h.shape[-1])
            w = torch.exp(self.pole_skala[i]).to(si.dtype)
            pole = si * w if pole is None else pole + si * w
        # [B, M, L] -> [B, C, L]; `pole_U` ma zeros_, wiec start jest tozsamy
        mod = self.pole_U(pole.transpose(1, 2)).transpose(1, 2)
        with torch.no_grad():
            # FALSYFIKATOR: amplituda modulacji wzgledem samej mapy. Zero po
            # kilku epokach znaczy, ze model nie chce modulacji pozycyjnej --
            # i to jest wynik, bo wyklucza hipoteze o interakcji motyw x pozycja.
            self._pole_amp = float(mod.abs().mean() / h.abs().mean().clamp_min(1e-9))
            self._pole_skala = [float(torch.exp(v)) for v in self.pole_skala]
        return h * (1.0 + mod)

    def _gate_occupancy(self, occ):
        """Identity. Overridden where motifs can be switched off.

        Placed at the SOURCE so one decision serves every consumer -- the
        prediction trunk, the graph, and (where a subclass projects motifs into
        the latent) the encoder as well. A motif rejected in one place and alive
        in another would be no rejection at all.
        """
        return occ

    def _szerokosc_wejscia_enkodera(self):
        """JEDNO ZRODLO PRAWDY o szerokosci wejscia enkodera.

        Musi zgadzac sie z `_codes_input` co do kanalu. Rozjazd tej pary jest
        awaria, ktora wychodzi az na pierwszym batchu -- i wyszla: `StemMotyw
        zbudowany na 4+32 kanalow, dostal 276`.
        """
        if self.kontekst_do_latentu:
            return 4 + self._occ_ch_lat + (self._graf_ch if self._graf_ch else 0)
        return 4 + (self._motyw_ch if self.motywy_do_latentu else 0)

    def _encoder_in_channels(self):
        """How wide the encoder's first convolution must be. Four: the bases.

        Paired with `_codes_input`, which produces the tensor of that width. The
        two must agree, and keeping them adjacent is the cheapest way to make a
        mismatch obvious.
        """
        return self._szerokosc_wejscia_enkodera()

    def mapa_wczesna(self, x):
        """Sam prefiks enkodera do wczesnej mapy -- BEZ kwantyzatora.

        Uzywane przez JEPA na drugim przebiegu. Wolanie calego `forward`
        nadpisaloby `reg_loss`, `_reg_recon` i `codes.indices`, a to jest
        w tym repo zapisana pulapka: przebieg pomocniczy oddawal Nashowi
        rekonstrukcje INNYCH sekwencji jako gracza.
        """
        st = self.codes
        h = st.stem(self._codes_input(x))
        if st.bloki_przed is not None:
            for _b in st.bloki_przed:
                h = _b(h)
        if getattr(st, "wieza", None) is not None:
            _tc, _, _h1 = st.wieza(h)
            h = _h1 + st.wieza_rzut(_tc)
        return h

    def strata_jepa(self, x, dl=40, n=None):
        """MSE miedzy reprezentacja okna z KONTEKSTU a jej wartoscia prawdziwa.

        Cel jest odciety; predykcja idzie z sekwencji ZAMASKOWANEJ.
        """
        # SPROSTOWANIE ZASIEGU (2026-09-09), PRZED starym opisem, bo stary
        # obiecuje wiecej, niz ten czlon robi. ZMIERZONE: gradient z JEPA
        # dosiega **4 z 34** tensorow enkodera -- wylacznie `codes.stem`
        # (max|grad| 4,6e-03 / 5,8e-03 / 8,9e-04 / 5,5e-03). Cale `downs`,
        # cala drabina i `brama_kontekst` dostaja gradient DOKLADNIE ZERO;
        # glowna strata dosiega 34 z 34. Obie strony MSE licza sie z tej samej
        # WCZESNEJ MAPY (patrz blok "SKALE LICZONE NA TYM SAMYM OBIEKCIE"),
        # wiec "hierarchia" jest po ZIARNISTOSCI PYTANIA (40/20/10/1 komorek),
        # a NIE po poziomach drabiny. Ten czlon jest REGULARYZATOREM STEMU,
        # nie nauczycielem hierarchii, i tak nalezy go raportowac.
        #
        # STARY OPIS, zachowany bo tlumaczy INTENCJE, ale nieaktualny co do
        # implementacji -- blok nizej sam odrzuca ta wersje jako zla:
        # JEPA HIERARCHICZNA: cel liczony na KAZDEJ skali drabiny, nie
        # tylko na wczesnej mapie. Powod jest strukturalny, nie estetyczny:
        # zamaskowane okno 40 bp da sie odtworzyc z kontekstu tym latwiej,
        # im GRUBSZA skala -- na L0 (16,6 bp na komorke) sasiedztwo niesie
        # duzo, na L2 (2 bp) prawie nic. Jedna skala miesza te dwa rezimy
        # w jedna liczbe i nie da sie odczytac, ktory poziom sie uczy.
        #
        # Kazdy poziom dostaje WLASNA wage `1/sqrt(liczba pozycji)`, zeby
        # poziom najdrobniejszy (124 komorki) nie zdominowal sumy samym
        # rozmiarem -- inaczej "hierarchiczna" byla by nazwa dla straty
        # zdominowanej przez jedna skale.
        tgt = getattr(self.codes, "_mapa_wczesna", None)
        _mp = getattr(self.codes, "_mapy_poziomow", None)
        if tgt is None:
            return None
        B, _, L = x.shape
        n = int(n or B)
        n = max(1, min(n, B))
        _g = torch.Generator(device="cpu").manual_seed(
            int(getattr(self, "current_epoch", 0)) * 1000003 + B)
        a0 = int(torch.randint(0, max(L - dl, 1), (1,), generator=_g))
        xm = x[:n].clone()
        xm[:, :, a0:a0 + dl] = 0.25            # maska: rozklad rownomierny
        from .negatives import BezAktualizacjiStatystyk
        with BezAktualizacjiStatystyk(self):
            pm = self.mapa_wczesna(xm)
        # SKALE LICZONE NA TYM SAMYM OBIEKCIE, w roznej ZIARNISTOSCI.
        #
        # POPRZEDNIA WERSJA BYLA ZLA i tu jest naprawiona. Porownywala
        # `adaptive_avg_pool1d(wczesna mapa)` z MAPAMI POZIOMOW, a te
        # powstaja przez UCZONE sploty ze stridem, agregacje (`max`/`std`/
        # `zakres`), bloki poziomu i `mot_l0`. To sa dwie ROZNE funkcje
        # wejscia, wiec MSE mierzylo systematyczna roznice miedzy
        # "usredniona mapa wczesna" a "mapa po stridzie", a nie to, ile
        # o zamaskowanym oknie wiadomo z kontekstu. Cel byl przy tym
        # odciety, wiec nacisk szedl na mape wczesna, zeby po usrednieniu
        # upodobnila sie do wyjscia `downs` -- objaw, nie zamiar.
        #
        # Teraz obie strony to TA SAMA wielkosc: wycinek wczesnej mapy
        # w oknie maski, usredniony do K komorek. K = 40 to rozdzielczosc
        # nukleotydowa, K = 1 to jedno zdanie o calym oknie. Hierarchia
        # jest wiec po ZIARNISTOSCI PYTANIA, a nie po niezgodnych mapach.
        #
        # Waga `1/sqrt(K)`: bez niej skala najdrobniejsza (40 komorek)
        # zdominowalaby sume samym rozmiarem i "hierarchiczna" byloby
        # nazwa dla straty jednoskalowej.
        _pw = pm[:, :, a0:a0 + dl]
        _tw = tgt[:n, :, a0:a0 + dl].detach()
        _skl = []
        for _K in (dl, max(dl // 2, 1), max(dl // 4, 1), 1):
            if _skl and _K == _skl[-1][0].shape[-1]:
                continue
            _pp = _pw if _K == dl else F.adaptive_avg_pool1d(_pw, _K)
            _tt = _tw if _K == dl else F.adaptive_avg_pool1d(_tw, _K)
            _skl.append((_pp, _tt, 1.0 / float(_K) ** 0.5))
        _p, _t = _skl[0][0], _skl[0][1]
        with torch.no_grad():
            # FALSYFIKATOR: wariancja predykcji MIEDZY sekwencjami wobec
            # wariancji celu. Cel rozwiazany STALA daje niska strate przy
            # zerowej wariancji -- zdarzylo sie juz w tym repo.
            _vp = _p.mean(dim=(1, 2)).var()
            _vt = _t.mean(dim=(1, 2)).var().clamp_min(1e-12)
            self._jepa_war = round(float(_vp / _vt), 4)
        _sum, _w = None, 0.0
        for _pp, _tt, _ww in _skl:
            _l = _ww * F.mse_loss(_pp, _tt)
            _sum = _l if _sum is None else _sum + _l
            _w += _ww
        with torch.no_grad():
            self._jepa_skal = len(_skl)
        return _sum / max(_w, 1e-6)

    def _codes_input(self, x):
        """What the ENCODER sees. Identity: the one-hot sequence.

        A hook, so a subclass can project shared motif occupancy or other
        per-position features into the latent without every `_parent_pooled`
        reimplementation having to be edited -- and there are three of them,
        which is how the trunk-modulation hook first ended up as dead code in
        the one class that mattered.
        """
        # KONTEKST LICZONY BEZWARUNKOWO. Trzon potrzebuje `h_occ` i `g`
        # w KAZDEJ konfiguracji -- takze wtedy, gdy latent ich nie dostaje.
        # Poprzednia wersja wychodzila tu wczesniej przy
        # `motywy_do_latentu: false` i `embed` walilo `AttributeError: _ctx`.
        # Od flag zalezy WYLACZNIE to, co wchodzi do ENKODERA.
        #
        # Uwaga informacyjna, ktora trzeba miec zapisana: H(oblozenie | x) = 0,
        # wiec kanal motywowy NIE DODAJE informacji. Moze zmienic wylacznie
        # PRZYDZIAL BITOW -- to jest rate-distortion, nie entropia.
        # Falsyfikatorem jest `bits_used_frac` i entropia per poziom, nie sama
        # rekonstrukcja.
        if self.kanon_rc:
            # TU, a nie w `embed`. `encode_codes` wola `self.codes(
            # self._codes_input(x))` i NIE przechodzi przez `embed` -- przy
            # kanonizacji tylko w `embed` niezmienniczosc byla ZEROWA:
            # zmierzone `encode(x) == encode(rc(x))` = 0,374/0,276/0,168,
            # czyli DOKLADNIE tyle, co zgodnosc dwoch ROZNYCH sekwencji
            # (0,354/0,275/0,168), a wiec poziom przypadku. `_codes_input`
            # jest jedynym wspolnym punktem obu drog. Ta sama usterka co
            # `decode_codes` z zaszytym `nearest`: dwa wejscia, jedno
            # zalatane. Kanonizacja jest IDEMPOTENTNA, wiec powtorne
            # zastosowanie w `embed` jest no-opem.
            # ============ ORIENTACJA NARZUCONA: PROTOKOL DLA WARIANTOW ============
            # `wybierz_orientacje` porownuje sekwencje z jej odwrotnym
            # komplementem LEKSYKOGRAFICZNIE i rozstrzyga na PIERWSZEJ roznicy.
            # ZMIERZONE na 512 sekwencjach walidacji: decyzja pada w 69,9 %
            # przypadkow na pozycji 0, w 97,5 % na pozycji <= 2, maksymalnie
            # na pozycji 5. Skutek dla zadania, dla ktorego ten projekt
            # istnieje -- przewidywania efektu wariantu -- jest powazny:
            #
            #   pozycja SNP |  0     |  1     |  2     | srodek |  247   |  248
            #   frakcja     | 0,4193 | 0,1139 | 0,0365 | 0,0000 | 0,1224 | 0,4030
            #   po losowej pozycji: 0,0126
            #
            # Czyli 1,26 % wszystkich podstawien PRZEWRACA nic kanoniczna,
            # a wtedy kody zmieniaja sie HURTOWO (nie o zmierzone 1,56 %
            # pozycji L2) i efekt wariantu jest NIECIAGLY. Przy skanie
            # nasycajacym 249 x 3 = 747 wariantow psuje to ~9 z nich, w tym
            # 42 % wariantow na samym skraju okna.
            #
            # Poprawki NIE robi sie w regule wyboru: probowalem policzyc
            # wariant "globalna statystyka zamiast leksykografii" (np. przewaga
            # puryn) i jest GORSZY -- statystyka globalna ma bliski remis dla
            # ~10 % sekwencji, wiec frakcja przewrocen rosnie do ~5 %.
            # Leksykografia jest dobra WLASNIE dlatego, ze decyduje o niej
            # 1-3 zasady, a nie cala sekwencja.
            #
            # Wlasciwa poprawka to PROTOKOL: przy ocenie wariantu kanonizuj
            # RAZ sekwencja odniesienia i narzuc te orientacje wariantom.
            # Wtedy przewrocenie jest niemozliwe Z KONSTRUKCJI. Sluzy do tego
            # `orientacja_odniesienia` (kontekst) albo `ustaw_orientacje`.
            _wym = getattr(self, "_orient_wymuszony", None)
            if _wym is not None:
                if _wym.shape[0] != x.shape[0]:
                    raise ValueError(
                        "narzucona orientacja ma %d wpisow, a batch ma %d -- "
                        "przy ocenie wariantow batch musi byc TEN SAM co dla "
                        "sekwencji odniesienia, inaczej narzuca sie orientacje "
                        "innej czasteczki"
                        % (int(_wym.shape[0]), int(x.shape[0])))
                from .kanon_rc import rc_onehot
                _m = _wym.to(x.dtype).view(-1, 1, 1)
                x = x * (1.0 - _m) + rc_onehot(x) * _m
                self._orient = _wym
            else:
                x, self._orient = kanonizuj(x)
            # JEDNO ZRODLO PRAWDY o nici kanonicznej. `embed` czyta stad
            # cel rekonstrukcji, zamiast liczyc kanonizacje drugi raz --
            # drugie wywolanie dostawalo nic JUZ kanoniczna i nadpisywalo
            # `_orient` zerami, przez co `kanon_frakcja` pokazywala 0,0
            # na prawdziwych sekwencjach (0,48 na losowych).
            self._x_kan = x
        occ = self._gate_occupancy(self.scanner(x))
        # ============ KONTEKST LICZONY TUTAJ, NIE CZYTANY Z CACHE'U ============
        # KIERUNEK ZALEZNOSCI JEST ISTOTNY. `_codes_input` liczy kontekst SAM
        # i wystawia go w `self._ctx` dla `embed`. Odwrotny kierunek --
        # `_codes_input` siegajacy po cache ustawiony w srodku `embed` -- byl
        # usterka rodziny Compact: samodzielne `encode_codes` podawalo drabinie
        # 4 kanaly zamiast 4+N i CICHO zwracalo inne kody niz trening.
        # `refiner` ma BatchNorm, wiec policzenie go DWA razy w trybie
        # treningu przepisalo by `running_*` dwukrotnie -- stad jedno miejsce.
        _h_occ = self.refiner(occ)
        if self.learned is not None:
            _h_occ = torch.cat([_h_occ, torch.sigmoid(self.learned(x))], dim=1)
        _h_occ = self.occ_proj(_h_occ)
        _g = self._graph(occ)
        self._ctx = {"occ": occ, "h_occ": _h_occ, "g": _g}
        # KLUCZE/WARTOSCI DLA ATENCJI DO L0 -- przez atrybut, nie przez
        # sygnature `forward`. Enkoder jest wolany rowniez z `encode_codes`
        # i `decode_codes`; zmiana sygnatury zlamalaby te drogi, a umowa
        # dyfuzji U2 wymaga, zeby przypisanie kodu bylo funkcja WYLACZNIE
        # sekwencji -- `_h_occ` nia jest (deterministyczny skan PWM).
        if getattr(self.codes, "mot_l0", None) is not None:
            self.codes._occ_kv = _h_occ
            self.codes._g_kv = _g
        if not self.kontekst_do_latentu:
            # LATENT CZYSTY Z SEKWENCJI, gdy oba kanaly kontekstu wylaczone.
            # To jest wariant docelowy przy atencji krzyzowej: kody niosa
            # WYLACZNIE sekwencje, wiec bijekcja i recon = 1,0000 zostaja,
            # a kontekst dochodzi do predykcji obok kwantyzatora.
            # WARUNEK NA WARSTWIE, nie na fladze. `motyw_proj` powstaje tylko
            # przy `motywy_do_latentu and not kontekst_do_latentu`, a tu jest
            # juz wiadomo, ze `kontekst_do_latentu` jest falszem -- wiec te dwa
            # warunki sa rownowazne. Pytanie o WARSTWE jest jednak odporne na
            # przyszla zmiane okablowania, a pytanie o flage nie: rozjazd
            # dawalby `TypeError: 'NoneType' object is not callable` na
            # pierwszym batchu zamiast czytelnej odmowy.
            if self.motyw_proj is None:
                return x
            return torch.cat([x, self.motyw_proj(occ)], dim=1)
        czesci = [x, _h_occ]
        if self.graf_proj is not None:
            # Wektor grafu jest STALY wzdluz sekwencji, wiec rozglaszamy waski
            # rzut, nie pelne 128 kanalow tej samej liczby.
            czesci.append(self.graf_proj(_g.unsqueeze(-1)).expand(
                -1, -1, x.shape[-1]))
        return torch.cat(czesci, dim=1)

    def renormalise_shells(self):
        """Przywraca promienie powlok po kroku optymalizatora.

        Wolane z petli treningowej obok istniejacego `level_dec.renormalise_gauge()`
        -- ten sam tryb awarii, ktory tam juz naprawiono, a kodebook zostal bez
        naprawy. Zwraca wspolczynniki skalowania per poziom do logu: wartosc
        odlegla od 1,0 znaczy, ze powloka realnie dryfowala w tym kroku.
        """
        if not bool(getattr(self, "renorm_shells", False)):
            return None
        return [vq.renormalise_radius() for vq in self.codes.vqs]

    def punkty_poziomow(self):
        """Kody jako PUNKTY NA HIPERBOLOIDZIE, jeden tensor na poziom.

        ``[B, L_k, dim+1]``. To jest to samo wyrazenie, ktore stoi juz w
        ``decode_codes`` -- wpis kodebooka lezy na hiperboloidzie z konstrukcji,
        wiec indeksowanie go nie wymaga zadnej projekcji.

        Wymaga wczesniejszego przebiegu ``self.codes(...)``; poza nim
        ``self.codes.indices`` nie istnieje i to jest uczciwy blad, nie cichy
        fallback -- ten sam wzorzec usterki, przez ktory w rodzinie Compact
        samodzielne ``encode_codes`` zwracalo inne kody niz trening.
        """
        idx = getattr(self.codes, "indices", None)
        if idx is None:
            raise RuntimeError(
                "punkty_poziomow() przed przebiegiem codes(...) -- brak indices")
        # ``st_points``, NIE ``embed[i]``. Dwa powody, oba zmierzone:
        #
        # 1. ``embed`` jest ``register_buffer`` przy ``vq_learnable: false``,
        #    wiec ``requires_grad = False``: cala galaz glowic hiperbolicznych
        #    mialaby ``d(strata)/d(enkoder) = 0`` STRUKTURALNIE. Glowice uczyly
        #    sie wylacznie wlasnych prototypow nad zamrozonym slownikiem, a
        #    gdyby ``realnosc`` weszla jako gracz Nasha, straznik zerowego
        #    gradientu rzucilby w pierwszym kroku.
        # 2. ``_update`` mutuje ``embed`` W MIEJSCU (``hyperbolic.py:823``)
        #    PRZED powrotem z ``forward``, a ``quant`` jest pobrany przed nim.
        #    Czytanie bufora po przebiegu dawalo glowicom kodebook o jeden krok
        #    EMA przesuniety wzgledem tego, co kwantyzator faktycznie przypisal.
        #
        # ``st_points = points + (quant - points).detach()`` jest co do wartosci
        # rowne ``embed[i]`` sprzed aktualizacji i lezy na hiperboloidzie; rozni
        # sie WYLACZNIE przepuszczeniem gradientu straight-through.
        if getattr(self.codes, "drabina", None) is not None:
            # Drabina stozkowa sklada punkty sama i trzyma je w `_punkty`.
            # `st_points` istnieje tam tylko na poziomie 0, wiec stara sciezka
            # zwrocilaby `None` dla poziomow stycznych i cicho przeszla do
            # gałęzi `embed[i]`, ktora dla wektorow stycznych zwraca WEKTORY,
            # nie punkty -- ksztalt by sie zgadzal (dim), a wymiar czasowy
            # nie, wiec padloby dopiero w glowicy.
            _p = getattr(self.codes, "_punkty", None)
            if _p is None:
                raise RuntimeError("punkty_poziomow() przed przebiegiem "
                                   "codes(...) -- brak _punkty drabiny")
            return _p
        punkty = [getattr(v, "st_points", None) for v in self.codes.vqs]
        if any(p is None for p in punkty):
            return [self.codes.vqs[k].embed[i] for k, i in enumerate(idx)]
        return punkty

    def _przechwyc_trzon(self, mod, we, wy):
        """Hak na ostatnim bloku trzonu -- jedyne zrodlo `_mapa_trzonu`.

        To jest wyjscie `blocks[-1]` PRZED jego FiLM i przed `krzyz`, czyli
        mapa po pieciu blokach -- ta sama definicja, ktora `HypCompactNet`
        podaje `level_dec` (`_capture_trunk`). Trzymamy ja identycznie, zeby
        `attn_dim` i sens atencji poziom->trzon byly porownywalne miedzy
        obiema rodzinami.
        """
        self._mapa_trzonu = wy

    def _promienie_poziomow(self):
        """PROMIENIE POWLOK, wyprowadzone z modelu -- NIE `level_radii`.

        `level_radii` w tej klasie to `radius_scale`, czyli SIGMA:
        `radius_scale = sinh(promien_cel)/sqrt(dim)`, a wiec 0,25 =
        sinh(1,4436)/8 przy dim 64 (komentarz przy `norm_poz` podaje ten sam
        rachunek). `HypCompactNet` przekazuje `codes.level_radii` wprost do
        `LevelResidualDecoder`; zrobienie tego TUTAJ ustawiloby prototypy na
        powlokach 0,25 / 0,6 / 1,2, gdy punkty leza na 1,44 / 2,27 / 2,95.
        Wtedy `d^2` wypada poza zakres [0, 2 sinh^2(r)], z ktorego wyprowadzone
        jest PRAWO TEMPERATURY, i caly odczyt geodezyjny mierzylby skale
        zamiast tresci. Repo ma to zapisane dwukrotnie: "level_radii to sigma,
        nie promien" oraz "promien do tau trzeba ZMIERZYC, nie odczytac".

        Drabina stozkowa: promien poziomu k to `promien_cel(L0) + k*krok`, i ta
        liczba jest JUZ policzona w konstruktorze stosu jako bufor `norm_poz`
        (`sinh(r_k)/sqrt(dim)`) -- odwracamy ja `asinh`, zamiast liczyc drugi
        raz z innej strony. Dwa niezalezne wyliczenia tej samej wielkosci to
        w tym repo zapisana rodzina awarii (rozjazd `_codes_input` z
        `_encoder_in_channels`).

        Bez drabiny promien powloki poziomu to wprost `promien_cel` jego VQ.
        """
        st = self.codes
        if getattr(st, "drabina", None) is not None:
            _np = getattr(st, "norm_poz", None)
            if _np is None:
                raise RuntimeError(
                    "glowice_poziomowe + drabina bez bufora `norm_poz` -- "
                    "promieni powlok nie da sie odczytac, a zgadniecie ich "
                    "unieważnia prawo temperatury. Formaty sie roznia.")
            _s = math.sqrt(float(st.dim))
            return [math.asinh(float(v) * _s) for v in _np.detach()]
        return [float(v.promien_cel) for v in st.vqs]

    def _podsumowania_poziomow(self):
        """Podsumowanie kodow per poziom: lista `n_levels` tensorow [B, dim].

        FORMAT JEST TEN SAM co `HypCompactNet._level_summaries` (sprawdzone
        czytaniem obu): ciecie prefiksu po kanalach plus srednia po pozycjach.
        Uklad kanalow `z` to `[q0..q_{L-1} | r0..r_{L-1} | slot]`
        (`HyperbolicCodeStack.out_channels`), wiec `k`-ty blok `dim` kanalow to
        kody poziomu `k`, a `level_decoder.sum_proj[k]` to `Linear(code_dim,
        hidden)` -- czyli oczekuje dokladnie [B, dim].

        Czytamy `self._z_last`, czyli tensor PO `latent_grad`, a nie surowe
        `z`. Rodzina Compact ma ten wybor zapisany wprost: podanie surowego
        `z` sprawialo, ze dekoder poziomow omijal profil gradientu CALKOWICIE
        (zmierzony profil [1,0; 1,7143; 1,141; 0,4789; 0,0248] wobec zadanego
        [1,0; 0,7; 0,3; 0,1; 0,0]). Konsekwencja przy `latent_grad: none` jest
        jawna i raportuje ja `raport_kluczy_bez_efektu`: galaz przewiduje, ale
        gradient nie dochodzi do enkodera.
        """
        z = getattr(self, "_z_last", None)
        if z is None:
            raise RuntimeError(
                "_podsumowania_poziomow() przed `_code_features` -- brak "
                "`_z_last`. To jest zepsute okablowanie, nie brak danych.")
        dim, n = int(self.codes.dim), int(self.codes.n_levels)
        if z.dim() != 3 or z.shape[1] < dim * n:
            raise RuntimeError(
                "widok trzonu ma ksztalt %r, a same kody wymagaja %d kanalow "
                "(dim=%d, n_levels=%d) -- format `_z_last` nie zgadza sie z "
                "tym, czego oczekuje `LevelResidualDecoder.forward`, i nie "
                "dopasowuje go na sile."
                % (tuple(z.shape), dim * n, dim, n))
        return [z[:, k * dim:(k + 1) * dim].mean(-1) for k in range(n)]

    def forward(self, x):
        # ZEROWANIE ZRODLA NA WEJSCIU. `_mapa_trzonu` ustawia hak na
        # `blocks[-1]`; bez zerowania mapa z POPRZEDNIEGO batcha przezylaby
        # przebieg, w ktorym bloki sie nie wykonaly, i glowice poziomowe
        # czytalyby klucze z innych sekwencji. To ta sama rodzina awarii co
        # `_h_occ_rek` i `_g_rek` w `embed` -- tam batch by sie zgadzal, wiec
        # nic by nie padlo.
        self._mapa_trzonu = None
        feat = self.embed(x)
        _fp = getattr(self, "_feat_per", None)
        feat_hk = _fp["hk"] if _fp is not None else feat
        if self.pm_dev is not None:
            # ZAPYTANIE Z LATENTU: sredni wektor styczny punktow L0. Oba
            # moduly dostaja to samo wejscie i te same wezly motywow --
            # rozni je WYLACZNIE uczone rzutowanie i cel, do ktorego ida.
            # Wezly biora sie z hipergrafu, wiec podprzestrzen pyta
            # o ZBIORY wspolwystepujacych motywow, nie o pojedyncze.
            _pk = getattr(self.codes, "_punkty", None)
            _wz = getattr(self, "_wezly_mot", None)
            if _pk and _wz is not None:
                _q = logmap0(_pk[0].double()).mean(dim=1).float()
                feat = torch.cat([feat, self.pm_dev(_q, _wz)], dim=-1)
                feat_hk = torch.cat([feat_hk, self.pm_hk(_q, _wz)], dim=-1)
        dev, hk = self.fc_dev(feat), self.fc_hk(feat_hk)

        # KOD GLOBALNY: wklad grupy DOKLADANY do predykcji glowicy.
        # `wy` jest zero-init, wiec w kroku zerowym to tozsamosc — i to
        # jest falsyfikator wpiecia, nie wada. Strata commitment ma dojsc
        # do optymalizatora przez `_kg_strata`; jesli nie dojdzie, modul
        # bedzie wygladal na dzialajacy (indeksy sie licza, entropia rosnie),
        # a kodebook zostanie na inicjalizacji. To ta sama rodzina awarii
        # co `_jepa_strata` ustawiane i nigdy nie czytane.
        if self.kod_glob is not None:
            _kgw, _kgi, _kgs = self.kod_glob(feat)
            dev = dev + _kgw[:, :1]
            hk = hk + _kgw[:, 1:]
            self._kg_strata = _kgs
            self._kg_idx = _kgi
        if self.gram is not None:
            _og = self.gram(self.punkty_poziomow())
            dev = dev + _og["dev"].unsqueeze(-1).to(dev.dtype)
            hk = hk + _og["hk"].unsqueeze(-1).to(hk.dtype)
        if self.lat_brama is not None:
            # ODCZYT NIEZMIENNICZY, ADDYTYWNIE. Trzon jest poziomem zerowym
            # predykcji i nie jest niezmienniczy; ta galaz jest, wiec suma
            # ma luke RC mniejsza niz sam trzon proporcjonalnie do udzialu
            # bramy. To jest jedyna droga, ktora ten modul ma do wyjscia --
            # `prefiksy()` nie istnieja, bo `OdczytZLatentu` nie sklada
            # predykcji per poziom i nie udaje, ze sklada.
            _ol = self.hyp_heads(self.punkty_poziomow())
            dev = dev + self.lat_brama[0] * _ol["dev"].unsqueeze(-1)
            hk = hk + self.lat_brama[1] * _ol["hk"].unsqueeze(-1)
            if self._lv_z_latentu:
                self._logvar_dev = _ol["logvar_dev"]
                self._logvar_hk = _ol["logvar_hk"]
                self._pre_lv = getattr(self.hyp_heads, "_pre_lv", None)
        if self.kal_wariancji and self.fc_dev_lv is not None:
            # TE SAME WAGI, cechy ODCIETE. Czlon kalibracyjny dziala
            # wylacznie tutaj, wiec nie ma drogi do trzonu ani enkodera --
            # dopasowanie sredniej jest chronione Z KONSTRUKCJI, a nie
            # przez mala wage. Wspolne wagi sprawiaja, ze kalibracja
            # przenosi sie na wyjscie uzywane przy wnioskowaniu.
            # TO SAMO OGRANICZENIE `2*tanh(x/2)`, co na wyjsciu uzywanym.
            # Pierwsza wersja go pomijala, czyli czlon kalibracyjny
            # optymalizowalby INNA FUNKCJE niz ta, ktora idzie do NLL
            # i do wnioskowania -- a wagi sa wspolne, wiec uczylby glowice
            # pod cel, ktorego nikt nie czyta.
            self._lv_kal = (
                2.0 * torch.tanh(self.fc_dev_lv(
                    self._wejscie_lv(feat.detach(), dev)).squeeze(-1) / 2.0),
                2.0 * torch.tanh(self.fc_hk_lv(
                    self._wejscie_lv(feat_hk.detach(), hk)).squeeze(-1) / 2.0))
        else:
            self._lv_kal = None
        if self.fc_dev_lv is not None:
            # Ograniczenie z obu stron: bez niego NLL ucieka do -inf przez
            # rozdmuchanie sigma. exp(-4)..exp(4) to 0,018..54,6 wariancji,
            # przy sd celu 1,52 (Dev) i 1,64 (Hk) -- z zapasem w obie strony.
            # OGRANICZENIE MIEKKIE, NIE `clamp`. Zmierzone na biegu
            # HypEncFinal (2026-09-05, epoka 5): `clamp(-2, 2)` to DRZWI
            # JEDNOKIERUNKOWE -- gdy glowica raz zjedzie ponizej -2, pochodna
            # `clamp` jest DOKLADNIE zerem dla kazdej probki, wiec ani waga,
            # ani obciazenie nie dostaja juz zadnego gradientu i stan jest
            # nieodwracalny do konca treningu. W logu widac to jako
            # `lv_sr = -2.0` (dokladnie prog) przy `lv_sd = 0.0000` (dokladnie
            # stala) -- glowica przestala byc predyktorem i zostala mnoznikiem.
            # `2*tanh(x/2)` ma ten sam zakres (-2, 2) i te sama pochodna 1 w
            # zerze, ale nigdzie nie zeruje gradientu, wiec zjazd do granicy
            # jest odwracalny.
            _pre_d = self.fc_dev_lv(
                self._wejscie_lv(feat, dev)).squeeze(-1)
            _pre_h = self.fc_hk_lv(
                self._wejscie_lv(feat_hk, hk)).squeeze(-1)
            self._logvar_dev = 2.0 * torch.tanh(_pre_d / 2.0)
            self._logvar_hk = 2.0 * torch.tanh(_pre_h / 2.0)
            with torch.no_grad():
                # B4. Po zamianie `clamp` na `tanh` SYGNATURA AWARII ZNIKA:
                # przy ucieczce x -> -inf bedzie `lv_sr = -1,9999...` i
                # `lv_sd ~ 1e-4` zamiast dokladnych -2,0000 / 0,0000. Stary
                # falsyfikator "lv_sd > 0" zaczalby wiec PRZECHODZIC przy
                # glowicy calkowicie martwej. Preaktywacja jest nieograniczona,
                # wiec pokazuje ucieczke, ktorej tanh juz nie pokaze.
                self._pre_lv = (float(_pre_d.abs().mean()),
                                float(_pre_h.abs().mean()))
        elif not self._lv_z_latentu:
            self._logvar_dev = self._logvar_hk = None
        self._realnosc = None
        self._prefiksy_dev = None
        self._prefiksy_hk = None
        if self.hyp_heads is not None and self.lat_brama is None:
            # ADDYTYWNIE, nie zamiast: trzon jest poziomem zerowym predykcji,
            # a odczyt z kodow go POPRAWIA. To jest boosting, i przy zerowym
            # wkladzie glowic bieg wraca do bazy sam z siebie.
            #
            # Brama startuje na 0,02, NIE na zerze. Zerowa brama nie przepuszcza
            # gradientu w gore i przez to gasi wlasne zrodlo -- blad popelniony
            # w tym repo czterokrotnie (patrz `film_init_std`, wycofane).
            o = self.hyp_heads(self.punkty_poziomow())
            # PREFIKSY DO STRATY HIERARCHICZNEJ. `prefiksy()[g][k]` to predykcja
            # z poziomow 0..k. Do tej pory byly LICZONE I NIGDY NIE UZYWANE --
            # `train_graph.py` nie wolal ich ani razu, wiec hierarchia byla
            # wylacznie strukturalna i nic nie wymuszalo, zeby poziom glebszy
            # POPRAWIAL predykcje plytszego. Wystawiamy je z brama, w tej samej
            # skali co wyjscie, zeby strata per poziom byla porownywalna z celem.
            self._prefiksy_dev = [self.fc_dev(feat).squeeze(-1) + self.hyp_gate[0] * v
                                  for v in self.hyp_heads.prefiksy()["dev"]]
            self._prefiksy_hk = [self.fc_hk(feat_hk).squeeze(-1) + self.hyp_gate[1] * v
                                 for v in self.hyp_heads.prefiksy()["hk"]]
            g = self.hyp_gate
            with torch.no_grad():
                # FALSYFIKATOR WKLADU, nie macierzy atencji. `attn_H` mowi,
                # czy atencja jest peaked, i moze byc zdrowe przez caly bieg
                # przy gasnacej bramie -- a wtedy modul nie wnosi NIC.
                # Zmierzone w tym repo: `hyp_gate` spadla 0,02 -> 1,42e-04
                # (mnoznik 7,1e-03) i glowice zwracaly stala o sd 5,93e-09,
                # a zaden klucz w logu tego nie pokazywal. Bramy sa dzis
                # zwolnione z weight decay (wzorzec "gate"), wiec ta liczba
                # jest testem, czy zwolnienie realnie zadzialalo.
                _wd = (g[0] * o["dev"].unsqueeze(-1)).pow(2).mean().sqrt()
                _wh = (g[1] * o["hk"].unsqueeze(-1)).pow(2).mean().sqrt()
                self._hyp_wklad = (
                    round(float(_wd / dev.pow(2).mean().sqrt().clamp_min(1e-9)), 4),
                    round(float(_wh / hk.pow(2).mean().sqrt().clamp_min(1e-9)), 4))
                self._hyp_brama = [round(float(v), 5) for v in g.detach()]
            # ---- CEL WLASNY GLOWIC KODOWYCH (omija brame) ----
            #
            # ZMIERZONA AWARIA, KTORA TO NAPRAWIA. `hyp_wklad` w biegu
            # HyppeDeepSTARR4 spada monotonicznie 0,1179 -> 0,0895 -> 0,0745
            # -> 0,0687 dla Dev, a dla Hk stoi na DOKLADNYM zerze przy
            # `hyp_brama[1] = 1,2e-04`. Weight decay tu nie zawinil -- wzorzec
            # "gate" zwalnia `hyp_gate` z decayu i bramka `smoke_maska` to
            # potwierdza (`wd = [0.0]`).
            #
            # Zawinila TOPOLOGIA. Gradient po bramie to `<dL/dhk, o_hk>`, wiec
            # dopoki zrodlo jest nieinformatywne, optimum bramy jest w zerze i
            # spadek gradientu znajduje je szybko. A gdy `g -> 0`, gradient do
            # ZRODLA wynosi `g * dL/dhk -> 0`. To stan POCHLANIAJACY, nie
            # przejsciowy. Repo ma to nazwane: "wygrzewac ZRODLO, nie SKALE".
            #
            # I nie ratuje tego czlon hierarchiczny: `_prefiksy_dev` to
            # `fc_dev(feat) + hyp_gate[0]*v`, czyli TEZ przez brame. Wszystkie
            # drogi od `o` do jakiejkolwiek straty przechodza przez `hyp_gate`,
            # wiec zrodlo jest osierocone strukturalnie.
            #
            # CEL WLASNY przecina te petle: `o` musi odtworzyc predykcje
            # trzonu Z SAMYCH RELACJI MIEDZY KODAMI, niezaleznie od tego, ile
            # brama przepuszcza. Gdy zrodlo stanie sie informatywne, brama ma
            # co podniesc. To ten sam wzorzec, ktory w tym repo jako JEDYNY
            # utrzymal galaz boczna przy zyciu (`trunk_aux_weight`, tor
            # trojkatny nizej) -- i jest to realizacja tezy, ze wzajemne
            # relacje w latencie moga podbic predykcje: tutaj nie tylko MOGA,
            # ale MUSZA, bo maja wlasny cel.
            #
            # CEL JEST ODCIETY i wziety PRZED dodaniem galezi, wiec nie ma
            # samoodniesienia i glowice nie ciagna trzonu do siebie.
            #
            # FALSYFIKATORY: `hyp_zrodlo` (sd `o` po batchu) -- udokumentowana
            # smierc tej galezi to stala o sd 5,93e-09, wiec ta liczba MUSI
            # odstawac od zera; `hyp_aux` (wartosc czlonu) i `hyp_wklad`,
            # ktory ma teraz ROSNAC, a nie opadac.
            # ZRODLO WYSTAWIONE Z GRADIENTEM. `train_graph` liczy z niego cel
            # na ETYKIETACH; tutaj zostaje wylacznie wariant "trzon".
            #
            # POPRAWKA PO ZMIERZONEJ PORAZCE. Wariant "trzon" (cel =
            # `predykcja_trzonu.detach()`) OZYWIL galaz i NIE dal PCC:
            # HD5 epoki 1-5 `hyp_wklad` 0,198/0,227 -> 0,434/0,356 przy
            # bramie 0,02 -> 0,488, a srednia PCC 0,63525 / 0,66540 /
            # 0,67960 / 0,69230 / 0,69155 wobec HD4 0,64040 / 0,66715 /
            # 0,68010 / 0,69220 / 0,69125 -- remis co do szumu.
            # Przyczyna jest konstrukcyjna: kazac relacjom w latencie
            # ODTWARZAC predykcje trzonu to kazac im byc DUPLIKATEM, a
            # duplikat nie dodaje informacji, wiec brama rosnie (suma
            # dwoch skorelowanych predyktorow lekko zbija wariancje) i PCC
            # stoi. Cel na ETYKIETACH zmusza galaz, zeby dolozyla to,
            # czego trzon NIE MA.
            self._hyp_o = (o["dev"], o["hk"])
            if self.hyp_aux_waga > 0.0 and self.hyp_aux_cel == "trzon":
                _cd = dev.detach().reshape(o["dev"].shape)
                _ch = hk.detach().reshape(o["hk"].shape)
                _ha = self.hyp_aux_waga * (F.mse_loss(o["dev"], _cd)
                                           + F.mse_loss(o["hk"], _ch))
                self.reg_loss = self.reg_loss + _ha
                with torch.no_grad():
                    self._hyp_aux = round(float(_ha), 5)
            with torch.no_grad():
                self._hyp_zrodlo = [round(float(o["dev"].std()), 5),
                                    round(float(o["hk"].std()), 5)]
            dev = dev + g[0] * o["dev"].unsqueeze(-1)
            hk = hk + g[1] * o["hk"].unsqueeze(-1)
            # `realnosc` TYLKO gdy glowica istnieje. Przy `glowica_realnosci:
            # false` `o` nie ma tego klucza, `g` ma dwa wpisy, a `_realnosc`
            # zostaje `None` -- czyli dokladnie ten stan, ktory asercja
            # `negatywy_frac` w `train_graph` uznaje za twardy blad
            # konfiguracji. To jest zamierzone: negatywy bez glowicy maja
            # KRZYCZEC, a glowica bez negatywow ma nie powstawac.
            if "realnosc" in o:
                self._realnosc = g[2] * o["realnosc"]
        if (self.trojkat is not None and self.troj_waga > 0
                and self._troj_aux is not None):
            # GALAZ Z WLASNYM CELEM. Reprezentacja parowa musi odtworzyc
            # wyjscie modelu z samych par -- cel jest ODCIETY, wiec nie
            # zaklóca glowic, ale galezi nie da sie wyzerowac bez kary.
            # To ten sam wzorzec, ktory w tym repo utrzymal galaz boczna
            # zywa (`trunk_aux_weight`), i jedyny, ktory to zrobil.
            # Bez etykiet, wiec bez zmian w `train_graph`.
            _cel = torch.cat([dev, hk], dim=-1).detach()
            _t = self.troj_waga * F.mse_loss(self._troj_aux, _cel)
            self._troj_strata = _t.detach()
            self.reg_loss = self.reg_loss + _t
        if self.level_dec is not None:
            # ---- GLOWICE POZIOMOWE: ADDYTYWNIE I NA SAMYM KONCU ----
            #
            # ADDYTYWNIE, dokladnie jak `lat_brama` i `hyp_heads`: trzon jest
            # poziomem zerowym predykcji, a odczyt per poziom go POPRAWIA.
            # Przy zerowym wkladzie bieg wraca do bazy sam z siebie.
            #
            # NA KONCU, i to nie jest obojetne. Kazda inna galaz w tym
            # `forward` liczy swoj cel z `dev`/`hk` ODCIETYCH
            # (`hyp_aux_cel: trzon` bierze `dev.detach()`, `_troj_aux` bierze
            # `cat([dev, hk]).detach()`), wiec wpiecie sie PRZED nimi
            # przestawiloby im cel. Jedna nowa flaga zmienialaby wtedy dwie
            # rzeczy naraz i zaden z tych mechanizmow nie dalby sie porownac
            # z wlasnym, wczesniejszym pomiarem.
            if self._mapa_trzonu is None:
                raise RuntimeError(
                    "glowice_poziomowe: brak `_mapa_trzonu`. Hak na "
                    "`blocks[-1]` nie odpalil, wiec mapy trzonu nie ma -- to "
                    "jest zepsute okablowanie, a nie martwy mechanizm. "
                    "(Dokladne zero w ablacji tej rodziny zawsze oznaczalo "
                    "zepsuta interwencje, nie brak sygnalu.)")
            _pk = self.punkty_poziomow()
            # JEDEN PUNKT NA POZIOM: centroid Lorentza po slotach, tak samo
            # jak `HypCompactNet._tree_points`. I to jest miejsce, w ktorym
            # ten wybor MA cene, zapisana w `geo_features`: centroid siedzi na
            # promieniu rho < r_k, wiec softmax po odleglosciach jest
            # plaszczony ARYTMETYCZNIE. Falsyfikator jest w logu jako
            # `gp_geo_rho_frac` = sinh(rho)/sinh(r) -- przy wartosci rzedu
            # 0,04 kazda liczba `gp_geo_*` jest bez wartosci i to konfiguracja
            # jest do poprawy, a nie wynik do interpretacji.
            _pl = [lorentz_centroid(p, dim=1) for p in _pk]
            _sc = None
            if self.level_dec.hyp_queries:
                # WARTOSCI = czesc przestrzenna punktu, czyli sam wpis
                # kodebooka per slot; [B, L_k, D+1] -> [B, D, L_k], bo
                # `hyp_attend` transponuje z powrotem. Uzasadnienie wyboru
                # (a nie cyfr FSQ) przy `_gp_fsq` w `__init__`.
                _sc = [p[..., 1:].transpose(1, 2) for p in _pk]
            _op = self.level_dec(self._podsumowania_poziomow(), _pl,
                                 self._mapa_trzonu,
                                 slot_points=_pk, slot_codes=_sc)
            # `train_graph` liczy z tego gleboki nadzor; `_parts` zostaje
            # w dekoderze, wiec przechowujemy SUME, nie kopiujemy skladnikow.
            self._pred_poziomow = _op
            with torch.no_grad():
                # FALSYFIKATOR WKLADU, ta sama definicja co `hyp_wklad`:
                # RMS(wklad galezi) / RMS(wyjscia), per zadanie. Bez tej
                # liczby galaz moze zwracac stala (udokumentowana smierc
                # `hyp_heads`: sd 5,93e-09 przy sd(Dev) = 1,61) i ANI JEDEN
                # klucz w logu tego nie pokaze. Rampa `gp_ramp` mnozy wyjscie,
                # wiec w epoce 1 liczba jest z definicji mniejsza -- alarm
                # jest na BRAK WZROSTU, nie na prog bezwzgledny.
                self._gp_wklad = (
                    round(float(_op[:, 0].pow(2).mean().sqrt()
                                / dev.pow(2).mean().sqrt().clamp_min(1e-9)), 4),
                    round(float(_op[:, 1].pow(2).mean().sqrt()
                                / hk.pow(2).mean().sqrt().clamp_min(1e-9)), 4))
            # PREDYKCJA SAMEGO TRZONU, PRZED dodaniem galezi. To jest
            # obserwabla, nie czlon straty, wiec zapisywana ZAWSZE gdy
            # galaz istnieje. `gp_wklad` nie zastapi jej z konstrukcji:
            # iloraz RMS rosnie i gdy galaz jest uzyteczna, i gdy trzon
            # zamiera, wiec dwie przeciwne diagnozy daja te sama liczbe.
            # Zmierzony precedens: po wyzerowaniu `level_dec` wlasny odczyt
            # trzonu korelowal z celem na 0,193.
            self._pred_trzonu = (dev, hk)
            dev = dev + _op[:, :1]
            hk = hk + _op[:, 1:2]
        return dev, hk

    def raport_trzonu(self):
        """`gp_r_trzon`: czy trzon PRZED dodaniem galezi jeszcze przewiduje.

        Zwraca `None`, gdy galezi nie ma -- wtedy pytanie nie istnieje.
        Korelacje liczy `train_graph`, bo tu nie ma etykiet; ta metoda
        oddaje tylko surowe predykcje trzonu.
        """
        return getattr(self, "_pred_trzonu", None)

    def hetero_report(self):
        """Falsyfikator glowicy wariancji: czy przewidywana sigma cokolwiek rozroznia.

        Glowica heteroscedastyczna moze sie NAUCZYC STALEJ -- wtedy NLL redukuje
        sie do przeskalowanego MSE i caly mechanizm jest ozdoba, a w PCC wyglada
        identycznie jak "nie pomaga". Dwie liczby to rozstrzygaja:
          *  -- odchylenie przewidywanego log-wariancji po batchu.
            Zero znaczy stala.
          *  -- korelacja przewidywanej sigmy z RZECZYWISTYM bledem
            |y - mu|. To jest wlasciwy test: glowica ma przewidywac, GDZIE model
            sie myli. Zero znaczy, ze zgaduje.
        """
        lv = getattr(self, "_logvar_dev", None)
        if lv is None:
            return None
        out = {"lv_sr": round(float(lv.mean()), 4),
               "lv_sd": round(float(lv.std()), 4)}
        lh = getattr(self, "_logvar_hk", None)
        if lh is not None:
            out["lv_sd_hk"] = round(float(lh.std()), 4)
        # `lv_r` -- WLASCIWY test, ktory ten docstring obiecywal, a ktorego
        # NIE BYLO w kodzie: korelacja przewidywanej log-wariancji z
        # RZECZYWISTYM kwadratem bledu. Bez niej `lv_sd > 0` mowi tylko, ze
        # glowica cokolwiek rozroznia -- nie, ze rozroznia WLASCIWIE. Blad
        # wstawia petla treningowa (`_err_dev`), bo model nie widzi celu.
        if lh is not None:
            out["lv_sr_hk"] = round(float(lh.mean()), 4)
        # SCIANA: ile procent probek siedzi przy granicy. Bez tego `lv_sd > 0`
        # przechodzi przy glowicy przyklejonej do -2 (patrz B4 wyzej).
        out["lv_scia"] = round(float((lv.abs() > 1.9).double().mean()), 4)
        _kr = getattr(self, "_kal_r", None)
        if _kr is not None:
            # `kal_r` to `lv_r` liczone na ODCIETYM wyjsciu w trakcie
            # treningu. Rozjazd `kal_r` rosnie / `lv_r` stoi znaczy, ze
            # odciecie rozdzielilo dwa wyjscia i wzorzec sie nie przenosi.
            out["kal_r"] = (round(_kr[0], 4), round(_kr[1], 4))
        _pre = getattr(self, "_pre_lv", None)
        if _pre is not None:
            out["pre_lv"] = (round(_pre[0], 3), round(_pre[1], 3))
        # KORELACJA Z RZECZYWISTYM BLEDEM -- wlasciwy test, akumulowany przez
        # petle treningowa (`_lvr_dev`/`_lvr_hk`), bo tylko tam `logvar` i blad
        # pochodza z TEGO SAMEGO batcha. Poprzednia wersja czytala je z dwoch
        # roznych batchy o rozmiarach 56 i 58 i nie wypisala sie ani razu.
        for _k, _a in (("lv_r", "_lvr_dev"), ("lv_r_hk", "_lvr_hk")):
            _st = getattr(self, _a, None)
            if not _st or _st[0] < 2:
                continue
            _n, _sx, _sy, _sxx, _syy, _sxy = _st
            _cov = _sxy / _n - (_sx / _n) * (_sy / _n)
            _vx = max(_sxx / _n - (_sx / _n) ** 2, 0.0)
            _vy = max(_syy / _n - (_sy / _n) ** 2, 0.0)
            _d = (_vx * _vy) ** 0.5
            out[_k] = round(_cov / _d, 4) if _d > 1e-12 else 0.0
            setattr(self, _a, [0.0] * 6)     # zeruj po odczycie: liczba ma byc
        return out                            # z JEDNEJ epoki, nie kumulowana

    def get_features(self, x):
        return self.embed(x)

    @torch.no_grad()
    def reconstruction_accuracy(self, x):
        self.embed(x)
        return float((self.recon_logits.argmax(1) == x.argmax(1)).float().mean())

    @torch.no_grad()
    def hierarchy_report(self):
        """Test the hierarchy hypothesis, level by level.

        The hypothesis: level 0 carries the most global statement about the
        sequence -- the regulatory grammar -- and each later level applies a
        progressively smaller refinement.  In hyperbolic space that predicts two
        things, and both are checkable here rather than assumed:

        ``radius_mean`` should **increase** with level.  Volume at radius r
        grows like ``sinh^(n-1)(r)``, so a shell near the origin holds few
        general codes and outer shells hold exponentially many specific ones.
        If the radii stay flat, curvature is idle and Euclidean would serve
        equally well -- which is the outcome to report honestly, not hide.

        ``contribution`` should **decrease** with level: measured in one common
        frame, each successive level should add less to the reconstruction than
        the one before -- "the first layer is the most global, and each next
        step is a smaller change".  A flat or rising profile falsifies the
        nesting and means the levels are competing to explain the same variance
        rather than refining one another.
        """
        radii = self.codes.radii()
        report = {
            "perplexity": self.codes.perplexities(),
            "radius_mean": [float(r.mean()) for r in radii],
            "radius_std": [float(r.std()) for r in radii],
            "contribution": list(self.codes._last_steps or []),
        }
        rm, st = report["radius_mean"], report["contribution"]
        report["radius_increases"] = all(a < b for a, b in zip(rm, rm[1:])) if len(rm) > 1 else None
        report["contribution_decreases"] = all(a > b for a, b in zip(st, st[1:])) if len(st) > 1 else None
        return report
