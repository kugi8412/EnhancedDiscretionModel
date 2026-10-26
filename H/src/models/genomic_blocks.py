from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dna import COMPLEMENT_PERMUTATION


class RCConv1d(nn.Module):
    """Convolution whose filters are tied to their reverse complements.

    One weight tensor produces two filter banks: ``W`` and its reverse
    complement, formed by reversing along the kernel axis and permuting input
    channels through the complement map.  Output channel ``i`` and ``i + out``
    are therefore an RC pair by construction, not by training.

    Only valid as the **first** layer, where input channels are literal
    nucleotides — deeper layers would need the pairing tracked through every
    intermediate channel.  That is where almost all of the benefit is anyway:
    it is the base-level filters that must be strand-symmetric.

    Extra input channels were tried here and REMOVED. The idea was that motif
    occupancy, being strand-invariant, would tie by reversal alone. Measured, it
    does not: |scanner(rc(x)) - reverse(scanner(x))| came out at 8.75e-01, no
    better than the no-reversal control. The scanner pads on the right only and
    its motifs have different widths, so each channel shifts by its own w-1
    under reversal and no single flip realigns them. Occupancy is invariant in
    VALUE, not in indexing.
    """

    def __init__(self, out_channels, kernel_size, padding=None, bias=True):
        super().__init__()
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.padding = (self.kernel_size // 2) if padding is None else int(padding)
        self.weight = nn.Parameter(
            torch.randn(out_channels, 4, kernel_size) * (1.0 / (4 * kernel_size) ** 0.5))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.register_buffer("perm", torch.tensor(COMPLEMENT_PERMUTATION), persistent=False)

    def rc_weight(self):
        # Reverse along the kernel axis AND complement the base axis.  Doing
        # only one of the two is the classic bug: it yields the reverse, not
        # the reverse complement, and silently trains the wrong symmetry.
        return self.weight.flip(-1)[:, self.perm]

    def forward(self, x):
        w = torch.cat([self.weight, self.rc_weight()], dim=0)
        b = torch.cat([self.bias, self.bias]) if self.bias is not None else None
        return F.conv1d(x, w, b, padding=self.padding)


class RCStemSplit(nn.Module):
    """Stem RC-wiazany na NUKLEOTYDACH, swobodny na kanalach dodatkowych.

    POWOD ISTNIENIA, ZMIERZONY. `HyperbolicCodeStack` tworzy `RCConv1d` tylko
    pod warunkiem `rc_stem and in_ch == 4`. Przy `motywy_do_latentu: true`
    doklada sie 8 kanalow motywowych, wiec `in_ch = 12` i warunek jest
    FALSZYWY -- stem jest zwyklym `Conv1d`, a `rc_stem: true` w configu to
    CICHY NO-OP. Sprawdzone na HypGram5, HypEncI i HypGram3: we wszystkich
    `stem[0] = Conv1d(12, 64, 5)`.
    Skutek: model nie ma ZADNEJ strukturalnej symetrii nici, nawet w pierwszej
    warstwie. Zmierzona luka: sr|f(x) - f(rc(x))| = 0,3104 przy sd predykcji
    1,1491 (Dev), czyli 27 % rozrzutu; korelacja f(x) z f(rc(x)) tylko 0,9415.
    RC-TTA odzyskuje z tego +0,0235 na Dev -- to jest miara marnowanego sygnalu.

    DLACZEGO ROZDZIELENIE, A NIE WIAZANIE CALOSCI. Wiazanie filtrow po kanalach
    motywowych bylo juz probowane i ZMIERZONE JAKO NIEUDANE (patrz docstring
    `RCConv1d`): obsadzenie motywu jest niezmiennicze CO DO WARTOSCI, ale nie
    CO DO INDEKSOWANIA -- skaner dopelnia tylko z prawej, a motywy maja rozne
    szerokosci, wiec kazdy kanal przesuwa sie o wlasne `w-1` i zadne jedno
    odbicie ich nie wyrownuje. Zmierzone |scanner(rc(x)) - reverse(scanner(x))|
    = 8,75e-01, czyli tyle co kontrola bez odbicia.

    Stad podzial: pierwsze 4 kanaly (one-hot, gdzie komplement JEST okreslony)
    ida przez `RCConv1d`, reszta przez swobodny `Conv1d`, a wyniki sie sumuja.
    Sciezka sekwencyjna staje sie RC-ekwiwariantna Z KONSTRUKCJI; motywowa nie.

    UCZCIWIE O ZAKRESIE: to jest naprawa CZESCIOWA. Calosc stemu nie bedzie
    RC-ekwiwariantna, dopoki kanaly motywowe nie zostana wyrownane centralnie.
    Falsyfikator w logu: `sonda_rc.py` musi pokazac SPADEK sr|f(x)-f(rc(x))|
    wobec 0,3104; brak spadku znaczy, ze skladowa motywowa dominuje i sama
    sciezka nukleotydowa nie wystarcza.
    """

    def __init__(self, out_channels, extra_ch, kernel_size=5, bias=True):
        super().__init__()
        self.rc = RCConv1d(out_channels // 2, kernel_size, bias=bias)
        self.extra_ch = int(extra_ch)
        self.dod = (nn.Conv1d(self.extra_ch, out_channels, kernel_size,
                              padding=kernel_size // 2, bias=False)
                    if self.extra_ch > 0 else None)
        if self.dod is not None:
            # Zero-init: start BIT W BIT jak sam `RCConv1d` na nukleotydach,
            # wiec galaz motywowa wchodzi dopiero, gdy sie czegos nauczy.
            nn.init.zeros_(self.dod.weight)

    def forward(self, x):
        y = self.rc(x[:, :4])
        if self.dod is not None and x.shape[1] > 4:
            y = y + self.dod(x[:, 4:4 + self.extra_ch])
        return y


class RCMaxBankWieloskalowy(nn.Module):
    """Bank filtrow de novo o ROZNYCH szerokosciach, niciowo niezmienniczy.

    DLACZEGO, ZMIERZONE: filtry swobodne wnosza do kanalu motywowego
    **+0,61092**, a oblozenie 296 profili JASPAR **+0,28005** -- czyli
    32 filtry jednej szerokosci daja ponad dwa razy wiecej niz caly
    wyselekcjonowany slownik. A szerokosc byla zaszyta: `RCMaxConv1d(n, 15)`.
    15 jest zla w OBIE strony: rdzenie GATA/ETS maja 6-8 bp (filtr marnuje
    7-9 pozycji), a `max_len` JASPAR to 21 bp (najszersze sie nie mieszcza).

    Konkatenacja bankow DOKLADNIE ekwiwariantnych jest dokladnie
    ekwiwariantna, wiec `f(rc x) = flip(f(x))` zachodzi dla calego banku --
    ale KAZDE jadro musi byc NIEPARZYSTE, bo `RCMaxConv1d` opiera dowod na
    `padding = k//2` wyrownanym do srodka okna. Straz nizej to wymusza,
    zamiast pozwolic na cichy dryf o pol pozycji.
    """

    def __init__(self, out_na_skale, jadra=(7, 11, 15, 21)):
        super().__init__()
        jadra = [int(k) for k in jadra]
        zle = [k for k in jadra if k % 2 == 0 or k < 3]
        if zle:
            raise ValueError(
                "RCMaxBankWieloskalowy wymaga jader NIEPARZYSTYCH >= 3 "
                "(dowod ekwiwariancji opiera sie na padding = k//2 "
                "wyrownanym do srodka okna); zle: %s" % zle)
        self.jadra = tuple(jadra)
        self.out_na_skale = int(out_na_skale)
        self.banki = nn.ModuleList(
            RCMaxConv1d(self.out_na_skale, k) for k in self.jadra)

    @property
    def out_channels(self):
        return self.out_na_skale * len(self.jadra)

    def forward(self, x):
        return torch.cat([b(x) for b in self.banki], dim=1)

    def extra_repr(self):
        return ("jadra=%r, %d na skale, razem %d kanalow"
                % (self.jadra, self.out_na_skale, self.out_channels))


class RCMaxConv1d(nn.Module):
    """Bank swobodnych filtrow NIEZMIENNICZY niciowo, wyrownany do srodka okna.

    `RCConv1d` daje PARE kanalow (W i rc(W)) i zostawia zlozenie sieci: kanal
    ``i`` i ``i + out`` sa parą RC, wiec operator RC na jej wyjsciu to zamiana
    polowek plus flip. To wystarcza dla stemu, ktory idzie do enkodera
    usredniajacego po nici -- ale NIE dla toru, ktory ma byc niezmienniczy sam
    w sobie, jak strumienie motywowe wchodzace do latentu.

    Tutaj obie nici sa skladane MAKSIMUM, wiec wyjscie jest jednym bankiem o
    ``out`` kanalach i spelnia
        f(rc x) = flip(f(x))
    DOKLADNIE. Wyprowadzenie: dla jadra nieparzystego z ``padding = k//2``
    indeks wyjscia ``t`` to okno ``[t - k//2, t + k//2]``, czyli jest juz
    wyrownany do SRODKA, a ``conv_W(rc x)[t] = conv_rc(W)(x)[L-1-t]``. Maksimum
    po parze jest wiec przenoszone przez `flip` bez reszty.

    Dlaczego to jest wazne akurat tu: dotad `learned` i `dev_filters`/
    `hk_filters` byly zwyklymi `Conv1d` na one-hot, wiec KAZDY tor, ktory je
    zawiera, byl nieekwiwariantny niciowo bez wzgledu na to, jak dobrze
    wyrownane jest oblozenie PWM. Skaner naprawiony w pojedynke nie kupuje
    dokladnosci, bo te filtry siedza w tym samym wektorze wejsciowym.

    Jadro MUSI byc nieparzyste (inaczej srodek okna nie istnieje) i tak jest w
    projekcie: `learned` ma 15, `task_filter_width` 15.

    Bez nowych parametrow wobec `Conv1d(4, out, k)`: druga nic to ta sama
    macierz wag przeczytana odwrotnie.
    """

    def __init__(self, out_channels, kernel_size, bias=True):
        super().__init__()
        k = int(kernel_size)
        if k % 2 == 0:
            raise ValueError("RCMaxConv1d wymaga jadra nieparzystego, jest %d" % k)
        self.out_channels = int(out_channels)
        self.kernel_size = k
        self.padding = k // 2
        self.weight = nn.Parameter(
            torch.randn(self.out_channels, 4, k) * (1.0 / (4 * k) ** 0.5))
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.register_buffer("perm", torch.tensor(COMPLEMENT_PERMUTATION),
                             persistent=False)

    def rc_weight(self):
        # Odwrocenie osi jadra ORAZ komplementacja osi zasad -- jedno bez
        # drugiego daje odwrotnosc, nie odwrotny komplement.
        return self.weight.flip(-1)[:, self.perm]

    def forward(self, x):
        w = torch.cat([self.weight, self.rc_weight()], dim=0)
        b = torch.cat([self.bias, self.bias]) if self.bias is not None else None
        y = F.conv1d(x, w, b, padding=self.padding)
        n = self.out_channels
        return torch.maximum(y[:, :n], y[:, n:])


class ParaAtencja(nn.Module):
    """Sprzezenie PAROWE po pozycjach: "reszta i oddzialuje z reszta j".

    POWOD, ZMIERZONY W TYM REPO. Zaleznosc od wszystkich 249 zasad zachodzi dzis
    trzema drogami -- bramka SE w kazdym bloku (**504 560 param. = 22,5 %
    modelu**), pula globalna w glowicy, FiLM z wektora grafu (288 960 param.) --
    ale wszystkie sa **permutacyjnie niezmiennicze po pozycjach** i moduluja
    **ranga 1** po kanale. Zadna nie wyraza konkretnej pary. To model
    billboardowy z konstrukcji.

    Zasiegu przy tym NIE brakuje: par o |i-j| > 202 jest 2 162 z 61 752, czyli
    3,5 %. Brakuje wlasnie **sprzezenia parowego, rozdzielczego pozycyjnie
    i zaleznego od tresci**.

    ``bias`` jest indeksowany przez ``|i-j|``, wiec jest PARZYSTY w przesunieciu
    z konstrukcji. To nie jest ozdoba: pod odwrotnym komplementem pozycje sie
    odwracaja, a czlon nieparzysty zlamalby symetrie, ktorej ten model pilnuje
    (RC-tied stem, RC-TTA warte +0,0085).

    Modul jest pomyslany jako ZAMIENNIK bramki SE przy ~stalej liczbie wag, zeby
    eksperyment mierzyl RODZAJ sprzezenia, a nie pojemnosc -- w tym projekcie
    kazdy wariant z wieksza pojemnoscia sie przeuczal (gap +0,05..+0,18).
    """

    def __init__(self, dim, heads=4, dim_head=32, max_len=256, dropout=0.0,
                 skala_galezi=0.1):
        super().__init__()
        self.h = int(heads)
        self.dk = int(dim_head)
        wew = self.h * self.dk
        self.q = nn.Conv1d(dim, wew, 1, bias=False)
        self.k = nn.Conv1d(dim, wew, 1, bias=False)
        self.v = nn.Conv1d(dim, wew, 1, bias=False)
        self.o = nn.Conv1d(wew, dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.h, int(max_len)))
        poz = torch.arange(int(max_len))
        self.register_buffer("_sep", (poz.view(-1, 1) - poz.view(1, -1)).abs(),
                             persistent=False)
        self.drop = nn.Dropout(dropout)
        # ---------------------------------------------------------------
        # SKALARNEJ BRAMY NIE MA, I TO JEST POPRAWKA ZMIERZONA.
        #
        # Poprzednio stalo tu `self.brama = nn.Parameter(tensor(0.05))`
        # z komentarzem "0,05, NIE zero, bo zerowa brama nie przepuszcza
        # gradientu w gore". Diagnoza byla trafna, lekarstwo nie: start
        # 0,05 naprawia INICJALIZACJE, a nie DYNAMIKE. Zmierzone na
        # HypPara2 (2026-10-11): `para_brama` 0,05 -> 0,0011 (ep1) ->
        # 0,0002 (ep2) -> -0,0004 (ep3). Przy mnozniku ~0 gradient do
        # CALEJ galezi jest ~0, wiec zero jest stanem POCHLANIAJACYM —
        # i to wyjasnia zapisane "para_brama ~1e-4 przez wszystkie
        # 21 epok" w poprzednim biegu. Sprzezenie parowe nie bylo wiec
        # nigdy przetestowane: trzy podejscia, trzy razy ta sama smierc.
        # Weight decay jest wykluczony — `brama` jest w `_WZORCE_BEZ_DECAY`.
        #
        # Lekarstwo jest w tym repo wyprowadzone przy wycofaniu
        # `film_init_std`: **zerowa WAGA nie glodzi galezi**, bo gradient
        # po wadze zalezy od WEJSCIA, nie od wagi (`film.0.weight` mial
        # gradient 5,21 przy W = 0). Glodzi wylacznie zerowy MNOZNIK
        # SKALARNY, bo ten mnozy caly strumien gradientu.
        #
        # Wiec: `o` startuje ZEREM (galaz nie wstrzykuje szumu na starcie,
        # czyli nie ma za co dostac kary), a skali uczy sie `o` samo —
        # bez stanu pochlaniajacego. To jest zapisana regula "wygrzewac
        # ZRODLO, nie SKALE".
        # ---------------------------------------------------------------
        # DRUGA POPRAWKA, po drugiej zmierzonej awarii. Pierwsza wersja
        # miala uczony mnoznik skalarny i on ZAPADAL sie do zera
        # (0,05 -> -0,0004 w trzech epokach), co jest stanem pochlaniajacym.
        # Druga miala `o` zerowane i ZADNEGO ograniczenia skali — galaz
        # eksplodowala: `para_wkl` = 1,399, czyli wyjscie galezi 1,4x
        # WIEKSZE od wejscia bloku, przy `para_H` = 0,0 (atencja zapadnieta
        # w delte) i PCC ep1 0,5246 wobec 0,6220 bazy.
        #
        # Mechanizm eksplozji jest arytmetyczny: wpiecie to czysty dodatek
        # rezydualny `y = y + para(y)` bez normalizacji, a AdamW przesuwa
        # wage o ~lr na krok NIEZALEZNIE od wielkosci gradientu — przy
        # lr 1e-3 i ~4200 krokach epoki waga startujaca z zera dochodzi
        # dokad chce.
        #
        # Wniosek, ktory z tych dwoch awarii zostaje: KAZDY uczony mnoznik
        # skalarny na galezi ma pochlaniajace zero, a brak jakiegokolwiek
        # ograniczenia daje eksplozje. Wiec skala NIE moze byc uczona ani
        # nieograniczona — galaz ma sie uczyc CZEGO dodac, nie ILE.
        #
        # `norma_wy` sprowadza wyjscie do RMS 1 (bez afinicznosci, wiec
        # nie da sie jej obejsc), a `skala` jest BUFOREM, nie parametrem.
        # Brak uczonego mnoznika => brak stanu pochlaniajacego; stala
        # normalizacja => brak eksplozji. `o` wraca do domyslnej
        # inicjalizacji, bo przy normalizacji zerowe `o` dawaloby 0/sqrt(eps)
        # i gradient rzedu 1/sqrt(eps) na pierwszym kroku.
        # TRZECIA POPRAWKA, po trzeciej zmierzonej awarii. v3 miala `skala`
        # BEZWZGLEDNA (wyjscie o RMS 0,1) i galaz wyszla WYCISZONA: RMS
        # strumienia rezydualnego urosl z 0,558 do ~12,6 przez dwie epoki,
        # wiec stale 0,1 to bylo 0,8% strumienia zamiast 10%. Przy tym
        # wnetrze galezi dryfowalo (`para_sygnal` 7,8 -> 20,5, surowe
        # wyjscie RMS ~258), bo normalizacja na WYJSCIU czyni skale `o`
        # nieidentyfikowalna, a atencja zapadla sie w delte (`para_H`
        # 0,002 przy sufcie 7,96).
        #
        # Stad dwie rzeczy naraz:
        #  * `norma_we` -- atencja widzi wejscie o RMS 1, wiec jej wnetrze
        #    nie dryfuje za rosnacym strumieniem;
        #  * wyjscie skalowane WZGLEDEM strumienia (`* rms(x)`), nie
        #    bezwzglednie, wiec udzial galezi jest `skala` NIEZALEZNIE od
        #    tego, jak strumien urosnie.
        #
        # Obie normalizacje sa PER PROBKA (GroupNorm z jedna grupa liczy po
        # C i L), a `rms` tez jest per probka — wiec wlasnosc "wejsciem jest
        # JEDNA sekwencja" (zmierzona: F2/F4 = 0,000e+00) zostaje nietknieta.
        self.norma_we = nn.GroupNorm(1, dim, affine=False)
        self.norma_wy = nn.GroupNorm(1, dim, affine=False)
        self.register_buffer("skala", torch.tensor(float(skala_galezi)),
                             persistent=False)
        self._ostatnia_entropia = None
        self._wklad = None
        self._sygnal = None

    def forward(self, x):
        B, C, L = x.shape
        xn = self.norma_we(x)
        q = self.q(xn).view(B, self.h, self.dk, L).transpose(-1, -2)  # [B,h,L,dk]
        k = self.k(xn).view(B, self.h, self.dk, L).transpose(-1, -2)
        v = self.v(xn).view(B, self.h, self.dk, L).transpose(-1, -2)
        logity = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)       # [B,h,L,L]
        logity = logity + self.bias[:, self._sep[:L, :L]].unsqueeze(0)
        w = torch.softmax(logity, dim=-1)
        # ENTROPIA LICZONA TAKZE W TRENINGU, i to nie jest kosmetyka.
        # Poprzednio stalo tu `if not self.training`, a `train_graph.py`
        # bierze migawke `band_report` PRZED `evaluate`, czyli w trybie
        # train -- wiec `para_H`, `para_H_max` i `para_brama` nie trafily
        # do logu ANI RAZ. To jest zapisana w tym repo cisza HypPara
        # "ani jednej linii w 29 epokach", i jej przyczyna to interakcja
        # DWOCH POPRAWNYCH poprawek: migawka musi byc treningowa, bo progi
        # `dr_*` / `dom_kontekst` / `rzad_ef` sa kalibracjami treningowymi,
        # a ta obserwabla istniala wylacznie w eval.
        #
        # `para_H` wobec `para_H_max = log2(L)` jest JEDYNYM falsyfikatorem
        # sprzezenia parowego: na sufcie atencja jest jednostajna, czyli
        # robi dokladnie to, co zastapiona przez nia srednia globalna SE --
        # i wtedy wymiana jest zerem, a nie porazka pomyslu.
        #
        # Koszt jest znikomy i policzony, nie oszacowany: `w` jest JUZ
        # zmaterializowane, wiec to jedna operacja elementowa plus redukcja
        # po [B,h,L,L]. Przy B=96, h=4, L=124 to 5,9 mln elementow wobec
        # 189 mln MAC samego `q @ k` tego bloku, czyli ~3% jego kosztu.
        # Rezim ma znaczenie: w treningu dropout jest aktywny, wiec to NIE
        # jest ta sama liczba co w eval -- i wlasnie dlatego ma byc czytana
        # tam, gdzie kalibruje sie prog.
        # PRZECHOWYWANY JEST TENSOR, NIE `float`. `float(e)` wymusilby
        # synchronizacje GPU->CPU na KAZDYM kroku i na kazdym bloku
        # parowym, czyli ~8 tys. punktow synchronizacji na epoke w gorącej
        # petli. Konwersja nalezy do `band_report`, ktory czyta to RAZ na
        # epoke. Zapisana w tym repo reguła brzmi, ze obserwabla moze zabic
        # przebieg wprzod — a najtanszym sposobem, zeby tego nie sprawdzac
        # pomiarem, jest nie wkladac do petli niczego, co czeka na GPU.
        with torch.no_grad():
            e = -(w * (w + 1e-12).log2()).sum(-1).mean()
            self._ostatnia_entropia = (e.detach(), math.log2(L))
        y = (w @ v).transpose(-1, -2).reshape(B, self.h * self.dk, L)
        surowe = self.o(y)
        # RMS PER PROBKA, nie po calym batchu — inaczej wyjscie zalezaloby
        # od towarzystwa w batchu i zlamalaby sie wlasnosc "wejsciem jest
        # jedna sekwencja", zmierzona jako F2/F4 = 0,000e+00.
        rms_x = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
        # DROPOUT PRZED NORMALIZACJA, nie po. Po normalizacji podnosil RMS
        # o 1/sqrt(1-p) = 1,118 przy p = 0,2 (zmierzone: `para_wkl` 0,1118
        # zamiast 0,1000, stabilnie przez 20 krokow — przesuniecie, nie
        # dryf). Wewnatrz: zerowanie dalej regularyzuje, a normalizacja
        # zdejmuje jego wplyw na SKALE, wiec udzial galezi jest dokladnie
        # `skala` i w treningu, i w eval. Bez tego falsyfikator musialby
        # znac `p`, czyli prog zalezalby od configu.
        wy = self.skala * rms_x * self.norma_wy(self.drop(surowe))
        # DWIE OBSERWABLE, bo jedna nie wystarczyla. `para_wkl` to udzial
        # galezi w strumieniu — po normalizacji jest z KONSTRUKCJI ~`skala`,
        # wiec sluzy jako straz, ze ograniczenie realnie trzyma (eksplozja
        # do 1,399 bylaby widoczna natychmiast). `para_sygnal` to RMS
        # wyjscia PRZED normalizacja, wzgledem wejscia — i to ono mowi, czy
        # galaz uczy sie czegokolwiek, bo normalizacja ukrywa jej wlasna
        # skale. Tensory, nie `float`: konwersja w `band_report`, raz na
        # epoke, zeby nie synchronizowac GPU na kazdym kroku.
        with torch.no_grad():
            we = x.pow(2).mean().sqrt().clamp_min(1e-8)
            self._wklad = (wy.pow(2).mean().sqrt() / we).detach()
            # `para_sygnal` jest teraz liczone wobec wejscia PO normalizacji
            # (RMS 1), wiec mierzy wzmocnienie SAMEJ galezi, a nie dryf
            # strumienia. W v3 liczylem je wobec surowego `x` i wartosc 20,5
            # mieszala dwie rzeczy: rosniecie `o` i rosniecie strumienia.
            self._sygnal = surowe.pow(2).mean().sqrt().detach()
        return wy


class MultiScaleDilatedBlock(nn.Module):
    """Parallel dilated branches with squeeze-excitation, residual-concatenated.

    Keeps LegNet's inverted-residual and SE structure — both measured to work —
    but replaces the single kernel with branches at several dilations, so one
    block sees motif-width and motif-spacing structure at the same time and at
    full resolution.  Dilations default to ``(1, 2, 4)``: with kernel 5 that
    spans 5, 9 and 17 bases, covering a JASPAR profile up to roughly one
    helical turn of spacing.
    """

    def __init__(self, dim, kernel_size=5, dilations=(1, 2, 4), expand=2,
                 se_reduction=4, dropout=0.1, para_atencja=False, bez_se=False,
                 para_zamiast_se=True, para_heads=4, para_dim_head=32,
                 ekstrema=False, ekstrema_ranga=74, se_stat="mean",
                 brama_glu=False, brama_glu_z_se=False, brama_glu_ranga=0):
        super().__init__()
        inner = dim * expand
        per = inner // len(dilations)
        inner = per * len(dilations)                # keep the split exact
        self.norm = nn.BatchNorm1d(dim)
        # ============ BRAMA GLU/SwiGLU ZAMIAST SE ============
        # SE jest brama RANGI 1: jeden skalar na kanal na CALE 249 bp,
        # a zmierzone jest, ze 83-93 % tego, co produkuje, jest STALE
        # miedzy sekwencjami -- czyli SE jest wyuczona maska kanalow,
        # nie modulacja. Dla DNA to zle obciazenie: "motyw A dziala tylko,
        # gdy obok NIE MA motywu B" jest interakcja MULTIPLIKATYWNA
        # i LOKALNA, a splot + SiLU jej nie daje wprost.
        #
        # GLU:  Y = branches(X W1) * SiLU(X W2)
        # Bramkowanie jest PER ELEMENT [B, C, L] -- zadnego uśredniania
        # po pozycjach. To ta sama konstrukcja, ktora w Llama/PaLM i w
        # modelach DNA wypara SE oraz zwykle MLP.
        #
        # BUDZET: `expand` rosnie z (dim -> inner) na (dim -> 2*inner),
        # ale SE ZNIKA. Netto dla trzonu (dim 160, inner 320, mean_max_sd):
        # +51 520 - 102 800 = **-51 280 na blok**. Zmiana jest wiec
        # TANSZA od SE, a nie drozsza -- co jest wazne metodycznie:
        # ewentualnej wygranej nie da sie przypisac pojemnosci.
        #
        # START = BRAMA 1,0 DOKLADNIE. Polowa `expand` odpowiadajaca bramie
        # ma wagi wyzerowane i bias 1,27846, bo SiLU(1,27846) = 1,0. Zimny
        # SwiGLU (bias 0) dawalby SiLU(0) = 0 i dusil blok w pierwszych
        # krokach; w tym repo skok bramy 0,02 -> 1,0 kosztowal juz -0,44 PCC.
        self.brama_glu = bool(brama_glu)
        self.brama_glu_ranga = int(brama_glu_ranga or 0)
        _glu_pelna = self.brama_glu and self.brama_glu_ranga <= 0
        self.expand = nn.Conv1d(dim, (2 * inner) if _glu_pelna else inner, 1)
        if _glu_pelna:
            with torch.no_grad():
                self.expand.weight[inner:].zero_()
                self.expand.bias[inner:].fill_(1.2784645)   # SiLU(x) = 1.0
        self.brama_g = None
        if self.brama_glu and self.brama_glu_ranga > 0:
            # BRAMA NISKIEGO RZEDU. Pelna brama podwaja wyjscie `expand`,
            # czyli kosztuje `dim*inner` na blok -- przy trzonie 224 i
            # inner 448 to 100 800 wag na blok, +22 % calego modelu przy
            # pieciu blokach. Za duzo: repo ma zmierzone, ze KAZDY wariant
            # z pelnym trzonem 4,1 M przeuczal sie (gap +0,05 ... +0,18),
            # a HypStozek ma gap juz +0,054.
            #
            # DLACZEGO NISKI RZAD JEST UZASADNIONY, a nie tylko tanszy:
            # zmierzone w tym repo, ze wariancja bramy SE to 83-93 %
            # MIEDZY KANALAMI, czyli brama jest w duzej czesci stala maska
            # kanalow. Brama o randze `r` reprezentuje dokladnie taka
            # strukture i nie moze reprezentowac tylko tego, czego pomiar
            # nie wykazal.
            #
            # START = 1,0 DOKLADNIE, tak samo jak brama pelna: druga warstwa
            # ma wagi wyzerowane i bias 1,27846, wiec SiLU(bias) = 1,0
            # NIEZALEZNIE od wejscia. Zadnej rampy -- wygrzewa sie ZRODLO,
            # nie SKALA.
            _r = self.brama_glu_ranga
            self.brama_g = nn.Sequential(
                nn.Conv1d(dim, _r, 1), nn.SiLU(), nn.Conv1d(_r, inner, 1))
            with torch.no_grad():
                self.brama_g[-1].weight.zero_()
                self.brama_g[-1].bias.fill_(1.2784645)
        self.branches = nn.ModuleList(
            nn.Conv1d(per, per, kernel_size, padding=d * (kernel_size // 2),
                      dilation=d, groups=per)
            for d in dilations)
        self.per = per
        squeezed = max(inner // se_reduction, 8)
        # ============ STATYSTYKA SQUEEZE ============
        # `mean` to zachowanie dotychczasowe, BIT W BIT. `mean_sd` doklada
        # ODCHYLENIE po pozycjach.
        #
        # ZMIERZONE (`sonda_se.py` na HypEncNapr_best, 4 032 sekwencje, sonda
        # liniowa Dev/Hk z faktycznego wejscia bramy, zysk SUROWY nad `mean`):
        #     blok        max        sd        mean+sd
        #     L0       -0,0559   +0,0291     +0,0672
        #     L1       -0,0365   +0,0265     +0,0285
        #     L2       -0,0334   +0,0071     +0,0135
        #     L3       -0,0429   -0,0081     -0,0036
        #     L4       -0,0258   -0,0019     +0,0025
        # KONTROLA POJEMNOSCI (te same cechy przetasowane po sekwencjach) jest
        # UJEMNA (-0,011 .. -0,030), czyli dodanie parametrow POGARSZA sonde --
        # pojemnosc niczego nie zawyza, a cechy informatywne musialy te kare
        # pokonac. Netto: +0,0976 / +0,0565 / +0,0372 / +0,0136 / +0,0135.
        #
        # SAMO MAKSIMUM JEST GORSZE NA KAZDYM BLOKU. Liczba 81x (srednia 0,004
        # wobec maksimum 0,329) dotyczyla OBLOZENIA PWM -- sygnalu rzadkiego
        # i ekstremalnego -- a nie mapy trzonu, ktora po BatchNorm+SiLU jest
        # gesta i wysrodkowana. Dlatego `mean_sd`, nie `mean_max`.
        #
        # Zysk jest skupiony na blokach WCZESNYCH, wiec `se_stat` warto wlaczac
        # tylko tam -- stad sterowanie per blok w `hyp_vq_net.py`.
        # TRZY OPCJE. `mean_max_sd` dodane po pomiarze na `HypFinal2_best`
        # (sonda liniowa out-of-sample z faktycznego wejscia bramy, L0),
        # gdzie NETTO po odjeciu kontroli pojemnosci wychodzi:
        #     sr+max      +0,00254
        #     sr+sd       +0,00563   (obecne `mean_sd`)
        #     sr+max+sd   **+0,01000**
        # Kontrola jest UJEMNA -- dodanie drugiej, przetasowanej kopii
        # sredniej kosztuje -0,0062 przy stalym lambda (0,70826 wobec
        # 0,71447), a trzeciej -0,0194 (0,69506). Dlatego SUROWE liczby
        # daja odwrotny wniosek (czysta srednia jest najwyzsza, 0,71447)
        # i trzeba czytac NETTO, nie surowe.
        self.se_stat = str(se_stat)
        _DOZW = {"mean": 1, "mean_sd": 2, "mean_max_sd": 3}
        if self.se_stat not in _DOZW:
            raise ValueError("se_stat musi byc jednym z %s, dostano %r"
                             % (sorted(_DOZW), se_stat))
        _we_se = inner * _DOZW[self.se_stat]
        self.se = nn.Sequential(
            nn.Linear(_we_se, squeezed), nn.SiLU(),
            nn.Linear(squeezed, inner), nn.Sigmoid())
        self.project = nn.Conv1d(inner, dim, 1)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.para_zamiast_se = bool(para_zamiast_se)
        self.para = (ParaAtencja(dim, para_heads, para_dim_head, dropout=dropout)
                     if para_atencja else None)
        if ekstrema:
            # ZAMIENNIK SE dostosowany do sekwencji regulatorowej: agregacja
            # miekkim maksimum z uczona temperatura zamiast sredniej po
            # pozycjach, plus jawnie oddzielona stala maska kanalow.
            # ZMIERZONE, czemu: obłozenie 20 PWM po SREDNIEJ daje 0,004,
            # a po MAKSIMUM 0,329 (sonda liniowa out-of-sample); a rozklad
            # wariancji bramy SE to 83-93 % MIEDZY KANALAMI, czyli stala maska.
            # Budzet dobrany tak, by zgadzal sie z SE co do 1 % -- eksperyment
            # ma mierzyc RODZAJ agregacji, nie pojemnosc.
            from .blok_ekstremow import BlokEkstremow
            self.se = BlokEkstremow(inner, ranga=int(ekstrema_ranga))
            self.ekstrema = True
        else:
            self.ekstrema = False
        self.brama_glu_z_se = bool(brama_glu_z_se) and bool(brama_glu)
        if self.brama_glu and not self.brama_glu_z_se:
            # WYMIANA, nie dolozenie. Zostawienie SE obok GLU sprawiloby,
            # ze bieg mierzy "GLU + SE" wobec "SE", a nie RODZAJ bramy.
            #
            # `brama_glu_z_se` zdejmuje to WYLACZNIE na zadanie. Domyslnie
            # False, wiec kazdy istniejacy config -- w tym dwanascie biegow
            # z tabeli -- buduje sie BIT W BIT jak dotad.
            self.se = None
        elif self.brama_glu_z_se and not getattr(self, "ekstrema", False):
            # ZLOZENIE ORTOGONALNYCH BRAM. GLU jest per POZYCJA (brama `_g`
            # z wejscia bloku, splot 1x1, zasieg lokalny); SE jest per KANAL
            # ze statystyki po WSZYSTKICH pozycjach, czyli rangi 1 po pozycji.
            # Kazda robi to, czego druga nie umie, wiec iloczyn ma pelna
            # rozdzielczosc pozycyjna I globalny kontekst kanalowy.
            #
            # AMPLITUDA NA STARCIE. `se` konczy sie `Sigmoid`, czyli ~0,5 przy
            # zerowym biasie -- dolozenie jej nad GLU POLOWILOBY galaz
            # rezydualna. Repo ma zmierzone, ze skok bramy 0,02 -> 1,0
            # kosztowal -0,44 PCC, a regula brzmi: wygrzewac ZRODLO, nie
            # SKALE. Wiec zadnej rampy -- brama startuje OTWARTA
            # (sigmoid(2,5) = 0,924) i gradient ja zamyka, jesli ma zamknac.
            with torch.no_grad():
                self.se[-2].bias.fill_(2.5)
        if bez_se:
            # Zdjecie bramki SE bez dokladania niczego w zamian --
            # uzywane przez HypGram, ktory zastepuje cala maszynerie
            # rangi 1 sprzezeniem parowym po slotach.
            self.se = None
        if self.para is not None and self.para_zamiast_se:
            # WYMIANA, nie dolozenie: SE ma 100 912 wag na blok, atencja parowa
            # przy 4 glowicach po 32 ma ~115 tys. Przy wymianie liczba wag
            # zostaje prawie ta sama, wiec porownanie mierzy RODZAJ sprzezenia.
            self.se = None

    def _brama_se(self, h):
        """Brama SE, JEDNO zrodlo prawdy dla obu galezi `forward`.

        Wydzielone, bo obie galezie (SE i GLU+SE) musza uzywac dokladnie tej
        samej statystyki. Zduplikowanie jej to wzorzec, ktory w tym repo
        zawiodl trzykrotnie: `decode_codes` z zaszytym `nearest` wobec
        `linear` w `forward`, kanonizacja RC wpieta tylko w `embed`, i
        `_codes_input` liczacy oblozenie drugi raz. Za kazdym razem dwa
        wejscia, jedno zalatane -- i rozjazd byl CICHY.
        """
        if self.se is None:
            return h
        # `BlokEkstremow` sam agreguje po pozycjach (potrzebuje calej mapy),
        # wiec dostaje `h`, a nie `h.mean(-1)`, i sam mnozy przez brame.
        if getattr(self, "ekstrema", False):
            return self.se(h)
        _ss = getattr(self, "se_stat", "mean")
        if _ss == "mean_max_sd":
            _st = torch.cat([h.mean(dim=-1), h.amax(dim=-1),
                             h.std(dim=-1)], dim=-1)
        elif _ss == "mean_sd":
            _st = torch.cat([h.mean(dim=-1), h.std(dim=-1)], dim=-1)
        else:
            _st = h.mean(dim=-1)
        return h * self.se(_st).unsqueeze(-1)

    def forward(self, x):
        if self.brama_glu:
            _n = self.norm(x)
            if self.brama_g is None:
                _hb = self.expand(_n)
                h, _g = _hb.chunk(2, dim=1)
            else:
                # Brama niskiego rzedu liczona z TEJ SAMEJ znormalizowanej
                # mapy co sygnal -- `norm` wolane raz, bo to BatchNorm
                # i drugie wywolanie przepisaloby `running_*` dwukrotnie
                # na krok. Ta sama pulapka, ktora byla w `_codes_input`
                # liczacym oblozenie po raz drugi.
                h = self.expand(_n)
                _g = self.brama_g(_n)
            h = self.act(h)
            h = torch.cat([br(part) for br, part in
                           zip(self.branches, h.split(self.per, dim=1))], dim=1)
            # BRAMA PER ELEMENT -- ksztalt [B, C, L], bez uśredniania.
            h = self.act(h) * self.act(_g)
            # SE PO GLU, nie przed. Statystyka ma opisywac to, co REALNIE
            # propaguje dalej -- liczona przed pomnozeniem przez `_g`
            # opisywalaby mape, ktora GLU zaraz wytlumi. Przy
            # `brama_glu_z_se: false` (domyslnie) `self.se` jest None
            # i to wywolanie jest tozsamoscia.
            h = self._brama_se(h)
            y = x + self.drop(self.project(h))
            if self.para is not None:
                y = y + self.para(y)
            return y
        h = self.act(self.expand(self.norm(x)))
        h = torch.cat([br(part) for br, part in
                       zip(self.branches, h.split(self.per, dim=1))], dim=1)
        h = self.act(h)
        h = self._brama_se(h)
        y = x + self.drop(self.project(h))
        if self.para is not None:
            y = y + self.para(y)
        return y


class MultiLevelVQ(nn.Module):
    """N-level hierarchical VQ, each level quantising what the ones above missed.

    Two levels separate "regional context" from "local detail".  Three separate
    base-level detail, motif-scale structure, and whole-enhancer context — which
    matches how the biology is organised, and gives an editing handle at each
    scale: touch the finest level for a point mutation, a coarser one to move a
    whole element.

    Each level is quantised on the *residual* left by the level above, so the
    codebooks specialise instead of redundantly re-encoding the same signal.
    """

    def __init__(self, in_ch=4, dim=64, strides=(2, 2, 4), codes=(512, 256, 128),
                 commit_weight=0.25, rc_stem=True, top_down=True):
        super().__init__()
        from .hier_vq import VQLayer
        assert len(strides) == len(codes), "one codebook size per level"
        self.strides = tuple(int(s) for s in strides)
        self.n_levels = len(strides)
        self.top_down = bool(top_down)

        # RC-tied first layer: strand symmetry is structural at the base level
        # rather than learned.  Emits 2*(dim//2) channels so the width matches.
        if rc_stem and in_ch == 4:
            self.stem = nn.Sequential(
                RCConv1d(dim // 2, 5), nn.BatchNorm1d(dim), nn.SiLU())
        else:
            self.stem = nn.Sequential(
                nn.Conv1d(in_ch, dim, 5, padding=2), nn.BatchNorm1d(dim), nn.SiLU())
        self.downs = nn.ModuleList()
        for s in self.strides:
            self.downs.append(nn.Sequential(
                nn.Conv1d(dim, dim, 2 * s, stride=s, padding=s // 2),
                nn.BatchNorm1d(dim), nn.SiLU()))
        self.vqs = nn.ModuleList(VQLayer(dim, c, commit_weight=commit_weight) for c in codes)
        self.dim = dim

    def forward(self, x):
        h = self.stem(x)
        feats = []
        for down in self.downs:
            h = down(h)
            feats.append(h)             # feats[0] finest ... feats[-1] coarsest

        quants = [None] * self.n_levels
        indices = [None] * self.n_levels
        loss = None

        if self.top_down:
            # VQ-VAE-2 ordering: quantise the COARSEST level first from a clean
            # signal, then let each finer level encode only what the coarser
            # reconstruction missed.  Bottom-up ordering was measured to invert
            # the hierarchy -- the finest level explained 75.6% of the variance
            # while the coarsest explained 22.0% of an increasingly noisy
            # residual, i.e. the coarse codebooks were quantising noise.
            cur = feats[-1]
            for li in range(self.n_levels - 1, -1, -1):
                q, idx, l = self.vqs[li](cur)
                quants[li], indices[li] = q, idx
                loss = l if loss is None else loss + l
                if li > 0:
                    up = F.interpolate(q.detach(), size=feats[li - 1].shape[-1],
                                       mode="nearest")
                    cur = feats[li - 1] - up
        else:
            for li, vq in enumerate(self.vqs):
                q, idx, l = vq(feats[li] if li == 0 else h)
                quants[li], indices[li] = q, idx
                loss = l if loss is None else loss + l
                h = (feats[li] if li == 0 else h) - q.detach()
                if li + 1 < self.n_levels:
                    h = F.interpolate(h, size=feats[li + 1].shape[-1], mode="nearest")

        self.loss = loss
        self.indices = indices

        # Bring every level to the FINEST resolution so the trunk sees all
        # scales aligned; coarse levels are upsampled, never the reverse, so no
        # positional precision is lost.
        target = feats[0].shape[-1]
        aligned = [q if q.shape[-1] == target
                   else F.interpolate(q, size=target, mode="nearest") for q in quants]
        aligned.append(feats[0] - aligned[0].detach())     # finest residual
        return torch.cat(aligned, dim=1)

    def decode_from(self, quants_or_indices, length):
        """Reconstruct nucleotide logits from codes -- used by the
        reconstruction-only VQ-VAE and by minimal-SNP editing."""
        raise NotImplementedError("use MultiLevelVQVAE for reconstruction")

    def out_channels(self):
        return self.dim * (self.n_levels + 1)

    @torch.no_grad()
    def perplexities(self):
        return [vq.perplexity() for vq in self.vqs]
