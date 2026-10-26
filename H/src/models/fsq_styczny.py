# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn

from .fsq import FSQ
from .hyperbolic import minkowski_dot, parallel_transport0

_LOG2 = math.log(2.0)


def poziomy_dla_bitow(bity, bazy=(3, 5, 7), tolerancja=0.6, max_polowa=6):
    """Siatka FSQ o budzecie ~`bity` na pozycje, parzysta i sparowana.

    Zwraca krotke `levels` dlugosci d (parzystej), w ktorej levels[i] ==
    levels[i + d/2] -- czyli taka, ktora dopuszcza parowanie RC. Wszystkie
    bazy sa NIEPARZYSTE (siatka symetryczna, zawiera zero).

    Regula: sposrod kandydatow o |bity - cel| <= `tolerancja` bierz ten o
    NAJWIEKSZEJ liczbie wymiarow, a przy remisie o najmniejszym bledzie.
    Wiecej wymiarow to wiecej kierunkow w przestrzeni stycznej przy tym samym
    budzecie -- a to kierunki, nie rozdzielczosc wzdluz jednego z nich,
    odpowiadaja za rekonstrukcje (L2 samo 0,7927 wobec L0 samo 0,2629 przy
    podlodze skladu 0,2947). Gdy nikt nie miesci sie w tolerancji, bierzemy
    najmniejszy blad bezwzgledny.

    Wyniki dla celow z tego projektu:
        7 bitow -> (3, 3, 3, 3)          81 kodow   6,3399   (blad -0,660)
        8 bitow -> (5, 3, 5, 3)         225 kodow   7,8138   (blad -0,186)
        9 bitow -> (3, 3, 3, 3, 3, 3)   729 kodow   9,5098   (blad +0,510)
    """
    cel = float(bity)
    bazy = tuple(sorted(int(b) for b in bazy))
    if any(b % 2 == 0 or b < 3 for b in bazy):
        raise ValueError("bazy musza byc NIEPARZYSTE i >= 3 (siatka "
                         "symetryczna, z zerem): %r" % (bazy,))
    kandydaci = []

    def rozwin(prefiks, od):
        if prefiks:
            b = 2.0 * sum(math.log(l, 2.0) for l in prefiks)
            kandydaci.append((tuple(prefiks), b))
        if len(prefiks) >= max_polowa:
            return
        for i in range(od, len(bazy)):
            rozwin(prefiks + [bazy[i]], i)

    rozwin([], 0)
    bliskie = [(p, b) for p, b in kandydaci if abs(b - cel) <= tolerancja]
    if bliskie:
        # -len(p): wiecej wymiarow wygrywa, ale TYLKO wsrod kandydatow
        # miesczacych sie w tolerancji; potem mniejszy blad, potem mniejsza baza.
        p, _ = min(bliskie,
                   key=lambda pb: (-len(pb[0]), abs(pb[1] - cel), max(pb[0])))
    else:
        # Poza tolerancja kolejnosc kryteriow MUSI byc odwrotna. Przy
        # pierwszenstwie dla liczby wymiarow cel 7 bitow zwracal
        # (3,) * 12 = 19,0196 bitu, czyli 2,7x budzet, i wygladalo to na
        # poprawnie dzialajacy dobor.
        p, _ = min(kandydaci,
                   key=lambda pb: (abs(pb[1] - cel), -len(pb[0]), max(pb[0])))
    polowa = tuple(sorted(p, reverse=True))
    return polowa + polowa


def bity_siatki(poziomy):
    """Zadeklarowany budzet siatki w bitach na pozycje."""
    return sum(math.log(int(l), 2.0) for l in poziomy)


class KwantyzatorFSQ(nn.Module):
    """Kwantyzuje USZCZEGOLOWIENIE siatka skalarna; brak kodebooka.

    Interfejs jak `KwantyzatorStyczny` (drabina_stozkowa.py:97):
    `forward(baza, f) -> (punkt, idx)`, `z_indeksow(baza, idx)`, `raport()`,
    `loss`, `indices`, `vq_update`, `codebook_size`.

    Argumenty:
      dim           -- wymiar czesci przestrzennej hiperboloidy (vq_dim = 64);
      poziomy       -- krotka liczb koszykow na wspolrzedna, NIEPARZYSTYCH;
                       `poziomy_dla_bitow(8)` / `(9)` daje domyslne siatki;
      dlugosc       -- APERTURA: maksymalna dlugosc geodezyjna kroku (dokladna);
      rc_pelne      -- wiaze `skala_we` i rame na zamiane polowek kanalow;
      waga_entropii -- OPCJONALNY czlon straty (domyslnie 0,0 = wylaczony
                       BIT W BIT). Uzywac dopiero, gdy `frakcja_bitow` z
                       raportu spadnie ponizej 0,85; w tym repo czlon entropii
                       dal 0,748 -> 0,788 (na innym zadaniu i przez entropie
                       WARUNKOWA, wiec to poszlaka, nie obietnica);
      seed          -- ziarno ramy (7898, jak reszta projektu).
    """

    def __init__(self, dim, poziomy=(3, 3, 3, 3, 3, 3), dlugosc=1.0,
                 rc_pelne=False, waga_entropii=0.0, temp_entropii=1.0,
                 decay_stat=0.99, seed=7898, eps=1e-3, wzmocnienie=1.0,
                 krok_promienia=0.0):
        super().__init__()
        poziomy = tuple(int(l) for l in poziomy)
        d = len(poziomy)
        if any(l % 2 == 0 or l < 3 for l in poziomy):
            # Uzasadnienie w naglowku: parzyste L daje siatke niesymetryczna
            # (dla L=4 wartosci {-1; -0,5; 0; +0,5}), a L=2 wywala `bound`
            # na `atanh(0.5/0.4995)`.
            raise ValueError("poziomy FSQ musza byc NIEPARZYSTE i >= 3: %r"
                             % (poziomy,))
        if d % 2:
            raise ValueError("liczba wymiarow siatki musi byc PARZYSTA "
                             "(parowanie RC), dostalem %d" % d)
        if int(dim) % 2:
            raise ValueError("dim musi byc parzyste (parowanie RC)")
        h = d // 2
        if any(poziomy[i] != poziomy[i + h] for i in range(h)):
            raise ValueError(
                "polowki siatki musza byc rowne (levels[i] == levels[i+d/2]), "
                "inaczej kwantyzacja nie komutuje z zamiana polowek kanalow "
                "i ekwiwariancja RC pada CICHO: %r" % (poziomy,))
        self.dim = int(dim)
        self.poziomy = poziomy
        self.d = d
        self.dlugosc = float(dlugosc)
        self.rc_pelne = bool(rc_pelne)
        self.waga_entropii = float(waga_entropii)
        self.temp_entropii = float(temp_entropii)
        self.decay_stat = float(decay_stat)
        self.wzmocnienie = float(wzmocnienie)
        # Umowa interfejsu: `vq_update` istnieje i jest respektowany. Nie ma
        # kodebooka do przepisania, wiec U6 przestaje byc ryzykiem -- flaga
        # wylacza jedyna rzecz, ktora dodatkowy przebieg moglby zepsuc, czyli
        # STATYSTYKE UZYCIA (inaczej liczylaby negatywy i kontrole RC).
        self.vq_update = True
        self.loss = None
        self.indices = None
        self._diag = {}

        # `autoscale=0.0` NIE JEST domyslnym lenistwem: wersja z autoskala
        # dzieli przez ODCHYLENIE BIEZACEGO BATCHA (fsq.py `_skaluj`), wiec
        # przypisanie kodu w treningu zalezaloby od skladu batcha -- dokladnie
        # to, czego zakazuje U2 i co juz raz zabilo `decode_codes` (70,4 %
        # slotow L0 zmienialo sie ze skladem batcha). Punkt pracy ustawiamy
        # INICJALIZACJA `skala_we`, ktora jest funkcja wylacznie wag.
        self.fsq = FSQ(levels=poziomy, eps=eps, autoscale=0.0)
        self.krok = self.dlugosc / math.sqrt(float(d))
        # ============ SKLADNIK PROMIENIOWY, PRZYPIETY ============
        # DEFEKT, KTORY TO NAPRAWIA. `zloz` to czysty `expmap(baza, PT(v))`,
        # wiec promien rosnie tylko dlatego, ze generyczny krok styczny
        # z promienia r zwieksza r SREDNIO. `KwantyzatorStyczny` ma na to
        # `_promieniowy` i ostrzega wprost: "bez tego skladnika kroki styczne
        # maja losowe kierunki i promien robi BLADZENIE PRZYPADKOWE --
        # zmierzone 1,3297 -> 1,2180 -> 1,2364, czyli poziomy NIE leza na
        # roznych glebokosciach, a wtedy apertura stozka jest na kazdym
        # poziomie taka sama".
        #
        # ZMIERZONE NA BIEGU HypStozek (31 epok, drabina z FSQ):
        #     dr_promien start [1,426; 1,676; 1,973] -> koniec [1,4395; 1,5325; 1,8562]
        #     czyli powloki ZBIEGLY SIE, nie rozeszly.
        #     sinh: 1,985 / 2,127 / 2,977 wobec planowanych 1,992 / 3,408 / 5,693
        # Do tego `drab_krok_promienia` w configu byl CICHYM NO-OPEM przy
        # `fsq_styczny: true`, bo trafial wylacznie do `KwantyzatorStyczny`.
        #
        # PRZYPIETY, NIE UCZONY. `KwantyzatorStyczny` trzyma to jako
        # `nn.Parameter` i to jest ta sama pulapka, w ktora w tym repo wpadly
        # `promien_specyf` (0,692 -> 0,422), `skala_reszty` (0,0542 -> 4e-05)
        # i bramy glowic (0,02 -> 1,42e-04): model konsekwentnie ZWIJA
        # mechanizmy rozwarstwiajace. Bufor odbiera mu te mozliwosc, a cena
        # jest zerowa -- to jedna liczba na poziom.
        self.krok_promienia = float(krok_promienia)
        self.register_buffer("_kr_prom",
                             torch.tensor(self.krok_promienia), persistent=False)

        # ---- redukcja dim -> d. NAZWA decyduje o grupie weight decay ----
        if self.rc_pelne:
            from .rc_ekwi import SplotParowy
            self.skala_we = SplotParowy(self.dim, d, 1, stride=1, padding=0)
            _w = self.skala_we.weight                      # [d/2, dim, 1]
        else:
            self.skala_we = nn.Conv1d(self.dim, d, 1)
            _w = self.skala_we.weight                      # [d, dim, 1]
        with torch.no_grad():
            # WZMOCNIENIE JEDNOSTKOWE: std(wyjscia) = std(wejscia). Domyslny
            # kaiming (a=sqrt(5)) dalby 0,577 * std(f), czyli std(z) ~ 0,58
            # i entropie okolo 78 % zamiast 98,6 %.
            #
            # `wzmocnienie` MNOZY te regule i domyslnie wynosi 1,0, wiec test
            # dymny (wejscie o std 1) jest bit w bit jak dotad. Przy wpieciu
            # w drabine wejsciem NIE jest szum jednostkowy, tylko mapa cech po
            # NormaParowa+SiLU: ZMIERZONE std_we = 0,0651 i 0,0590, a siatka
            # ma odstep koszykow 1,0 -- czyli KAZDA pozycja zaokraglala sie do
            # kodu zerowego. Zmierzony skutek: `uzyte` 1 z 729, `bity` -0,0,
            # `zero_frac` 1,0000. Siatka niosla ZERO bitow, a `raport()`
            # pokazywal to poprawnie -- to jest ten falsyfikator, ktory tu
            # zadzialal.
            #
            # `skala_we` jest UCZONA, wiec to jest wylacznie punkt startowy;
            # falsyfikatorem w biegu pozostaje `frakcja_bitow` (prog 0,85).
            _w.normal_(0.0, float(wzmocnienie) / math.sqrt(float(self.dim)))
            if self.skala_we.bias is not None:
                self.skala_we.bias.zero_()

        # ---- rama: d ortonormalnych kierunkow w R^dim, STALA ----
        g = torch.Generator().manual_seed(int(seed))
        if self.rc_pelne:
            a = torch.randn(self.dim // 2, h, generator=g)
            a, _ = torch.linalg.qr(a)                      # [dim/2, h]
            a = a.t().contiguous()                         # [h, dim/2]
            rama = torch.zeros(d, self.dim)
            rama[:h, :self.dim // 2] = a
            rama[h:, self.dim // 2:] = a
        else:
            a = torch.randn(self.dim, d, generator=g)
            a, _ = torch.linalg.qr(a)                      # [dim, d]
            rama = a.t().contiguous()                      # [d, dim]
        self.register_buffer("rama", rama.float())         # float32: patrz naglowek

        # ---- permutacja alfabetu przy zamianie polowek cyfr (para RC) ----
        K = self.fsq.codebook_size
        _b = self.fsq._basis.clone()
        _l = torch.tensor(poziomy, dtype=torch.long)
        _k = torch.arange(K, dtype=torch.long)
        _dg = (_k.unsqueeze(-1) // _b) % _l                # [K, d]
        _dg2 = torch.cat([_dg[:, h:], _dg[:, :h]], dim=-1)
        self.register_buffer("_para", (_dg2 * _b).sum(-1), persistent=False)
        self.register_buffer("_cyfry", _dg, persistent=False)

        # ---- statystyki (persistent=False: poza `state_dict`, poza EMA) ----
        self.register_buffer("_hist", torch.zeros(K), persistent=False)
        self.register_buffer("_widziane", torch.zeros(K, dtype=torch.bool),
                             persistent=False)
        self.register_buffer("_hist_wsp", torch.zeros(d, max(poziomy)),
                             persistent=False)
        self._n_stat = 0

        # ---- centra koszykow (tylko dla opcjonalnego czlonu entropii) ----
        Lmax = max(poziomy)
        centra = torch.zeros(d, Lmax)
        maska = torch.zeros(d, Lmax, dtype=torch.bool)
        for i, l in enumerate(poziomy):
            for j in range(l):
                centra[i, j] = float(j - l // 2)
                maska[i, j] = True
        self.register_buffer("_centra", centra, persistent=False)
        self.register_buffer("_maska_centr", maska, persistent=False)

    # ------------------------------------------------------------ wlasnosci
    @torch.no_grad()
    def sparuj(self):
        """NIC NIE ROBI, i to jest wynik konstrukcji, nie przeoczenie.

        `KwantyzatorStyczny.sparuj` musi domykac kodebook na zamiane polowek,
        bo kodebook jest UCZONY i EMA rozjezdza polowki. Tutaj nie ma czego
        domykac: siatka FSQ istnieje niezaleznie od danych, a parowanie siedzi
        w RAMIE (`rama[:h]` dziala na pierwszej polowce kanalow, `rama[h:]` na
        drugiej tymi samymi wspolczynnikami) i w `skala_we` (`SplotParowy`).
        Obie sa STALE albo wiazane z konstrukcji.

        Metoda istnieje, zeby wolajacy NIE musial pytac `hasattr` -- cichy
        `hasattr`, ktory pominalby prawdziwe domkniecie, jest dokladnie tym
        trybem awarii, ktory `rc_stem` mial przez warunek na `in_ch`.
        Zmierzone: zgodnosc indeksow z para 1,0000 bez zadnego wolania.
        """
        return None

    @property
    def codebook_size(self):
        """Rozmiar SIATKI. Nie mylic z liczba kodow REALNIE uzywanych -- ta
        jest w raporcie jako `uzyte`, a jej entropia jako `bity`."""
        return self.fsq.codebook_size

    def bity_zadeklarowane(self):
        return bity_siatki(self.poziomy)

    @property
    def embed(self):
        """Cala siatka jako [K, dim] -- zgodnosc z `KwantyzatorStyczny.embed`.

        Liczona na zadanie (729 * 64 = 46 656 liczb), nie trzymana: to jest
        funkcja ramy, a rama jest stala. Do sond, nie do petli treningu.
        """
        idx = torch.arange(self.codebook_size, device=self.rama.device)
        return self.wektor_z_indeksow(idx)

    # --------------------------------------------------------------- kroki
    def wektor_z_indeksow(self, idx, dtype=None):
        """INDEKS -> wektor styczny w origo. Czysta funkcja indeksu (U1).

        Dekodowanie mieszane o podstawie `poziomy`: cyfra i-ta to
        (idx // basis_i) % levels_i, wartosc to (cyfra - levels_i//2) /
        (levels_i//2), a wektor to `krok * wartosci @ rama`. Zadnej statystyki
        batcha, zadnej wielkosci z enkodera, zadnej wagi uczonej.

        `dtype` MUSI byc ten sam co w `forward` (tam `baza.dtype`, czyli
        float64), inaczej rownosc z przebiegiem wprzod jest tylko rzedu 1e-07:
        mnozenie w float32 i rzutowanie na float64 to nie to samo, co mnozenie
        po rzutowaniu. Umowa dyfuzji brzmi 0,000000e+00, a nie "male".
        """
        q = self.fsq.indices_to_codes(idx)                 # [..., d]
        dt = self.rama.dtype if dtype is None else dtype
        return self.krok * (q.to(dt) @ self.rama.to(dt))

    def _promieniowy(self, baza):
        """Jednostkowy wektor STYCZNY w `baza`, wskazujacy NA ZEWNATRZ.

        Rzut ambientnego `(0, u)` na przestrzen styczna, gdzie `u` to kierunek
        czesci przestrzennej `baza`. Ten sam wzor co
        `KwantyzatorStyczny._promieniowy`, celowo powtorzony: oba poziomy
        drabiny musza skladac sie tym SAMYM wzorem, a kopia szesciu linii jest
        tansza niz zaleznosc miedzy dwoma modulami.
        """
        from .hyperbolic import project_to_tangent
        sp = baza[..., 1:]
        u = sp / sp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        amb = torch.cat([torch.zeros_like(baza[..., :1]), u], dim=-1)
        t = project_to_tangent(baza, amb)
        return t / torch.sqrt(minkowski_dot(t, t).clamp_min(1e-12))

    def _krok(self, baza, v_styczny):
        """Transport + skladnik promieniowy + expmap. JEDNA metoda dla
        `forward` i `z_indeksow` -- gdyby byly dwie, rozejscie sie tych drog
        zabija dyfuzje PO CICHU."""
        if self.krok_promienia == 0.0:
            return self.zloz(baza, v_styczny)
        vt = parallel_transport0(baza, v_styczny)
        vt = vt + self._kr_prom.to(vt.dtype) * self._promieniowy(baza)
        return self._expmap_bezpieczny(baza, vt)

    @staticmethod
    def _expmap_bezpieczny(baza, vt):
        """`expmap` z poprawnym KODEM ZEROWYM -- wyjete z `zloz`, zeby
        `_krok` i `zloz` liczyly to samo."""
        n2 = minkowski_dot(vt, vt).clamp_min(0.0)
        n = n2.clamp_min(1e-30).sqrt()
        duze = n > 1e-9
        n_bez = torch.where(duze, n, torch.ones_like(n))
        sinc = torch.where(duze, torch.sinh(n_bez) / n_bez, torch.ones_like(n))
        return torch.cosh(n) * baza + sinc * vt

    @staticmethod
    def zloz(baza, v_styczny):
        """`expmap(baza, PT(origo -> baza, v))` -- jeden krok geodezyjny.

        Wzor ten sam co `KwantyzatorStyczny.zloz` (drabina_stozkowa.py:190),
        celowo powtorzony zamiast importowany: drabina moze miec poziomy obu
        typow i musza skladac sie tym SAMYM wzorem, a kopia trzech linii jest
        tansza niz zaleznosc miedzy dwoma nieprzetestowanymi modulami.
        Transport rownolegly jest konieczny -- wektor styczny w origo nie jest
        wektorem stycznym w `baza`.

        JEDNA ROZNICA, i jest nia KOD ZEROWY -- dlatego `hyperbolic.expmap`
        nie jest tu wolane. Tamto liczy `norm = sqrt(<v,v>_L.clamp_min(1e-7))`,
        wiec dla v = 0 zwraca `cosh(3,1623e-04) * x = (1 + 5,0e-08) * x`:
        punkt POZA hiperboloida (|<x,x>_L + 1| = 1,0e-07 przy dopuszczalnym
        1e-16) i przesuniety RADIALNIE NA ZEWNATRZ. Kodebook uczony nigdy tego
        nie pokazal, bo jego wpisy nie sa dokladnie zerowe (init
        `linspace(0.25, 1.0)`, drabina_stozkowa.py:124). Siatka FSQ ma kod
        zerowy z KONSTRUKCJI i trafia w niego okolo 0,5 % pozycji, a jego
        znaczenie to "rodzic wystarczy" -- musi wiec zostawiac punkt DOKLADNIE
        na miejscu, inaczej "brak uszczegolowienia" jest systematycznym
        dryfem na zewnatrz, powtorzonym na kazdym poziomie.

        Tutaj `sinh(n)/n` jest zastapione jedynka dla n < 1e-9, a `sqrt` ma
        `clamp_min(1e-30)`, zeby przy n = 0 pochodna `d sqrt / d n2` byla
        wyzerowana przez clamp, a nie nieskonczona (klasyczne NaN "podwojnego
        where": maskowanie wartosci nie maskuje gradientu). Dla n >> 1e-4 wzor
        jest identyczny z `expmap` co do 1e-16. Gradient przy kodzie zerowym
        plynie przez czlon liniowy `sinc * vt`, wiec pozycja z kodem zerowym
        NIE jest martwa.
        """
        vt = parallel_transport0(baza, v_styczny)
        n2 = minkowski_dot(vt, vt).clamp_min(0.0)          # [..., 1] >= 0
        n = n2.clamp_min(1e-30).sqrt()
        duze = n > 1e-9
        n_bez = torch.where(duze, n, torch.ones_like(n))
        sinc = torch.where(duze, torch.sinh(n_bez) / n_bez, torch.ones_like(n))
        return torch.cosh(n) * baza + sinc * vt

    def forward(self, baza, f):
        """`baza`: [B, L, dim+1] punkty rodzica. `f`: [B, dim, L] mapa cech."""
        if f.dim() != 3 or f.shape[1] != self.dim:
            raise ValueError("f ma byc [B, %d, L], jest %s"
                             % (self.dim, tuple(f.shape)))
        if (baza.dim() != 3 or baza.shape[1] != f.shape[-1]
                or baza.shape[-1] != self.dim + 1):
            # Glosno, bo cicha niezgodnosc dlugosci konczy sie broadcastem,
            # ktory "dziala" i daje bezsens.
            raise ValueError("baza ma byc [B, %d, %d], jest %s"
                             % (f.shape[-1], self.dim + 1, tuple(baza.shape)))
        z = self.skala_we(f).transpose(1, 2)               # [B, L, d]
        y = self.fsq.bound(z)                              # przed zaokragleniem
        hw = (self.fsq._levels.to(y.dtype) // 2).clamp_min(1)
        idx = self.fsq.codes_to_indices((torch.round(y) / hw).detach())
        # STRAIGHT-THROUGH O DOKLADNEJ WARTOSCI. Zwykle `x + (round(x) -
        # x).detach()` (fsq.py `round_ste`) NIE daje bit w bit wartosci
        # zaokraglonej: `r - x` jest zaokraglane, a suma nie musi wrocic do
        # `r`. Blad rzedu 1 ULP przechodzi potem przez `expmap` i `z_indeksow`
        # rozjezdza sie z przebiegiem wprzod o ~1e-07. W tym repo umowa
        # dekodowania jest weryfikowana jako 0,000000e+00 i taka ma zostac,
        # wiec skladamy STE na WEKTORZE: wartosc bierzemy z siatki (dokladnie
        # ta sama droga co `z_indeksow`), a gradient z galezi ciaglej.
        # `v_s - v_s.detach()` to dokladne zero (te same liczby), wiec
        # `v = v_h` co do bitu, a d v / d z = d v_s / d z.
        dt = baza.dtype
        R = self.rama.to(dt)
        v_h = self.krok * (self.fsq.indices_to_codes(idx).to(dt) @ R)
        v_s = self.krok * ((y / hw).to(dt) @ R)
        v = v_h + (v_s - v_s.detach())
        self.indices = idx

        if self.waga_entropii > 0.0:
            self.loss = self.waga_entropii * self._czlon_entropii(y)
        else:
            # Zero, ale TENSOR: `DrabinaStozkowa.forward` sumuje `kr.loss`.
            self.loss = torch.zeros((), device=f.device, dtype=torch.float32)

        with torch.no_grad():
            if self.vq_update:
                self._statystyka(idx)
            _n = v.norm(dim=-1)
            _u = torch.bincount(idx.reshape(-1),
                                minlength=self.codebook_size).double()
            _u = _u / _u.sum().clamp_min(1.0)
            _uu = _u[_u > 0]
            _y = y.detach()
            _hw = hw.to(_y.dtype)
            self._diag = {
                # `perplexity` zostaje pod ta sama nazwa co w
                # `KwantyzatorStyczny`, zeby log dalo sie porownac wprost.
                "perplexity": round(float(torch.exp(-(_uu * _uu.log()).sum())), 2),
                "dl_sr": round(float(_n.mean()), 4),
                "dl_max": round(self.dlugosc, 3),
                "std_we": round(float(z.detach().float().std()), 4),
                # ========== APERTURA: ILE BUDZETU IDZIE W UZYCIE ==========
                # `poza_stozkiem` w raporcie drabiny jest ZEREM ARYTMETYCZNIE:
                # ||v|| = krok * ||wartosci @ rama|| <= (dlugosc / sqrt(d)) *
                # sqrt(d) = dlugosc, bo wartosci siatki leza w [-1, 1]^d,
                # a rama jest ortonormalna. Kat kroku jest wiec ograniczony
                # przez `dlugosc / sinh(r)` i przy aperturze 1,0 nie ma jak jej
                # przekroczyc. Zapalalne jest WYKORZYSTANIE, nie przekroczenie:
                #   `wyk_apert` -- ||v|| / dlugosc, 0..1. Zmierzone w tej
                #                  rodzinie 0,95-0,96, czyli krok siedzi na
                #                  krancu apertury i model DUSI SIE nia;
                #   `nasyc`     -- frakcja WSPOLRZEDNYCH siatki w koszyku
                #                  skrajnym. Blisko 1 razem z `wyk_apert`
                #                  ~1 znaczy, ze siatka pracuje jak binarna
                #                  (ten sam objaw co przy `fsq_wzmocnienie:
                #                  15`, gdzie `tanh` byl wysycony), a blisko
                #                  0 przy `wyk_apert` ~0, ze wejscie zapadlo
                #                  do kodu zerowego (`std_we` 0,065 -> `bity`
                #                  -0,0, awaria juz raz zmierzona);
                #   `blad_kw`   -- ||v_s - v_h|| / ||v_s||, wzgledny blad
                #                  kwantyzacji, MIANOWNIK PO STRONIE ENKODERA.
                #                  To jest ROZJAZD ENKODER-SIATKA i jedyna
                #                  wersja tej wielkosci, ktora nie jest
                #                  tozsamoscia -- promien enkodera i promien
                #                  powloki sa przypiete do tej samej liczby,
                #                  wiec ich roznica jest zerem z konstrukcji,
                #                  a ta nie.
                #                  MIANOWNIK MA ZNACZENIE: przy ||v_h|| (kwant)
                #                  zapad wejscia do kodu zerowego dawal
                #                  5,0e+08 zamiast czytelnego 1,0000 ("cala
                #                  tresc zgubiona"), bo dzielilo sie przez
                #                  zero. Zmierzone w bramce.
                "wyk_apert": round(float(_n.mean()) / max(self.dlugosc, 1e-12), 4),
                "nasyc": round(float((_y.abs() >= (_hw - 0.5)).double().mean()), 4),
                "blad_kw": round(float((v_s.detach() - v_h).norm(dim=-1).mean()
                                       / v_s.detach().norm(dim=-1).mean()
                                       .clamp_min(1e-12)), 4),
            }
            self._diag.update(self.raport_entropii())
        return self._krok(baza, v), idx

    @torch.no_grad()
    def z_indeksow(self, baza, idx):
        """Odtworzenie kroku z SAMEGO INDEKSU -- sciezka dyfuzji (U1)."""
        return self._krok(baza, self.wektor_z_indeksow(idx, dtype=baza.dtype))

    def para_indeksu(self, idx):
        """Indeks kodu odbicia RC: zamiana POLOWEK CYFR.

        Odpowiednik `sparuj_kodebook` dla siatki -- z ta roznica, ze permutacja
        alfabetu jest znana z konstrukcji i nic nie trzeba domykac. Przy
        `rc_pelne` zachodzi `idx(rc x)[L-1-t] == para_indeksu(idx(x)[t])`
        DOKLADNIE, i to jest test, ktory ten modul ma przechodzic na 1,0000
        (kontrola bez flagi: 0,0000).
        """
        return self._para[idx]

    # --------------------------------------------------------- diagnostyka
    @torch.no_grad()
    def _statystyka(self, idx):
        p = idx.reshape(-1)
        n = torch.bincount(p, minlength=self.codebook_size).to(self._hist.dtype)
        self._hist.mul_(self.decay_stat).add_(n, alpha=1.0 - self.decay_stat)
        self._widziane |= n > 0
        dg = self._cyfry[p]                                # [N, d]
        for i in range(self.d):
            c = torch.bincount(dg[:, i], minlength=self._hist_wsp.shape[1])
            self._hist_wsp[i].mul_(self.decay_stat).add_(
                c.to(self._hist_wsp.dtype), alpha=1.0 - self.decay_stat)
        self._n_stat += int(p.numel())

    @torch.no_grad()
    def raport_entropii(self):
        """Entropia kodu -- JEDYNY uczciwy falsyfikator budzetu bitowego.

        `bits_used_frac` i pokrycie siatki tego NIE mierza: zmierzone w tym
        repo FSQ dotykalo 7-8 koszykow z 8 na kanal i niosło 29,3 bitu z 48,
        bo rozklad byl spiczasty (70,8 % masy w jednym koszyku).

        Zwraca:
          bity          -- entropia LACZNA rozkladu kodow (EMA, decay 0,99);
          bity_marg     -- suma entropii BRZEGOWYCH po wspolrzednych; roznica
                           `bity_marg - bity` to redundancja miedzy
                           wspolrzednymi (przy niezaleznosci = 0);
          bity_zad      -- budzet zadeklarowany, log2(prod(poziomy));
          frakcja_bitow -- bity / bity_zad. PROG: >= 0,85;
          uzyte         -- ile roznych kodow padlo od ostatniego zerowania;
          zero_frac     -- udzial kodu ZEROWEGO (brak uszczegolowienia).
                           Oczekiwane 0,005-0,007 przy inicjalizacji; > 0,20
                           znaczy, ze zapadla skala wejscia (weight decay!).
        """
        s = float(self._hist.sum())
        if s <= 0.0:
            return {}
        p = (self._hist / s).double()
        pp = p[p > 0]
        bity = float(-(pp * pp.log()).sum() / _LOG2)
        marg = 0.0
        for i, l in enumerate(self.poziomy):
            hi = self._hist_wsp[i, :l].double()
            hi = hi / hi.sum().clamp_min(1e-30)
            hi = hi[hi > 0]
            marg += float(-(hi * hi.log()).sum() / _LOG2)
        zad = self.bity_zadeklarowane()
        # Kod zerowy: wszystkie cyfry rowne levels//2.
        z_idx = int((torch.tensor([l // 2 for l in self.poziomy])
                     * self.fsq._basis.cpu()).sum())
        return {
            "bity": round(bity, 3),
            "bity_marg": round(marg, 3),
            "bity_zad": round(zad, 3),
            "frakcja_bitow": round(bity / zad, 4),
            "uzyte": int(self._widziane.sum()),
            "siatka": int(self.codebook_size),
            "zero_frac": round(float(p[z_idx]), 5),
        }

    def raport(self):
        return dict(self._diag)

    @torch.no_grad()
    def zeruj_statystyki(self):
        """Statystyka ma opisywac TEN przebieg, a nie ogon po treningu.

        Wolac przed ewaluacja. Bez tego EMA (decay 0,99, okno ~100 krokow)
        miesza rozklad walidacyjny z treningowym -- ta sama pulapka, przez
        ktora `fsq_grid_used` czytalo [0.0, 0.0, 0.0] i wygladalo na calkowity
        zapad, bedac brakiem pomiaru (fsq.py, `forward`).
        """
        self._hist.zero_()
        self._hist_wsp.zero_()
        self._widziane.zero_()
        self._n_stat = 0

    # ------------------------------------------------- opcjonalna regulacja
    def _czlon_entropii(self, y):
        """Miekka entropia przypisania: H(p) na probke MINUS H(sredniego p).

        `y` to wartosc po `bound`, przed zaokragleniem; koszyki maja srodki w
        liczbach calkowitych, wiec odleglosc do srodka jest w tych samych
        jednostkach co `temp_entropii`.

        UWAGA NA ZAKRES: ten czlon ZALEZY OD SKLADU BATCHA (drugi skladnik to
        srednia po batchu). To jest dozwolone -- U2 zakazuje zaleznosci od
        batcha w SCIEZCE STERUJACEJ KWANTYZACJA, a nie w stracie. Przypisanie
        kodu pozostaje funkcja samej sekwencji, wiec `decode_codes` i dyfuzja
        sa nietkniete. Domyslnie waga 0,0, czyli powrot do bazy BIT W BIT.
        """
        c = self._centra.to(y.dtype)
        m = self._maska_centr
        logity = -(y.reshape(-1, self.d).unsqueeze(-1)
                   - c.unsqueeze(0)).pow(2) / self.temp_entropii
        logity = logity.masked_fill(~m.unsqueeze(0), float("-inf"))
        p = torch.softmax(logity, dim=-1)
        lp = torch.log_softmax(logity, dim=-1)
        # `masked_fill(-inf)` daje p=0 i lp=-inf, czyli 0 * -inf = NaN.
        #
        # POPRZEDNIA WERSJA -- `torch.where(m, p * lp, 0)` -- naprawiala
        # WARTOSC i NIE naprawiala GRADIENTU, i to zabijalo caly bieg.
        # `where` nie chroni niewybranej galezi: backward liczy `p * lp`
        # dla WSZYSTKICH elementow, dostaje NaN na zamaskowanych, a potem
        # mnozy go przez zero maski -- a `0 * NaN = NaN`. NaN wracal wiec
        # do calego enkodera przy pierwszym kroku.
        #
        # Poprawka: uczynic niewybrana galaz SKONCZONA ZANIM dojdzie do
        # mnozenia. `p` jest dokladnie 0 na zamaskowanych, wiec `p * 0`
        # daje tam poprawne 0 w wartosci, a gradient jest skonczony.
        lp = torch.where(m.unsqueeze(0), lp, torch.zeros_like(lp))
        plp = p * lp
        h_prob = -plp.sum(-1).mean()                       # chcemy MALE
        pm = p.mean(0)                                     # [d, Lmax]
        h_marg = -torch.where(m, pm * pm.clamp_min(1e-12).log(),
                              torch.zeros_like(pm)).sum(-1).mean()
        return h_prob - h_marg                             # chcemy DUZE h_marg

    def extra_repr(self):
        return ("dim=%d, siatka=%r (%d kodow, %.4f bitu), apertura=%.3f, rc=%s"
                % (self.dim, self.poziomy, self.codebook_size,
                   self.bity_zadeklarowane(), self.dlugosc, self.rc_pelne))


@torch.no_grad()
def kontrola_kodow(indeksy, rozmiary, nazwy=None):
    """KONTROLA: ile kodow realnie zyje i ile z nich mowi juz rodzic.

    `indeksy`  -- lista tensorow [B, L_k] typu long, w kolejnosci DRABINY,
                  czyli od NAJGRUBSZEGO poziomu (tak jak `drabina.indices`
                  i `HypVQNet.encode_codes`);
    `rozmiary` -- lista rozmiarow alfabetu (dla FSQ `codebook_size`).

    Liczy na poziom:
      n            liczba pozycji w probce,
      uzyte / K    ile roznych kodow padlo,
      H            entropia empiryczna w bitach (plug-in),
      H_mm         to samo z poprawka Millera-Madowa (m-1)/(2 N ln2) -- podana
                   OSOBNO, a nie wstawiona po cichu, bo dla rzadkich tablic
                   plug-in jest obciazony W DOL i latwo pomylic to z brakiem
                   entropii,
      frakcja      H / log2(K) -- falsyfikator budzetu (prog 0,85),
      H_war        H(kod | kod rodzica) w bitach,
      I            H - H_war, informacja wzajemna z rodzicem,
      I_bias       obciazenie plug-in dla `I` (m_lacz - m_rodzic)/(2 N ln2).

    PO CO `I`. Drabina rezydualna miala zmierzone kor(reszta, -kod rodzica) =
    0,9997 i 0,9985, czyli 97,4 % i 94,6 % reszty bylo WLASNYM KODEM
    poprzedniego poziomu ze znakiem minus -- drabina kwantyzowala sama siebie.
    Ten sam blad w wersji stozkowej wyszedlby jako wysokie `I`: dziecko
    powtarzajace rodzica nie niesie nowych bitow, niezaleznie od tego, jak
    ladnie wyglada perplexity. PROG ROBOCZY: I < 1,0 bitu.

    OGRANICZENIE, ktore trzeba znac: rodzic dla pozycji `t` dziecka brany jest
    NAJBLIZSZY, `floor((t + 0.5) * L_rodzic / L_dziecko)`, podczas gdy drabina
    podnosi punkt rodzica interpolacja LINIOWA (drabina_stozkowa.py:250), czyli
    miesza dwoch sasiadow. Ta kontrola ZANIZA wiec zaleznosc; jest gorna
    granica na "dziecko jest niezalezne od rodzica", a nie dolna.
    """
    wyn = []
    for k, idx in enumerate(indeksy):
        K = int(rozmiary[k])
        p = idx.reshape(-1).long()
        n = int(p.numel())
        licz = torch.bincount(p, minlength=K).double()
        pr = licz / licz.sum().clamp_min(1.0)
        prp = pr[pr > 0]
        H = float(-(prp * prp.log()).sum() / _LOG2)
        m = int((licz > 0).sum())
        r = {
            "poziom": (nazwy[k] if nazwy else "L%d" % k),
            "n": n, "K": K, "uzyte": m,
            "H": round(H, 4),
            "H_mm": round(H + (m - 1) / (2.0 * n * _LOG2), 4),
            "frakcja": round(H / math.log(K, 2.0), 4),
        }
        if k > 0:
            rod = indeksy[k - 1].reshape(idx.shape[0], -1)
            Lc, Lp = idx.shape[-1], rod.shape[-1]
            poz = ((torch.arange(Lc, device=idx.device, dtype=torch.float64)
                    + 0.5) * Lp / Lc).floor().long().clamp(0, Lp - 1)
            rp = rod[:, poz].reshape(-1).long()
            Kp = int(rozmiary[k - 1])
            wsp = torch.bincount(rp * K + p, minlength=Kp * K).double()
            pj = wsp / wsp.sum().clamp_min(1.0)
            pjp = pj[pj > 0]
            Hj = float(-(pjp * pjp.log()).sum() / _LOG2)
            lr = torch.bincount(rp, minlength=Kp).double()
            prr = lr / lr.sum().clamp_min(1.0)
            prrp = prr[prr > 0]
            Hr = float(-(prrp * prrp.log()).sum() / _LOG2)
            Hw = Hj - Hr
            r["H_war"] = round(Hw, 4)
            r["I"] = round(H - Hw, 4)
            r["I_bias"] = round((int((wsp > 0).sum()) - int((lr > 0).sum()))
                                / (2.0 * n * _LOG2), 4)
        wyn.append(r)
    return wyn


__all__ = ["KwantyzatorFSQ", "poziomy_dla_bitow", "bity_siatki",
           "kontrola_kodow"]
