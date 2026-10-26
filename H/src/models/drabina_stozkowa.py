# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperbolic import (expmap, expmap0, lorentz_distance, minkowski_dot,
                         origin_like, parallel_transport0,
                         project_to_hyperboloid, radius)


class KwantyzatorStyczny(nn.Module):
    """Kwantyzuje USZCZEGOLOWIENIE: wektor styczny w origo, potem transport.

    Kodebook trzyma `K` wektorow stycznych W ORIGO (czesc przestrzenna,
    `[K, dim]`). Przestrzen styczna w origo JEST euklidesowa, wiec
    przypisanie liczymy euklidesowo i nie udajemy, ze jest inaczej.

    `dlugosc` ogranicza norme kroku. To nie jest regularyzacja -- to jest
    APERTURA STOZKA: krok o dlugosci `d` z promienia `r` obejmuje kat okolo
    `d / sinh(r)`. Bez ograniczenia dziecko moze wyladowac gdziekolwiek i
    zawieranie znika, czyli wracamy do drabiny rezydualnej.
    """

    def __init__(self, dim, codebook_size, dlugosc=1.0, decay=0.99,
                 commit_weight=0.25, eps=1e-5, krok_promienia=0.5,
                 dead_threshold=0.5, dtype=torch.float64):
        super().__init__()
        self.dim = int(dim)
        self.codebook_size = int(codebook_size)
        self.dlugosc = float(dlugosc)
        self.decay = float(decay)
        self.commit_weight = float(commit_weight)
        self.eps = float(eps)
        self.dead_threshold = float(dead_threshold)
        # KIERUNKI rozlozone na sferze, DLUGOSCI rozne -- zeby kodebook od
        # startu pokrywal zarowno "male poprawki", jak i "duze skoki".
        g = torch.Generator().manual_seed(7898)
        v = torch.randn(self.codebook_size, self.dim, generator=g, dtype=dtype)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        skala = torch.linspace(0.25, 1.0, self.codebook_size, dtype=dtype)
        self.register_buffer("embed", v * skala.unsqueeze(-1) * self.dlugosc)
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size,
                                                         dtype=dtype))
        self.register_buffer("embed_avg", self.embed.clone())
        # `krok_promienia` NIE ma w nazwie wzorca bez decay ("gate",
        # "brama", "_skala", "skala_", ...), a jest MALYM skalarem --
        # czyli dokladnie tym, co weight decay zabija (zmierzone:
        # brama 0,02 -> 1,42e-04). Stad nazwa z przedrostkiem "skala_".
        self.skala_promienia = nn.Parameter(torch.tensor(float(krok_promienia)))
        self.vq_update = True
        self.loss = None
        self.indices = None
        self._diag = {}

    # ------------------------------------------------------------------ kody
    def kwantyzuj(self, v):
        """Euklidesowy argmin po wpisach; `v` ma ksztalt [N, dim]."""
        d = (v.pow(2).sum(-1, keepdim=True)
             - 2.0 * v @ self.embed.t()
             + self.embed.pow(2).sum(-1).unsqueeze(0))
        idx = d.argmin(dim=-1)
        return self.embed[idx], idx

    @torch.no_grad()
    def _ema(self, plaskie, idx):
        oh = torch.zeros(plaskie.shape[0], self.codebook_size,
                         device=plaskie.device, dtype=plaskie.dtype)
        oh.scatter_(1, idx.unsqueeze(1), 1.0)
        n = oh.sum(0)
        self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
        suma = oh.t() @ plaskie
        self.embed_avg.mul_(self.decay).add_(suma, alpha=1 - self.decay)
        m = self.cluster_size.sum()
        c = ((self.cluster_size + self.eps)
             / (m + self.codebook_size * self.eps) * m)
        nowe = self.embed_avg / c.unsqueeze(-1)
        # OGRANICZENIE DLUGOSCI po aktualizacji, nie przed: EMA moze wyprowadzic
        # wpis poza aperture, a wtedy stozek przestaje cokolwiek ograniczac.
        nn_ = nowe.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        skal = torch.clamp(nn_, max=self.dlugosc) / nn_
        self.embed.data.copy_(nowe * skal)

        # ================= RESTART MARTWYCH KODOW =================
        # BEZ TEGO ZAPAD JEST GWARANTOWANY, nie prawdopodobny: kod, ktory
        # przestal wygrywac, nie dostaje ani gradientu, ani aktualizacji EMA,
        # wiec utrata pojemnosci jest POCHLANIAJACA. Zmierzone w tym repo na
        # biegu z wylaczonym restartem: `bits_used_frac` 0,2970 w epoce 1
        # (perplexity rzedu 4 kodow na 128) i recon 0,8227 wobec 0,9952 --
        # i byl to artefakt implementacji, nie wynik o rozmaitosci.
        #
        # PILNE WLASNIE TU: przy inicjalizacji zmierzono perplexity 1,87 na
        # 256 i 9,52 na 512 -- mapa cech ma silna skladowa WSPOLNA miedzy
        # pozycjami, wiec argmin wskazuje jeden kod dla prawie kazdej pozycji.
        # Bez restartu drabina wystartowalaby z 2 uzywanych kodow i nie miala
        # jak z tego wyjsc.
        martwe = self.cluster_size < self.dead_threshold
        if martwe.any() and plaskie.shape[0] > 0:
            wyb = torch.randint(0, plaskie.shape[0], (int(martwe.sum()),),
                                device=plaskie.device)
            nowe_v = plaskie[wyb]
            # APERTURA OBOWIAZUJE TAKZE RESTART. Wpis dluzszy niz `dlugosc`
            # przestaje byc krokiem w stozku i zawieranie znika dla kazdej
            # pozycji, ktora go wybierze.
            n2 = nowe_v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            nowe_v = nowe_v * (torch.clamp(n2, max=self.dlugosc) / n2)
            self.embed.data[martwe] = nowe_v
            self.embed_avg.data[martwe] = nowe_v
            self.cluster_size[martwe] = 1.0

    def forward(self, baza, f):
        """`baza`: [B, L, dim+1] punkty rodzica. `f`: [B, dim, L] mapa cech."""
        B, D, L = f.shape
        v = f.transpose(1, 2).to(self.embed.dtype)             # [B, L, dim]
        # SKALOWANIE, NIE OBCINANIE: obcinanie zeruje gradient dla kazdego
        # wektora poza apertura, czyli dokladnie dla tych, ktore trzeba
        # sciagnac. Skalowanie zachowuje kierunek i gradient.
        n = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        v = v * (torch.clamp(n, max=self.dlugosc) / n)
        plaskie = v.reshape(-1, self.dim)
        q, idx = self.kwantyzuj(plaskie)
        q = q.view(B, L, self.dim)
        idx = idx.view(B, L)
        if self.training and self.vq_update:
            self._ema(plaskie.detach(), idx.reshape(-1))
        self.loss = self.commit_weight * (v - q.detach()).pow(2).mean().float()
        st = v + (q - v).detach()                              # straight-through
        self.indices = idx
        with torch.no_grad():
            _u = F.one_hot(idx.reshape(-1), self.codebook_size).double().mean(0)
            _u = _u[_u > 0]
            _uzyte = int((self.cluster_size >= self.dead_threshold).sum())
            self._diag = {
                "perplexity": round(float(torch.exp(-(_u * _u.log()).sum())), 2),
                # WARUNEK NIEZBEDNY ALARMU ZAPADU. `perplexity` liczona z
                # JEDNEGO batcha myli sie w obie strony przy 8 x 62 = 496
                # probkach na 256 kodow; `uzyte` czyta EMA, wiec pamieta
                # historie i jest ta wielkoscia, ktorej spadek jest
                # falsyfikatorem (patrz `bits_used_frac`).
                "uzyte": _uzyte,
                "frakcja_kodow": round(_uzyte / self.codebook_size, 4),
                "dl_sr": round(float(q.norm(dim=-1).mean()), 4),
                "dl_max": round(self.dlugosc, 3),
                # ========== APERTURA: CO MODEL CHCE, A CO WOLNO ==========
                # `dl_sr / dl_max` bylo w logu, ale jako dwie osobne liczby,
                # i nikt go nie liczyl. `poza_stozkiem` (raport drabiny) jest
                # ZEREM ARYTMETYCZNIE, bo krok ma dlugosc najwyzej `dlugosc`,
                # wiec kat <= dlugosc / sinh(r) i przy aperturze 1,0 nie ma
                # jak jej przekroczyc. Falsyfikator "model dusi sie apertura"
                # musi wiec stac na WYKORZYSTANIU budzetu, nie na jego
                # przekroczeniu.
                #   `dl_we_sr`  -- srednia norma wektora stycznego PRZED
                #                  obcieciem, czyli dlugosc, ktorej enkoder
                #                  SIE DOMAGA. Wieksza od `dl_max` znaczy, ze
                #                  apertura tnie;
                #   `zaciete`   -- frakcja pozycji realnie obcietych (0..1);
                #   `wyk_apert` -- `||q|| / dlugosc`, 0..1. Zmierzone w tej
                #                  rodzinie 0,95-0,96, czyli kodebook siedzi
                #                  na krancu apertury;
                #   `blad_kw`   -- `||v - q|| / ||v||`, wzgledny blad
                #                  kwantyzacji. To jest ROZJAZD ENKODER-
                #                  KODEBOOK w przestrzeni stycznej i JEDYNA
                #                  wersja tej wielkosci, ktora nie jest
                #                  tozsamoscia (promien enkodera i promien
                #                  powloki sa przypiete do tej samej liczby).
                "dl_we_sr": round(float(n.mean()), 4),
                "zaciete": round(float((n.reshape(-1) > self.dlugosc)
                                       .double().mean()), 4),
                "wyk_apert": round(float(q.norm(dim=-1).mean())
                                   / max(self.dlugosc, 1e-12), 4),
                "blad_kw": round(float((v - q).norm(dim=-1).mean()
                                       / v.norm(dim=-1).mean().clamp_min(1e-12)), 4),
            }
        return self._krok(baza, st), idx

    def _promieniowy(self, baza):
        """Jednostkowy wektor STYCZNY w `baza` wskazujacy NA ZEWNATRZ.

        Kierunek promieniowy w punkcie `x` to rzut wektora `(0, u)` na
        przestrzen styczna, gdzie `u` to kierunek czesci przestrzennej `x`.
        Bez tego skladnika kroki styczne maja losowe kierunki i promien robi
        BLADZENIE PRZYPADKOWE -- zmierzone w pierwszym tescie: 1,3297 ->
        1,2180 -> 1,2364, czyli poziomy NIE leza na roznych glebokosciach,
        a wtedy apertura stozka jest na kazdym poziomie taka sama i cala
        teza o "silniej w punkcie, slabiej dalej" nie ma na czym stanac.
        """
        from .hyperbolic import project_to_tangent
        sp = baza[..., 1:]
        u = sp / sp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        amb = torch.cat([torch.zeros_like(baza[..., :1]), u], dim=-1)
        t = project_to_tangent(baza, amb)
        return t / torch.sqrt(minkowski_dot(t, t).clamp_min(1e-12))

    def _krok(self, baza, v_styczny):
        """JEDEN krok geodezyjny: transport + skladnik promieniowy + expmap.

        JEDNA metoda dla `forward` i dla `z_indeksow`. Gdyby byly dwie, mogly
        by sie rozejsc -- a rozejscie sie tych dwoch sciezek to dokladnie
        awaria, ktora zabija dyfuzje PO CICHU: `decode_codes` zwracaloby
        sensownie wygladajaca sekwencje, ktora nie odpowiada przebiegowi
        wprzod. Test dymny sprawdza rownosc CO DO BITU (dzis 5,6e-17).

        SKLADNIK PROMIENIOWY jest uczony jednym skalarem na poziom, a jego
        KIERUNEK jest wyznaczony geometrycznie. Model steruje wiec tylko tym,
        JAK GLEBOKO schodzi poziom, a nie DOKAD -- to zamyka droge ucieczki,
        ktora w tym repo raz rozbila bieg (promien L0 1,44 -> 5,93 -> 8,22,
        commit 213 -> 3428, recon 0,9986 -> 0,4155).
        """
        krok = parallel_transport0(baza, v_styczny)
        krok = krok + self.skala_promienia.to(krok.dtype) * self._promieniowy(baza)
        return expmap(baza, krok)

    @staticmethod
    def zloz(baza, v_styczny):
        """`expmap(baza, PT(origo -> baza, v))` -- jeden krok geodezyjny.

        Transport rownolegly jest KONIECZNY: wektor styczny w origo NIE JEST
        wektorem stycznym w `baza`, a uzycie go wprost jest najczestszym
        bledem w kodzie riemannowskim (w tym repo raz juz kosztowal
        10^11-krotne wzmocnienie pedu przez pomylke pierwszenstwa operatorow).
        """
        vt = parallel_transport0(baza, v_styczny)
        return expmap(baza, vt)

    @torch.no_grad()
    def sparuj(self):
        """Domyka kodebook STYCZNY na zamiane polowek -- odpowiednik
        `sparuj_kodebook` dla wektorow, nie dla punktow.

        Pod RC wektor styczny w origo przechodzi na siebie z ZAMIENIONYMI
        POLOWKAMI czesci przestrzennej. Zeby INDEKS byl ekwiwariantny, kodebook
        musi byc na te zamiane zamkniety -- inaczej symetria ginie dokladnie
        tam, gdzie powstaje latent, i `rc_pelne` daje ekwiwariantny enkoder
        karmiacy niesymetryczny slownik.

        ROZNICA WOBEC `sparuj_kodebook`: tamten BIERZE pierwsza polowe i
        odbija ja na druga, wiec statystyka EMA drugiej polowy jest
        wyrzucana. Tutaj polowki sa USREDNIANE (`embed_avg` i `cluster_size`
        razem), bo przy augmentacji `RandomRC` obie polowy widza te sama
        liczbe przykladow w oczekiwaniu i sredniowanie jest po prostu lepszym
        estymatorem tej samej wielkosci. Nie ma tu rzutu na hiperboloide --
        przestrzen styczna jest liniowa i rzut bylby bledem typu.
        """
        K, d = self.embed.shape
        h, dh = K // 2, d // 2

        def _zam(t):
            return torch.cat([t[:, dh:], t[:, :dh]], dim=-1)

        n = 0.5 * (self.cluster_size[:h] + self.cluster_size[h:])
        a = 0.5 * (self.embed_avg[:h] + _zam(self.embed_avg[h:]))
        self.cluster_size.copy_(torch.cat([n, n], dim=0))
        self.embed_avg.copy_(torch.cat([a, _zam(a)], dim=0))
        m = self.cluster_size.sum()
        c = ((self.cluster_size + self.eps)
             / (m + self.codebook_size * self.eps) * m)
        nowe = self.embed_avg / c.unsqueeze(-1)
        nn_ = nowe.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.embed.copy_(nowe * (torch.clamp(nn_, max=self.dlugosc) / nn_))
        return self.embed

    @torch.no_grad()
    def z_indeksow(self, baza, idx):
        """Odtworzenie kroku z SAMEGO INDEKSU -- sciezka dyfuzji."""
        return self._krok(baza, self.embed[idx])

    @torch.no_grad()
    def perplexity(self):
        """Perplexity Z EMA, nie z batcha -- umowa `band_report`.

        `band_report` szuka warstw po `hasattr(m, "cluster_size") and
        hasattr(m, "perplexity")`. Bez tej metody poziomy styczne bylyby
        w raporcie NIEWIDOCZNE, a warunkiem NIEZBEDNYM alarmu zapadu jest
        spadek wykorzystania kodow -- czyli alarm bylby slepy dokladnie na
        tych poziomach, ktore ta przebudowa zmienia. To ta sama slepota,
        ktora `band_report` juz raz w tym repo zamknal.
        """
        p = self.cluster_size / self.cluster_size.sum().clamp_min(1e-12)
        return float(torch.exp(-(p * (p + 1e-10).log()).sum()))

    def raport(self):
        return dict(self._diag)


class DrabinaStozkowa(nn.Module):
    """Zlozenie: pozycja na poziomie 0, uszczegolowienia na kolejnych.

    Poziom 0 uzywa PODANEGO kwantyzatora pozycji (`HyperbolicVQ`), bo jego
    wpisy sa punktami hiperboloidy -- i to one sa tensorami NA ROZMAITOSCI,
    ktore `RiemannianAdam` moze optymalizowac. Poziomy dalsze uzywaja
    `KwantyzatorStyczny`, ktorego wpisy sa wektorami stycznymi w origo,
    czyli obiektami euklidesowymi -- i tak nalezy je optymalizowac.

    To rozdzielenie nie jest kompromisem, tylko poprawnym typowaniem: pozycja
    lezy na rozmaitosci, uszczegolowienie lezy w przestrzeni stycznej.

    ====== KTORE LICZBY W LOGU NIE SA TU OBSERWABLAMI (audyt 10.09.2026) ======
    Falsyfikator, ktory nie moze sie zapalic, jest GORSZY od jego braku. Przy
    tej konstrukcji nastepujace klucze logu sa **niezapalalne** i nie wolno
    czytac ich wartosci jako wyniku:

      `dr_poza_stozkiem`  ZERO ARYTMETYCZNIE. Skladowa styczna kroku ma norme
                          najwyzej `dlugosc`, wiec kat <= dlugosc / sinh(r);
                          przy 0,25-0,5 wobec sinh(1,44) = 1,97 sufit to
                          0,13-0,25 przy aperturze 1,0. Do tego
                          `waga_stozka = 0`. Czytac `kat_wzg`, `kat_maks`,
                          `kat_sufit` i `stozek_moze` (ten ostatni mowi
                          wprost, czy pytanie jest zadawalne).
      `kor_reszty`,       NaN Z KONSTRUKCJI: sa liczone wylacznie w galezi
      `udzial_sekw`       REZYDUALNEJ (`hyp_vq_net.py`, `HyperbolicCodeStack.forward`,
                          przypisanie `self._kor_reszty[k]`), a drabina
                          stozkowa nie ma reszt. Lista jest inicjowana NaN-ami
                          i drukowana bezwarunkowo (`band_report`, `getattr(
                          self.codes, "_kor_reszty")`), wiec zasmieca kazda
                          linie logu i uczy czytelnika ignorowac NaN-y.
                          Poprawka nalezy do `hyp_vq_net.py`, nie tutaj.
      `enc_radius`,       JEDNA LICZBA: przy `enc_na_powloce` promien punktu
      `radius_mean`,      enkodera jest USTAWIANY na `promien_cel`
      `powloka`           (`hyperbolic.py`, `HyperbolicVQ._na_powloke`),
                          a kodebook jest tam przypinany przez
                          `renormalise_shells()`. Roznica jest zerem
                          z konstrukcji. Czytac `l0_d_enk` wobec `l0_d_kod`
                          (odleglosc KATOWA, nie promieniowa) oraz
                          `l0_r_tozs`, ktory raportuje sama tozsamosc.
      `emp_H_sufit`       liczony jako `log2(B)` z OSTATNIEGO, czesciowego
                          batcha (`hyperbolic.py`,
                          `HyperbolicVQ.uzycie_empiryczne`), wiec skacze
                          miedzy epokami bez zmiany rozkladu kodow. Poprawka
                          nalezy do `hyperbolic.py`.

    (Numery linii sa tu SWIADOMIE pominiete: `hyp_vq_net.py` jest w tej sesji
    edytowany rownolegle i kazdy numer zdazyl sie przesunac miedzy grepem
    a zapisem tego komentarza.)
    """

    def __init__(self, vq_pozycji, kwantyzatory_styczne, waga_stozka=0.0,
                 apertura=1.0):
        super().__init__()
        self.vq0 = vq_pozycji
        self.kroki = nn.ModuleList(kwantyzatory_styczne)
        self.n_levels = 1 + len(self.kroki)
        # ============ KARA STOZKOWA (ENTAILMENT) ============
        # ZMIERZONY DEFEKT: `sonda_zawierania.py` na HypStozek_best dala
        #     przejscie   d_rodzic  d_losowy  iloraz  rozrzut  krok/rozrzut
        #     L0->L1       0,3985    0,5316   1,334   0,2657     1,500
        #     L1->L2       0,9379    1,0887   1,161   0,5649     1,660
        # przy kontroli (drabina rezydualna) ilorazie DOKLADNIE 1,000.
        # Zawieranie POWSTALO, ale `krok/rozrzut > 1` znaczy, ze dziecko moze
        # doskoczyc do DOWOLNEGO rodzica -- warunek nie ogranicza POZYCJI.
        #
        # Kara dziala na SKLADOWEJ KATOWEJ kroku, nie na jego dlugosci.
        # Rozpietosc katowa kroku o skladowej stycznej `t` z promienia `r`
        # wynosi okolo `||t|| / sinh(r)`; to jest wlasnie apertura stozka,
        # o ktorej mowi naglowek tego modulu. Kara to
        #     relu(||t|| / sinh(r_rodzica) - apertura)^2
        # czyli zero wewnatrz stozka i kwadratowa poza nim. Ograniczenie
        # `dlugosc` jest TWARDE i dotyczy DLUGOSCI; ta kara jest MIEKKA
        # i dotyczy KATA -- to dwie rozne wielkosci i dopiero razem
        # realizuja "ten sam kod znaczy mniej, im dalej od origa".
        self.waga_stozka = float(waga_stozka)
        self.apertura = float(apertura)
        self.punkty = None
        self.indices = None
        self.loss = None
        self.strata_stozka = None

    def forward(self, mapy):
        """`mapy`: lista [B, dim, L_k], od NAJGRUBSZEJ. Zwraca punkty i indeksy."""
        if len(mapy) != self.n_levels:
            raise ValueError("oczekiwano %d map, dostalem %d"
                             % (self.n_levels, len(mapy)))
        p, idx0 = self.vq0(mapy[0].double())
        # ============ DWIE DROGI GRADIENTU DO JEDNEJ WARTOSCI ============
        # `st_points` to `points + (quant - points).detach()`: WARTOSC jest
        # wpisem kodebooka, ale gradient idzie WYLACZNIE do enkodera. To jest
        # dokladnie ten detach, przez ktory rekonstrukcja NIGDY nie docierala
        # do kodebooka -- zmierzony skutek: rozrzut promieni kodow zapadl do
        # sd 0,0002 (L0) i 0,0023 (L1), bo kodebook widzial tylko czlon
        # commitment ciagnacy go DO POWLOKI.
        #
        # `embed[idx]` ma te sama WARTOSC, ale gradient idzie do KODEBOOKA.
        # Suma `embed[idx] + (st - st.detach())` ma wartosc `embed[idx]`
        # (drugi skladnik jest DOKLADNIE zerem), a gradient plynie w OBIE
        # strony. Dopiero wtedy `RiemannianAdam` ma co optymalizowac -- do
        # dzis `manifold_parameters()` zwracalo pusta liste i flaga
        # `riemannian: true` byla cichym no-opem.
        _st = self.vq0.st_points
        _kod = self.vq0.embed[idx0]
        baza = _kod + (_st - _st.detach())
        punkty, indeksy = [baza], [idx0]
        strata = self.vq0.loss.float()
        for k, kr in enumerate(self.kroki):
            # Rodzic musi byc podniesiony do rozdzielczosci dziecka. Skalowanie
            # SYMETRYCZNE NA ODBICIE (`linear`), bo `nearest` uzywa `floor`
            # i lamie ekwiwariancje RC -- zmierzone w tej samej sesji.
            b = baza
            if b.shape[1] != mapy[k + 1].shape[-1]:
                b = F.interpolate(b.transpose(1, 2).float(),
                                  size=mapy[k + 1].shape[-1], mode="linear",
                                  align_corners=False).transpose(1, 2)
                b = project_to_hyperboloid(b[..., 1:].to(baza.dtype))
            _rodzic = b
            baza, idx = kr(b, mapy[k + 1])
            punkty.append(baza)
            indeksy.append(idx)
            strata = strata + kr.loss
            if self.waga_stozka > 0.0:
                strata = strata + self.waga_stozka * self._kara_stozka(
                    _rodzic, baza)
        self.punkty = punkty
        self.indices = indeksy
        self.loss = strata
        return punkty, indeksy

    def _kara_stozka(self, rodzic, dziecko):
        """`relu(kat - apertura)^2`, gdzie kat ~ ||skladowa styczna|| / sinh(r).

        MIERZY KAT, NIE DLUGOSC. Skladowa RADIALNA kroku jest dozwolona bez
        ograniczenia -- to ona realizuje zejscie na glebsza powloke i jej
        karanie zabilo by rozwarstwienie, o ktore cala konstrukcja idzie.
        Karana jest wylacznie skladowa KATOWA, i to znormalizowana przez
        `sinh(r_rodzica)`, wiec ten sam krok w bezwzglednych jednostkach jest
        na glebszej powloce TANSZY -- dokladnie jak w geometrii hiperbolicznej.
        """
        from .hyperbolic import logmap
        v = logmap(rodzic, dziecko)                      # wektor styczny w rodzicu
        rad = self._promieniowy_pkt(rodzic)
        # rzut na kierunek promieniowy w metryce Minkowskiego (rad jest
        # jednostkowy, wiec wspolczynnik to samo <v, rad>_L)
        wsp = minkowski_dot(v, rad)
        styczna = v - wsp * rad
        n_st = torch.sqrt(minkowski_dot(styczna, styczna).clamp_min(1e-30))
        sinh_r = torch.sinh(radius(rodzic).unsqueeze(-1)).clamp_min(1e-6)
        kat = n_st / sinh_r
        return F.relu(kat - self.apertura).pow(2).mean().float()

    @staticmethod
    def _promieniowy_pkt(baza):
        """Jednostkowy wektor styczny wskazujacy na zewnatrz -- jak w krokach."""
        from .hyperbolic import project_to_tangent
        sp = baza[..., 1:]
        u = sp / sp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        amb = torch.cat([torch.zeros_like(baza[..., :1]), u], dim=-1)
        t = project_to_tangent(baza, amb)
        return t / torch.sqrt(minkowski_dot(t, t).clamp_min(1e-12))

    @torch.no_grad()
    def z_indeksow(self, indeksy, dlugosci=None):
        """Punkty z SAMYCH INDEKSOW -- to jest umowa dyfuzji."""
        baza = self.vq0.embed[indeksy[0]]
        punkty = [baza]
        for k, kr in enumerate(self.kroki):
            b = baza
            if dlugosci is not None and b.shape[1] != dlugosci[k + 1]:
                b = F.interpolate(b.transpose(1, 2).float(),
                                  size=dlugosci[k + 1], mode="linear",
                                  align_corners=False).transpose(1, 2)
                b = project_to_hyperboloid(b[..., 1:].to(baza.dtype))
            baza = kr.z_indeksow(b, indeksy[k + 1])
            punkty.append(baza)
        return punkty

    def raport(self):
        # BRAK PRZEBIEGU = PUSTY RAPORT, nie wyjatek. `band_report` wola to
        # w bloku `try: ... except Exception: continue`, wiec jeden `TypeError`
        # na `None` gubi CALA linie falsyfikatorow, nie tylko klucze drabiny.
        if not self.punkty:
            return {}
        # `.detach()` OBOWIAZKOWO: `self.punkty` sa czescia grafu, a `float()`
        # na tensorze z gradientem wypisuje UserWarning w kazdej epoce i zasmieca
        # log, w ktorym szuka sie falsyfikatorow.
        r = {"promien": [round(float(radius(p.detach()).mean()), 4)
                         for p in self.punkty]}
        # ========== PLAN PROMIENIA WOBEC POMIARU: DWIE ROZNE LICZBY ==========
        # `radius_mean`, `enc_radius` i `powloka` to w tym modelu JEDNA liczba
        # (przy `enc_na_powloce` promien enkodera jest ustawiany na
        # `promien_cel`, a `renormalise_shells` przypina tam kodebook), wiec
        # falsyfikator "rozjazd enkoder-kodebook" nie moze sie zapalic.
        # Rozwarstwienie drabiny NIE jest niczym przypiete i jest wielkoscia
        # NIEZALEZNA: plan to `r_{k+1} = r_k + krok_promienia`, a pomiar
        # wychodzi z geodezyjnego zlozenia. ZMIERZONE w HypStozek: plan
        # 1,44 / 1,94 / 2,44 wobec pomiaru 1,437 / 1,499 / 1,811, czyli kroki
        # radialne 0,06 i 0,31 zamiast 0,5 -- model TRZECI raz w tym repo
        # zwinal mechanizm rozwarstwiajacy. Ta liczba byla dotad wyliczana
        # RECZNIE poza logiem, wiec falsyfikator istnial tylko w notatce.
        _plan = [r["promien"][0]]
        _nom = []
        for kr in self.kroki:
            _s = getattr(kr, "skala_promienia", None)
            if _s is None:
                _s = float(getattr(kr, "krok_promienia", 0.0))
            else:
                _s = float(_s.detach())
            _nom.append(round(_s, 4))
            _plan.append(_plan[-1] + _s)
        r["promien_plan"] = [round(v, 4) for v in _plan]
        r["promien_luka"] = [round(_plan[i] - r["promien"][i], 4)
                             for i in range(len(r["promien"]))]
        r["krok_prom"] = [round(r["promien"][i + 1] - r["promien"][i], 4)
                          for i in range(len(r["promien"]) - 1)]
        r["krok_prom_nom"] = _nom
        for k, kr in enumerate(self.kroki):
            for a, b in kr.raport().items():
                r.setdefault(a, []).append(b)
        with torch.no_grad():
            # FALSYFIKATOR ZAWIERANIA: cos(kierunek dziecka, kierunek rodzica)
            # wobec KONTROLI LOSOWEGO PAROWANIA. Dzis (drabina rezydualna)
            # obie liczby sa rowne: 0,0981 i 0,0981. Jesli po tej przebudowie
            # nadal beda rowne, zawieranie NIE POWSTALO i cala konstrukcja
            # jest ozdoba -- niezaleznie od tego, co pokaze PCC.
            # ============ KAT KROKU WOBEC APERTURY ============
            # `poza_stozkiem` jest ZEREM ARYTMETYCZNIE, i to jest TOZSAMOSC
            # STRUKTURALNA, nie wynik: skladowa styczna kroku ma norme
            # najwyzej `dlugosc` (kwantyzator obcina), wiec
            #     kat = ||styczna|| / sinh(r_rodzica) <= dlugosc / sinh(r)
            # i przy `dlugosc` 0,25-0,5 wobec `sinh(1,44) = 1,97` sufit kata
            # wynosi 0,13-0,25 przy aperturze 1,0. Do tego `waga_stozka = 0`
            # w kazdym configu tej rodziny, czyli kara nie liczy sie wcale.
            # Falsyfikator nie moze sie zapalic -- i to jest gorsze od jego
            # braku, bo `poza_stozkiem = 0,0000` czyta sie jako "stozek
            # trzyma", a znaczy "pytanie jest niezadawalne".
            #
            # DOKLADANE OBOK (stare klucze zostaja, sa w historii biegow):
            #   `kat_sufit`   -- `dlugosc_k / sinh(min r_rodzica)`, czyli
            #                    arytmetyczny sufit `kat` na tym przejsciu;
            #   `stozek_moze` -- 1, gdy `kat_sufit > apertura` I
            #                    `waga_stozka > 0`, czyli gdy `poza_stozkiem`
            #                    MOZE byc niezerowe; 0 znaczy: nie czytaj
            #                    `poza_stozkiem` jako wyniku;
            #   `kat_wzg`     -- `kat / apertura` (srednio) i
            #   `kat_maks`    -- `max kat / apertura`. Te dwie MIERZA, bo nic
            #                    ich nie przypina: rosna, gdy krok wypelnia
            #                    stozek, i to jest wersja pytania "czy model
            #                    dusi sie apertura", ktora da sie zapalic.
            r["kat"], r["poza_stozkiem"] = [], []
            r["kat_wzg"], r["kat_maks"] = [], []
            r["kat_sufit"], r["stozek_moze"] = [], []
            r["waga_stozka"] = round(float(self.waga_stozka), 4)
            r["apertura"] = round(float(self.apertura), 4)
            for a_, b_ in zip(self.punkty[:-1], self.punkty[1:]):
                pa = a_.detach()
                if pa.shape[1] != b_.shape[1]:
                    _q = F.interpolate(pa.transpose(1, 2).float(),
                                       size=b_.shape[1], mode="linear",
                                       align_corners=False).transpose(1, 2)
                    pa = project_to_hyperboloid(_q[..., 1:].to(b_.dtype))
                from .hyperbolic import logmap
                v = logmap(pa, b_.detach())
                rad = self._promieniowy_pkt(pa)
                st_ = v - minkowski_dot(v, rad) * rad
                n_st = torch.sqrt(minkowski_dot(st_, st_).clamp_min(1e-30))
                kat = n_st / torch.sinh(radius(pa).unsqueeze(-1)).clamp_min(1e-6)
                r["kat"].append(round(float(kat.mean()), 4))
                r["poza_stozkiem"].append(
                    round(float((kat > self.apertura).double().mean()), 4))
                _ap = max(float(self.apertura), 1e-12)
                r["kat_wzg"].append(round(float(kat.mean()) / _ap, 4))
                r["kat_maks"].append(round(float(kat.max()) / _ap, 4))
                _k = len(r["kat"]) - 1
                _dl = float(getattr(self.kroki[_k], "dlugosc", float("nan")))
                _sh = float(torch.sinh(radius(pa)).min().clamp_min(1e-6))
                _suf = _dl / _sh
                r["kat_sufit"].append(round(_suf, 4))
                r["stozek_moze"].append(
                    1 if (_suf > self.apertura and self.waga_stozka > 0.0) else 0)
            r["cos_rodzic"], r["cos_losowy"] = [], []
            for a, b in zip([p.detach() for p in self.punkty[:-1]],
                            [p.detach() for p in self.punkty[1:]]):
                ka = a[..., 1:]
                ka = ka / ka.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                kb = b[..., 1:]
                kb = kb / kb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                if ka.shape[1] != kb.shape[1]:
                    ka = F.interpolate(ka.transpose(1, 2).float(),
                                       size=kb.shape[1], mode="linear",
                                       align_corners=False).transpose(1, 2)
                    ka = ka / ka.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                ka = ka.to(kb.dtype)
                perm = torch.randperm(ka.shape[1], device=ka.device)
                r["cos_rodzic"].append(
                    round(float((ka * kb).sum(-1).mean()), 4))
                r["cos_losowy"].append(
                    round(float((ka[:, perm] * kb).sum(-1).mean()), 4))
                # ============ WERSJA UCENTROWANA ============
                # ZMIERZONE, dlaczego surowa wersja nie wystarcza: na LOSOWYCH
                # mapach `cos_losowy` wynosi 0,10 i separacja to +0,89, ale na
                # prawdziwym enkoderze (HypStozek ep19) kontrola siedzi na
                # 0,9597 / 0,9019, a separacja spada do +0,015 / +0,030.
                # Przyczyna: mapy enkodera maja silna skladowa WSPOLNA miedzy
                # pozycjami, wiec losowo sparowany "rodzic" wskazuje prawie ten
                # sam kierunek co wlasciwy. Kontrola jest wysycona i miara
                # traci rozdzielczosc DOKLADNIE tam, gdzie jest potrzebna --
                # ta sama pulapka co `geo_mi_bits`, gdzie poziom szumu
                # przewyzszal wszystkie obserwacje.
                #
                # Odjecie sredniej po POZYCJACH I PO BATCHU zostawia to, o co
                # pytamy: czy dziecko idzie za SWOIM rodzicem, a nie za
                # kierunkiem wspolnym dla calej mapy. Normalizacja po
                # centrowaniu, bo inaczej dlugosc reszty mieszalaby sie z katem.
                def _ucentruj(t):
                    u = t - t.mean(dim=(0, 1), keepdim=True)
                    return u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                kac, kbc = _ucentruj(ka), _ucentruj(kb)
                r.setdefault("cos_rodzic_c", []).append(
                    round(float((kac * kbc).sum(-1).mean()), 4))
                r.setdefault("cos_losowy_c", []).append(
                    round(float((kac[:, perm] * kbc).sum(-1).mean()), 4))
                # ====== WLASCIWY PRZYRZAD: ODLEGLOSC GEODEZYJNA ======
                # Kosinus kierunkow -- surowy I ucentrowany -- jest ZLYM
                # przyrzadem, i to jest wyprowadzenie, nie przypuszczenie.
                # Na powloce o promieniu r dwa punkty w STALEJ odleglosci
                # geodezyjnej maja kat tym mniejszy, im wieksze r (kat ~ d /
                # sinh r). Kosinus miesza wiec odleglosc z promieniem.
                # ZMIERZONE na sztucznej skladowej wspolnej o amplitudzie 10:
                # surowy daje 0,9952 wobec kontroli 0,9952 (wysycenie), a
                # ucentrowany -0,0028 wobec 0,0130 -- czyli "brak zawierania"
                # tam, gdzie zawieranie jest Z KONSTRUKCJI, bo krok ma dlugosc
                # najwyzej `dlugosc`. Oba klamia, w przeciwne strony.
                #
                # Odleglosc geodezyjna dziecko-RODZIC wobec dziecko-LOSOWY
                # rodzic nie ma tej wady: jest niezmiennicza na izometrie,
                # wiec skladowa wspolna (obrot i przesuniecie wzdluz
                # rozmaitosci) jej nie rusza.
                if a.shape[1] == b.shape[1]:
                    _pa = a
                else:
                    # Rodzic na rozdzielczosc dziecka -- `linear`, tak samo jak
                    # w `forward`. `nearest` uzywa `floor` i nie komutuje
                    # z odbiciem, wiec dalby inne pary niz te, ktore realnie
                    # powstaly.
                    _q = F.interpolate(a.transpose(1, 2).float(),
                                       size=b.shape[1], mode="linear",
                                       align_corners=False).transpose(1, 2)
                    _pa = project_to_hyperboloid(_q[..., 1:].to(b.dtype))
                _dr = lorentz_distance(_pa, b)
                _dl = lorentz_distance(_pa[:, perm], b)
                r.setdefault("d_rodzic", []).append(round(float(_dr.mean()), 4))
                r.setdefault("d_losowy", []).append(round(float(_dl.mean()), 4))
                r.setdefault("d_iloraz", []).append(
                    round(float(_dl.mean() / _dr.mean().clamp_min(1e-9)), 3))
        # ===== L0: ROZJAZD ENKODER-KODEBOOK, WERSJA NIE-TOZSAMOSCIOWA =====
        # `enc_radius`, `radius_mean` i `powloka` to JEDNA liczba: przy
        # `enc_na_powloce` metoda `_na_powloke` USTAWIA promien punktu na
        # `promien_cel`, a `renormalise_shells()` przypina tam kodebook co
        # krok. Ich roznica jest wiec zerem Z KONSTRUKCJI i falsyfikator
        # "enkoder odjechal od kodebooka" nie moze sie zapalic -- `l0_r_tozs`
        # mowi to WPROST, zamiast pozwolic czytac zero jako zdrowie.
        #
        # Wielkoscia, ktora TE SAMO pytanie zadaje zapalalnie, jest odleglosc
        # KATOWA: punkt enkodera i jego kod leza na tej samej powloce, ale nie
        # w tym samym miejscu. `l0_d_enk` to sqrt(2 * commit_raw) (bo
        # `lorentz_sqdist` = d^2/2 w wiodacym rzedzie), a `l0_d_kod` to
        # MEDIANA odleglosci miedzy kodami. Iloraz >= 1 znaczy, ze enkoder
        # jest dalej od swojego kodu niz kody sa od siebie, czyli przypisanie
        # przestalo isc za kierunkiem -- zapisane w tym repo jako zmierzone
        # 0,1371 / 0,3008 / 0,8413 dla odleglosci miedzy kodami.
        with torch.no_grad():
            try:
                _v0 = self.vq0
                _emb = _v0.embed.detach()
                _cm = getattr(_v0, "_commit_raw", None)
                _de = (math.sqrt(max(2.0 * float(_cm), 0.0))
                       if _cm is not None else float("nan"))
                # Podprobka DETERMINISTYCZNA (co n-ty wpis). `randperm` tutaj
                # przesunalby globalny strumien RNG, a od niego zalezy
                # powtarzalnosc ziarna 7898 -- raport nie ma prawa dotykac
                # przebiegu.
                _st = max(1, _emb.shape[0] // 128)
                _kk = _emb[::_st][:128]
                _dm = lorentz_distance(_kk.unsqueeze(1), _kk.unsqueeze(0))
                _msk = ~torch.eye(_kk.shape[0], dtype=torch.bool,
                                  device=_dm.device)
                _dk = float(_dm[_msk].median())
                r["l0_d_enk"] = round(_de, 4)
                r["l0_d_kod"] = round(_dk, 4)
                r["l0_iloraz"] = round(_de / max(_dk, 1e-9), 3)
                # TRZY PROMIENIE, i tylko JEDNA para z nich jest tozsamoscia:
                #   `l0_r_kod` -- srednia po WSZYSTKICH wpisach; to jest to
                #                 samo, co `radius_mean` w `band_report`,
                #                 i to z NIM promien enkodera jest zlepiony
                #                 (`_na_powloke` ustawia go na `promien_cel`,
                #                 czyli na te srednia z inicjalizacji);
                #   `l0_r_uz`  -- srednia po wpisach REALNIE WYBRANYCH. To
                #                 jest inna liczba i nikt jej nie przypina;
                #   `l0_r_sel` -- roznica enkoder minus wybrane, czyli
                #                 STRONNICZOSC PRZYPISANIA. Zmierzone
                #                 w bramce na inicjalizacji: +0,1097, czyli
                #                 przypisanie systematycznie wybiera powloke
                #                 WEWNETRZNIEJSZA niz ta, na ktorej lezy
                #                 enkoder -- liczba, ktorej para
                #                 `enc_radius`/`radius_mean` nie mogla pokazac.
                _rk = float(radius(_emb).mean())
                _ru = float(radius(_emb[self.indices[0].reshape(-1)]).mean())
                r["l0_r_kod"] = round(_rk, 4)
                r["l0_r_uz"] = round(_ru, 4)
                _er = getattr(_v0, "_enc_radius", None)
                if _er is not None:
                    r["l0_r_enk"] = round(float(_er[0]), 4)
                    r["l0_r_luka"] = round(float(_er[0]) - _rk, 4)
                    r["l0_r_sel"] = round(float(_er[0]) - _ru, 4)
                r["l0_r_tozs"] = int(
                    bool(getattr(_v0, "enc_na_powloce", False))
                    and float(getattr(_v0, "pasmo_r", 0.0)) == 0.0
                    and float(getattr(_v0, "promien_specyf", 0.0)) == 0.0)
            except Exception as _e:
                # Raport NIE MOZE rzucic: `band_report` lapie wyjatek i gubi
                # CALA linie falsyfikatorow, a nie tylko te klucze.
                r["l0_blad"] = type(_e).__name__
        return r
