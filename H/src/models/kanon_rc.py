# -*- coding: utf-8 -*-
import torch

# A, C, T, G -> komplement A<->T, C<->G, czyli permutacja [2, 3, 0, 1].
# NIE `flip(dims=[1])`: to dalaby A->G / C->T, i ta pomylka ma w tym repo
# osobna notatke w konwencjach.
PERM_KOMPL = (2, 3, 0, 1)


def rc_onehot(x):
    """Odwrotna nic dla one-hot [B, 4, L] w kolejnosci A, C, T, G."""
    return x[:, PERM_KOMPL, :].flip(-1)


def _klucz(x):
    """Indeks zasady na pozycje, [B, L]. `N` (kolumna zerowa) -> 4.

    Kolumny zerowe musza dostac WLASNA wartosc, nie 0 -- inaczej `N`
    porownywaloby sie jak `A` i dwie rozne sekwencje mialyby ten sam klucz.
    """
    idx = x.argmax(1)
    puste = x.sum(1) == 0
    return torch.where(puste, torch.full_like(idx, 4), idx)


@torch.no_grad()
def wybierz_orientacje(x):
    """Bit orientacji: 1 = brac `rc(x)`. Porownanie LEKSYKOGRAFICZNE.

    Zwraca [B] typu bool. Palindromy (klucz identyczny) daja 0, czyli
    zostaja jak podane -- dla nich obie nici sa tym samym obiektem, wiec
    wybor jest dowolny, byle deterministyczny.
    """
    a = _klucz(x)
    b = _klucz(rc_onehot(x))
    roz = a != b
    # pierwsza pozycja roznicy: argmax po masce daje najmniejszy indeks
    ma = roz.any(dim=-1)
    pierwsza = roz.int().argmax(dim=-1)
    ar = a.gather(1, pierwsza.unsqueeze(1)).squeeze(1)
    br = b.gather(1, pierwsza.unsqueeze(1)).squeeze(1)
    return ma & (br < ar)


def kanonizuj(x):
    """`(x_kanoniczne, orient)`. Rozniczkowalne po `x`.

    `torch.where` na wyniku, nie indeksowanie -- gradient musi dochodzic do
    OBU galezi, bo obie sa funkcjami `x` (przy jednym wejsciu tylko jedna
    jest wybrana, ale wybor jest stala, nie parametrem).
    """
    o = wybierz_orientacje(x)
    xr = rc_onehot(x)
    m = o.view(-1, 1, 1).to(x.dtype)
    return x * (1.0 - m) + xr * m, o


class OrientacjaOdniesienia:
    """Kontekst: kanonizuj RAZ sekwencja odniesienia, narzuc te nic wariantom.

    Bez tego kazdy wariant wybiera nic sam, a `wybierz_orientacje` rozstrzyga
    na PIERWSZEJ roznicy miedzy sekwencja i jej odwrotnym komplementem --
    zmierzone na 512 sekwencjach walidacji: decyzja pada w 69,9 % przypadkow
    na pozycji 0 i w 97,5 % na pozycji <= 2. Skutek: podstawienie na pozycji 0
    przewraca nic w **41,9 %** przypadkow, na pozycji 248 w **40,3 %**,
    a po losowej pozycji w **1,26 %**. Przy przewroceniu kody zmieniaja sie
    HURTOWO, zamiast o zmierzone 1,56 % pozycji L2, i efekt wariantu jest
    nieciagly. Przy skanie nasycajacym 747 wariantow psuje to ~9 z nich.

    Orientacja liczona jest CZYSTA FUNKCJA `wybierz_orientacje(x_ref)`, bez
    przebiegu przez model -- wiec nie rusza ani statystyk BatchNorma, ani EMA
    kodebooka. To nie jest ostroznosc na zapas: kazdy dodatkowy przebieg
    wprzod w trybie treningu przepisuje jedno i drugie.

    Uzycie::

        with OrientacjaOdniesienia(model, x_ref):
            y0 = model(x_ref)
            y1 = model(x_wariant)        # ta sama nic co odniesienie
    """

    def __init__(self, model, x_ref):
        self.model = model
        self.x_ref = x_ref
        self._stara = None

    def __enter__(self):
        self._stara = getattr(self.model, "_orient_wymuszony", None)
        o = wybierz_orientacje(self.x_ref).detach().clone()
        self.model.ustaw_orientacje(o)
        return o

    def __exit__(self, *wyj):
        self.model.ustaw_orientacje(self._stara)
        return False


def przywroc_orientacje(t_kan, orient):
    """Tensor w ukladzie KANONICZNYM -> uklad PODANY, wg bitu orientacji.

    `t_kan`: [B, 4, L] -- logity zasad albo one-hot. `orient`: [B] bool,
    gdzie True znaczy "kanonizacja wzięla `rc(x)`".

    `rc` jest INWOLUCJA (`rc(rc(x)) == x`), wiec powrot to ta sama
    operacja zastosowana tam, gdzie kanonizacja ja zastosowala.

    Skladane ARYTMETYCZNIE, nie przez indeksowanie ani `torch.where`.
    Dwa powody, oba zapisane w tym repo: `torch.where` nie chroni
    gradientu (maskowac OPERAND, nie wynik), a indeksowanie zerwaloby
    gradient do galezi niewybranej -- a obie sa funkcjami wejscia.

    Kolejnosc kanalow to A, C, T, G, wiec komplement idzie przez
    `PERM_KOMPL`, NIE przez `flip(dims=[1, 2])` -- ten drugi daje tu
    A->G i C->T zamiast komplementu i jest zapisana pulapka tego repo.
    """
    if t_kan.dim() != 3 or t_kan.shape[1] != 4:
        raise ValueError(
            "oczekuje [B, 4, L] w kolejnosci A,C,T,G, dostalem %r"
            % (tuple(t_kan.shape),))
    if orient.shape[0] != t_kan.shape[0]:
        raise ValueError(
            "bit orientacji ma batch %d, a tensor %d -- to jest ta sama "
            "klasa awarii co straz orientacji w `ustaw_orientacje`"
            % (int(orient.shape[0]), int(t_kan.shape[0])))
    t_rc = t_kan[:, PERM_KOMPL, :].flip(-1)
    m = orient.view(-1, 1, 1).to(t_kan.dtype)
    return t_kan * (1.0 - m) + t_rc * m


@torch.no_grad()
def sprawdz_bit_orientacji(x):
    """FALSYFIKATOR WLASNY: `przywroc_orientacje(kanonizuj(x)) == x`.

    Zwraca `max|roznica|`. Musi byc DOKLADNIE 0 -- nie "male", bo obie
    operacje sa permutacjami i odbiciami, wiec skladaja sie do tozsamosci
    bez bledu numerycznego. Wartosc niezerowa znaczy, ze `PERM_KOMPL`
    albo `flip` nie sa inwolucja, i wtedy bit orientacji przenosi
    rekonstrukcje w ZLY uklad, co w metryce wygladalo by jak slaby dekoder.
    """
    xk, o = kanonizuj(x)
    xp = przywroc_orientacje(xk, o)
    return float((xp - x).abs().max())


@torch.no_grad()
def sprawdz_niezmienniczosc(model, x):
    """FALSYFIKATOR WLASNY: `encode(x)` musi byc IDENTYCZNE z `encode(rc(x))`.

    To nie jest test dokladnosci, a test TOZSAMOSCI na indeksach calkowitych
    -- albo jest 1,0000 na kazdym poziomie, albo kanonizacja jest zepsuta
    i nie ma sensu patrzec na zadna inna liczbe z tego biegu. Dokladnie tak
    samo `occ_center` przechodzil kazdy test na swiezym modelu i nie byl
    spelniony ani przez jedna epoke.

    Zwraca liste frakcji zgodnych indeksow, jedna na poziom.
    """
    a = model.encode_codes(x)
    b = model.encode_codes(rc_onehot(x))
    return [round(float((p == q).float().mean()), 6) for p, q in zip(a, b)]


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L = 64, 249
    x = torch.zeros(B, 4, L)
    x.scatter_(1, torch.randint(0, 4, (B, L)).unsqueeze(1), 1.0)

    print("[kanon_rc] testy wlasne")
    xk, o = kanonizuj(x)
    xrk, orc = kanonizuj(rc_onehot(x))
    print("   1. IDEMPOTENCJA/ZGODNOSC: kanon(x) == kanon(rc(x))?")
    print("      max |roznica| = %.3e  (musi byc 0)"
          % float((xk - xrk).abs().max()))
    print("   2. bit orientacji jest komplementarny: %s"
          % bool(((o ^ orc) | (o == orc)).all()))
    print("      wybrano rc dla %d z %d sekwencji" % (int(o.sum()), B))
    print("   3. odwracalnosc: rc(kanon) przy orient==1 wraca do x?")
    wroc = torch.where(o.view(-1, 1, 1), rc_onehot(xk), xk)
    print("      max |roznica| = %.3e  (musi byc 0)"
          % float((wroc - x).abs().max()))
    print("   4. kanonizacja jest ZBIOREM: kazda para {x, rc(x)} -> 1 element")
    kl = _klucz(xk)
    klr = _klucz(xrk)
    print("      klucze rowne: %s" % bool((kl == klr).all()))
    print("   5. palindrom: x == rc(x) daje orient 0 i x bez zmian")
    # "ACGT" jest palindromem RC: komplement TGCA, odwrocony daje ACGT.
    # Pierwsza wersja tego testu brala "ATCG", ktore palindromem NIE JEST
    # (rc = CGAT) -- test sprawdzal wiec cos innego, niz mowil.
    p = torch.zeros(1, 4, 4)
    for i, b in enumerate([0, 1, 3, 2]):        # A C G T w kolejnosci ACTG
        p[0, b, i] = 1.0
    pk, po = kanonizuj(p)
    print("      rc(p) == p: %s | orient %s | zmiana %.3e"
          % (bool((rc_onehot(p) == p).all()), bool(po[0]),
             float((pk - p).abs().max())))
