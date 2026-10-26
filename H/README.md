# HypBaza

Model przewiduje aktywność enhancera (STARR-seq, komórki S2 *D. melanogaster*)
dla sekwencji **249 nt**, osobno dla promotora rozwojowego (**Dev**) i
housekeeping (**Hk**). Wyjście w jednostkach `log2 enrichment`.

Pakiet jest samodzielny nie wymaga nic więcej.

## Start

```bash
pip install torch numpy pyyaml
python test_install.py              # musi wypisać "All checks passed"
python predict.py --input seqs.fa --output pred.tsv
```

```python
from deepstarr import DeepSTARRModel
model = DeepSTARRModel()
out = model.predict(["ACGT...249nt..."])    # {"Dev": array, "Hk": array}
```

Wczytanie modelu ~3 s, więc twórz obiekt raz. Predykcja: 20 tys. sekwencji
w 13 s na GPU.

## Wyniki na teście

Split testowy DeepSTARR, 20 593 loci, RC-TTA (`results/test_report.txt`):

| | Dev | Hk | mean |
|---|---|---|---|
| **HypBaza** | **0,7333** | **0,7954** | **0,7643** |
| DeepSTARR (publikacja), te same loci | 0,6757 | 0,7450 | 0,7104 |

Spearman: 0,6947 (Dev) / 0,6246 (Hk). MSE: 1,144 / 1,101.

W podziale na klasy loci (Dev / Hk PCC): peak_849bp_region 0,712 / 0,777 ·
positive_peaks 0,630 / 0,879 · negative 0,471 / 0,410 · Other_DHSs 0,473 /
0,384 · Other_enhancers 0,516 / 0,395.

SE metryki ≈ 0,003. Różnica poniżej ~0,005 to remis, nie przewaga.

## Funkcje

**RC-TTA** (domyślnie włączone) — uśrednia predykcję z obu nici, +0,006 PCC.
Wyłączenie: `rc_tta=False` / `--no-tta`.

**Mutageneza in silico** — efekt każdej z 747 możliwych substytucji:

```python
effects = model.mutagenesis(seq)
effects["Dev"]     # macierz (249, 4), kolejność zasad A, C, G, T
                   # wartość = predykcja(wariant) − predykcja(referencji)
```

**Kalibracja** (`calibrated=True` / `--calibrated`); poprawka afiniczna
dopasowana na walidacji, obniża MSE o 3–5 %; PCC i Spearman bez zmian.
Uwaga: `b` = 0,83 (Dev) / 0,88 (Hk), więc dodatkowo ściska predykcje. Do
szukania sekwencji skrajnie aktywnych używaj surowych.

**Ewaluacja na danych z etykietami:**

```bash
python evaluate.py --fasta Sequences_Test.fa \
                   --activity Sequences_activity_Test.txt \
                   --name test --output results \
                   --deepstarr-tsv Test_counts_DeepSTARR_predictions.tsv
```

Pliki DeepSTARR trzymają obie nici (wiersz `i+n/2` to odwrotny komplement
wiersza `i`) — skrypt wykrywa to i liczy metryki **na loci**, nie na wierszach.

## Uwagi

* Kolejność kanałów one-hot to **A, C, T, G**, nie A, C, G, T. Koduj przez
  `model.encode()`, a odwrotny komplement przez `dna.reverse_complement_tensor`
  (`torch.flip(x, dims=[1,2])` daje tu transwersję, nie komplement).
* Długość musi być dokładnie 249 nt. `--fit-length` przycina centralnie lub
  dopełnia N, ale sekwencje z dużym udziałem N są poza rozkładem treningowym.
* Model liczy w pełnym FP32. TF32 (domyślne w PyTorchu) przestawiało kody
  kwantyzatora w 2,4 % sekwencji i uzależniało wynik od sprzętu do ~1e-2.
* Nie importuj w procesie, w którym jest już inny moduł `models` lub `dna` —
  wrapper to wykrywa i rzuca błąd.

## Zawartość

```
deepstarr.py        wrapper (klasa DeepSTARRModel)
predict.py          CLI: sekwencje → predykcje
evaluate.py         CLI: sekwencje + etykiety → metryki
test_install.py     weryfikacja instalacji
example.py
model/              weights.pt, config.yaml, calibration.json
src/                kod modelu (kopia z repo KAAMMB, nie edytować)
examples/           16 sekwencji + predykcje referencyjne
results/            raport i predykcje dla splitu testowego
```

Model: HypVQNet, 2 277 210 parametrów, checkpoint z epoki 20 (wagi EMA),
ziarno 7898, trening na dendrite (RTX 5090, torch 2.10).
