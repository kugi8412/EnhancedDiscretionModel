# KAMB — Deep Learning for DNA Enhancer Activity Prediction

A PyTorch framework for predicting *Drosophila* enhancer activity (Developmental and Housekeeping expression) from one-hot encoded DNA sequences. Features 20+ model architectures, EvoAug data augmentation, VQ-VAE generative models, oracle-guided training, crosscoder SAE interpretability, MC Dropout uncertainty, active learning, and single-output ablation.

---

## Project Structure

```
KAMB/
├── config/                 # YAML experiment configs (one per model/run)
├── data/
│   ├── deepSTARR/          # DeepSTARR train/val/test splits (.fa, .txt)
│   └── drosophila_genome/  # dm6 reference genome
├── envs/                   # Conda environment (pytorch.yml)
├── results/
│   └── models/             # Saved model weights (best checkpoints)
├── src/
│   ├── py/                 # All Python source code
│   ├── R/                  # Expression and motif analysis
│   └── sh/                 # Data download scripts
├── experiments.md          # Experiment tracker with results tables
└── README.md
```

### Training Scripts (`src/py/`)

| Script | Purpose | Command |
|--------|---------|---------|
| `train_cnn.py` | CNN regression models (MSE loss) | `python train_cnn.py -c ../../config/<model>.yaml` |
| `train_vq.py` | VQ-VAE generative models | `python train_vq.py -c ../../config/<vqvae>.yaml` |
| `train_nucleotide.py` | NucleotideCNN + MLP predictor + uncertainty + active learning | `python train_nucleotide.py -c ../../config/NucleotideCNN.yaml` |
| `finetune.py` | Fine-tune any model on new/mutated data | `python finetune.py -c ../../config/Finetune.yaml` |
| `crosscoder_sae.py` | Crosscoder shared dictionary learning | `python crosscoder_sae.py -c ../../config/Crosscoder.yaml` |
| `cross_sae.py` | Cross-model SAE comparison + CKA | `python cross_sae.py --sae_config ../../config/CrossSAE.yaml` |
| `evaluate.py` | Evaluation with PCC/SCC plots | `python evaluate.py -c <config> -w <weights> -s Test` |
| `ensemble.py` | Multi-model ensemble + MC Dropout | `python ensemble.py --configs ... --weights ...` |
| `generate.py` | Synthetic enhancer generation via cVQVAE | `python generate.py -c <vqvae_config> -w <weights>` |

### Models (`src/py/models/`)

All models are built via `build_model(config)` using decorator registry `@register_model("Name")`.

| Registered Name | File | Type | Description |
|---|---|---|---|
| `DeepSTARR` | `deepstarr.py` | CNN | Original DeepSTARR 4-layer CNN |
| `DeepSTARR_Siamese` | `deepstarr.py` | CNN | RC-averaging Siamese |
| `DeepSTARR_2D_Fusion` | `deepstarr.py` | CNN | 2D dual-strand convolutions |
| `LegNetOriginal` | `legnet.py` | CNN | Original SeqNN (18-bin soft-classification) |
| `LegNet` | `legnet.py` | CNN | Regression-adapted LegNet |
| `LegNetV2` | `legnet.py` | CNN | EfficientNet-style blocks |
| `LegNetPlus` | `legnetplus.py` | CNN | Multi-scale stem + GLU + SE + DropPath |
| `ConvNeXt_DNA` | `convnext.py` | CNN | ConvNeXt-T for DNA |
| `DilatedConvNeXt` | `dilated_convnext.py` | CNN | ASAP-adapted dilated residual CNN |
| `DeepEPI` | `deep_epi.py` | CNN | DeepEPI with FiLM promoter conditioning |
| `SEResNet` | `senet.py` | CNN | SE-ResNet bottleneck |
| `BassetNetwork` | `regseqnet.py` | CNN | Basset with SuperKernel |
| `CustomNetwork` | `regseqnet.py` | CNN | Configurable Basset variant |
| `ReverseNet_SuperKernel` | `reversnet.py` | CNN | RC-equivariant weight sharing |
| `HydraDNA_cVQVAE` | `dnacvqvae.py` | VQ-VAE | Conditional VQ-VAE with FiLM + GRU |
| `cVQVAE_MultiTask` | `dnacvqvae.py` | VQ-VAE | Multitask cVQVAE with direct heads |
| `cVQVAE_Asymmetric` | `dnacvqvae.py` | VQ-VAE | Asymmetric encoder/decoder |
| `LegNet_VQVAE` | `emavqvae.py` | VQ-VAE | LegNet-backbone VQ-VAE |
| `DNA_PixelCNN` | `pixelcnn.py` | Generative | Autoregressive causal model |
| `DNA_PixelCNN_Conditioned` | `pixelcnn.py` | Generative | PixelCNN conditioned on VQ latent |
| `NucleotideCNN` | `pixelcnn.py` | Uncertainty | Per-position nucleotide predictor |
| `DNABert2` | `gllm.py` | gLLM | DNABERT-2 with linear heads |
| `Evo2` | `gllm.py` | gLLM | Evo-2 SSM with linear heads |

---

## Quick Start

```bash
# 1. Setup
conda env create -f envs/pytorch.yml
conda activate deepstarr
pip install tltorch captum

# 2. Download data
bash src/sh/get_original_DeepSTARR.sh

# 3. Train (run from src/py/)
cd src/py
python train_cnn.py -c ../../config/LegNetArch.yaml
```

---

## Config Format (Standard)

All experiments follow this standard YAML structure:

```yaml
experiment_name: "ModelName_Variant"
seed: 42

model:
  name: "RegisteredName"      # Must match @register_model decorator
  output_head: "both"         # "both" (default), "dev", or "hk" (single-output ablation)
  kwargs:
    seq_len: 249              # Model-specific keyword arguments

data:
  dataset_path: "../../data/deepSTARR"
  batch_size: 128
  augment: false
  num_workers: 4
  target_noise:
    apply: false
    std: 0.05
  evoaug:                     # Only when augment: true
    max_augs_per_seq: 2
    distribution: "poisson"
    poisson_lambda: 0.5
    augmentations: [...]

training:
  lr: 0.0005
  weight_decay: 0.0001
  epochs: 100
  early_stop: 15
  log_dir: "train_logs"       # All checkpoints saved here
  scheduler:
    apply: true
    type: "cosine"
    warmup_fraction: 0.05     # Fraction of epochs for linear warmup
    eta_min: 1.0e-6

comet:                        # Optional (gracefully skipped if not installed)
  api_key: ""
  project_name: "kamb"
  workspace: ""
```

### Key Convention Notes

- All paths are **relative to `src/py/`** (the working directory for all scripts)
- Data path: `../../data/deepSTARR` → `KAMB/data/deepSTARR/`
- Config path: `../../config/X.yaml` → `KAMB/config/X.yaml`
- Model saves: `train_logs/<experiment_name>_seed<N>.pth`
- `log_dir` must be under `training:` (not top-level)
- Use `warmup_fraction` (not `warmup_epochs`) for scheduler warmup

---

## Single-Output Ablation

To train a model predicting only Dev or only Hk (multitask ablation study):

```yaml
model:
  name: "LegNetPlus"
  output_head: "dev"          # Only Dev loss and metrics
```

The model architecture stays the same, but only one head receives gradient. Compare with `output_head: "both"` to quantify multitask transfer.

---

## NucleotideCNN Pipeline

Per-position nucleotide probability predictor with uncertainty estimation:

```bash
# Train nucleotide predictor (frozen backbone)
python train_nucleotide.py -c ../../config/NucleotideCNN.yaml

# Train MLP predictor from latent features → Dev/Hk
python train_nucleotide.py -c ../../config/NucleotideCNN.yaml --train-mlp

# Compute MC Dropout + MSE uncertainty
python train_nucleotide.py -c ../../config/NucleotideCNN.yaml --skip-train --uncertainty

# Active learning on worst sequences
python train_nucleotide.py -c ../../config/NucleotideCNN.yaml --skip-train --active-learning --al-rounds 3
```

Options: `--uncertainty-method mc|mse|combined`, `--mc-samples 20`

---

## Fine-Tuning on Mutated Data

```bash
python finetune.py -c ../../config/Finetune.yaml
python finetune.py -c ../../config/Finetune.yaml --freeze-layers 3
python finetune.py -c ../../config/Finetune.yaml --freeze-prefix stem blocks.0
```

---

## Crosscoder SAE (Cross-Model Interpretability)

Shared dictionary learning across multiple models ([Anthropic Crosscoders](https://transformer-circuits.pub/2024/crosscoders/index.html)):

```bash
python crosscoder_sae.py -c ../../config/Crosscoder.yaml
```

---

## Key Dependencies

- **PyTorch** ≥ 2.0
- **tltorch** — tensor-train for LegNet bilinear layers
- **Captum** — XAI attribution methods
- **Streamlit** — interactive dashboards
- **Comet ML** — experiment tracking (optional)
- **BioPython** — FASTA parsing

---

## References

- de Almeida, B.P. et al. *DeepSTARR predicts enhancer activity from DNA sequence.* Nature Genetics, 2022.
- Penzar, D. et al. *LegNet: a best-in-class model for short regulatory regions.* Bioinformatics, 2023.
- Liu, Z. et al. *A ConvNet for the 2020s.* CVPR, 2022.
- Bricken, T. et al. *Towards Monosemanticity.* Anthropic, 2023.
- Lindsey, J. et al. *Crosscoders.* Transformer Circuits, 2024.
# KAMB — Deep Learning for DNA Enhancer Activity Prediction

A PyTorch framework for predicting Drosophila enhancer activity (Developmental and Housekeeping expression) from one-hot encoded DNA sequences. Includes 15 model architectures, EvoAug data augmentation, VQ-VAE generative models, oracle-guided training, RL-based sequence optimisation, cross-model Sparse Autoencoder comparison, and interactive Streamlit dashboards.

---

## Project Structure

```
KAMB/
├── config/                 # YAML experiment configs (one per model/run)
├── data/
│   ├── deepSTARR/          # DeepSTARR train/val/test splits (.txt, .fa, .fasta)
│   └── drosophila_genome/  # dm6 reference genome and fragment TSVs
├── doc/                    # Documentation
├── envs/                   # Conda environment (pytorch.yml)
├── filtering-seq-pipeline/ # k-mer deduplication pipeline (kmer-db)
├── results/
│   └── models/             # Saved model weights (best_model.pth per architecture)
├── src/
│   ├── py/                 # All Python source code (see below)
│   ├── R/                  # Expression and motif analysis scripts
│   └── sh/                 # Data download scripts
└── README.md
```

### Source Code (`src/py/`)

| File | Description |
|------|-------------|
| `train.py` | Universal training script for CNN, VQ-VAE, and oracle-conditioned models |
| `train_cnn.py` | Simplified CNN-only trainer with extracted helpers |
| `train_vq.py` | Specialised VQ-VAE trainer with unified loss function |
| `evaluate.py` | Evaluation with correlation plots and metric reporting |
| `datasets.py` | `DNADataset` with EvoAug augmentation pipeline |
| `augment.py` | EvoAug augmentation library (9 modules) |
| `utils.py` | Config loading, seeding, one-hot encoding, data helpers |
| `generate.py` | Inverse design via conditional VQ-VAE (synthetic enhancer generation) |
| `change_sequences.py` | Greedy beam-search sequence optimisation with pruning |
| `cross_sae.py` | Cross-model Sparse Autoencoder comparison (penultimate + multi-layer CKA) |
| `rl_latent.py` | Reinforcement learning navigator in VQ-VAE latent space |
| `prepare_filtered_data.py` | Data preparation after k-mer filtering |

### Models (`src/py/models/`)

All models are registered via a decorator-based factory (`registry.py`) and built from YAML config with `build_model(config)`.

| Registered Name | File | Description |
|-----------------|------|-------------|
| `LegNetOriginal` | `legnet.py` | Faithful port of [autosome-ru/LegNet](https://github.com/autosome-ru/LegNet) (SeqNN) with 18-bin soft-classification head |
| `LegNet` | `legnet.py` | Regression-adapted LegNet with dual Dev/Hk heads |
| `LegNetV2` | `legnet.py` | Modernised variant with EfficientNet-style blocks and residual concatenation |
| `LegNetPlus` | `legnetplus.py` | Enhanced LegNet with multi-scale stem, GLU, DropPath, and positional SE |
| `DeepSTARR` | `deepstarr.py` | Original DeepSTARR 4-layer CNN |
| `DeepSTARR_Siamese` | `deepstarr.py` | RC-averaging Siamese variant |
| `DeepSTARR_2D_Fusion` | `deepstarr.py` | 2D convolution variant processing both strands |
| `ConvNeXt_DNA` | `convnext.py` | ConvNeXt adapted for DNA with RC-equivariant fusion |
| `SEResNet` | `senet.py` | SE-ResNet bottleneck architecture |
| `ReverseNet_SuperKernel` | `reversnet.py` | RC-sharing model with learnable SuperKernel masks |
| `BassetNetwork` | `regseqnet.py` | Basset architecture with 2D SuperKernel convolutions |
| `CustomNetwork` | `regseqnet.py` | Configurable Basset variant |
| `HydraDNA_cVQVAE` | `dnacvqvae.py` | Conditional VQ-VAE with FiLM modulation and GRU bottleneck |
| `cVQVAE_MultiTask` | `dnacvqvae.py` | Heavy-encoder / light-decoder cVQVAE with direct Dev+Hk regression heads and sequence rewriting |
| `LegNet_VQVAE` | `emavqvae.py` | LegNet-backbone VQ-VAE with EMA quantiser |
| `DNA_PixelCNN` | `pixelcnn.py` | Autoregressive causal model for unconditional sequence generation |
| `DNA_PixelCNN_Conditioned` | `pixelcnn.py` | PixelCNN conditioned on cVQVAE quantized latent for guided generation |

### Streamlit Dashboards (`src/py/streamlit/`)

| App | Description |
|-----|-------------|
| `results_app.py` | Interactive evaluation dashboard: per-model PCC/SCC, hexbin plots, Dev vs Hk structure analysis, error distribution |
| `xai_app.py` | Explainable AI: Integrated Gradients, DeepLift, GradientShap, Saliency, Grad-CAM, Feature Ablation (via Captum) |
| `filters_app.py` | CNN filter PCA visualisation and JASPAR motif matching |
| `ensemble_app.py` | Ensemble + MC Dropout uncertainty explorer: upload multiple model checkpoints, evaluate sequences, flag high-variance predictions |
| `sae_app.py` | Sparse Autoencoder feature browser: per-feature k-mer enrichment, activation heatmaps, cross-model comparison |

---

## Quick Start

### 1. Environment Setup

```bash
conda env create -f envs/pytorch.yml
conda activate deepstarr
pip install tltorch captum comet-ml
```

### 2. Download Data

```bash
bash src/sh/get_original_DeepSTARR.sh
bash src/sh/get_Drosophila_melanogaster_genome.sh
```

### 3. Train a Model

Every model is driven by a YAML config. Training is launched with a single command:

```bash
cd src/py
python train.py -c ../../config/LegNetPlus.yaml
```

Available configs:

| Config | Model |
|--------|-------|
| `DeepSTARR.yaml` | DeepSTARR baseline |
| `DeepSTARRPlus.yaml` | DeepSTARR with augmentation |
| `LegNetPlus.yaml` | LegNet/LegNetV2 with augmentation |
| `LegNetPlusArch.yaml` | LegNetPlus architecture |
| `LegNetOriginal.yaml` | Original LegNet (SeqNN) |
| `LegNetNewPlus.yaml` | LegNetV2 tuned config |
| `SENet.yaml` / `SENetPlus.yaml` | SE-ResNet |
| `ConvNeXtPlus.yaml` | ConvNeXt DNA |
| `BassetPlus.yaml` | Basset SuperKernel |
| `RegSeqNetPlus.yaml` | Custom Basset |
| `ReverseNet.yaml` | ReverseNet SuperKernel |
| `LegNetVQVAEPlus.yaml` | LegNet VQ-VAE |
| `HydraDNA_cVQVAEPlus.yaml` | Conditional VQ-VAE |
| `cVQVAEOracle.yaml` | Oracle-guided cVQ-VAE training |
| `LegNetOracle.yaml` | LegNet as frozen oracle |
| `cVQVAE_MultiTask.yaml` | Multitask cVQVAE (heavy encoder + direct heads) |
| `PixelCNN_Conditioned.yaml` | Latent-conditioned PixelCNN |

### 4. Evaluate

```bash
python evaluate.py -c ../../config/LegNetPlus.yaml \
                   -w ../../results/models/LegNet/best_model.pth \
                   -s Test
```

### 5. Cross-Model Comparison (Sparse Autoencoders)

**Penultimate-layer comparison** — trains SAEs on each model's last hidden layer and correlates learned dictionary atoms:

```bash
python cross_sae.py --sae_config ../../config/CrossSAE_DeepSTARR_vs_LegNetV2.yaml
```

**Multi-layer comparison** — extracts activations from every layer via forward hooks, computes CKA heatmaps, and optionally trains per-layer SAEs:

```bash
python cross_sae.py \
    --sae_config ../../config/CrossSAE_MultiLayer.yaml \
    --multilayer --sae_per_layer --top_k_pairs 6
```

### 6. Streamlit Dashboards

Launch the unified multi-page hub (recommended):

```bash
streamlit run src/py/streamlit/app.py
```

Or run individual apps:

```bash
streamlit run src/py/streamlit/results_app.py   # multi-model benchmark
streamlit run src/py/streamlit/xai_app.py        # attribution + ISM
streamlit run src/py/streamlit/filters_app.py    # CNN filter viewer
streamlit run src/py/streamlit/ensemble_app.py   # ensemble + MC Dropout
streamlit run src/py/streamlit/sae_app.py        # SAE feature explorer
```

### 7. Ensemble Prediction with Uncertainty

`ensemble.py` builds an ensemble of any registered models and reports MC Dropout confidence intervals:

```bash
cd src/py
python ensemble.py \
    --configs ../../config/LegNetPlus.yaml ../../config/DeepSTARR.yaml \
    --weights ../../results/models/LegNet/best_model.pth \
             ../../results/models/DeepSTARR/DeepSTARR.pth \
    --fasta   ../../data/deepSTARR/Sequences_Test.fa \
    --output  ../../results/ensemble_predictions.csv \
    --mc_passes 30 \
    --var_threshold_dev 0.5 --var_threshold_hk 0.5
```

### 8. Sparse Autoencoder Pipeline

Train a SAE on frozen model activations, then analyse feature-to-motif linkage:

```bash
# Step 1 — train SAE
cd src/py
python -m sparse_ae.train \
    --model_config  ../../config/cVQVAE_MultiTask.yaml \
    --model_weights ../../results/models/cVQVAE/best_model.pth \
    --hook_layer    encoder_gru \
    --fasta         ../../data/deepSTARR/Sequences_Train.fa \
    --activity      ../../data/deepSTARR/Sequences_activity_Train.txt \
    --dict_size     1024 --l1_coeff 1e-3 --epochs 50 \
    --output_dir    ../../results/sae/

# Step 2 — analyse k-mer enrichment
python -m sparse_ae.analyze \
    --sae_checkpoint ../../results/sae/sae.pth \
    --model_config   ../../config/cVQVAE_MultiTask.yaml \
    --model_weights  ../../results/models/cVQVAE/best_model.pth \
    --fasta          ../../data/deepSTARR/Sequences_Test.fa \
    --activity       ../../data/deepSTARR/Sequences_activity_Test.txt \
    --output_dir     ../../results/sae/analysis/
```

### 9. Sequence Generation and Optimisation

```bash
# Generate synthetic enhancers via conditional VQ-VAE
python generate.py -c ../../config/HydraDNA_cVQVAEPlus.yaml \
                   -w ../../results/models/cVQVAE/best_model.pth

# Greedy beam-search optimisation
python change_sequences.py -c ../../config/LegNetPlus.yaml \
                           -w ../../results/models/LegNet/best_model.pth
```

---

## Config Format

All experiments are defined in YAML. A minimal config:

```yaml
experiment_name: "MyExperiment"
seed: 42

model:
  name: "LegNetV2"          # Must match a @register_model name
  kwargs:
    in_ch: 4
    stem_ch: 256
    seq_len: 249

data:
  dataset_path: "../../data/deepSTARR"
  batch_size: 128
  augment: true

training:
  lr: 0.001
  epochs: 100
  early_stop: 20
```

See `config/README.md` for the full schema and all supported fields.

---

## Key Dependencies

- **PyTorch** — core deep learning framework
- **tltorch** — tensor-train decomposition for LegNet bilinear layers
- **Captum** — XAI attribution methods (Integrated Gradients, DeepLift, etc.)
- **Streamlit** — interactive web dashboards
- **Comet ML** — experiment tracking (optional)
- **BioPython** — FASTA parsing for genome pipeline
- **scipy / scikit-learn** — metrics, PCA, statistical tests

---

## End-to-End Experiment Design

This section describes the complete experimental pipeline from raw data to interpretable results.

### Pipeline overview

```
Raw Data
│
├─ data/deepSTARR/          (DeepSTARR train/val/test FASTA + activity labels)
└─ data/drosophila_genome/  (dm6.fa for k-mer deduplication)
        │
        ▼
  [1] Preprocessing & filtering
        │  src/filtering-pipeline/   (kmer-db — remove near-duplicates across splits)
        │  src/py/prepare_filtered_data.py
        ▼
  [2] Model training
        │  python train.py -c config/<model>.yaml
        │  Logged to: results/models/<arch>/  (best checkpoint + metrics.json)
        ▼
  [3] Standard evaluation
        │  python evaluate.py  →  PCC / SCC per split
        │  Streamlit: results_app.py  (multi-model benchmark + hexbin plots)
        ▼
  [4] Ensemble + uncertainty
        │  python ensemble.py  →  per-sequence 95 % CI (MC Dropout)
        │  Streamlit: ensemble_app.py
        ▼
  [5] Model comparison (CKA / SAE)
        │  python cross_sae.py  →  CKA matrices + SAE atom correlation
        │  Streamlit: sae_app.py
        ▼
  [6] Explainability
        │  Streamlit: xai_app.py  (IG, DeepLIFT, GradCAM, ISM heatmap, seq logo)
        ▼
  [7] Sequence design
        │  python generate.py        (cVQVAE-based synthetic enhancers)
        │  python change_sequences.py (greedy beam-search optimisation)
        ▼
  [8] Motif analysis
        │  src/py/motifs-pipeline/   (TomTom + JASPAR matching)
        │  src/R/motif_analysis.R
```

### Step-by-step commands for a complete run

```bash
# ── 0. Setup ──────────────────────────────────────────────────────────────────
conda env create -f envs/pytorch.yml && conda activate deepstarr
bash src/sh/get_original_DeepSTARR.sh
bash src/sh/get_Drosophila_melanogaster_genome.sh

# ── 1. Filtering (removes near-duplicate sequences across splits) ─────────────
cd src/filtering-pipeline
bash run.sh                          # builds kmer-db indices and runs queries
cd ../..
python src/py/prepare_filtered_data.py

# ── 2. Train all baseline models (run in parallel or sequentially) ────────────
cd src/py
for CFG in DeepSTARR LegNetPlus LegNetPlusArch SENetPlus ConvNeXtPlus BassetPlus ReverseNet; do
    python train.py -c ../../config/${CFG}.yaml
done

# ── 3. Train generative models ────────────────────────────────────────────────
python train_vq.py -c ../../config/LegNetVQVAEPlus.yaml
python train_vq.py -c ../../config/HydraDNA_cVQVAEPlus.yaml
python train_vq.py -c ../../config/cVQVAEOracle.yaml

# ── 4. Evaluate all checkpoints on the test set ───────────────────────────────
for CFG in DeepSTARR LegNetPlus LegNetPlusArch SENetPlus ConvNeXtPlus BassetPlus ReverseNet; do
    python evaluate.py -c ../../config/${CFG}.yaml -s Test
done

# ── 5. Build ensemble + MC Dropout uncertainty ────────────────────────────────
python ensemble.py \
    --configs ../../config/LegNetPlus.yaml ../../config/DeepSTARR.yaml \
    --weights ../../results/models/LegNet/best_model.pth \
             ../../results/models/DeepSTARR/DeepSTARR.pth \
    --fasta   ../../data/deepSTARR/Sequences_Test.fa \
    --output  ../../results/ensemble_test.csv \
    --mc_passes 30

# ── 6. Cross-model comparison (penultimate SAE + CKA) ─────────────────────────
python cross_sae.py --sae_config ../../config/CrossSAE_DeepSTARR_vs_LegNetV2.yaml
python cross_sae.py --sae_config ../../config/CrossSAE_MultiLayer.yaml \
    --multilayer --sae_per_layer

# ── 7. Train SAE for feature analysis ─────────────────────────────────────────
python -m sparse_ae.train \
    --model_config  ../../config/cVQVAE_MultiTask.yaml \
    --model_weights ../../results/models/cVQVAE/best_model.pth \
    --hook_layer    encoder_gru \
    --fasta         ../../data/deepSTARR/Sequences_Train.fa \
    --activity      ../../data/deepSTARR/Sequences_activity_Train.txt \
    --dict_size 1024 --l1_coeff 1e-3 --epochs 50 \
    --output_dir ../../results/sae/

# ── 8. Launch the unified dashboard ───────────────────────────────────────────
streamlit run src/py/streamlit/app.py
```

---

## Architecture Benchmark

Expected Pearson correlation coefficients (PCC) on the DeepSTARR test set.
Values are approximate and depend on seed, augmentation, and hyperparameter tuning.

| Architecture | Dev PCC | Hk PCC | Avg PCC | Parameters | Notes |
|---|---|---|---|---|---|
| DeepSTARR | 0.68 | 0.74 | 0.71 | ~1.5 M | Baseline CNN |
| DeepSTARR + aug | 0.70 | 0.76 | 0.73 | ~1.5 M | EvoAug augmentation |
| LegNet (SeqNN) | 0.70 | 0.75 | 0.73 | ~1.3 M | Original architecture |
| LegNetV2 | 0.72 | 0.77 | 0.75 | ~2.0 M | EfficientNet-style blocks |
| LegNetPlus | 0.73 | 0.78 | 0.76 | ~3.5 M | Multi-scale stem + GLU + SE |
| SEResNet | 0.71 | 0.76 | 0.74 | ~2.8 M | SE bottleneck blocks |
| ConvNeXt_DNA | 0.72 | 0.77 | 0.75 | ~3.2 M | ConvNeXt adapted for DNA |
| BassetNetwork | 0.69 | 0.74 | 0.72 | ~1.8 M | Basset + SuperKernel 2D |
| ReverseNet | 0.71 | 0.76 | 0.74 | ~2.1 M | RC-equivariant weight-sharing |
| cVQVAE_MultiTask | 0.66 | 0.72 | 0.69 | ~5.0 M | VQ-VAE + regression heads |
| Ensemble (2–4 models) | **0.75** | **0.80** | **0.78** | — | MC Dropout ensemble |

> **Strand correction**: Models that are not RC-equivariant (all except ReverseNet)
> benefit from strand-correcting the input via `load_fasta_with_strand_correction()`.
> The minus-strand sequences in DeepSTARR are identified by `_-_` in the FASTA header
> and are automatically reverse-complemented.

---

## gLLM Integration (Optional)

The `models/gllm.py` module provides wrappers for genomic large language models:

| Class | Registry key | Backbone | Dependency |
|---|---|---|---|
| `DNABert2Wrapper` | `"DNABert2"` | DNABERT-2 (HuggingFace) | `transformers >= 4.35` |
| `Evo2Wrapper` | `"Evo2"` | Evo-2 (Arc Institute) | `evo2` package |

Both expose:
- `forward(x)` → `(pred_dev, pred_hk)` after a two-head linear classifier
- `get_features(x)` → CLS embedding (for SAE feature extraction)
- `freeze_backbone()` — freeze backbone weights, train only heads
- LoRA support via `lora_rank` in the config

### Setup

```bash
# DNABert2
pip install transformers sentencepiece

# Evo2 (requires NVIDIA GPU, ~40 GB VRAM for 7B)
pip install git+https://github.com/ArcInstitute/evo2.git
```

### Config example (`config/DNABert2.yaml`)

```yaml
model:
  name: "DNABert2"
  kwargs:
    lora_rank: 16
    freeze_backbone: true

training:
  lr: 1e-4
  epochs: 20
  early_stop: 5
```

---



- de Almeida, B.P. et al. *DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers.* Nature Genetics, 2022.
- Penzar, D., Nogina, D. et al. *LegNet: a best-in-class deep learning model for short DNA regulatory regions.* Bioinformatics, 2023.
- Liu, Z. et al. *A ConvNet for the 2020s.* CVPR, 2022.
- Bricken, T. et al. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Anthropic, 2023.
- Kornblith, S. et al. *Similarity of Neural Network Representations Revisited.* ICML, 2019.
