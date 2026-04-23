#!/usr/bin/env python
"""KAMB — unified Streamlit multi-page hub.

Launch
------
    streamlit run src/py/streamlit/app.py

Pages are implemented as independent modules imported lazily so that
the hub itself starts instantly even if optional dependencies (captum,
biopython, etc.) are not all installed.
"""

import streamlit as st

st.set_page_config(
    page_title="KAMB — DNA Expression Model Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "🏠  Home": "home",
    "📊  Model Evaluation": "results_app",
    "🔍  XAI Explorer": "xai_app",
    "🎛️  CNN Filters": "filters_app",
    "🤝  Ensemble & Uncertainty": "ensemble_app",
}

st.sidebar.title("🧬 KAMB")
st.sidebar.caption("DNA Expression Model Suite")
st.sidebar.markdown("---")
choice = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
page_key = PAGES[choice]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Docs:** [README](https://github.com/your-repo/KAMB)  \n"
    "**Data:** `data/deepSTARR/`  \n"
    "**Models:** `src/py/models/`"
)

# ---------------------------------------------------------------------------
# Page dispatch
# ---------------------------------------------------------------------------

if page_key == "home":
    st.title("KAMB — DNA Regulatory Sequence Modelling")
    st.markdown(
        """
Welcome to the **KAMB** analysis suite for predicting *Drosophila* enhancer activity
from DNA sequence.

---

### Available tools

| Page | Description |
|---|---|
| **Model Evaluation** | Benchmark one or more trained models on a held-out test set.  Upload config YAMLs + weight files and a FASTA + label file to get PCC / SCC / MSE tables, hexbin scatter plots, strand-bias analysis, and rank-order comparison. |
| **XAI Explorer** | Explain model decisions with Saliency, SmoothGrad, Integrated Gradients, DeepLIFT, GradientShap, Feature Ablation, and Grad-CAM.  ISM heatmaps use per-base colour coding; IG can be rendered as a sequence logo. |
| **CNN Filters** | Visualise learned first-layer convolutional filters and their activation patterns across sequences. |
| **Ensemble & Uncertainty** | Combine multiple models into a deep ensemble with MC Dropout uncertainty.  Produces per-sequence 95 % CI, variance histograms, and rank-order Dev/Hk correlation. |

---

### Quick-start

```bash
# 1. Train a model
python src/py/train.py --config config/LegNetPlus.yaml

# 2. Evaluate on the test set
python src/py/evaluate.py \\
    --config  config/LegNetPlus.yaml \\
    --weights results/models/LegNet/LegNetPlus.pth \\
    --fasta   data/deepSTARR/Sequences_Test.fa \\
    --labels  data/deepSTARR/Sequences_activity_Test.txt

# 3. Launch this dashboard
streamlit run src/py/streamlit/app.py
```

---

### Architecture overview

```
data/deepSTARR/          raw FASTA + activity labels (Train / Val / Test)
config/                  per-model YAML configs
src/py/
  train.py               supervised CNN training
  train_vq.py            VQ-VAE / cVQVAE training
  evaluate.py            CLI evaluation (PCC / SCC)
  ensemble.py            deep-ensemble + MC Dropout
  sparse_ae/             Sparse Auto-Encoder (L1 + TopK)
  models/                DeepSTARR, LegNet, SEResNet, ConvNeXt, ReverseNet, …
  streamlit/             this dashboard
results/models/          saved checkpoints
```
        """
    )

else:
    # Lazy-import the selected page module and re-execute its top-level code
    # We use exec() on the module's __file__ so that each page has its own
    # Streamlit widget namespace while sharing the same process.
    import importlib, sys, os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        # Remove any previously cached version so re-entry works
        if page_key in sys.modules:
            del sys.modules[page_key]
        mod = importlib.import_module(page_key)
    except ModuleNotFoundError as exc:
        st.error(f"Could not load page **{page_key}**: {exc}")
        st.info("Make sure all dependencies are installed (`pip install -r requirements.txt`).")
