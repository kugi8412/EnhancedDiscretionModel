# models/__init__.py

from .registry import build_model, register_model

from . import layers
from . import wrappers

from . import deepstarr
from . import regseqnet
from . import reversnet
from . import senet
from . import legnet
from . import convnext
from . import dilated_convnext
from . import deep_epi
from . import emavqvae
from . import dnacvqvae
from . import pixelcnn
from . import legnetplus

# Optional: gLLM adapters require transformers / evo2 packages
try:
    from . import gllm  # noqa: F401  registers DNABert2 and Evo2
except Exception as _gllm_err:  # pragma: no cover
    import warnings
    warnings.warn(f"gLLM adapters not loaded: {_gllm_err}", ImportWarning, stacklevel=2)
