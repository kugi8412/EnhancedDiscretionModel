from .registry import build_model, register_model
from . import hyp_vq_net  # noqa: F401

__all__ = ["build_model", "register_model"]
