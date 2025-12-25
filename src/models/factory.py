from __future__ import annotations

from typing import Type

from src.models.baseline_gpt import BaselineGPT
from src.models.falcon_gpt import FalconGPT


_MODEL_REGISTRY: dict[str, Type] = {
    "baseline": BaselineGPT,
    "falcon": FalconGPT,
}

def build_model(cfg):
    model_type = getattr(cfg.model, "type", "baseline")
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model.type={model_type!r}. "
            f"Valid: {sorted(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[model_type](cfg)
