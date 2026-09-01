"""LeJEPA-SAE: sparse JEPA-style features for LLM residual streams."""

from .config import ExperimentConfig, load_config
from .models import (
    DimensionDenoisingSAE,
    ProposedModel,
    StandardSAE,
    build_model,
)

__all__ = [
    "ExperimentConfig",
    "DimensionDenoisingSAE",
    "ProposedModel",
    "StandardSAE",
    "build_model",
    "load_config",
]

__version__ = "0.1.0"
