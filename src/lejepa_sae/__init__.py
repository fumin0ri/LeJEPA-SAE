"""LeJEPA-SAE: sparse JEPA-style features for LLM residual streams."""

from .config import ExperimentConfig, load_config
from .models import (
    BatchTopKSAE,
    JumpReLUSAE,
    MatryoshkaSAE,
    ProposedModel,
    build_model,
)

__all__ = [
    "ExperimentConfig",
    "BatchTopKSAE",
    "JumpReLUSAE",
    "MatryoshkaSAE",
    "ProposedModel",
    "build_model",
    "load_config",
]

__version__ = "0.1.0"
