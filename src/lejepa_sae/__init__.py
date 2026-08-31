"""LeJEPA-SAE: sparse JEPA-style features for LLM residual streams."""

from .config import ExperimentConfig, load_config
from .models import SparseJEPA, StandardSAE, WindowAutoencoder, build_model

__all__ = [
    "ExperimentConfig",
    "SparseJEPA",
    "StandardSAE",
    "WindowAutoencoder",
    "build_model",
    "load_config",
]

__version__ = "0.1.0"
