from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

ModelType = Literal[
    "proposed",
    "sparse_jepa_full_view",
    "jepa_sigreg",
    "standard_sae",
    "window_autoencoder",
    "single_token_jepa",
    "dimension_denoising_sae",
]

TOKEN_VIEW_MODEL_TYPES = {"proposed", "sparse_jepa_full_view", "jepa_sigreg"}
DIMENSION_VIEW_MODEL_TYPES = {"single_token_jepa", "dimension_denoising_sae"}


@dataclass
class DataConfig:
    activation_dir: str = "data/pythia-6.9b/layer-16"
    window_size: int = 10
    train_stride: int = 1
    eval_stride: int = 10
    num_workers: int = 4
    cache_shards_per_worker: int = 1


@dataclass
class ModelConfig:
    type: ModelType = "proposed"
    d_llm: int = 4096
    d_encoder: int = 256
    num_layers: int = 3
    num_heads: int = 4
    mlp_ratio: int = 4
    feature_dim: int = 8192
    dropout: float = 0.0
    attention: Literal["causal", "bidirectional"] = "causal"
    num_local_views: int = 4
    local_tokens: int = 3
    dimension_keep_fraction: float = 0.5
    sparse_bias: float = 0.0


@dataclass
class LossConfig:
    lambda_rdm: float = 1.0
    rdm_projections: int = 32
    target_active_fraction: float = 0.10
    target_sigma: float = 1.0
    invariance_weight: float = 1.0
    reconstruction_weight: float = 1.0
    sae_l1_coefficient: float = 1e-3


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    precision: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    batch_size: int = 64
    gradient_accumulation_steps: int = 8
    max_steps: int = 100_000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2_000
    max_grad_norm: float = 1.0
    log_every: int = 20
    eval_every: int = 1_000
    checkpoint_every: int = 5_000
    eval_batches: int = 50
    output_dir: str = "runs/proposed-k3"
    resume_from: str | None = None


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        if self.data.window_size < 1:
            raise ValueError("data.window_size must be positive")
        if (
            self.model.type in TOKEN_VIEW_MODEL_TYPES
            and not 1 <= self.model.local_tokens <= self.data.window_size
        ):
            raise ValueError("model.local_tokens must be in [1, data.window_size]")
        if self.model.d_encoder % self.model.num_heads:
            raise ValueError("model.d_encoder must be divisible by model.num_heads")
        if self.model.num_local_views < 1:
            raise ValueError("model.num_local_views must be positive")
        if not 0.0 < self.model.dimension_keep_fraction <= 1.0:
            raise ValueError("model.dimension_keep_fraction must be in (0, 1]")
        if (
            self.model.type in DIMENSION_VIEW_MODEL_TYPES
            and self.data.window_size != 1
        ):
            raise ValueError("dimension-view models require data.window_size=1")
        if not 0.0 < self.loss.target_active_fraction < 1.0:
            raise ValueError("loss.target_active_fraction must be in (0, 1)")
        if self.train.gradient_accumulation_steps < 1:
            raise ValueError("train.gradient_accumulation_steps must be positive")
        if self.model.type == "sparse_jepa_full_view":
            self.model.local_tokens = self.data.window_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(cls: type, values: dict[str, Any] | None):
    fields = values or {}
    return cls(**fields)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    unknown = set(raw) - {"data", "model", "loss", "train"}
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    config = ExperimentConfig(
        data=_merge_dataclass(DataConfig, raw.get("data")),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        loss=_merge_dataclass(LossConfig, raw.get("loss")),
        train=_merge_dataclass(TrainConfig, raw.get("train")),
    )
    config.validate()
    return config
