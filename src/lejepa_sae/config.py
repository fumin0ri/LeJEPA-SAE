from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

ModelType = Literal["proposed", "standard_sae", "dimension_denoising_sae"]
MODEL_TYPES = {"proposed", "standard_sae", "dimension_denoising_sae"}
FEATURE_ACTIVATIONS = {"relu", "relu_forward_leaky_backward"}


@dataclass
class DataConfig:
    activation_dir: str = "data/the-pile/pythia-6.9b/layer-16-ctx1024-100m"
    window_size: int = 1
    train_stride: int = 1
    eval_stride: int = 1
    num_workers: int = 4
    cache_shards_per_worker: int = 1


@dataclass
class ModelConfig:
    type: ModelType = "proposed"
    d_llm: int = 4096
    feature_dim: int = 16384
    num_local_views: int = 4
    dimension_keep_fraction: float = 0.5
    feature_activation: str = "relu"
    leaky_backward_slope: float = 0.01


@dataclass
class LossConfig:
    lambda_rdm: float = 125.0
    rdm_projections: int = 8192
    axis_projections: int = 512
    axis_weight: float = 1.0
    target_distribution: str = "rectified_lp_distribution"
    lp_norm_parameter: float = 1.0
    expected_l0_fraction: float | None = 0.009765625
    mean_shift_value: float = 0.0
    mode_of_sigma: str = "sigma_GN"
    projection_vectors_type: str = "random"
    invariance_weight: float = 25.0
    reconstruction_weight: float = 1.0
    sae_l1_coefficient: float = 1e-3


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    precision: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    batch_size: int = 512
    gradient_accumulation_steps: int = 1
    max_steps: int | Literal["one_epoch"] = "one_epoch"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2_000
    max_grad_norm: float = 1.0
    log_every: int = 20
    eval_every: int = 1_000
    checkpoint_every: int = 10_000
    eval_batches: int = 12
    output_dir: str = "runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed"
    resume_from: str | None = None


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        if self.model.type not in MODEL_TYPES:
            raise ValueError(f"model.type must be one of {sorted(MODEL_TYPES)}")
        if self.data.window_size != 1:
            raise ValueError("single-token models require data.window_size=1")
        if self.model.d_llm < 1 or self.model.feature_dim < 1:
            raise ValueError("model dimensions must be positive")
        if self.model.num_local_views < 1:
            raise ValueError("model.num_local_views must be positive")
        if not 0.0 < self.model.dimension_keep_fraction <= 1.0:
            raise ValueError("model.dimension_keep_fraction must be in (0, 1]")
        if self.model.feature_activation not in FEATURE_ACTIVATIONS:
            raise ValueError(
                f"model.feature_activation must be one of {sorted(FEATURE_ACTIVATIONS)}"
            )
        if not 0.0 < self.model.leaky_backward_slope <= 1.0:
            raise ValueError("model.leaky_backward_slope must be in (0, 1]")
        if (
            self.model.feature_activation == "relu_forward_leaky_backward"
            and self.model.type != "proposed"
        ):
            raise ValueError(
                "relu_forward_leaky_backward is an ablation for model.type=proposed"
            )
        if self.loss.rdm_projections < 1:
            raise ValueError("loss.rdm_projections must be positive")
        if not 1 <= self.loss.axis_projections <= self.model.feature_dim:
            raise ValueError("loss.axis_projections must be in [1, model.feature_dim]")
        if self.loss.axis_weight <= 0:
            raise ValueError("loss.axis_weight must be positive")
        if self.model.type == "proposed":
            if self.loss.target_distribution != "rectified_lp_distribution":
                raise ValueError(
                    "proposed requires loss.target_distribution=rectified_lp_distribution"
                )
            if self.loss.lp_norm_parameter <= 0:
                raise ValueError("loss.lp_norm_parameter must be positive")
            if self.loss.expected_l0_fraction is not None and not (
                0.0 < self.loss.expected_l0_fraction < 1.0
            ):
                raise ValueError("loss.expected_l0_fraction must be in (0, 1) or null")
            if self.loss.mode_of_sigma != "sigma_GN":
                raise ValueError("proposed currently supports loss.mode_of_sigma=sigma_GN")
            if self.loss.projection_vectors_type != "random":
                raise ValueError("proposed currently supports projection_vectors_type=random")
        if self.train.gradient_accumulation_steps < 1:
            raise ValueError("train.gradient_accumulation_steps must be positive")
        if self.train.batch_size < 1:
            raise ValueError("train.batch_size must be positive")
        if self.train.max_steps != "one_epoch" and (
            not isinstance(self.train.max_steps, int) or self.train.max_steps < 1
        ):
            raise ValueError("train.max_steps must be a positive integer or one_epoch")

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
