from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import ExperimentConfig, ModelConfig


@dataclass
class ModelOutput:
    features: torch.Tensor
    reconstruction: torch.Tensor | None = None


class SparseLinearFeatureEncoder(nn.Module):
    """Shared pre-bias and sparse linear encoder for one residual activation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.pre_bias = nn.Parameter(torch.zeros(config.d_llm))
        self.encoder = nn.Linear(config.d_llm, config.feature_dim)
        nn.init.zeros_(self.encoder.bias)

    def prepare_input(
        self,
        residuals: torch.Tensor,
        dimension_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residuals.ndim < 2 or residuals.shape[-1] != self.pre_bias.numel():
            raise ValueError("residuals must have shape [..., d_llm]")
        centered = residuals - self.pre_bias
        if dimension_mask is None:
            return centered
        if dimension_mask.shape != residuals.shape:
            raise ValueError("dimension_mask must have the same shape as residuals")
        retained_fraction = dimension_mask.float().mean(dim=-1, keepdim=True)
        if torch.any(retained_fraction == 0):
            raise ValueError("dimension_mask must retain at least one coordinate per item")
        return centered * dimension_mask.to(centered.dtype) / retained_fraction

    def encode_features(
        self,
        residuals: torch.Tensor,
        dimension_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(self.prepare_input(residuals, dimension_mask)).relu()


class ProposedModel(SparseLinearFeatureEncoder):
    """Single-token dimension-mask JEPA proposed model."""

    def forward(
        self,
        residuals: torch.Tensor,
        dimension_mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        return ModelOutput(features=self.encode_features(residuals, dimension_mask))


class StandardSAE(SparseLinearFeatureEncoder):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.decoder = nn.Linear(config.feature_dim, config.d_llm, bias=False)

    def forward(
        self,
        residuals: torch.Tensor,
        dimension_mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        features = self.encode_features(residuals, dimension_mask)
        reconstruction = self.decoder(features) + self.pre_bias
        return ModelOutput(features=features, reconstruction=reconstruction)


class DimensionDenoisingSAE(StandardSAE):
    """SAE baseline that reconstructs a complete token from masked coordinates."""


def build_model(config: ExperimentConfig) -> nn.Module:
    if config.model.type == "proposed":
        return ProposedModel(config.model)
    if config.model.type == "standard_sae":
        return StandardSAE(config.model)
    if config.model.type == "dimension_denoising_sae":
        return DimensionDenoisingSAE(config.model)
    raise ValueError(f"Unsupported model type: {config.model.type}")
