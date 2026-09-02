from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import ExperimentConfig, ModelConfig


@dataclass
class ModelOutput:
    features: torch.Tensor
    reconstruction: torch.Tensor | None = None


class _ReLUForwardLeakyBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, negative_slope: float) -> torch.Tensor:
        ctx.save_for_backward(inputs > 0)
        ctx.negative_slope = negative_slope
        return inputs.relu()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (positive_region,) = ctx.saved_tensors
        return (
            torch.where(
                positive_region,
                grad_output,
                grad_output * ctx.negative_slope,
            ),
            None,
        )


def relu_forward_leaky_backward(
    inputs: torch.Tensor, negative_slope: float
) -> torch.Tensor:
    """Exact ReLU values with a leaky surrogate derivative in the non-positive region."""
    return _ReLUForwardLeakyBackward.apply(inputs, negative_slope)


class SparseLinearFeatureEncoder(nn.Module):
    """Shared pre-bias and sparse linear encoder for one residual activation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.pre_bias = nn.Parameter(torch.zeros(config.d_llm))
        self.encoder = nn.Linear(config.d_llm, config.feature_dim)
        self.feature_activation = config.feature_activation
        self.leaky_backward_slope = config.leaky_backward_slope
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
        preactivations = self.encoder(self.prepare_input(residuals, dimension_mask))
        if self.feature_activation == "relu":
            return preactivations.relu()
        if self.feature_activation == "relu_forward_leaky_backward":
            return relu_forward_leaky_backward(
                preactivations, self.leaky_backward_slope
            )
        raise RuntimeError(f"Unsupported feature activation: {self.feature_activation}")


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
