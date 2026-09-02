from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .config import BaselineConfig, ExperimentConfig, ModelConfig


@dataclass
class ModelOutput:
    features: torch.Tensor
    reconstruction: torch.Tensor | None = None
    preactivations: torch.Tensor | None = None


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


class _JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, thresholds, bandwidth):
        ctx.save_for_backward(inputs, thresholds)
        ctx.bandwidth = bandwidth
        return inputs * (inputs > thresholds).to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, thresholds = ctx.saved_tensors
        gate = (inputs > thresholds).to(inputs.dtype)
        rectangle = ((inputs - thresholds).abs() <= ctx.bandwidth / 2).to(inputs.dtype)
        threshold_grad = -thresholds * rectangle * grad_output / ctx.bandwidth
        return grad_output * gate, threshold_grad.sum_to_size(thresholds.shape), None


class _RectangleStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, thresholds, bandwidth):
        ctx.save_for_backward(inputs, thresholds)
        ctx.bandwidth = bandwidth
        return (inputs > thresholds).to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, thresholds = ctx.saved_tensors
        rectangle = ((inputs - thresholds).abs() <= ctx.bandwidth / 2).to(inputs.dtype)
        return None, (-rectangle * grad_output / ctx.bandwidth).sum_to_size(thresholds.shape), None


class SAEBase(nn.Module):
    """Shared untied SAE with a learned decoder bias and constrained decoder columns."""

    def __init__(self, model_config: ModelConfig, baseline_config: BaselineConfig, k: int) -> None:
        super().__init__()
        self.pre_bias = nn.Parameter(torch.zeros(model_config.d_llm))
        self.encoder = nn.Linear(model_config.d_llm, model_config.feature_dim)
        self.decoder = nn.Linear(model_config.feature_dim, model_config.d_llm, bias=False)
        self.k = k
        self.k_aux = min(baseline_config.k_aux, model_config.feature_dim)
        self.auxk_coefficient = baseline_config.auxk_coefficient
        self.dead_feature_window_tokens = baseline_config.dead_feature_window_tokens
        self.register_buffer(
            "last_active_token", torch.zeros(model_config.feature_dim, dtype=torch.long)
        )
        self.register_buffer("tokens_seen", torch.zeros((), dtype=torch.long))
        self.register_buffer("calibrated_threshold", torch.tensor(float("nan")))
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        self.normalize_decoder_()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.weight.T)
            nn.init.zeros_(self.encoder.bias)

    def preactivations(self, residuals: torch.Tensor) -> torch.Tensor:
        return self.encoder(residuals - self.pre_bias)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features) + self.pre_bias

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))

    @torch.no_grad()
    def project_decoder_gradients_(self) -> None:
        if self.decoder.weight.grad is None:
            return
        weights = self.decoder.weight
        gradient = self.decoder.weight.grad
        gradient.sub_(weights * (weights * gradient).sum(dim=0, keepdim=True))

    @torch.no_grad()
    def update_dead_features_(self, features: torch.Tensor) -> None:
        batch = features.shape[0]
        self.tokens_seen.add_(batch)
        active = features.detach().reshape(-1, features.shape[-1]).gt(0).any(dim=0)
        self.last_active_token[active] = self.tokens_seen

    def dead_mask(self) -> torch.Tensor:
        return (self.tokens_seen - self.last_active_token) >= self.dead_feature_window_tokens

    def auxiliary_loss(
        self,
        residuals: torch.Tensor,
        reconstruction: torch.Tensor,
        preactivations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dead = self.dead_mask()
        if not bool(dead.any()) or self.auxk_coefficient == 0:
            return reconstruction.new_zeros((), dtype=torch.float32)
        pre = preactivations if preactivations is not None else self.preactivations(residuals)
        pre = pre.relu().masked_fill(~dead, -torch.inf)
        count = min(self.k_aux, int(dead.sum()))
        values, indices = pre.topk(count, dim=-1)
        aux_features = torch.zeros_like(pre).scatter(-1, indices, values.clamp_min(0))
        residual_target = (residuals - reconstruction.detach()).float()
        aux_reconstruction = self.decoder(aux_features).float()
        numerator = (aux_reconstruction - residual_target).square().sum(dim=-1).mean()
        centered = residual_target - residual_target.mean(dim=0, keepdim=True)
        denominator = centered.square().sum(dim=-1).mean()
        return (numerator / denominator.clamp_min(1e-12)).nan_to_num(0.0)


def batch_topk(values: torch.Tensor, k: int) -> torch.Tensor:
    """Keep exactly batch*k candidates from post-ReLU activations."""
    if values.ndim != 2:
        raise ValueError("BatchTopK expects [batch, features]")
    count = min(values.numel(), values.shape[0] * k)
    flat = values.flatten()
    selected_values, selected_indices = flat.topk(count)
    return torch.zeros_like(flat).scatter(0, selected_indices, selected_values).view_as(values)


class BatchTopKSAE(SAEBase):
    def encode(self, residuals: torch.Tensor, *, pointwise: bool | None = None) -> torch.Tensor:
        positive = self.preactivations(residuals).relu()
        if pointwise is None:
            pointwise = not self.training and torch.isfinite(self.calibrated_threshold)
        if pointwise:
            if not torch.isfinite(self.calibrated_threshold):
                raise RuntimeError("BatchTopK pointwise encoding requires a calibrated threshold")
            return positive * (positive > self.calibrated_threshold).to(positive.dtype)
        return batch_topk(positive, self.k)

    def forward(self, residuals: torch.Tensor, *, pointwise: bool | None = None) -> ModelOutput:
        pre = self.preactivations(residuals)
        positive = pre.relu()
        if pointwise is None:
            pointwise = not self.training and torch.isfinite(self.calibrated_threshold)
        features = (
            positive * (positive > self.calibrated_threshold).to(positive.dtype)
            if pointwise
            else batch_topk(positive, self.k)
        )
        return ModelOutput(features, self.decode(features), pre)


class JumpReLUSAE(SAEBase):
    def __init__(self, model_config: ModelConfig, baseline_config: BaselineConfig, k: int) -> None:
        super().__init__(model_config, baseline_config, k)
        self.log_threshold = nn.Parameter(
            torch.full(
                (model_config.feature_dim,),
                math.log(baseline_config.jump_relu_initial_threshold),
            )
        )
        self.bandwidth = baseline_config.jump_relu_bandwidth

    @property
    def threshold(self) -> torch.Tensor:
        return self.log_threshold.exp()

    def encode(self, residuals: torch.Tensor, *, pointwise: bool | None = None) -> torch.Tensor:
        pre = self.preactivations(residuals)
        return _JumpReLU.apply(pre, self.threshold, self.bandwidth)

    def l0_surrogate(self, preactivations: torch.Tensor) -> torch.Tensor:
        return _RectangleStep.apply(preactivations, self.threshold, self.bandwidth)

    def forward(self, residuals: torch.Tensor, *, pointwise: bool | None = None) -> ModelOutput:
        pre = self.preactivations(residuals)
        features = _JumpReLU.apply(pre, self.threshold, self.bandwidth)
        return ModelOutput(features, self.decode(features), pre)


class MatryoshkaSAE(BatchTopKSAE):
    def __init__(self, model_config: ModelConfig, baseline_config: BaselineConfig, k: int) -> None:
        super().__init__(model_config, baseline_config, k)
        cumulative = []
        total = 0
        for size in baseline_config.matryoshka_group_sizes:
            total += size
            cumulative.append(total)
        self.prefix_widths = tuple(cumulative)
        weight_sum = sum(baseline_config.matryoshka_weights)
        self.prefix_weights = tuple(
            weight / weight_sum for weight in baseline_config.matryoshka_weights
        )


def build_model(config: ExperimentConfig) -> nn.Module:
    if config.model.type == "proposed":
        return ProposedModel(config.model)
    if config.model.type == "batch_topk_sae":
        return BatchTopKSAE(config.model, config.baseline, config.target_l0)
    if config.model.type == "jump_relu_sae":
        return JumpReLUSAE(config.model, config.baseline, config.target_l0)
    if config.model.type == "matryoshka_sae":
        return MatryoshkaSAE(config.model, config.baseline, config.target_l0)
    raise ValueError(f"Unsupported model type: {config.model.type}")
