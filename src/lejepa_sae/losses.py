from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from statistics import NormalDist

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RDMRegularizationOutput:
    loss: torch.Tensor
    random_loss: torch.Tensor
    axis_loss: torch.Tensor
    random_view_losses: torch.Tensor
    axis_view_losses: torch.Tensor


def generalized_gaussian_unit_variance_sigma(lp_norm_parameter: float) -> float:
    """Scale for unit pre-rectification variance under the paper's GN_p convention."""
    if lp_norm_parameter <= 0:
        raise ValueError("lp_norm_parameter must be positive")
    p = lp_norm_parameter
    log_sigma = 0.5 * (math.lgamma(1.0 / p) - math.lgamma(3.0 / p))
    log_sigma = log_sigma - math.log(p) / p
    return math.exp(log_sigma)


@lru_cache(maxsize=128)
def generalized_gaussian_mean_shift_for_active_fraction(
    lp_norm_parameter: float,
    expected_active_fraction: float,
    sigma: float | None = None,
) -> float:
    """Return mu such that ReLU(mu + sigma * GN_p) has the requested L0 fraction."""
    if lp_norm_parameter <= 0:
        raise ValueError("lp_norm_parameter must be positive")
    if not 0.0 < expected_active_fraction < 1.0:
        raise ValueError("expected_active_fraction must be in (0, 1)")
    if sigma is None:
        sigma = generalized_gaussian_unit_variance_sigma(lp_norm_parameter)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if expected_active_fraction == 0.5:
        return 0.0

    p = lp_norm_parameter
    if p == 1.0:
        if expected_active_fraction < 0.5:
            return sigma * math.log(2.0 * expected_active_fraction)
        return -sigma * math.log(2.0 * (1.0 - expected_active_fraction))
    if p == 2.0:
        return sigma * NormalDist().inv_cdf(expected_active_fraction)

    tail_probability = 2.0 * min(
        expected_active_fraction, 1.0 - expected_active_fraction
    )
    target_cdf = min(1.0 - tail_probability, 1.0 - 1e-15)
    shape = torch.tensor(1.0 / p, dtype=torch.float64)
    target = torch.tensor(target_cdf, dtype=torch.float64)
    lower, upper = 0.0, 1.0
    while (
        float(torch.special.gammainc(shape, torch.tensor(upper, dtype=torch.float64)))
        < target_cdf
    ):
        upper *= 2.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        cdf = torch.special.gammainc(
            shape, torch.tensor(midpoint, dtype=torch.float64)
        )
        if bool(cdf < target):
            lower = midpoint
        else:
            upper = midpoint
    magnitude = (p * (lower + upper) / 2.0) ** (1.0 / p)
    sign = -1.0 if expected_active_fraction < 0.5 else 1.0
    return sign * sigma * magnitude


def sample_rectified_generalized_gaussian_like(
    values: torch.Tensor,
    lp_norm_parameter: float,
    mean_shift_value: float = 0.0,
    sigma: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ReLU(GN_p(mu, sigma)) with the Rectified LpJEPA parameterization."""
    if lp_norm_parameter <= 0:
        raise ValueError("lp_norm_parameter must be positive")
    if sigma is None:
        sigma = generalized_gaussian_unit_variance_sigma(lp_norm_parameter)
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    sample_dtype = (
        torch.float32 if values.dtype in {torch.float16, torch.bfloat16} else values.dtype
    )
    shape = values.shape
    device = values.device
    p = lp_norm_parameter

    if p == 1.0:
        uniform = torch.rand(shape, device=device, dtype=sample_dtype, generator=generator)
        epsilon = torch.finfo(sample_dtype).eps
        uniform = uniform.clamp(min=epsilon, max=1.0 - epsilon) - 0.5
        noise = -uniform.sign() * torch.log1p(-2.0 * uniform.abs())
    elif p == 2.0:
        noise = torch.randn(shape, device=device, dtype=sample_dtype, generator=generator)
    else:
        concentration = torch.full(shape, 1.0 / p, device=device, dtype=sample_dtype)
        gamma = torch._standard_gamma(concentration, generator=generator)
        signs = torch.empty(shape, device=device, dtype=sample_dtype).bernoulli_(
            0.5, generator=generator
        )
        noise = (2.0 * signs - 1.0) * (p * gamma).pow(1.0 / p)

    samples = (mean_shift_value + sigma * noise).relu()
    return samples.to(dtype=values.dtype)


def random_unit_projections(
    num_projections: int,
    feature_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw row-wise unit vectors, normalizing in float32 before mixed-precision use."""
    if num_projections < 1:
        raise ValueError("num_projections must be positive")
    if feature_dim < 1:
        raise ValueError("feature_dim must be positive")
    projections = torch.randn(
        num_projections,
        feature_dim,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    projections = F.normalize(projections, dim=1)
    return projections.to(dtype=dtype)


def random_axis_indices(
    num_axes: int,
    feature_dim: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Choose coordinate axes without replacement for one optimizer step."""
    if not 1 <= num_axes <= feature_dim:
        raise ValueError("num_axes must be in [1, feature_dim]")
    return torch.randperm(feature_dim, device=device, generator=generator)[:num_axes]


def sliced_wasserstein_2_with_projections(
    values: torch.Tensor,
    targets: torch.Tensor,
    projection_vectors: torch.Tensor,
) -> torch.Tensor:
    """Empirical sliced W2, optionally vectorized over a leading view dimension."""
    if values.shape != targets.shape or values.ndim not in {2, 3}:
        raise ValueError(
            "values and targets must have the same [batch, features] or "
            "[views, batch, features] shape"
        )
    if projection_vectors.ndim != 2 or projection_vectors.shape[1] != values.shape[-1]:
        raise ValueError("projection_vectors must have shape [projections, features]")

    leading_shape = values.shape[:-1]
    projected_values = (values.reshape(-1, values.shape[-1]) @ projection_vectors.T).reshape(
        *leading_shape, projection_vectors.shape[0]
    )
    projected_values = projected_values.sort(dim=-2).values
    with torch.no_grad():
        projected_targets = (
            targets.reshape(-1, targets.shape[-1]) @ projection_vectors.T
        ).reshape(*leading_shape, projection_vectors.shape[0])
        projected_targets = projected_targets.sort(dim=-2).values
    difference = projected_values.float() - projected_targets.float()
    return difference.square().mean(dim=(-2, -1))


def sliced_wasserstein_2_on_axes(
    values: torch.Tensor,
    targets: torch.Tensor,
    axis_indices: torch.Tensor,
) -> torch.Tensor:
    """Empirical W2 on selected coordinate marginals without materializing one-hot vectors."""
    if values.shape != targets.shape or values.ndim not in {2, 3}:
        raise ValueError(
            "values and targets must have the same [batch, features] or "
            "[views, batch, features] shape"
        )
    if axis_indices.ndim != 1 or axis_indices.numel() < 1:
        raise ValueError("axis_indices must be a non-empty vector")
    if axis_indices.dtype != torch.long:
        raise ValueError("axis_indices must have dtype torch.long")

    selected_values = values.index_select(-1, axis_indices).sort(dim=-2).values
    with torch.no_grad():
        selected_targets = targets.index_select(-1, axis_indices).sort(dim=-2).values
    difference = selected_values.float() - selected_targets.float()
    return difference.square().mean(dim=(-2, -1))


def rectified_lp_rdm_regularization(
    feature_views: list[torch.Tensor] | torch.Tensor,
    num_projections: int,
    num_axis_projections: int,
    axis_weight: float,
    lp_norm_parameter: float,
    mean_shift_value: float,
    generator: torch.Generator | None = None,
    projection_vectors: torch.Tensor | None = None,
    axis_indices: torch.Tensor | None = None,
) -> RDMRegularizationOutput:
    """Multi-view RDMReg with equally weighted global and local view groups."""
    if isinstance(feature_views, list):
        if not feature_views:
            raise ValueError("At least one feature view is required")
        if any(view.shape != feature_views[0].shape for view in feature_views):
            raise ValueError("all feature views must have the same shape")
        feature_tensor = torch.stack(feature_views)
    else:
        feature_tensor = feature_views
    if feature_tensor.ndim != 3 or feature_tensor.shape[0] < 1:
        raise ValueError("feature views must have shape [views, batch, features]")
    reference = feature_tensor[0]

    if projection_vectors is None:
        projection_vectors = random_unit_projections(
            num_projections,
            reference.shape[1],
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
    elif projection_vectors.shape != (num_projections, reference.shape[1]):
        raise ValueError("projection_vectors shape does not match num_projections/features")
    projection_vectors = projection_vectors.to(device=reference.device, dtype=reference.dtype)
    if axis_weight < 0:
        raise ValueError("axis_weight must be non-negative")
    if axis_indices is None:
        axis_indices = random_axis_indices(
            num_axis_projections,
            reference.shape[1],
            device=reference.device,
            generator=generator,
        )
    elif axis_indices.shape != (num_axis_projections,):
        raise ValueError("axis_indices shape does not match num_axis_projections")
    axis_indices = axis_indices.to(device=reference.device, dtype=torch.long)

    sigma = generalized_gaussian_unit_variance_sigma(lp_norm_parameter)
    targets = sample_rectified_generalized_gaussian_like(
        feature_tensor,
        lp_norm_parameter,
        mean_shift_value,
        sigma,
        generator,
    )
    random_view_losses = sliced_wasserstein_2_with_projections(
        feature_tensor, targets, projection_vectors
    )
    axis_view_losses = sliced_wasserstein_2_on_axes(
        feature_tensor, targets, axis_indices
    )
    if random_view_losses.numel() == 1:
        random_loss = random_view_losses[0]
        axis_loss = axis_view_losses[0]
    else:
        random_loss = 0.5 * (
            random_view_losses[0] + random_view_losses[1:].mean()
        )
        axis_loss = 0.5 * (axis_view_losses[0] + axis_view_losses[1:].mean())
    return RDMRegularizationOutput(
        loss=random_loss + axis_weight * axis_loss,
        random_loss=random_loss,
        axis_loss=axis_loss,
        random_view_losses=random_view_losses,
        axis_view_losses=axis_view_losses,
    )


@torch.no_grad()
def l1_sparsity_metric(features: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """Paper metric: mean (||z||_1 / ||z||_2)^2 / D over samples."""
    width = features.shape[1]
    work = features.float()
    l1_norm = torch.linalg.vector_norm(work, ord=1, dim=1)
    l2_norm = torch.linalg.vector_norm(work, ord=2, dim=1)
    return ((l1_norm / (l2_norm + epsilon)).square() / width).mean()
