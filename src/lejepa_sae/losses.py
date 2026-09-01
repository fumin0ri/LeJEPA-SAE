from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def invariance_loss(
    global_features: torch.Tensor, local_features: list[torch.Tensor]
) -> torch.Tensor:
    if not local_features:
        raise ValueError("At least one local feature tensor is required")
    return torch.stack([F.mse_loss(local, global_features) for local in local_features]).mean()


def rectified_gaussian_parameters(
    active_fraction: float,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mu and sigma such that P(ReLU(N(mu, sigma²)) > 0) = active_fraction."""
    if not 0.0 < active_fraction < 1.0:
        raise ValueError("active_fraction must be in (0, 1)")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    probability = torch.tensor(active_fraction, device=device, dtype=dtype)
    mu_over_sigma = math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
    scale = torch.tensor(sigma, device=device, dtype=dtype)
    return mu_over_sigma * scale, scale


def sample_rectified_gaussian_like(
    values: torch.Tensor,
    active_fraction: float,
    sigma: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    mu, scale = rectified_gaussian_parameters(
        active_fraction,
        sigma,
        device=values.device,
        dtype=values.dtype,
    )
    noise = torch.randn(values.shape, device=values.device, dtype=values.dtype, generator=generator)
    return (noise * scale + mu).relu()


def generalized_gaussian_unit_variance_sigma(lp_norm_parameter: float) -> float:
    """Scale for unit pre-rectification variance under the paper's GN_p convention."""
    if lp_norm_parameter <= 0:
        raise ValueError("lp_norm_parameter must be positive")
    p = lp_norm_parameter
    log_sigma = 0.5 * (math.lgamma(1.0 / p) - math.lgamma(3.0 / p))
    log_sigma = log_sigma - math.log(p) / p
    return math.exp(log_sigma)


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


def sliced_wasserstein_2(
    values: torch.Tensor,
    targets: torch.Tensor,
    num_projections: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sliced W2 distance using random unit directions and empirical quantiles."""
    if values.shape != targets.shape or values.ndim != 2:
        raise ValueError("values and targets must have the same [batch, features] shape")
    if num_projections < 1:
        raise ValueError("num_projections must be positive")

    directions = torch.randn(
        values.shape[1],
        num_projections,
        device=values.device,
        dtype=values.dtype,
        generator=generator,
    )
    directions = F.normalize(directions, dim=0)
    projected_values = (values @ directions).sort(dim=0).values
    projected_targets = (targets @ directions).sort(dim=0).values
    return F.mse_loss(projected_values, projected_targets)


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


def rectified_lp_rdm_regularization(
    feature_views: list[torch.Tensor] | torch.Tensor,
    num_projections: int,
    lp_norm_parameter: float,
    mean_shift_value: float,
    generator: torch.Generator | None = None,
    projection_vectors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Paper-aligned multi-view RDMReg with shared directions and independent RGG targets."""
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

    sigma = generalized_gaussian_unit_variance_sigma(lp_norm_parameter)
    targets = sample_rectified_generalized_gaussian_like(
        feature_tensor,
        lp_norm_parameter,
        mean_shift_value,
        sigma,
        generator,
    )
    view_loss_tensor = sliced_wasserstein_2_with_projections(
        feature_tensor, targets, projection_vectors
    )
    view_losses = list(view_loss_tensor.unbind())
    return view_loss_tensor.mean(), view_losses


@torch.no_grad()
def l1_sparsity_metric(features: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """Paper metric: mean (||z||_1 / ||z||_2)^2 / D over samples."""
    width = features.shape[1]
    work = features.float()
    l1_norm = torch.linalg.vector_norm(work, ord=1, dim=1)
    l2_norm = torch.linalg.vector_norm(work, ord=2, dim=1)
    return ((l1_norm / (l2_norm + epsilon)).square() / width).mean()


def rdm_regularization(
    features: torch.Tensor,
    num_projections: int,
    active_fraction: float,
    sigma: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    work = features.float()
    target = sample_rectified_gaussian_like(work, active_fraction, sigma, generator)
    return sliced_wasserstein_2(work, target, num_projections, generator)


def gaussian_distribution_regularization(
    features: torch.Tensor,
    num_projections: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    work = features.float()
    target = torch.randn(work.shape, device=work.device, dtype=work.dtype, generator=generator)
    return sliced_wasserstein_2(work, target, num_projections, generator)
