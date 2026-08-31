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
