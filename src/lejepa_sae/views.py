from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResidualView:
    residuals: torch.Tensor
    positions: torch.Tensor
    indices: torch.Tensor


def sample_local_views(
    residuals: torch.Tensor,
    num_views: int,
    retained_tokens: int,
    generator: torch.Generator | None = None,
) -> list[ResidualView]:
    """Drop tokens before the encoder while preserving their original positions.

    Every batch item receives an independent subset. Returned tokens are ordered by
    their original position, so causal attention retains its intended meaning.
    """
    if residuals.ndim != 3:
        raise ValueError("residuals must have shape [batch, window, d_llm]")
    batch, window, width = residuals.shape
    if not 1 <= retained_tokens <= window:
        raise ValueError("retained_tokens must be in [1, window]")
    if num_views < 1:
        raise ValueError("num_views must be positive")

    views: list[ResidualView] = []
    for _ in range(num_views):
        scores = torch.rand(
            batch,
            window,
            device=residuals.device,
            generator=generator,
        )
        indices = scores.topk(retained_tokens, dim=1, largest=False).indices
        indices = indices.sort(dim=1).values
        gather_index = indices.unsqueeze(-1).expand(batch, retained_tokens, width)
        dropped = residuals.gather(dim=1, index=gather_index)
        views.append(ResidualView(dropped, indices, indices))
    return views


def full_view(residuals: torch.Tensor) -> ResidualView:
    if residuals.ndim != 3:
        raise ValueError("residuals must have shape [batch, window, d_llm]")
    positions = torch.arange(residuals.shape[1], device=residuals.device)
    positions = positions.expand(residuals.shape[0], -1)
    return ResidualView(residuals, positions, positions)


def sample_dimension_masks(
    values: torch.Tensor,
    num_views: int,
    keep_fraction: float,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Sample exact-k coordinate masks independently for every item and view."""
    if values.ndim != 2:
        raise ValueError("values must have shape [batch, dimensions]")
    if num_views < 1:
        raise ValueError("num_views must be positive")
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    batch, dimensions = values.shape
    retained = max(1, min(dimensions, round(dimensions * keep_fraction)))
    masks = []
    for _ in range(num_views):
        scores = torch.rand(
            batch,
            dimensions,
            device=values.device,
            generator=generator,
        )
        indices = scores.topk(retained, dim=1, largest=False).indices
        mask = torch.zeros(batch, dimensions, dtype=torch.bool, device=values.device)
        mask.scatter_(1, indices, True)
        masks.append(mask)
    return masks
