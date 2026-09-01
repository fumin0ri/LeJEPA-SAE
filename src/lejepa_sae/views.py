from __future__ import annotations

import torch


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
