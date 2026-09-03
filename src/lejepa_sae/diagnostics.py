from __future__ import annotations

import torch

ACTIVE_FRACTION_THRESHOLDS = (
    ("0", 0.0),
    ("1e-4", 1e-4),
    ("1e-3", 1e-3),
    ("1e-2", 1e-2),
    ("5e-2", 5e-2),
    ("1e-1", 1e-1),
)


@torch.no_grad()
def thresholded_active_counts(features: torch.Tensor) -> dict[str, torch.Tensor]:
    """Count strict z > threshold without modifying features or consuming RNG.

    Compare in float32 so mixed precision does not round the diagnostic thresholds.
    Process thresholds sequentially rather than allocating [threshold, batch, feature].
    """
    values = features.detach().float()
    return {
        f"active_fraction_gt_{label}": values.gt(threshold).sum()
        for label, threshold in ACTIVE_FRACTION_THRESHOLDS
    }
