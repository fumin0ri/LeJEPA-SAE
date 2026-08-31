import torch

from lejepa_sae.losses import (
    rdm_regularization,
    sample_rectified_gaussian_like,
    sliced_wasserstein_2,
)


def test_rectified_gaussian_matches_requested_active_fraction():
    reference = torch.empty(300_000)
    target = sample_rectified_gaussian_like(
        reference,
        active_fraction=0.1,
        generator=torch.Generator().manual_seed(4),
    )
    assert abs(float((target > 0).float().mean()) - 0.1) < 0.005
    assert (target >= 0).all()


def test_sliced_wasserstein_is_zero_for_identical_samples():
    values = torch.randn(32, 16)
    loss = sliced_wasserstein_2(
        values,
        values,
        num_projections=8,
        generator=torch.Generator().manual_seed(3),
    )
    assert loss == 0


def test_rdm_has_finite_gradient():
    features = torch.randn(32, 16).relu().requires_grad_(True)
    loss = rdm_regularization(features, 8, 0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(features.grad).all()
