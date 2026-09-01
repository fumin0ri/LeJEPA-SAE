import math

import torch

from lejepa_sae.losses import (
    generalized_gaussian_unit_variance_sigma,
    random_unit_projections,
    rectified_lp_rdm_regularization,
    sample_rectified_generalized_gaussian_like,
    sliced_wasserstein_2_with_projections,
)


def test_generalized_gaussian_unit_variance_scales():
    assert math.isclose(generalized_gaussian_unit_variance_sigma(1.0), 1 / math.sqrt(2))
    assert math.isclose(generalized_gaussian_unit_variance_sigma(2.0), 1.0)


def test_rectified_lp_fast_paths_are_half_active_and_reproducible():
    reference = torch.empty(200_000)
    for p in (1.0, 2.0):
        first = sample_rectified_generalized_gaussian_like(
            reference,
            p,
            generator=torch.Generator().manual_seed(11),
        )
        second = sample_rectified_generalized_gaussian_like(
            reference,
            p,
            generator=torch.Generator().manual_seed(11),
        )
        torch.testing.assert_close(first, second)
        assert abs(float((first > 0).float().mean()) - 0.5) < 0.005
        assert torch.isfinite(first).all()


def test_generalized_rectified_lp_sampler_supports_nonstandard_p():
    reference = torch.empty(4096)
    target = sample_rectified_generalized_gaussian_like(
        reference,
        0.5,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.isfinite(target).all()
    assert (target >= 0).all()


def test_random_projections_are_unit_norm():
    projections = random_unit_projections(
        32,
        64,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(3),
    )
    torch.testing.assert_close(projections.norm(dim=1), torch.ones(32), atol=1e-6, rtol=1e-6)


def test_explicit_sliced_wasserstein_is_permutation_invariant():
    values = torch.randn(32, 16)
    targets = torch.randn(32, 16)
    projections = random_unit_projections(
        8,
        16,
        device=values.device,
        dtype=values.dtype,
        generator=torch.Generator().manual_seed(4),
    )
    baseline = sliced_wasserstein_2_with_projections(values, targets, projections)
    permuted = sliced_wasserstein_2_with_projections(
        values[torch.randperm(32)], targets[torch.randperm(32)], projections
    )
    torch.testing.assert_close(baseline, permuted)


def test_vectorized_sliced_wasserstein_matches_view_loop_and_gradient():
    vectorized_values = torch.randn(5, 12, 16, requires_grad=True)
    loop_values = vectorized_values.detach().clone().requires_grad_(True)
    targets = torch.randn(5, 12, 16)
    projections = random_unit_projections(
        8,
        16,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(14),
    )

    vectorized_losses = sliced_wasserstein_2_with_projections(
        vectorized_values, targets, projections
    )
    loop_losses = torch.stack(
        [
            sliced_wasserstein_2_with_projections(view, target, projections)
            for view, target in zip(loop_values, targets, strict=True)
        ]
    )
    torch.testing.assert_close(vectorized_losses, loop_losses)

    vectorized_losses.mean().backward()
    loop_losses.mean().backward()
    torch.testing.assert_close(vectorized_values.grad, loop_values.grad)


def test_paper_rdm_uses_shared_projections_and_independent_targets():
    features = [torch.randn(32, 16).relu().requires_grad_(True) for _ in range(5)]
    projections = random_unit_projections(
        12,
        16,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(5),
    )
    loss, view_losses = rectified_lp_rdm_regularization(
        features,
        12,
        1.0,
        0.0,
        generator=torch.Generator().manual_seed(6),
        projection_vectors=projections,
    )
    loss.backward()
    torch.testing.assert_close(loss, torch.stack(view_losses).mean())
    manual_generator = torch.Generator().manual_seed(6)
    sigma = generalized_gaussian_unit_variance_sigma(1.0)
    expected_view_losses = []
    for view in features:
        target = sample_rectified_generalized_gaussian_like(
            view.detach(), 1.0, 0.0, sigma, manual_generator
        )
        expected_view_losses.append(
            sliced_wasserstein_2_with_projections(view.detach(), target, projections)
        )
    for observed, expected in zip(view_losses, expected_view_losses, strict=True):
        torch.testing.assert_close(observed.detach(), expected)
    assert len(view_losses) == 5
    assert len({round(float(item.detach()), 7) for item in view_losses}) > 1
    assert all(view.grad is not None and torch.isfinite(view.grad).all() for view in features)


def test_paper_rdm_pushes_post_relu_collapsed_features():
    features = torch.zeros(64, 16, requires_grad=True)
    loss, _ = rectified_lp_rdm_regularization(
        [features],
        16,
        1.0,
        0.0,
        generator=torch.Generator().manual_seed(9),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert features.grad is not None
    assert float(features.grad.abs().sum()) > 0


def test_paper_rdm_supports_bfloat16_projection_path():
    features = [torch.randn(16, 8, dtype=torch.bfloat16).relu().requires_grad_(True)]
    loss, _ = rectified_lp_rdm_regularization(
        features,
        4,
        1.0,
        0.0,
        generator=torch.Generator().manual_seed(10),
    )
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert features[0].grad is not None
