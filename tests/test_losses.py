import math

import pytest
import torch

from lejepa_sae.losses import (
    generalized_gaussian_mean_shift_for_active_fraction,
    generalized_gaussian_unit_variance_sigma,
    random_axis_indices,
    random_unit_projections,
    rectified_lp_rdm_regularization,
    sample_rectified_generalized_gaussian_like,
    sliced_wasserstein_2_on_axes,
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


def test_mean_shift_matches_requested_l0_fraction_for_supported_lp_targets():
    expected = 0.009765625
    reference = torch.empty(500_000)
    for p in (0.5, 1.0, 2.0):
        mean_shift = generalized_gaussian_mean_shift_for_active_fraction(p, expected)
        target = sample_rectified_generalized_gaussian_like(
            reference,
            p,
            mean_shift,
            generator=torch.Generator().manual_seed(31),
        )
        observed = float((target > 0).float().mean())
        assert abs(observed - expected) < 0.001


def test_mean_shift_active_fraction_validation():
    for expected in (0.0, 1.0):
        with pytest.raises(ValueError, match="expected_active_fraction"):
            generalized_gaussian_mean_shift_for_active_fraction(1.0, expected)


def test_random_projections_are_unit_norm():
    projections = random_unit_projections(
        32,
        64,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(3),
    )
    torch.testing.assert_close(projections.norm(dim=1), torch.ones(32), atol=1e-6, rtol=1e-6)


def test_random_axis_indices_are_unique_and_reproducible():
    first = random_axis_indices(
        12,
        64,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(21),
    )
    second = random_axis_indices(
        12,
        64,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(21),
    )
    torch.testing.assert_close(first, second)
    assert torch.unique(first).numel() == 12


def test_axis_wasserstein_matches_explicit_one_hot_projections_and_gradient():
    axis_values = torch.randn(5, 12, 16, requires_grad=True)
    projected_values = axis_values.detach().clone().requires_grad_(True)
    targets = torch.randn(5, 12, 16)
    indices = torch.tensor([1, 4, 7, 11])
    one_hot = torch.eye(16).index_select(0, indices)

    axis_losses = sliced_wasserstein_2_on_axes(axis_values, targets, indices)
    projected_losses = sliced_wasserstein_2_with_projections(
        projected_values, targets, one_hot
    )
    torch.testing.assert_close(axis_losses, projected_losses)

    axis_losses.mean().backward()
    projected_losses.mean().backward()
    torch.testing.assert_close(axis_values.grad, projected_values.grad)


def test_axis_wasserstein_pushes_collapsed_selected_coordinates():
    values = torch.zeros(1, 64, 16, requires_grad=True)
    targets = torch.randn(1, 64, 16).relu()
    indices = torch.tensor([0, 3, 7, 12])
    loss = sliced_wasserstein_2_on_axes(values, targets, indices).mean()
    loss.backward()

    assert loss > 0
    assert values.grad is not None
    assert values.grad.index_select(-1, indices).abs().sum() > 0


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
    axis_indices = torch.tensor([1, 5, 9, 13])
    result = rectified_lp_rdm_regularization(
        features,
        12,
        4,
        0.75,
        1.0,
        0.0,
        generator=torch.Generator().manual_seed(6),
        projection_vectors=projections,
        axis_indices=axis_indices,
    )
    result.loss.backward()
    torch.testing.assert_close(
        result.loss, result.random_loss + 0.75 * result.axis_loss
    )
    manual_generator = torch.Generator().manual_seed(6)
    sigma = generalized_gaussian_unit_variance_sigma(1.0)
    targets = []
    for view in features:
        targets.append(
            sample_rectified_generalized_gaussian_like(
                view.detach(), 1.0, 0.0, sigma, manual_generator
            )
        )
    target_tensor = torch.stack(targets)
    feature_tensor = torch.stack([view.detach() for view in features])
    expected_random = sliced_wasserstein_2_with_projections(
        feature_tensor, target_tensor, projections
    )
    expected_axis = sliced_wasserstein_2_on_axes(
        feature_tensor, target_tensor, axis_indices
    )
    torch.testing.assert_close(result.random_view_losses.detach(), expected_random)
    torch.testing.assert_close(result.axis_view_losses.detach(), expected_axis)
    torch.testing.assert_close(
        result.random_loss.detach(),
        0.5 * expected_random[0] + 0.5 * expected_random[1:].mean(),
    )
    torch.testing.assert_close(
        result.axis_loss.detach(),
        0.5 * expected_axis[0] + 0.5 * expected_axis[1:].mean(),
    )
    assert len({round(float(item), 7) for item in result.axis_view_losses.detach()}) > 1
    assert all(view.grad is not None and torch.isfinite(view.grad).all() for view in features)


def test_rdm_gradient_weights_global_and_local_groups_equally():
    features = torch.randn(5, 8, 6, requires_grad=True)
    result = rectified_lp_rdm_regularization(
        features,
        4,
        3,
        1.0,
        1.0,
        0.0,
        generator=torch.Generator().manual_seed(17),
    )

    (random_weights,) = torch.autograd.grad(
        result.random_loss,
        result.random_view_losses,
        retain_graph=True,
    )
    (axis_weights,) = torch.autograd.grad(
        result.axis_loss,
        result.axis_view_losses,
    )
    expected = torch.tensor([0.5, 0.125, 0.125, 0.125, 0.125])
    torch.testing.assert_close(random_weights, expected)
    torch.testing.assert_close(axis_weights, expected)


def test_paper_rdm_pushes_post_relu_collapsed_features():
    features = torch.zeros(64, 16, requires_grad=True)
    result = rectified_lp_rdm_regularization(
        [features],
        16,
        8,
        1.0,
        1.0,
        0.0,
        generator=torch.Generator().manual_seed(9),
    )
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert result.axis_loss > 0
    assert features.grad is not None
    assert float(features.grad.abs().sum()) > 0


def test_paper_rdm_supports_bfloat16_projection_path():
    features = [torch.randn(16, 8, dtype=torch.bfloat16).relu().requires_grad_(True)]
    result = rectified_lp_rdm_regularization(
        features,
        4,
        4,
        1.0,
        1.0,
        0.0,
        generator=torch.Generator().manual_seed(10),
    )
    result.loss.backward()
    assert result.loss.dtype == torch.float32
    assert torch.isfinite(result.loss)
    assert features[0].grad is not None
