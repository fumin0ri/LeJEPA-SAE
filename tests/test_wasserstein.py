import pytest
import torch
import yaml

from lejepa_sae.config import ExperimentConfig, load_config
from lejepa_sae.losses import (
    rectified_lp_rdm_regularization,
    sliced_wasserstein_2_on_axes,
    sliced_wasserstein_2_with_projections,
    sliced_wasserstein_on_axes,
    sliced_wasserstein_with_projections,
)
from lejepa_sae.train import apply_overrides


@pytest.mark.parametrize("power, expected", [(1, 1.0), (2, 5 / 3)])
@pytest.mark.parametrize("path", ["random", "axis"])
def test_known_1d_transport_cost_and_identity(power, expected, path):
    values = torch.tensor([[4.0], [0.0], [1.0]], requires_grad=True)
    targets = torch.tensor([[0.0], [2.0], [0.0]])
    function, selection = (
        (sliced_wasserstein_with_projections, torch.ones(1, 1))
        if path == "random" else (sliced_wasserstein_on_axes, torch.tensor([0]))
    )
    loss = function(values, targets, selection, wasserstein_power=power)
    assert float(loss.detach()) == pytest.approx(expected)
    loss.backward()
    expected_grad = [[1 / 3], [0.0], [1 / 3]] if power == 1 else [[4 / 3], [0.0], [2 / 3]]
    torch.testing.assert_close(values.grad, torch.tensor(expected_grad))
    identical = function(values, values.detach().flip(0), selection, wasserstein_power=power)
    assert identical == 0
    (gradient,) = torch.autograd.grad(identical, values)
    assert torch.isfinite(gradient).all() and not gradient.count_nonzero()


@pytest.mark.parametrize("power", [1, 2])
@pytest.mark.parametrize("views", [False, True])
def test_axis_equals_one_hot_projection_values_and_gradients(power, views):
    shape = (3, 12, 8) if views else (12, 8)
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(shape, generator=generator, requires_grad=True)
    targets = torch.randn(shape, generator=generator)
    indices = torch.tensor([0, 3, 7])
    axis = sliced_wasserstein_on_axes(values, targets, indices, wasserstein_power=power)
    projected = sliced_wasserstein_with_projections(
        values, targets, torch.eye(8)[indices], wasserstein_power=power,
    )
    torch.testing.assert_close(axis, projected)
    (axis_grad,) = torch.autograd.grad(axis.sum(), values, retain_graph=True)
    (projected_grad,) = torch.autograd.grad(projected.sum(), values)
    torch.testing.assert_close(axis_grad, projected_grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_default_w2_matches_historical_formula_exactly(dtype):
    generator = torch.Generator().manual_seed(18)
    values = torch.randn(3, 12, 8, generator=generator, dtype=dtype, requires_grad=True)
    targets = torch.randn(3, 12, 8, generator=generator, dtype=dtype)
    projections = torch.randn(4, 8, generator=generator, dtype=dtype)
    axes = torch.tensor([1, 3, 6])
    for modern, legacy, selection in (
        (sliced_wasserstein_with_projections, sliced_wasserstein_2_with_projections, projections),
        (sliced_wasserstein_on_axes, sliced_wasserstein_2_on_axes, axes),
    ):
        if selection is projections:
            left = (values.reshape(-1, 8) @ projections.T).reshape(3, 12, 4)
            right = (targets.reshape(-1, 8) @ projections.T).reshape(3, 12, 4)
        else:
            left, right = values[..., axes], targets[..., axes]
        expected = (left.sort(dim=-2).values.float() - right.sort(dim=-2).values.float())
        expected = expected.square().mean(dim=(-2, -1))
        (expected_grad,) = torch.autograd.grad(expected.sum(), values, retain_graph=True)
        for actual in (
            modern(values, targets, selection),
            modern(values, targets, selection, wasserstein_power=2),
            legacy(values, targets, selection),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            (gradient,) = torch.autograd.grad(actual.sum(), values, retain_graph=True)
            torch.testing.assert_close(gradient, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("power", [1, 2])
@pytest.mark.parametrize("views", [1, 3])
def test_rdm_switch_controls_both_terms_and_preserves_view_weights(monkeypatch, power, views):
    values = torch.arange(views * 12, dtype=torch.float32).reshape(views, 4, 3) / 7
    values.requires_grad_()
    targets = torch.tensor([[[0.0, 0, 2], [0, 1, 0], [3, 0, 0], [0, 0, 0]]]).expand_as(values)
    monkeypatch.setattr(
        "lejepa_sae.losses.sample_rectified_generalized_gaussian_like",
        lambda *_args: targets,
    )
    projections = torch.tensor([[1.0, 0, 0], [0, 0.6, 0.8]])
    axes = torch.tensor([1, 2])
    actual = rectified_lp_rdm_regularization(
        values, 2, 2, 4, 1, -1.6, projection_vectors=projections, axis_indices=axes,
        target_scale=1.5, wasserstein_power=power,
    )
    scaled = 1.5 * targets
    expected = []
    for left, right in (
        (values @ projections.T, scaled @ projections.T),
        (values[..., axes], scaled[..., axes]),
    ):
        differences = left.sort(dim=-2).values - right.sort(dim=-2).values
        per_view = differences.abs().pow(power).mean(dim=(-2, -1))
        expected.append(per_view[0] if views == 1 else 0.5 * (per_view[0] + per_view[1:].mean()))
    torch.testing.assert_close(actual.random_loss, expected[0])
    torch.testing.assert_close(actual.axis_loss, expected[1])
    torch.testing.assert_close(actual.loss, expected[0] + 4 * expected[1])
    actual.loss.backward()
    assert torch.isfinite(values.grad).all()


def test_metric_switch_does_not_change_sampling_rng_and_bfloat16_backward():
    features = torch.randn(1, 16, 8, dtype=torch.bfloat16).relu().requires_grad_()
    states = []
    for power in (1, 2):
        generator = torch.Generator().manual_seed(42)
        result = rectified_lp_rdm_regularization(
            features, 4, 4, 4, 1, -1.6, generator=generator, wasserstein_power=power,
        )
        states.append(generator.get_state())
        assert result.loss.dtype == torch.float32 and torch.isfinite(result.loss)
        (grad,) = torch.autograd.grad(result.loss, features)
        assert torch.isfinite(grad).all()
    assert torch.equal(*states)


@pytest.mark.parametrize("invalid", [0, 3, -1, 1.5, 2.0, True, None, "1"])
def test_invalid_transport_power_is_rejected(invalid):
    config = ExperimentConfig()
    config.loss.rdm_wasserstein_power = invalid
    with pytest.raises(ValueError, match="rdm_wasserstein_power"):
        config.validate()
    values = torch.zeros(1, 4, 2)
    with pytest.raises(ValueError, match="wasserstein_power"):
        sliced_wasserstein_with_projections(values, values, torch.eye(2), wasserstein_power=invalid)
    with pytest.raises(ValueError, match="wasserstein_power"):
        sliced_wasserstein_on_axes(values, values, torch.tensor([0]), wasserstein_power=invalid)
    with pytest.raises(ValueError, match="wasserstein_power"):
        rectified_lp_rdm_regularization(values, 2, 2, 1, 1, 0, wasserstein_power=invalid)


@pytest.mark.parametrize("power", [1, 2])
def test_config_override_roundtrip_keeps_target_shape_separate(tmp_path, power):
    config = load_config("configs/pythia-6.9b-layer16-rdm-sae.yaml")
    apply_overrides(config, [f"loss.rdm_wasserstein_power={power}", "loss.lp_norm_parameter=2.0"])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    reloaded = load_config(path)
    assert reloaded.loss.rdm_wasserstein_power == power
    assert reloaded.loss.lp_norm_parameter == 2
