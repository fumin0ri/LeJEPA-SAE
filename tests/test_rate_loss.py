import copy

import pytest
import torch
import yaml

from lejepa_sae.config import DataConfig, ExperimentConfig, ModelConfig, load_config
from lejepa_sae.losses import straight_through_gate, target_rate_regularization
from lejepa_sae.models import build_model
from lejepa_sae.train import compute_loss


def rate_config():
    config = ExperimentConfig(
        data=DataConfig(window_size=1, num_workers=0),
        model=ModelConfig(d_llm=8, feature_dim=16, num_local_views=4, mask_scaling="sqrt"),
    )
    config.loss.rdm_projections = config.loss.axis_projections = 4
    config.loss.rate_weight = 0.3
    config.loss.expected_l0_fraction = 0.05
    return config


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gate_exact_forward_and_sigmoid_backward(dtype):
    pre = torch.tensor([-2, -0.1, 0, 0.1, 2], dtype=dtype, requires_grad=True)
    temperature = torch.tensor(0.2)
    gate = straight_through_gate(pre, temperature)
    assert gate.dtype == torch.float32
    assert torch.equal(gate, (pre > 0).float())
    gate.sum().backward()
    soft = torch.sigmoid(pre.detach().float() / temperature)
    torch.testing.assert_close(pre.grad, (soft * (1 - soft) / temperature).to(dtype))


def test_rate_aggregation_and_analytic_surrogate_gradient():
    pre = torch.tensor([
        [[-2., -1., 0., 1.]],
        [[-1., 0., 1., 2.]],
        [[0., 1., 2., 3.]],
        [[-3., -2., -1., 0.]],
        [[1., 2., 3., 4.]],
    ], requires_grad=True)
    rho, tau = 0.5, 0.7
    result = target_rate_regularization(pre, rho, tau)
    expected_rates = torch.tensor([0.25, 0.5, 0.75, 0, 1])
    expected_losses = (expected_rates - rho).square() / (2 * rho * (1 - rho))
    torch.testing.assert_close(result.rates, expected_rates)
    torch.testing.assert_close(result.view_losses, expected_losses)
    torch.testing.assert_close(
        result.loss, 0.5 * expected_losses[0] + 0.5 * expected_losses[1:].mean()
    )
    assert not result.scale.requires_grad
    result.loss.backward()
    scale = pre[0].detach().std(unbiased=False)
    weights = torch.tensor([0.5, 0.125, 0.125, 0.125, 0.125])
    soft = torch.sigmoid(pre.detach() / (tau * scale))
    expected_grad = (
        (weights * (expected_rates - rho) / (rho * (1 - rho) * 4))[:, None, None]
        * soft * (1 - soft) / (tau * scale)
    )
    torch.testing.assert_close(pre.grad, expected_grad)
    # Repeating locals must not increase their group's weight.
    duplicated = torch.cat([pre.detach()[:1], pre.detach()[1:].repeat(2, 1, 1)])
    torch.testing.assert_close(target_rate_regularization(duplicated, rho, tau).loss, result.loss)


@pytest.mark.parametrize("on", [False, True])
def test_boundary_rates_have_finite_loss_and_correct_gradient(on):
    values = [0.1, 0.2, 0.3, 0.4] if on else [-0.1, -0.2, -0.3, 0.0]
    pre = torch.tensor([[values]] * 5, requires_grad=True)
    result = target_rate_regularization(pre, 0.05, temperature=1.0)
    assert torch.all(result.rates == int(on))
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert torch.all(torch.isfinite(pre.grad))
    assert torch.all(pre.grad > 0) if on else torch.all(pre.grad < 0)


def test_rate_at_target_and_zero_variance_scale_floor():
    pre = torch.tensor([[[-1., 1.]]] * 5, requires_grad=True)
    result = target_rate_regularization(pre, 0.5)
    result.loss.backward()
    assert result.loss == 0
    assert torch.count_nonzero(pre.grad) == 0
    constant = torch.zeros(5, 1, 1, requires_grad=True)
    result = target_rate_regularization(constant, 0.05)
    result.loss.backward()
    assert result.scale == pytest.approx(1e-6)
    assert torch.isfinite(result.loss) and torch.isfinite(constant.grad).all()
    assert torch.all(constant.grad < 0)


@pytest.mark.parametrize("activation", ["relu", "relu_forward_leaky_backward"])
@pytest.mark.parametrize("autocast", [False, True])
def test_rate_forward_backward_diagnostics_do_not_change_gradients(activation, autocast):
    config = rate_config()
    config.model.feature_activation = activation
    model = build_model(config)
    other = copy.deepcopy(model)
    residuals = torch.randn(6, 1, 8)
    config.loss.rate_gradient_diagnostics = True
    torch.manual_seed(77)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        loss, metrics = compute_loss(model, residuals, config)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert metrics["rate_to_base_grad_ratio"] >= 0
    torch.testing.assert_close(loss, metrics["base_loss"] + metrics["rate_contribution"])
    torch.testing.assert_close(
        metrics["rate_global_active_fraction"], metrics["global_active_fraction"]
    )
    torch.testing.assert_close(
        metrics["rate_local_active_fraction"], metrics["local_active_fraction"]
    )
    config.loss.rate_gradient_diagnostics = False
    torch.manual_seed(77)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        other_loss, _ = compute_loss(other, residuals, config)
    other_loss.backward()
    torch.testing.assert_close(loss, other_loss)
    for p, q in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad)
    config.loss.rate_gradient_diagnostics = True
    with torch.no_grad():
        _, validation = compute_loss(model, residuals, config)
    assert "rate_loss" in validation
    assert "rate_to_base_grad_ratio" not in validation
    _, ordinary = compute_loss(model, residuals, config, include_diagnostics=False)
    assert "rate_loss" in ordinary
    assert "rate_to_base_grad_ratio" not in ordinary
    assert "feature_std" not in ordinary


def test_disabled_rate_preserves_legacy_path_and_checkpoint(tmp_path, monkeypatch):
    config = rate_config()
    config.loss.rate_weight = 0
    raw = config.to_dict()
    for name in list(raw["loss"]):
        if name.startswith("rate_"):
            del raw["loss"][name]
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    legacy = load_config(path)
    assert legacy.loss.rate_weight == 0
    model = build_model(legacy)
    new_model = build_model(config)
    new_model.load_state_dict(model.state_dict(), strict=True)

    def fail(*_args, **_kwargs):
        raise AssertionError("Disabled rate must not compute sigmoid gates or std")

    monkeypatch.setattr("lejepa_sae.train.target_rate_regularization", fail)
    residuals = torch.randn(6, 1, 8)
    torch.manual_seed(88)
    loss, metrics = compute_loss(model, residuals, legacy)
    torch.manual_seed(88)
    other_loss, _ = compute_loss(new_model, residuals, config)
    assert "rate_loss" not in metrics
    torch.testing.assert_close(loss, other_loss, rtol=0, atol=0)
    torch.testing.assert_close(
        loss, config.loss.invariance_weight * metrics["invariance"]
        + config.loss.lambda_rdm * metrics["distribution"], rtol=0, atol=0,
    )
    loss.backward()
    other_loss.backward()
    for p, q in zip(model.parameters(), new_model.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize(("field", "value"), [
    ("rate_weight", -1), ("rate_weight", float("nan")), ("rate_weight", float("inf")),
    ("rate_temperature", 0), ("rate_temperature", float("inf")),
    ("rate_scale_floor", 0), ("rate_scale_floor", float("nan")),
    ("rate_gradient_diagnostics", "true"), ("expected_l0_fraction", None),
])
def test_rate_validation(field, value):
    config = rate_config()
    setattr(config.loss, field, value)
    with pytest.raises(ValueError):
        config.validate()


def test_rate_rejects_sae_baseline():
    config = rate_config()
    config.model.type = "batch_topk_sae"
    with pytest.raises(ValueError, match="only supported"):
        config.validate()


def test_rate_uses_one_encoder_forward_and_no_additional_rng():
    config = rate_config()
    model = build_model(config)
    calls = []
    handle = model.encoder.register_forward_hook(lambda *_args: calls.append(1))
    compute_loss(model, torch.randn(6, 1, 8), config)
    handle.remove()
    assert len(calls) == 1
    pre = torch.randn(5, 6, 16, requires_grad=True)
    state = torch.get_rng_state()
    target_rate_regularization(pre, 0.05).loss.backward()
    assert torch.equal(state, torch.get_rng_state())
