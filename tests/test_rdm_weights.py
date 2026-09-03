import copy

import pytest
import torch
import yaml

from lejepa_sae.config import ExperimentConfig, ModelConfig, load_config
from lejepa_sae.losses import rectified_lp_rdm_regularization
from lejepa_sae.models import build_model
from lejepa_sae.train import apply_overrides, compute_loss


def direct_config(random_weight=1.0, axis_weight=4.0):
    config = ExperimentConfig(
        model=ModelConfig(type="rdm_sae", d_llm=4, feature_dim=8, num_local_views=0)
    )
    config.loss.invariance_weight = 0
    config.loss.rdm_projections = 4
    config.loss.axis_projections = 4
    config.loss.rdm_target_scale = 1.5
    config.loss.rdm_random_wasserstein_power = 2
    config.loss.rdm_axis_wasserstein_power = 1
    config.loss.rdm_random_weight = random_weight
    config.loss.rdm_axis_weight = axis_weight
    config.loss.rdm_gradient_diagnostics = True
    config.train.device = "cpu"
    config.train.precision = "float32"
    config.validate()
    return config


@pytest.mark.parametrize("random_weight, axis_weight", [(0, 0), (0, 4), (0.25, 0), (0.25, 4)])
def test_direct_coefficients_match_known_loss_and_gradient(monkeypatch, random_weight, axis_weight):
    values = torch.tensor([[[4.0], [0.0], [1.0]]], requires_grad=True)
    targets = torch.tensor([[[0.0], [2.0], [0.0]]])
    monkeypatch.setattr(
        "lejepa_sae.losses.sample_rectified_generalized_gaussian_like", lambda *_args: targets,
    )
    result = rectified_lp_rdm_regularization(
        values, 1, 1, axis_weight, 1, 0, random_weight=random_weight,
        projection_vectors=torch.ones(1, 1), axis_indices=torch.tensor([0]),
        random_wasserstein_power=2, axis_wasserstein_power=1,
    )
    assert float(result.loss.detach()) == pytest.approx(random_weight * 5 / 3 + axis_weight)
    result.loss.backward()
    expected = torch.tensor([[[4 / 3], [0.0], [2 / 3]]]) * random_weight
    expected += torch.tensor([[[1 / 3], [0.0], [1 / 3]]]) * axis_weight
    torch.testing.assert_close(values.grad, expected)


def test_random_coefficient_does_not_scale_axis_and_legacy_multiplier_is_ignored():
    config = direct_config()
    config.loss.lambda_rdm = 0  # A direct axis-only loss must still run with this legacy value.
    config.loss.axis_weight = 99
    config.validate()
    model = build_model(config)
    inputs = torch.randn(6, 1, 4)
    records = []
    for random_weight in (0, 0.25, 1):
        config.loss.rdm_random_weight = random_weight
        assert config.loss.rdm_enabled
        torch.manual_seed(17)
        loss, metrics = compute_loss(model, inputs, config)
        assert float(metrics["rdm_random_contribution"]) == pytest.approx(
            random_weight * float(metrics["random_distribution"])
        )
        assert float(metrics["rdm_axis_contribution"]) == pytest.approx(
            4 * float(metrics["axis_distribution"])
        )
        torch.testing.assert_close(
            loss.detach(), metrics["reconstruction_contribution"]
            + metrics["rdm_random_contribution"] + metrics["rdm_axis_contribution"],
        )
        if random_weight == 0:
            assert metrics["rdm_random_preactivation_grad_rms"] == 0
            assert metrics["rdm_axis_preactivation_grad_rms"] > 0
        records.append(metrics)
    for record in records[1:]:
        torch.testing.assert_close(
            record["rdm_axis_contribution"], records[0]["rdm_axis_contribution"],
        )
        torch.testing.assert_close(
            record["rdm_axis_preactivation_grad_rms"],
            records[0]["rdm_axis_preactivation_grad_rms"],
        )


def test_old_yaml_maps_to_direct_coefficients_and_matches_explicit_config(tmp_path):
    legacy = direct_config()
    raw = legacy.to_dict()
    raw["loss"].pop("rdm_random_weight")
    raw["loss"].pop("rdm_axis_weight")
    raw["loss"]["lambda_rdm"] = 0.5
    raw["loss"]["axis_weight"] = 4
    path = tmp_path / "old-config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    legacy = load_config(path)
    assert legacy.loss.effective_rdm_random_weight == 0.5
    assert legacy.loss.effective_rdm_axis_weight == 2
    exported = legacy.to_dict()["loss"]
    assert exported["rdm_random_weight"] == 0.5 and exported["rdm_axis_weight"] == 2
    assert "lambda_rdm" not in exported and "axis_weight" not in exported
    explicit = copy.deepcopy(legacy)
    apply_overrides(explicit, ["loss.rdm_random_weight=0.5", "loss.rdm_axis_weight=2"])
    model = build_model(legacy)
    inputs = torch.randn(6, 1, 4)
    torch.manual_seed(52)
    old_loss, _ = compute_loss(model, inputs, legacy)
    old_grads = torch.autograd.grad(old_loss, tuple(model.parameters()))
    torch.manual_seed(52)
    new_loss, metrics = compute_loss(model, inputs, explicit)
    new_grads = torch.autograd.grad(new_loss, tuple(model.parameters()))
    torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
    for old, new in zip(old_grads, new_grads, strict=True):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    expected = metrics["reconstruction_contribution"] + 0.5 * (
        metrics["random_distribution"] + 4 * metrics["axis_distribution"]
    )
    torch.testing.assert_close(old_loss.detach(), expected)
    apply_overrides(legacy, ["loss.rdm_random_weight=0"])
    assert legacy.loss.effective_rdm_random_weight == 0
    assert legacy.loss.effective_rdm_axis_weight == 2
    path.write_text(yaml.safe_dump(legacy.to_dict()), encoding="utf-8")
    assert load_config(path).loss.effective_rdm_axis_weight == 2


@pytest.mark.parametrize("term", ["random", "axis"])
@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_direct_weight_rejected_by_config_and_loss(term, invalid):
    config = direct_config()
    setattr(config.loss, f"rdm_{term}_weight", invalid)
    with pytest.raises(ValueError, match=f"rdm_{term}_weight"):
        config.validate()
    weights = {"random_weight": 1, "axis_weight": 4, f"{term}_weight": invalid}
    with pytest.raises(ValueError, match=f"{term}_weight"):
        rectified_lp_rdm_regularization(
            torch.zeros(1, 4, 2), 2, 2, weights["axis_weight"], 1, 0,
            random_weight=weights["random_weight"],
        )


def test_direct_coefficients_are_specific_to_rdm_sae():
    config = ExperimentConfig()
    config.loss.rdm_random_weight = 1
    config.loss.rdm_axis_weight = 4
    with pytest.raises(ValueError, match="only supported for rdm_sae"):
        config.validate()
