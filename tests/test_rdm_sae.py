import copy
import json
import os
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from lejepa_sae.config import ExperimentConfig, ModelConfig, load_config
from lejepa_sae.losses import (
    generalized_gaussian_mean_shift_for_active_fraction,
    rectified_lp_rdm_regularization,
)
from lejepa_sae.models import RDMSAE, build_model
from lejepa_sae.probing import ProbeSAEAdapter
from lejepa_sae.reporting import load_training_history, write_training_curves_svg
from lejepa_sae.train import apply_overrides, compute_loss, evaluate_loss


def rdm_config():
    config = ExperimentConfig(
        model=ModelConfig(type="rdm_sae", d_llm=8, feature_dim=16, num_local_views=0)
    )
    config.loss.invariance_weight = 0
    config.loss.expected_l0_fraction = 0.05
    config.loss.rdm_projections = 4
    config.loss.axis_projections = 4
    config.loss.axis_weight = 2
    config.loss.lambda_rdm = 0.3
    config.loss.reconstruction_weight = 1.5
    config.loss.rdm_target_scale = 2
    config.loss.rdm_gradient_diagnostics = True
    config.train.device = "cpu"
    config.train.precision = "float32"
    config.validate()
    return config


def forbidden(*_args, **_kwargs):
    pytest.fail("RDM SAE must not mask, apply TopK/AuxK/rate loss, or calibrate thresholds")


@pytest.mark.parametrize("activation", ["relu", "relu_forward_leaky_backward"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize(
    "power, random_power, axis_power", [(1, None, None), (2, None, None), (2, 2, 1)],
)
def test_rdm_sae_matches_direct_objective_and_parameter_gradients(
    monkeypatch, activation, mixed, power, random_power, axis_power
):
    config = rdm_config()
    config.loss.rdm_wasserstein_power = power
    config.loss.rdm_random_wasserstein_power = random_power
    config.loss.rdm_axis_wasserstein_power = axis_power
    if axis_power == 1:
        config.loss.rdm_target_scale = 1.5
        config.loss.axis_weight = 4
        config.loss.lambda_rdm = config.loss.reconstruction_weight = 1
    config.model.feature_activation = activation
    config.model.leaky_backward_slope = 0.1
    model = build_model(config)
    assert isinstance(model, RDMSAE)
    assert not hasattr(model, "calibrated_threshold")
    torch.testing.assert_close(model.decoder.weight.norm(dim=0), torch.ones(16))
    torch.testing.assert_close(model.encoder.weight, model.decoder.weight.T)
    assert model.encoder.weight.data_ptr() != model.decoder.weight.data_ptr()
    with torch.no_grad():
        model.pre_bias.copy_(torch.linspace(-1, 1, 8))
    reference = copy.deepcopy(model)
    residuals = torch.randn(6, 1, 8)
    axes = torch.arange(4)
    shapes = []
    model.register_forward_pre_hook(lambda _model, inputs: shapes.append(inputs[0].shape))
    monkeypatch.setattr("lejepa_sae.train.sample_dimension_masks", forbidden)
    monkeypatch.setattr("lejepa_sae.train.target_rate_regularization", forbidden)
    monkeypatch.setattr("lejepa_sae.models.batch_topk", forbidden)
    monkeypatch.setattr(model, "auxiliary_loss", forbidden)

    def autocast():
        return torch.autocast("cpu", dtype=torch.bfloat16) if mixed else nullcontext()

    torch.manual_seed(37)
    with autocast():
        actual, metrics = compute_loss(model, residuals, config, axis_indices=axes)
    torch.manual_seed(37)
    with autocast():
        output = reference(residuals[:, 0])
        reconstruction = config.loss.reconstruction_weight * F.mse_loss(
            output.reconstruction.float(), residuals[:, 0].float()
        )
        rdm = rectified_lp_rdm_regularization(
            output.features.unsqueeze(0), 4, 4, config.loss.axis_weight, 1,
            generalized_gaussian_mean_shift_for_active_fraction(1, 0.05),
            axis_indices=axes, target_scale=config.loss.rdm_target_scale, wasserstein_power=power,
            random_wasserstein_power=random_power, axis_wasserstein_power=axis_power,
        )
        distribution = config.loss.lambda_rdm * rdm.loss
        expected = reconstruction + distribution
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(metrics["reconstruction_contribution"], reconstruction.detach())
    torch.testing.assert_close(metrics["rdm_contribution"], distribution.detach())
    torch.testing.assert_close(metrics["distribution"], rdm.loss.detach())
    assert actual.dtype == torch.float32
    assert metrics["rdm_random_wasserstein_power"] == (random_power or power)
    assert metrics["rdm_axis_wasserstein_power"] == (axis_power or power)
    assert shapes == [(6, 8)]
    assert not any("local" in name for name in metrics)
    assert {"invariance", "rate_loss", "auxk", "l0_penalty", "threshold"}.isdisjoint(metrics)
    for key, term in [
        ("reconstruction", reconstruction),
        ("rdm", distribution),
        ("rdm_random", config.loss.lambda_rdm * rdm.random_loss),
        ("rdm_axis", config.loss.lambda_rdm * config.loss.axis_weight * rdm.axis_loss),
    ]:
        grad = torch.autograd.grad(term, output.preactivations, retain_graph=True)[0]
        rms = grad.float().square().mean().sqrt()
        torch.testing.assert_close(metrics[f"{key}_preactivation_grad_rms"], rms)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.isfinite(value) and not value.requires_grad for value in metrics.values())
    actual.backward()
    expected.backward()
    for parameter, target in zip(model.parameters(), reference.parameters(), strict=True):
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, target.grad)


def test_rdm_sae_pointwise_encoding_and_no_threshold_calibration(monkeypatch):
    config = rdm_config()
    model = build_model(config)
    inputs = torch.randn(4, 3, 8)
    adapter = ProbeSAEAdapter(model, config)
    expected = model.encode(inputs)
    torch.testing.assert_close(adapter.encode(inputs), expected)
    torch.testing.assert_close(adapter.encode(inputs[:1]), expected[:1])
    torch.testing.assert_close(adapter.encode(inputs.flip(0)), expected.flip(0))
    model.eval()
    torch.testing.assert_close(model(inputs).features, expected)
    monkeypatch.setattr("lejepa_sae.train.calibrate_batch_topk_threshold", forbidden)
    metrics = evaluate_loss(model, [{"residuals": inputs[:, :1]}], config)
    assert "reconstruction" in metrics and "distribution" in metrics
    assert "active_fraction" in metrics and "feature_std" in metrics
    assert "rdm_to_reconstruction_grad_ratio" not in metrics
    assert "threshold" not in metrics


@pytest.mark.parametrize("power", [1, 2])
def test_rdm_sae_diagnostics_are_optional_and_do_not_change_backward(power):
    config = rdm_config()
    config.loss.rdm_wasserstein_power = power
    model = build_model(config)
    reference = copy.deepcopy(model)
    inputs = torch.randn(6, 1, 8)
    torch.manual_seed(52)
    loss, metrics = compute_loss(model, inputs, config, include_diagnostics=False)
    torch.manual_seed(52)
    other, full_metrics = compute_loss(reference, inputs, config)
    assert "distribution" in metrics and "reconstruction_contribution" in metrics
    for key in (
        "l0", "active_fraction", "feature_std", "rdm_to_reconstruction_grad_ratio",
        "active_fraction_gt_1e-3", "rdm_random_preactivation_grad_rms",
        "rdm_axis_preactivation_grad_rms",
    ):
        assert key not in metrics and key in full_metrics
    loss.backward()
    other.backward()
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_zero_rdm_weight_is_pure_reconstruction_without_auxk_or_rng(monkeypatch):
    config = rdm_config()
    config.loss.lambda_rdm = 0
    config.loss.expected_l0_fraction = None  # No baseline target_k is needed.
    config.baseline.auxk_coefficient = 1000
    config.validate()
    model = build_model(config)
    monkeypatch.setattr("lejepa_sae.train.compute_rdm", forbidden)
    monkeypatch.setattr(model, "auxiliary_loss", forbidden)
    inputs = torch.randn(6, 1, 8)
    rng = torch.get_rng_state().clone()
    loss, metrics = compute_loss(model, inputs, config)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    expected = config.loss.reconstruction_weight * F.mse_loss(
        model(inputs[:, 0]).reconstruction, inputs[:, 0]
    )
    torch.testing.assert_close(loss, expected)
    assert metrics["rdm_contribution"] == metrics["rdm_preactivation_grad_rms"] == 0
    assert metrics["rdm_random_preactivation_grad_rms"] == 0
    assert metrics["rdm_axis_preactivation_grad_rms"] == 0
    assert "distribution" not in metrics
    loss.backward()
    assert model.decoder.weight.grad is not None


def test_target_scale_multiplies_whole_target_and_preserves_support(monkeypatch):
    targets = []

    def capture(values, target, _projections, *, wasserstein_power):
        targets.append(target.detach().clone())
        return (values.float() - target.float()).square().mean(dim=(-2, -1))

    monkeypatch.setattr("lejepa_sae.losses.sliced_wasserstein_with_projections", capture)
    features = torch.randn(1, 64, 16).relu()
    for scale in [1, 3]:
        rectified_lp_rdm_regularization(
            features, 4, 4, 1, 1, -1.6,
            generator=torch.Generator().manual_seed(11), target_scale=scale,
        )
    torch.testing.assert_close(targets[1], 3 * targets[0])
    assert torch.equal(targets[1] > 0, targets[0] > 0)


@pytest.mark.parametrize("slope", [0.0, 0.1])
def test_negative_features_have_only_the_configured_surrogate_gradient(slope):
    config = rdm_config()
    config.model.feature_activation = "relu_forward_leaky_backward" if slope else "relu"
    config.model.leaky_backward_slope = slope or 0.1
    model = build_model(config)
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.fill_(-1)
    output = model(torch.ones(2, 8))
    assert not output.features.count_nonzero()
    output.features.sum().backward()
    torch.testing.assert_close(model.encoder.bias.grad, torch.full((16,), 2 * slope))


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "num_local_views", 4),
        ("loss", "invariance_weight", 1),
        ("loss", "rate_weight", 1),
        ("loss", "reconstruction_weight", 0),
        ("loss", "reconstruction_weight", float("nan")),
        ("loss", "lambda_rdm", -1),
        ("loss", "lambda_rdm", float("inf")),
        ("loss", "rdm_target_scale", 0),
        ("loss", "rdm_target_scale", float("nan")),
        ("loss", "rdm_gradient_diagnostics", "true"),
        ("loss", "expected_l0_fraction", 1),
        ("loss", "lp_norm_parameter", float("inf")),
        ("loss", "target_distribution", "gaussian"),
    ],
)
def test_rdm_config_rejects_invalid_options(section, field, value):
    config = rdm_config()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        config.validate()


def test_rdm_preset_and_legacy_defaults():
    config = load_config("configs/pythia-6.9b-layer16-rdm-sae.yaml")
    assert config.model.type == "rdm_sae"
    assert config.model.feature_dim == 16384
    assert config.model.feature_activation == "relu_forward_leaky_backward"
    assert config.model.leaky_backward_slope == 0.1
    assert config.loss.lambda_rdm == config.loss.reconstruction_weight == 1
    assert config.loss.rdm_wasserstein_power == 2
    assert config.loss.expected_l0_fraction == 0.05
    assert config.train.batch_size == 512 and config.train.gradient_accumulation_steps == 1
    assert config.train.max_steps == 10000 and config.train.eval_batches == 12
    assert config.train.resume_from is None
    legacy = load_config("configs/pythia-6.9b-layer16.yaml")
    assert legacy.loss.rdm_target_scale == 1
    assert legacy.loss.rdm_wasserstein_power == 2
    assert not legacy.loss.rdm_gradient_diagnostics
    assert legacy.model.type == "proposed"


def test_rdm_diagnostic_curves_hide_absent_local_metrics(tmp_path):
    config = rdm_config()
    _, metrics = compute_loss(build_model(config), torch.randn(6, 1, 8), config)
    record = {"kind": "train", "step": 20, **{k: float(v) for k, v in metrics.items()}}
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(record), encoding="utf-8")
    history = load_training_history(path)
    assert history[0]["active_fraction_gt_1e-3"] == record["active_fraction_gt_1e-3"]
    assert history[0]["rdm_axis_preactivation_grad_rms"] == (
        record["rdm_axis_preactivation_grad_rms"]
    )
    assert history[0]["rdm_wasserstein_power"] == 2
    assert history[0]["rdm_random_wasserstein_power"] == 2
    assert history[0]["rdm_axis_wasserstein_power"] == 2
    assert history[0]["rdm_to_reconstruction_grad_ratio"] == (
        record["rdm_to_reconstruction_grad_ratio"]
    )
    chart = tmp_path / "curves.svg"
    write_training_curves_svg(chart, history)
    svg = chart.read_text(encoding="utf-8")
    assert "Weighted reconstruction vs RDMReg" in svg
    assert "RDM / reconstruction gradient RMS" in svg
    assert "Active fraction" in svg and "Reconstruction and FVU" in svg
    assert "Thresholded active fractions (train)" in svg
    assert "Global-local" not in svg and "Gate transitions" not in svg


@pytest.mark.parametrize(
    "power, random_power, axis_power",
    [(None, None, None), ("1", None, None), ("2", None, None),
     ("2", "2", "1"), ("1", "2", None), (None, None, "1")],
)
def test_launcher_overrides_and_fresh_output_guard(tmp_path, power, random_power, axis_power):
    bash = shutil.which("bash")
    if sys.platform == "win32":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(candidate) if candidate.is_file() else None
    if not bash:
        pytest.skip("Bash is required for launcher test")
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "lejepa-train"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n', newline="\n", encoding="utf-8")
    stub.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_rdm_sae.sh"
    output = tmp_path / "new run"
    env = {
        **os.environ,
        "RDM_WEIGHT": "0.5", "RECONSTRUCTION_WEIGHT": "2", "RDM_TARGET_SCALE": "3",
        "EXPECTED_L0_FRACTION": "0.03", "FEATURE_ACTIVATION": "relu",
        "BATCH_SIZE": "256", "GRAD_ACCUM": "2", "OUTPUT_DIR": "ignored-positional-wins",
    }
    for key, value in (
        ("RDM_WASSERSTEIN_POWER", power),
        ("RDM_RANDOM_WASSERSTEIN_POWER", random_power),
        ("RDM_AXIS_WASSERSTEIN_POWER", axis_power),
    ):
        env.pop(key, None)
        if value is not None:
            env[key] = value
    command = [
        bash, "-c",
        'export PATH="$(cd "$1" && pwd):$(cd "$2" && pwd):$PATH"; shift 2; exec "$@"',
        "launcher-test", str(stub_dir), str(Path(bash).parent), bash, str(script), str(output),
    ]
    result = subprocess.run(command, capture_output=True, encoding="utf-8", check=True, env=env)
    args = result.stdout.splitlines()
    config = load_config(args[1])
    apply_overrides(config, args[3::2])
    assert config.model.type == "rdm_sae" and config.model.num_local_views == 0
    assert config.model.feature_activation == "relu"
    assert config.loss.lambda_rdm == 0.5 and config.loss.reconstruction_weight == 2
    assert config.loss.rdm_wasserstein_power == int(power or 2)
    assert config.loss.random_wasserstein_power == int(random_power or power or 2)
    assert config.loss.axis_wasserstein_power == int(axis_power or power or 2)
    assert config.loss.rdm_random_wasserstein_power == (int(random_power) if random_power else None)
    assert config.loss.rdm_axis_wasserstein_power == (int(axis_power) if axis_power else None)
    assert config.loss.rdm_target_scale == 3 and config.loss.expected_l0_fraction == 0.03
    assert config.train.batch_size == 256 and config.train.gradient_accumulation_steps == 2
    assert config.train.seed == 42 and config.train.max_steps == 10000
    assert config.train.resume_from is None and config.train.output_dir == str(output)
    default_env = {key: value for key, value in env.items() if key != "OUTPUT_DIR"}
    default_run = subprocess.run(
        command[:-1], capture_output=True, encoding="utf-8", check=True, env=default_env,
    )
    default_output = default_run.stdout.splitlines()[-1]
    expected_random = int(random_power or power or 2)
    expected_axis = int(axis_power or power or 2)
    tag = (f"wp{expected_random}" if expected_random == expected_axis
           else f"wpr{expected_random}-wpa{expected_axis}")
    assert f"-{tag}-axis" in default_output
    output.mkdir()
    blocked = subprocess.run(command, capture_output=True, encoding="utf-8", env=env)
    assert blocked.returncode == 1 and "Refusing existing output" in blocked.stderr
    assert not blocked.stdout
    for key in (
        "RDM_WASSERSTEIN_POWER", "RDM_RANDOM_WASSERSTEIN_POWER", "RDM_AXIS_WASSERSTEIN_POWER",
    ):
        invalid = subprocess.run(
            command, capture_output=True, encoding="utf-8", env={**env, key: "3"},
        )
        assert invalid.returncode == 2 and f"{key} must be 1 or 2" in invalid.stderr
        assert not invalid.stdout
