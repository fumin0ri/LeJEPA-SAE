import copy
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import yaml

from lejepa_sae.config import ExperimentConfig, ModelConfig, load_config
from lejepa_sae.losses import (
    generalized_gaussian_mean_shift_for_active_fraction,
    rectified_lp_rdm_regularization,
)
from lejepa_sae.models import build_model
from lejepa_sae.probing import ProbeSAEAdapter
from lejepa_sae.train import apply_overrides, compute_loss


def global_config():
    config = ExperimentConfig(model=ModelConfig(d_llm=8, feature_dim=16, num_local_views=0))
    config.loss.invariance_weight = 0
    config.loss.rate_weight = 0
    config.loss.rate_gradient_diagnostics = True  # Must stay inactive with rate_weight=0.
    config.loss.expected_l0_fraction = 0.05
    config.loss.rdm_projections = 4
    config.loss.axis_projections = 4
    config.loss.axis_weight = 2.0
    config.train.device = "cpu"
    config.train.precision = "float32"
    config.validate()
    return config


@pytest.mark.parametrize("activation", ["relu", "relu_forward_leaky_backward"])
@pytest.mark.parametrize("mask_scaling", ["inverted", "sqrt", "none"])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_global_only_matches_direct_rdm_loss_and_gradients(
    monkeypatch, activation, mask_scaling, mixed_precision
):
    config = global_config()
    config.model.feature_activation = activation
    config.model.leaky_backward_slope = 0.1
    config.model.mask_scaling = mask_scaling
    model = build_model(config)
    with torch.no_grad():
        model.pre_bias.copy_(torch.linspace(-1, 1, 8))
    reference = copy.deepcopy(model)
    residuals = torch.randn(8, 1, 8)
    axes = torch.arange(4)
    forward_shapes = []
    rdm_shapes = []

    def check_forward(_model, inputs):
        assert len(inputs) == 1  # No mask, including an all-ones mask.
        forward_shapes.append(tuple(inputs[0].shape))

    def capture_rdm(features, *args, **kwargs):
        rdm_shapes.append(tuple(features.shape))
        return rectified_lp_rdm_regularization(features, *args, **kwargs)

    def forbidden(*_args, **_kwargs):
        pytest.fail("global-only training must not generate masks or compute invariance/rate")

    model.register_forward_pre_hook(check_forward)
    monkeypatch.setattr("lejepa_sae.train.sample_dimension_masks", forbidden)
    monkeypatch.setattr("lejepa_sae.train.stack_dimension_views", forbidden)
    monkeypatch.setattr("lejepa_sae.train.F.mse_loss", forbidden)
    monkeypatch.setattr("lejepa_sae.train.target_rate_regularization", forbidden)
    monkeypatch.setattr("lejepa_sae.train.rectified_lp_rdm_regularization", capture_rdm)

    def autocast():
        return torch.autocast("cpu", dtype=torch.bfloat16) if mixed_precision else nullcontext()

    torch.manual_seed(71)
    with autocast():
        loss, metrics = compute_loss(model, residuals, config, axis_indices=axes)
    torch.manual_seed(71)
    with autocast():
        expected_features = reference(residuals[:, 0]).features
        expected_rdm = rectified_lp_rdm_regularization(
            expected_features.unsqueeze(0),
            config.loss.rdm_projections,
            config.loss.axis_projections,
            config.loss.axis_weight,
            config.loss.lp_norm_parameter,
            generalized_gaussian_mean_shift_for_active_fraction(
                config.loss.lp_norm_parameter, config.loss.expected_l0_fraction
            ),
            axis_indices=axes,
        )
        expected_loss = config.loss.lambda_rdm * expected_rdm.loss
    assert forward_shapes == [(8, 8)]
    assert rdm_shapes == [(1, 8, 16)]
    torch.testing.assert_close(loss, expected_loss)
    # The single-view RDM contribution is 1.0, not the multi-view global factor 0.5.
    torch.testing.assert_close(
        metrics["distribution"],
        expected_rdm.random_view_losses[0]
        + config.loss.axis_weight * expected_rdm.axis_view_losses[0],
    )
    torch.testing.assert_close(metrics["global_rdm_contribution"], metrics["distribution"])
    torch.testing.assert_close(metrics["global_distribution"], metrics["distribution"])
    torch.testing.assert_close(metrics["active_fraction"], metrics["global_active_fraction"])
    torch.testing.assert_close(metrics["feature_std"], metrics["global_feature_std"])
    assert "invariance" not in metrics
    assert "off_to_on" not in metrics
    assert "on_to_off" not in metrics
    assert "rate_loss" not in metrics
    assert not any("local" in key for key in metrics)
    assert all(torch.isfinite(value).all() for value in metrics.values())
    loss.backward()
    expected_loss.backward()
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad)


def test_global_only_skips_diagnostics_and_probe_still_encodes_full_input():
    config = global_config()
    model = build_model(config)
    inputs = torch.randn(6, 1, 8)
    _, metrics = compute_loss(model, inputs, config, include_diagnostics=False)
    assert "global_distribution" in metrics
    assert "feature_std" not in metrics
    assert "global_active_fraction" not in metrics
    assert "invariance" not in metrics
    torch.testing.assert_close(
        ProbeSAEAdapter(model, config).encode(inputs), model(inputs).features
    )


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("model", "num_local_views", -1, "non-negative integer"),
        ("model", "num_local_views", 0.5, "non-negative integer"),
        ("model", "num_local_views", False, "non-negative integer"),
        ("model", "type", "batch_topk_sae", "only supported"),
        ("loss", "invariance_weight", 25.0, "invariance_weight=0"),
        ("loss", "rate_weight", 1.0, "rate_weight=0"),
        ("loss", "lambda_rdm", 0.0, "positive lambda_rdm"),
        ("loss", "lambda_rdm", float("nan"), "positive lambda_rdm"),
    ],
)
def test_global_only_config_rejects_incompatible_losses(section, field, value, message):
    config = global_config()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_global_only_launcher_inherits_config_and_refuses_existing_output(tmp_path):
    bash = shutil.which("bash")
    if sys.platform == "win32":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(candidate) if candidate.is_file() else None
    if not bash:
        pytest.skip("Bash is required for the launcher smoke test")
    source = tmp_path / "base run"
    source.mkdir()
    config = ExperimentConfig()
    config.model.feature_activation = "relu_forward_leaky_backward"
    config.model.leaky_backward_slope = 0.1
    config.model.mask_scaling = "sqrt"
    config.loss.expected_l0_fraction = 0.05
    config.train.seed = 43  # The launcher deliberately fixes seed 42 and 10k steps.
    config_path = source / "config.resolved.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "lejepa-train"
    stub.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n', encoding="utf-8", newline="\n"
    )
    stub.chmod(0o755)
    output = tmp_path / "new run"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_global_rdm_only.sh"
    command = [
        bash, "-c",
        'export PATH="$(cd "$1" && pwd):$(cd "$2" && pwd):$PATH"; shift 2; exec "$@"',
        "test-launcher", str(stub_dir), str(Path(bash).parent),
        bash, str(script), str(source), str(output),
    ]
    result = subprocess.run(command, capture_output=True, encoding="utf-8", check=True)
    arguments = result.stdout.splitlines()
    assert arguments[0] == "--config"
    assert Path(arguments[1]) == config_path
    assert all(value == "--set" for value in arguments[2::2])
    resolved = load_config(config_path)
    apply_overrides(resolved, arguments[3::2])
    assert resolved.model.num_local_views == 0
    assert resolved.loss.invariance_weight == resolved.loss.rate_weight == 0
    assert resolved.train.max_steps == resolved.train.checkpoint_every == 10000
    assert resolved.train.seed == 42
    assert resolved.train.resume_from is None
    assert resolved.train.output_dir == str(output)
    assert resolved.model.feature_activation == config.model.feature_activation
    assert resolved.model.leaky_backward_slope == 0.1
    assert resolved.loss.lambda_rdm == 125
    assert resolved.loss.rdm_projections == 8192
    assert resolved.loss.axis_projections == 512
    assert resolved.loss.expected_l0_fraction == 0.05
    assert resolved.train.batch_size == 512
    assert resolved.train.gradient_accumulation_steps == 1
    assert load_config(config_path).to_dict() == config.to_dict()
    output.mkdir()
    blocked = subprocess.run(command, capture_output=True, encoding="utf-8")
    assert blocked.returncode == 1
    assert "Refusing existing output" in blocked.stderr
    assert not blocked.stdout
