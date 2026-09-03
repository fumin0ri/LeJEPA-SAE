import itertools
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from lejepa_sae.config import ExperimentConfig, ModelConfig, load_config
from lejepa_sae.models import ProposedModel
from lejepa_sae.train import apply_overrides
from lejepa_sae.views import sample_dimension_masks

MODES = [("inverted", 2.0), ("sqrt", 2.0**0.5), ("none", 1.0)]


@pytest.mark.parametrize(("mode", "multiplier"), MODES)
def test_mask_scaling_values_and_pre_bias_gradients(mode, multiplier):
    model = ProposedModel(ModelConfig(d_llm=4, feature_dim=8, mask_scaling=mode))
    with torch.no_grad():
        model.pre_bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    residuals = torch.tensor([[3.0, 5.0, 7.0, 9.0]], requires_grad=True)
    mask = torch.tensor([[True, False, True, False]])
    prepared = model.prepare_input(residuals, mask)
    torch.testing.assert_close(prepared, torch.tensor([[2.0, 0.0, 4.0, 0.0]]) * multiplier)
    prepared.sum().backward()
    expected_gradient = mask.float() * multiplier
    torch.testing.assert_close(residuals.grad, expected_gradient)
    torch.testing.assert_close(model.pre_bias.grad, -expected_gradient[0])


@pytest.mark.parametrize(("mode", "_multiplier"), MODES)
def test_global_full_mask_matches_unmasked_values_and_gradients(mode, _multiplier):
    model = ProposedModel(ModelConfig(d_llm=4, feature_dim=8, mask_scaling=mode))
    residuals = torch.randn(2, 3, 4)
    unmasked = model(residuals).features
    masked = model(residuals, torch.ones_like(residuals, dtype=torch.bool)).features
    torch.testing.assert_close(masked, unmasked)
    parameters = tuple(model.parameters())
    expected_gradients = torch.autograd.grad(unmasked.sum(), parameters)
    gradients = torch.autograd.grad(masked.sum(), parameters)
    for expected, actual in zip(expected_gradients, gradients, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("mode", "_multiplier"), MODES)
def test_mask_counts_and_scaling_use_actual_rounded_keep_rate(mode, _multiplier):
    model = ProposedModel(
        ModelConfig(d_llm=7, feature_dim=8, dimension_keep_fraction=0.3, mask_scaling=mode)
    )
    residuals = torch.ones(8, 7)
    mask = sample_dimension_masks(residuals, 1, 0.3)[0]
    assert torch.all(mask.sum(-1) == 2)
    divisor = {"inverted": 2 / 7, "sqrt": (2 / 7)**0.5, "none": 1.0}[mode]
    torch.testing.assert_close(model.prepare_input(residuals, mask), mask.float() / divisor)


@pytest.mark.parametrize(("mode", "multiplier"), MODES)
def test_half_mask_expectations_over_all_exact_masks(mode, multiplier):
    # Exhaustive masks avoid a probabilistic tolerance: sqrt preserves input energy, not mean.
    model = ProposedModel(ModelConfig(d_llm=4, feature_dim=8, mask_scaling=mode))
    residual = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    inputs = []
    for coordinates in itertools.combinations(range(4), 2):
        mask = torch.zeros_like(residual, dtype=torch.bool)
        mask[:, list(coordinates)] = True
        inputs.append(model.prepare_input(residual, mask))
    stacked = torch.stack(inputs)
    torch.testing.assert_close(stacked.mean(0), residual * (0.5 * multiplier))
    expected_energy = residual.square().sum() * (0.5 * multiplier**2)
    torch.testing.assert_close(stacked.square().sum(-1).mean(), expected_energy)


def test_scaling_config_overrides_roundtrip_and_legacy_default(tmp_path):
    config = ExperimentConfig(model=ModelConfig(d_llm=4, feature_dim=8))
    config.loss.axis_projections = 4
    apply_overrides(config, ["model.mask_scaling=none", "model.dimension_keep_fraction=0.5"])
    path = tmp_path / "config.yaml"
    raw = config.to_dict()
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_config(path).model.mask_scaling == "none"
    del raw["model"]["mask_scaling"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_config(path).model.mask_scaling == "inverted"
    legacy_model = ProposedModel(load_config(path).model)
    new_model = ProposedModel(ModelConfig(d_llm=4, feature_dim=8, mask_scaling="sqrt"))
    new_model.load_state_dict(legacy_model.state_dict(), strict=True)


def test_unknown_mask_scaling_is_rejected():
    config = ExperimentConfig()
    with pytest.raises(ValueError, match="model.mask_scaling"):
        apply_overrides(config, ["model.mask_scaling=invalid"])


def _native_bash():
    if os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        if candidate.is_file():
            return str(candidate)
    else:
        executable = shutil.which("bash")
        if executable:
            return executable
    pytest.skip("Native bash is unavailable")


@pytest.mark.parametrize("mode", ["inverted", "sqrt", "none"])
@pytest.mark.parametrize("script", ["run_proposed.sh", "run_leaky_backward.sh"])
@pytest.mark.parametrize("rate_weight", ["0", "0.3"])
def test_script_passes_mask_scaling_and_separates_output_paths(mode, script, rate_weight):
    environment = os.environ.copy()
    environment.update(
        MASK_SCALING=mode,
        DIMENSION_KEEP_FRACTION="0.5",
        FEATURE_ACTIVATION="relu",
        FEATURE_DIM="16384",
        EXPECTED_L0_FRACTION="0.009765625",
        AXIS_PROJECTIONS="512",
        LEAKY_BACKWARD_SLOPE="0.01",
        RATE_WEIGHT=rate_weight,
        RATE_TEMPERATURE="0.2", RATE_SCALE_FLOOR="0.000001", RATE_GRAD_DIAGNOSTICS="true",
    )
    completed = subprocess.run(
        [
            _native_bash(), "-c",
            'lejepa-train() { printf "%s\\n" "$@"; }; export -f lejepa-train; '
            'exec "$BASH" "$@"',
            "mask-scaling-test", f"scripts/{script}", "configs/pythia-6.9b-layer16.yaml",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    arguments = completed.stdout.splitlines()
    assert f"model.mask_scaling={mode}" in arguments
    assert "model.dimension_keep_fraction=0.5" in arguments
    assert f"loss.rate_weight={rate_weight}" in arguments
    assert "loss.rate_temperature=0.2" in arguments
    assert "loss.rate_gradient_diagnostics=true" in arguments
    output = next(arg for arg in arguments if arg.startswith("train.output_dir="))
    if rate_weight != "0":
        assert output.endswith("-rate0.3-tau0.2-floor0.000001")
        output = output.removesuffix("-rate0.3-tau0.2-floor0.000001")
    if mode == "inverted":
        assert "-mask-" not in output
    else:
        assert output.endswith(f"-q0.5-mask-{mode}")
    if script == "run_leaky_backward.sh":
        assert "model.feature_activation=relu_forward_leaky_backward" in arguments
    assert "train.resume_from=null" in arguments


@pytest.mark.parametrize("mode", ["base", "rate", "both"])
def test_rate_comparison_launcher_same_settings_fresh_outputs(tmp_path, mode):
    environment = os.environ.copy()
    root = tmp_path / "rate runs"
    environment.update(
        RATE_COMPARISON_ROOT=root.as_posix(),
        RATE_WEIGHT="0.2", RATE_TEMPERATURE="0.1", RATE_SCALE_FLOOR="0.000001",
        EXPECTED_L0_FRACTION="0.05", LEAKY_BACKWARD_SLOPE="0.1", MASK_SCALING="sqrt",
        DIMENSION_KEEP_FRACTION="0.5", TRAIN_SEED="42", MAX_STEPS="2",
        RATE_GRAD_DIAGNOSTICS="true",
        PATH=str(Path(sys.executable).parent) + os.pathsep + environment["PATH"],
    )
    completed = subprocess.run(
        [
            _native_bash(), "-c",
            'lejepa-train() { printf "__RUN__\\n"; printf "%s\\n" "$@"; }; '
            'export -f lejepa-train; exec "$BASH" "$@"',
            "rate-test", "scripts/run_rate_comparison.sh", mode,
        ],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, check=True, timeout=30,
    )
    runs = [part.splitlines() for part in completed.stdout.split("__RUN__\n")[1:]]
    assert len(runs) == (2 if mode == "both" else 1)
    for index, arguments in enumerate(runs):
        is_base = mode == "base" or (mode == "both" and index == 0)
        assert f"loss.rate_weight={'0' if is_base else '0.2'}" in arguments
        assert f"train.output_dir={root.as_posix()}/{'base' if is_base else 'rate'}" in arguments
        assert "loss.expected_l0_fraction=0.05" in arguments
        assert "model.leaky_backward_slope=0.1" in arguments
        assert "model.mask_scaling=sqrt" in arguments
        assert "model.dimension_keep_fraction=0.5" in arguments
        assert "train.seed=42" in arguments
        assert "train.max_steps=2" in arguments
        assert "train.resume_from=null" in arguments
    if mode == "both":
        def common(args):
            return [
                arg for arg in args
                if not arg.startswith(("loss.rate_weight=", "train.output_dir="))
            ]

        assert common(runs[0]) == common(runs[1])


def test_rate_comparison_refuses_existing_destination_before_any_training(tmp_path):
    root = tmp_path / "existing"
    (root / "rate").mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        RATE_COMPARISON_ROOT=root.as_posix(), RATE_WEIGHT="1",
        PATH=str(Path(sys.executable).parent) + os.pathsep + environment["PATH"],
    )
    completed = subprocess.run(
        [_native_bash(), "scripts/run_rate_comparison.sh", "both"],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode != 0
    assert "Refusing existing output" in completed.stderr
    assert not (root / "base").exists()


def test_rate_comparison_rejects_zero_coefficient_before_training(tmp_path):
    environment = os.environ.copy()
    environment.update(
        RATE_COMPARISON_ROOT=tmp_path.as_posix(), RATE_WEIGHT="0",
        PATH=str(Path(sys.executable).parent) + os.pathsep + environment["PATH"],
    )
    completed = subprocess.run(
        [_native_bash(), "scripts/run_rate_comparison.sh", "both"],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode != 0
    assert "RATE_WEIGHT must be finite and positive" in completed.stderr
    assert not (tmp_path / "base").exists()
