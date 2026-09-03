import copy

import pytest
import torch
import torch.nn.functional as F

from lejepa_sae.config import DataConfig, ExperimentConfig, ModelConfig, load_config
from lejepa_sae.models import (
    BatchTopKSAE,
    JumpReLUSAE,
    MatryoshkaSAE,
    ProposedModel,
    batch_topk,
    build_model,
    relu_forward_leaky_backward,
)
from lejepa_sae.train import compute_loss, resolve_training_steps, stack_dimension_views


def tiny_config(model_type: str = "proposed") -> ExperimentConfig:
    config = ExperimentConfig(
        data=DataConfig(window_size=1, num_workers=0),
        model=ModelConfig(
            type=model_type,
            d_llm=8,
            feature_dim=16,
            num_local_views=2,
        ),
    )
    config.loss.rdm_projections = 4
    config.loss.axis_projections = 4
    config.baseline.k = 2
    config.baseline.k_aux = 4
    config.baseline.matryoshka_group_sizes = [2, 2, 4, 4, 4]
    config.validate()
    return config


@pytest.mark.parametrize(
    "model_type", ["proposed", "batch_topk_sae", "jump_relu_sae", "matryoshka_sae"]
)
def test_all_models_complete_backward_step(model_type):
    torch.manual_seed(2)
    config = tiny_config(model_type)
    model = build_model(config)
    residuals = torch.randn(4, 1, 8)
    loss, metrics = compute_loss(model, residuals, config)
    loss.backward()
    assert torch.isfinite(loss)
    assert "loss" in metrics
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_proposed_is_the_dimension_mask_jepa_model():
    assert isinstance(build_model(tiny_config()), ProposedModel)


@pytest.mark.parametrize(
    ("model_type", "expected_type"),
    [
        ("batch_topk_sae", BatchTopKSAE),
        ("jump_relu_sae", JumpReLUSAE),
        ("matryoshka_sae", MatryoshkaSAE),
    ],
)
def test_baseline_builders(model_type, expected_type):
    assert isinstance(build_model(tiny_config(model_type)), expected_type)


def test_batch_topk_keeps_exact_batch_times_k_and_is_permutation_equivariant():
    values = torch.arange(1, 49, dtype=torch.float32).reshape(6, 8)
    selected = batch_topk(values, 2)
    assert int((selected > 0).sum()) == 12
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    torch.testing.assert_close(batch_topk(values[permutation], 2), selected[permutation])


def test_sae_decoder_columns_are_unit_norm_and_encoder_is_tied_at_initialization():
    model = build_model(tiny_config("batch_topk_sae"))
    torch.testing.assert_close(model.decoder.weight.norm(dim=0), torch.ones(16))
    torch.testing.assert_close(model.encoder.weight, model.decoder.weight.T)


def test_jumprelu_threshold_is_positive_and_l0_has_threshold_gradient():
    model = build_model(tiny_config("jump_relu_sae"))
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.fill_(0.001)
    residuals = torch.randn(8, 8)
    output = model(residuals)
    penalty = model.l0_surrogate(output.preactivations).sum()
    penalty.backward()
    assert torch.all(model.threshold > 0)
    assert model.log_threshold.grad is not None
    assert torch.count_nonzero(model.log_threshold.grad) == model.log_threshold.numel()


def test_matryoshka_prefixes_cover_full_width():
    model = build_model(tiny_config("matryoshka_sae"))
    assert model.prefix_widths == (2, 4, 8, 12, 16)


def test_matryoshka_loss_is_equal_weighted_prefix_mean():
    config = tiny_config("matryoshka_sae")
    config.baseline.auxk_coefficient = 0.0
    model = build_model(config)
    loss, metrics = compute_loss(model, torch.randn(6, 1, 8), config)
    expected = torch.stack(
        [metrics[f"prefix_{width}_mse"] for width in model.prefix_widths]
    ).mean()
    torch.testing.assert_close(loss.detach(), expected)


def test_baseline_expensive_diagnostics_can_be_skipped():
    config = tiny_config("batch_topk_sae")
    model = build_model(config)
    _, metrics = compute_loss(
        model, torch.randn(6, 1, 8), config, include_diagnostics=False
    )
    assert "reconstruction" in metrics
    assert "active_fraction" not in metrics
    assert "feature_std" not in metrics


def test_decoder_gradient_projection_removes_radial_component():
    model = build_model(tiny_config("batch_topk_sae"))
    model.decoder.weight.grad = model.decoder.weight.detach().clone()
    model.project_decoder_gradients_()
    radial = (model.decoder.weight * model.decoder.weight.grad).sum(dim=0)
    torch.testing.assert_close(radial, torch.zeros_like(radial), atol=1e-6, rtol=0)


def test_relu_forward_leaky_backward_has_relu_values_and_surrogate_gradient():
    inputs = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
    outputs = relu_forward_leaky_backward(inputs, negative_slope=0.05)

    torch.testing.assert_close(outputs, torch.tensor([0.0, 0.0, 3.0]))
    outputs.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.tensor([0.05, 0.05, 1.0]))


def test_leaky_backward_can_update_features_with_negative_preactivations():
    regular_config = tiny_config()
    leaky_config = tiny_config()
    leaky_config.model.feature_activation = "relu_forward_leaky_backward"
    leaky_config.model.leaky_backward_slope = 0.02
    leaky_config.validate()

    regular_model = build_model(regular_config)
    leaky_model = build_model(leaky_config)
    leaky_model.load_state_dict(regular_model.state_dict())
    for model in (regular_model, leaky_model):
        with torch.no_grad():
            model.encoder.weight.zero_()
            model.encoder.bias.fill_(-1.0)

    residuals = torch.ones(4, regular_config.model.d_llm)
    regular_features = regular_model(residuals).features
    leaky_features = leaky_model(residuals).features
    torch.testing.assert_close(regular_features, leaky_features)
    assert torch.count_nonzero(leaky_features) == 0

    regular_features.sum().backward()
    leaky_features.sum().backward()
    assert torch.count_nonzero(regular_model.encoder.bias.grad) == 0
    torch.testing.assert_close(
        leaky_model.encoder.bias.grad,
        torch.full_like(leaky_model.encoder.bias, 4 * 0.02),
    )


def test_proposed_reports_collapse_diagnostics():
    config = tiny_config()
    model = build_model(config)
    residuals = torch.randn(8, 1, config.model.d_llm)
    loss, metrics = compute_loss(model, residuals, config)
    expected = {
        "global_distribution",
        "local_distribution",
        "global_rdm_contribution",
        "local_rdm_contribution",
        "random_distribution",
        "axis_distribution",
        "global_random_distribution",
        "local_random_distribution",
        "global_axis_distribution",
        "local_axis_distribution",
        "l0_sparsity",
        "l1_sparsity",
        "global_active_fraction",
        "local_active_fraction",
        "off_to_on",
        "on_to_off",
        "local_global_active_fraction_gap",
        "transition_rate_gap",
        "global_feature_std",
        "local_feature_std",
        "global_dead_feature_fraction",
        "local_dead_feature_fraction",
        "expected_l0_fraction",
    }
    assert expected <= metrics.keys()
    torch.testing.assert_close(
        metrics["local_active_fraction"] - metrics["global_active_fraction"],
        metrics["off_to_on"] - metrics["on_to_off"],
    )
    assert torch.isfinite(loss)
    torch.testing.assert_close(
        metrics["distribution"],
        metrics["global_rdm_contribution"] + metrics["local_rdm_contribution"],
    )


def test_proposed_can_skip_expensive_diagnostics():
    config = tiny_config()
    model = build_model(config)
    residuals = torch.randn(8, 1, config.model.d_llm)
    _, metrics = compute_loss(model, residuals, config, include_diagnostics=False)

    assert {
        "loss",
        "invariance",
        "distribution",
        "random_distribution",
        "axis_distribution",
        "global_distribution",
        "local_distribution",
        "global_rdm_contribution",
        "local_rdm_contribution",
        "global_random_distribution",
        "local_random_distribution",
        "global_axis_distribution",
        "local_axis_distribution",
        "expected_l0_fraction",
    } == metrics.keys()
    assert "feature_std" not in metrics


@pytest.mark.parametrize("mask_scaling", ["inverted", "sqrt", "none"])
def test_batched_dimension_views_match_individual_forwards_and_gradients(mask_scaling):
    torch.manual_seed(15)
    config = tiny_config()
    config.model.mask_scaling = mask_scaling
    batched_model = build_model(config)
    loop_model = copy.deepcopy(batched_model)
    residuals = torch.randn(6, config.model.d_llm)
    local_masks = [
        torch.tensor(
            [
                [
                    (column + row + view) % 2 == 0
                    for column in range(config.model.d_llm)
                ]
                for row in range(residuals.shape[0])
            ]
        )
        for view in range(config.model.num_local_views)
    ]

    residual_views, masks = stack_dimension_views(
        residuals, local_masks, include_global=True
    )
    batched_features = batched_model(residual_views, masks).features
    loop_features = torch.stack(
        [
            loop_model(residuals).features,
            *(loop_model(residuals, mask).features for mask in local_masks),
        ]
    )
    torch.testing.assert_close(batched_features, loop_features)

    batched_features.square().mean().backward()
    loop_features.square().mean().backward()
    for batched_parameter, loop_parameter in zip(
        batched_model.parameters(), loop_model.parameters(), strict=True
    ):
        torch.testing.assert_close(batched_parameter.grad, loop_parameter.grad)


def test_masked_encoder_fills_missing_coordinates_with_pre_bias():
    model = build_model(tiny_config())
    with torch.no_grad():
        model.pre_bias.copy_(torch.arange(8).float())
    residuals = torch.arange(8).float().unsqueeze(0) + 2.0
    mask = torch.tensor([[True, False, True, False, True, False, True, False]])
    prepared = model.prepare_input(residuals, mask)
    assert torch.equal(prepared[~mask], torch.zeros(4))
    torch.testing.assert_close(prepared[mask], torch.full((4,), 4.0))


def test_dense_mask_projection_matches_coordinate_indexed_projection():
    torch.manual_seed(5)
    model = build_model(tiny_config())
    residuals = torch.randn(3, 8)
    masks = torch.tensor(
        [
            [1, 1, 0, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 0, 0, 1],
        ],
        dtype=torch.bool,
    )
    dense = model.encoder(model.prepare_input(residuals, masks))
    indexed = []
    for row in range(residuals.shape[0]):
        selected = masks[row]
        centered = residuals[row, selected] - model.pre_bias[selected]
        indexed.append(
            F.linear(centered / 0.5, model.encoder.weight[:, selected], model.encoder.bias)
        )
    torch.testing.assert_close(dense, torch.stack(indexed))


def test_all_model_types_require_one_token():
    config = tiny_config()
    config.data.window_size = 2
    with pytest.raises(ValueError, match="window_size=1"):
        config.validate()


@pytest.mark.parametrize("legacy_type", ["standard_sae", "dimension_denoising_sae"])
def test_removed_baseline_types_are_rejected(legacy_type):
    config = tiny_config()
    config.model.type = legacy_type
    with pytest.raises(ValueError, match="model.type"):
        config.validate()


def test_default_baseline_target_l0_is_derived_from_fraction():
    config = load_config("configs/pythia-6.9b-layer16.yaml")
    config.model.type = "batch_topk_sae"
    config.validate()
    assert config.target_l0 == 160


@pytest.mark.parametrize("keep_fraction", [0.0, 1.1])
def test_dimension_keep_fraction_is_validated(keep_fraction):
    config = tiny_config()
    config.model.dimension_keep_fraction = keep_fraction
    with pytest.raises(ValueError, match="dimension_keep_fraction"):
        config.validate()


@pytest.mark.parametrize("slope", [0.0, -0.01, 1.01])
def test_leaky_backward_slope_is_validated(slope):
    config = tiny_config()
    config.model.leaky_backward_slope = slope
    with pytest.raises(ValueError, match="leaky_backward_slope"):
        config.validate()


def test_feature_activation_is_validated():
    config = tiny_config()
    config.model.feature_activation = "gelu"
    with pytest.raises(ValueError, match="feature_activation"):
        config.validate()


@pytest.mark.parametrize("model_type", ["batch_topk_sae", "jump_relu_sae", "matryoshka_sae"])
def test_leaky_backward_is_not_enabled_on_topk_or_jumprelu_baselines(model_type):
    config = tiny_config(model_type)
    config.model.feature_activation = "relu_forward_leaky_backward"
    with pytest.raises(ValueError, match="only supported for proposed and rdm_sae"):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rdm_projections", 0, "rdm_projections"),
        ("axis_projections", 0, "axis_projections"),
        ("axis_projections", 17, "axis_projections"),
        ("axis_weight", 0.0, "axis_weight"),
        ("lp_norm_parameter", 0.0, "lp_norm_parameter"),
        ("expected_l0_fraction", 0.0, "expected_l0_fraction"),
        ("expected_l0_fraction", 1.0, "expected_l0_fraction"),
        ("target_distribution", "rectified_gaussian", "target_distribution"),
        ("mode_of_sigma", "sigma_RGN", "mode_of_sigma"),
        ("projection_vectors_type", "svd", "projection_vectors_type"),
    ],
)
def test_proposed_paper_rdm_config_is_validated(field, value, message):
    config = tiny_config()
    setattr(config.loss, field, value)
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_main_preset_uses_single_token_paper_defaults():
    config = load_config("configs/pythia-6.9b-layer16.yaml")
    assert config.model.type == "proposed"
    assert config.data.window_size == 1
    assert config.model.num_local_views == 4
    assert config.model.mask_scaling == "inverted"
    assert config.model.feature_dim == 16384
    assert config.model.feature_activation == "relu"
    assert config.model.leaky_backward_slope == 0.01
    assert config.loss.target_distribution == "rectified_lp_distribution"
    assert config.loss.lp_norm_parameter == 1.0
    assert config.loss.expected_l0_fraction == 0.009765625
    assert config.loss.mean_shift_value == 0.0
    assert config.loss.mode_of_sigma == "sigma_GN"
    assert config.loss.projection_vectors_type == "random"
    assert config.loss.rdm_projections == 8192
    assert config.loss.axis_projections == 512
    assert config.loss.axis_weight == 1.0
    assert config.loss.invariance_weight == 25.0
    assert config.loss.lambda_rdm == 125.0
    assert config.target_l0 == 160
    assert config.baseline.auxk_coefficient == pytest.approx(1 / 32)
    assert config.baseline.dead_feature_window_tokens == 10_000_000
    assert config.baseline.k_aux == 2048
    assert config.baseline.jump_relu_initial_threshold == 0.001
    assert config.baseline.jump_relu_bandwidth == 0.001
    assert config.baseline.jump_relu_sparsity_warmup_steps == 10_000
    assert config.baseline.matryoshka_group_sizes == [512, 1024, 2048, 4096, 8704]
    assert config.baseline.matryoshka_weights == [0.2] * 5
    assert config.train.batch_size == 512
    assert config.train.gradient_accumulation_steps == 1
    assert config.train.max_steps == "one_epoch"
    assert config.train.eval_batches == 12
    assert config.train.checkpoint_every == 10000
    assert config.train.output_dir.endswith(
        "/proposed-d16384-l0-0.009765625-axis512"
    )
    assert config.train.resume_from is None


def test_one_epoch_steps_use_complete_optimizer_batches():
    assert resolve_training_steps("one_epoch", 191_406, 1) == 191_406
    assert resolve_training_steps("one_epoch", 382_812, 2) == 191_406
    assert resolve_training_steps(10_000, 191_406, 1) == 10_000


def test_one_epoch_requires_at_least_one_optimizer_step():
    with pytest.raises(ValueError, match="does not contain enough batches"):
        resolve_training_steps("one_epoch", 1, 2)
