import copy

import pytest
import torch
import torch.nn.functional as F

from lejepa_sae.config import DataConfig, ExperimentConfig, ModelConfig, load_config
from lejepa_sae.models import ProposedModel, build_model
from lejepa_sae.train import compute_loss, stack_dimension_views


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
    config.validate()
    return config


@pytest.mark.parametrize("model_type", ["proposed", "standard_sae", "dimension_denoising_sae"])
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


def test_proposed_reports_collapse_diagnostics():
    config = tiny_config()
    model = build_model(config)
    residuals = torch.randn(8, 1, config.model.d_llm)
    loss, metrics = compute_loss(model, residuals, config)
    expected = {
        "global_distribution",
        "local_distribution",
        "l0_sparsity",
        "l1_sparsity",
        "global_active_fraction",
        "local_active_fraction",
        "global_feature_std",
        "local_feature_std",
        "global_dead_feature_fraction",
        "local_dead_feature_fraction",
    }
    assert expected <= metrics.keys()
    assert torch.isfinite(loss)


def test_proposed_can_skip_expensive_diagnostics():
    config = tiny_config()
    model = build_model(config)
    residuals = torch.randn(8, 1, config.model.d_llm)
    _, metrics = compute_loss(model, residuals, config, include_diagnostics=False)

    assert {
        "loss",
        "invariance",
        "distribution",
        "global_distribution",
        "local_distribution",
    } == metrics.keys()
    assert "feature_std" not in metrics


def test_batched_dimension_views_match_individual_forwards_and_gradients():
    torch.manual_seed(15)
    config = tiny_config()
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


@pytest.mark.parametrize("keep_fraction", [0.0, 1.1])
def test_dimension_keep_fraction_is_validated(keep_fraction):
    config = tiny_config()
    config.model.dimension_keep_fraction = keep_fraction
    with pytest.raises(ValueError, match="dimension_keep_fraction"):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rdm_projections", 0, "rdm_projections"),
        ("lp_norm_parameter", 0.0, "lp_norm_parameter"),
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
    assert config.loss.target_distribution == "rectified_lp_distribution"
    assert config.loss.lp_norm_parameter == 1.0
    assert config.loss.mean_shift_value == 0.0
    assert config.loss.mode_of_sigma == "sigma_GN"
    assert config.loss.projection_vectors_type == "random"
    assert config.loss.rdm_projections == 8192
    assert config.loss.invariance_weight == 25.0
    assert config.loss.lambda_rdm == 125.0
    assert config.train.batch_size == 512
    assert config.train.gradient_accumulation_steps == 1
    assert config.train.max_steps == 10000
    assert config.train.eval_batches == 12
    assert config.train.checkpoint_every == 10000
    assert config.train.output_dir.endswith("/proposed")
    assert config.train.resume_from is None
