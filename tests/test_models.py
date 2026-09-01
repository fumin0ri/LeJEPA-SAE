import pytest
import torch
import torch.nn.functional as F

from lejepa_sae.config import DataConfig, ExperimentConfig, ModelConfig, load_config
from lejepa_sae.models import SharedWindowEncoder, build_model
from lejepa_sae.train import compute_loss


def tiny_config(model_type: str = "proposed") -> ExperimentConfig:
    window_size = 1 if model_type in {"single_token_jepa", "dimension_denoising_sae"} else 5
    config = ExperimentConfig(
        data=DataConfig(window_size=window_size, num_workers=0),
        model=ModelConfig(
            type=model_type,
            d_llm=8,
            d_encoder=8,
            num_layers=2,
            num_heads=2,
            mlp_ratio=2,
            feature_dim=16,
            num_local_views=2,
            local_tokens=min(2, window_size),
        ),
    )
    config.loss.rdm_projections = 4
    config.validate()
    return config


def test_causal_cls_attention_mask():
    mask = SharedWindowEncoder.attention_mask(4, "causal")
    assert mask.shape == (5, 5)
    assert not mask[0].any()  # CLS reads every residual token.
    assert mask[1:, 0].all()  # Residual tokens cannot read CLS.
    assert mask[1, 2:].all()
    assert not mask[4, 1:].any()


def test_bidirectional_mask_still_blocks_token_to_cls():
    mask = SharedWindowEncoder.attention_mask(4, "bidirectional")
    assert not mask[0].any()
    assert mask[1:, 0].all()
    assert not mask[1:, 1:].any()


@pytest.mark.parametrize(
    "model_type",
    [
        "proposed",
        "sparse_jepa_full_view",
        "jepa_sigreg",
        "standard_sae",
        "window_autoencoder",
        "single_token_jepa",
        "dimension_denoising_sae",
    ],
)
def test_all_models_complete_backward_step(model_type):
    torch.manual_seed(2)
    config = tiny_config(model_type)
    model = build_model(config)
    residuals = torch.randn(4, config.data.window_size, 8)
    loss, metrics = compute_loss(model, residuals, config)
    loss.backward()
    assert torch.isfinite(loss)
    assert "loss" in metrics
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_single_token_jepa_reports_collapse_diagnostics():
    config = tiny_config("single_token_jepa")
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


def test_positions_affect_encoding():
    config = tiny_config()
    model = build_model(config).eval()
    residuals = torch.randn(1, 2, 8)
    first = model(residuals, torch.tensor([[0, 1]])).features
    second = model(residuals, torch.tensor([[0, 4]])).features
    assert not torch.equal(first, second)


def test_masked_encoder_fills_missing_coordinates_with_pre_bias():
    config = tiny_config("single_token_jepa")
    model = build_model(config)
    with torch.no_grad():
        model.pre_bias.copy_(torch.arange(8).float())
    residuals = torch.arange(8).float().unsqueeze(0) + 2.0
    mask = torch.tensor([[True, False, True, False, True, False, True, False]])
    prepared = model.prepare_input(residuals, mask)
    assert torch.equal(prepared[~mask], torch.zeros(4))
    torch.testing.assert_close(prepared[mask], torch.full((4,), 4.0))


def test_dense_mask_projection_matches_coordinate_indexed_projection():
    torch.manual_seed(5)
    config = tiny_config("single_token_jepa")
    model = build_model(config)
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


def test_dimension_view_config_requires_one_token():
    config = tiny_config("single_token_jepa")
    config.data.window_size = 2
    with pytest.raises(ValueError, match="window_size=1"):
        config.validate()


@pytest.mark.parametrize("keep_fraction", [0.0, 1.1])
def test_dimension_keep_fraction_is_validated(keep_fraction):
    config = tiny_config("single_token_jepa")
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
def test_single_token_paper_rdm_config_is_validated(field, value, message):
    config = tiny_config("single_token_jepa")
    setattr(config.loss, field, value)
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_single_token_preset_uses_paper_stabilization_defaults():
    config = load_config("configs/pythia-6.9b-layer16-single-token.yaml")
    assert config.model.type == "single_token_jepa"
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
    assert config.train.batch_size == 128
    assert config.train.gradient_accumulation_steps == 4
    assert config.train.resume_from is None
