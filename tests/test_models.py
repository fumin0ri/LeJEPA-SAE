import pytest
import torch

from lejepa_sae.config import DataConfig, ExperimentConfig, ModelConfig
from lejepa_sae.models import SharedWindowEncoder, build_model
from lejepa_sae.train import compute_loss


def tiny_config(model_type: str = "proposed") -> ExperimentConfig:
    config = ExperimentConfig(
        data=DataConfig(window_size=5, num_workers=0),
        model=ModelConfig(
            type=model_type,
            d_llm=8,
            d_encoder=8,
            num_layers=2,
            num_heads=2,
            mlp_ratio=2,
            feature_dim=16,
            num_local_views=2,
            local_tokens=2,
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
    ],
)
def test_all_models_complete_backward_step(model_type):
    torch.manual_seed(2)
    config = tiny_config(model_type)
    model = build_model(config)
    residuals = torch.randn(4, 5, 8)
    loss, metrics = compute_loss(model, residuals, config)
    loss.backward()
    assert torch.isfinite(loss)
    assert "loss" in metrics
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_positions_affect_encoding():
    config = tiny_config()
    model = build_model(config).eval()
    residuals = torch.randn(1, 2, 8)
    first = model(residuals, torch.tensor([[0, 1]])).features
    second = model(residuals, torch.tensor([[0, 4]])).features
    assert not torch.equal(first, second)
