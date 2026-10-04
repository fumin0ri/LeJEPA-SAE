import pytest
import torch

from lejepa_sae.config import ExperimentConfig, ModelConfig
from lejepa_sae.losses import (
    generalized_gaussian_mean_shift_for_active_fraction,
    rectified_target_variance,
    sample_orthonormal_sketch,
    sketched_covariance_loss,
)
from lejepa_sae.models import build_model
from lejepa_sae.train import compute_loss


def test_sketch_is_orthonormal():
    q = sample_orthonormal_sketch(32, 8, device=torch.device("cpu"))
    torch.testing.assert_close(q.T @ q, torch.eye(8), atol=1e-5, rtol=0)


def test_cov_loss_is_small_for_matching_covariance_and_large_otherwise():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(4096, 64, generator=generator) * 0.5
    q = sample_orthonormal_sketch(64, 8, device=torch.device("cpu"), generator=generator)
    assert sketched_covariance_loss(x, q, 0.25) < 0.01
    assert sketched_covariance_loss(x, q, 1.0) > 0.05
    correlated = x[:, :1].expand(-1, 64) + 0.01 * x
    assert sketched_covariance_loss(correlated, q, 0.25) > sketched_covariance_loss(x, q, 0.25)


def test_cov_loss_is_scale_invariant_with_matching_variance():
    x = torch.randn(512, 32)
    q = sample_orthonormal_sketch(32, 4, device=torch.device("cpu"))
    torch.testing.assert_close(
        sketched_covariance_loss(x, q, 1.0),
        sketched_covariance_loss(3 * x, q, 9.0),
        rtol=1e-5, atol=1e-7,
    )


def test_cov_loss_rejects_bad_shapes():
    q = sample_orthonormal_sketch(8, 4, device=torch.device("cpu"))
    with pytest.raises(ValueError):
        sketched_covariance_loss(torch.randn(4, 8), q, 1.0)
    with pytest.raises(ValueError):
        sketched_covariance_loss(torch.randn(16, 8), q, 0.0)


def test_laplace_target_variance_matches_closed_form():
    # ReLU(mu + b*L), L ~ Laplace(1), b = 1/sqrt(2); rho = 0.05 -> mu = b*log(2*rho) < 0.
    rho = 0.05
    mu = generalized_gaussian_mean_shift_for_active_fraction(1.0, rho)
    b = 2 ** -0.5
    # Given X > 0 the excess over mu is exponential with scale b: E[X]=rho*(mu+b)... derive via MC
    samples = torch.distributions.Laplace(0.0, 1.0).sample((2_000_000,))
    values = (mu + b * samples).relu()
    assert rectified_target_variance(1.0, mu, 1.0) == pytest.approx(
        float(values.var(unbiased=False)), rel=0.03
    )
    assert rectified_target_variance(1.0, mu, 2.0) == pytest.approx(
        4 * rectified_target_variance(1.0, mu, 1.0), rel=1e-9
    )


def _config(cov_weight):
    config = ExperimentConfig(
        model=ModelConfig(type="rdm_sae", d_llm=8, feature_dim=16, num_local_views=0)
    )
    config.loss.invariance_weight = 0
    config.loss.expected_l0_fraction = 0.25
    config.loss.rdm_projections = 4
    config.loss.axis_projections = 4
    config.loss.rdm_random_weight = 1.0
    config.loss.rdm_axis_weight = 0.0
    config.loss.rdm_cov_weight = cov_weight
    config.loss.rdm_cov_sketch_dim = 4
    config.train.device = "cpu"
    config.train.precision = "float32"
    config.train.batch_size = 32
    config.validate()
    return config


def test_cov_term_adds_to_loss_and_zero_weight_draws_no_rng():
    residuals = torch.randn(32, 1, 8)
    off = _config(0.0)
    torch.manual_seed(1)
    model = build_model(off)
    state = torch.get_rng_state().clone()
    loss_off, metrics_off = compute_loss(model, residuals, off, include_diagnostics=False)
    assert "rdm_covariance" not in metrics_off
    del state
    on = _config(2.0)
    torch.manual_seed(2)
    loss_on, metrics_on = compute_loss(model, residuals, on, include_diagnostics=False)
    assert metrics_on["rdm_covariance"] > 0
    torch.testing.assert_close(
        metrics_on["rdm_cov_contribution"], 2.0 * metrics_on["rdm_covariance"]
    )
    loss_on.backward()
    assert model.encoder.weight.grad is not None and torch.isfinite(model.encoder.weight.grad).all()


def test_config_rejects_oversized_sketch_and_wrong_model():
    config = _config(1.0)
    config.loss.rdm_cov_sketch_dim = 32
    with pytest.raises(ValueError):
        config.validate()
    config = _config(1.0)
    config.model.type = "proposed"
    with pytest.raises(ValueError):
        config.validate()
