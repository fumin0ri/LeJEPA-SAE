import torch

from lejepa_sae.distribution_report import distribution_metrics
from lejepa_sae.losses import (
    generalized_gaussian_mean_shift_for_active_fraction,
    rectified_target_variance,
    sample_rectified_generalized_gaussian_like,
)


def test_target_like_features_are_near_noise_floor_and_collapsed_ones_are_not():
    mu = generalized_gaussian_mean_shift_for_active_fraction(1.0, 0.1)
    variance = rectified_target_variance(1.0, mu, 1.0)
    draw = lambda seed: sample_rectified_generalized_gaussian_like(  # noqa: E731
        torch.empty(2048, 256), 1.0, mu, generator=torch.Generator().manual_seed(seed)
    )
    kwargs = dict(target_variance=variance, projections=64, axes=32, sketch_dim=8,
                  cov_features=64, seed=0)
    good = distribution_metrics(draw(1), draw(2), **kwargs)
    shared = draw(3)[:, :1].expand(-1, 256).contiguous()  # perfectly correlated features
    bad = distribution_metrics(shared, draw(2), **kwargs)
    assert abs(good["active_fraction"] - 0.1) < 0.01
    assert good["mean_abs_offdiag_corr"] < 0.1 < bad["mean_abs_offdiag_corr"]
    assert good["cov_effective_rank_frac"] > 3 * bad["cov_effective_rank_frac"]
    assert good["sketched_cov_loss"] < bad["sketched_cov_loss"]
