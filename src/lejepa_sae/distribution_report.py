"""Compare how well trained rdm_sae checkpoints match the rectified target distribution.

Every metric is computed on held-out test tokens with fixed seeds, and the same statistic is
computed on i.i.d. target samples (the "reference" row), which is the finite-sample noise
floor: a run is "well shaped" when it approaches the reference, not zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, load_config
from .data import ActivationWindowDataset
from .evaluate import load_model
from .losses import (
    generalized_gaussian_mean_shift_for_active_fraction,
    random_axis_indices,
    random_unit_projections,
    rectified_target_variance,
    sample_orthonormal_sketch,
    sample_rectified_generalized_gaussian_like,
    sketched_covariance_loss,
    sliced_wasserstein_on_axes,
    sliced_wasserstein_with_projections,
)
from .train import autocast_context


def effective_rank(eigenvalues: torch.Tensor) -> float:
    values = eigenvalues.clamp_min(0).double()
    probabilities = values / values.sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


@torch.no_grad()
def distribution_metrics(
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    target_variance: float,
    projections: int,
    axes: int,
    sketch_dim: int,
    cov_features: int,
    seed: int,
) -> dict[str, float]:
    """features/target: [N, D] on the same device; target is an i.i.d. target draw."""
    features = features.float()
    target = target.float()
    n, d = features.shape
    device = features.device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = random_unit_projections(
        projections, d, device=device, dtype=torch.float32, generator=generator
    )
    axis_indices = random_axis_indices(axes, d, device=device, generator=generator)
    sketch = sample_orthonormal_sketch(d, sketch_dim, device=device, generator=generator)
    subset = torch.randperm(d, generator=generator)[:cov_features].to(device)

    mean = features.mean(0)
    variance = features.var(0, unbiased=False)
    target_mean = target.mean(0)
    sub = features[:, subset]
    centered = sub - sub.mean(0, keepdim=True)
    covariance = centered.T @ centered / (n - 1)
    std = covariance.diagonal().clamp_min(1e-12).sqrt()
    correlation = covariance / (std[:, None] * std[None, :])
    off_diagonal = ~torch.eye(len(subset), dtype=torch.bool, device=device)
    eigenvalues = torch.linalg.eigvalsh(covariance.double())
    normalized = covariance / target_variance
    identity = torch.eye(len(subset), device=device)
    quantiles = torch.tensor([0.5, 0.9, 0.99], device=device)
    feature_q = torch.quantile(features[:, axis_indices].T.contiguous(), quantiles, dim=1)
    target_q = torch.quantile(target[:, axis_indices].T.contiguous(), quantiles, dim=1)
    return {
        "sw_random": float(
            sliced_wasserstein_with_projections(features[None], target[None], projection)[0]
        ),
        "axis_w2sq": float(
            sliced_wasserstein_on_axes(features[None], target[None], axis_indices)[0]
        ),
        "axis_w1": float(
            sliced_wasserstein_on_axes(
                features[None], target[None], axis_indices, wasserstein_power=1
            )[0]
        ),
        "active_fraction": float(features.gt(0).float().mean()),
        "dead_feature_fraction": float(features.amax(0).le(0).float().mean()),
        "mean_abs_error": float((mean - target_mean).abs().mean()),
        "variance_ratio_median": float((variance / target_variance).median()),
        "variance_ratio_p10": float((variance / target_variance).quantile(0.1)),
        "variance_ratio_p90": float((variance / target_variance).quantile(0.9)),
        "quantile_abs_error_p50": float((feature_q[0] - target_q[0]).abs().mean()),
        "quantile_abs_error_p90": float((feature_q[1] - target_q[1]).abs().mean()),
        "quantile_abs_error_p99": float((feature_q[2] - target_q[2]).abs().mean()),
        "sketched_cov_loss": float(
            sketched_covariance_loss(features, sketch, target_variance)
        ),
        "cov_fro_dev": float(
            (normalized - identity).square().mean().sqrt()
        ),
        "mean_abs_offdiag_corr": float(correlation[off_diagonal].abs().mean()),
        "cov_effective_rank_frac": effective_rank(eigenvalues) / len(subset),
        "cov_top_eig_over_mean": float(eigenvalues.max() / eigenvalues.mean().clamp_min(1e-30)),
    }


@torch.no_grad()
def collect_features(config: ExperimentConfig, checkpoint: str, tokens: int):
    device = config.train.device
    dataset = ActivationWindowDataset(
        config.data.activation_dir, "test", config.data.window_size,
        config.data.eval_stride, config.data.cache_shards_per_worker,
    )
    loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=0)
    model = load_model(config, checkpoint, device)
    features, residuals_seen = [], 0
    mse_sum, var_sum, count = 0.0, 0.0, 0
    for batch in loader:
        if residuals_seen >= tokens:
            break
        residuals = batch["residuals"][: tokens - residuals_seen].to(device)[:, 0]
        with autocast_context(config):
            output = model(residuals)
        features.append(output.features.float())
        h = residuals.float()
        mse_sum += float((output.reconstruction.float() - h).square().mean()) * len(h)
        var_sum += float((h - h.mean(0, keepdim=True)).square().mean()) * len(h)
        count += len(h)
        residuals_seen += len(h)
    return torch.cat(features), mse_sum / count / max(var_sum / count, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="NAME=RUN_DIR (needs config.resolved.yaml and the checkpoint)")
    parser.add_argument("--checkpoint-name", default="checkpoint-00010000.pt")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--projections", type=int, default=8192)
    parser.add_argument("--axes", type=int, default=512)
    parser.add_argument("--sketch-dim", type=int, default=64)
    parser.add_argument("--cov-features", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="distribution_report.json")
    args = parser.parse_args()

    results: dict[str, dict[str, float]] = {}
    for spec in args.run:
        name, _, run_dir = spec.partition("=")
        config = load_config(Path(run_dir) / "config.resolved.yaml")
        features, fvu = collect_features(
            config, str(Path(run_dir) / args.checkpoint_name), args.tokens
        )
        mu = (
            generalized_gaussian_mean_shift_for_active_fraction(
                config.loss.lp_norm_parameter, config.loss.expected_l0_fraction
            )
            if config.loss.expected_l0_fraction is not None
            else config.loss.mean_shift_value
        )
        variance = rectified_target_variance(
            config.loss.lp_norm_parameter, mu, config.loss.rdm_target_scale
        )
        target_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        target = sample_rectified_generalized_gaussian_like(
            torch.empty_like(features, device="cpu"), config.loss.lp_norm_parameter, mu,
            generator=target_generator,
        ).to(features.device) * config.loss.rdm_target_scale
        kwargs = dict(
            target_variance=variance, projections=args.projections, axes=args.axes,
            sketch_dim=args.sketch_dim, cov_features=args.cov_features, seed=args.seed,
        )
        results[name] = {"fvu": fvu, **distribution_metrics(features, target, **kwargs)}
        if "reference" not in results:
            # Noise floor: a second independent target draw against the first.
            other = sample_rectified_generalized_gaussian_like(
                torch.empty_like(features, device="cpu"), config.loss.lp_norm_parameter, mu,
                generator=torch.Generator(device="cpu").manual_seed(args.seed + 2),
            ).to(features.device) * config.loss.rdm_target_scale
            results["reference"] = {"fvu": float("nan"), **distribution_metrics(other, target, **kwargs)}
        print(f"done {name}")

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    names = list(results)
    metrics = list(next(iter(results.values())))
    print("| metric | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for metric in metrics:
        print(f"| {metric} | " + " | ".join(f"{results[n][metric]:.4g}" for n in names) + " |")


if __name__ == "__main__":
    main()
