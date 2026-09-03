from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .config import ExperimentConfig, load_config
from .data import (
    ActivationWindowDataset,
    ShardAwareRandomSampler,
    validate_document_disjointness,
)
from .diagnostics import thresholded_active_counts
from .losses import (
    generalized_gaussian_mean_shift_for_active_fraction,
    l1_sparsity_metric,
    random_axis_indices,
    rectified_lp_rdm_regularization,
    target_rate_regularization,
)
from .models import RDMSAE, BatchTopKSAE, JumpReLUSAE, SAEBase, build_model
from .views import sample_dimension_masks


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_overrides(config: ExperimentConfig, overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be section.key=value, got {override!r}")
        path, raw_value = override.split("=", 1)
        parts = path.split(".")
        if len(parts) != 2 or not hasattr(config, parts[0]):
            raise ValueError(f"Unknown override path: {path}")
        section = getattr(config, parts[0])
        if not hasattr(section, parts[1]):
            raise ValueError(f"Unknown override path: {path}")
        setattr(section, parts[1], yaml.safe_load(raw_value))
    config.validate()


def autocast_context(config: ExperimentConfig):
    if config.train.device.startswith("cuda") and config.train.precision != "float32":
        return torch.autocast("cuda", dtype=getattr(torch, config.train.precision))
    return nullcontext()


def stack_dimension_views(
    token_residuals: torch.Tensor,
    dimension_masks: list[torch.Tensor],
    *,
    include_global: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast residuals and stack masks so all dimension views use one Linear call."""
    if not dimension_masks:
        raise ValueError("At least one dimension mask is required")
    masks = torch.stack(dimension_masks)
    if include_global:
        global_mask = torch.ones_like(token_residuals, dtype=torch.bool).unsqueeze(0)
        masks = torch.cat((global_mask, masks), dim=0)
    residual_views = token_residuals.unsqueeze(0).expand(masks.shape[0], -1, -1)
    return residual_views, masks


def compute_rdm(
    feature_views: torch.Tensor,
    config: ExperimentConfig,
    axis_indices: torch.Tensor | None,
):
    return rectified_lp_rdm_regularization(
        feature_views,
        config.loss.rdm_projections,
        config.loss.axis_projections,
        config.loss.axis_weight,
        config.loss.lp_norm_parameter,
        (
            config.loss.mean_shift_value
            if config.loss.expected_l0_fraction is None
            else generalized_gaussian_mean_shift_for_active_fraction(
                config.loss.lp_norm_parameter, config.loss.expected_l0_fraction
            )
        ),
        axis_indices=axis_indices,
        target_scale=config.loss.rdm_target_scale,
        wasserstein_power=config.loss.rdm_wasserstein_power,
    )


def compute_loss(
    model: nn.Module,
    residuals: torch.Tensor,
    config: ExperimentConfig,
    include_diagnostics: bool = True,
    axis_indices: torch.Tensor | None = None,
    step: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model_type = config.model.type
    if residuals.shape[1] != 1:
        raise ValueError(f"{model_type} requires one-token residual windows")
    token_residuals = residuals[:, 0]

    if model_type in {"rdm_sae", "batch_topk_sae", "jump_relu_sae", "matryoshka_sae"}:
        output = model(token_residuals, pointwise=False if model.training else None)
        if model.training:
            model.update_dead_features_(output.features)
        reconstruction = F.mse_loss(output.reconstruction.float(), token_residuals.float())
        residual_float = token_residuals.float()
        reconstruction_variance = (
            residual_float - residual_float.mean(dim=0, keepdim=True)
        ).square().mean().clamp_min(1e-12)
        metrics = {
            "reconstruction": reconstruction.detach(),
            "fvu": (reconstruction / reconstruction_variance).detach(),
        }
        if model_type == "matryoshka_sae":
            prefix_losses = []
            for width, weight in zip(
                model.prefix_widths, model.prefix_weights, strict=True
            ):
                prefix_reconstruction = (
                    F.linear(output.features[:, :width], model.decoder.weight[:, :width])
                    + model.pre_bias
                )
                prefix_loss = F.mse_loss(
                    prefix_reconstruction.float(), token_residuals.float()
                )
                prefix_losses.append(prefix_loss * weight)
                metrics[f"prefix_{width}_mse"] = prefix_loss.detach()
                metrics[f"prefix_{width}_fvu"] = (
                    prefix_loss / reconstruction_variance
                ).detach()
            reconstruction_objective = torch.stack(prefix_losses).sum()
        else:
            reconstruction_objective = reconstruction

        auxk = (
            output.reconstruction.new_zeros((), dtype=torch.float32)
            if isinstance(model, JumpReLUSAE | RDMSAE)
            else model.auxiliary_loss(
                token_residuals, output.reconstruction, output.preactivations
            )
        )
        loss = config.loss.reconstruction_weight * reconstruction_objective
        if isinstance(model, RDMSAE):
            reconstruction_contribution = loss
            rdm_contribution = loss.new_zeros(())
            if config.loss.lambda_rdm > 0:
                rdm = compute_rdm(output.features.unsqueeze(0), config, axis_indices)
                rdm_contribution = config.loss.lambda_rdm * rdm.loss
                metrics.update({
                    "distribution": rdm.loss.detach(),
                    "random_distribution": rdm.random_loss.detach(),
                    "axis_distribution": rdm.axis_loss.detach(),
                })
            loss = reconstruction_contribution + rdm_contribution
            metrics.update({
                "reconstruction_contribution": reconstruction_contribution.detach(),
                "rdm_contribution": rdm_contribution.detach(),
                "rdm_target_scale": loss.new_tensor(config.loss.rdm_target_scale),
                "rdm_wasserstein_power": loss.new_tensor(config.loss.rdm_wasserstein_power),
            })
            if config.loss.expected_l0_fraction is not None:
                metrics["expected_l0_fraction"] = loss.new_tensor(config.loss.expected_l0_fraction)
            if (
                include_diagnostics
                and config.loss.rdm_gradient_diagnostics
                and torch.is_grad_enabled()
            ):
                reconstruction_grad = torch.autograd.grad(
                    reconstruction_contribution, output.preactivations, retain_graph=True
                )[0]
                reconstruction_rms = reconstruction_grad.detach().float().square().mean().sqrt()
                rdm_rms = loss.new_zeros(())
                if config.loss.lambda_rdm > 0:
                    rdm_grad = torch.autograd.grad(
                        rdm_contribution, output.preactivations, retain_graph=True
                    )[0]
                    rdm_rms = rdm_grad.detach().float().square().mean().sqrt()
                metrics.update({
                    "reconstruction_preactivation_grad_rms": reconstruction_rms,
                    "rdm_preactivation_grad_rms": rdm_rms,
                    "rdm_to_reconstruction_grad_ratio": (
                        rdm_rms / reconstruction_rms.clamp_min(1e-12)
                    ),
                })
                for name in ("random", "axis"):
                    component_rms = loss.new_zeros(())
                    if config.loss.lambda_rdm > 0:
                        component = rdm.random_loss if name == "random" else rdm.axis_loss
                        weight = config.loss.lambda_rdm
                        if name == "axis":
                            weight *= config.loss.axis_weight
                        component_grad = torch.autograd.grad(
                            weight * component, output.preactivations, retain_graph=True
                        )[0]
                        component_rms = component_grad.detach().float().square().mean().sqrt()
                        del component_grad
                    metrics[f"rdm_{name}_preactivation_grad_rms"] = component_rms
            if not torch.isfinite(loss):
                raise FloatingPointError("rdm_sae produced a non-finite loss")
        else:
            loss = loss + config.baseline.auxk_coefficient * auxk
            metrics["auxk"] = auxk.detach()
        if isinstance(model, JumpReLUSAE):
            l0_surrogate = model.l0_surrogate(output.preactivations).sum(dim=-1).mean()
            warmup = min(
                1.0,
                step / max(1, config.baseline.jump_relu_sparsity_warmup_steps),
            )
            sparsity_coefficient = config.baseline.jump_relu_lambda * warmup
            loss = loss + sparsity_coefficient * l0_surrogate
            metrics["l0_penalty"] = l0_surrogate.detach()
            metrics["sparsity_coefficient"] = torch.tensor(
                sparsity_coefficient, device=loss.device
            )
            metrics["mean_threshold"] = model.threshold.detach().float().mean()
        metrics["loss"] = loss.detach()
        if include_diagnostics:
            detached = output.features.detach()
            active_counts = detached.gt(0).sum(dim=-1).float()
            metrics.update(
                {
                    "l0": active_counts.mean(),
                    "active_fraction": detached.gt(0).float().mean(),
                    "dead_feature_fraction": detached.amax(dim=0).le(0).float().mean(),
                    "tracker_dead_feature_fraction": model.dead_mask().float().mean(),
                    "feature_std": detached.float().std(dim=0, unbiased=False).mean(),
                }
            )
            if isinstance(model, RDMSAE):
                metrics.update({
                    key: count.float() / detached.numel()
                    for key, count in thresholded_active_counts(detached).items()
                })
                metrics["l0_sparsity"] = metrics["active_fraction"]
                metrics["l1_sparsity"] = l1_sparsity_metric(detached)
        return loss, metrics

    has_local_views = config.model.num_local_views > 0
    if has_local_views:
        dimension_masks = sample_dimension_masks(
            token_residuals,
            config.model.num_local_views,
            config.model.dimension_keep_fraction,
        )
        residual_views, masks = stack_dimension_views(
            token_residuals, dimension_masks, include_global=True
        )
        output = model(residual_views, masks)
        feature_views = output.features
    else:
        # A single unmasked forward, not duplicate full-mask local views.
        output = model(token_residuals)
        feature_views = output.features.unsqueeze(0)
    preactivations = output.preactivations if config.loss.rate_weight > 0 else None
    del output
    global_features = feature_views[0]
    local_features = feature_views[1:]
    invariance = None
    if has_local_views:
        invariance = F.mse_loss(
            local_features,
            global_features.unsqueeze(0).expand_as(local_features),
        )
    rdm = compute_rdm(feature_views, config, axis_indices)
    base_loss = config.loss.lambda_rdm * rdm.loss
    if invariance is not None:
        base_loss = config.loss.invariance_weight * invariance + base_loss
    loss = base_loss
    rate = None
    if config.loss.rate_weight > 0:
        rate = target_rate_regularization(
            preactivations,
            config.loss.expected_l0_fraction,
            config.loss.rate_temperature,
            config.loss.rate_scale_floor,
        )
        loss = loss + config.loss.rate_weight * rate.loss
    if not torch.isfinite(loss):
        raise FloatingPointError("proposed model produced a non-finite loss")
    view_distribution_losses = (
        rdm.random_view_losses + config.loss.axis_weight * rdm.axis_view_losses
    )
    global_distribution = view_distribution_losses[0]
    metrics = {
        "loss": loss.detach(),
        "distribution": rdm.loss.detach(),
        "random_distribution": rdm.random_loss.detach(),
        "axis_distribution": rdm.axis_loss.detach(),
        "global_distribution": global_distribution.detach(),
        "global_rdm_contribution": (
            (0.5 if has_local_views else 1.0) * global_distribution
        ).detach(),
        "global_random_distribution": rdm.random_view_losses[0].detach(),
        "global_axis_distribution": rdm.axis_view_losses[0].detach(),
    }
    if has_local_views:
        local_distribution = view_distribution_losses[1:].mean()
        metrics.update({
            "invariance": invariance.detach(),
            "local_distribution": local_distribution.detach(),
            "local_rdm_contribution": (0.5 * local_distribution).detach(),
            "local_random_distribution": rdm.random_view_losses[1:].mean().detach(),
            "local_axis_distribution": rdm.axis_view_losses[1:].mean().detach(),
        })
    if config.loss.expected_l0_fraction is not None:
        metrics["expected_l0_fraction"] = torch.tensor(
            config.loss.expected_l0_fraction,
            device=feature_views.device,
        )
    if rate is not None:
        metrics.update({
            "base_loss": base_loss.detach(),
            "rate_loss": rate.loss.detach(),
            "global_rate_loss": rate.view_losses[0].detach(),
            "local_rate_loss": rate.view_losses[1:].mean().detach(),
            "rate_contribution": (config.loss.rate_weight * rate.loss).detach(),
            "rate_global_active_fraction": rate.rates[0].detach(),
            "rate_local_active_fraction": rate.rates[1:].mean().detach(),
            "rate_scale": rate.scale,
        })
        if (
            include_diagnostics
            and config.loss.rate_gradient_diagnostics
            and torch.is_grad_enabled()
        ):
            base_grad = torch.autograd.grad(base_loss, preactivations, retain_graph=True)[0]
            rate_grad = torch.autograd.grad(
                config.loss.rate_weight * rate.loss, preactivations, retain_graph=True
            )[0]
            base_rms = base_grad.detach().float().square().mean().sqrt()
            rate_rms = rate_grad.detach().float().square().mean().sqrt()
            metrics.update({
                "base_preactivation_grad_rms": base_rms,
                "rate_preactivation_grad_rms": rate_rms,
                "rate_to_base_grad_ratio": rate_rms / base_rms.clamp_min(1e-12),
            })
    if not include_diagnostics:
        return loss, metrics

    flattened = feature_views.flatten(0, 1).detach()
    global_detached = global_features.detach()
    global_float = global_detached.float()
    metrics.update(
        {
            "active_fraction": (flattened > 0).float().mean().detach(),
            "l0_sparsity": (flattened > 0).float().mean().detach(),
            "l1_sparsity": l1_sparsity_metric(flattened).detach(),
            "feature_std": flattened.float().std(dim=0, unbiased=False).mean(),
            "global_active_fraction": (global_detached > 0).float().mean(),
            "global_feature_std": global_float.std(dim=0, unbiased=False).mean(),
            "global_dead_feature_fraction": (
                global_detached.amax(dim=0) <= 0
            ).float().mean(),
        }
    )
    if has_local_views:
        flattened_locals = local_features.flatten(0, 1).detach()
        metrics.update(activation_transition_metrics(global_features, local_features))
        metrics.update({
            "local_feature_std": flattened_locals.float().std(dim=0, unbiased=False).mean(),
            "local_dead_feature_fraction": (
                flattened_locals.amax(dim=0) <= 0
            ).float().mean(),
        })
    return loss, metrics


@torch.no_grad()
def activation_transition_metrics(
    global_features: torch.Tensor,
    local_features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Unconditional gate probabilities over [local view, sample, coordinate].

    For both proposed activations, forward is exact ReLU, so z > 0 iff a > 0.
    Use strict zero, not an evaluation support epsilon. No extra forward is needed.
    """
    if (
        global_features.ndim != 2
        or local_features.ndim != 3
        or local_features.shape[1:] != global_features.shape
        or local_features.numel() == 0
    ):
        raise ValueError("Expected global [B, D] and nonempty local [V, B, D] features")
    global_on = global_features > 0
    local_on = local_features > 0
    off_to_on = ((~global_on).unsqueeze(0) & local_on).float().mean()
    on_to_off = (global_on.unsqueeze(0) & ~local_on).float().mean()
    global_active = global_on.float().mean()
    local_active = local_on.float().mean()
    return {
        "off_to_on": off_to_on,
        "on_to_off": on_to_off,
        "global_active_fraction": global_active,
        "local_active_fraction": local_active,
        "local_global_active_fraction_gap": local_active - global_active,
        "transition_rate_gap": off_to_on - on_to_off,
        "support_disagreement": off_to_on + on_to_off,
    }


def make_loader(
    config: ExperimentConfig,
    split: str,
    shuffle: bool,
) -> DataLoader:
    stride = config.data.train_stride if split == "train" else config.data.eval_stride
    dataset = ActivationWindowDataset(
        config.data.activation_dir,
        split,
        config.data.window_size,
        stride,
        config.data.cache_shards_per_worker,
        include_metadata=False,
    )
    if not len(dataset):
        raise ValueError(f"No {split} windows found in {config.data.activation_dir}")
    sampler = ShardAwareRandomSampler(dataset, config.train.seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.train.device.startswith("cuda"),
        persistent_workers=config.data.num_workers > 0,
        drop_last=shuffle,
    )


def aggregate_metrics(metrics: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    keys = set().union(*(item.keys() for item in metrics))
    return {
        key: float(
            torch.stack([item[key].float().cpu() for item in metrics if key in item]).mean()
        )
        for key in sorted(keys)
    }


@torch.no_grad()
def calibrate_batch_topk_threshold(
    model: BatchTopKSAE,
    loader: DataLoader,
    config: ExperimentConfig,
) -> dict[str, float]:
    """Fit the pointwise threshold to validation preactivations at the training target L0."""
    was_training = model.training
    model.eval()
    samples = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= config.baseline.threshold_calibration_batches:
            break
        residuals = batch["residuals"][:, 0].to(config.train.device, non_blocking=True)
        with autocast_context(config):
            samples.append(model.preactivations(residuals).relu().float().cpu())
    if not samples:
        raise ValueError("No validation activations were available for threshold calibration")
    activations = torch.cat(samples)
    desired = min(activations.numel(), activations.shape[0] * config.target_l0)
    kth_value = activations.flatten().topk(desired).values[-1]
    threshold = torch.nextafter(kth_value, torch.tensor(-torch.inf))
    model.calibrated_threshold.copy_(threshold.to(model.calibrated_threshold.device))
    measured_l0 = float((activations > threshold).float().sum(dim=-1).mean())
    if was_training:
        model.train()
    return {
        "threshold": float(threshold),
        "calibrated_l0": measured_l0,
        "calibration_samples": float(activations.shape[0]),
    }


def resolve_training_steps(
    requested_max_steps: int | str,
    train_batches: int,
    gradient_accumulation_steps: int,
) -> int:
    if requested_max_steps != "one_epoch":
        return int(requested_max_steps)
    optimizer_steps = train_batches // gradient_accumulation_steps
    if optimizer_steps < 1:
        raise ValueError(
            "One epoch does not contain enough batches for one optimizer step; reduce "
            "train.batch_size or train.gradient_accumulation_steps"
        )
    return optimizer_steps


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
) -> dict[str, float]:
    calibration = None
    if isinstance(model, BatchTopKSAE):
        calibration = calibrate_batch_topk_threshold(model, loader, config)
    model.eval()
    collected = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= config.train.eval_batches:
            break
        residuals = batch["residuals"].to(config.train.device, non_blocking=True)
        with autocast_context(config):
            _, metrics = compute_loss(model, residuals, config)
        collected.append(metrics)
    model.train()
    result = aggregate_metrics(collected)
    if calibration is not None:
        result.update(calibration)
    return result


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    config: ExperimentConfig,
) -> Path:
    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config.to_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if isinstance(model, BatchTopKSAE) and torch.isfinite(model.calibrated_threshold):
        checkpoint["threshold_calibration"] = {
            "threshold": float(model.calibrated_threshold),
            "target_l0": config.target_l0,
        }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    final_path = output_dir / f"checkpoint-{step:08d}.pt"
    temporary_path = output_dir / f".checkpoint-{step:08d}.tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, final_path)
    latest_path = output_dir / "latest.json"
    latest = json.dumps({"checkpoint": final_path.name, "step": step})
    latest_path.write_text(latest, encoding="utf-8")
    return final_path


def train(config: ExperimentConfig) -> Path:
    seed_everything(config.train.seed)
    if config.train.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    validate_document_disjointness(config.data.activation_dir)
    train_loader = make_loader(config, "train", shuffle=True)
    validation_loader = make_loader(config, "validation", shuffle=False)
    if train_loader.dataset.d_llm != config.model.d_llm:
        raise ValueError(
            f"Config d_llm={config.model.d_llm} does not match activation d_llm="
            f"{train_loader.dataset.d_llm}"
        )

    requested_max_steps = config.train.max_steps
    config.train.max_steps = resolve_training_steps(
        requested_max_steps,
        len(train_loader),
        config.train.gradient_accumulation_steps,
    )

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    training_plan = {
        "requested_max_steps": requested_max_steps,
        "resolved_max_steps": config.train.max_steps,
        "train_samples": len(train_loader.dataset),
        "train_batches": len(train_loader),
        "batch_size": config.train.batch_size,
        "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
        "consumed_samples": (
            config.train.max_steps
            * config.train.batch_size
            * config.train.gradient_accumulation_steps
        ),
    }
    training_plan["sample_delta_from_one_epoch"] = (
        training_plan["train_samples"] - training_plan["consumed_samples"]
    )
    (output_dir / "training_plan.json").write_text(
        json.dumps(training_plan, indent=2), encoding="utf-8"
    )
    print(json.dumps({"kind": "training_plan", **training_plan}), flush=True)
    metrics_path = output_dir / "metrics.jsonl"

    model = build_model(config).to(config.train.device)
    optimizer_kwargs: dict[str, Any] = {
        "lr": config.train.learning_rate,
        "weight_decay": config.train.weight_decay,
    }
    if config.train.device.startswith("cuda"):
        optimizer_kwargs["fused"] = True
    optimizer = AdamW(model.parameters(), **optimizer_kwargs)

    def lr_multiplier(step: int) -> float:
        if step < config.train.warmup_steps:
            return float(step + 1) / max(1, config.train.warmup_steps)
        progress = (step - config.train.warmup_steps) / max(
            1, config.train.max_steps - config.train.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = LambdaLR(optimizer, lr_multiplier)
    start_step = 0
    if config.train.resume_from:
        checkpoint = torch.load(config.train.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    model.train()
    optimizer.zero_grad(set_to_none=True)
    data_iterator = iter(train_loader)
    interval_metrics: list[dict[str, torch.Tensor]] = []
    started = time.monotonic()
    last_log_time = started
    excluded_interval_seconds = 0.0
    interval_samples = 0
    interval_steps = 0
    cuda_device = (
        torch.device(config.train.device) if config.train.device.startswith("cuda") else None
    )
    deferred_peak_allocated = 0
    deferred_peak_reserved = 0
    if cuda_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_device)
    last_checkpoint: Path | None = None

    for step in range(start_step + 1, config.train.max_steps + 1):
        step_axis_indices = None
        if config.model.type == "proposed" or (
            config.model.type == "rdm_sae" and config.loss.lambda_rdm > 0
        ):
            step_axis_indices = random_axis_indices(
                config.loss.axis_projections,
                config.model.feature_dim,
                device=torch.device(config.train.device),
            )
        for accumulation_index in range(config.train.gradient_accumulation_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                batch = next(data_iterator)
            residuals = batch["residuals"].to(config.train.device, non_blocking=True)
            include_diagnostics = (
                step % config.train.log_every == 0
                and accumulation_index == config.train.gradient_accumulation_steps - 1
            )
            with autocast_context(config):
                loss, metrics = compute_loss(
                    model,
                    residuals,
                    config,
                    include_diagnostics=include_diagnostics,
                    axis_indices=step_axis_indices,
                    step=step,
                )
                scaled_loss = loss / config.train.gradient_accumulation_steps
            scaled_loss.backward()
            interval_metrics.append(metrics)
            interval_samples += residuals.shape[0]

        if isinstance(model, SAEBase):
            model.project_decoder_gradients_()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        if isinstance(model, SAEBase):
            model.normalize_decoder_()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        interval_steps += 1

        if step % config.train.log_every == 0:
            aggregated = aggregate_metrics(interval_metrics)
            logged_at = time.monotonic()
            interval_seconds = max(
                logged_at - last_log_time - excluded_interval_seconds,
                1e-12,
            )
            record = {
                "kind": "train",
                "step": step,
                **aggregated,
                "grad_norm": float(grad_norm),
                "learning_rate": scheduler.get_last_lr()[0],
                "samples_per_second": interval_samples / interval_seconds,
                "optimizer_steps_per_second": interval_steps / interval_seconds,
                "elapsed_seconds": logged_at - started,
            }
            if cuda_device is not None:
                peak_allocated = max(
                    deferred_peak_allocated,
                    torch.cuda.max_memory_allocated(cuda_device),
                )
                peak_reserved = max(
                    deferred_peak_reserved,
                    torch.cuda.max_memory_reserved(cuda_device),
                )
                record["cuda_peak_allocated_mib"] = peak_allocated / 2**20
                record["cuda_peak_reserved_mib"] = peak_reserved / 2**20
                torch.cuda.reset_peak_memory_stats(cuda_device)
                deferred_peak_allocated = 0
                deferred_peak_reserved = 0
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            interval_metrics.clear()
            interval_samples = 0
            interval_steps = 0
            last_log_time = logged_at
            excluded_interval_seconds = 0.0

        if step % config.train.eval_every == 0:
            if cuda_device is not None:
                deferred_peak_allocated = max(
                    deferred_peak_allocated,
                    torch.cuda.max_memory_allocated(cuda_device),
                )
                deferred_peak_reserved = max(
                    deferred_peak_reserved,
                    torch.cuda.max_memory_reserved(cuda_device),
                )
                torch.cuda.reset_peak_memory_stats(cuda_device)
            evaluation_started = time.monotonic()
            record = {
                "kind": "validation",
                "step": step,
                **evaluate_loss(model, validation_loader, config),
            }
            excluded_interval_seconds += time.monotonic() - evaluation_started
            if cuda_device is not None:
                torch.cuda.reset_peak_memory_stats(cuda_device)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

        if step % config.train.checkpoint_every == 0:
            checkpoint_started = time.monotonic()
            last_checkpoint = save_checkpoint(output_dir, step, model, optimizer, scheduler, config)
            excluded_interval_seconds += time.monotonic() - checkpoint_started

    if isinstance(model, BatchTopKSAE):
        calibration = calibrate_batch_topk_threshold(model, validation_loader, config)
        (output_dir / "threshold_calibration.json").write_text(
            json.dumps(calibration, indent=2), encoding="utf-8"
        )
    # Always rewrite the final checkpoint after calibration so pointwise inference is portable.
    last_checkpoint = save_checkpoint(
        output_dir, config.train.max_steps, model, optimizer, scheduler, config
    )
    return last_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LeJEPA-SAE and its required baselines")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a scalar as section.key=value (repeatable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args.overrides)
    checkpoint = train(config)
    print(f"Final checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
