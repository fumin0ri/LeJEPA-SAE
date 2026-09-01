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
from .losses import (
    generalized_gaussian_mean_shift_for_active_fraction,
    l1_sparsity_metric,
    random_axis_indices,
    rectified_lp_rdm_regularization,
)
from .models import build_model
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


def compute_loss(
    model: nn.Module,
    residuals: torch.Tensor,
    config: ExperimentConfig,
    include_diagnostics: bool = True,
    axis_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model_type = config.model.type
    if residuals.shape[1] != 1:
        raise ValueError(f"{model_type} requires one-token residual windows")
    token_residuals = residuals[:, 0]

    if model_type == "standard_sae":
        output = model(token_residuals)
        reconstruction = F.mse_loss(
            output.reconstruction.float(), token_residuals.float()
        )
        sparsity = output.features.float().abs().mean()
        loss = config.loss.reconstruction_weight * reconstruction
        loss = loss + config.loss.sae_l1_coefficient * sparsity
        return loss, {
            "loss": loss.detach(),
            "reconstruction": reconstruction.detach(),
            "l1": sparsity.detach(),
            "active_fraction": (output.features > 0).float().mean().detach(),
        }

    dimension_masks = sample_dimension_masks(
        token_residuals,
        config.model.num_local_views,
        config.model.dimension_keep_fraction,
    )
    if model_type == "dimension_denoising_sae":
        residual_views, masks = stack_dimension_views(
            token_residuals, dimension_masks, include_global=False
        )
        output = model(residual_views, masks)
        target = token_residuals.unsqueeze(0).expand_as(output.reconstruction)
        reconstruction = F.mse_loss(output.reconstruction.float(), target.float())
        sparsity = output.features.float().abs().mean()
        loss = config.loss.reconstruction_weight * reconstruction
        loss = loss + config.loss.sae_l1_coefficient * sparsity
        flattened = output.features.flatten(0, 1)
        return loss, {
            "loss": loss.detach(),
            "reconstruction": reconstruction.detach(),
            "l1": sparsity.detach(),
            "active_fraction": (flattened > 0).float().mean().detach(),
        }

    residual_views, masks = stack_dimension_views(
        token_residuals, dimension_masks, include_global=True
    )
    feature_views = model(residual_views, masks).features
    global_features = feature_views[0]
    local_features = feature_views[1:]
    invariance = F.mse_loss(
        local_features,
        global_features.unsqueeze(0).expand_as(local_features),
    )
    rdm = rectified_lp_rdm_regularization(
        feature_views,
        config.loss.rdm_projections,
        config.loss.axis_projections,
        config.loss.axis_weight,
        config.loss.lp_norm_parameter,
        (
            config.loss.mean_shift_value
            if config.loss.expected_l0_fraction is None
            else generalized_gaussian_mean_shift_for_active_fraction(
                config.loss.lp_norm_parameter,
                config.loss.expected_l0_fraction,
            )
        ),
        axis_indices=axis_indices,
    )
    loss = config.loss.invariance_weight * invariance + config.loss.lambda_rdm * rdm.loss
    if not torch.isfinite(loss):
        raise FloatingPointError("proposed model produced a non-finite loss")
    view_distribution_losses = (
        rdm.random_view_losses + config.loss.axis_weight * rdm.axis_view_losses
    )
    metrics = {
        "loss": loss.detach(),
        "invariance": invariance.detach(),
        "distribution": rdm.loss.detach(),
        "random_distribution": rdm.random_loss.detach(),
        "axis_distribution": rdm.axis_loss.detach(),
        "global_distribution": view_distribution_losses[0].detach(),
        "local_distribution": view_distribution_losses[1:].mean().detach(),
        "global_random_distribution": rdm.random_view_losses[0].detach(),
        "local_random_distribution": rdm.random_view_losses[1:].mean().detach(),
        "global_axis_distribution": rdm.axis_view_losses[0].detach(),
        "local_axis_distribution": rdm.axis_view_losses[1:].mean().detach(),
    }
    if config.loss.expected_l0_fraction is not None:
        metrics["expected_l0_fraction"] = torch.tensor(
            config.loss.expected_l0_fraction,
            device=feature_views.device,
        )
    if not include_diagnostics:
        return loss, metrics

    flattened = feature_views.flatten(0, 1).detach()
    flattened_locals = local_features.flatten(0, 1).detach()
    global_detached = global_features.detach()
    global_float = global_detached.float()
    local_float = flattened_locals.float()
    metrics.update(
        {
            "active_fraction": (flattened > 0).float().mean().detach(),
            "l0_sparsity": (flattened > 0).float().mean().detach(),
            "l1_sparsity": l1_sparsity_metric(flattened).detach(),
            "global_active_fraction": (global_detached > 0).float().mean(),
            "local_active_fraction": (flattened_locals > 0).float().mean().detach(),
            "feature_std": flattened.float().std(dim=0, unbiased=False).mean(),
            "global_feature_std": global_float.std(dim=0, unbiased=False).mean(),
            "local_feature_std": local_float.std(dim=0, unbiased=False).mean(),
            "global_dead_feature_fraction": (
                global_detached.amax(dim=0) <= 0
            ).float().mean(),
            "local_dead_feature_fraction": (
                flattened_locals.amax(dim=0) <= 0
            ).float().mean().detach(),
        }
    )
    return loss, metrics


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
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
) -> dict[str, float]:
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
    return aggregate_metrics(collected)


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

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
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
        if config.model.type == "proposed":
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
                )
                scaled_loss = loss / config.train.gradient_accumulation_steps
            scaled_loss.backward()
            interval_metrics.append(metrics)
            interval_samples += residuals.shape[0]

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
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

    checkpoint_step = int(last_checkpoint.stem.split("-")[-1]) if last_checkpoint else -1
    if checkpoint_step != config.train.max_steps:
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
