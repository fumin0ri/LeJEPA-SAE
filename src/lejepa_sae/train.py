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
    gaussian_distribution_regularization,
    invariance_loss,
    rdm_regularization,
)
from .models import build_model
from .views import full_view, sample_local_views


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


def compute_loss(
    model: nn.Module,
    residuals: torch.Tensor,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model_type = config.model.type
    complete = full_view(residuals)

    if model_type == "standard_sae":
        batch_indices = torch.arange(residuals.shape[0], device=residuals.device)
        token_indices = torch.randint(
            residuals.shape[1], (residuals.shape[0],), device=residuals.device
        )
        token_residuals = residuals[batch_indices, token_indices]
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

    if model_type == "window_autoencoder":
        output = model(complete.residuals, complete.positions)
        reconstruction = F.mse_loss(output.reconstruction.float(), residuals.float())
        sparsity = output.features.float().abs().mean()
        loss = config.loss.reconstruction_weight * reconstruction
        loss = loss + config.loss.sae_l1_coefficient * sparsity
        return loss, {
            "loss": loss.detach(),
            "reconstruction": reconstruction.detach(),
            "l1": sparsity.detach(),
            "active_fraction": (output.features > 0).float().mean().detach(),
        }

    global_features = model(complete.residuals, complete.positions).features
    local_views = sample_local_views(
        residuals,
        config.model.num_local_views,
        config.model.local_tokens,
    )
    local_features = [model(view.residuals, view.positions).features for view in local_views]
    invariance = invariance_loss(global_features, local_features)

    all_features = [global_features, *local_features]
    if model_type == "jepa_sigreg":
        distribution_losses = [
            gaussian_distribution_regularization(features, config.loss.rdm_projections)
            for features in all_features
        ]
    else:
        distribution_losses = [
            rdm_regularization(
                features,
                config.loss.rdm_projections,
                config.loss.target_active_fraction,
                config.loss.target_sigma,
            )
            for features in all_features
        ]
    distribution = torch.stack(distribution_losses).mean()
    loss = config.loss.invariance_weight * invariance + config.loss.lambda_rdm * distribution
    flattened = torch.cat(all_features, dim=0)
    return loss, {
        "loss": loss.detach(),
        "invariance": invariance.detach(),
        "distribution": distribution.detach(),
        "active_fraction": (flattened > 0).float().mean().detach(),
        "feature_std": flattened.float().std(dim=0).mean().detach(),
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
    keys = metrics[0].keys()
    return {
        key: float(torch.stack([item[key].float().cpu() for item in metrics]).mean())
        for key in keys
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
    last_checkpoint: Path | None = None

    for step in range(start_step + 1, config.train.max_steps + 1):
        for _ in range(config.train.gradient_accumulation_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                batch = next(data_iterator)
            residuals = batch["residuals"].to(config.train.device, non_blocking=True)
            with autocast_context(config):
                loss, metrics = compute_loss(model, residuals, config)
                scaled_loss = loss / config.train.gradient_accumulation_steps
            scaled_loss.backward()
            interval_metrics.append(metrics)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % config.train.log_every == 0:
            record = {
                "kind": "train",
                "step": step,
                **aggregate_metrics(interval_metrics),
                "grad_norm": float(grad_norm),
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.monotonic() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            interval_metrics.clear()

        if step % config.train.eval_every == 0:
            record = {
                "kind": "validation",
                "step": step,
                **evaluate_loss(model, validation_loader, config),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

        if step % config.train.checkpoint_every == 0:
            last_checkpoint = save_checkpoint(output_dir, step, model, optimizer, scheduler, config)

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
