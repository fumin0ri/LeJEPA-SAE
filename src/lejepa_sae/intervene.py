from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import load_config
from .data import ActivationWindowDataset
from .evaluate import load_model


def local_gradient_intervention(
    model,
    residuals: torch.Tensor,
    feature_index: int,
    alpha: float,
) -> dict[str, torch.Tensor | float | int]:
    """Construct a sample-dependent residual intervention from ∇_H z_k."""
    residuals = residuals.detach().float().clone().requires_grad_(True)
    if residuals.shape[1] != 1:
        raise ValueError("ProposedModel intervention requires a one-token window")
    features = model(residuals[:, 0]).features
    if not 0 <= feature_index < features.shape[-1]:
        raise ValueError(f"feature_index must be in [0, {features.shape[-1] - 1}]")
    activation = features[:, feature_index].sum()
    (gradient,) = torch.autograd.grad(activation, residuals)
    direction = gradient / gradient.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None]
    modified = residuals.detach() + alpha * direction
    return {
        "residuals": residuals.detach().cpu(),
        "modified_residuals": modified.cpu(),
        "gradient": gradient.cpu(),
        "direction": direction.cpu(),
        "feature_index": feature_index,
        "baseline_activation": float(activation.detach()),
        "alpha": alpha,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a local-gradient feature intervention")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--token-index", type=int, required=True)
    parser.add_argument("--feature-index", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.model.type != "proposed":
        raise ValueError("Local-gradient intervention expects the proposed model")
    model = load_model(config, args.checkpoint, config.train.device)
    dataset = ActivationWindowDataset(
        config.data.activation_dir,
        args.split,
        config.data.window_size,
        config.data.eval_stride,
        config.data.cache_shards_per_worker,
    )
    item = dataset[args.token_index]
    residuals = item["residuals"].unsqueeze(0).to(config.train.device)
    result = local_gradient_intervention(
        model, residuals, args.feature_index, args.alpha
    )
    result["token_ids"] = item["token_ids"]
    result["document_id"] = item["document_id"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
