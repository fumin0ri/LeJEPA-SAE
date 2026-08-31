from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import TOKEN_VIEW_MODEL_TYPES, ExperimentConfig, load_config
from .data import ActivationWindowDataset
from .models import build_model
from .train import autocast_context, seed_everything
from .views import full_view, sample_dimension_masks, sample_local_views


def span_features(model, residuals: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    if config.model.type == "standard_sae":
        token_features = model(residuals).features
        return token_features.max(dim=1).values
    if config.model.type in {"single_token_jepa", "dimension_denoising_sae"}:
        if residuals.shape[1] != 1:
            raise ValueError("dimension-view evaluation requires one-token windows")
        return model(residuals[:, 0]).features
    complete = full_view(residuals)
    return model(complete.residuals, complete.positions).features


def load_model(config: ExperimentConfig, checkpoint_path: str | Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def update_topk(
    current_scores: torch.Tensor,
    current_indices: torch.Tensor,
    features: torch.Tensor,
    first_index: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = features.float().T.cpu()
    indices = torch.arange(first_index, first_index + features.shape[0]).expand(
        features.shape[1], -1
    )
    combined_scores = torch.cat((current_scores, scores), dim=1)
    combined_indices = torch.cat((current_indices, indices), dim=1)
    kept_scores, selection = combined_scores.topk(min(top_k, combined_scores.shape[1]), dim=1)
    kept_indices = combined_indices.gather(1, selection)
    return kept_scores, kept_indices


@torch.inference_mode()
def evaluate(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    max_windows: int,
    top_k: int,
    support_epsilon: float,
    concept_labels_path: str | None = None,
) -> dict[str, float]:
    seed_everything(config.train.seed)
    device = config.train.device
    dataset = ActivationWindowDataset(
        config.data.activation_dir,
        "test",
        config.data.window_size,
        config.data.eval_stride,
        config.data.cache_shards_per_worker,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    model = load_model(config, checkpoint_path, device)
    feature_dim = config.model.feature_dim
    top_scores = torch.empty(feature_dim, 0)
    top_indices = torch.empty(feature_dim, 0, dtype=torch.long)
    token_rows: list[torch.Tensor] = []
    document_ids: list[str] = []
    active_sum = torch.zeros(feature_dim, dtype=torch.float64)
    value_sum = torch.zeros(feature_dim, dtype=torch.float64)
    square_sum = torch.zeros(feature_dim, dtype=torch.float64)
    maxima = torch.full((feature_dim,), -torch.inf)
    invariance_total = 0.0
    jaccard_total = 0.0
    full_reconstruction_total = 0.0
    masked_reconstruction_total = 0.0
    evaluated = 0

    concept_labels: dict[str, str] = {}
    concept_active: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(feature_dim))
    concept_examples: dict[str, int] = defaultdict(int)
    if concept_labels_path:
        concept_labels = json.loads(Path(concept_labels_path).read_text(encoding="utf-8"))

    for batch in loader:
        if evaluated >= max_windows:
            break
        remaining = max_windows - evaluated
        residuals = batch["residuals"][:remaining].to(device, non_blocking=True)
        batch_token_ids = batch["token_ids"][:remaining]
        batch_document_ids = list(batch["document_id"][:remaining])
        with autocast_context(config):
            features = span_features(model, residuals, config)
        feature_cpu = features.float().cpu()
        support = feature_cpu > support_epsilon
        count = features.shape[0]

        active_sum += support.sum(dim=0)
        value_sum += feature_cpu.sum(dim=0, dtype=torch.float64)
        square_sum += feature_cpu.square().sum(dim=0, dtype=torch.float64)
        maxima = torch.maximum(maxima, feature_cpu.max(dim=0).values)
        top_scores, top_indices = update_topk(
            top_scores, top_indices, feature_cpu, evaluated, top_k
        )
        token_rows.extend(batch_token_ids.cpu())
        document_ids.extend(batch_document_ids)

        if config.model.type in TOKEN_VIEW_MODEL_TYPES:
            complete = full_view(residuals)
            local = sample_local_views(residuals, 1, config.model.local_tokens)[0]
            with autocast_context(config):
                global_features = model(complete.residuals, complete.positions).features.float()
                local_features = model(local.residuals, local.positions).features.float()
            invariance_total += torch.nn.functional.mse_loss(
                global_features, local_features, reduction="sum"
            ).item() / feature_dim
            global_support = global_features > support_epsilon
            local_support = local_features > support_epsilon
            intersection = (global_support & local_support).sum(dim=1)
            union = (global_support | local_support).sum(dim=1)
            jaccard_total += (intersection / union.clamp_min(1)).sum().item()

        elif config.model.type == "single_token_jepa":
            token_residuals = residuals[:, 0]
            dimension_mask = sample_dimension_masks(
                token_residuals, 1, config.model.dimension_keep_fraction
            )[0]
            with autocast_context(config):
                global_features = model(token_residuals).features.float()
                local_features = model(token_residuals, dimension_mask).features.float()
            invariance_total += torch.nn.functional.mse_loss(
                global_features, local_features, reduction="sum"
            ).item() / feature_dim
            global_support = global_features > support_epsilon
            local_support = local_features > support_epsilon
            intersection = (global_support & local_support).sum(dim=1)
            union = (global_support | local_support).sum(dim=1)
            jaccard_total += (intersection / union.clamp_min(1)).sum().item()

        elif config.model.type == "dimension_denoising_sae":
            token_residuals = residuals[:, 0]
            dimension_mask = sample_dimension_masks(
                token_residuals, 1, config.model.dimension_keep_fraction
            )[0]
            with autocast_context(config):
                full_output = model(token_residuals)
                masked_output = model(token_residuals, dimension_mask)
            full_reconstruction_total += torch.nn.functional.mse_loss(
                full_output.reconstruction.float(),
                token_residuals.float(),
                reduction="sum",
            ).item() / config.model.d_llm
            masked_reconstruction_total += torch.nn.functional.mse_loss(
                masked_output.reconstruction.float(),
                token_residuals.float(),
                reduction="sum",
            ).item() / config.model.d_llm

        elif config.model.type == "standard_sae":
            with autocast_context(config):
                full_output = model(residuals)
            elements_per_window = residuals.shape[1] * config.model.d_llm
            full_reconstruction_total += torch.nn.functional.mse_loss(
                full_output.reconstruction.float(),
                residuals.float(),
                reduction="sum",
            ).item() / elements_per_window

        for row, document_id in enumerate(batch_document_ids):
            label = concept_labels.get(str(document_id))
            if label is not None:
                concept_active[label] += support[row].float()
                concept_examples[label] += 1
        evaluated += count

    if evaluated == 0:
        raise ValueError("No test windows were evaluated")
    mean = value_sum / evaluated
    variance = (square_sum / evaluated - mean.square()).clamp_min(0)
    result = {
        "windows": float(evaluated),
        "mean_active_fraction": float(active_sum.sum() / (evaluated * feature_dim)),
        "dead_feature_fraction": float((maxima <= support_epsilon).float().mean()),
        "mean_feature_std": float(variance.sqrt().mean()),
    }
    if config.model.type in {*TOKEN_VIEW_MODEL_TYPES, "single_token_jepa"}:
        result["global_local_mse"] = invariance_total / evaluated
        result["support_jaccard"] = jaccard_total / evaluated
    if config.model.type in {"standard_sae", "dimension_denoising_sae"}:
        result["full_reconstruction_mse"] = full_reconstruction_total / evaluated
    if config.model.type == "dimension_denoising_sae":
        result["masked_reconstruction_mse"] = masked_reconstruction_total / evaluated

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dataset.metadata
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"], revision=manifest.get("revision", "main")
    )
    with (output / "top_spans.jsonl").open("w", encoding="utf-8") as handle:
        for feature_index in range(feature_dim):
            examples = []
            for score, row_index in zip(
                top_scores[feature_index].tolist(), top_indices[feature_index].tolist(), strict=True
            ):
                if row_index < 0 or row_index >= len(token_rows):
                    continue
                examples.append(
                    {
                        "score": score,
                        "document_id": document_ids[row_index],
                        "text": tokenizer.decode(token_rows[row_index].tolist()),
                        "token_ids": token_rows[row_index].tolist(),
                    }
                )
            record = {"feature": feature_index, "examples": examples}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if concept_active:
        labels = sorted(concept_active)
        rates = torch.stack(
            [concept_active[label] / concept_examples[label] for label in labels], dim=0
        )
        dominant_rate, dominant_index = rates.max(dim=0)
        dominant_labels = [labels[index] for index in dominant_index.tolist()]
        merging = (rates > 0.1).sum(dim=0).float()
        splitting = {
            label: dominant_labels.count(label)
            for label in labels
        }
        semantic = {
            "labels": labels,
            "mean_dominant_concept_consistency": float(dominant_rate.mean()),
            "mean_concepts_per_feature_at_10pct": float(merging.mean()),
            "features_dominated_by_concept": splitting,
        }
        (output / "semantic_metrics.json").write_text(
            json.dumps(semantic, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        result["mean_dominant_concept_consistency"] = semantic[
            "mean_dominant_concept_consistency"
        ]
        result["mean_concepts_per_feature_at_10pct"] = semantic[
            "mean_concepts_per_feature_at_10pct"
        ]

    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sparse features and token-drop robustness"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-windows", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--support-epsilon", type=float, default=0.0)
    parser.add_argument("--concept-labels", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = evaluate(
        config,
        args.checkpoint,
        args.output_dir,
        args.max_windows,
        args.top_k,
        args.support_epsilon,
        args.concept_labels,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
