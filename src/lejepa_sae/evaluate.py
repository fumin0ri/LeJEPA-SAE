from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import ExperimentConfig, load_config
from .data import ActivationWindowDataset
from .diagnostics import thresholded_active_counts
from .models import RDMSAE, SAEBase, build_model
from .reporting import load_training_history, write_evaluation_report
from .train import activation_transition_metrics, autocast_context, seed_everything
from .views import sample_dimension_masks


def encode_features(model, residuals: torch.Tensor) -> torch.Tensor:
    if residuals.shape[1] != 1:
        raise ValueError("evaluation requires one-token windows")
    return model(residuals[:, 0]).features


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
    max_tokens: int,
    top_k: int,
    support_epsilon: float,
    concept_labels_path: str | None = None,
    training_metrics_path: str | Path | None = None,
) -> dict[str, float]:
    seed_everything(config.train.seed)
    if training_metrics_path is not None:
        history_path = Path(training_metrics_path)
        if not history_path.is_file():
            raise FileNotFoundError(f"Training metrics not found: {history_path}")
    else:
        history_path = Path(config.train.output_dir) / "metrics.jsonl"
    training_history = (
        load_training_history(history_path) if history_path.is_file() else []
    )
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
    transition_totals: dict[str, float] = defaultdict(float)
    full_reconstruction_total = 0.0
    residual_sum = torch.zeros(config.model.d_llm, dtype=torch.float64)
    residual_square_sum = torch.zeros(config.model.d_llm, dtype=torch.float64)
    evaluated = 0
    threshold_counts: dict[str, int] = defaultdict(int)

    concept_labels: dict[str, str] = {}
    concept_active: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(feature_dim))
    concept_examples: dict[str, int] = defaultdict(int)
    if concept_labels_path:
        concept_labels = json.loads(Path(concept_labels_path).read_text(encoding="utf-8"))

    for batch in loader:
        if evaluated >= max_tokens:
            break
        remaining = max_tokens - evaluated
        residuals = batch["residuals"][:remaining].to(device, non_blocking=True)
        batch_token_ids = batch["token_ids"][:remaining]
        batch_document_ids = list(batch["document_id"][:remaining])
        with autocast_context(config):
            features = encode_features(model, residuals)
        feature_cpu = features.float().cpu()
        if isinstance(model, RDMSAE):
            for key, active_count in thresholded_active_counts(feature_cpu).items():
                threshold_counts[key] += int(active_count)
        support = feature_cpu > support_epsilon
        count = features.shape[0]
        residual_float = residuals[:, 0].float()
        residual_cpu = residual_float.cpu().double()
        residual_sum += residual_cpu.sum(dim=0)
        residual_square_sum += residual_cpu.square().sum(dim=0)

        active_sum += support.sum(dim=0)
        value_sum += feature_cpu.sum(dim=0, dtype=torch.float64)
        square_sum += feature_cpu.square().sum(dim=0, dtype=torch.float64)
        maxima = torch.maximum(maxima, feature_cpu.max(dim=0).values)
        top_scores, top_indices = update_topk(
            top_scores, top_indices, feature_cpu, evaluated, top_k
        )
        token_rows.extend(batch_token_ids.cpu())
        document_ids.extend(batch_document_ids)

        if config.model.type == "proposed" and config.model.num_local_views > 0:
            token_residuals = residuals[:, 0]
            dimension_mask = sample_dimension_masks(
                token_residuals, 1, config.model.dimension_keep_fraction
            )[0]
            with autocast_context(config):
                global_features = features.float()
                local_features = model(token_residuals, dimension_mask).features.float()
            for key, value in activation_transition_metrics(
                global_features, local_features.unsqueeze(0)
            ).items():
                transition_totals[key] += value.item() * count
            invariance_total += torch.nn.functional.mse_loss(
                global_features, local_features, reduction="sum"
            ).item() / feature_dim
            global_support = global_features > support_epsilon
            local_support = local_features > support_epsilon
            intersection = (global_support & local_support).sum(dim=1)
            union = (global_support | local_support).sum(dim=1)
            jaccard_total += (intersection / union.clamp_min(1)).sum().item()

        elif isinstance(model, SAEBase):
            with autocast_context(config):
                full_output = model(residuals[:, 0])
            full_reconstruction_total += torch.nn.functional.mse_loss(
                full_output.reconstruction.float(),
                residuals[:, 0].float(),
                reduction="sum",
            ).item() / config.model.d_llm

        for row, document_id in enumerate(batch_document_ids):
            label = concept_labels.get(str(document_id))
            if label is not None:
                concept_active[label] += support[row].float()
                concept_examples[label] += 1
        evaluated += count

    if evaluated == 0:
        raise ValueError("No test tokens were evaluated")
    mean = value_sum / evaluated
    variance = (square_sum / evaluated - mean.square()).clamp_min(0)
    result = {
        "tokens": float(evaluated),
        "mean_active_fraction": float(active_sum.sum() / (evaluated * feature_dim)),
        "dead_feature_fraction": float((maxima <= support_epsilon).float().mean()),
        "mean_feature_std": float(variance.sqrt().mean()),
    }
    result.update({
        key: count / (evaluated * feature_dim) for key, count in threshold_counts.items()
    })
    if config.model.type == "proposed" and config.model.num_local_views > 0:
        result["global_local_mse"] = invariance_total / evaluated
        result["support_jaccard"] = jaccard_total / evaluated
        result.update({key: value / evaluated for key, value in transition_totals.items()})
    if isinstance(model, SAEBase):
        result["full_reconstruction_mse"] = full_reconstruction_total / evaluated
        residual_mean = residual_sum / evaluated
        residual_variance = float(
            (residual_square_sum / evaluated - residual_mean.square()).clamp_min(0).mean()
        )
        result["fvu"] = result["full_reconstruction_mse"] / max(residual_variance, 1e-12)
        result["mean_l0"] = result["mean_active_fraction"] * feature_dim

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dataset.metadata
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"], revision=manifest.get("revision", "main")
    )
    top_records = []
    with (output / "top_tokens.jsonl").open("w", encoding="utf-8") as handle:
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
            top_records.append(record)
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
    active_rates = active_sum / evaluated
    feature_std = variance.sqrt()
    feature_rows = [
        {
            "feature": feature_index,
            "active_fraction": float(active_rates[feature_index]),
            "mean": float(mean[feature_index]),
            "std": float(feature_std[feature_index]),
            "maximum": float(maxima[feature_index]),
        }
        for feature_index in range(feature_dim)
    ]
    write_evaluation_report(
        output,
        result,
        feature_rows,
        top_records,
        training_history=training_history,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize single-token sparse features"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--support-epsilon", type=float, default=0.0)
    parser.add_argument("--concept-labels", default=None)
    parser.add_argument(
        "--training-metrics",
        default=None,
        help="Training metrics JSONL (defaults to <train.output_dir>/metrics.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = evaluate(
        config,
        args.checkpoint,
        args.output_dir,
        args.max_tokens,
        args.top_k,
        args.support_epsilon,
        args.concept_labels,
        args.training_metrics,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
