from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

from .comparison import METRIC_ALIASES, _canonical_metric
from .config import ExperimentConfig, load_config
from .evaluate import load_model
from .probing import (
    HOOK_NAME,
    MODEL_NAME,
    ProbeSAEAdapter,
    _cache_spec,
    _checkpoint_digest,
    _release_memory,
    _write_json,
    prepare_activation_cache,
    run_hook_parity_preflight,
)

DEFAULT_CS = [10.0 ** (5.0 - 10.0 * index / 9.0) for index in range(10)]
DEFAULT_MAX_ITER = 1_000
DEFAULT_TOLERANCE_GRAD = 1e-7
DEFAULT_TOLERANCE_CHANGE = 1e-9


@dataclass
class _LogisticFit:
    weight: torch.Tensor
    bias: torch.Tensor
    loss: float
    gradient_max: float
    iterations: int
    function_evaluations: int
    converged: bool


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"Dense z probe received non-finite {name}")


def _validate_binary_labels(name: str, labels: torch.Tensor) -> None:
    _require_finite(name, labels.float())
    values = set(labels.detach().cpu().tolist())
    if values != {0, 1}:
        raise ValueError(f"Dense z probe requires both binary classes in {name}; got {values}")


def _fit_logistic(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    c_value: float,
    device: str | torch.device,
    max_iter: int = DEFAULT_MAX_ITER,
    tolerance_grad: float = DEFAULT_TOLERANCE_GRAD,
    tolerance_change: float = DEFAULT_TOLERANCE_CHANGE,
) -> _LogisticFit:
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("Dense z probe features/labels have incompatible shapes")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("Dense z probe features must be nonempty")
    if not math.isfinite(c_value) or c_value <= 0:
        raise ValueError("Dense z probe C must be finite and positive")
    if max_iter < 1 or tolerance_grad <= 0 or tolerance_change <= 0:
        raise ValueError("Dense z probe optimizer limits/tolerances must be positive")
    _require_finite("features", features)
    _validate_binary_labels("labels", labels)

    target_device = torch.device(device)
    x = features.detach().to(device=target_device, dtype=torch.float32)
    y = labels.detach().to(device=target_device, dtype=torch.float32)
    weight = torch.nn.Parameter(torch.zeros(x.shape[1], device=target_device))
    bias = torch.nn.Parameter(torch.zeros((), device=target_device))
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    regularization = 1.0 / (c_value * x.shape[0])

    def objective(backward: bool) -> torch.Tensor:
        logits = x.mv(weight) + bias
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss = loss + 0.5 * regularization * weight.square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Dense z probe objective became non-finite")
        if backward:
            loss.backward()
            for parameter in (weight, bias):
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Dense z probe gradient became non-finite")
        return loss

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        return objective(backward=True)

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    final_loss = objective(backward=True)
    gradient_max = max(float(weight.grad.abs().max()), float(bias.grad.abs()))
    state = optimizer.state[weight]
    iterations = int(state.get("n_iter", 0))
    function_evaluations = int(state.get("func_evals", 0))
    return _LogisticFit(
        weight=weight.detach(),
        bias=bias.detach(),
        loss=float(final_loss.detach()),
        gradient_max=gradient_max,
        iterations=iterations,
        function_evaluations=function_evaluations,
        converged=iterations < max_iter or gradient_max <= tolerance_grad,
    )


def _metrics(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y_true = labels.detach().cpu().numpy()
    probabilities = logits.detach().float().sigmoid().cpu().numpy()
    predictions = probabilities >= 0.5
    result = {
        "f1": float(f1_score(y_true, predictions, average="weighted")),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "auroc": float(roc_auc_score(y_true, probabilities)),
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Dense z probe produced a non-finite metric")
    return result


def _logits(features: torch.Tensor, fit: _LogisticFit, device: str | torch.device) -> torch.Tensor:
    x = features.detach().to(device=device, dtype=torch.float32)
    return (x.mv(fit.weight) + fit.bias).cpu()


def fit_dense_logistic_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    device: str | torch.device,
    cs: list[float] = DEFAULT_CS,
    max_iter: int = DEFAULT_MAX_ITER,
    tolerance_grad: float = DEFAULT_TOLERANCE_GRAD,
    tolerance_change: float = DEFAULT_TOLERANCE_CHANGE,
) -> dict[str, Any]:
    """Fit an all-feature L2 logistic probe with the official validation splits."""
    from sae_probes.utils_training import get_cv, get_splits

    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Dense z probe train/test features must be rank-2 with equal width")
    if x_train.shape[0] != y_train.shape[0] or x_test.shape[0] != y_test.shape[0]:
        raise ValueError("Dense z probe train/test labels do not match feature rows")
    if not cs or any(not math.isfinite(value) or value <= 0 for value in cs):
        raise ValueError("Dense z probe C grid must contain finite positive values")
    if len(set(cs)) != len(cs):
        raise ValueError("Dense z probe C grid must not contain duplicates")
    for name, value in (
        ("training features", x_train),
        ("test features", x_test),
    ):
        _require_finite(name, value)
    _validate_binary_labels("training labels", y_train)
    _validate_binary_labels("test labels", y_test)

    target_device = torch.device(device)
    x_train_device = x_train.detach().to(device=target_device, dtype=torch.float32)
    x_test_device = x_test.detach().to(device=target_device, dtype=torch.float32)
    y_train_cpu = y_train.detach().cpu()
    validation_scores: list[dict[str, Any]] = []
    if x_train.shape[0] > 3:
        cv = get_cv(x_train.detach().cpu().numpy())
        splits = get_splits(cv, x_train.detach().cpu().numpy(), y_train_cpu.numpy())
        if not splits:
            raise ValueError("Dense z probe validation produced no binary-class splits")
        for c_value in cs:
            fold_scores = []
            fold_diagnostics = []
            for train_indices, validation_indices in splits:
                fold_labels = y_train_cpu[train_indices]
                if len(set(fold_labels.tolist())) < 2:
                    fold_scores.append(0.5)
                    fold_diagnostics.append(
                        {
                            "iterations": 0,
                            "function_evaluations": 0,
                            "converged": False,
                            "single_class_training_fold": True,
                        }
                    )
                    continue
                fit = _fit_logistic(
                    x_train_device[train_indices],
                    fold_labels,
                    c_value=c_value,
                    device=device,
                    max_iter=max_iter,
                    tolerance_grad=tolerance_grad,
                    tolerance_change=tolerance_change,
                )
                validation_metrics = _metrics(
                    y_train_cpu[validation_indices],
                    _logits(x_train_device[validation_indices], fit, target_device),
                )
                fold_scores.append(validation_metrics["auroc"])
                fold_diagnostics.append(
                    {
                        "iterations": fit.iterations,
                        "function_evaluations": fit.function_evaluations,
                        "converged": fit.converged,
                    }
                )
            validation_scores.append(
                {
                    "c": c_value,
                    "auroc": mean(fold_scores),
                    "folds": fold_diagnostics,
                }
            )
        best_index = max(
            range(len(validation_scores)),
            key=lambda index: validation_scores[index]["auroc"],
        )
        best_c = cs[best_index]
        validation_auroc = float(validation_scores[best_index]["auroc"])
    else:
        # Match sklearn LogisticRegression's default used by the existing evaluator
        # when there are too few examples for validation.
        best_c = 1.0
        validation_auroc = math.nan

    final_fit = _fit_logistic(
        x_train_device,
        y_train_cpu,
        c_value=best_c,
        device=device,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
    )
    train_metrics = _metrics(y_train_cpu, _logits(x_train_device, final_fit, target_device))
    if math.isnan(validation_auroc):
        validation_auroc = train_metrics["auroc"]
    test_metrics = _metrics(y_test, _logits(x_test_device, final_fit, target_device))
    return {
        "feature_dim": int(x_train.shape[1]),
        "best_c": float(best_c),
        "val_auc": validation_auroc,
        "test_f1": test_metrics["f1"],
        "test_acc": test_metrics["accuracy"],
        "test_auc": test_metrics["auroc"],
        "validation": validation_scores,
        "optimizer": {
            "name": "torch.optim.LBFGS",
            "max_iter": max_iter,
            "tolerance_grad": tolerance_grad,
            "tolerance_change": tolerance_change,
            "line_search_fn": "strong_wolfe",
            "final_loss": final_fit.loss,
            "final_gradient_max": final_fit.gradient_max,
            "iterations": final_fit.iterations,
            "function_evaluations": final_fit.function_evaluations,
            "converged": final_fit.converged,
        },
    }


def _dense_result_path(output: Path, dataset: str) -> Path:
    return output / "dense_z_gpu" / "normal_setting" / f"{dataset}_{HOOK_NAME}_l2.json"


def _validate_dense_result(path: Path, dataset: str, feature_dim: int) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid dense z probe result: {path}") from error
    if (
        not isinstance(result, dict)
        or result.get("dataset") != dataset
        or result.get("feature_dim") != feature_dim
        or result.get("method") != "dense_z_gpu_logistic_regression"
        or not isinstance(result.get("best_c"), int | float)
        or not math.isfinite(float(result["best_c"]))
        or float(result["best_c"]) <= 0
    ):
        raise RuntimeError(f"Incompatible dense z probe result: {path}")
    try:
        metrics = {name: _canonical_metric(result, name) for name in METRIC_ALIASES}
    except ValueError as error:
        raise RuntimeError(f"Incomplete dense z probe result: {path}") from error
    if any(not math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Non-finite dense z probe result: {path}")
    return result


def _validate_dense_manifest(output: Path, spec: dict[str, Any]) -> None:
    path = output / "dense_probe_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != spec:
            raise ValueError(
                "Dense probe settings/checkpoint differ from existing results; "
                "use a new results-path"
            )
    elif (output / "dense_z_gpu").exists():
        raise ValueError("Existing dense probe results have no manifest; use a new results-path")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(path, spec)


def _summarize_dense_results(output: Path, datasets: list[str], feature_dim: int) -> None:
    rows = [
        _validate_dense_result(_dense_result_path(output, dataset), dataset, feature_dim)
        for dataset in datasets
    ]
    _write_json(
        output / "dense_z_probe_summary.json",
        {
            "complete": True,
            "tasks": datasets,
            "feature_dim": feature_dim,
            "macro_metrics": {
                metric: mean(_canonical_metric(row, metric) for row in rows)
                for metric in METRIC_ALIASES
            },
        },
    )


def run_dense_z_probes(
    config: ExperimentConfig,
    checkpoint: str | Path,
    results_path: str | Path,
    model_cache_path: str | Path,
    *,
    datasets: list[str] | None = None,
    parity: bool = True,
    llm_precision: str = "auto",
    activation_batch_size: int = 1,
    max_seq_len: int = 1024,
    smoke_test: bool = False,
    cs: list[float] = DEFAULT_CS,
    max_iter: int = DEFAULT_MAX_ITER,
) -> None:
    try:
        from sae_probes import DATASETS
        from sae_probes.generate_sae_activations import generate_sae_activations
    except ImportError as error:
        raise RuntimeError(
            "Install probing dependencies with: pip install -e '.[probes]'"
        ) from error

    config.validate()
    device = torch.device(config.train.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Dense z probe requires an available CUDA device")
    datasets = sorted(set(DATASETS if datasets is None else datasets))
    unknown = set(datasets) - set(DATASETS)
    if not datasets or unknown:
        raise ValueError(f"Select nonempty official datasets; unknown datasets: {sorted(unknown)}")
    if smoke_test:
        datasets = datasets[:1]
    if activation_batch_size < 1 or not 1 <= max_seq_len <= 2048:
        raise ValueError("activation batch size must be positive and max_seq_len in [1, 2048]")
    if max_iter < 1:
        raise ValueError("Dense z probe max_iter must be positive")
    cs = [float(value) for value in cs]
    if not cs or any(not math.isfinite(value) or value <= 0 for value in cs):
        raise ValueError("Dense z probe C grid must contain finite positive values")
    if len(set(cs)) != len(cs):
        raise ValueError("Dense z probe C grid must not contain duplicates")

    cache_spec = _cache_spec(config.train.device, llm_precision, max_seq_len)
    output = Path(results_path)
    manifest = {
        "schema": 1,
        "checkpoint_sha256": _checkpoint_digest(checkpoint),
        "config": config.to_dict(),
        "feature_dim": config.model.feature_dim,
        "datasets": datasets,
        "setting": "normal",
        "method": "dense_z_gpu_logistic_regression",
        "penalty": "l2",
        "normalization": "none",
        "seed": 42,
        "cs": cs,
        "optimizer": {
            "name": "torch.optim.LBFGS",
            "max_iter": max_iter,
            "tolerance_grad": DEFAULT_TOLERANCE_GRAD,
            "tolerance_change": DEFAULT_TOLERANCE_CHANGE,
            "line_search_fn": "strong_wolfe",
        },
        "probe_device": str(device),
        "probe_dtype": "torch.float32",
        "cache": cache_spec,
        "smoke_test": smoke_test,
    }
    _validate_dense_manifest(output, manifest)

    adapter = ProbeSAEAdapter(load_model(config, checkpoint, "cpu"), config)
    print(
        f"Dense z GPU probe tasks={len(datasets)}, width={config.model.feature_dim}, "
        f"smoke_test={smoke_test}",
        flush=True,
    )
    if parity:
        parity_result = run_hook_parity_preflight(
            config.train.device, llm_precision, output / "hook_parity.json"
        )
        _write_json(output / "hook_parity.json", parity_result)
    cache = prepare_activation_cache(
        model_cache_path,
        datasets,
        config.train.device,
        llm_precision=llm_precision,
        batch_size=activation_batch_size,
        max_seq_len=max_seq_len,
    )
    print(f"Dense z probe activation cache: {cache}", flush=True)
    adapter.to(config.train.device).eval()
    try:
        for dataset in datasets:
            result_path = _dense_result_path(output, dataset)
            if result_path.is_file():
                _validate_dense_result(result_path, dataset, config.model.feature_dim)
                print(f"Skipping completed dense z probe task {dataset}", flush=True)
                continue
            activations = generate_sae_activations(
                sae=adapter,
                setting="normal",
                dataset=dataset,
                hook_name=HOOK_NAME,
                model_name=MODEL_NAME,
                device=config.train.device,
                num_train=None,
                frac=None,
                model_cache_path=cache,
                batch_size=128,
                seed=42,
            )
            result = fit_dense_logistic_probe(
                activations.X_train,
                activations.y_train,
                activations.X_test,
                activations.y_test,
                device=config.train.device,
                cs=cs,
                max_iter=max_iter,
            )
            if result["feature_dim"] != config.model.feature_dim:
                raise RuntimeError(
                    "Dense z probe encoder width differs from checkpoint configuration"
                )
            result.update(
                {
                    "schema": 1,
                    "dataset": dataset,
                    "hook_name": HOOK_NAME,
                    "method": "dense_z_gpu_logistic_regression",
                    "backend": "torch",
                    "device": str(device),
                    "dtype": "torch.float32",
                    "penalty": "l2",
                    "normalization": "none",
                    "seed": 42,
                    "cs": cs,
                }
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(result_path, result)
            del activations, result
            _release_memory(config.train.device)
    finally:
        del adapter
        _release_memory(config.train.device)
    _summarize_dense_results(output, datasets, config.model.feature_dim)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an all-feature L2 logistic probe on SAE z using CUDA"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--model-cache-path", required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument(
        "--llm-precision",
        choices=("auto", "bfloat16", "float32"),
        default="auto",
        help="LLM/cache precision: auto uses bfloat16 on CUDA",
    )
    parser.add_argument("--activation-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--cs", nargs="+", type=float, default=DEFAULT_CS)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the first selected official task; use a separate results directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dense_z_probes(
        load_config(args.config),
        args.checkpoint,
        args.results_path,
        args.model_cache_path,
        datasets=args.datasets,
        parity=not args.skip_parity,
        llm_precision=args.llm_precision,
        activation_batch_size=args.activation_batch_size,
        max_seq_len=args.max_seq_len,
        smoke_test=args.smoke_test,
        cs=args.cs,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
