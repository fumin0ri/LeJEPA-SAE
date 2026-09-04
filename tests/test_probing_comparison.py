import json

import pytest
import torch

from lejepa_sae.calibration import choose_jumprelu_lambda, next_grid_lambda
from lejepa_sae.comparison import (
    aggregate_dense_probe_runs,
    aggregate_probe_runs,
    generate_comparison,
    validate_complete_dense_tasks,
    validate_complete_tasks,
)
from lejepa_sae.config import DataConfig, ExperimentConfig, ModelConfig
from lejepa_sae.models import build_model
from lejepa_sae.probing import ProbeSAEAdapter, assert_hook_parity
from lejepa_sae.train import calibrate_batch_topk_threshold


def baseline_config(model_type="batch_topk_sae"):
    config = ExperimentConfig(
        data=DataConfig(window_size=1, num_workers=0),
        model=ModelConfig(type=model_type, d_llm=4, feature_dim=8),
    )
    config.loss.axis_projections = 2
    config.train.device = "cpu"
    config.train.precision = "float32"
    config.baseline.k = 2
    config.baseline.k_aux = 2
    config.baseline.matryoshka_group_sizes = [1, 1, 2, 2, 2]
    config.validate()
    return config


def test_calibrated_batchtopk_is_pointwise_and_hits_target_l0():
    config = baseline_config()
    model = build_model(config)
    batches = [
        {"residuals": torch.randn(6, 1, 4)},
        {"residuals": torch.randn(6, 1, 4)},
    ]
    result = calibrate_batch_topk_threshold(model, batches, config)
    all_residuals = torch.cat([batch["residuals"][:, 0] for batch in batches])
    direct = model.encode(all_residuals, pointwise=True)
    split = torch.cat(
        [model.encode(batch["residuals"][:, 0], pointwise=True) for batch in batches]
    )
    torch.testing.assert_close(direct, split)
    assert result["calibrated_l0"] == pytest.approx(2.0)


def test_probe_adapter_matches_direct_encoding_and_preserves_leading_shape():
    config = baseline_config("jump_relu_sae")
    model = build_model(config).eval()
    adapter = ProbeSAEAdapter(model, config)
    inputs = torch.randn(2, 3, 4)
    expected = model.encode(inputs.reshape(-1, 4), pointwise=True).reshape(2, 3, 8)
    torch.testing.assert_close(adapter.encode(inputs), expected)


def test_hook_parity_reports_errors_and_rejects_mismatch():
    activation = torch.randn(2, 3, 4)
    result = assert_hook_parity(activation, activation + 1e-5)
    assert result["max_abs_error"] > 0
    with pytest.raises(ValueError, match="Hook parity failed"):
        assert_hook_parity(activation, activation + 1.0)


def test_jumprelu_lambda_selection_and_grid_extension():
    results = [
        {"lambda": 1e-4, "l0": 220.0, "fvu": 0.1},
        {"lambda": 1e-3, "l0": 165.0, "fvu": 0.2},
    ]
    assert choose_jumprelu_lambda(results, 160.0)["lambda"] == 1e-3
    assert next_grid_lambda(results, 100.0) == pytest.approx(1e-2)
    assert next_grid_lambda(results, 300.0) == pytest.approx(1e-5)


def _probe_rows(tasks=("task-a", "task-b"), ks=(1, 16), offset=0.0):
    return [
        {
            "task": task,
            "k": k,
            "f1": 0.5 + offset,
            "auroc": 0.6 + offset,
            "accuracy": 0.7 + offset,
        }
        for task in tasks
        for k in ks
    ]


def test_task_completeness_and_seed_aggregation():
    records = {"a": _probe_rows(), "b": _probe_rows(offset=0.1)}
    assert validate_complete_tasks(records, [1, 16]) == ["task-a", "task-b"]
    with pytest.raises(ValueError, match="mismatched"):
        validate_complete_tasks({"a": records["a"], "b": records["b"][:-1]}, [1, 16])
    runs = [
        {"name": "a", "series": "proposed_relu", "seed": 42},
        {"name": "b", "series": "proposed_relu", "seed": 43},
    ]
    summary, task_rows = aggregate_probe_runs(runs, records)
    assert len(summary) == 2
    assert summary[0]["f1_mean"] == pytest.approx(0.55)
    assert len(task_rows) == 8


def test_dense_task_completeness_and_seed_aggregation():
    records = {
        "a": [
            {"task": task, "feature_dim": 8, "best_c": 1.0, **metrics}
            for task, metrics in (
                ("task-a", {"f1": 0.5, "auroc": 0.6, "accuracy": 0.7}),
                ("task-b", {"f1": 0.7, "auroc": 0.8, "accuracy": 0.9}),
            )
        ]
    }
    validate_complete_dense_tasks(records, ["task-a", "task-b"])
    with pytest.raises(ValueError, match="dense z probe"):
        validate_complete_dense_tasks({"a": records["a"][:-1]}, ["task-a", "task-b"])
    runs = [{"name": "a", "series": "proposed_relu", "seed": 42, "feature_dim": 8}]
    summary, task_rows = aggregate_dense_probe_runs(runs, records)
    assert summary[0]["f1_mean"] == pytest.approx(0.6)
    assert len(task_rows) == 2


def test_comparison_report_artifacts(tmp_path):
    runs = []
    for seed, offset in ((42, 0.0), (43, 0.1), (44, 0.2)):
        run_dir = tmp_path / f"run-{seed}"
        result_dir = tmp_path / f"results-{seed}" / "nested"
        run_dir.mkdir()
        result_dir.mkdir(parents=True)
        (run_dir / "metrics.jsonl").write_text(
            json.dumps(
                {
                    "kind": "validation",
                    "step": 2,
                    "l0": 160.0,
                    "fvu": 0.2,
                    "dead_feature_fraction": 0.1,
                }
            ),
            encoding="utf-8",
        )
        official_rows = [
            {
                "dataset": row["task"],
                "k": row["k"],
                "f1_score": row["f1"],
                "roc_auc": row["auroc"],
                "accuracy": row["accuracy"],
            }
            for row in _probe_rows(ks=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512), offset=offset)
        ]
        (result_dir / "tasks.json").write_text(json.dumps(official_rows), encoding="utf-8")
        dense_result_dir = tmp_path / f"dense-results-{seed}" / "dense_z_gpu"
        dense_result_dir.mkdir(parents=True)
        for task in ("task-a", "task-b"):
            (dense_result_dir / f"{task}.json").write_text(
                json.dumps(
                    {
                        "dataset": task,
                        "feature_dim": 16384,
                        "best_c": 1.0,
                        "method": "dense_z_gpu_logistic_regression",
                        "test_f1": 0.55 + offset,
                        "test_auc": 0.65 + offset,
                        "test_acc": 0.75 + offset,
                    }
                ),
                encoding="utf-8",
            )
        runs.append(
            {
                "name": f"proposed-{seed}",
                "series": "proposed_relu",
                "seed": seed,
                "feature_dim": 16384,
                "run_dir": str(run_dir),
                "results_dir": str(result_dir.parent),
                "dense_results_dir": str(dense_result_dir.parent),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    output = tmp_path / "comparison"
    generate_comparison(manifest, output)
    for name in (
        "index.html",
        "summary.md",
        "summary.json",
        "probe_summary.csv",
        "probe_summary.json",
        "probe_task_results.csv",
        "probe_task_results.json",
        "paired_task_deltas.csv",
        "representation_diagnostics.csv",
        "k_curves.svg",
        "dense_z_probe_summary.csv",
        "dense_z_probe_summary.json",
        "dense_z_probe_task_results.csv",
        "dense_z_probe_task_results.json",
    ):
        assert (output / name).is_file()
    dense_summary = json.loads((output / "dense_z_probe_summary.json").read_text())
    assert len(dense_summary) == 1
    assert dense_summary[0]["f1_mean"] == pytest.approx(0.65)
