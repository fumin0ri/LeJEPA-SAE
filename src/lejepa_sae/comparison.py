from __future__ import annotations

# ruff: noqa: E501
import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
METRIC_ALIASES = {
    "f1": ("f1", "f1_score", "test_f1"),
    "auroc": ("auroc", "auc", "roc_auc", "test_auc"),
    "accuracy": ("accuracy", "acc", "test_acc", "test_accuracy"),
}


def _canonical_metric(record: dict[str, Any], name: str) -> float:
    for key in METRIC_ALIASES[name]:
        value = record.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    raise ValueError(f"Probe result has no {name} metric (accepted: {METRIC_ALIASES[name]})")


def load_probe_records(results_dir: str | Path) -> list[dict[str, Any]]:
    records = []
    for path in Path(results_dir).rglob("*.json"):
        if path.name == "hook_parity.json":
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict) or "dataset" not in item or "k" not in item:
                continue
            records.append(
                {
                    "task": str(item["dataset"]),
                    "k": int(item["k"]),
                    **{name: _canonical_metric(item, name) for name in METRIC_ALIASES},
                }
            )
    return records


def load_raw_residual_records(results_dir: str | Path) -> list[dict[str, Any]]:
    records = []
    for path in Path(results_dir).rglob("*.json"):
        if "baseline_results_" not in str(path):
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, dict) and "dataset" in item:
                records.append(
                    {
                        "task": str(item["dataset"]),
                        **{name: _canonical_metric(item, name) for name in METRIC_ALIASES},
                    }
                )
    return records


def load_dense_probe_records(results_dir: str | Path) -> list[dict[str, Any]]:
    records = []
    for path in Path(results_dir).rglob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or raw.get("method") != "dense_z_gpu_logistic_regression"
            or "dataset" not in raw
        ):
            continue
        try:
            record = {
                "task": str(raw["dataset"]),
                "feature_dim": int(raw["feature_dim"]),
                "best_c": float(raw["best_c"]),
                **{name: _canonical_metric(raw, name) for name in METRIC_ALIASES},
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid dense z probe result: {path}") from error
        numeric = [record["best_c"], *(record[name] for name in METRIC_ALIASES)]
        if record["feature_dim"] < 1 or record["best_c"] <= 0 or not all(
            math.isfinite(value) for value in numeric
        ):
            raise ValueError(f"Invalid dense z probe result: {path}")
        records.append(record)
    return records


def validate_complete_tasks(
    records_by_run: dict[str, list[dict[str, Any]]],
    expected_ks: list[int] = KS,
    expected_tasks: list[str] | None = None,
) -> list[str]:
    if not records_by_run:
        raise ValueError("No probe runs were supplied")
    reference_name = next(iter(records_by_run))
    reference = {(row["task"], row["k"]) for row in records_by_run[reference_name]}
    if not reference:
        raise ValueError(f"No probe results found for {reference_name}")
    reference_tasks = sorted(expected_tasks or {task for task, _ in reference})
    expected = {(task, k) for task in reference_tasks for k in expected_ks}
    problems = []
    for name, records in records_by_run.items():
        observed = {(row["task"], row["k"]) for row in records}
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        duplicate_count = len(records) - len(observed)
        if missing or extra or duplicate_count:
            problems.append(
                f"{name}: missing={missing}, extra={extra}, duplicates={duplicate_count}"
            )
    if problems:
        raise ValueError("Incomplete or mismatched probe task sets:\n" + "\n".join(problems))
    return reference_tasks


def validate_complete_dense_tasks(
    records_by_run: dict[str, list[dict[str, Any]]],
    expected_tasks: list[str],
) -> None:
    expected = set(expected_tasks)
    problems = []
    for name, records in records_by_run.items():
        observed = {row["task"] for row in records}
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        duplicate_count = len(records) - len(observed)
        if missing or extra or duplicate_count:
            problems.append(
                f"{name}: missing={missing}, extra={extra}, duplicates={duplicate_count}"
            )
    if problems:
        raise ValueError(
            "Incomplete or mismatched dense z probe task sets:\n" + "\n".join(problems)
        )


def aggregate_probe_runs(
    manifest_runs: list[dict[str, Any]], records_by_run: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed = []
    task_rows = []
    for run in manifest_runs:
        name = str(run["name"])
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in records_by_run[name]:
            grouped[row["k"]].append(row)
            task_rows.append({"series": run["series"], "seed": run["seed"], **row})
        for k, rows in sorted(grouped.items()):
            per_seed.append(
                {
                    "series": run["series"],
                    "seed": int(run["seed"]),
                    "k": k,
                    **{
                        metric: mean(float(row[metric]) for row in rows)
                        for metric in METRIC_ALIASES
                    },
                }
            )
    summary = []
    grouped_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        grouped_seed[(row["series"], row["k"])].append(row)
    for (series, k), rows in sorted(grouped_seed.items()):
        record: dict[str, Any] = {"series": series, "k": k, "seeds": len(rows)}
        for metric in METRIC_ALIASES:
            values = [float(row[metric]) for row in rows]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(record)
    return summary, task_rows


def aggregate_dense_probe_runs(
    manifest_runs: list[dict[str, Any]], records_by_run: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed = []
    task_rows = []
    for run in manifest_runs:
        rows = records_by_run[str(run["name"])]
        expected_width = run.get("feature_dim")
        if expected_width is not None and any(
            row["feature_dim"] != int(expected_width) for row in rows
        ):
            raise ValueError(f"Dense z probe feature width differs for {run['name']}")
        task_rows.extend(
            {"series": run["series"], "seed": int(run["seed"]), **row} for row in rows
        )
        per_seed.append(
            {
                "series": run["series"],
                "seed": int(run["seed"]),
                **{
                    metric: mean(float(row[metric]) for row in rows)
                    for metric in METRIC_ALIASES
                },
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        grouped[str(row["series"])].append(row)
    summary = []
    for series, rows in sorted(grouped.items()):
        record: dict[str, Any] = {"series": series, "seeds": len(rows)}
        for metric in METRIC_ALIASES:
            values = [float(row[metric]) for row in rows]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(record)
    return summary, task_rows


def _last_validation(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "metrics.jsonl"
    latest: dict[str, Any] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "validation":
                latest = row
    return latest


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_k_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    series_names = sorted({str(row["series"]) for row in rows})
    colors = ["#4f46e5", "#db2777", "#059669", "#d97706", "#0891b2", "#7c3aed"]
    width, height = 1050, 520
    x0, y0, plot_width, plot_height = 85, 55, 900, 370
    values = [float(row["f1_mean"]) for row in rows]
    low, high = min(values), max(values)
    pad = max((high - low) * 0.1, 0.02)
    low, high = max(0.0, low - pad), min(1.0, high + pad)
    if high <= low:
        high = low + 0.1
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}.t{font-size:20px;font-weight:700}.s{font-size:12px;fill:#64748b}</style>',
        '<text class="t" x="35" y="30">k-sparse probing: macro F1</text>',
    ]
    for index, k in enumerate(KS):
        x = x0 + index / (len(KS) - 1) * plot_width
        pieces.append(f'<text class="s" text-anchor="middle" x="{x}" y="{y0 + plot_height + 24}">{k}</text>')
    for fraction in (0.0, 0.5, 1.0):
        y = y0 + plot_height * (1 - fraction)
        value = low + fraction * (high - low)
        pieces.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + plot_width}" y2="{y}" stroke="#e2e8f0"/>')
        pieces.append(f'<text class="s" text-anchor="end" x="{x0 - 8}" y="{y + 4}">{value:.3f}</text>')
    for series_index, series in enumerate(series_names):
        color = colors[series_index % len(colors)]
        selected = {int(row["k"]): row for row in rows if row["series"] == series}
        points = []
        for index, k in enumerate(KS):
            if k not in selected:
                continue
            x = x0 + index / (len(KS) - 1) * plot_width
            y = y0 + plot_height * (1 - (float(selected[k]["f1_mean"]) - low) / (high - low))
            points.append((x, y))
        pieces.append('<polyline fill="none" stroke="{}" stroke-width="2.5" points="{}"/>'.format(color, " ".join(f"{x:.1f},{y:.1f}" for x, y in points)))
        pieces.append(f'<line x1="{x0 + series_index * 175}" y1="480" x2="{x0 + 20 + series_index * 175}" y2="480" stroke="{color}" stroke-width="3"/>')
        pieces.append(f'<text class="s" x="{x0 + 25 + series_index * 175}" y="484">{html.escape(series)}</text>')
    pieces.append("</svg>")
    path.write_text("".join(pieces), encoding="utf-8")


def generate_comparison(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    records_by_run = {
        str(run["name"]): load_probe_records(run["results_dir"]) for run in runs
    }
    tasks = validate_complete_tasks(
        records_by_run, expected_tasks=manifest.get("expected_tasks")
    )
    summary_rows, task_rows = aggregate_probe_runs(runs, records_by_run)
    dense_summary = None
    dense_task_rows = None
    dense_paths = [run.get("dense_results_dir") for run in runs]
    if any(dense_paths):
        if not all(dense_paths):
            raise ValueError("Dense z probe results must be declared for every comparison run")
        dense_records = {
            str(run["name"]): load_dense_probe_records(run["dense_results_dir"])
            for run in runs
        }
        validate_complete_dense_tasks(dense_records, tasks)
        dense_summary, dense_task_rows = aggregate_dense_probe_runs(runs, dense_records)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "probe_summary.csv", summary_rows)
    _write_csv(output / "probe_task_results.csv", task_rows)
    (output / "probe_summary.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )
    (output / "probe_task_results.json").write_text(
        json.dumps(task_rows, indent=2), encoding="utf-8"
    )
    _write_k_curves(output / "k_curves.svg", summary_rows)

    if dense_summary is not None and dense_task_rows is not None:
        _write_csv(output / "dense_z_probe_summary.csv", dense_summary)
        _write_csv(output / "dense_z_probe_task_results.csv", dense_task_rows)
        (output / "dense_z_probe_summary.json").write_text(
            json.dumps(dense_summary, indent=2), encoding="utf-8"
        )
        (output / "dense_z_probe_task_results.json").write_text(
            json.dumps(dense_task_rows, indent=2), encoding="utf-8"
        )

    raw_records = load_raw_residual_records(runs[0]["results_dir"])
    raw_summary = None
    if raw_records:
        raw_tasks = {row["task"] for row in raw_records}
        if raw_tasks != set(tasks):
            raise ValueError(
                "Raw residual reference task set does not match SAE runs: "
                f"missing={sorted(set(tasks) - raw_tasks)}, extra={sorted(raw_tasks - set(tasks))}"
            )
        raw_summary = {
            metric: mean(float(row[metric]) for row in raw_records)
            for metric in METRIC_ALIASES
        }
        _write_csv(output / "raw_residual_reference.csv", raw_records)

    diagnostic_rows = []
    for run in runs:
        validation = _last_validation(run["run_dir"])
        measured_l0 = validation.get("l0")
        if measured_l0 is None and "active_fraction" in validation:
            measured_l0 = float(validation["active_fraction"]) * int(run["feature_dim"])
        diagnostic_rows.append(
            {
                "series": run["series"],
                "seed": run["seed"],
                "l0": measured_l0 if measured_l0 is not None else math.nan,
                "fvu": validation.get("fvu", math.nan),
                "dead_feature_fraction": validation.get(
                    "tracker_dead_feature_fraction",
                    validation.get(
                        "dead_feature_fraction",
                        validation.get("global_dead_feature_fraction", math.nan),
                    ),
                ),
            }
        )
    _write_csv(output / "representation_diagnostics.csv", diagnostic_rows)

    prefix_runs = []
    prefix_records: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for width, results_dir in run.get("prefix_results_dirs", {}).items():
            name = f"matryoshka-prefix-{width}-seed-{run['seed']}"
            prefix_runs.append(
                {
                    "name": name,
                    "series": f"matryoshka_prefix_{width}",
                    "seed": run["seed"],
                }
            )
            prefix_records[name] = load_probe_records(results_dir)
    if prefix_runs:
        validate_complete_tasks(prefix_records, expected_tasks=tasks)
        prefix_summary, _ = aggregate_probe_runs(prefix_runs, prefix_records)
        _write_csv(output / "matryoshka_prefix_probe_summary.csv", prefix_summary)

    proposed = {(row["seed"], row["task"], row["k"]): row for row in task_rows if row["series"] == "proposed_relu"}
    paired = []
    for row in task_rows:
        reference = proposed.get((row["seed"], row["task"], row["k"]))
        if reference is not None:
            paired.append(
                {
                    **row,
                    **{f"delta_{metric}": row[metric] - reference[metric] for metric in METRIC_ALIASES},
                }
            )
    _write_csv(output / "paired_task_deltas.csv", paired)
    headline = [row for row in summary_rows if row["k"] in {1, 16}]
    result = {
        "tasks": tasks,
        "runs": len(runs),
        "headline": headline,
        "dense_z_gpu": dense_summary,
        "raw_residual_reference": raw_summary,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    markdown = [
        "# SAE comparison",
        "",
        f"Complete official `normal` sae-probes tasks: {len(tasks)}; runs: {len(runs)}.",
        "",
        "## Headline macro F1",
        "",
        "| Series | k | F1 mean | F1 std | AUROC | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in headline:
        markdown.append(
            f'| {row["series"]} | {row["k"]} | {row["f1_mean"]:.4f} | {row["f1_std"]:.4f} | {row["auroc_mean"]:.4f} | {row["accuracy_mean"]:.4f} |'
        )
    if dense_summary is not None:
        markdown.extend(
            [
                "",
                "## Full-z GPU L2 logistic probe",
                "",
                "All SAE coordinates are used; no top-k feature selection or input normalization.",
                "",
                "| Series | F1 mean | F1 std | AUROC | Accuracy |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in dense_summary:
            markdown.append(
                f'| {row["series"]} | {row["f1_mean"]:.4f} | {row["f1_std"]:.4f} | {row["auroc_mean"]:.4f} | {row["accuracy_mean"]:.4f} |'
            )
    if raw_summary is not None:
        markdown.extend(
            [
                "",
                "## Raw residual dense logistic reference",
                "",
                "This is a reference ceiling and is not a k-sparse SAE result.",
                "",
                f'F1 `{raw_summary["f1"]:.4f}` · AUROC `{raw_summary["auroc"]:.4f}` · accuracy `{raw_summary["accuracy"]:.4f}`',
            ]
        )
    markdown.extend(["", "![All-k macro F1 curves](k_curves.svg)", ""])
    (output / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    rows_html = "".join(
        f'<tr><td>{html.escape(str(row["series"]))}</td><td>{row["k"]}</td><td>{row["f1_mean"]:.4f} ± {row["f1_std"]:.4f}</td><td>{row["auroc_mean"]:.4f}</td><td>{row["accuracy_mean"]:.4f}</td></tr>'
        for row in headline
    )
    dense_html = ""
    if dense_summary is not None:
        dense_rows_html = "".join(
            f'<tr><td>{html.escape(str(row["series"]))}</td><td>{row["f1_mean"]:.4f} ± {row["f1_std"]:.4f}</td><td>{row["auroc_mean"]:.4f}</td><td>{row["accuracy_mean"]:.4f}</td></tr>'
            for row in dense_summary
        )
        dense_html = (
            "<h2>Full-z GPU L2 logistic probe</h2>"
            "<p>All SAE coordinates; no top-k selection or input normalization.</p>"
            "<table><thead><tr><th>Series</th><th>Macro F1</th><th>AUROC</th>"
            f"<th>Accuracy</th></tr></thead><tbody>{dense_rows_html}</tbody></table>"
        )
    raw_html = (
        f'<h2>Raw residual reference</h2><p>Dense logistic probe: F1 {raw_summary["f1"]:.4f}, AUROC {raw_summary["auroc"]:.4f}, accuracy {raw_summary["accuracy"]:.4f}. This is a reference ceiling, not a k-sparse SAE result.</p>'
        if raw_summary is not None
        else ""
    )
    (output / "index.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>SAE comparison</title><style>body{{max-width:1100px;margin:40px auto;font:15px Inter,Arial;color:#172033}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%;margin-top:30px}}</style><h1>SAE comparison</h1><p>{len(tasks)} complete tasks · {len(runs)} runs · mean ± standard deviation over seeds</p><table><thead><tr><th>Series</th><th>k</th><th>Macro F1</th><th>AUROC</th><th>Accuracy</th></tr></thead><tbody>{rows_html}</tbody></table>{dense_html}{raw_html}<img src="k_curves.svg" alt="All-k macro F1 curves">',
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate complete sae-probes comparison runs")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(generate_comparison(args.manifest, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
