from __future__ import annotations

# ruff: noqa: E501
import csv
import html
import math
from pathlib import Path
from typing import Any

METRIC_LABELS = {
    "tokens": ("Evaluated tokens", "Number of single-token samples evaluated"),
    "mean_active_fraction": ("Active fraction", "Mean fraction of active latent features"),
    "dead_feature_fraction": ("Dead features", "Features never active in this evaluation"),
    "mean_feature_std": ("Feature std", "Mean per-feature standard deviation"),
    "global_local_mse": ("Global-local MSE", "Invariance error; lower is better"),
    "support_jaccard": ("Support Jaccard", "Global/local active-set overlap; higher is better"),
    "full_reconstruction_mse": ("Full reconstruction MSE", "Full-input SAE reconstruction"),
    "masked_reconstruction_mse": (
        "Masked reconstruction MSE",
        "Denoising SAE reconstruction from masked input",
    ),
}


def _format_metric(key: str, value: float) -> str:
    if key == "tokens":
        return f"{int(value):,}"
    if "fraction" in key or key == "support_jaccard":
        return f"{value:.2%}"
    return f"{value:.6g}"


def collapse_assessment(metrics: dict[str, float]) -> tuple[str, str]:
    collapsed = (
        metrics["mean_active_fraction"] <= 1e-5
        or metrics["mean_feature_std"] <= 1e-8
        or metrics["dead_feature_fraction"] >= 0.99
    )
    if collapsed:
        return "Possible collapse", "warning"
    return "Non-collapse checks passed", "healthy"


def _histogram(values: list[float], bins: int, fixed_max: float | None = None):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [0] * bins, 0.0, 1.0
    lower = 0.0 if min(finite) >= 0 else min(finite)
    upper = fixed_max if fixed_max is not None else max(finite)
    if upper <= lower:
        upper = lower + 1.0
    counts = [0] * bins
    for value in finite:
        position = min(bins - 1, max(0, int((value - lower) / (upper - lower) * bins)))
        counts[position] += 1
    return counts, lower, upper


def write_feature_diagnostics_svg(
    output_path: Path,
    feature_rows: list[dict[str, float | int]],
) -> None:
    panels = [
        ("Active-rate distribution", "active_fraction", 1.0),
        ("Feature standard deviation", "std", None),
        ("Maximum activation", "maximum", None),
    ]
    width, height = 1200, 420
    panel_width = 360
    chart_height = 260
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}'
        '.title{font-size:18px;font-weight:700}.tick{font-size:12px;fill:#596579}</style>',
    ]
    for panel_index, (title, key, fixed_max) in enumerate(panels):
        origin_x = 30 + panel_index * 390
        origin_y = 75
        counts, lower, upper = _histogram(
            [float(row[key]) for row in feature_rows], 24, fixed_max
        )
        tallest = max(max(counts), 1)
        bar_width = panel_width / len(counts)
        pieces.append(
            f'<text class="title" x="{origin_x}" y="38">{html.escape(title)}</text>'
        )
        pieces.append(
            f'<line x1="{origin_x}" y1="{origin_y + chart_height}" '
            f'x2="{origin_x + panel_width}" y2="{origin_y + chart_height}" '
            'stroke="#94a3b8"/>'
        )
        for index, count in enumerate(counts):
            bar_height = chart_height * count / tallest
            x = origin_x + index * bar_width + 1
            y = origin_y + chart_height - bar_height
            pieces.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 2, 1):.2f}" '
                f'height="{bar_height:.2f}" rx="2" fill="#5b6df9"/>'
            )
        pieces.extend(
            [
                f'<text class="tick" x="{origin_x}" y="{origin_y + chart_height + 24}">'
                f'{lower:.3g}</text>',
                f'<text class="tick" text-anchor="end" x="{origin_x + panel_width}" '
                f'y="{origin_y + chart_height + 24}">{upper:.3g}</text>',
                f'<text class="tick" x="{origin_x}" y="{origin_y - 10}">'
                f'max bin: {tallest:,} features</text>',
            ]
        )
    pieces.append("</svg>")
    output_path.write_text("".join(pieces), encoding="utf-8")


def write_evaluation_report(
    output: Path,
    metrics: dict[str, float],
    feature_rows: list[dict[str, float | int]],
    top_records: list[dict[str, Any]],
) -> None:
    with (output / "feature_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature", "active_fraction", "mean", "std", "maximum"],
        )
        writer.writeheader()
        writer.writerows(feature_rows)

    write_feature_diagnostics_svg(output / "feature_diagnostics.svg", feature_rows)
    status, status_class = collapse_assessment(metrics)
    ranked = sorted(feature_rows, key=lambda row: float(row["std"]), reverse=True)
    record_by_feature = {int(record["feature"]): record for record in top_records}

    markdown = [
        "# Evaluation summary",
        "",
        f"**Status:** {status}",
        "",
        "## Core metrics",
        "",
        "| Metric | Value | Meaning |",
        "|---|---:|---|",
    ]
    for key, value in metrics.items():
        label, description = METRIC_LABELS.get(key, (key.replace("_", " ").title(), ""))
        markdown.append(f"| {label} | {_format_metric(key, value)} | {description} |")
    markdown.extend(
        [
            "",
            "![Feature diagnostics](feature_diagnostics.svg)",
            "",
            "## Highest-variance features",
            "",
            "| Feature | Active | Std | Max | Top example |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in ranked[:20]:
        record = record_by_feature[int(row["feature"])]
        examples = record.get("examples", [])
        text = str(examples[0]["text"]).replace("|", "\\|").replace("\n", " ") if examples else ""
        markdown.append(
            f'| {row["feature"]} | {float(row["active_fraction"]):.2%} | '
            f'{float(row["std"]):.4g} | {float(row["maximum"]):.4g} | {text[:120]} |'
        )
    markdown.extend(
        [
            "",
            "Open `index.html` for the interactive, searchable feature report. Raw data remains in",
            "`metrics.json`, `feature_metrics.csv`, and `top_tokens.jsonl`.",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(markdown), encoding="utf-8")

    cards = []
    for key, value in metrics.items():
        label, description = METRIC_LABELS.get(key, (key.replace("_", " ").title(), ""))
        cards.append(
            '<article class="card">'
            f'<div class="card-label">{html.escape(label)}</div>'
            f'<div class="card-value">{html.escape(_format_metric(key, value))}</div>'
            f'<div class="card-help">{html.escape(description)}</div></article>'
        )

    feature_sections = []
    for row in ranked[:50]:
        feature = int(row["feature"])
        examples = record_by_feature[feature].get("examples", [])[:5]
        example_html = "".join(
            '<li><span class="score">'
            f'{float(example["score"]):.4g}</span> '
            f'{html.escape(str(example["text"]))}'
            f'<small>{html.escape(str(example["document_id"]))}</small></li>'
            for example in examples
        )
        feature_sections.append(
            f'<details data-search="feature {feature} {html.escape(" ".join(str(item["text"]) for item in examples))}">'
            f'<summary>Feature {feature} <span>active {float(row["active_fraction"]):.2%} '
            f'· std {float(row["std"]):.4g} · max {float(row["maximum"]):.4g}</span></summary>'
            f'<ol>{example_html}</ol></details>'
        )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>LeJEPA-SAE evaluation</title><style>
:root{{--ink:#172033;--muted:#64748b;--panel:#fff;--line:#dce3ee;--accent:#5b6df9}}
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fb;color:var(--ink);font:15px/1.55 Inter,Arial,sans-serif}}
main{{max-width:1180px;margin:auto;padding:42px 24px 80px}}h1{{font-size:34px;margin:0 0 8px}}h2{{margin-top:38px}}
.status{{display:inline-block;padding:7px 12px;border-radius:999px;font-weight:700}}
.healthy{{background:#dcfce7;color:#166534}}.warning{{background:#fee2e2;color:#991b1b}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:24px 0}}
.card,details,.chart{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 24px #1720330d}}
.card{{padding:18px}}.card-label{{color:var(--muted);font-weight:700}}.card-value{{font-size:27px;font-weight:800;margin:5px 0}}
.card-help,small{{color:var(--muted)}}.chart{{width:100%;padding:12px}}input{{width:100%;padding:13px 15px;border:1px solid var(--line);border-radius:10px;font:inherit;margin-bottom:12px}}
details{{margin:9px 0;padding:13px 16px}}summary{{cursor:pointer;font-weight:800}}summary span{{float:right;color:var(--muted);font-weight:500}}
li{{margin:10px 0}}.score{{display:inline-block;min-width:68px;color:var(--accent);font-weight:800}}small{{display:block;margin-left:72px}}
</style></head><body><main><h1>Single-token JEPA evaluation</h1>
<div class="status {status_class}">{html.escape(status)}</div>
<section class="cards">{"".join(cards)}</section><h2>Feature distributions</h2>
<img class="chart" src="feature_diagnostics.svg" alt="Feature diagnostic histograms">
<h2>Highest-variance features</h2><p>Showing 50 features. Search decoded top activations.</p>
<input id="search" type="search" placeholder="Search feature number or activation text">
<section id="features">{"".join(feature_sections)}</section></main>
<script>const q=document.querySelector('#search');q.addEventListener('input',()=>{{const s=q.value.toLowerCase();document.querySelectorAll('details').forEach(e=>e.hidden=!e.dataset.search.toLowerCase().includes(s))}});</script>
</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
