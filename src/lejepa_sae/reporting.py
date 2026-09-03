from __future__ import annotations

# ruff: noqa: E501
import csv
import html
import json
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
    "fvu": ("FVU", "Fraction of residual activation variance left unexplained"),
    "mean_l0": ("Mean L0", "Mean active features per token"),
    "global_active_fraction": ("Global active fraction", "Strict ReLU gate: a_G > 0"),
    "local_active_fraction": ("Local active fraction", "Strict ReLU gate: a_L > 0"),
    "off_to_on": ("OFF → ON", "P(a_G ≤ 0, a_L > 0); fraction of all paired coordinates"),
    "on_to_off": ("ON → OFF", "P(a_G > 0, a_L ≤ 0); fraction of all paired coordinates"),
    "local_global_active_fraction_gap": ("Local − global active fraction", "Sparsity gap in percentage points"),
    "transition_rate_gap": ("OFF→ON − ON→OFF", "Must equal local − global active fraction, up to rounding"),
    "support_disagreement": ("Support disagreement", "OFF→ON + ON→OFF; fraction of paired coordinates that flip"),
}


def _format_metric(key: str, value: float) -> str:
    if key == "tokens":
        return f"{int(value):,}"
    if key in {"local_global_active_fraction_gap", "transition_rate_gap"}:
        return f"{100 * value:+.3f} pp"
    if "fraction" in key or key in {"support_jaccard", "off_to_on", "on_to_off", "support_disagreement"}:
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


TRAINING_HISTORY_FIELDS = [
    "kind",
    "step",
    "active_fraction",
    "expected_l0_fraction",
    "global_active_fraction",
    "local_active_fraction",
    "off_to_on",
    "on_to_off",
    "local_global_active_fraction_gap",
    "transition_rate_gap",
    "support_disagreement",
    "base_loss",
    "rate_loss",
    "global_rate_loss",
    "local_rate_loss",
    "rate_contribution",
    "rate_global_active_fraction",
    "rate_local_active_fraction",
    "rate_scale",
    "base_preactivation_grad_rms",
    "rate_preactivation_grad_rms",
    "rate_to_base_grad_ratio",
    "invariance",
    "random_distribution",
    "axis_distribution",
    "global_distribution",
    "local_distribution",
    "global_rdm_contribution",
    "local_rdm_contribution",
    "feature_std",
    "loss",
    "distribution",
    "rdm_target_scale",
    "reconstruction_contribution",
    "rdm_contribution",
    "reconstruction_preactivation_grad_rms",
    "rdm_preactivation_grad_rms",
    "rdm_to_reconstruction_grad_ratio",
    "l0",
    "reconstruction",
    "fvu",
    "auxk",
    "mean_threshold",
    "sparsity_coefficient",
    "dead_feature_fraction",
    "tracker_dead_feature_fraction",
    "global_dead_feature_fraction",
    "local_dead_feature_fraction",
]


def load_training_history(metrics_path: Path) -> list[dict[str, Any]]:
    """Load finite scalar training metrics while tolerating interrupted JSONL tails."""
    history: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        step = raw.get("step")
        if (
            not isinstance(step, int | float)
            or isinstance(step, bool)
            or not math.isfinite(float(step))
        ):
            continue
        record: dict[str, Any] = {
            "kind": str(raw.get("kind", "train")),
            "step": float(step),
        }
        for key in TRAINING_HISTORY_FIELDS[2:]:
            value = raw.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                value = float(value)
                if math.isfinite(value):
                    record[key] = value
        history.append(record)
    return sorted(history, key=lambda row: (float(row["step"]), str(row["kind"])))


def _write_training_history_csv(
    output_path: Path, history: list[dict[str, Any]]
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_HISTORY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def _training_series(
    history: list[dict[str, Any]], key: str, kind: str
) -> list[tuple[float, float]]:
    return [
        (float(row["step"]), float(row[key]))
        for row in history
        if row.get("kind") == kind and key in row
    ]


def _chart_coordinate(
    step: float,
    value: float,
    *,
    log_scale: bool,
    min_step: float,
    max_step: float,
    min_value: float,
    max_value: float,
    x0: float,
    y0: float,
    plot_width: float,
    plot_height: float,
) -> tuple[float, float] | None:
    if log_scale and value <= 0:
        return None
    scaled_value = math.log10(value) if log_scale else value
    x = x0 + (step - min_step) / (max_step - min_step) * plot_width
    y = y0 + plot_height - (scaled_value - min_value) / (max_value - min_value) * plot_height
    return x, y


def write_training_curves_svg(
    output_path: Path, history: list[dict[str, Any]]
) -> None:
    panels = [
        (
            "Active fraction",
            False,
            [
                ("active_fraction", "train", "Train", "#4f46e5"),
                ("active_fraction", "validation", "Validation", "#db2777"),
                ("expected_l0_fraction", "train", "Target L0", "#64748b"),
                ("global_active_fraction", "train", "Global", "#059669"),
                ("local_active_fraction", "train", "Local", "#d97706"),
            ],
        ),
        (
            "Global-local MSE",
            True,
            [
                ("invariance", "train", "Train", "#4f46e5"),
                ("invariance", "validation", "Validation", "#db2777"),
            ],
        ),
        (
            "Random vs axis RDMReg",
            True,
            [
                ("random_distribution", "train", "Random train", "#2563eb"),
                ("axis_distribution", "train", "Axis train", "#ea580c"),
                ("random_distribution", "validation", "Random val", "#0891b2"),
                ("axis_distribution", "validation", "Axis val", "#be123c"),
            ],
        ),
        (
            "Global vs local RDMReg contribution",
            True,
            [
                ("global_rdm_contribution", "train", "Global train", "#2563eb"),
                ("local_rdm_contribution", "train", "Local train", "#ea580c"),
                ("global_rdm_contribution", "validation", "Global val", "#0891b2"),
                ("local_rdm_contribution", "validation", "Local val", "#be123c"),
            ],
        ),
        (
            "Feature standard deviation",
            False,
            [
                ("feature_std", "train", "Train", "#4f46e5"),
                ("feature_std", "validation", "Validation", "#db2777"),
            ],
        ),
        (
            "Reconstruction and FVU",
            True,
            [
                ("reconstruction", "train", "MSE train", "#4f46e5"),
                ("reconstruction", "validation", "MSE val", "#db2777"),
                ("fvu", "train", "FVU train", "#059669"),
                ("fvu", "validation", "FVU val", "#d97706"),
            ],
        ),
        (
            "Gate transitions (%)",
            False,
            [
                ("off_to_on", "train", "OFF→ON train", "#2563eb"),
                ("on_to_off", "train", "ON→OFF train", "#ea580c"),
                ("off_to_on", "validation", "OFF→ON val", "#0891b2"),
                ("on_to_off", "validation", "ON→OFF val", "#be123c"),
            ],
        ),
        (
            "Sparsity gap (percentage points)",
            False,
            [
                ("local_global_active_fraction_gap", "train", "L−G train", "#2563eb"),
                ("transition_rate_gap", "train", "Flow diff train", "#ea580c"),
                ("local_global_active_fraction_gap", "validation", "L−G val", "#0891b2"),
                ("transition_rate_gap", "validation", "Flow diff val", "#be123c"),
            ],
        ),
    ]
    if any("rate_loss" in row for row in history):
        panels.extend([
            (
                "Target-rate penalty",
                True,
                [
                    ("rate_loss", "train", "Raw train", "#2563eb"),
                    ("rate_loss", "validation", "Raw val", "#0891b2"),
                    ("rate_contribution", "train", "Weighted tr", "#ea580c"),
                    ("rate_contribution", "validation", "Weighted val", "#be123c"),
                ],
            ),
            (
                "Rate / base preactivation gradient RMS",
                False,
                [("rate_to_base_grad_ratio", "train", "Rate / base", "#2563eb")],
            ),
        ])
    if any("reconstruction_contribution" in row for row in history):
        panels.extend([
            (
                "Weighted reconstruction vs RDMReg",
                True,
                [
                    ("reconstruction_contribution", "train", "Recon train", "#2563eb"),
                    ("rdm_contribution", "train", "RDM train", "#ea580c"),
                    ("reconstruction_contribution", "validation", "Recon val", "#0891b2"),
                    ("rdm_contribution", "validation", "RDM val", "#be123c"),
                ],
            ),
            (
                "Weighted preactivation gradient RMS",
                True,
                [
                    ("reconstruction_preactivation_grad_rms", "train", "Recon", "#2563eb"),
                    ("rdm_preactivation_grad_rms", "train", "RDM", "#ea580c"),
                ],
            ),
            (
                "RDM / reconstruction gradient RMS",
                False,
                [("rdm_to_reconstruction_grad_ratio", "train", "RDM / recon", "#2563eb")],
            ),
        ])
        # This model has no local branch: do not show empty JEPA-only panels.
        panels = [
            panel for panel in panels
            if any(key in row for key, *_ in panel[2] for row in history)
        ]
    width, height = 1200, 40 + 360 * math.ceil(len(panels) / 2)
    panel_width, panel_height = 550, 300
    plot_left, plot_top, plot_width, plot_height = 62, 58, 458, 190
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}.title{font-size:18px;font-weight:700}'
        '.tick{font-size:11px;fill:#64748b}.legend{font-size:11px;fill:#334155}</style>',
    ]
    for panel_index, (title, log_scale, definitions) in enumerate(panels):
        panel_x = 30 + (panel_index % 2) * 585
        panel_y = 25 + (panel_index // 2) * 360
        x0, y0 = panel_x + plot_left, panel_y + plot_top
        series = [
            (label, color, kind, _training_series(history, key, kind))
            for key, kind, label, color in definitions
        ]
        series = [item for item in series if item[3]]
        points = [point for _, _, _, values in series for point in values]
        pieces.extend(
            [
                f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" rx="14" fill="#fff" stroke="#dce3ee"/>',
                f'<text class="title" x="{panel_x + 22}" y="{panel_y + 32}">{html.escape(title)}</text>',
            ]
        )
        if not points:
            pieces.append(
                f'<text class="tick" x="{x0}" y="{y0 + 90}">No recorded values</text>'
            )
            continue
        min_step, max_step = min(p[0] for p in points), max(p[0] for p in points)
        if max_step <= min_step:
            max_step = min_step + 1.0
        transformed = [
            math.log10(value) if log_scale and value > 0 else value
            for _, value in points
            if not log_scale or value > 0
        ]
        if not transformed:
            pieces.append(
                f'<text class="tick" x="{x0}" y="{y0 + 90}">No positive finite values</text>'
            )
            continue
        min_value, max_value = min(transformed), max(transformed)
        percent_axis = title in {"Gate transitions (%)", "Sparsity gap (percentage points)"}
        min_padding = 0.001 if percent_axis else (0.02 if not log_scale else 0.08)
        padding = max((max_value - min_value) * 0.08, min_padding)
        min_value -= padding
        max_value += padding

        for fraction in (0.0, 0.5, 1.0):
            y = y0 + plot_height * (1 - fraction)
            raw_value = min_value + (max_value - min_value) * fraction
            label = f"{10 ** raw_value:.2g}" if log_scale else f"{raw_value:.3g}"
            if percent_axis:
                label = f"{100 * raw_value:.3g}"
            pieces.extend(
                [
                    f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + plot_width}" y2="{y:.2f}" stroke="#e2e8f0"/>',
                    f'<text class="tick" text-anchor="end" x="{x0 - 8}" y="{y + 4:.2f}">{label}</text>',
                ]
            )
        pieces.extend(
            [
                f'<text class="tick" x="{x0}" y="{y0 + plot_height + 20}">{min_step:g}</text>',
                f'<text class="tick" text-anchor="end" x="{x0 + plot_width}" y="{y0 + plot_height + 20}">{max_step:g} steps</text>',
            ]
        )
        legend_x = panel_x + 22
        for series_index, (label, color, kind, values) in enumerate(series):
            dash = ' stroke-dasharray="7 5"' if kind == "validation" else ""
            valid = [
                _chart_coordinate(
                    step,
                    value,
                    log_scale=log_scale,
                    min_step=min_step,
                    max_step=max_step,
                    min_value=min_value,
                    max_value=max_value,
                    x0=x0,
                    y0=y0,
                    plot_width=plot_width,
                    plot_height=plot_height,
                )
                for step, value in values
            ]
            valid = [point for point in valid if point is not None]
            if not valid:
                continue
            polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in valid)
            pieces.append(
                f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>'
            )
            for x, y in valid:
                pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.7" fill="{color}"/>')
            legend_y = panel_y + 272 + (series_index // 4) * 17
            item_x = legend_x + (series_index % 4) * 124
            pieces.extend(
                [
                    f'<line x1="{item_x}" y1="{legend_y - 4}" x2="{item_x + 18}" y2="{legend_y - 4}" stroke="{color}" stroke-width="2.5"{dash}/>',
                    f'<text class="legend" x="{item_x + 23}" y="{legend_y}">{html.escape(label)}</text>',
                ]
            )
    pieces.append("</svg>")
    output_path.write_text("".join(pieces), encoding="utf-8")


def write_evaluation_report(
    output: Path,
    metrics: dict[str, float],
    feature_rows: list[dict[str, float | int]],
    top_records: list[dict[str, Any]],
    training_history: list[dict[str, Any]] | None = None,
) -> None:
    with (output / "feature_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature", "active_fraction", "mean", "std", "maximum"],
        )
        writer.writeheader()
        writer.writerows(feature_rows)

    write_feature_diagnostics_svg(output / "feature_diagnostics.svg", feature_rows)
    training_history = training_history or []
    if training_history:
        _write_training_history_csv(output / "training_history.csv", training_history)
        write_training_curves_svg(output / "training_curves.svg", training_history)
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
    if training_history:
        markdown.extend(
            [
                "",
                "## Training curves",
                "",
                "![Training curves](training_curves.svg)",
            ]
        )
    raw_artifacts = ["`metrics.json`", "`feature_metrics.csv`"]
    if training_history:
        raw_artifacts.append("`training_history.csv`")
    raw_artifacts.append("`top_tokens.jsonl`")
    markdown.extend(
        [
            "",
            "## Feature distributions",
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
            f'{", ".join(raw_artifacts[:-1])}, and {raw_artifacts[-1]}.',
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

    training_section = (
        '<h2>Training curves</h2><p>Train and validation history from metrics.jsonl. '
        'Global-local MSE and RDMReg use logarithmic scales. '
        'Gate transitions use all paired coordinates as their denominator, not just initially active/inactive ones. '
        'OFF→ON − ON→OFF equals local − global active fraction; the gap curves should overlap.</p>'
        '<img class="chart" src="training_curves.svg" alt="Active fraction, global-local MSE, RDMReg, feature standard deviation, and gate transitions over training">'
        if training_history
        else '<h2>Training curves</h2><p class="card-help">No training metrics history was found. Pass <code>--training-metrics</code> to include it.</p>'
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
</style></head><body><main><h1>Single-token sparse representation evaluation</h1>
<div class="status {status_class}">{html.escape(status)}</div>
<section class="cards">{"".join(cards)}</section>{training_section}<h2>Feature distributions</h2>
<img class="chart" src="feature_diagnostics.svg" alt="Feature diagnostic histograms">
<h2>Highest-variance features</h2><p>Showing 50 features. Search decoded top activations.</p>
<input id="search" type="search" placeholder="Search feature number or activation text">
<section id="features">{"".join(feature_sections)}</section></main>
<script>const q=document.querySelector('#search');q.addEventListener('input',()=>{{const s=q.value.toLowerCase();document.querySelectorAll('details').forEach(e=>e.hidden=!e.dataset.search.toLowerCase().includes(s))}});</script>
</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
