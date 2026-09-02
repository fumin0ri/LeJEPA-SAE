from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from .calibration import calibrate
from .comparison import generate_comparison
from .config import ExperimentConfig, load_config
from .probing import run_probes
from .train import train

SERIES = (
    ("proposed_relu", "proposed", "relu"),
    ("proposed_leaky_backward", "proposed", "relu_forward_leaky_backward"),
    ("batch_topk", "batch_topk_sae", "relu"),
    ("jump_relu", "jump_relu_sae", "relu"),
    ("matryoshka_batch_topk", "matryoshka_sae", "relu"),
)
SEEDS = (42, 43, 44)


def _complete_checkpoint(run_dir: Path) -> Path | None:
    latest_path = run_dir / "latest.json"
    plan_path = run_dir / "training_plan.json"
    if not latest_path.is_file() or not plan_path.is_file():
        return None
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if int(latest["step"]) != int(plan["resolved_max_steps"]):
        return None
    checkpoint = run_dir / latest["checkpoint"]
    return checkpoint if checkpoint.is_file() else None


def _latest_checkpoint(run_dir: Path) -> Path | None:
    latest_path = run_dir / "latest.json"
    if not latest_path.is_file():
        return None
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint = run_dir / latest["checkpoint"]
    return checkpoint if checkpoint.is_file() else None


def build_run_config(
    base: ExperimentConfig,
    root: Path,
    series: str,
    model_type: str,
    activation: str,
    seed: int,
    jump_lambda: float,
) -> ExperimentConfig:
    config = copy.deepcopy(base)
    config.model.type = model_type
    config.model.feature_activation = activation
    config.train.seed = seed
    config.train.output_dir = str(root / "training" / series / f"seed-{seed}")
    config.train.resume_from = None
    config.baseline.jump_relu_lambda = jump_lambda
    config.validate()
    return config


def make_run_manifest(
    base: ExperimentConfig, root: Path, jump_lambda: float
) -> dict[str, Any]:
    runs = []
    for series, model_type, activation in SERIES:
        for seed in SEEDS:
            config = build_run_config(
                base, root, series, model_type, activation, seed, jump_lambda
            )
            run_dir = Path(config.train.output_dir)
            checkpoint = _complete_checkpoint(run_dir)
            runs.append(
                {
                    "name": f"{series}-seed-{seed}",
                    "series": series,
                    "seed": seed,
                    "model_type": model_type,
                    "feature_dim": config.model.feature_dim,
                    "run_dir": str(run_dir),
                    "config": str(run_dir / "config.resolved.yaml"),
                    "checkpoint": str(checkpoint) if checkpoint else None,
                    "results_dir": str(root / "probes" / series / f"seed-{seed}"),
                    "prefix_results_dirs": (
                        {
                            str(width): str(
                                root
                                / "matryoshka-prefix-probes"
                                / f"prefix-{width}"
                                / f"seed-{seed}"
                            )
                            for width in (512, 1536, 3584, 7680, 16384)
                        }
                        if model_type == "matryoshka_sae"
                        else {}
                    ),
                }
            )
    return {
        "model": "pythia-6.9b",
        "hook": "blocks.16.hook_resid_post",
        "seeds": list(SEEDS),
        "jump_relu_lambda": jump_lambda,
        "runs": runs,
    }


def _load_selected_lambda(calibration_path: Path) -> float:
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"JumpReLU calibration is missing: {calibration_path}. Run train mode first."
        )
    return float(json.loads(calibration_path.read_text(encoding="utf-8"))["selected_lambda"])


def run_pipeline(
    mode: str,
    config_path: str,
    root_dir: str | Path,
    model_cache_path: str | Path,
    pilot_steps: int = 20_000,
) -> Path:
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    base = load_config(config_path)
    if base.train.batch_size != 512 or base.train.gradient_accumulation_steps != 1:
        raise ValueError(
            "Primary comparison requires train.batch_size=512 and "
            "train.gradient_accumulation_steps=1 because BatchTopK is microbatch-dependent"
        )
    calibration_dir = root / "jumprelu-pilot"
    calibration_path = calibration_dir / "jumprelu_calibration.json"

    if mode in {"train", "all"}:
        if not calibration_path.is_file():
            calibrate(config_path, calibration_dir, pilot_steps=pilot_steps)
        jump_lambda = _load_selected_lambda(calibration_path)
        for series, model_type, activation in SERIES:
            for seed in SEEDS:
                config = build_run_config(
                    base, root, series, model_type, activation, seed, jump_lambda
                )
                if _complete_checkpoint(Path(config.train.output_dir)) is None:
                    partial = _latest_checkpoint(Path(config.train.output_dir))
                    config.train.resume_from = str(partial) if partial else None
                    train(config)

    jump_lambda = _load_selected_lambda(calibration_path)
    manifest = make_run_manifest(base, root, jump_lambda)
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if mode in {"probe", "all"}:
        try:
            from sae_probes import DATASETS
        except ImportError as error:
            raise RuntimeError(
                "Install probing dependencies with: pip install -e '.[probes]'"
            ) from error
        manifest["expected_tasks"] = list(DATASETS)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        missing = [run["name"] for run in manifest["runs"] if run["checkpoint"] is None]
        if missing:
            raise RuntimeError("Refusing to probe incomplete training runs: " + ", ".join(missing))
        for run in manifest["runs"]:
            if run["model_type"] in {"batch_topk_sae", "matryoshka_sae"}:
                state = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
                threshold = state["model"].get("calibrated_threshold")
                if threshold is None or not torch.isfinite(threshold):
                    raise RuntimeError(f"Missing calibrated threshold: {run['name']}")
        for index, run in enumerate(manifest["runs"]):
            config = load_config(run["config"])
            run_probes(
                config,
                run["checkpoint"],
                run["results_dir"],
                model_cache_path,
                parity=index == 0,
                raw_residual=index == 0,
            )
            for width, prefix_results in run["prefix_results_dirs"].items():
                run_probes(
                    config,
                    run["checkpoint"],
                    prefix_results,
                    model_cache_path,
                    parity=False,
                    prefix_width=int(width),
                )
        generate_comparison(manifest_path, root / "comparison")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the five-series SAE comparison")
    parser.add_argument("mode", choices=("train", "probe", "all"))
    parser.add_argument("--config", default="configs/pythia-6.9b-layer16.yaml")
    parser.add_argument(
        "--root-dir",
        default="runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/comparison-d16384-l0-160",
    )
    parser.add_argument(
        "--model-cache-path",
        default="data/sae-probes/pythia-6.9b-layer16",
    )
    parser.add_argument("--pilot-steps", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_pipeline(
        args.mode,
        args.config,
        args.root_dir,
        args.model_cache_path,
        args.pilot_steps,
    )
    print(f"Run manifest: {manifest}")


if __name__ == "__main__":
    main()
