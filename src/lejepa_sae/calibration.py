from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .config import load_config
from .train import train

DEFAULT_LAMBDAS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def choose_jumprelu_lambda(
    results: list[dict[str, float]], target_l0: float
) -> dict[str, float]:
    if not results:
        raise ValueError("No JumpReLU calibration results")
    required = {"lambda", "l0", "fvu"}
    if any(not required <= result.keys() for result in results):
        raise ValueError("Each calibration result requires lambda, l0, and fvu")
    return min(results, key=lambda result: (abs(result["l0"] - target_l0), result["fvu"]))


def next_grid_lambda(
    results: list[dict[str, float]], target_l0: float
) -> float | None:
    """Extend one decade toward stronger or weaker sparsity when target is not bracketed."""
    if all(result["l0"] > target_l0 for result in results):
        return max(result["lambda"] for result in results) * 10.0
    if all(result["l0"] < target_l0 for result in results):
        return min(result["lambda"] for result in results) / 10.0
    return None


def _last_validation(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "validation" and "l0" in record and "fvu" in record:
            records.append(record)
    return records[-1] if records else None


def _checkpoint_at_step(run_dir: Path, step: int) -> Path | None:
    latest_path = run_dir / "latest.json"
    if not latest_path.is_file():
        return None
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if int(latest.get("step", -1)) != step:
        return None
    checkpoint = run_dir / str(latest["checkpoint"])
    return checkpoint if checkpoint.is_file() else None


def _partial_checkpoint(run_dir: Path) -> Path | None:
    latest_path = run_dir / "latest.json"
    if not latest_path.is_file():
        return None
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint = run_dir / str(latest["checkpoint"])
    return checkpoint if checkpoint.is_file() else None


def calibrate(
    config_path: str,
    output_dir: str | Path,
    *,
    pilot_steps: int = 20_000,
    maximum_extensions: int = 3,
) -> dict[str, object]:
    base = load_config(config_path)
    base.model.type = "jump_relu_sae"
    base.model.feature_activation = "relu"
    base.train.seed = 42
    base.train.max_steps = pilot_steps
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, float]] = []
    candidates = list(DEFAULT_LAMBDAS)
    extensions = 0

    while candidates:
        value = candidates.pop(0)
        run_dir = output / f"lambda-{value:.0e}"
        finished = _checkpoint_at_step(run_dir, pilot_steps)
        record = _last_validation(run_dir / "metrics.jsonl") if finished else None
        if record is None:
            config = copy.deepcopy(base)
            config.baseline.jump_relu_lambda = value
            config.train.output_dir = str(run_dir)
            config.train.eval_every = min(config.train.eval_every, pilot_steps)
            config.train.checkpoint_every = min(
                config.train.checkpoint_every, pilot_steps
            )
            partial = _partial_checkpoint(run_dir)
            config.train.resume_from = str(partial) if partial else None
            config.validate()
            train(config)
            record = _last_validation(run_dir / "metrics.jsonl")
        if record is None:
            raise RuntimeError(f"Pilot did not produce validation L0/FVU: {run_dir}")
        results.append(
            {"lambda": value, "l0": float(record["l0"]), "fvu": float(record["fvu"])}
        )
        if not candidates:
            selected = choose_jumprelu_lambda(results, base.target_l0)
            if abs(selected["l0"] - base.target_l0) <= 0.1 * base.target_l0:
                break
            extension = next_grid_lambda(results, base.target_l0)
            if extension is None or extensions >= maximum_extensions:
                break
            candidates.append(extension)
            extensions += 1

    selected = choose_jumprelu_lambda(results, base.target_l0)
    summary: dict[str, object] = {
        "target_l0": base.target_l0,
        "tolerance_fraction": 0.1,
        "pilot_steps": pilot_steps,
        "extensions": extensions,
        "results": sorted(results, key=lambda row: row["lambda"]),
        "selected_lambda": selected["lambda"],
        "selected_l0": selected["l0"],
        "selected_fvu": selected["fvu"],
    }
    (output / "jumprelu_calibration.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate JumpReLU lambda to target L0")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pilot-steps", type=int, default=20_000)
    parser.add_argument("--maximum-extensions", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = calibrate(
        args.config,
        args.output_dir,
        pilot_steps=args.pilot_steps,
        maximum_extensions=args.maximum_extensions,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
