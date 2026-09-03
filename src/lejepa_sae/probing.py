from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from .comparison import METRIC_ALIASES, _canonical_metric
from .config import ExperimentConfig, load_config
from .evaluate import load_model
from .models import JumpReLUSAE, SAEBase
from .train import autocast_context

DEFAULT_KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
MODEL_NAME = "pythia-6.9b"
HOOK_NAME = "blocks.16.hook_resid_post"
HF_MODEL_NAME = "EleutherAI/pythia-6.9b"


class ProbeSAEAdapter(nn.Module):
    """Minimal SAELens-compatible adapter used by sae-probes 0.4."""

    def __init__(self, model: nn.Module, config: ExperimentConfig, prefix_width: int | None = None):
        super().__init__()
        self.model = model
        self.config = config
        self.prefix_width = prefix_width
        if prefix_width is not None and not 1 <= prefix_width <= config.model.feature_dim:
            raise ValueError("prefix_width must be within the checkpoint feature width")
        if (
            hasattr(model, "calibrated_threshold")
            and not torch.isfinite(model.calibrated_threshold)
            and not isinstance(model, JumpReLUSAE)
        ):
            raise RuntimeError("checkpoint has no calibrated pointwise threshold")
        self.cfg = SimpleNamespace(
            d_in=config.model.d_llm,
            d_sae=prefix_width or config.model.feature_dim,
            device=config.train.device,
            dtype=config.train.precision,
            hook_name=HOOK_NAME,
            model_name=MODEL_NAME,
        )

    @torch.inference_mode()
    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        original_shape = activations.shape[:-1]
        parameter = next(self.model.parameters())
        flat = activations.reshape(-1, activations.shape[-1]).to(
            device=parameter.device, dtype=parameter.dtype
        )
        # Match training/threshold calibration precision, then give sklearn float32,
        # including when the official cache contains bfloat16 residuals.
        with autocast_context(self.config):
            if isinstance(self.model, SAEBase):
                features = self.model.encode(flat, pointwise=True)
            else:
                features = self.model(flat).features
        if self.prefix_width is not None:
            features = features[..., : self.prefix_width]
        return features.reshape(*original_shape, features.shape[-1]).float()


def resolve_llm_dtype(device: str, precision: str = "auto") -> torch.dtype:
    if precision == "auto":
        precision = "bfloat16" if torch.device(device).type == "cuda" else "float32"
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("LLM precision must be auto, float32, or bfloat16")
    return getattr(torch, precision)


def _release_memory(device: str) -> None:
    # TransformerLens hooks can participate in reference cycles.
    gc.collect()
    if torch.device(device).type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def _load_transformer_lens(device: str, dtype: torch.dtype):
    from transformer_lens import HookedTransformer

    # Weight centering changes residual coordinates, even if logits are equivalent.
    return HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME, device=device, dtype=dtype
    ).eval()


def assert_hook_parity(
    hf_activation: torch.Tensor,
    transformer_lens_activation: torch.Tensor,
    *,
    atol: float = 5e-3,
    rtol: float = 5e-3,
) -> dict[str, float]:
    if hf_activation.shape != transformer_lens_activation.shape:
        raise ValueError(
            "Hook parity failed: Hugging Face and TransformerLens shapes differ: "
            f"{tuple(hf_activation.shape)} vs {tuple(transformer_lens_activation.shape)}"
        )
    difference = (hf_activation.float() - transformer_lens_activation.float()).abs()
    if not torch.allclose(
        hf_activation.float(), transformer_lens_activation.float(), atol=atol, rtol=rtol
    ):
        raise ValueError(
            "Hook parity failed: layer 16 activations differ "
            f"(max_abs={float(difference.max()):.6g}, mean_abs={float(difference.mean()):.6g})"
        )
    return {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
    }


@torch.inference_mode()
def run_hook_parity_preflight(device: str, llm_precision: str = "auto") -> dict[str, Any]:
    """Verify that extraction's HF block output is the sae-probes TransformerLens hook."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional heavyweight dependency
        raise RuntimeError(
            "Install probing dependencies with: pip install -e '.[probes]'"
        ) from error

    dtype = resolve_llm_dtype(device, llm_precision)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    tokens = tokenizer("Hook parity preflight for LeJEPA-SAE.", return_tensors="pt").input_ids
    hf_model = (
        AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME, torch_dtype=dtype).to(device).eval()
    )
    try:
        hf_output = hf_model(tokens.to(device), output_hidden_states=True)
        hf_activation = hf_output.hidden_states[17].float().cpu()
        del hf_output
    finally:
        del hf_model
        _release_memory(device)

    tl_model = _load_transformer_lens(device, dtype)
    try:
        _, cache = tl_model.run_with_cache(
            tokens.to(device), names_filter=[HOOK_NAME], stop_at_layer=17
        )
        tl_activation = cache[HOOK_NAME].float().cpu()
        del cache
    finally:
        del tl_model
        _release_memory(device)
    result = assert_hook_parity(hf_activation, tl_activation)
    result.update(
        {"model": MODEL_NAME, "hook": HOOK_NAME, "processing": "none", "dtype": str(dtype)}
    )
    return result


def _runtime_versions() -> dict[str, str]:
    return {
        name: version(name)
        for name in (
            "sae-probes",
            "transformer-lens",
            "transformers",
            "torch",
            "sae-lens",
            "scikit-learn",
        )
    }


def _write_json(path: Path, value: Any) -> None:
    # Do not leave a truncated manifest if interrupted while saving.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _cache_spec(device: str, llm_precision: str, max_seq_len: int) -> dict[str, Any]:
    return {
        "schema": 1,
        "model": MODEL_NAME,
        "hook": HOOK_NAME,
        "processing": "none",
        "dtype": str(resolve_llm_dtype(device, llm_precision)),
        # sklearn cannot convert a bfloat16 tensor to NumPy. Widening preserves
        # all computed BF16 values and also supports the raw-residual reference.
        "storage_dtype": "torch.float32",
        "max_seq_len": max_seq_len,
        "versions": _runtime_versions(),
    }


def _cache_directory(root: Path, spec: dict[str, Any]) -> Path:
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    return root / f"hf-resid-post-v1-{fingerprint}"


def _finalize_activation_files(paths: list[Path]) -> None:
    # Check even on resume: a previous run may have stopped after generation but
    # before conversion. mmap avoids reading already-float32 tensors in full.
    for path in paths:
        activations = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(activations, torch.Tensor) or activations.ndim != 2:
            raise RuntimeError(f"Invalid residual activation cache: {path}")
        if activations.dtype != torch.float32:
            widened = activations.float()
            del activations  # release the file mapping before replace (Windows)
            temporary = path.with_suffix(".pt.tmp")
            torch.save(widened, temporary)
            temporary.replace(path)


@torch.inference_mode()
def prepare_activation_cache(
    root: str | Path,
    datasets: list[str],
    device: str,
    *,
    llm_precision: str = "auto",
    batch_size: int = 1,
    max_seq_len: int = 1024,
) -> Path:
    """Populate the official cache before its evaluator can load a default fp32 LLM."""
    from sae_probes.generate_model_activations import ensure_dataset_activations

    if batch_size < 1 or not 1 <= max_seq_len <= 2048:
        raise ValueError("activation batch size must be positive and max_seq_len in [1, 2048]")
    spec = _cache_spec(device, llm_precision, max_seq_len)
    cache = _cache_directory(Path(root), spec)
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "activation_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != spec:
            raise ValueError(f"Incompatible activation cache manifest: {manifest_path}")
    elif any(cache.glob("model_activations_*/*.pt")):
        raise ValueError(f"Refusing activation cache without provenance: {cache}")
    _write_json(manifest_path, spec)
    expected = [
        cache / f"model_activations_{MODEL_NAME}" / f"{dataset}_{HOOK_NAME}.pt"
        for dataset in datasets
    ]
    if all(path.is_file() for path in expected):
        _finalize_activation_files(expected)
        return cache

    model = _load_transformer_lens(device, resolve_llm_dtype(device, llm_precision))
    try:
        ensure_dataset_activations(
            model_name=MODEL_NAME,
            dataset_short_names=datasets,
            hook_names=[HOOK_NAME],
            model_cache_path=str(cache),
            device=device,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            model=model,
        )
    finally:
        del model
        _release_memory(device)
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Activation generation did not produce requested files: " + ", ".join(missing)
        )
    _finalize_activation_files(expected)
    return cache


def _checkpoint_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_result_manifest(output: Path, spec: dict[str, Any]) -> None:
    path = output / "probe_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != spec:
            raise ValueError(
                "Probe settings/checkpoint differ from existing results; use a new results-path"
            )
    elif any(output.glob("sae_probes_*")):
        raise ValueError("Existing probe results have no manifest; use a new results-path")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(path, spec)


def _summarize_results(output: Path, datasets: list[str], ks: list[int]) -> None:
    from sae_probes.run_sae_evals import get_save_metrics_path

    rows = []
    for dataset in datasets:
        path = get_save_metrics_path(
            dataset=dataset,
            hook_name=HOOK_NAME,
            reg_type="l1",
            model_name=MODEL_NAME,
            sae_results_path=output,
            setting="normal",
        )
        if not path.is_file():
            raise RuntimeError(f"Missing probe results: {path}. Rerun the same command to resume.")
        records = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(records, list)
            or len(records) != len(ks)
            or any(not isinstance(row, dict) for row in records)
            or {row.get("k") for row in records} != set(ks)
            or any(row.get("dataset") != dataset for row in records)
        ):
            raise RuntimeError(f"Incomplete task/k results in {path}; use a new results-path")
        try:
            normalized = [
                {
                    "k": row["k"],
                    **{metric: _canonical_metric(row, metric) for metric in METRIC_ALIASES},
                }
                for row in records
            ]
        except ValueError as error:
            raise RuntimeError(f"Missing probe metrics in {path}: {error}") from error
        if any(not math.isfinite(row[metric]) for row in normalized for metric in METRIC_ALIASES):
            raise RuntimeError(f"Non-finite or missing probe metrics in {path}")
        rows.extend(normalized)
    _write_json(
        output / "probe_summary.json",
        {
            "complete": True,
            "tasks": datasets,
            "ks": ks,
            "macro_metrics": {
                str(k): {
                    metric: mean(row[metric] for row in rows if row["k"] == k)
                    for metric in METRIC_ALIASES
                }
                for k in ks
            },
        },
    )


def run_probes(
    config: ExperimentConfig,
    checkpoint: str | Path,
    results_path: str | Path,
    model_cache_path: str | Path,
    *,
    ks: list[int] = DEFAULT_KS,
    datasets: list[str] | None = None,
    parity: bool = True,
    prefix_width: int | None = None,
    raw_residual: bool = False,
    llm_precision: str = "auto",
    activation_batch_size: int = 1,
    max_seq_len: int = 1024,
    smoke_test: bool = False,
) -> None:
    try:
        from sae_probes import DATASETS, run_baseline_evals, run_sae_evals
    except ImportError as error:
        raise RuntimeError(
            "Install probing dependencies with: pip install -e '.[probes]'"
        ) from error

    config.validate()
    datasets = sorted(set(DATASETS if datasets is None else datasets))
    unknown = set(datasets) - set(DATASETS)
    if not datasets or unknown:
        raise ValueError(f"Select nonempty official datasets; unknown datasets: {sorted(unknown)}")
    if smoke_test:
        datasets = datasets[:1]
        ks = [1, 16]
    ks = sorted(set(ks))
    if not ks or any(k < 1 or k > (prefix_width or config.model.feature_dim) for k in ks):
        raise ValueError("Probe ks must be within the encoded feature width")
    if activation_batch_size < 1 or not 1 <= max_seq_len <= 2048:
        raise ValueError("activation batch size must be positive and max_seq_len in [1, 2048]")
    spec = _cache_spec(config.train.device, llm_precision, max_seq_len)
    output = Path(results_path)
    _validate_result_manifest(
        output,
        {
            "schema": 1,
            "checkpoint_sha256": _checkpoint_digest(checkpoint),
            "config": config.to_dict(),
            "prefix_width": prefix_width,
            "ks": ks,
            "datasets": datasets,
            "setting": "normal",
            "reg_type": "l1",
            "normalization": "mean",
            "seed": 42,
            "cache": spec,
            "smoke_test": smoke_test,
        },
    )
    # Validate the checkpoint (including calibrated thresholds) before loading a 6.9B LLM.
    adapter = ProbeSAEAdapter(load_model(config, checkpoint, "cpu"), config, prefix_width)
    print(f"Probe tasks={len(datasets)}, ks={ks}, smoke_test={smoke_test}", flush=True)
    if parity:
        parity_result = run_hook_parity_preflight(config.train.device, llm_precision)
        _write_json(output / "hook_parity.json", parity_result)
    cache = prepare_activation_cache(
        model_cache_path,
        datasets,
        config.train.device,
        llm_precision=llm_precision,
        batch_size=activation_batch_size,
        max_seq_len=max_seq_len,
    )
    print(f"Probe activation cache: {cache}", flush=True)
    adapter.to(config.train.device).eval()
    try:
        run_sae_evals(
            sae=adapter,
            model_name=MODEL_NAME,
            hook_name=HOOK_NAME,
            reg_type="l1",
            setting="normal",
            results_path=str(output),
            model_cache_path=str(cache),
            ks=ks,
            datasets=datasets,
            device=config.train.device,
            mean_diff_normalization="mean",
            seed=42,
        )
    finally:
        del adapter
        _release_memory(config.train.device)
    _summarize_results(output, datasets, ks)
    if raw_residual:
        run_baseline_evals(
            model_name=MODEL_NAME,
            hook_name=HOOK_NAME,
            setting="normal",
            results_path=str(output),
            model_cache_path=str(cache),
            device=config.train.device,
            datasets=datasets,
            method="logreg",
            seed=42,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official sae-probes on a trained model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--model-cache-path", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--matryoshka-prefix", type=int, default=None)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--raw-residual", action="store_true")
    parser.add_argument(
        "--llm-precision",
        choices=("auto", "bfloat16", "float32"),
        default="auto",
        help="LLM/cache precision: auto uses bfloat16 on CUDA and float32 on CPU",
    )
    parser.add_argument("--activation-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the first selected task at k=1,16; use a separate results directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_probes(
        config,
        args.checkpoint,
        args.results_path,
        args.model_cache_path,
        ks=args.ks,
        datasets=args.datasets,
        parity=not args.skip_parity,
        prefix_width=args.matryoshka_prefix,
        raw_residual=args.raw_residual,
        llm_precision=args.llm_precision,
        activation_batch_size=args.activation_batch_size,
        max_seq_len=args.max_seq_len,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
