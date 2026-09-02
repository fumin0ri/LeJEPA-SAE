from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .config import ExperimentConfig, load_config
from .evaluate import load_model
from .models import JumpReLUSAE, SAEBase

DEFAULT_KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
MODEL_NAME = "pythia-6.9b"
HOOK_NAME = "blocks.16.hook_resid_post"


class ProbeSAEAdapter(nn.Module):
    """Minimal SAELens-compatible adapter used by sae-probes 0.4."""

    def __init__(self, model: nn.Module, config: ExperimentConfig, prefix_width: int | None = None):
        super().__init__()
        self.model = model
        self.prefix_width = prefix_width
        self.cfg = SimpleNamespace(
            d_in=config.model.d_llm,
            d_sae=prefix_width or config.model.feature_dim,
            device=config.train.device,
            dtype=config.train.precision,
            hook_name=HOOK_NAME,
            model_name=MODEL_NAME,
        )

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        original_shape = activations.shape[:-1]
        flat = activations.reshape(-1, activations.shape[-1])
        if isinstance(self.model, SAEBase):
            if hasattr(self.model, "calibrated_threshold") and not torch.isfinite(
                self.model.calibrated_threshold
            ) and not isinstance(self.model, JumpReLUSAE):
                raise RuntimeError("checkpoint has no calibrated pointwise threshold")
            features = self.model.encode(flat, pointwise=True)
        else:
            features = self.model(flat).features
        if self.prefix_width is not None:
            features = features[..., : self.prefix_width]
        return features.reshape(*original_shape, features.shape[-1])


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
def run_hook_parity_preflight(device: str) -> dict[str, float]:
    """Verify that extraction's HF block output is the sae-probes TransformerLens hook."""
    try:
        from transformer_lens import HookedTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional heavyweight dependency
        raise RuntimeError(
            "Install probing dependencies with: pip install -e '.[probes]'"
        ) from error

    hf_name = "EleutherAI/pythia-6.9b"
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    tokens = tokenizer("Hook parity preflight for LeJEPA-SAE.", return_tensors="pt").input_ids.to(
        device
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_name, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32
    ).to(device)
    hf_output = hf_model(tokens, output_hidden_states=True)
    hf_activation = hf_output.hidden_states[17]
    del hf_model, hf_output
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    tl_model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    _, cache = tl_model.run_with_cache(tokens, names_filter=[HOOK_NAME])
    result = assert_hook_parity(hf_activation, cache[HOOK_NAME])
    result.update({"model": MODEL_NAME, "hook": HOOK_NAME})
    return result


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
) -> None:
    try:
        from sae_probes import run_baseline_evals, run_sae_evals
    except ImportError as error:
        raise RuntimeError(
            "Install probing dependencies with: pip install -e '.[probes]'"
        ) from error

    output = Path(results_path)
    output.mkdir(parents=True, exist_ok=True)
    if parity:
        parity_result = run_hook_parity_preflight(config.train.device)
        (output / "hook_parity.json").write_text(
            json.dumps(parity_result, indent=2), encoding="utf-8"
        )
    model = load_model(config, checkpoint, config.train.device)
    adapter = ProbeSAEAdapter(model, config, prefix_width)
    run_sae_evals(
        sae=adapter,
        model_name=MODEL_NAME,
        hook_name=HOOK_NAME,
        reg_type="l1",
        setting="normal",
        results_path=str(output),
        model_cache_path=str(model_cache_path),
        ks=ks,
        datasets=datasets,
        device=config.train.device,
        mean_diff_normalization="mean",
        seed=42,
    )
    if raw_residual:
        run_baseline_evals(
            model_name=MODEL_NAME,
            hook_name=HOOK_NAME,
            setting="normal",
            results_path=str(output),
            model_cache_path=str(model_cache_path),
            device=config.train.device,
            datasets=datasets,
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
    )


if __name__ == "__main__":
    main()
