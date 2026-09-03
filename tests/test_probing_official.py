"""Optional, offline integration with the installed official probe implementation.

Only residuals/labels are synthetic. Encoding, splitting, feature selection, CV,
logistic fits, raw residual reference, JSON output and aggregation are real.
No pretrained LLM is downloaded or loaded by this test.
"""

import importlib
import json

import pytest
import torch

from lejepa_sae import probing
from lejepa_sae.config import ExperimentConfig
from lejepa_sae.models import build_model
from lejepa_sae.probe_parity import hf_streamed_residual, tl_streamed_residual


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_tiny_neox_layer16_hook_parity_without_weight_processing(dtype):
    tl = pytest.importorskip("transformer_lens")
    from transformer_lens.pretrained.weight_conversions.neox import convert_neox_weights
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    # At least 18 blocks: HF's last hidden state is final-LN output, whereas
    # hidden_states[17] must be the unnormalized residual after block 16.
    hf_config = GPTNeoXConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=18,
        num_attention_heads=4,
        vocab_size=32,
        max_position_embeddings=32,
        rotary_pct=0.5,
        hidden_act="gelu",
        attention_dropout=0,
        hidden_dropout=0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        hf = GPTNeoXForCausalLM(hf_config).to(dtype).eval()
        tl_config = tl.HookedTransformerConfig(
            d_model=16,
            d_head=4,
            n_heads=4,
            d_mlp=32,
            n_layers=18,
            n_ctx=32,
            d_vocab=32,
            act_fn="gelu",
            eps=hf_config.layer_norm_eps,
            parallel_attn_mlp=True,
            positional_embedding_type="rotary",
            rotary_adjacent_pairs=False,
            rotary_dim=2,
            normalization_type="LN",
            dtype=dtype,
            device="cpu",
        )
        model = tl.HookedTransformer(tl_config).eval()
        model.load_state_dict(convert_neox_weights(hf, tl_config), strict=False)
    tokens = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    hf_residual = hf(tokens, output_hidden_states=True).hidden_states[17]
    _, cache = model.run_with_cache(tokens, names_filter=[probing.HOOK_NAME], stop_at_layer=17)
    result = probing.assert_hook_parity(hf_residual, cache[probing.HOOK_NAME])
    assert result["max_abs_error"] < 0.005
    # Exercise the exact fallback with real HF/TL modules. No complete model is
    # promoted: all original CPU parameter storage/dtypes must be restored.
    hf_before = {name: p.clone() for name, p in hf.named_parameters()}
    tl_before = {name: p.clone() for name, p in model.named_parameters()}
    hf_reference = hf_streamed_residual(hf, tokens, "cpu")
    tl_reference = tl_streamed_residual(model, tokens, "cpu")
    probing.assert_hook_parity(hf_reference, tl_reference)
    for name, parameter in hf.named_parameters():
        torch.testing.assert_close(parameter, hf_before[name], rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, tl_before[name], rtol=0, atol=0)
    assert model.cfg.dtype == dtype and model.cfg.device == "cpu"


@pytest.mark.parametrize("model_type", ["proposed", "batch_topk_sae", "rdm_sae"])
def test_official_cpu_end_to_end_from_cached_residuals(tmp_path, monkeypatch, model_type):
    official = pytest.importorskip("sae_probes")
    import numpy as np
    from sae_probes import utils_data
    from transformer_lens import HookedTransformer

    generate = importlib.import_module("sae_probes.generate_sae_activations")
    dataset = sorted(official.DATASETS)[0]
    monkeypatch.setitem(generate.DATASET_SIZES, dataset, 256)
    monkeypatch.setattr(utils_data, "get_yvals", lambda name: np.tile([0, 1], 128))

    def forbid_llm(*args, **kwargs):
        pytest.fail("A complete cache must not load an LLM (including the official fp32 fallback)")

    monkeypatch.setattr(probing, "_load_transformer_lens", forbid_llm)
    monkeypatch.setattr(HookedTransformer, "from_pretrained_no_processing", forbid_llm)

    config = ExperimentConfig()
    config.model.type = model_type
    if model_type == "rdm_sae":
        config.model.num_local_views = 0
        config.model.feature_activation = "relu_forward_leaky_backward"
        config.loss.invariance_weight = 0
    config.model.d_llm = 4
    config.model.feature_dim = 16
    config.loss.axis_projections = 2
    config.train.device = "cpu"
    config.train.precision = "float32"
    config.baseline.k = 4
    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        model = build_model(config).eval()
        if model_type == "batch_topk_sae":
            model.calibrated_threshold.fill_(0.2)
        residuals = torch.randn(256, 4)
        residuals[:, 0] += torch.tensor(np.tile([-1, 1], 128))

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    cache_root = tmp_path / "cache"
    spec = probing._cache_spec("cpu", "auto", 1024)
    cache_dir = probing._cache_directory(cache_root, spec)
    activation_dir = cache_dir / f"model_activations_{probing.MODEL_NAME}"
    activation_dir.mkdir(parents=True)
    probing._write_json(cache_dir / "activation_manifest.json", spec)
    # Also exercise an interrupted BF16 -> float32 cache finalization on resume.
    torch.save(residuals.bfloat16(), activation_dir / f"{dataset}_{probing.HOOK_NAME}.pt")
    results = tmp_path / "results"
    probing.run_probes(
        config,
        checkpoint,
        results,
        cache_root,
        ks=[1, 16],
        datasets=[dataset],
        parity=False,
        raw_residual=True,
    )
    summary = json.loads((results / "probe_summary.json").read_text())
    assert summary["complete"] and summary["tasks"] == [dataset]
    assert summary["ks"] == [1, 16]
    assert set(summary["macro_metrics"]["1"]) == {"f1", "auroc", "accuracy"}
    for metrics in summary["macro_metrics"].values():
        assert all(0 <= score <= 1 for score in metrics.values())
    official_files = list((results / f"sae_probes_{probing.MODEL_NAME}").rglob("*.json"))
    raw = json.loads(official_files[0].read_text())
    assert {row["k"] for row in raw} == {1, 16}
    assert summary["macro_metrics"]["1"]["f1"] == raw[0]["test_f1"]
    baseline_files = list((results / f"baseline_results_{probing.MODEL_NAME}").rglob("*.json"))
    assert len(baseline_files) == 1
    assert json.loads(baseline_files[0].read_text())[0]["method"] == "logreg"

    # Existing task files must be skipped, without refitting any classifier.
    eval_module = importlib.import_module("sae_probes.run_sae_evals")
    monkeypatch.setattr(eval_module, "run_sae_eval", forbid_llm)
    probing.run_probes(
        config,
        checkpoint,
        results,
        cache_root,
        ks=[1, 16],
        datasets=[dataset],
        parity=False,
    )
    assert json.loads((results / "probe_summary.json").read_text()) == summary
