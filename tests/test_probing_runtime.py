"""Offline contracts for the memory-safe official sae-probes entry point."""

import copy
import json
import os
import shutil
import subprocess
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lejepa_sae import probing
from lejepa_sae.config import ExperimentConfig
from lejepa_sae.models import build_model


@pytest.fixture(autouse=True)
def deterministic_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        yield


def small_config(model_type="proposed"):
    config = ExperimentConfig()
    config.model.type = model_type
    config.model.d_llm = 4
    config.model.feature_dim = 16
    config.loss.axis_projections = 2
    config.train.device = "cpu"
    config.train.precision = "float32"
    config.baseline.k = 2
    config.baseline.matryoshka_group_sizes = [2, 2, 4, 4, 4]
    config.validate()
    return config


@pytest.mark.parametrize(
    "model_type",
    [
        "proposed",
        "batch_topk_sae",
        "jump_relu_sae",
        "matryoshka_sae",
    ],
)
def test_adapter_accepts_bfloat16_cache_and_returns_finite_float32(model_type):
    config = small_config(model_type)
    model = build_model(config).eval()
    if hasattr(model, "calibrated_threshold"):
        model.calibrated_threshold.fill_(0.1)
    inputs = torch.randn(2, 3, 4).bfloat16()
    adapter = probing.ProbeSAEAdapter(model, config)
    result = adapter.encode(inputs)
    expected = (
        model(inputs.float().reshape(-1, 4)).features
        if model_type == "proposed"
        else model.encode(inputs.float().reshape(-1, 4), pointwise=True)
    ).reshape(2, 3, 16)
    torch.testing.assert_close(result, expected)
    assert result.dtype == torch.float32
    assert not result.requires_grad
    assert torch.isfinite(result).all()
    torch.testing.assert_close(
        result, torch.cat([adapter.encode(row.unsqueeze(0)) for row in inputs])
    )


def test_adapter_rejects_uncalibrated_threshold_and_invalid_prefix():
    config = small_config("batch_topk_sae")
    with pytest.raises(RuntimeError, match="no calibrated"):
        probing.ProbeSAEAdapter(build_model(config), config)
    config.model.type = "proposed"
    for prefix in (0, 17):
        with pytest.raises(ValueError, match="prefix_width"):
            probing.ProbeSAEAdapter(build_model(config), config, prefix)


def test_llm_dtype_defaults_and_validation():
    assert probing.resolve_llm_dtype("cuda:0") == torch.bfloat16
    assert probing.resolve_llm_dtype("cpu") == torch.float32
    assert probing.resolve_llm_dtype("cuda", "float32") == torch.float32
    with pytest.raises(ValueError, match="precision"):
        probing.resolve_llm_dtype("cuda", "float16")


def test_transformer_lens_loading_has_no_processing_and_explicit_dtype(monkeypatch):
    calls = []
    model = torch.nn.Linear(4, 4)
    module = ModuleType("transformer_lens")
    module.HookedTransformer = SimpleNamespace(
        from_pretrained_no_processing=lambda *args, **kwargs: (
            calls.append((args, kwargs)) or model
        )
    )
    monkeypatch.setitem(sys.modules, "transformer_lens", module)
    assert probing._load_transformer_lens("cuda:0", torch.bfloat16) is model
    assert calls == [((probing.MODEL_NAME,), {"device": "cuda:0", "dtype": torch.bfloat16})]
    assert not model.training


def test_parity_releases_hf_before_loading_tl_and_uses_identical_tokens(monkeypatch):
    import transformers

    tokens = torch.tensor([[1, 2, 3]])
    activation = torch.randn(1, 3, 4)
    refs = []

    class FakeHF(torch.nn.Module):
        def forward(self, input_ids, *, output_hidden_states):
            assert not self.training
            assert output_hidden_states
            torch.testing.assert_close(input_ids, tokens)
            return SimpleNamespace(hidden_states=[None] * 17 + [activation])

    def load_hf(name, **kwargs):
        assert name == probing.HF_MODEL_NAME
        assert kwargs.get("dtype", kwargs.get("torch_dtype")) == torch.float32
        model = FakeHF()
        refs.append(weakref.ref(model))
        return model

    class FakeTL(torch.nn.Module):
        def run_with_cache(self, input_ids, **kwargs):
            torch.testing.assert_close(input_ids, tokens)
            assert kwargs == {"names_filter": [probing.HOOK_NAME], "stop_at_layer": 17}
            return activation, {probing.HOOK_NAME: activation.clone()}

    def load_tl(device, dtype):
        assert refs[0]() is None, "HF and TL must not coexist in GPU memory"
        model = FakeTL()
        refs.append(weakref.ref(model))
        return model

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_hf)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda name: (lambda *args, **kwargs: SimpleNamespace(input_ids=tokens)),
    )
    monkeypatch.setattr(probing, "_load_transformer_lens", load_tl)
    result = probing.run_hook_parity_preflight("cpu")
    assert result["max_abs_error"] == 0
    assert result["processing"] == "none"
    assert all(ref() is None for ref in refs)


@pytest.fixture
def official_stub(monkeypatch):
    package = ModuleType("sae_probes")
    package.DATASETS = ["task-b", "task-a"]
    generation = ModuleType("sae_probes.generate_model_activations")
    evaluation = ModuleType("sae_probes.run_sae_evals")
    calls = {"generate": [], "evaluate": [], "baseline": []}

    def results_path(*, dataset, hook_name, reg_type, model_name, sae_results_path, setting):
        return Path(sae_results_path) / (
            f"sae_probes_{model_name}/{setting}_setting/{dataset}_{hook_name}_{reg_type}.json"
        )

    def generate(**kwargs):
        calls["generate"].append(kwargs)
        root = Path(kwargs["model_cache_path"]) / f"model_activations_{kwargs['model_name']}"
        root.mkdir(parents=True, exist_ok=True)
        for dataset in kwargs["dataset_short_names"]:
            torch.save(torch.randn(4, 4).bfloat16(), root / f"{dataset}_{probing.HOOK_NAME}.pt")

    def evaluate(**kwargs):
        calls["evaluate"].append(kwargs)
        encoded = kwargs["sae"].encode(torch.randn(4, 4).bfloat16())
        assert encoded.dtype == torch.float32
        assert torch.isfinite(encoded).all()
        for dataset in kwargs["datasets"]:
            path = results_path(
                dataset=dataset,
                hook_name=kwargs["hook_name"],
                reg_type=kwargs["reg_type"],
                model_name=kwargs["model_name"],
                sae_results_path=kwargs["results_path"],
                setting=kwargs["setting"],
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    [
                        {
                            "dataset": dataset,
                            "k": k,
                            "test_f1": 0.5,
                            "test_auc": 0.6,
                            "test_acc": 0.7,
                        }
                        for k in kwargs["ks"]
                    ]
                ),
                encoding="utf-8",
            )

    package.run_sae_evals = evaluate
    package.run_baseline_evals = lambda **kwargs: calls["baseline"].append(kwargs)
    generation.ensure_dataset_activations = generate
    evaluation.get_save_metrics_path = results_path
    monkeypatch.setitem(sys.modules, "sae_probes", package)
    monkeypatch.setitem(sys.modules, generation.__name__, generation)
    monkeypatch.setitem(sys.modules, evaluation.__name__, evaluation)
    monkeypatch.setattr(probing, "_runtime_versions", lambda: {"sae-probes": "0.4.0"})
    monkeypatch.setattr(probing, "_load_transformer_lens", lambda *args: torch.nn.Linear(4, 4))
    monkeypatch.setattr(probing, "run_hook_parity_preflight", lambda *args: {"max_abs_error": 0})
    return calls


def test_cache_prepopulation_is_small_batch_and_reused(tmp_path, official_stub):
    root = probing.prepare_activation_cache(tmp_path, ["task-a"], "cpu", batch_size=2)
    assert official_stub["generate"][0]["batch_size"] == 2
    assert official_stub["generate"][0]["max_seq_len"] == 1024
    assert "model" in official_stub["generate"][0]
    cached = torch.load(
        root / f"model_activations_{probing.MODEL_NAME}/task-a_{probing.HOOK_NAME}.pt",
        weights_only=True,
    )
    assert cached.dtype == torch.float32
    assert probing.prepare_activation_cache(tmp_path, ["task-a"], "cpu") == root
    assert len(official_stub["generate"]) == 1
    other = probing.prepare_activation_cache(tmp_path, ["task-a"], "cpu", max_seq_len=512)
    assert other != root
    assert len(official_stub["generate"]) == 2


def test_cache_requires_generation_success(tmp_path, official_stub, monkeypatch):
    module = sys.modules["sae_probes.generate_model_activations"]
    monkeypatch.setattr(module, "ensure_dataset_activations", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="did not produce"):
        probing.prepare_activation_cache(tmp_path, ["task-a"], "cpu")


def test_checkpoint_to_smoke_summary_and_resume_contract(tmp_path, official_stub):
    config = small_config()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": build_model(config).state_dict()}, checkpoint)
    output = tmp_path / "results"
    kwargs = dict(smoke_test=True, raw_residual=True)
    probing.run_probes(config, checkpoint, output, tmp_path / "cache", **kwargs)
    call = official_stub["evaluate"][0]
    assert call["datasets"] == ["task-a"]
    assert call["ks"] == [1, 16]
    assert call["setting"] == "normal" and call["reg_type"] == "l1"
    assert call["seed"] == 42 and call["mean_diff_normalization"] == "mean"
    assert official_stub["baseline"][0]["method"] == "logreg"
    summary = json.loads((output / "probe_summary.json").read_text())
    assert summary["complete"]
    assert summary["macro_metrics"]["16"]["f1"] == 0.5
    assert (output / "hook_parity.json").is_file()
    probing.run_probes(config, checkpoint, output, tmp_path / "cache", **kwargs)
    assert len(official_stub["generate"]) == 1
    with pytest.raises(ValueError, match="new results-path"):
        probing.run_probes(config, checkpoint, output, tmp_path / "cache", ks=[1])
    changed_config = copy.deepcopy(config)
    changed_config.model.feature_activation = "relu_forward_leaky_backward"
    with pytest.raises(ValueError, match="new results-path"):
        probing.run_probes(changed_config, checkpoint, output, tmp_path / "cache", **kwargs)
    torch.save({"model": build_model(config).state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="new results-path"):
        probing.run_probes(config, checkpoint, output, tmp_path / "cache", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"datasets": []},
        {"datasets": ["unknown"]},
        {"ks": [17]},
        {"ks": []},
        {"activation_batch_size": 0},
        {"max_seq_len": 2049},
        {"llm_precision": "float16"},
    ],
)
def test_invalid_options_fail_before_model_load(tmp_path, official_stub, kwargs):
    with pytest.raises(ValueError):
        probing.run_probes(small_config(), "missing.pt", tmp_path, tmp_path / "cache", **kwargs)
    assert not official_stub["generate"]


def test_missing_task_results_are_not_summarized(tmp_path, official_stub):
    with pytest.raises(RuntimeError, match="Missing probe results"):
        probing._summarize_results(tmp_path, ["task-a"], [1, 16])
    assert not (tmp_path / "probe_summary.json").exists()


@pytest.mark.parametrize("corruption", ["missing_k", "duplicate", "nan", "bad_record"])
def test_summary_rejects_incomplete_or_invalid_results(tmp_path, official_stub, corruption):
    module = sys.modules["sae_probes.run_sae_evals"]
    path = module.get_save_metrics_path(
        dataset="task-a",
        hook_name=probing.HOOK_NAME,
        reg_type="l1",
        model_name=probing.MODEL_NAME,
        sae_results_path=tmp_path,
        setting="normal",
    )
    path.parent.mkdir(parents=True)
    rows = [
        {"dataset": "task-a", "k": k, "f1_score": 0.5, "roc_auc": 0.6, "accuracy": 0.7}
        for k in [1, 16]
    ]
    if corruption == "missing_k":
        rows.pop()
    elif corruption == "duplicate":
        rows[1]["k"] = 1
    elif corruption == "nan":
        rows[1]["f1_score"] = float("nan")
    else:
        rows[1] = 42
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(RuntimeError):
        probing._summarize_results(tmp_path, ["task-a"], [1, 16])
    assert not (tmp_path / "probe_summary.json").exists()


@pytest.mark.parametrize(
    ("mode", "module", "smoke", "sparse"),
    [
        ("smoke", "lejepa_sae.probing", True, True),
        ("probe", "lejepa_sae.probing", False, True),
        ("dense-smoke", "lejepa_sae.dense_probing", True, False),
        ("dense", "lejepa_sae.dense_probing", False, False),
    ],
)
def test_pilot_launcher_preserves_paths_and_precision_flags(
    tmp_path, mode, module, smoke, sparse
):
    bash = (
        str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe")
        if os.name == "nt"
        else shutil.which("bash")
    )
    if not bash or not Path(bash).is_file():
        pytest.skip("Native bash is unavailable")
    for name in ("config.resolved.yaml", "checkpoint-00010000.pt"):
        (tmp_path / name).touch()
    env = dict(os.environ, ACTIVATION_BATCH_SIZE="2", LLM_PRECISION="bfloat16")
    script = Path(__file__).resolve().parents[1] / "scripts/run_probe_pilot.sh"
    result = subprocess.run(
        [
            bash,
            "-c",
            'python() { printf "%s\\n" "$@"; }; export -f python; ' 'exec "$BASH" "$@"',
            "probe-launcher-test",
            script.as_posix(),
            mode,
            tmp_path.as_posix(),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env=env,
    )
    args = result.stdout.splitlines()
    assert args[:2] == ["-m", module]
    assert args[args.index("--activation-batch-size") + 1] == "2"
    assert args[args.index("--llm-precision") + 1] == "bfloat16"
    assert ("--ks" in args) == sparse
    if sparse:
        assert args[args.index("--ks") + 1 : args.index("--ks") + 3] == ["1", "16"]
    assert ("--smoke-test" in args) == smoke
    assert "--skip-parity" not in args
