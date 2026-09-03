import json
from types import SimpleNamespace

import pytest
import torch

from lejepa_sae import probing
from lejepa_sae.probe_parity import streamed_float32


def test_bf16_residual_addition_order_can_differ_by_16():
    residual, attention, mlp = torch.tensor([2048.0, 8.0, 8.0], dtype=torch.bfloat16)
    hf = (mlp + attention) + residual
    tl = (residual + attention) + mlp
    assert abs(float(hf - tl)) == 16
    torch.testing.assert_close(
        mlp.float() + attention.float() + residual.float(),
        residual.float() + attention.float() + mlp.float(),
    )


@pytest.mark.parametrize("fail", [False, True])
def test_streaming_restores_dtype_storage_hooks_and_tf32_on_exception(fail):
    stage = torch.nn.Linear(4, 4).bfloat16()
    original = stage.weight.detach().clone()
    pointer = stage.weight.data_ptr()
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    with torch.inference_mode():
        try:
            with streamed_float32([stage], "cpu"):
                assert not torch.backends.cuda.matmul.allow_tf32
                result = stage(torch.ones(1, 4))
                assert result.dtype == torch.float32
                assert stage.weight.dtype == torch.bfloat16
                if fail:
                    raise RuntimeError("test failure")
        except RuntimeError:
            assert fail
    assert torch.backends.cuda.matmul.allow_tf32 == original_tf32
    assert stage.weight.data_ptr() == pointer
    torch.testing.assert_close(stage.weight, original, rtol=0, atol=0)
    assert not stage._forward_pre_hooks and not stage._forward_hooks


def test_only_one_stage_is_promoted_and_forward_errors_restore_storage():
    first = torch.nn.Linear(4, 4).bfloat16()
    second = torch.nn.Linear(4, 4).bfloat16()

    def check(module, inputs):
        assert first.weight.dtype == torch.bfloat16
        assert second.weight.dtype == torch.float32

    with torch.inference_mode(), streamed_float32([first, second], "cpu"):
        handle = second.register_forward_pre_hook(check)
        try:
            second(first(torch.ones(1, 4)))
            with pytest.raises(RuntimeError):
                second(torch.ones(1, 7))
            assert second.weight.dtype == torch.bfloat16
        finally:
            handle.remove()


@pytest.mark.parametrize("version,key", [("4.55.0", "torch_dtype"), ("4.57.6", "dtype")])
def test_hf_loader_uses_version_appropriate_dtype(monkeypatch, version, key):
    import transformers

    calls = []
    monkeypatch.setattr(transformers, "__version__", version)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: calls.append(kwargs) or torch.nn.Linear(2, 2),
    )
    probing._load_hf_parity_model("cpu", torch.bfloat16)
    assert calls == [{key: torch.bfloat16, "low_cpu_mem_usage": True}]


@pytest.fixture
def parity_models(monkeypatch):
    import transformers

    hf_activation = torch.ones(1, 3, 4)
    hf_activation[..., 0] = 2048
    tl_activation = hf_activation.clone()
    tl_activation[..., 0] -= 16

    class HF(torch.nn.Module):
        def forward(self, tokens, **kwargs):
            return SimpleNamespace(hidden_states=[None] * 17 + [hf_activation])

    class TL:
        def run_with_cache(self, *args, **kwargs):
            return tl_activation, {probing.HOOK_NAME: tl_activation}

    monkeypatch.setattr(probing, "_load_hf_parity_model", lambda *args: HF())
    monkeypatch.setattr(probing, "_load_transformer_lens", lambda *args: TL())
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda name: (lambda *args, **kwargs: SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))),
    )
    return hf_activation, tl_activation


def test_bf16_requires_fp32_reference_and_saves_direct_discrepancy(
    parity_models,
    monkeypatch,
    tmp_path,
):
    hf, tl = parity_models
    calls = []

    def reference(*args):
        calls.append(args)
        return hf, hf.clone()

    monkeypatch.setattr(probing, "_float32_parity_reference", reference)
    report = tmp_path / "hook_parity.json"
    result = probing.run_hook_parity_preflight("cpu", "bfloat16", report)
    assert len(calls) == 1
    assert result["passed"] and not result["elementwise_allclose"]
    assert result["verification"] == "float32_reference_on_bf16_weights"
    assert result["max_abs_error"] == 16
    assert result["relative_rmse"] < 0.01
    assert result["float32_reference"]["elementwise_allclose"]
    assert json.loads(report.read_text()) == result


def test_true_fp32_hook_mismatch_still_fails_and_is_recorded(parity_models, monkeypatch, tmp_path):
    hf, tl = parity_models
    monkeypatch.setattr(probing, "_float32_parity_reference", lambda *args: (hf, hf + 1))
    report = tmp_path / "hook_parity.json"
    with pytest.raises(ValueError, match="Hook parity failed"):
        probing.run_hook_parity_preflight("cpu", "bfloat16", report)
    saved = json.loads(report.read_text())
    assert not saved["passed"]
    assert not saved["float32_reference"]["elementwise_allclose"]


def test_large_bf16_error_is_not_blessed_by_a_reference(parity_models, monkeypatch, tmp_path):
    hf, tl = parity_models
    tl.mul_(0.5)
    monkeypatch.setattr(probing, "_float32_parity_reference", lambda *args: pytest.fail("unsafe"))
    report = tmp_path / "hook_parity.json"
    with pytest.raises(ValueError, match="safety bound"):
        probing.run_hook_parity_preflight("cpu", "bfloat16", report)
    assert not json.loads(report.read_text())["passed"]


def test_fp32_mode_never_uses_bf16_fallback(parity_models, monkeypatch, tmp_path):
    monkeypatch.setattr(
        probing, "_float32_parity_reference", lambda *args: pytest.fail("unexpected")
    )
    with pytest.raises(ValueError, match="Hook parity failed"):
        probing.run_hook_parity_preflight("cpu", "float32", tmp_path / "hook_parity.json")
