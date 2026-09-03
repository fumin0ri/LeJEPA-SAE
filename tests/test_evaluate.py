import csv
import json

import pytest
import torch
from safetensors.torch import save_file

from lejepa_sae.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from lejepa_sae.evaluate import evaluate
from lejepa_sae.models import build_model
from lejepa_sae.reporting import _format_metric, load_training_history


class FakeTokenizer:
    def decode(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


def make_test_store(tmp_path):
    activation_dir = tmp_path / "activations"
    test_dir = activation_dir / "test"
    test_dir.mkdir(parents=True)
    save_file(
        {
            "activations": torch.randn(5, 8),
            "token_ids": torch.arange(5, dtype=torch.int32),
        },
        str(test_dir / "shard-00000.safetensors"),
    )
    manifest = {
        "model": "fake-tokenizer",
        "revision": "test",
        "d_llm": 8,
        "shards": [
            {
                "file": "test/shard-00000.safetensors",
                "split": "test",
                "sequences": [
                    {
                        "offset": 0,
                        "length": 5,
                        "document_id": "test-document",
                        "segment_index": 0,
                    }
                ],
            }
        ],
    }
    (activation_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return activation_dir


@pytest.mark.parametrize(
    ("model_type", "expected_metrics"),
    [
        ("batch_topk_sae", {"full_reconstruction_mse", "fvu", "mean_l0"}),
        ("jump_relu_sae", {"full_reconstruction_mse", "fvu", "mean_l0"}),
        ("matryoshka_sae", {"full_reconstruction_mse", "fvu", "mean_l0"}),
        ("proposed", {"global_local_mse", "support_jaccard"}),
    ],
)
@pytest.mark.parametrize("support_epsilon", [0.0, 100.0])
def test_evaluation_metrics_and_report_artifacts(
    tmp_path, monkeypatch, model_type, expected_metrics, support_epsilon
):
    activation_dir = make_test_store(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    training_records = [
        {
            "kind": "train",
            "step": 1,
            "active_fraction": 0.48,
            "global_active_fraction": 0.5,
            "local_active_fraction": 0.46,
            "off_to_on": 0.03,
            "on_to_off": 0.07,
            "local_global_active_fraction_gap": -0.04,
            "transition_rate_gap": -0.04,
            "invariance": 0.4,
            "random_distribution": 0.8,
            "axis_distribution": 0.7,
            "feature_std": 0.2,
        },
        {
            "kind": "train",
            "step": 2,
            "active_fraction": 0.51,
            "global_active_fraction": 0.52,
            "local_active_fraction": 0.5,
            "off_to_on": 0.04,
            "on_to_off": 0.06,
            "local_global_active_fraction_gap": -0.02,
            "transition_rate_gap": -0.02,
            "invariance": 0.2,
            "random_distribution": 0.5,
            "axis_distribution": 0.4,
            "feature_std": 0.3,
        },
        {
            "kind": "validation",
            "step": 2,
            "active_fraction": 0.49,
            "global_active_fraction": 0.47,
            "local_active_fraction": 0.495,
            "off_to_on": 0.041,
            "on_to_off": 0.016,
            "local_global_active_fraction_gap": 0.025,
            "transition_rate_gap": 0.025,
            "invariance": 0.24,
            "random_distribution": 0.54,
            "axis_distribution": 0.45,
            "feature_std": 0.28,
        },
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in training_records), encoding="utf-8"
    )
    config = ExperimentConfig(
        data=DataConfig(
            activation_dir=str(activation_dir),
            window_size=1,
            eval_stride=1,
            num_workers=0,
        ),
        model=ModelConfig(
            type=model_type,
            d_llm=8,
            feature_dim=16,
            num_local_views=2,
            dimension_keep_fraction=0.5,
        ),
        train=TrainConfig(
            device="cpu", precision="float32", batch_size=2, output_dir=str(run_dir)
        ),
    )
    config.loss.axis_projections = 4
    config.baseline.k = 2
    config.baseline.matryoshka_group_sizes = [2, 2, 4, 4, 4]
    config.validate()
    checkpoint = tmp_path / f"{model_type}.pt"
    torch.save({"model": build_model(config).state_dict()}, checkpoint)
    monkeypatch.setattr(
        "lejepa_sae.evaluate.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )

    result = evaluate(
        config,
        checkpoint,
        tmp_path / f"evaluation-{model_type}",
        max_tokens=5,
        top_k=2,
        support_epsilon=support_epsilon,
    )

    assert expected_metrics <= result.keys()
    assert result["tokens"] == 5
    if model_type == "proposed":
        assert result["off_to_on"] - result["on_to_off"] == pytest.approx(
            result["local_active_fraction"] - result["global_active_fraction"], abs=1e-7
        )
        assert result["transition_rate_gap"] == pytest.approx(
            result["local_global_active_fraction_gap"], abs=1e-7
        )
        if support_epsilon == 0:
            assert result["global_active_fraction"] == pytest.approx(
                result["mean_active_fraction"]
            )
        else:
            assert result["global_active_fraction"] > result["mean_active_fraction"] == 0
    assert len(
        (tmp_path / f"evaluation-{model_type}" / "top_tokens.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 16
    output_dir = tmp_path / f"evaluation-{model_type}"
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "feature_metrics.csv").exists()
    assert "Evaluation summary" in (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "<svg" in (output_dir / "feature_diagnostics.svg").read_text(encoding="utf-8")
    training_chart = (output_dir / "training_curves.svg").read_text(encoding="utf-8")
    assert "Active fraction" in training_chart
    assert "Global-local MSE" in training_chart
    assert "Random vs axis RDMReg" in training_chart
    assert "Global vs local RDMReg contribution" in training_chart
    assert "Feature standard deviation" in training_chart
    assert "Gate transitions (%)" in training_chart
    assert "Sparsity gap (percentage points)" in training_chart
    assert "OFF→ON val" in training_chart
    assert "ON→OFF train" in training_chart
    assert len(
        (output_dir / "training_history.csv").read_text(encoding="utf-8").splitlines()
    ) == 4
    dashboard = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "Single-token sparse representation evaluation" in dashboard
    assert "Training curves" in dashboard
    assert "Highest-variance features" in dashboard
    with (output_dir / "training_history.csv").open(encoding="utf-8", newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert float(exported[-1]["off_to_on"]) == 0.041
    assert float(exported[-1]["on_to_off"]) == 0.016
    assert float(exported[-1]["transition_rate_gap"]) == 0.025


def test_transition_units_and_legacy_history(tmp_path):
    assert _format_metric("off_to_on", 0.041) == "4.10%"
    assert _format_metric("on_to_off", 0.016) == "1.60%"
    assert _format_metric("transition_rate_gap", 0.025) == "+2.500 pp"
    path = tmp_path / "legacy.jsonl"
    path.write_text('{"step": 20, "active_fraction": 0.5}\n', encoding="utf-8")
    history = load_training_history(path)
    assert len(history) == 1
    assert "off_to_on" not in history[0]
