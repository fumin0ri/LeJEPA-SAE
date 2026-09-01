import json

import pytest
import torch
from safetensors.torch import save_file

from lejepa_sae.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from lejepa_sae.evaluate import evaluate
from lejepa_sae.models import build_model


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
        ("standard_sae", {"full_reconstruction_mse"}),
        ("proposed", {"global_local_mse", "support_jaccard"}),
        (
            "dimension_denoising_sae",
            {"full_reconstruction_mse", "masked_reconstruction_mse"},
        ),
    ],
)
def test_evaluation_metrics_and_report_artifacts(
    tmp_path, monkeypatch, model_type, expected_metrics
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
            "invariance": 0.2,
            "random_distribution": 0.5,
            "axis_distribution": 0.4,
            "feature_std": 0.3,
        },
        {
            "kind": "validation",
            "step": 2,
            "active_fraction": 0.49,
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
        max_tokens=4,
        top_k=2,
        support_epsilon=0.0,
    )

    assert expected_metrics <= result.keys()
    assert result["tokens"] == 4
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
    assert "Feature standard deviation" in training_chart
    assert len(
        (output_dir / "training_history.csv").read_text(encoding="utf-8").splitlines()
    ) == 4
    dashboard = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "Single-token JEPA evaluation" in dashboard
    assert "Training curves" in dashboard
    assert "Highest-variance features" in dashboard
