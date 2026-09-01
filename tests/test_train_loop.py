import json

import pytest
import torch
from safetensors.torch import save_file

from lejepa_sae.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from lejepa_sae.train import train


@pytest.mark.parametrize("model_type", ["proposed", "standard_sae", "dimension_denoising_sae"])
def test_end_to_end_cpu_training_and_checkpoint(tmp_path, model_type):
    activation_dir = tmp_path / "activations"
    shards = []
    for split_index, split in enumerate(("train", "validation", "test")):
        split_dir = activation_dir / split
        split_dir.mkdir(parents=True)
        filename = split_dir / "shard-00000.safetensors"
        generator = torch.Generator().manual_seed(split_index)
        save_file(
            {
                "activations": torch.randn(6, 8, generator=generator),
                "token_ids": torch.arange(6, dtype=torch.int32),
            },
            str(filename),
        )
        shards.append(
            {
                "file": f"{split}/shard-00000.safetensors",
                "split": split,
                "sequences": [
                    {
                        "offset": 0,
                        "length": 6,
                        "document_id": f"document-{split}",
                        "segment_index": 0,
                    }
                ],
            }
        )
    (activation_dir / "manifest.json").write_text(
        json.dumps({"d_llm": 8, "shards": shards}), encoding="utf-8"
    )

    config = ExperimentConfig(
        data=DataConfig(
            activation_dir=str(activation_dir),
            window_size=1,
            num_workers=0,
        ),
        model=ModelConfig(
            type=model_type,
            d_llm=8,
            feature_dim=16,
            num_local_views=2,
        ),
        train=TrainConfig(
            device="cpu",
            precision="float32",
            batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=2,
            warmup_steps=1,
            log_every=1,
            eval_every=1,
            checkpoint_every=1,
            eval_batches=1,
            output_dir=str(tmp_path / f"run-{model_type}"),
        ),
    )
    config.loss.rdm_projections = 4
    config.loss.axis_projections = 4
    checkpoint = train(config)

    assert checkpoint.exists()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["step"] == 2
    run_dir = tmp_path / f"run-{model_type}"
    assert (run_dir / "config.resolved.yaml").exists()
    records = [
        json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(records) == 4
    train_records = [record for record in records if record["kind"] == "train"]
    assert all(record["samples_per_second"] > 0 for record in train_records)
    assert all(record["optimizer_steps_per_second"] > 0 for record in train_records)
    if model_type == "proposed":
        assert all("feature_std" in record for record in train_records)
