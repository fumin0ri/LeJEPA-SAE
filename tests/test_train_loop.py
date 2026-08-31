import json

import pytest
import torch
from safetensors.torch import save_file

from lejepa_sae.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from lejepa_sae.train import train


@pytest.mark.parametrize(
    ("model_type", "window_size"),
    [
        ("proposed", 5),
        ("standard_sae", 1),
        ("single_token_jepa", 1),
        ("dimension_denoising_sae", 1),
    ],
)
def test_end_to_end_cpu_training_and_checkpoint(tmp_path, model_type, window_size):
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
            window_size=window_size,
            num_workers=0,
        ),
        model=ModelConfig(
            type=model_type,
            d_llm=8,
            d_encoder=8,
            num_layers=1,
            num_heads=2,
            mlp_ratio=2,
            feature_dim=16,
            num_local_views=2,
            local_tokens=min(2, window_size),
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
    checkpoint = train(config)

    assert checkpoint.exists()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["step"] == 2
    run_dir = tmp_path / f"run-{model_type}"
    assert (run_dir / "config.resolved.yaml").exists()
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) == 4
