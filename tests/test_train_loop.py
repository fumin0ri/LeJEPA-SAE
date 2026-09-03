import json

import pytest
import torch
from safetensors.torch import save_file

from lejepa_sae.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from lejepa_sae.train import train


@pytest.mark.parametrize(
    ("model_type", "mask_scaling", "rate_activation", "num_local_views", "rdm_power", "axis_power"),
    [
        ("proposed", "inverted", None, 2, 2, None),
        ("proposed", "sqrt", None, 2, 2, None),
        ("proposed", "none", None, 2, 2, None),
        ("proposed", "sqrt", "relu", 2, 2, None),
        ("proposed", "sqrt", "relu_forward_leaky_backward", 2, 2, None),
        ("proposed", "sqrt", None, 0, 2, None),
        ("rdm_sae", "none", None, 0, 2, None),
        ("rdm_sae", "none", None, 0, 1, None),
        ("rdm_sae", "none", None, 0, 2, 1),
        ("batch_topk_sae", "inverted", None, 2, 2, None),
        ("jump_relu_sae", "inverted", None, 2, 2, None),
        ("matryoshka_sae", "inverted", None, 2, 2, None),
    ],
)
def test_end_to_end_cpu_training_and_checkpoint(
    tmp_path, model_type, mask_scaling, rate_activation, num_local_views, rdm_power, axis_power
):
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
            num_local_views=num_local_views,
            mask_scaling=mask_scaling,
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
    config.loss.rdm_wasserstein_power = rdm_power
    config.loss.rdm_axis_wasserstein_power = axis_power
    if axis_power is not None:
        config.loss.rdm_target_scale = 1.5
        config.loss.axis_weight = 4
    config.loss.axis_projections = 4
    if model_type == "rdm_sae":
        config.loss.lambda_rdm = 1
        config.loss.rdm_gradient_diagnostics = True
    if num_local_views == 0:
        config.loss.invariance_weight = 0
        config.model.feature_activation = "relu_forward_leaky_backward"
        config.model.leaky_backward_slope = 0.1
    config.baseline.k = 2
    config.baseline.k_aux = 4
    config.baseline.dead_feature_window_tokens = 2
    config.baseline.matryoshka_group_sizes = [2, 2, 4, 4, 4]
    if rate_activation:
        config.model.feature_activation = rate_activation
        config.loss.rate_weight = 1.0
        config.loss.rate_gradient_diagnostics = True
        config.loss.expected_l0_fraction = 0.05
    checkpoint = train(config)

    assert checkpoint.exists()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["step"] == 2
    assert state["config"]["model"]["mask_scaling"] == mask_scaling
    assert state["config"]["loss"]["rate_weight"] == config.loss.rate_weight
    assert state["config"]["loss"]["rdm_wasserstein_power"] == rdm_power
    assert state["config"]["loss"]["rdm_axis_wasserstein_power"] == axis_power
    run_dir = tmp_path / f"run-{model_type}"
    assert (run_dir / "config.resolved.yaml").exists()
    training_plan = json.loads((run_dir / "training_plan.json").read_text())
    assert training_plan["requested_max_steps"] == 2
    assert training_plan["resolved_max_steps"] == 2
    assert training_plan["sample_delta_from_one_epoch"] == 2
    records = [
        json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(records) == 4
    train_records = [record for record in records if record["kind"] == "train"]
    assert all(record["samples_per_second"] > 0 for record in train_records)
    assert all(record["optimizer_steps_per_second"] > 0 for record in train_records)
    if model_type == "proposed":
        assert all("feature_std" in record for record in train_records)
        for record in records:
            if num_local_views == 0:
                assert "invariance" not in record
                assert "off_to_on" not in record
                assert not any("local" in key for key in record)
                assert record["distribution"] == pytest.approx(record["global_distribution"])
                assert record["loss"] == pytest.approx(
                    config.loss.lambda_rdm * record["distribution"], rel=1e-6
                )
                continue
            assert 0 <= record["off_to_on"] <= 1
            assert 0 <= record["on_to_off"] <= 1
            assert record["off_to_on"] - record["on_to_off"] == pytest.approx(
                record["local_active_fraction"] - record["global_active_fraction"],
                abs=1e-7,
            )
            assert record["transition_rate_gap"] == pytest.approx(
                record["local_global_active_fraction_gap"], abs=1e-7
            )
        assert all(
            record["expected_l0_fraction"] == pytest.approx(config.loss.expected_l0_fraction)
            for record in train_records
        )
        if rate_activation:
            assert all("rate_loss" in record for record in records)
            assert all("rate_to_base_grad_ratio" in record for record in train_records)
            for record in records:
                assert record["loss"] == pytest.approx(
                    record["base_loss"] + record["rate_contribution"], rel=1e-6
                )
    else:
        assert all("off_to_on" not in record for record in records)
        assert all("reconstruction" in record for record in train_records)
        torch.testing.assert_close(
            state["model"]["decoder.weight"].norm(dim=0),
            torch.ones(config.model.feature_dim),
            atol=1e-5,
            rtol=1e-5,
        )
        if model_type in {"batch_topk_sae", "matryoshka_sae"}:
            assert (run_dir / "threshold_calibration.json").exists()
            assert torch.isfinite(state["model"]["calibrated_threshold"])
        if model_type == "rdm_sae":
            from lejepa_sae.evaluate import load_model
            from lejepa_sae.probing import ProbeSAEAdapter

            assert not (run_dir / "threshold_calibration.json").exists()
            assert "threshold_calibration" not in state
            assert "calibrated_threshold" not in state["model"]
            loaded = load_model(config, checkpoint, "cpu")
            sample = torch.randn(3, 1, 8)
            torch.testing.assert_close(
                ProbeSAEAdapter(loaded, config).encode(sample), loaded.encode(sample)
            )
            for record in records:
                assert record["rdm_wasserstein_power"] == rdm_power
                assert record["rdm_random_wasserstein_power"] == rdm_power
                assert record["rdm_axis_wasserstein_power"] == (axis_power or rdm_power)
                assert record["active_fraction_gt_0"] == record["active_fraction"]
                assert 0 <= record["active_fraction_gt_1e-1"] <= record["active_fraction_gt_0"]
                assert record["loss"] == pytest.approx(
                    record["reconstruction_contribution"] + record["rdm_contribution"], rel=1e-6
                )
                assert "auxk" not in record and "invariance" not in record
                if record["kind"] == "train":
                    assert "rdm_to_reconstruction_grad_ratio" in record
                    assert "rdm_axis_preactivation_grad_rms" in record
                    assert "rdm_random_preactivation_grad_rms" in record
                else:
                    assert "rdm_to_reconstruction_grad_ratio" not in record
                    assert "rdm_axis_preactivation_grad_rms" not in record
    if model_type == "batch_topk_sae" or rate_activation or num_local_views == 0:
        config.train.resume_from = str(checkpoint)
        config.train.max_steps = 3
        resumed = train(config)
        resumed_state = torch.load(resumed, map_location="cpu", weights_only=False)
        assert resumed_state["step"] == 3
        assert resumed_state["config"]["loss"]["rdm_wasserstein_power"] == rdm_power
        assert resumed_state["config"]["loss"]["rdm_axis_wasserstein_power"] == axis_power
