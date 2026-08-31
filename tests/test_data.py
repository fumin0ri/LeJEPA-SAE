import json

import torch
from safetensors.torch import save_file

from lejepa_sae.data import (
    ActivationWindowDataset,
    ShardAwareRandomSampler,
    document_split,
    validate_document_disjointness,
)


def make_activation_store(tmp_path):
    (tmp_path / "train").mkdir()
    activations = torch.arange(13 * 4).reshape(13, 4).float()
    token_ids = torch.arange(13, dtype=torch.int32)
    save_file(
        {"activations": activations, "token_ids": token_ids},
        str(tmp_path / "train" / "shard-00000.safetensors"),
    )
    manifest = {
        "d_llm": 4,
        "shards": [
            {
                "file": "train/shard-00000.safetensors",
                "split": "train",
                "sequences": [
                    {"offset": 0, "length": 6, "document_id": "a", "segment_index": 0},
                    {"offset": 6, "length": 7, "document_id": "b", "segment_index": 0},
                ],
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_windows_never_cross_sequence_boundaries(tmp_path):
    make_activation_store(tmp_path)
    dataset = ActivationWindowDataset(tmp_path, "train", window_size=5, stride=1)
    assert len(dataset) == 5
    assert dataset[1]["token_ids"].tolist() == [1, 2, 3, 4, 5]
    assert dataset[2]["token_ids"].tolist() == [6, 7, 8, 9, 10]
    assert dataset[-1]["token_ids"].tolist() == [8, 9, 10, 11, 12]


def test_shard_aware_sampler_covers_each_window_once(tmp_path):
    make_activation_store(tmp_path)
    dataset = ActivationWindowDataset(tmp_path, "train", window_size=5, stride=1)
    first_epoch = list(ShardAwareRandomSampler(dataset, seed=3))
    assert sorted(first_epoch) == list(range(len(dataset)))


def test_document_split_is_stable():
    split = document_split("same-document", seed=123)
    assert split == document_split("same-document", seed=123)
    assert split in {"train", "validation", "test"}


def test_manifest_disjointness_check_detects_leakage(tmp_path):
    manifest = {
        "d_llm": 4,
        "shards": [
            {
                "file": "a",
                "split": "train",
                "sequences": [{"document_id": "duplicate"}],
            },
            {
                "file": "b",
                "split": "test",
                "sequences": [{"document_id": "duplicate"}],
            },
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    try:
        validate_document_disjointness(tmp_path)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("Expected leakage validation to fail")
