import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from lejepa_sae import extract as extract_module
from lejepa_sae.activation_store import load_resume_state, plan_trim, scan_shards
from lejepa_sae.data import ActivationWindowDataset


def write_shard(root: Path, split: str, index: int, rows: int, width: int = 4) -> None:
    (root / split).mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "activations": torch.zeros(rows, width, dtype=torch.bfloat16),
            "token_ids": torch.arange(rows, dtype=torch.int32),
        },
        str(root / split / f"shard-{index:05d}.safetensors"),
    )


def test_scan_shards_orders_numerically_and_uses_placeholder_metadata(tmp_path):
    for index in (10, 2, 1):
        write_shard(tmp_path, "train", index, 3)
    shards = scan_shards(tmp_path)
    assert [s["file"] for s in shards] == [
        "train/shard-00001.safetensors",
        "train/shard-00002.safetensors",
        "train/shard-00010.safetensors",
    ]
    assert shards[0]["sequences"][0]["length"] == 3


def test_plan_trim_keeps_leading_shards_per_split(tmp_path):
    for index in range(4):
        write_shard(tmp_path, "train", index, 10)
    for index in range(2):
        write_shard(tmp_path, "validation", index, 5)
    keep, drop = plan_trim(scan_shards(tmp_path), 0.5)
    assert [s["file"] for s in keep] == [
        "train/shard-00000.safetensors",
        "train/shard-00001.safetensors",
        "validation/shard-00000.safetensors",
    ]
    assert len(drop) == 3
    assert plan_trim(scan_shards(tmp_path), 1.0)[1] == []
    with pytest.raises(ValueError):
        plan_trim(scan_shards(tmp_path), 0)


def test_resume_state_without_manifest_uses_shard_tokens(tmp_path):
    write_shard(tmp_path, "train", 0, 7)
    write_shard(tmp_path, "test", 0, 2)
    state = load_resume_state(tmp_path)
    assert state.source_tokens == 9
    assert state.counts == {"train": 7, "test": 2}
    with pytest.raises(FileNotFoundError):
        load_resume_state(tmp_path / "empty")


def test_resume_state_prefers_manifest_counters_only_when_it_matches_disk(tmp_path):
    write_shard(tmp_path, "train", 0, 6)
    shards = scan_shards(tmp_path)
    manifest = {"shards": shards, "source_tokens_processed": 8, "documents_processed": 3}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_resume_state(tmp_path).source_tokens == 8
    write_shard(tmp_path, "train", 1, 4)  # orphan shard written after the manifest
    assert load_resume_state(tmp_path).source_tokens == 10


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        return {"input_ids": torch.tensor([[ord(c) % 50 for c in text]])}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=50, hidden_size=4)
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.embed = torch.nn.Embedding(50, 4)

    def to(self, *_args, **_kwargs):
        return self

    def forward(self, input_ids, use_cache=False):
        hidden = self.embed(input_ids)
        self.layers[0](hidden)
        return hidden


def run_extract(monkeypatch, tmp_path, documents, max_source_tokens, resume=False):
    model = FakeModel()
    monkeypatch.setattr(extract_module, "load_dataset", lambda *a, **k: iter(documents))
    monkeypatch.setattr(
        extract_module.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setattr(extract_module.AutoModel, "from_pretrained", lambda *a, **k: model)
    argv = [
        "lejepa-extract", "--dataset", "json", "--device", "cpu", "--dtype", "float32",
        "--context-length", "8", "--shard-tokens", "10", "--layer", "0",
        "--max-source-tokens", str(max_source_tokens), "--output-dir", str(tmp_path),
    ]  # fmt: skip
    if resume:
        argv.append("--resume")
    monkeypatch.setattr(sys, "argv", argv)
    return extract_module.extract(extract_module.parse_args())


def test_resume_extends_without_duplicating_or_overwriting(monkeypatch, tmp_path):
    documents = [{"text": f"document number {i:03d}"} for i in range(30)]  # 19 tokens each
    run_extract(monkeypatch, tmp_path, documents, 190)
    first = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert first["source_tokens_processed"] == 190
    first_files = {s["file"] for s in first["shards"]}
    first_bytes = {f: (tmp_path / f).read_bytes() for f in first_files}

    with pytest.raises(FileExistsError, match="--resume"):
        run_extract(monkeypatch, tmp_path, documents, 380)
    run_extract(monkeypatch, tmp_path, documents, 380, resume=True)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_tokens_processed"] == 380
    assert manifest["resumed_from_source_tokens"] == 190
    assert manifest["documents_processed"] == 20
    assert sum(manifest["tokens_by_split"].values()) == 380
    assert first_files < {s["file"] for s in manifest["shards"]}
    assert all((tmp_path / f).read_bytes() == data for f, data in first_bytes.items())
    assert list(tmp_path.glob("manifest.pre-resume-*.json"))

    # The resumed directory reads like one extraction: no document id appears twice.
    ids = [
        seq["document_id"]
        for shard in manifest["shards"]
        for seq in shard["sequences"]
        if seq["segment_index"] == 0
    ]
    assert len(ids) == len(set(ids)) == 20
    assert len(ActivationWindowDataset(tmp_path, "train")) == manifest["tokens_by_split"]["train"]


def test_resume_rejects_a_budget_not_above_what_exists(monkeypatch, tmp_path):
    documents = [{"text": f"document number {i:03d}"} for i in range(30)]
    run_extract(monkeypatch, tmp_path, documents, 190)
    with pytest.raises(ValueError, match="not above"):
        run_extract(monkeypatch, tmp_path, documents, 190, resume=True)


def test_resume_rejects_a_different_layer(monkeypatch, tmp_path):
    documents = [{"text": f"document number {i:03d}"} for i in range(30)]
    run_extract(monkeypatch, tmp_path, documents, 190)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["layer"] = 5
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="layer"):
        run_extract(monkeypatch, tmp_path, documents, 380, resume=True)
