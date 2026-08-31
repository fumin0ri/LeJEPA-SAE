from __future__ import annotations

import bisect
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset, Sampler

SPLITS = ("train", "validation", "test")


def document_split(
    document_id: str,
    validation_fraction: float = 0.01,
    test_fraction: float = 0.01,
    seed: int = 42,
) -> str:
    """Stable document-level split; all segments/windows from a document stay together."""
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions cannot be negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than one")
    digest = hashlib.blake2b(f"{seed}:{document_id}".encode(), digest_size=8).digest()
    unit = int.from_bytes(digest, "big") / 2**64
    if unit < test_fraction:
        return "test"
    if unit < test_fraction + validation_fraction:
        return "validation"
    return "train"


@dataclass(frozen=True)
class SequenceRecord:
    shard_path: Path
    offset: int
    length: int
    document_id: str
    segment_index: int
    window_count: int


class ActivationWindowDataset(Dataset[dict[str, Any]]):
    """Map-style windows over compact, non-overlapping activation sequences."""

    def __init__(
        self,
        activation_dir: str | Path,
        split: str,
        window_size: int = 10,
        stride: int = 1,
        cache_shards: int = 2,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if window_size < 1 or stride < 1:
            raise ValueError("window_size and stride must be positive")
        self.root = Path(activation_dir)
        self.window_size = window_size
        self.stride = stride
        self.cache_shards = max(cache_shards, 1)
        self._cache: OrderedDict[Path, dict[str, torch.Tensor]] = OrderedDict()

        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Activation manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["d_llm"]) < 1:
            raise ValueError("Invalid d_llm in manifest")
        self.d_llm = int(manifest["d_llm"])
        self.metadata = manifest
        self.records: list[SequenceRecord] = []
        self.cumulative_windows: list[int] = []
        running = 0
        for shard in manifest["shards"]:
            if shard["split"] != split:
                continue
            shard_path = self.root / shard["file"]
            for sequence in shard["sequences"]:
                length = int(sequence["length"])
                count = max(0, (length - window_size) // stride + 1)
                if count == 0:
                    continue
                self.records.append(
                    SequenceRecord(
                        shard_path=shard_path,
                        offset=int(sequence["offset"]),
                        length=length,
                        document_id=str(sequence["document_id"]),
                        segment_index=int(sequence["segment_index"]),
                        window_count=count,
                    )
                )
                running += count
                self.cumulative_windows.append(running)

    def __len__(self) -> int:
        return self.cumulative_windows[-1] if self.cumulative_windows else 0

    def _load_shard(self, path: Path) -> dict[str, torch.Tensor]:
        cached = self._cache.pop(path, None)
        if cached is not None:
            self._cache[path] = cached
            return cached
        tensors = load_file(str(path), device="cpu")
        self._cache[path] = tensors
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return tensors

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        record_index = bisect.bisect_right(self.cumulative_windows, index)
        previous = self.cumulative_windows[record_index - 1] if record_index else 0
        within_sequence = index - previous
        record = self.records[record_index]
        start = record.offset + within_sequence * self.stride
        stop = start + self.window_size
        shard = self._load_shard(record.shard_path)
        return {
            "residuals": shard["activations"][start:stop],
            "token_ids": shard["token_ids"][start:stop].long(),
            "positions": torch.arange(self.window_size, dtype=torch.long),
            "document_id": record.document_id,
            "segment_index": record.segment_index,
            "window_start": within_sequence * self.stride,
        }


class ShardAwareRandomSampler(Sampler[int]):
    """Shuffle shards, sequences, and starts without allocating randperm(len(dataset))."""

    def __init__(self, dataset: ActivationWindowDataset, seed: int = 42) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.records_by_shard: dict[Path, list[int]] = {}
        for record_index, record in enumerate(dataset.records):
            self.records_by_shard.setdefault(record.shard_path, []).append(record_index)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shard_paths = list(self.records_by_shard)
        shard_order = torch.randperm(len(shard_paths), generator=generator).tolist()
        for shard_index in shard_order:
            records = self.records_by_shard[shard_paths[shard_index]]
            record_order = torch.randperm(len(records), generator=generator).tolist()
            for order_index in record_order:
                record_index = records[order_index]
                record = self.dataset.records[record_index]
                base = self.dataset.cumulative_windows[record_index - 1] if record_index else 0
                starts = torch.randperm(record.window_count, generator=generator).tolist()
                for start in starts:
                    yield base + start


def validate_document_disjointness(activation_dir: str | Path) -> None:
    manifest = json.loads((Path(activation_dir) / "manifest.json").read_text(encoding="utf-8"))
    seen: dict[str, str] = {}
    for shard in manifest["shards"]:
        split = shard["split"]
        for sequence in shard["sequences"]:
            document_id = str(sequence["document_id"])
            previous = seen.setdefault(document_id, split)
            if previous != split:
                raise ValueError(
                    f"Document {document_id!r} occurs in both {previous!r} and {split!r}"
                )
