"""Helpers for inspecting, trimming and resuming an on-disk activation extraction.

Shards are the source of truth: each one is written atomically, while manifest.json is written
only at the end of an extraction. Per-document metadata is reused from the manifest when it lists
a shard, and replaced with one placeholder sequence per shard otherwise (exact for window_size=1).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

SPLITS = ("train", "validation", "test")
_SHARD_PATTERN = re.compile(r"shard-(\d+)\.safetensors$")


def shard_number(file: str | Path) -> int:
    match = _SHARD_PATTERN.search(Path(file).name)
    if match is None:
        raise ValueError(f"Not a shard file name: {file}")
    return int(match.group(1))


def load_manifest(root: Path) -> dict[str, Any] | None:
    path = root / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def scan_shards(root: Path, manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """List shards present on disk, ordered by split then shard number."""
    known = {shard["file"]: shard for shard in (manifest or {}).get("shards", [])}
    shards: list[dict[str, Any]] = []
    for split in SPLITS:
        paths = sorted((root / split).glob("shard-*.safetensors"), key=shard_number)
        for path in paths:
            relative = path.relative_to(root).as_posix()
            with safe_open(str(path), framework="pt") as handle:
                rows = handle.get_slice("activations").get_shape()[0]
                ids = handle.get_slice("token_ids").get_shape()[0]
            if rows != ids:
                raise ValueError(f"{path}: {rows} activations but {ids} token_ids")
            previous = known.get(relative)
            if previous is not None and int(previous["num_tokens"]) == rows:
                shards.append(previous)
                continue
            shards.append(
                {
                    "file": relative,
                    "split": split,
                    "num_tokens": rows,
                    "sequences": [
                        {
                            "offset": 0,
                            "length": rows,
                            "document_id": f"rebuilt:{split}/{path.stem}",
                            "segment_index": 0,
                        }
                    ],
                }
            )
    return shards


def tokens_by_split(shards: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for shard in shards:
        counts[shard["split"]] = counts.get(shard["split"], 0) + int(shard["num_tokens"])
    return counts


@dataclass(frozen=True)
class ResumeState:
    shards: list[dict[str, Any]]
    counts: dict[str, int]
    source_tokens: int
    documents: int
    manifest: dict[str, Any] | None


def load_resume_state(root: Path) -> ResumeState:
    """Where an extraction stopped: the source-token position to skip to and shards to keep."""
    manifest = load_manifest(root)
    shards = scan_shards(root, manifest)
    if not shards:
        raise FileNotFoundError(f"No shard-*.safetensors under {root}; nothing to resume")
    counts = tokens_by_split(shards)
    total = sum(counts.values())
    source_tokens, documents = total, 0
    # Trust the manifest's own counters only if it describes exactly the shards on disk;
    # otherwise orphan shards were written after it and the shard token total is the position.
    if manifest is not None and {s["file"] for s in manifest["shards"]} == {
        s["file"] for s in shards
    }:
        source_tokens = int(manifest.get("source_tokens_processed", total))
        documents = int(manifest.get("documents_processed", 0))
    return ResumeState(shards, counts, source_tokens, documents, manifest)


def plan_trim(
    shards: list[dict[str, Any]], keep_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the leading shards of every split covering >= keep_fraction of its tokens."""
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    keep: list[dict[str, Any]] = []
    drop: list[dict[str, Any]] = []
    for split in SPLITS:
        members = sorted(
            (s for s in shards if s["split"] == split), key=lambda s: shard_number(s["file"])
        )
        target = math.ceil(sum(int(s["num_tokens"]) for s in members) * keep_fraction)
        running = 0
        for shard in members:
            (keep if running < target else drop).append(shard)
            running += int(shard["num_tokens"])
    return keep, drop
