#!/usr/bin/env python
"""Shrink an activation extraction to its leading shards to free disk space.

Every split keeps its first shards until at least --keep-fraction of that split's tokens are
covered, so train/validation/test stay in the same proportion. manifest.json is rewritten for the
kept shards (the old one is saved as manifest.pre-trim-<unix>.json) BEFORE any shard is deleted.

The deleted tail can be regenerated later, deterministically, with
``lejepa-extract ... --resume --max-source-tokens <original budget>`` (same dataset and seed).

Dry run by default; pass --apply to modify anything. A directory without manifest.json is
refused: run scripts/rebuild_manifest.py first.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lejepa_sae.activation_store import load_manifest, plan_trim, scan_shards, tokens_by_split


def _gib(num_bytes: int) -> str:
    return f"{num_bytes / 2**30:.1f} GiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activation_dir")
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument(
        "--apply", action="store_true", help="Actually write the manifest and delete shards"
    )
    args = parser.parse_args()

    root = Path(args.activation_dir)
    manifest = load_manifest(root)
    if manifest is None:
        raise SystemExit(f"{root}/manifest.json is missing; run scripts/rebuild_manifest.py first")
    shards = scan_shards(root, manifest)
    keep, drop = plan_trim(shards, args.keep_fraction)
    if not drop:
        raise SystemExit("Nothing to trim at this keep-fraction")

    sizes = {s["file"]: (root / s["file"]).stat().st_size for s in drop}
    before, after = tokens_by_split(shards), tokens_by_split(keep)
    print(f"tokens by split: {before} -> {after}")
    print(f"deleting {len(drop)} shards, freeing {_gib(sum(sizes.values()))}")
    if not args.apply:
        print("Dry run. Re-run with --apply to trim.")
        return

    backup = root / f"manifest.pre-trim-{int(time.time())}.json"
    backup.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    trimmed = {k: v for k, v in manifest.items() if k not in ("documents_processed",)}
    trimmed["shards"] = keep
    trimmed["tokens_by_split"] = after
    trimmed["source_tokens_processed"] = sum(after.values())
    trimmed["trimmed_from_tokens"] = sum(before.values())
    trimmed["trim_keep_fraction"] = args.keep_fraction
    tmp = root / "manifest.json.tmp"
    tmp.write_text(json.dumps(trimmed, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(root / "manifest.json")
    for shard in drop:
        (root / shard["file"]).unlink()
    print(f"Trimmed. Old manifest saved as {backup.name}")


if __name__ == "__main__":
    main()
