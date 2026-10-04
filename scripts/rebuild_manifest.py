#!/usr/bin/env python
"""Rebuild a missing manifest.json from the shard files of an interrupted extraction.

lejepa-extract writes manifest.json only at the very end, so a crashed or copied-without-it
directory has valid shards (each is written atomically) but no manifest. The per-document
boundaries live only in the manifest, so each shard becomes ONE pseudo-sequence. That is exact
for window_size=1 (every token is an independent window; splits are already separate
directories, so train/validation/test stay document-disjoint). document_id / segment_index are
placeholders, so do not use this for document-level analyses (e.g. top-activating contexts).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from safetensors import safe_open

SPLITS = ("train", "validation", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activation_dir")
    parser.add_argument("--model", default="EleutherAI/pythia-6.9b")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    root = Path(args.activation_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"Refusing to overwrite {manifest_path}")
    shards, counts, d_llm = [], {}, None
    for split in SPLITS:
        for path in sorted((root / split).glob("shard-*.safetensors")):
            with safe_open(str(path), framework="pt") as handle:
                rows, width = handle.get_slice("activations").get_shape()
                ids = handle.get_slice("token_ids").get_shape()[0]
            if rows != ids:
                raise SystemExit(f"{path}: {rows} activations but {ids} token_ids")
            if d_llm not in (None, width):
                raise SystemExit(f"{path}: width {width} differs from {d_llm}")
            d_llm = width
            counts[split] = counts.get(split, 0) + rows
            shards.append({
                "file": path.relative_to(root).as_posix(),
                "split": split,
                "num_tokens": rows,
                "sequences": [{
                    "offset": 0, "length": rows,
                    "document_id": f"rebuilt:{split}/{path.stem}", "segment_index": 0,
                }],
            })
    if not shards:
        raise SystemExit(f"No shard-*.safetensors under {root}/{{{','.join(SPLITS)}}}")
    missing = [s for s in SPLITS if s not in counts]
    if missing:
        print(f"WARNING: no shards for split(s): {missing}")
    manifest = {
        "format_version": 1, "created_unix": time.time(), "rebuilt_from_shards": True,
        "model": args.model, "revision": args.revision,
        "hook_point": f"block_output:{args.layer}", "layer": args.layer, "d_llm": int(d_llm),
        "dtype": args.dtype, "context_length": args.context_length, "minimum_window_size": 1,
        "tokens_by_split": counts, "shards": shards,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}: {len(shards)} shards, d_llm={d_llm}, tokens={counts}")


if __name__ == "__main__":
    main()
