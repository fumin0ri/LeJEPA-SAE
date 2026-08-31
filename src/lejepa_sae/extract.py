from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .data import document_split


class ShardWriter:
    def __init__(self, output_dir: Path, shard_tokens: int) -> None:
        self.output_dir = output_dir
        self.shard_tokens = shard_tokens
        self.pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.pending_tokens: dict[str, int] = defaultdict(int)
        self.shard_indices: dict[str, int] = defaultdict(int)
        self.shards: list[dict[str, Any]] = []

    def add(
        self,
        split: str,
        activations: torch.Tensor,
        token_ids: torch.Tensor,
        document_id: str,
        segment_index: int,
    ) -> None:
        self.pending[split].append(
            {
                "activations": activations.contiguous(),
                "token_ids": token_ids.contiguous(),
                "document_id": document_id,
                "segment_index": segment_index,
            }
        )
        self.pending_tokens[split] += token_ids.numel()
        if self.pending_tokens[split] >= self.shard_tokens:
            self.flush(split)

    def flush(self, split: str) -> None:
        sequences = self.pending[split]
        if not sequences:
            return
        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        index = self.shard_indices[split]
        relative_path = Path(split) / f"shard-{index:05d}.safetensors"
        final_path = self.output_dir / relative_path
        temporary_path = final_path.with_suffix(".tmp")

        activations = torch.cat([item["activations"] for item in sequences])
        token_ids = torch.cat([item["token_ids"] for item in sequences]).to(torch.int32)
        save_file({"activations": activations, "token_ids": token_ids}, str(temporary_path))
        temporary_path.replace(final_path)

        offset = 0
        sequence_metadata = []
        for item in sequences:
            length = int(item["token_ids"].numel())
            sequence_metadata.append(
                {
                    "offset": offset,
                    "length": length,
                    "document_id": item["document_id"],
                    "segment_index": item["segment_index"],
                }
            )
            offset += length
        self.shards.append(
            {
                "file": relative_path.as_posix(),
                "split": split,
                "num_tokens": offset,
                "sequences": sequence_metadata,
            }
        )
        self.pending[split] = []
        self.pending_tokens[split] = 0
        self.shard_indices[split] += 1

    def finish(self) -> list[dict[str, Any]]:
        for split in list(self.pending):
            self.flush(split)
        return self.shards


def _layer_stack(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidates = (
        (None, "layers"),
        ("gpt_neox", "layers"),
        ("model", "layers"),
        ("transformer", "h"),
    )
    for parent_name, child_name in candidates:
        parent = model if parent_name is None else getattr(model, parent_name, None)
        layers = getattr(parent, child_name, None) if parent is not None else None
        if layers is not None:
            return layers
    raise ValueError("Could not locate transformer blocks on this model architecture")


def _document_identifier(
    example: dict[str, Any], text: str, index: int, id_column: str | None
) -> str:
    if id_column and id_column in example:
        return str(example[id_column])
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=12).hexdigest()
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract split-safe Pythia residual shards")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument(
        "--data-files",
        action="append",
        default=None,
        help="Local/remote data file or glob (repeatable; useful with --dataset json)",
    )
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--token-ids-column",
        default=None,
        help="Use pre-tokenized IDs from this column instead of tokenizing text",
    )
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--model", default="EleutherAI/pythia-6.9b")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--layer", type=int, default=16, help="Zero-based block index")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--shard-tokens", type=int, default=50_000)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def extract(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing extraction: {output_dir}")
    if args.context_length < args.window_size:
        raise ValueError("context-length must be at least window-size")

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    layers = _layer_stack(model)
    if not 0 <= args.layer < len(layers):
        raise ValueError(f"layer must be in [0, {len(layers) - 1}]")

    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach())

    handle = layers[args.layer].register_forward_hook(hook)
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        data_files=args.data_files,
        revision=args.dataset_revision,
        split=args.source_split,
        streaming=args.streaming,
    )
    writer = ShardWriter(output_dir, args.shard_tokens)
    counts: dict[str, int] = defaultdict(int)
    processed_documents = 0
    progress = tqdm(dataset, total=args.max_documents, desc="extracting documents")
    try:
        for index, example in enumerate(progress):
            if args.max_documents is not None and processed_documents >= args.max_documents:
                break
            if args.token_ids_column:
                raw_token_ids = example.get(args.token_ids_column)
                if raw_token_ids is None:
                    continue
                token_ids = torch.as_tensor(raw_token_ids, dtype=torch.long).flatten()
                if token_ids.numel() == 0:
                    continue
                identity = hashlib.blake2b(
                    token_ids.numpy().tobytes(), digest_size=12
                ).hexdigest()
            else:
                text = example.get(args.text_column)
                if not isinstance(text, str) or not text.strip():
                    continue
                identity = text
                token_ids = tokenizer(
                    text, add_special_tokens=False, return_tensors="pt"
                )
                token_ids = token_ids["input_ids"][0]
            document_id = _document_identifier(
                example, identity, index, args.id_column
            )
            split = document_split(
                document_id,
                args.validation_fraction,
                args.test_fraction,
                args.split_seed,
            )
            if token_ids.min() < 0 or token_ids.max() >= model.config.vocab_size:
                raise ValueError(
                    f"Token IDs for source unit {document_id!r} are outside model vocabulary"
                )
            segment_index = 0
            for start in range(0, token_ids.numel(), args.context_length):
                segment = token_ids[start : start + args.context_length]
                if segment.numel() < args.window_size:
                    continue
                captured.clear()
                with torch.inference_mode():
                    model(segment.unsqueeze(0).to(args.device), use_cache=False)
                if len(captured) != 1:
                    raise RuntimeError(f"Expected one layer-hook result, got {len(captured)}")
                activations = captured[0][0].to(device="cpu", dtype=dtype)
                writer.add(split, activations, segment.cpu(), document_id, segment_index)
                counts[split] += int(segment.numel())
                segment_index += 1
            processed_documents += 1
            progress.set_postfix(documents=processed_documents, tokens=sum(counts.values()))
    finally:
        handle.remove()

    shards = writer.finish()
    manifest = {
        "format_version": 1,
        "created_unix": time.time(),
        "model": args.model,
        "revision": args.revision,
        "hook_point": f"block_output:{args.layer}",
        "layer": args.layer,
        "d_llm": int(model.config.hidden_size),
        "dtype": args.dtype,
        "context_length": args.context_length,
        "minimum_window_size": args.window_size,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "data_files": args.data_files,
        "source_split": args.source_split,
        "input_format": "token_ids" if args.token_ids_column else "text",
        "text_column": None if args.token_ids_column else args.text_column,
        "token_ids_column": args.token_ids_column,
        "id_column": args.id_column,
        "split_unit": "source_sequence" if args.token_ids_column else "document",
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "documents_processed": processed_documents,
        "tokens_by_split": dict(counts),
        "shards": shards,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def main() -> None:
    manifest = extract(parse_args())
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
