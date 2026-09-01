#!/usr/bin/env bash
set -euo pipefail

# 100M width-4096 bf16 activations require about 763 GiB before filesystem
# overhead. Source tokens are consumed once and the final document is truncated
# so the configured budget is not exceeded.
# The first raw Pile train shard contains far more than the 100M-token default
# budget, so stream only that shard unless a local path/glob is supplied.
data_files="${DATA_FILES:-https://mystic.the-eye.eu/public/AI/pile/train/00.jsonl.zst}"
max_source_tokens="${MAX_SOURCE_TOKENS:-100000000}"
output_dir="${OUTPUT_DIR:-data/the-pile/pythia-6.9b/layer-16-ctx1024-100m}"

lejepa-extract \
  --dataset json \
  --data-files "$data_files" \
  --source-split train \
  --text-column text \
  --model EleutherAI/pythia-6.9b \
  --revision main \
  --layer 16 \
  --context-length 1024 \
  --window-size 1 \
  --dtype bfloat16 \
  --shard-tokens 50000 \
  --max-source-tokens "$max_source_tokens" \
  --output-dir "$output_dir"
