#!/usr/bin/env bash
set -euo pipefail

# 100M width-4096 bf16 activations require about 763 GiB before filesystem
# overhead. Source tokens are consumed once and the final document is truncated
# so the configured budget is not exceeded.
# Use a Hugging Face-hosted Parquet mirror so streaming does not depend on the
# external The Eye server. Supplying DATA_FILES switches to local JSON shards.
dataset="${DATASET:-monology/pile-uncopyrighted-parquet}"
data_file_args=()
if [[ -n "${DATA_FILES:-}" ]]; then
  dataset="${DATASET:-json}"
  data_file_args=(--data-files "$DATA_FILES")
fi
max_source_tokens="${MAX_SOURCE_TOKENS:-100000000}"
output_dir="${OUTPUT_DIR:-data/the-pile/pythia-6.9b/layer-16-ctx1024-100m}"

lejepa-extract \
  --dataset "$dataset" \
  "${data_file_args[@]}" \
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
