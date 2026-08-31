#!/usr/bin/env bash
set -euo pipefail

# Each official random-sample row contains 64 Pythia tokens. 10,000 rows are
# 640k source tokens and about 4.9 GiB of width-4096 bf16 residuals.
max_sequences="${MAX_SEQUENCES:-10000}"

lejepa-extract \
  --dataset EleutherAI/pile-duped-pythia-random-sampled \
  --dataset-revision 49487e95e42f4532534e8d7d8bc17d42795b5af8 \
  --source-split train \
  --token-ids-column Tokens \
  --id-column Index \
  --model EleutherAI/pythia-6.9b \
  --revision main \
  --layer 16 \
  --context-length 512 \
  --window-size 10 \
  --dtype bfloat16 \
  --shard-tokens 50000 \
  --max-documents "$max_sequences" \
  --output-dir data/the-pile/pythia-6.9b/layer-16
