#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/pythia-6.9b-layer16-single-token.yaml}"
output_dir="${2:-runs/the-pile/pythia-6.9b-layer16/single-token/pilot-paper-rdmreg-p1-mu0-b512}"
batch_size="${BATCH_SIZE:-512}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
max_steps="${MAX_STEPS:-10000}"
eval_batches="${EVAL_BATCHES:-12}"

lejepa-train --config "$config" \
  --set model.type=single_token_jepa \
  --set "train.batch_size=$batch_size" \
  --set "train.gradient_accumulation_steps=$gradient_accumulation_steps" \
  --set "train.max_steps=$max_steps" \
  --set "train.eval_batches=$eval_batches" \
  --set "train.checkpoint_every=$max_steps" \
  --set "train.output_dir=$output_dir" \
  --set train.resume_from=null
