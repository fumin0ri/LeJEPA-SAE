#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/pythia-6.9b-layer16.yaml}"
feature_dim="${FEATURE_DIM:-16384}"
expected_l0_fraction="${EXPECTED_L0_FRACTION:-0.009765625}"
axis_projections="${AXIS_PROJECTIONS:-512}"
axis_weight="${AXIS_WEIGHT:-1.0}"
output_dir="${2:-runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed-d$feature_dim-l0-$expected_l0_fraction-axis$axis_projections}"
batch_size="${BATCH_SIZE:-512}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
max_steps="${MAX_STEPS:-10000}"
eval_batches="${EVAL_BATCHES:-12}"

lejepa-train --config "$config" \
  --set model.type=proposed \
  --set "model.feature_dim=$feature_dim" \
  --set "loss.expected_l0_fraction=$expected_l0_fraction" \
  --set "loss.axis_projections=$axis_projections" \
  --set "loss.axis_weight=$axis_weight" \
  --set "train.batch_size=$batch_size" \
  --set "train.gradient_accumulation_steps=$gradient_accumulation_steps" \
  --set "train.max_steps=$max_steps" \
  --set "train.eval_batches=$eval_batches" \
  --set "train.checkpoint_every=$max_steps" \
  --set "train.output_dir=$output_dir" \
  --set train.resume_from=null
