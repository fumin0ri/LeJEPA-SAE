#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/pythia-6.9b-layer16.yaml}"
feature_dim="${FEATURE_DIM:-16384}"
expected_l0_fraction="${EXPECTED_L0_FRACTION:-0.009765625}"
feature_activation="${FEATURE_ACTIVATION:-relu}"
leaky_backward_slope="${LEAKY_BACKWARD_SLOPE:-0.01}"
dimension_keep_fraction="${DIMENSION_KEEP_FRACTION:-0.5}"
mask_scaling="${MASK_SCALING:-inverted}"
axis_projections="${AXIS_PROJECTIONS:-512}"
axis_weight="${AXIS_WEIGHT:-1.0}"
rate_weight="${RATE_WEIGHT:-0}"
rate_temperature="${RATE_TEMPERATURE:-0.1}"
rate_scale_floor="${RATE_SCALE_FLOOR:-0.000001}"
rate_gradient_diagnostics="${RATE_GRAD_DIAGNOSTICS:-false}"
activation_suffix=""
if [[ "$feature_activation" != "relu" ]]; then
  activation_suffix="-$feature_activation-s$leaky_backward_slope"
fi
mask_suffix=""
if [[ "$mask_scaling" != "inverted" || "$dimension_keep_fraction" != "0.5" ]]; then
  mask_suffix="-q$dimension_keep_fraction-mask-$mask_scaling"
fi
rate_suffix=""
if [[ "$rate_weight" != "0" && "$rate_weight" != "0.0" ]]; then
  rate_suffix="-rate$rate_weight-tau$rate_temperature-floor$rate_scale_floor"
fi
output_dir="${2:-runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed-d$feature_dim-l0-$expected_l0_fraction-axis$axis_projections$activation_suffix$mask_suffix$rate_suffix}"
batch_size="${BATCH_SIZE:-512}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
eval_batches="${EVAL_BATCHES:-12}"
max_step_args=()
seed_args=()
if [[ -n "${TRAIN_SEED:-}" ]]; then
  seed_args=(--set "train.seed=$TRAIN_SEED")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  max_step_args=(
    --set "train.max_steps=$MAX_STEPS"
    --set "train.checkpoint_every=$MAX_STEPS"
  )
fi

lejepa-train --config "$config" \
  --set model.type=proposed \
  --set "model.feature_dim=$feature_dim" \
  --set "loss.expected_l0_fraction=$expected_l0_fraction" \
  --set "model.feature_activation=$feature_activation" \
  --set "model.leaky_backward_slope=$leaky_backward_slope" \
  --set "model.dimension_keep_fraction=$dimension_keep_fraction" \
  --set "model.mask_scaling=$mask_scaling" \
  --set "loss.axis_projections=$axis_projections" \
  --set "loss.axis_weight=$axis_weight" \
  --set "loss.rate_weight=$rate_weight" \
  --set "loss.rate_temperature=$rate_temperature" \
  --set "loss.rate_scale_floor=$rate_scale_floor" \
  --set "loss.rate_gradient_diagnostics=$rate_gradient_diagnostics" \
  --set "train.batch_size=$batch_size" \
  --set "train.gradient_accumulation_steps=$gradient_accumulation_steps" \
  --set "train.eval_batches=$eval_batches" \
  "${max_step_args[@]}" \
  "${seed_args[@]}" \
  --set "train.output_dir=$output_dir" \
  --set train.resume_from=null
