#!/usr/bin/env bash
set -euo pipefail

# Fresh standalone pilot; never launch comparison runs or reuse JEPA checkpoints.
if (( $# > 1 )); then
  echo "Usage: bash scripts/run_rdm_sae.sh [OUTPUT_DIR]" >&2
  exit 2
fi
config="${CONFIG:-$(dirname "$0")/../configs/pythia-6.9b-layer16-rdm-sae.yaml}"
feature_dim="${FEATURE_DIM:-16384}"
rho="${EXPECTED_L0_FRACTION:-0.05}"
activation="${FEATURE_ACTIVATION:-relu_forward_leaky_backward}"
slope="${LEAKY_BACKWARD_SLOPE:-0.1}"
rdm_weight="${RDM_WEIGHT:-1.0}"
reconstruction_weight="${RECONSTRUCTION_WEIGHT:-1.0}"
target_scale="${RDM_TARGET_SCALE:-1.0}"
wasserstein_power="${RDM_WASSERSTEIN_POWER:-2}"
axis_weight="${AXIS_WEIGHT:-1.0}"
if [[ "$wasserstein_power" != 1 && "$wasserstein_power" != 2 ]]; then
  echo "RDM_WASSERSTEIN_POWER must be 1 or 2" >&2
  exit 2
fi
random_power="${RDM_RANDOM_WASSERSTEIN_POWER:-$wasserstein_power}"
axis_power="${RDM_AXIS_WASSERSTEIN_POWER:-$wasserstein_power}"
if [[ "$random_power" != 1 && "$random_power" != 2 ]]; then
  echo "RDM_RANDOM_WASSERSTEIN_POWER must be 1 or 2" >&2
  exit 2
fi
if [[ "$axis_power" != 1 && "$axis_power" != 2 ]]; then
  echo "RDM_AXIS_WASSERSTEIN_POWER must be 1 or 2" >&2
  exit 2
fi
metric_tag="wp${random_power}"
if [[ "$random_power" != "$axis_power" ]]; then
  metric_tag="wpr${random_power}-wpa${axis_power}"
fi
seed="${SEED:-42}"
steps="${MAX_STEPS:-10000}"
output_dir="${1:-${OUTPUT_DIR:-runs/rdm-sae/d${feature_dim}-rho${rho}-${activation}-slope${slope}-rec${reconstruction_weight}-rdm${rdm_weight}-scale${target_scale}-${metric_tag}-axis${axis_weight}-seed${seed}-steps${steps}}}"
if [[ ! -f "$config" ]]; then
  echo "Missing config: $config" >&2
  exit 1
fi
if [[ -e "$output_dir" || -L "$output_dir" ]]; then
  echo "Refusing existing output: $output_dir. Choose a fresh OUTPUT_DIR." >&2
  exit 1
fi

exec lejepa-train --config "$config" \
  --set model.type=rdm_sae \
  --set model.num_local_views=0 \
  --set "model.feature_dim=$feature_dim" \
  --set "model.feature_activation=$activation" \
  --set "model.leaky_backward_slope=$slope" \
  --set loss.invariance_weight=0.0 \
  --set loss.rate_weight=0.0 \
  --set loss.rate_gradient_diagnostics=false \
  --set "loss.reconstruction_weight=$reconstruction_weight" \
  --set "loss.lambda_rdm=$rdm_weight" \
  --set "loss.rdm_target_scale=$target_scale" \
  --set "loss.rdm_wasserstein_power=$wasserstein_power" \
  --set "loss.rdm_random_wasserstein_power=${RDM_RANDOM_WASSERSTEIN_POWER:-null}" \
  --set "loss.rdm_axis_wasserstein_power=${RDM_AXIS_WASSERSTEIN_POWER:-null}" \
  --set "loss.expected_l0_fraction=$rho" \
  --set "loss.rdm_projections=${RDM_PROJECTIONS:-8192}" \
  --set "loss.axis_projections=${AXIS_PROJECTIONS:-512}" \
  --set "loss.axis_weight=$axis_weight" \
  --set "loss.rdm_gradient_diagnostics=${RDM_GRADIENT_DIAGNOSTICS:-true}" \
  --set "train.batch_size=${BATCH_SIZE:-512}" \
  --set "train.gradient_accumulation_steps=${GRAD_ACCUM:-1}" \
  --set "train.eval_batches=${EVAL_BATCHES:-12}" \
  --set "train.seed=$seed" \
  --set "train.max_steps=$steps" \
  --set "train.checkpoint_every=${CHECKPOINT_EVERY:-10000}" \
  --set train.resume_from=null \
  --set "train.output_dir=$output_dir"
