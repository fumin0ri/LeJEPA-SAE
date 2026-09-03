#!/usr/bin/env bash
set -euo pipefail

# Inherit the chosen run's encoder, data, RDMReg, optimizer and batch settings.
if (( $# < 1 || $# > 2 )); then
  echo "Usage: bash scripts/run_global_rdm_only.sh BASE_RUN_DIR [OUTPUT_DIR]" >&2
  exit 2
fi
base_run="${1%/}"
output_dir="${2:-${base_run}-global-rdm-only-seed42-steps10000}"
config="$base_run/config.resolved.yaml"
if [[ ! -f "$config" ]]; then
  echo "Missing base config: $config" >&2
  exit 1
fi
if [[ -e "$output_dir" || -L "$output_dir" ]]; then
  echo "Refusing existing output: $output_dir. Choose a fresh OUTPUT_DIR." >&2
  exit 1
fi

exec lejepa-train --config "$config" \
  --set model.type=proposed \
  --set model.num_local_views=0 \
  --set loss.invariance_weight=0.0 \
  --set loss.rate_weight=0.0 \
  --set loss.rate_gradient_diagnostics=false \
  --set train.seed=42 \
  --set train.max_steps=10000 \
  --set train.checkpoint_every=10000 \
  --set train.resume_from=null \
  --set "train.output_dir=$output_dir"
