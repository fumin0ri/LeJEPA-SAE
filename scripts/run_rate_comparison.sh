#!/usr/bin/env bash
set -euo pipefail

# Only the existing proposed model and its +rate ablation; no SAE baseline training.
mode="${1:-both}"
case "$mode" in
  base|rate|both) ;;
  *) echo "Usage: bash scripts/run_rate_comparison.sh [base|rate|both]" >&2; exit 2 ;;
esac
config="${CONFIG:-configs/pythia-6.9b-layer16.yaml}"
export EXPECTED_L0_FRACTION="${EXPECTED_L0_FRACTION:-0.05}"
export LEAKY_BACKWARD_SLOPE="${LEAKY_BACKWARD_SLOPE:-0.1}"
export MASK_SCALING="${MASK_SCALING:-sqrt}"
export DIMENSION_KEEP_FRACTION="${DIMENSION_KEEP_FRACTION:-0.5}"
export FEATURE_DIM="${FEATURE_DIM:-16384}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export RATE_TEMPERATURE="${RATE_TEMPERATURE:-0.1}"
export RATE_SCALE_FLOOR="${RATE_SCALE_FLOOR:-0.000001}"
export RATE_GRAD_DIAGNOSTICS="${RATE_GRAD_DIAGNOSTICS:-true}"
rate_weight="${RATE_WEIGHT:-1.0}"  # Pilot starting value, not a calibrated optimum.
if [[ "$mode" != "base" ]]; then
  python -c 'import math, sys; w = float(sys.argv[1]); sys.exit(0 if math.isfinite(w) and w > 0 else "RATE_WEIGHT must be finite and positive for the +rate run")' "$rate_weight"
fi
root_dir="${RATE_COMPARISON_ROOT:-runs/rate-ablation/d$FEATURE_DIM-rho$EXPECTED_L0_FRACTION-q$DIMENSION_KEEP_FRACTION-$MASK_SCALING-slope$LEAKY_BACKWARD_SLOPE-rate$rate_weight-tau$RATE_TEMPERATURE-floor$RATE_SCALE_FLOOR-seed$TRAIN_SEED-steps$MAX_STEPS}"

# Check all selected destinations before starting either run. Never append a fresh
# run to an old metrics file or overwrite an earlier checkpoint in this launcher.
conditions=()
[[ "$mode" == "rate" ]] || conditions+=(base)
[[ "$mode" == "base" ]] || conditions+=(rate)
for condition in "${conditions[@]}"; do
  if [[ -e "$root_dir/$condition" ]]; then
    echo "Refusing existing output: $root_dir/$condition. Set RATE_COMPARISON_ROOT to a fresh path." >&2
    exit 1
  fi
done
for condition in "${conditions[@]}"; do
  weight=0
  [[ "$condition" == "base" ]] || weight="$rate_weight"
  RATE_WEIGHT="$weight" bash "$(dirname "$0")/run_leaky_backward.sh" "$config" "$root_dir/$condition"
done
