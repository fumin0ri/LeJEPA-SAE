#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"
config="${CONFIG:-configs/pythia-6.9b-layer16.yaml}"
root_dir="${COMPARISON_ROOT:-runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/comparison-d16384-l0-160}"
cache_dir="${PROBE_CACHE:-data/sae-probes/pythia-6.9b-layer16}"
pilot_steps="${PILOT_STEPS:-20000}"

python -m lejepa_sae.pipeline "$mode" \
  --config "$config" \
  --root-dir "$root_dir" \
  --model-cache-path "$cache_dir" \
  --pilot-steps "$pilot_steps"
