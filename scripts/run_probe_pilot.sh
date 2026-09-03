#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_probe_pilot.sh smoke|probe RUN_DIR [RUN_DIR ...]
if (( $# < 2 )); then
  echo "Usage: bash scripts/run_probe_pilot.sh smoke|probe RUN_DIR [RUN_DIR ...]" >&2
  exit 2
fi
mode="$1"
shift
case "$mode" in
  smoke) extra_args=(--smoke-test); suffix="probe-smoke-k1-k16" ;;
  probe) extra_args=(); suffix="probes-normal-k1-k16" ;;
  *) echo "Unknown mode: $mode (expected smoke or probe)" >&2; exit 2 ;;
esac

# Check every run before loading any LLM. Never train or change existing checkpoints.
for run_dir in "$@"; do
  for name in config.resolved.yaml checkpoint-00010000.pt; do
    if [[ ! -f "$run_dir/$name" ]]; then
      echo "Missing $run_dir/$name" >&2
      exit 1
    fi
  done
done

for run_dir in "$@"; do
  python -m lejepa_sae.probing \
    --config "$run_dir/config.resolved.yaml" \
    --checkpoint "$run_dir/checkpoint-00010000.pt" \
    --results-path "$run_dir/$suffix" \
    --model-cache-path "${PROBE_CACHE:-data/sae-probes/pythia-6.9b-layer16}" \
    --llm-precision "${LLM_PRECISION:-auto}" \
    --activation-batch-size "${ACTIVATION_BATCH_SIZE:-1}" \
    --max-seq-len 1024 \
    --ks 1 16 \
    "${extra_args[@]}"
done
