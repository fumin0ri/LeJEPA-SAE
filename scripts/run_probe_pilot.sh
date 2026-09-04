#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_probe_pilot.sh smoke|probe|dense-smoke|dense RUN_DIR [RUN_DIR ...]
if (( $# < 2 )); then
  echo "Usage: bash scripts/run_probe_pilot.sh smoke|probe|dense-smoke|dense RUN_DIR [RUN_DIR ...]" >&2
  exit 2
fi
mode="$1"
shift
case "$mode" in
  smoke)
    module="lejepa_sae.probing"
    extra_args=(--ks 1 16 --smoke-test)
    suffix="probe-smoke-k1-k16"
    ;;
  probe)
    module="lejepa_sae.probing"
    extra_args=(--ks 1 16)
    suffix="probes-normal-k1-k16"
    ;;
  dense-smoke)
    module="lejepa_sae.dense_probing"
    extra_args=(--smoke-test)
    suffix="dense-z-gpu-smoke"
    ;;
  dense)
    module="lejepa_sae.dense_probing"
    extra_args=()
    suffix="dense-z-gpu-normal"
    ;;
  *) echo "Unknown mode: $mode (expected smoke, probe, dense-smoke, or dense)" >&2; exit 2 ;;
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
  python -m "$module" \
    --config "$run_dir/config.resolved.yaml" \
    --checkpoint "$run_dir/checkpoint-00010000.pt" \
    --results-path "$run_dir/$suffix" \
    --model-cache-path "${PROBE_CACHE:-data/sae-probes/pythia-6.9b-layer16}" \
    --llm-precision "${LLM_PRECISION:-auto}" \
    --activation-batch-size "${ACTIVATION_BATCH_SIZE:-1}" \
    --max-seq-len 1024 \
    "${extra_args[@]}"
done
