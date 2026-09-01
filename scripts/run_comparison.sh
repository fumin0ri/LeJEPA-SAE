#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/pythia-6.9b-layer16.yaml}"

for model_type in standard_sae dimension_denoising_sae proposed; do
  output_name="$model_type"
  if [[ "$model_type" == "proposed" ]]; then
    output_name="proposed-l0-0.009765625-axis512"
  fi
  lejepa-train --config "$config" \
    --set "model.type=$model_type" \
    --set "train.output_dir=runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/d16384/$output_name"
done
