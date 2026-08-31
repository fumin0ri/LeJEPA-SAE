#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/pythia-6.9b-layer16-single-token.yaml}"

for model_type in standard_sae dimension_denoising_sae single_token_jepa; do
  lejepa-train --config "$config" \
    --set "model.type=$model_type" \
    --set "train.output_dir=runs/the-pile/pythia-6.9b-layer16/single-token/$model_type"
done
