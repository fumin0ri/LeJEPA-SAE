#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/pythia-6.9b-layer16.yaml}"

for baseline in standard_sae window_autoencoder sparse_jepa_full_view jepa_sigreg; do
  lejepa-train --config "$config" \
    --set "model.type=$baseline" \
    --set "train.output_dir=runs/the-pile/pythia-6.9b-layer16/$baseline"
done
