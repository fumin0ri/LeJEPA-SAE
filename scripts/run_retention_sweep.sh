#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/pythia-6.9b-layer16.yaml}"

for k in 1 2 3 5 8 10; do
  lejepa-train --config "$config" \
    --set "model.local_tokens=$k" \
    --set "train.output_dir=runs/the-pile/pythia-6.9b-layer16/proposed-k$k"
done
