#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/pythia-6.9b-layer16-single-token.yaml}"
output_dir="${2:-runs/the-pile/pythia-6.9b-layer16/single-token/paper-rdmreg-p1-mu0}"

lejepa-train --config "$config" \
  --set model.type=single_token_jepa \
  --set "train.output_dir=$output_dir" \
  --set train.resume_from=null
