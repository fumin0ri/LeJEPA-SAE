#!/usr/bin/env bash
set -euo pipefail

export FEATURE_ACTIVATION=relu_forward_leaky_backward
export LEAKY_BACKWARD_SLOPE="${LEAKY_BACKWARD_SLOPE:-0.01}"

exec bash "$(dirname "$0")/run_proposed.sh" "$@"
