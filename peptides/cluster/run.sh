#!/bin/bash
# One training run per amlt job:  bash cluster/run.sh <name> [digin.train flags]
set -eo pipefail
NAME=$1; shift
echo "=== START $(date) host=$(hostname) name=${NAME} args='$*' ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
OUT=${AMLT_OUTPUT_DIR:-$PWD/runs}/${NAME}
mkdir -p "$OUT"
python -u -m digin.train --out "$OUT" --wandb_name "$NAME" "$@"
echo "=== DONE $(date) ==="
