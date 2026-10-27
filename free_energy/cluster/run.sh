#!/bin/bash
# One training run per amlt job:  bash cluster/run.sh <name> [digin_bridge.train flags]
# FEAT's ALDP data is downloaded from Hugging Face to node-local disk; a
# restarted job resumes from its last checkpoint and from the same W&B run.
set -eo pipefail
NAME=$1; shift
echo "=== START $(date) host=$(hostname) name=${NAME} args='$*' ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
OUT=${AMLT_OUTPUT_DIR:-$PWD/runs}/${NAME}
DATA_ROOT=${TMPDIR:-/tmp}/feat_aldp
mkdir -p "$OUT"
for i in 1 2 3; do python -c "from digin_bridge import data; data.ensure('$DATA_ROOT')" && break; sleep 30; done
python -u -m digin_bridge.train --out "$OUT" --data_root "$DATA_ROOT" --wandb_name "$NAME" "$@"
echo "=== DONE $(date) ==="
