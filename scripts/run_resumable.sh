#!/usr/bin/env bash
# Retry wrapper for one training run on preemptible capacity (optional).
#
#     bash scripts/run_resumable.sh --out runs/lattice/ising25/ordered/digin/s0 \
#         target=ising25_ordered seed=0 ...      # the discrete_sampler.train overrides
#
# Preemption contract, in order:
#   1. The scheduler sends SIGTERM.  discrete_sampler/train.py catches it,
#      checkpoints at the next step boundary and exits 17.
#   2. This wrapper restarts train.py with the SAME run directory.
#   3. train.py finds the checkpoint there and continues from that step.
#
# The guard that matters is the NO-PROGRESS one: restart on any failure, but
# give up after N consecutive attempts that advance zero steps, otherwise a
# deterministic bug (a bad override, a missing file) holds a GPU for hours
# failing identically.  Any attempt that advances the step counter resets it.
#
# --out must be on storage that survives the node: container-local state is
# gone the moment the node is reclaimed, and the job would restart from zero
# on every preemption while looking like it was making progress.
#
# Kept LF-only by .gitattributes (*.sh text eol=lf).
set -uo pipefail

OUT=""
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
MAX_NOPROGRESS="${MAX_NOPROGRESS:-3}"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$OUT" ]]; then
  echo "run_shard.sh: --out is required" >&2
  exit 2
fi
mkdir -p "$OUT"
ls "$OUT" >/dev/null 2>&1 || true    # refresh a stale network-mount lookup

progress() {
  # Steps the checkpoint pointer says are complete; -1 when there is none.
  python - "$OUT" <<'PY' 2>/dev/null || echo -1
import json, os, sys
p = os.path.join(sys.argv[1], "checkpoint.ptr.json")
print(json.load(open(p))["step"] if os.path.exists(p) else -1)
PY
}

# The configured step budget, so the retry guard can tell "stalled" from
# "training finished, still evaluating" (see the loop below).
TOTAL_STEPS=""
for a in "${ARGS[@]}"; do
  case "$a" in n_steps=*) TOTAL_STEPS="${a#n_steps=}" ;; esac
done

echo "[wrapper] host=$(hostname) out=$OUT"

# Idempotence comes FIRST, before any hardware check: re-submitting a finished
# sweep must be free, and must not depend on the health of the GPU on whatever
# node the no-op happens to land on.
if [[ -f "$OUT/result.json" ]]; then
  echo "[wrapper] result.json present; run already complete"
  exit 0
fi

# Log the real device, not the one that was requested.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  || echo "[wrapper] no nvidia-smi"
# Fail fast instead of silently crawling on CPU: a node with a wedged driver
# reports zero devices and train.py's device=auto would quietly fall back.
# DDS_ALLOW_CPU is for exercising this script off-cluster; never set it in a job.
if [[ "${DDS_ALLOW_CPU:-0}" != "1" ]]; then
  python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "[wrapper] FATAL: no CUDA device visible on $(hostname)" >&2; exit 1; }
fi

noprog=0
for (( attempt=1; attempt<=MAX_ATTEMPTS; attempt++ )); do
  if [[ -f "$OUT/result.json" ]]; then
    echo "[wrapper] result.json present; run complete"
    exit 0
  fi
  before="$(progress)"
  echo "[wrapper] attempt $attempt/$MAX_ATTEMPTS (step=$before)"
  python -m discrete_sampler.train "${ARGS[@]}" "hydra.run.dir=$OUT"
  rc=$?
  if [[ -f "$OUT/result.json" ]]; then
    echo "[wrapper] run finished (rc=$rc)"
    exit 0
  fi
  after="$(progress)"
  if [[ "$after" -gt "$before" ]]; then
    noprog=0
    echo "[wrapper] progressed $before -> $after (rc=$rc); retrying"
  elif [[ -n "$TOTAL_STEPS" && "$after" -ge "$TOTAL_STEPS" ]]; then
    # Training is done and what remains is the final evaluation, which writes
    # no checkpoint and can run for hours -- the alloy grids score 2349 state
    # points and take longer than the training did.  Judging that
    # by step progress would retire a fully trained run after three
    # preemptions, throwing away the entire job at the last hurdle.  The
    # evaluation is idempotent, so let it keep trying.
    noprog=0
    echo "[wrapper] at n_steps=$TOTAL_STEPS (rc=$rc); final evaluation is not"
    echo "[wrapper] checkpointed, so this is not counted as a stall; retrying"
  else
    noprog=$((noprog + 1))
    echo "[wrapper] NO PROGRESS at $after (rc=$rc), strike $noprog/$MAX_NOPROGRESS"
    if [[ "$noprog" -ge "$MAX_NOPROGRESS" ]]; then
      echo "[wrapper] giving up: $MAX_NOPROGRESS consecutive attempts made no progress" >&2
      exit 1
    fi
  fi
  sleep 5
done

echo "[wrapper] exhausted $MAX_ATTEMPTS attempts without finishing" >&2
exit 1
