#!/usr/bin/env bash
# Round 4 of the tool's GPU runs (judgment 45ccd1d D): PREPARED, NOT RUN — start it only when the GPU is free.
#   tool/tests/gpu_round4.sh [--wait]
# One training, 5 seeds, full sizes, D = 1024, batch 8, lr 0.02 — the regime of round3 run D (reference legend v2, to / by / from given open),
# ONE change: the presence thresholds are set on the training sample + the CALIBRATION half of every held-out formulation split
# (sop train --calibrate_on_heldout_formulations; the halves by hash, written to splits.json before training; the other half reported).
#   reference_F_calibration_with_heldout_formulations
# EXPECTATION (written before the run, judgment 45ccd1d D): with the presence thresholds set on calibration rows that include held-out FORMULATIONS, the recipient's presence on the reported half of the unseen formulations recovers toward the every-role-read value (round 3 D: role recipient 0.619 decided -> 0.839 every role read) without reading every role. If it does not, presence calibration across formulations is a recorded limit of the reader, not of the tool.
# Lock and guards as tests/p0_full_gpu.sh: the main checkout's experiment/logs/.run.lock (one run at a time; --wait polls the pid lock only,
# rule 26), a clean code tree, 40 GB free. Output (commit afterwards): tool/runs/round4_<commit>/<run>/{train.log, metrics.json, splits.json,
# command.txt, REPORT.md, config.json, legend.json}, round4_table.md, round4.log. No checkpoints, no models.
set -euo pipefail
TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO="$(dirname "$TOOL")"; SOP="$TOOL/bin/sop"
PY="${SOP_PYTHON:-}"; for c in "$REPO/.venv/bin/python3" "$REPO/../../../.venv/bin/python3"; do [ -z "$PY" ] && [ -x "$c" ] && PY="$c"; done; PY="${PY:-python3}"
cd "$REPO"
if [ -n "$(git status --porcelain 2>/dev/null | grep -v "logs/" | grep -v "\.md$" || true)" ]; then
  echo "uncommitted code changes: commit first, so that the run folder corresponds to a commit"; exit 1; fi
COMMIT=$(git rev-parse --short HEAD); OUT="$TOOL/runs/round4_$COMMIT"
MAIN="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
LOCK="${P0_LOCK:-$MAIN/experiment/logs/.run.lock}"; mkdir -p "$(dirname "$LOCK")"; echo "GPU lock: $LOCK"
held() { [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; }
if held; then
  if [ "${1:-}" = "--wait" ]; then while held; do sleep 30; done
  else echo "the GPU lock is held (pid $(cat "$LOCK"), $LOCK): one run at a time; rerun with --wait or later"; exit 1; fi
fi
avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
[ "$avail_gb" -ge 40 ] || { echo "only ${avail_gb} GB available; refusing to start (need 40, as run.sh)"; exit 1; }
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VSA_MEM_FRACTION=${VSA_MEM_FRACTION:-0.5} VSA_MIN_AVAILABLE_GB=${VSA_MIN_AVAILABLE_GB:-16} VSA_MAX_HOURS=${VSA_MAX_HOURS:-6}
mkdir -p "$OUT"; WORK="$(mktemp -d "${TMPDIR:-/tmp}/sop_round4_XXXX")"; LOG="$OUT/round4.log"
echo "# round 4, commit $COMMIT, $(date -Iseconds); worlds in $WORK; runs in $OUT" | tee "$LOG"
echo "# EXPECTATION (written before the run, judgment 45ccd1d D): with the presence thresholds set on calibration rows that include held-out FORMULATIONS, the recipient's presence on the reported half of the unseen formulations recovers toward the every-role-read value (round 3 D: role recipient 0.619 decided -> 0.839 every role read) without reading every role. If it does not, presence calibration across formulations is a recorded limit of the reader, not of the tool." | tee -a "$LOG"
"$SOP" export --out "$WORK" --n 30000 --n_eval 30 --worlds reference 2>&1 | grep -v "^  T" | tee -a "$LOG"
COMMON="--D 1024 --seeds 5 --vary_stars --device cuda --n_eval 600 --log_every 500 --no_ckpt"
name=reference_F_calibration_with_heldout_formulations
echo "## $name: sop train $WORK/reference $COMMON --calibrate_on_heldout_formulations" | tee -a "$LOG"
"$SOP" train "$WORK/reference" $COMMON --calibrate_on_heldout_formulations --run_dir "$OUT/$name" 2>&1 | grep "\[eval\]\|\[done\]\|end:\|numpy reader\|GIVEN\|\[splits\]" | tee -a "$LOG"
cp "$WORK/reference/legend.json" "$OUT/$name/legend.json"
"$PY" "$TOOL/tests/round4_table.py" --root "$OUT" | tee -a "$LOG"
echo "done: commit $OUT (git add tool/runs/round4_$COMMIT)" | tee -a "$LOG"
