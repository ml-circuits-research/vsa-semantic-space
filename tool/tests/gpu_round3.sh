#!/usr/bin/env bash
# Round 3 of the tool's GPU runs (coordinator, after round 2): PREPARED, NOT RUN — start it only when the GPU is free.
#   tool/tests/gpu_round3.sh [--wait]
# Two trainings, 5 seeds each, full sizes, D = 1024, batch 8, lr 0.02 (the regime of round2_815e5f9), ONE change each:
#   reference_D_presence_not_decided   = reference_A_gates_given retrained (no kept models), scored ALSO with presence not decided (every role
#                                         read, as v16): the unseen-transitions split next to A's 0.428 and 10b-quinquies' 0.627
#   commands_E_depth_opener_kinds      = commands_depth_from_gates with the primary depth estimator counting the level-opener KINDS
#                                         (legend reader.level_openers), independent of the learned gates; gates learned, nothing given open
# Lock and guards as tests/p0_full_gpu.sh: the main checkout's experiment/logs/.run.lock (one run at a time; --wait polls the pid lock only,
# rule 26), a clean code tree, 40 GB free. Output (commit afterwards): tool/runs/round3_<commit>/<run>/{train.log, metrics.json, splits.json,
# command.txt, REPORT.md, config.json, legend.json}, round3_table.md, round3.log. No checkpoints, no models.
set -euo pipefail
TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO="$(dirname "$TOOL")"; SOP="$TOOL/bin/sop"
PY="${SOP_PYTHON:-}"; for c in "$REPO/.venv/bin/python3" "$REPO/../../../.venv/bin/python3"; do [ -z "$PY" ] && [ -x "$c" ] && PY="$c"; done; PY="${PY:-python3}"
cd "$REPO"
if [ -n "$(git status --porcelain 2>/dev/null | grep -v "logs/" | grep -v "\.md$" || true)" ]; then
  echo "uncommitted code changes: commit first, so that the run folder corresponds to a commit"; exit 1; fi
COMMIT=$(git rev-parse --short HEAD); OUT="$TOOL/runs/round3_$COMMIT"
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
mkdir -p "$OUT"; WORK="$(mktemp -d "${TMPDIR:-/tmp}/sop_round3_XXXX")"; LOG="$OUT/round3.log"
echo "# round 2, commit $COMMIT, $(date -Iseconds); worlds in $WORK; runs in $OUT" | tee "$LOG"
"$SOP" export --out "$WORK" --n 30000 --n_eval 30 --worlds reference 2>&1 | grep -v "^  T" | tee -a "$LOG"
"$SOP" export --out "$WORK" --n 40000 --n_eval 30 --worlds commands 2>&1 | tee -a "$LOG"
COMMON="--D 1024 --seeds 5 --vary_stars --device cuda --n_eval 600 --log_every 500 --no_ckpt"
run() {  # name world extra...
  local name=$1 w=$2; shift 2
  echo "## $name: sop train $WORK/$w $COMMON $*" | tee -a "$LOG"
  "$SOP" train "$WORK/$w" $COMMON "$@" --run_dir "$OUT/$name" 2>&1 | grep "\[eval\]\|\[done\]\|end:\|numpy reader\|GIVEN" | tee -a "$LOG"
  cp "$WORK/$w/legend.json" "$OUT/$name/legend.json"
}
run reference_D_presence_not_decided reference --also_presence_all
run commands_E_depth_opener_kinds commands --depth_estimator kinds
"$PY" "$TOOL/tests/round3_table.py" --root "$OUT" | tee -a "$LOG"
echo "done: commit $OUT (git add tool/runs/round3_$COMMIT)" | tee -a "$LOG"
