#!/usr/bin/env bash
# The 5-seed GPU P0 round trip on the three worlds (reference; commands; quantifiers VERSION 2 = C2c), at the repository's sizes
# (reference 30,000 sentences, commands / quantifiers 40,000; D = 1024, batch 8, lr 0.02, gates learned from a closed start; identities drawn
# per seed, rule 6: --vary_stars). PREPARED, NOT RUN: start it only when the GPU is free.
#   tool/tests/p0_full_gpu.sh [--wait]
# One run at a time on this machine: the script takes the repository's GPU lock experiment/logs/.run.lock (the pid lock of
# experiment/run.sh and queue.sh); it refuses to start when the lock is held (with --wait it polls the pid lock only, rule 26), and it
# refuses an unclean code tree (the run folder is named by the commit). Output, committed afterwards:
#   tool/runs/p0_gpu_5seed_<commit>/<world>/{train.log, metrics.json, splits.json, command.txt, REPORT.md, config.json}   (no checkpoints)
#   tool/runs/p0_gpu_5seed_<commit>/p0_table.md   (tests/p0_compare.py: structure estimated and depth given, side by side)
set -euo pipefail
TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO="$(dirname "$TOOL")"; SOP="$TOOL/bin/sop"
PY="${SOP_PYTHON:-}"; for c in "$REPO/.venv/bin/python3" "$REPO/../../../.venv/bin/python3"; do [ -z "$PY" ] && [ -x "$c" ] && PY="$c"; done; PY="${PY:-python3}"
cd "$REPO"
if [ -n "$(git status --porcelain 2>/dev/null | grep -v "logs/" | grep -v "\.md$" | grep -v "^?? tool/worlds" || true)" ]; then
  echo "uncommitted code changes: commit first, so that the run folder corresponds to a commit"; exit 1; fi
COMMIT=$(git rev-parse --short HEAD); OUT="$TOOL/runs/p0_gpu_5seed_$COMMIT"
# the lock of the MAIN checkout (run.sh / queue.sh run there): from a git worktree, the checkout that owns the common .git directory
MAIN="$(dirname "$(cd "$REPO" && git rev-parse --path-format=absolute --git-common-dir)")"
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
mkdir -p "$OUT"; WORK="$(mktemp -d "${TMPDIR:-/tmp}/sop_p0_gpu_XXXX")"
echo "# P0 GPU 5 seeds, commit $COMMIT, $(date -Iseconds); worlds exported to $WORK; runs in $OUT" | tee "$OUT/p0_gpu.log"
"$SOP" export --out "$WORK" --n 30000 --n_eval 30 --worlds reference 2>&1 | grep -v "^  T" | tee -a "$OUT/p0_gpu.log"
"$SOP" export --out "$WORK" --n 40000 --n_eval 30 --worlds commands,quantifiers 2>&1 | tee -a "$OUT/p0_gpu.log"
for w in reference commands quantifiers; do
  "$SOP" train "$WORK/$w" --D 1024 --seeds 5 --vary_stars --device cuda --n_eval 600 --log_every 500 --no_ckpt --run_dir "$OUT/$w" 2>&1 \
    | grep "\[eval\]\|\[done\]\|end:\|numpy reader" | tee -a "$OUT/p0_gpu.log"
  cp "$WORK/$w/legend.json" "$OUT/$w/legend.json"
done
"$PY" "$TOOL/tests/p0_compare.py" --root "$OUT" | tee "$OUT/p0_table.md"
echo "done: commit $OUT (git add tool/runs/p0_gpu_5seed_$COMMIT)"
