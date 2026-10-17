#!/usr/bin/env bash
# Round 5 of the tool's GPU runs (judgment d8f8a98 Q5): PREPARED, NOT RUN — start it only when the GPU is free.
#   tool/tests/gpu_round5.sh [--wait]
# One training, 5 seeds, full sizes, D = 1024, batch 8, lr 0.02 — the regime of round 4 F (reference legend v2, to / by / from given open,
# --calibrate_on_heldout_formulations), NO other change, that STORES the per-sentence presence ratios (sop train --store_presence_ratios):
# present class (calibration halves / reported halves of the held-out formulations), absent class (training recipient-less / held-out
# sentences recipient-less), per seed, in the run folder (presence_ratios_seed<k>.json; no models).
#   reference_G_presence_ratios
# Readouts (tests/round5_table.py): (i) the pooled midpoint (the product today); (ii) a threshold FITTED on the calibration half alone (max
# balanced present / absent accuracy, stated); the overlap (share of present ratios below the absent 90th percentile; best balanced accuracy).
# DECISION RULE (written before the run, verbatim from judgment d8f8a98 Q5): if the present-class ratios on unseen formulations overlap the absent class (no threshold separates them above 0.9 present / 0.9 absent), the limit is the READER's and stays recorded as it is; if a threshold fitted on the held-out formulations alone separates them, the limit is the midpoint RULE's, and the tool's calibration becomes "threshold fitted on calibration rows that include held-out formulations, by a fit, not the pooled midpoint" (rule 25; calibration split disjoint from the report split), re-run once as round 5.
# Lock and guards as tests/p0_full_gpu.sh: the main checkout's experiment/logs/.run.lock (one run at a time; --wait polls the pid lock only,
# rule 26), a clean code tree, 40 GB free. Output (commit afterwards): tool/runs/round5_<commit>/<run>/{train.log, metrics.json, splits.json,
# command.txt, REPORT.md, config.json, legend.json, presence_ratios_seed<k>.json}, round5_table.md, round5.log. No checkpoints, no models.
set -euo pipefail
TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO="$(dirname "$TOOL")"; SOP="$TOOL/bin/sop"
PY="${SOP_PYTHON:-}"; for c in "$REPO/.venv/bin/python3" "$REPO/../../../.venv/bin/python3"; do [ -z "$PY" ] && [ -x "$c" ] && PY="$c"; done; PY="${PY:-python3}"
cd "$REPO"
if [ -n "$(git status --porcelain 2>/dev/null | grep -v "logs/" | grep -v "\.md$" || true)" ]; then
  echo "uncommitted code changes: commit first, so that the run folder corresponds to a commit"; exit 1; fi
COMMIT=$(git rev-parse --short HEAD); OUT="$TOOL/runs/round5_$COMMIT"
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
mkdir -p "$OUT"; WORK="$(mktemp -d "${TMPDIR:-/tmp}/sop_round5_XXXX")"; LOG="$OUT/round5.log"
echo "# round 5, commit $COMMIT, $(date -Iseconds); worlds in $WORK; runs in $OUT" | tee "$LOG"
echo "# DECISION RULE (written before the run, verbatim from judgment d8f8a98 Q5): if the present-class ratios on unseen formulations overlap the absent class (no threshold separates them above 0.9 present / 0.9 absent), the limit is the READER's and stays recorded as it is; if a threshold fitted on the held-out formulations alone separates them, the limit is the midpoint RULE's, and the tool's calibration becomes \"threshold fitted on calibration rows that include held-out formulations, by a fit, not the pooled midpoint\" (rule 25; calibration split disjoint from the report split), re-run once as round 5." | tee -a "$LOG"
"$SOP" export --out "$WORK" --n 30000 --n_eval 30 --worlds reference 2>&1 | grep -v "^  T" | tee -a "$LOG"
COMMON="--D 1024 --seeds 5 --vary_stars --device cuda --n_eval 600 --log_every 500 --no_ckpt"
name=reference_G_presence_ratios
echo "## $name: sop train $WORK/reference $COMMON --calibrate_on_heldout_formulations --store_presence_ratios" | tee -a "$LOG"
"$SOP" train "$WORK/reference" $COMMON --calibrate_on_heldout_formulations --store_presence_ratios --run_dir "$OUT/$name" 2>&1 | grep "\[eval\]\|\[done\]\|end:\|numpy reader\|GIVEN\|\[splits\]\|\[presence ratios\]" | tee -a "$LOG"
cp "$WORK/reference/legend.json" "$OUT/$name/legend.json"
"$PY" "$TOOL/tests/round5_table.py" --root "$OUT" | tee -a "$LOG"
echo "done: commit $OUT (git add tool/runs/round5_$COMMIT)" | tee -a "$LOG"
