#!/usr/bin/env bash
# Round 6 of the tool's GPU runs (judgment 752853c §2): PREPARED, NOT RUN — start it only when the GPU is free.
#   tool/tests/gpu_round6.sh [--wait]
# One training, 5 seeds, full sizes, D = 1024, batch 8, lr 0.02 (reference legend v2, to / by / from given open), the regime of round 5.
#   reference_H_fitted_rule
# CHANGE (one run, the regime of round 5, judgment 752853c §2): the presence thresholds by the FITTED rule (argmax balanced accuracy on the calibration rows: training sample + calibration half of the held-out formulations; sop train --threshold_rule fit, the default) and the PRESENCE MARGIN on the wire (_presence_margin.<role> = |ratio - threshold|; the role's _margin = min(identity, presence), so a near-threshold reading abstains). Product numbers first (presence decided, fitted rule); the pooled midpoint of the same rows beside it as the comparison. EXPECTATION (judgment 752853c §2): most of the product's presence errors lie inside the abstention band (|ratio - threshold| <= tau or identity margin < tau); if not, the asymmetry option is reported, not built.
# Lock and guards as tests/p0_full_gpu.sh: the main checkout's experiment/logs/.run.lock (one run at a time; --wait polls the pid lock only,
# rule 26), a clean code tree, 40 GB free. Output (commit afterwards): tool/runs/round6_<commit>/<run>/{train.log, metrics.json, splits.json,
# command.txt, REPORT.md, config.json, legend.json, presence_ratios_seed<k>.json}, round6_table.md, round6.log. No checkpoints, no models.
set -euo pipefail
TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO="$(dirname "$TOOL")"; SOP="$TOOL/bin/sop"
PY="${SOP_PYTHON:-}"; for c in "$REPO/.venv/bin/python3" "$REPO/../../../.venv/bin/python3"; do [ -z "$PY" ] && [ -x "$c" ] && PY="$c"; done; PY="${PY:-python3}"
cd "$REPO"
if [ -n "$(git status --porcelain 2>/dev/null | grep -v "logs/" | grep -v "\.md$" || true)" ]; then
  echo "uncommitted code changes: commit first, so that the run folder corresponds to a commit"; exit 1; fi
COMMIT=$(git rev-parse --short HEAD); OUT="$TOOL/runs/round6_$COMMIT"
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
mkdir -p "$OUT"; WORK="$(mktemp -d "${TMPDIR:-/tmp}/sop_round6_XXXX")"; LOG="$OUT/round6.log"
echo "# round 6, commit $COMMIT, $(date -Iseconds); worlds in $WORK; runs in $OUT" | tee "$LOG"
echo "# CHANGE (one run, the regime of round 5, judgment 752853c §2): the presence thresholds by the FITTED rule (argmax balanced accuracy on the calibration rows: training sample + calibration half of the held-out formulations; sop train --threshold_rule fit, the default) and the PRESENCE MARGIN on the wire (_presence_margin.<role> = |ratio - threshold|; the role's _margin = min(identity, presence), so a near-threshold reading abstains). Product numbers first (presence decided, fitted rule); the pooled midpoint of the same rows beside it as the comparison. EXPECTATION (judgment 752853c §2): most of the product's presence errors lie inside the abstention band (|ratio - threshold| <= tau or identity margin < tau); if not, the asymmetry option is reported, not built." | tee -a "$LOG"
"$SOP" export --out "$WORK" --n 30000 --n_eval 30 --worlds reference 2>&1 | grep -v "^  T" | tee -a "$LOG"
COMMON="--D 1024 --seeds 5 --vary_stars --device cuda --n_eval 600 --log_every 500 --no_ckpt"
name=reference_H_fitted_rule
echo "## $name: sop train $WORK/reference $COMMON --calibrate_on_heldout_formulations --threshold_rule fit --store_presence_ratios" | tee -a "$LOG"
"$SOP" train "$WORK/reference" $COMMON --calibrate_on_heldout_formulations --threshold_rule fit --store_presence_ratios --run_dir "$OUT/$name" 2>&1 | grep "\[eval\]\|\[done\]\|end:\|numpy reader\|GIVEN\|\[splits\]\|\[presence ratios\]" | tee -a "$LOG"
cp "$WORK/reference/legend.json" "$OUT/$name/legend.json"
"$PY" "$TOOL/tests/round6_table.py" --root "$OUT" | tee -a "$LOG"
echo "done: commit $OUT (git add tool/runs/round6_$COMMIT)" | tee -a "$LOG"
