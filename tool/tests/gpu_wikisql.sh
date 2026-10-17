#!/usr/bin/env bash
# The P1 rehearsal on real text: WikiSQL through the tool (judgment 934fbae E2): PREPARED, NOT RUN — start it only when the GPU is free.
#   tool/tests/gpu_wikisql.sh [--wait]
# One world, 3 seeds: the WikiSQL world exported by tool/export_wikisql.py (questions as text, gold frames from the SQL, the
# hand-written legend with the condition SET value), sop train, sop formalize, the SOP round trip.
# FIRST ATTEMPT (commit f9e27b0, 2026-10-17 14:42): stopped after 20 minutes in seed 0 for the quadratic padding cost (every batch padded to the longest question of the whole set; the chain code is quadratic in the length) and produced nothing; this run batches by length (--length_buckets: v11's bucketing, each batch trimmed to its longest question — exact, tested in test 7).
# CRITERION (written before the run, judgment 934fbae E2): the package reproduces RESULTS 8c / 8d within seed variance — dev: selected column 0.810±0.004 (8c) / 0.816 (8d 56k), condition-1 column 0.632±0.004 / 0.626 (here: the first gold condition column in the decoded SET), FULL query 0.376±0.003 / 0.368 — and the frames round-trip through SOP Lang (write -> check -> read, byte-identical; frames equal the SQL). Sizes: 56,355 train questions, 3 epochs, D = 512, batch 32, lr 0.02, 3 seeds, identities fixed across seeds (as v11), question only; condition as a SET value (one role, & of entity wires), not indexed slots (judgment d8f8a98 Q1).
# Lock and guards as tests/p0_full_gpu.sh: the main checkout's experiment/logs/.run.lock (one run at a time; --wait polls the pid lock only,
# rule 26), a clean code tree, 40 GB free. Output (commit afterwards): tool/runs/wikisql_<commit>/<run>/{train.log, metrics.json, splits.json,
# command.txt, REPORT.md, config.json, legend.json, presence_ratios_seed<k>.json}, wikisql_table.md, wikisql.log. No checkpoints, no models.
set -euo pipefail
TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO="$(dirname "$TOOL")"; SOP="$TOOL/bin/sop"
PY="${SOP_PYTHON:-}"; for c in "$REPO/.venv/bin/python3" "$REPO/../../../.venv/bin/python3"; do [ -z "$PY" ] && [ -x "$c" ] && PY="$c"; done; PY="${PY:-python3}"
cd "$REPO"
if [ -n "$(git status --porcelain 2>/dev/null | grep -v "logs/" | grep -v "\.md$" || true)" ]; then
  echo "uncommitted code changes: commit first, so that the run folder corresponds to a commit"; exit 1; fi
COMMIT=$(git rev-parse --short HEAD); OUT="$TOOL/runs/wikisql_$COMMIT"
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
mkdir -p "$OUT"; WORK="$(mktemp -d "${TMPDIR:-/tmp}/sop_wikisql_XXXX")"; LOG="$OUT/wikisql.log"
echo "# wikisql, commit $COMMIT, $(date -Iseconds); worlds in $WORK; runs in $OUT" | tee "$LOG"
echo "# FIRST ATTEMPT (commit f9e27b0, 2026-10-17 14:42): stopped after 20 minutes in seed 0 for the quadratic padding cost (every batch padded to the longest question of the whole set; the chain code is quadratic in the length) and produced nothing; this run batches by length (--length_buckets: v11's bucketing, each batch trimmed to its longest question — exact, tested in test 7)." | tee -a "$LOG"
echo "# CRITERION (written before the run, judgment 934fbae E2): the package reproduces RESULTS 8c / 8d within seed variance — dev: selected column 0.810±0.004 (8c) / 0.816 (8d 56k), condition-1 column 0.632±0.004 / 0.626 (here: the first gold condition column in the decoded SET), FULL query 0.376±0.003 / 0.368 — and the frames round-trip through SOP Lang (write -> check -> read, byte-identical; frames equal the SQL). Sizes: 56,355 train questions, 3 epochs, D = 512, batch 32, lr 0.02, 3 seeds, identities fixed across seeds (as v11), question only; condition as a SET value (one role, & of entity wires), not indexed slots (judgment d8f8a98 Q1)." | tee -a "$LOG"
"$PY" "$TOOL/export_wikisql.py" --out "$WORK" 2>&1 | tee -a "$LOG"
echo "## wikisql: sop train $WORK/wikisql --D 512 --epochs 3 --batch 32 --length_buckets --seeds 3 --device cuda --n_eval 8421 --eval_modes primary --candidate_sets catalog --no_ckpt" | tee -a "$LOG"
"$SOP" train "$WORK/wikisql" --D 512 --epochs 3 --batch 32 --length_buckets --seeds 3 --device cuda --n_eval 8421 --log_every 1000 --eval_modes primary --candidate_sets catalog --no_ckpt \
  --run_dir "$OUT/wikisql" 2>&1 | grep "\[eval\]\|\[done\]\|end:\|numpy reader\|\[splits\]\|\[schema\]" | tee -a "$LOG"
cp "$WORK/wikisql/legend.json" "$OUT/wikisql/legend.json"; cp "$WORK/wikisql/kinds.tsv" "$OUT/wikisql/kinds.tsv"
head -c 20000 "$WORK/wikisql/examples/dev.sop" > "$OUT/wikisql/dev_examples_head.sop"
"$PY" "$TOOL/tests/wikisql_table.py" --root "$OUT" --world "$WORK/wikisql" | tee -a "$LOG"
echo "done: commit $OUT (git add tool/runs/wikisql_$COMMIT)" | tee -a "$LOG"
