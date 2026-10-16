#!/usr/bin/env bash
# P0 round trip at the repositories' full sizes (reference N=30000, commands / quantifiers N=40000, D=1024, batch 8, lr 0.02), on CPU,
# one seed per world (the RESULTS tables are 5 seeds on the GPU: for those, add --seeds 5 --device cuda through experiment/queue.sh).
# Output: tool/worlds_p0/<world>/model/REPORT.md and metrics.json; logs in tool/worlds_p0/<world>/runs/.
set -euo pipefail
TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; OUT="${P0_OUT:-$TOOL/worlds_p0}"; SEEDS="${P0_SEEDS:-1}"; DEV="${P0_DEVICE:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
"$TOOL/bin/sop" export --out "$OUT" --n 30000 --n_eval 30 --worlds reference
"$TOOL/bin/sop" export --out "$OUT" --n 40000 --n_eval 30 --worlds commands,quantifiers
for w in reference commands quantifiers; do
  "$TOOL/bin/sop" train "$OUT/$w" --D 1024 --seeds "$SEEDS" --device "$DEV" --n_eval 600 --log_every 500
done
