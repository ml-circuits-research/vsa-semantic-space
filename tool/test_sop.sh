#!/usr/bin/env bash
# The CPU smoke test of the sop package: export the three worlds, `sop check` the examples, a smoke training on the reference world
# (1 seed, D=256, 2000 sentences, 3 epochs), `sop formalize` on three sentences, `sop diff` on two formalizations, `sop stats`.
set -euo pipefail
TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; SOP="$TOOL/bin/sop"; W="$TOOL/worlds"; OUT="$TOOL/tests/out"; mkdir -p "$OUT"
"$SOP" export --out "$W" --n 2000 --n_eval 20 | grep -v "^  T"
for w in reference commands quantifiers; do "$SOP" check --legend "$W/$w/legend.json" "$W/$w"/examples/*.sop | tail -1; done
"$SOP" train "$W/reference" --D 256 --epochs 3 --n_eval 200 --log_every 200 | grep "\[train seed 0\] \(start\|end\)\|numpy reader\|\[eval\]\|checklist"
printf 'p3 says that p0 gives o1 to p2 .\np5 receives o2 from p1 .\no4 is sold to p2 by p7 .\n' > "$OUT/three.txt"
"$SOP" formalize --model "$W/reference/model" "$OUT/three.txt" -o "$OUT/three_a.sop"
cat "$OUT/three_a.sop"
printf 'p3 says that p0 gives o1 to p2 .\np5 receives o2 from p9 .\no4 is sold to p2 by p7 .\n' > "$OUT/three_b.txt"
"$SOP" formalize --model "$W/reference/model" "$OUT/three_b.txt" -o "$OUT/three_b.sop"
"$SOP" check --legend "$W/reference/legend.json" "$OUT/three_a.sop" "$OUT/three_b.sop" | tail -1
"$SOP" diff "$OUT/three_a.sop" "$OUT/three_b.sop" --model "$W/reference/model"
"$SOP" stats --model "$W/reference/model" | head -30
echo "test_sop.sh: PASS"
