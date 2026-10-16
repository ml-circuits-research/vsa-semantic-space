#!/usr/bin/env bash
# Test 2 (build order step 2): `sop formalize` on the repository's three constructed worlds.
# Exports the worlds (small: 2000 training-distribution sentences each), runs `sop check` on every example, trains a SMOKE reader per world
# on CPU (D=256, 3 epochs; the full-size P0 run is tests/p0_full_cpu.sh), formalizes every eval sentence twice and checks: the output passes
# `sop check` against the world's legend, it is byte-identical between the two runs (determinism), and `sop stats` prints the table.
# The smoke numbers are a pipeline check, not the P0 round trip (that needs the full sizes: README, "P0 round trip").
set -euo pipefail
TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; SOP="$TOOL/bin/sop"; W="${WORLDS_DIR:-$TOOL/worlds}"; OUT="$TOOL/tests/out"; mkdir -p "$OUT"
FAILS=0
"$SOP" export --out "$W" --n 2000 --n_eval 20 | grep -v "^  T"
for w in reference commands quantifiers; do
  echo "== $w"
  "$SOP" check --legend "$W/$w/legend.json" "$W/$w"/examples/*.sop | tail -1
  if [ "${SKIP_TRAIN:-0}" != "1" ] || [ ! -f "$W/$w/model/config.json" ]; then
    "$SOP" train "$W/$w" --D 256 --epochs 3 --n_eval 200 --log_every 1000 > "$OUT/train_$w.log" 2>&1 || { echo "FAIL training $w (see $OUT/train_$w.log)"; FAILS=$((FAILS+1)); continue; }
    grep "numpy reader vs torch\|\[eval\]" "$OUT/train_$w.log"
  fi
  grep -h "^_text=" "$W/$w"/examples/eval_*.sop | sed 's/^_text=//' > "$OUT/sentences_$w.txt"
  "$SOP" formalize --model "$W/$w/model" "$OUT/sentences_$w.txt" -o "$OUT/formalized_$w.sop"
  "$SOP" formalize --model "$W/$w/model" "$OUT/sentences_$w.txt" -o "$OUT/formalized_${w}_again.sop"
  if cmp -s "$OUT/formalized_$w.sop" "$OUT/formalized_${w}_again.sop"; then echo "PASS $w: formalize deterministic ($(wc -l < "$OUT/sentences_$w.txt") sentences, byte-identical twice)"; else echo "FAIL $w: formalize not deterministic"; FAILS=$((FAILS+1)); fi
  if "$SOP" check --legend "$W/$w/legend.json" "$OUT/formalized_$w.sop" > /dev/null; then echo "PASS $w: formalize output passes sop check"; else echo "FAIL $w: formalize output fails sop check"; "$SOP" check --legend "$W/$w/legend.json" "$OUT/formalized_$w.sop" | head; FAILS=$((FAILS+1)); fi
  "$SOP" formalize --model "$W/$w/model" --json "$OUT/sentences_$w.txt" -o "$OUT/formalized_$w.jsonl" && python3 -c "import json,sys; [json.loads(l) for l in open(sys.argv[1])]" "$OUT/formalized_$w.jsonl" && echo "PASS $w: --json output (one JSON object per wire)"
  "$SOP" stats --model "$W/$w/model" > "$OUT/stats_$w.txt" && echo "PASS $w: sop stats ($(wc -l < "$OUT/stats_$w.txt") lines)" && sed -n '4,12p' "$OUT/stats_$w.txt"
done
# exit code 2: model missing / legend mismatch
set +e
"$SOP" formalize --model "$OUT/no_such_model" "$OUT/sentences_reference.txt" >/dev/null 2>&1; c=$?
[ $c -eq 2 ] && echo "PASS exit 2 on a missing model" || { echo "FAIL exit $c on a missing model"; FAILS=$((FAILS+1)); }
"$SOP" formalize --model "$W/reference/model" --legend "$W/commands/legend.json" "$OUT/sentences_reference.txt" >/dev/null 2>&1; c=$?
[ $c -eq 2 ] && echo "PASS exit 2 on a legend mismatch" || { echo "FAIL exit $c on a legend mismatch"; FAILS=$((FAILS+1)); }
# the dispatcher by fit: every reader reads, the smallest residual wins and fills _world
head -3 "$OUT/sentences_reference.txt" > "$OUT/mixed.txt"; head -3 "$OUT/sentences_commands.txt" >> "$OUT/mixed.txt"
"$SOP" formalize --model "$W/reference/model" --model "$W/commands/model" "$OUT/mixed.txt" -o "$OUT/dispatched.sop"
nr=$(grep -c "^_world=reference" "$OUT/dispatched.sop"); nc=$(grep -c "^_world=commands" "$OUT/dispatched.sop")
[ "$nr" = 3 ] && [ "$nc" = 3 ] && echo "PASS dispatcher by fit: 3 sentences to reference, 3 to commands" || { echo "FAIL dispatcher: reference $nr, commands $nc"; FAILS=$((FAILS+1)); }
"$SOP" check --legends "$W" "$OUT/dispatched.sop" > /dev/null && echo "PASS the dispatched file (two worlds) passes sop check --legends" || { echo "FAIL dispatched check"; FAILS=$((FAILS+1)); }
set -e
echo; [ $FAILS -eq 0 ] && echo "test2 (formalize): PASS" || echo "test2: $FAILS FAILED"
exit $FAILS
