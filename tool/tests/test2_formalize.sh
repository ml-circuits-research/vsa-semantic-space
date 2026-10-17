#!/usr/bin/env bash
# Test 2 (build order step 2): `sop formalize` on the repository's three constructed worlds.
# Exports the worlds (small: 2000 training-distribution sentences each), runs `sop check` on every example, trains a SMOKE reader per world
# on CPU (D=256, 3 epochs; the full-size P0 run is tests/p0_full_cpu.sh), formalizes every eval sentence twice and checks: the output passes
# `sop check` against the world's legend, it is byte-identical between the two runs (determinism), and `sop stats` prints the table.
# The smoke numbers are a pipeline check, not the P0 round trip (that needs the full sizes: README, "P0 round trip").
set -euo pipefail
TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; SOP="$TOOL/bin/sop"; W="${WORLDS_DIR:-$TOOL/worlds}"; OUT="$TOOL/tests/out"; mkdir -p "$OUT"
FAILS=0
PY="${SOP_PYTHON:-}"; for c in "$TOOL/../.venv/bin/python3" "$TOOL/../../../../.venv/bin/python3"; do [ -z "$PY" ] && [ -x "$c" ] && PY="$c"; done; PY="${PY:-python3}"
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
# the quantifier world v2 (C2c, `_scope`): v1 kept under versions/; _scope emitted on the ambiguous (never trained) sentences; FULL with
# the depth given and estimated side by side (author C3)
Q="$W/quantifiers"
"$PY" - "$Q" "$OUT/formalized_quantifiers.sop" "$OUT/sentences_quantifiers.txt" <<'EOF2'
import json, os, sys
q, form, sents = sys.argv[1], sys.argv[2], sys.argv[3]; fails = 0
def ok(c, m):
    global fails; print(("PASS " if c else "FAIL ") + m); fails += 0 if c else 1
v3 = json.load(open(os.path.join(q, "legend.json"))); v1 = json.load(open(os.path.join(q, "versions", "v1", "legend.json")))
v2 = json.load(open(os.path.join(q, "versions", "v2", "legend.json")))
ok(v3["version"] == 3 and v2["version"] == 2 and v1["version"] == 1 and "scope_terms" in v3["reader"] and "scope_terms" in v2["reader"]
   and "scope_terms" not in v1.get("reader", {}) and v3["reader"].get("level_openers") and not v2["reader"].get("level_openers"),
   "quantifiers: legend v3 (v2 = C2c + level openers) current; v2 and v1 (no scope) under versions/")
text = open(form).read(); n_scope = text.count("\n_scope=")
amb = [l.strip() for l in open(os.path.join(q, "examples", "eval_ambiguous_d0.sop")) if l.startswith("_text=")]
ok(n_scope > 0, f"quantifiers: sop formalize emits _scope ({n_scope} wires; {text.count('_scope=both')} 'both')")
m = json.load(open(os.path.join(q, "model", "metrics.json")))["per_seed"][0]
for s in ("ambiguous_d0 (two unmarked quantifiers, never trained: _scope=both)", "heldout_sentences"):
    a = m[s]["catalog candidates"]["acc"]
    ok("FULL [depth given]" in a and "_scope (wide role / both / absent)" in a and "FULL incl. det (previous definition)" in a,
       f"quantifiers {s[:12]}: FULL est. {a['FULL']:.3f} / depth given {a['FULL [depth given]']:.3f}; _scope {a['_scope (wide role / both / absent)']:.3f}; FULL incl. det {a['FULL incl. det (previous definition)']:.3f}")
r = json.load(open(os.path.join(os.path.dirname(q), "reference", "legend.json")))
ok(r["version"] == 2 and r["reader"]["gates_given"]["open"] == ["to", "by", "from"] and os.path.exists(os.path.join(os.path.dirname(q), "reference", "versions", "v1", "legend.json")),
   "reference: legend v2 with gates_given to / by / from open and level openers; v1 under versions/v1/")
rm = json.load(open(os.path.join(os.path.dirname(q), "reference", "model", "metrics.json")))["per_seed"][0]["heldout_depth3"]["catalog candidates"]["acc"]
cfg = json.load(open(os.path.join(os.path.dirname(q), "reference", "model", "config.json")))
ok(cfg.get("depth_estimator") == "kinds" and "depth estimate: primary (kinds)" in rm and "FULL [gate-count estimate]" in rm,
   "the default primary depth estimator is the level-opener kind count (config.json depth_estimator=kinds); the gate count and the fit are checks")
rep_ = open(os.path.join(os.path.dirname(q), "reference", "model", "REPORT.md")).read()
ok("## Unseen formulations: the readouts" in rep_ and "v16 protocol: every role read, not available to the product" in rep_ and cfg.get("presence_readout") == "decided",
   "REPORT.md prints both readouts on the unseen-formulation splits; the default readout decides presence")
ok("depth estimate: open-gate count" in rm and "FULL [fit estimate]" in rm,
   f"reference depth 3: depth from the level gates {rm['depth estimate: open-gate count']:.3f} vs the fit {rm['depth estimate: fit (check)']:.3f}; "
   f"FULL gates / fit / given {rm['FULL']:.3f} / {rm['FULL [fit estimate]']:.3f} / {rm['FULL [depth given]']:.3f}")
rt = open(form.replace("quantifiers", "reference")).read()
ok("_depth.gates=" in rt and "_depth.kinds=" in rt and "_depth.fit=" in rt, "formalize writes _depth.gates, _depth.kinds and _depth.fit on the top-level wire")
ok("depth estimate: opener-kind count" in rm and "FULL [gate-count estimate]" in rm and "FULL [kind-count estimate]" not in rm,
   f"reference depth 3: the opener-kind count (primary, so it is FULL) right on {rm.get('depth estimate: opener-kind count', float('nan')):.3f}; FULL {rm['FULL']:.3f}, FULL [gate-count] {rm.get('FULL [gate-count estimate]', float('nan')):.3f}")
import subprocess
sop = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(form))), "bin", "sop")
md = os.path.join(os.path.dirname(q), "reference", "model"); txt = "p3 gives o1 .\n"
d1 = subprocess.run([sop, "formalize", "--model", md, "-"], input=txt, capture_output=True, text=True).stdout
d2 = subprocess.run([sop, "formalize", "--model", md, "--presence", "all", "-"], input=txt, capture_output=True, text=True).stdout
ok("recipient=" not in d1.split("@w")[1] and "recipient=" in d2, "--presence all reads every role (a recipient-less sentence gets a recipient line); the default decides presence")
sys.exit(fails)
EOF2
FAILS=$((FAILS+$?))
# round 4 (judgment 45ccd1d D): presence thresholds set on the training sample + the calibration half of the held-out formulations
"$SOP" train "$W/reference" --D 256 --epochs 1 --n_eval 60 --log_every 5000 --no_ckpt --calibrate_on_heldout_formulations --out "$OUT/ref_cal_model" > "$OUT/train_ref_cal.log" 2>&1
"$PY" - "$W/reference" "$OUT/ref_cal_model" <<'EOF3'
import glob, json, os, sys
wd, md = sys.argv[1], sys.argv[2]; fails = 0
def ok(c, m):
    global fails; print(("PASS " if c else "FAIL ") + m); fails += 0 if c else 1
sp = json.load(open(sorted(glob.glob(os.path.join(wd, "runs", "*", "splits.json")), key=os.path.getmtime)[-1]))
cal = [k for k in sp["rows"] if k.startswith("calibration_")]; rep = [k for k in sp["rows"] if k.startswith("heldout_formulations")]
ok(cal and rep and all(not (set(sp["texts"][c]) & set(sp["texts"][c[len("calibration_"):]])) for c in cal) and "calibration_of_heldout_formulations" in sp["definitions"],
   f"splits.json (written before training) holds the calibration halves {cal}, disjoint from the reported halves")
cfg = json.load(open(os.path.join(md, "config.json")))
ok("thresholds_training_rows_only" in cfg and "calibration half" in cfg["thresholds"]["calibration_split"].lower(), "config.json: the product thresholds (training + held-out-formulation calibration rows) and the training-rows-only ones")
r = open(os.path.join(md, "REPORT.md")).read(); i, j, k = r.find("product readout: presence decided"), r.find("thresholds from training rows only (the round-3"), r.find("v16 protocol: every role read, not available to the product")
ok(0 < i < j < k, "REPORT.md: the product number first, then the round-3 thresholds, then the v16 protocol, each labelled")
sys.exit(fails)
EOF3
FAILS=$((FAILS+$?))
set -e
echo; [ $FAILS -eq 0 ] && echo "test2 (formalize): PASS" || echo "test2: $FAILS FAILED"
exit $FAILS
