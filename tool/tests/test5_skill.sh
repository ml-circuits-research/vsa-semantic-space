#!/usr/bin/env bash
# Test 5 (build order step 5): the train-world skill's loop, offline.
# (a) the full loop on the constructed commands world with its sentences as raw/ (legend given): annotate (oracle = generator gold, the offline
#     stand-in for the annotate model), train, second look; model/ complete, REPORT.md with every field, second-look rate reported.
# (b) one evolution step with a DELIBERATELY INCOMPLETE legend (role 'destination' removed, sentences kept): the loop must PROPOSE the role back
#     from the _unplaced clusters (residual directions, no LLM), migrate the examples, retrain, and accept it by the criterion of spec 1.1 step 5.
#     The naming of the role needs the evolve model (API): offline it gets a placeholder name, and the test checks the role by its FILLERS.
set -uo pipefail
TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; SOP="$TOOL/bin/sop"; OUT="$TOOL/tests/out/skill"; rm -rf "$OUT"; mkdir -p "$OUT"
PY="${SOP_PYTHON:-}"
for c in "$TOOL/../.venv/bin/python3" "$TOOL/../../../../.venv/bin/python3"; do [ -z "$PY" ] && [ -x "$c" ] && PY="$c"; done; PY="${PY:-python3}"
LOOP="$TOOL/skills/train-world/scripts/loop.py"; TRAIN_ARGS="${TRAIN_ARGS:---D 256 --epochs 6 --n_eval 150 --log_every 1000 --seeds 1 --no_ckpt}"
FAILS=0; ok() { if [ "$1" = "1" ]; then echo "PASS $2"; else echo "FAIL $2"; FAILS=$((FAILS+1)); fi; }

make_world() {   # name drop_role
  "$PY" - "$OUT" "$1" "$2" <<'EOF'
import sys, os, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[1]), "..", ".."))
out, name, drop = sys.argv[1], sys.argv[2], sys.argv[3] or None
import export_worlds as ex
wd = ex.export_commands(out, 1500, 15, 0, name=name, drop_role=drop, raw=True)
gold = os.path.join(out, name + "_gold"); shutil.move(os.path.join(wd, "examples"), gold); os.makedirs(os.path.join(wd, "examples"))
h = json.load(open(os.path.join(wd, "split_hints.json")))
hints = {"heldout_sentence_share": 0.1, "new_identities": h["new_identities"], "heldout_depth_min": 2}
if drop: hints["heldout_formulations"] = "none: the evolution test measures the missing role, not formulations (stated)"
json.dump(hints, open(os.path.join(wd, "split_hints.json"), "w"))
wj = json.load(open(os.path.join(wd, "world.json"))); wj["min_examples"] = 500; json.dump(wj, open(os.path.join(wd, "world.json"), "w"), indent=1)
EOF
}

echo "== (a) full loop, legend given, one version"
make_world commands_loop "" > /dev/null
W="$OUT/commands_loop"
"$PY" "$LOOP" run "$W" --annotator "oracle:$OUT/commands_loop_gold" --train-args "$TRAIN_ARGS" --steps 0 > "$OUT/loop_a.json" 2> "$OUT/loop_a.err"
ok "$([ $? -eq 0 ] && echo 1)" "loop run (a) exit 0"
for f in config.json legend.json catalog.json routebook.npz background.npz metrics.json REPORT.md kinds.tsv second_look.json; do
  [ -f "$W/model/$f" ] || { echo "FAIL model/$f missing"; FAILS=$((FAILS+1)); }; done
echo "PASS model/ complete: $(ls "$W/model" | tr '\n' ' ')"
for field in "## Given and learned" "## Data" "## Splits" "## Kinds" "## Training" "## Thresholds" "## Results" "## Gates" "## What is weak" "## Structure given vs estimated" "## Checklist" "## Loop: annotation and second look" "## Acceptance of this training"; do
  grep -q "$field" "$W/model/REPORT.md" || { echo "FAIL REPORT.md lacks '$field'"; FAILS=$((FAILS+1)); }; done
echo "PASS REPORT.md has every section"
"$PY" -c "import json,sys; s=json.load(open(sys.argv[1]))['second_look']; print(f\"PASS second-look rate reported: {s['disagreement_rate']:.3f} ({s['disagreements']} of {s['rows_read']}); changed {s['changed']}; {s['note'][:120]}\")" "$OUT/loop_a.json" \
  || { echo "FAIL second look"; FAILS=$((FAILS+1)); }
grep -q '"label": "BUILT"' "$W/status.json" && echo "PASS status.json BUILT" || { echo "FAIL status.json"; FAILS=$((FAILS+1)); }
"$PY" - "$OUT/loop_a.json" "$W" <<'EOF'
import json, sys, os
log = json.load(open(sys.argv[1])); wd = sys.argv[2]; fails = 0
def ok(c, m):
    global fails; print(("PASS " if c else "FAIL ") + m); fails += 0 if c else 1
inv = log["inventory"]; ok(inv["stop"] is None and inv["legend"]["errors"] == [], "6.1 inventory: legend consistent, nothing to stop on")
ok(log["seeds"].startswith("as given"), f"6.5 seeds stated: {log['seeds']}")
sl = log["second_look"]; ok(sl["rows"].startswith("the training split"), "6b second look reads the TRAINING split only")
ok(os.path.exists(os.path.join(wd, "audit", "disagreements.sop")), "6b audit/disagreements.sop written (both readings)")
ok(log["package"]["n"] > 0 and all(os.path.exists(os.path.join(wd, "audit", f)) for f in ("heldout_reader.sop", "heldout_annotation.sop", "heldout_verdicts.sop")),
   f"6.7 package: held-out slice formalized next to the annotation, verdicts {log['package']['verdicts']}")
acc = log["acceptance_of_training"]
for k, v in acc["checks"].items(): ok(v, f"acceptance of the training: {k}")
sp = json.load(open(sorted(__import__('glob').glob(os.path.join(wd, "runs", "*", "splits.json")))[0]))
ok({"train", "heldout_sentences", "new_identities"} <= set(sp["rows"]) and any(k.startswith("heldout_formulations") for k in sp["rows"]),
   f"6.4 the four splits in splits.json: {sorted((k, len(v)) for k, v in sp['rows'].items())}")
m = json.load(open(os.path.join(wd, "model", "metrics.json")))["per_seed"][0]["heldout_sentences"]["catalog candidates"]["acc"]
ok("heads FULL (lexical bag baseline, trained on the same training rows)" in m and "FULL [depth given]" in m and "occupancy: not given" in m,
   f"6.6 fields: bag baseline {m.get('heads FULL (lexical bag baseline, trained on the same training rows)'):.3f}, FULL est. / depth given "
   f"{m['FULL']:.3f} / {m['FULL [depth given]']:.3f}, occupancy known / not given {m.get('occupancy: known filler', float('nan')):.3f} / {m['occupancy: not given']:.3f}")
sys.exit(fails)
EOF
FAILS=$((FAILS+$?))
"$SOP" stats --model "$W/model" | sed -n '1,3p'

echo "== (b) evolution step: legend without 'destination', sentences kept"
make_world commands_missing destination > /dev/null
W="$OUT/commands_missing"
grep -q destination "$W/legend.json" && { echo "FAIL the legend still has destination"; FAILS=$((FAILS+1)); }
"$PY" "$LOOP" run "$W" --annotator "oracle:$OUT/commands_missing_gold" --train-args "$TRAIN_ARGS" --steps 1 > "$OUT/loop_b.json" 2> "$OUT/loop_b.err"
ok "$([ $? -eq 0 ] && echo 1)" "loop run (b) exit 0"
"$PY" - "$OUT/loop_b.json" "$W" <<'EOF'
import json, sys, os
log = json.load(open(sys.argv[1])); wd = sys.argv[2]; fails = 0
def ok(c, m):
    global fails; print(("PASS " if c else "FAIL ") + m); fails += 0 if c else 1
s = log["steps"][0]; cl = s["feedback"]["unplaced_clusters"]
print("  _unplaced clusters (words, kinds, sentences, routes, residual share, supported, novel):")
for c in cl: print("   ", c["words"], c["kinds"], c["sentences"], c["routes"], round(c["residual_share_mean"], 3), c["supported"], c["novel"])
dest = {"archive", "cloud", "trash", "folder"}
ch = s["evolve"]["changes"]; ok(len(ch) == 1, f"one change set item proposed: {[(c['role'], c['fillers'], c['naming'][:60]) for c in ch]}")
ok(bool(ch) and set(ch[0]["fillers"]) == dest, "the proposed role's fillers are exactly the removed role's fillers (archive, cloud, trash, folder)")
acc = s.get("acceptance", {}); print("  acceptance:", json.dumps({k: acc.get(k) for k in ("old", "new", "criteria", "met")}, default=str))
ok(acc.get("met") is True and s.get("accepted") is True, "v2 accepted by the criterion (FULL within tolerance, residual down, unknown not up, confident-wrong not up)")
lg = json.load(open(os.path.join(wd, "legend.json"))); ok(lg["version"] == 2 and len(lg["roles"]) == 3, f"legend.json is v2 with the role added: {[r['id'] for r in lg['roles']]}")
ok(os.path.exists(os.path.join(wd, "versions", "v1", "REPORT.md")) and os.path.exists(os.path.join(wd, "versions", "v2", "CHANGES.md")), "versions/v1 (previous, with REPORT) and versions/v2/CHANGES.md written")
st = json.load(open(os.path.join(wd, "status.json"))); ok(st.get("legend_version") == 2, "status.json names v2")
sys.exit(fails)
EOF
FAILS=$((FAILS+$?))
"$SOP" check --legend "$W/legend.json" "$W"/examples/*.sop | tail -1
sed -n '/## Acceptance/,$p' "$W/versions/v2/CHANGES.md"
echo; [ $FAILS -eq 0 ] && echo "test5 (skill loop + missing-role recovery): PASS" || echo "test5: $FAILS FAILED"
exit $FAILS
