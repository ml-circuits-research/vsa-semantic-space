"""
Test 1 (build order step 1): SOP Lang parser/writer and `sop check`.
  - the worked example of spec 3.5 parses and round-trips byte for byte;
  - with a legend matching it (hash written into the file), the example passes `sop check`;
  - every negative case of spec 3.6 is rejected with a line number and a reason: duplicate id, dangling $ref, reference cycle, unknown key,
    '&' and '|' mixed, _asked without _mood=question, legend hash mismatch (plus: unknown wire type, two _text lines, bad quoting).
Run: python3 tool/tests/test1_soplang.py   (exit 0 = pass)
"""
import json, os, subprocess, sys, tempfile
TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, TOOL)
from sop import soplang
from sop.legend import Legend
from sop.check import check_file

EXAMPLE = """!world commands
!legend sha256:3f9a…
!generated sop 0.1, model commands/model (2026-10-16)

@w1 says
_text=Maria says that Paul does not give Anna the unsaved files.
_sentence=0
speaker=Maria
scope=$w2
_margin=0.19
_status=ok
_depth=0

@w2 not
scope=$w3
_depth=1

@w3 transfer
agent=Paul
object=$w4
recipient=Anna
_empty=time
_margin=0.24
_depth=2

@w4 entity
head=files
attr=$w5

@w5 not
scope=saved
"""
LEGEND = {"version": 1, "roles": [{"id": "agent", "question": "who gives?"}, {"id": "object", "question": "what?"},
                                  {"id": "recipient", "question": "to whom?"}, {"id": "time", "question": "when?"}],
          "events": [{"id": "transfer", "verbs": ["give", "receive", "hand"]}, {"id": "sale", "verbs": ["sell", "buy"]}],
          "operators": ["not", "must", "may", "if", "says"], "moods": ["statement", "command", "question"],
          "kinds": {"source": "lexicon", "lexicon": "kinds.tsv"}}

fails = 0
def ok(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond: fails += 1

doc = soplang.parse(EXAMPLE)
ok(len(doc.wires) == 5 and [w.type for w in doc.wires] == ["says", "not", "transfer", "entity", "not"], "spec 3.5 example parses: 5 wires, types says/not/transfer/entity/not")
ok(soplang.write(doc) == EXAMPLE, "spec 3.5 example round-trips byte for byte")
ok([w.id for w in doc.top_level()] == ["w1"], "top-level wire is w1 (the others are referenced)")
v, sep = soplang.parse_value('"Paul" & "Peter"'); ok([a.text for a in v] == ["Paul", "Peter"] and sep == "&", 'set value "Paul" & "Peter"')
v, sep = soplang.parse_value('Paul|Peter'); ok(sep == "|" and len(v) == 2, "alternatives without spaces Paul|Peter")
v, _ = soplang.parse_value(r'"aisle \"B\" \\ x"'); ok(v[0].text == 'aisle "B" \\ x', "quoted value with \\\" and \\\\ escapes")
w = soplang.Wire("x", "transfer"); w.set("place", "New York"); w.set("agent", ("&", ["Paul", "$odd"]))
ok(soplang.write(soplang.Doc(wires=[w])) == '@x transfer\nplace="New York"\nagent=Paul & "$odd"\n', "writer quotes values with spaces / reserved first char")

tmp = tempfile.mkdtemp(prefix="sop_t1_")
with open(os.path.join(tmp, "kinds.tsv"), "w") as f: f.write("Paul\tP\nAnna\tP\nMaria\tP\nfiles\tN\nsaved\tA\ngives\tgives\tgive\n")
lp = os.path.join(tmp, "legend.json"); json.dump(LEGEND, open(lp, "w"), indent=1); lg = Legend(lp)
ok(not lg.errors, f"legend loads ({lg.sha[:20]}…)")
good = EXAMPLE.replace("sha256:3f9a…", lg.sha); gp = os.path.join(tmp, "good.sop"); open(gp, "w").write(good)
rep, _ = check_file(gp, lg); ok(rep.ok, "spec 3.5 example (with this legend's hash) passes sop check")

NEG = {
    "duplicate id": (good.replace("@w4 entity", "@w2 entity").replace("object=$w4", "object=$w2"), "duplicate wire id"),
    "dangling $ref": (good.replace("scope=$w3", "scope=$w9"), "dangling reference $w9"),
    "reference cycle": (good.replace("scope=saved", "scope=$w4"), "reference cycle"),
    "unknown key": (good.replace("recipient=Anna", "receiver=Anna"), "unknown key 'receiver'"),
    "& and | mixed": (good.replace("recipient=Anna", "recipient=Anna & Maria | Paul"), "'&' and '|' mixed"),
    "_asked without question mood": (good.replace("recipient=Anna", "recipient=?\n_asked=recipient"), "_asked is only allowed with _mood=question"),
    "legend hash mismatch": (EXAMPLE, "legend hash mismatch"),
    "unknown wire type": (good.replace("@w3 transfer", "@w3 donation"), "unknown wire type 'donation'"),
    "two _text lines": (good.replace("_sentence=0", "_text=again"), "has 2 _text lines"),
    "bare value with spaces": (good.replace("recipient=Anna", "recipient=Anna Maria"), "must be quoted"),
    "_scope on a non-event wire": (good.replace("head=files", "head=files\n_scope=agent"), "_scope is only allowed on event wires"),
    "_scope with a value that is not a role": (good.replace("recipient=Anna", "recipient=Anna\n_scope=price"), "must be a legend role"),
    "event without verbs (legend)": (None, None),
}
for name, (text, want) in NEG.items():
    if text is None: continue
    p = os.path.join(tmp, name.replace(" ", "_").replace("$", "") + ".sop"); open(p, "w").write(text)
    rep, _ = check_file(p, lg)
    hit = [(l, m) for l, m in rep.errors if want in m]
    ok(bool(hit) and hit[0][0] is not None, f"negative case '{name}' rejected: " + (f"line {hit[0][0]}: {hit[0][1]}" if hit else f"got {rep.errors}"))

# _scope (spec 3.3, quantifier world v2): accepted on an event wire with a legend role or 'both'
for v in ("agent", "both"):
    p = os.path.join(tmp, f"scope_{v}.sop"); open(p, "w").write(good.replace("recipient=Anna", f"recipient=Anna\n_scope={v}"))
    rep, _ = check_file(p, lg); ok(rep.ok, f"_scope={v} on the event wire accepted")
bad_leg = dict(LEGEND, events=[{"id": "transfer", "verbs": []}]); nv = tempfile.mkdtemp(prefix="sop_t1nv_"); os.makedirs(os.path.join(nv, "noverbs"), exist_ok=True)
json.dump(bad_leg, open(os.path.join(nv, "noverbs", "legend.json"), "w")); ok(any("has no verbs" in e for e in Legend(os.path.join(nv, "noverbs", "legend.json")).errors),
                                                                              "legend inconsistency: an event without verbs is an error (spec 6.1)")

# several worlds in one file (amendment): each top-level wire is checked against the legend of ITS _world
os.makedirs(os.path.join(tmp, "commands"), exist_ok=True); os.makedirs(os.path.join(tmp, "shop"), exist_ok=True)
import shutil
shutil.copy(lp, os.path.join(tmp, "commands", "legend.json")); shutil.copy(os.path.join(tmp, "kinds.tsv"), os.path.join(tmp, "commands", "kinds.tsv"))
SHOP = {"version": 1, "roles": [{"id": "buyer", "question": "who buys?"}, {"id": "item", "question": "what?"}], "events": [{"id": "purchase", "verbs": ["buy"]}],
        "operators": ["not"], "moods": ["statement"], "kinds": {"source": "lexicon", "lexicon": "kinds.tsv"}}
json.dump(SHOP, open(os.path.join(tmp, "shop", "legend.json"), "w")); shutil.copy(os.path.join(tmp, "kinds.tsv"), os.path.join(tmp, "shop", "kinds.tsv"))
lc, ls = Legend(os.path.join(tmp, "commands", "legend.json")), Legend(os.path.join(tmp, "shop", "legend.json"))
MIX = (f"!legend commands {lc.sha}\n!legend shop {ls.sha}\n\n" + good.split("\n\n", 1)[1].replace("_sentence=0", "_sentence=0\n_world=commands")
       + '\n@s1 purchase\n_text=Anna buys a book.\n_world="shop"\nbuyer=Anna\nitem=book\n')
mp = os.path.join(tmp, "mixed.sop"); open(mp, "w").write(MIX)
rep, _ = check_file(mp, {"commands": lc, "shop": ls}); ok(rep.ok, f"a file mixing two worlds passes with two legends {rep.errors}")
rep, _ = check_file(mp, {"commands": lc}); ok(not rep.ok and any("no legend given for world 'shop'" in m for _, m in rep.errors), "the same file fails with one legend: " + (rep.errors[0][1] if rep.errors else ""))
bad = MIX.replace(f"!legend shop {ls.sha}", f"!legend shop {lc.sha}"); bp = os.path.join(tmp, "mixed_badhash.sop"); open(bp, "w").write(bad)
rep, _ = check_file(bp, {"commands": lc, "shop": ls}); ok(any("legend hash mismatch for world 'shop'" in m for _, m in rep.errors), "per-world hash check: a wrong !legend shop hash is rejected")
r = subprocess.run([os.path.join(TOOL, "bin", "sop"), "check", "--legends", tmp, mp], capture_output=True, text=True); ok(r.returncode == 0, "sop check --legends <dir> on the mixed file: exit 0")
r = subprocess.run([os.path.join(TOOL, "bin", "sop"), "check", "--legend", os.path.join(tmp, "commands", "legend.json"), mp], capture_output=True, text=True); ok(r.returncode == 1, "sop check with one --legend on the mixed file: exit 1")

# the CLI: exit 1 on errors, 0 when clean
sop = os.path.join(TOOL, "bin", "sop")
r = subprocess.run([sop, "check", "--legend", lp, gp], capture_output=True, text=True); ok(r.returncode == 0, "sop check exit 0 on the good file")
r = subprocess.run([sop, "check", "--legend", lp, gp, os.path.join(tmp, "reference_cycle.sop")], capture_output=True, text=True)
ok(r.returncode == 1 and "reference cycle" in r.stdout, "sop check exit 1 with the error printed as file:line")
print(f"\ntest1: {'PASS' if not fails else f'{fails} FAILED'}"); sys.exit(1 if fails else 0)
