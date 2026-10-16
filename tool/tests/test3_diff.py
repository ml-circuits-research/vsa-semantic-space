"""
Test 3 (build order step 3): `sop diff` (frame.py's diff, four verdicts) and `sop stats`, with INJECTED-ERROR pairs.

The protocol of experiment/v17_verifier (RESULTS 10c): the formalization to verify is the GOLD frame with ONE injected error per instance;
the reader reads the sentence; `sop diff reader.sop proposed.sop` gives a verdict; per error type we count detected (contradiction, or
compatible-incomplete for a dropped / added slot) / unverifiable (the reader's margin on the disputed slot under tau) / missed (agreement:
the reader confirms the error with margin >= tau = confident-wrong), and on the gold itself false alarm / unverifiable / confirmed.
Adaptation (stated): v17's propose_with_error injects errors into WikiSQL SQL dicts (select column, aggregator, conditions) and needs the
WikiSQL data and the v11 reader; it cannot be applied to SOP frames as is. The same one-error-per-instance scheme is applied here to the
constructed commands world, with error types mapped by function: change_action / change_patient (~ change_sel), flip_negation (~ flip_agg,
an operator), flip_attr_negation (scope inside the patient), drop_destination (~ drop_cond), add_destination (~ add_cond).
Test criterion (the 10c shape, not its numbers): on role errors the reader detects more often than it raises false alarms on gold, and the
missed (confident-wrong) rate stays at most the gold false-alarm rate. Needs tool/worlds/commands/model (tests/test2_formalize.sh).
Run: python3 tool/tests/test3_diff.py
"""
import collections, copy, os, subprocess, sys
import numpy as np
TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, TOOL)
from sop import soplang, world
from sop.infer import Reader, formalize_text, idgen
from sop.diff import diff_docs

W = os.environ.get("WORLDS_DIR", os.path.join(TOOL, "worlds")); OUT = os.path.join(TOOL, "tests", "out"); os.makedirs(OUT, exist_ok=True)
MD = os.path.join(W, "commands", "model")
if not os.path.exists(os.path.join(MD, "config.json")): print("test3: no commands model; run tests/test2_formalize.sh first"); sys.exit(1)
rd = Reader(MD); lg = rd.legend
gold_doc = soplang.load(os.path.join(W, "commands", "examples", "eval_new_identities.sop"))
gtops = gold_doc.top_level(); by = gold_doc.by_id()                  # new patients (never in training), all forms, depth 0
frames = [world.frame_of(w, by, lg) for w in gtops]; texts = [w.value("_text") for w in gtops]
actions = sorted(w for w, e in rd.words.items() if e["kind"] == "ACT"); pats = sorted(w for w, e in rd.words.items() if e["kind"] == "PAT")
dests = sorted(w for w, e in rd.words.items() if e["kind"] == "DEST"); rng = np.random.default_rng(0)

def mutate(f, kind):
    g = copy.deepcopy(f); R = g["roles"]
    if kind == "change_action": R["action"] = ["word", str(rng.choice([a for a in actions if a != R["action"][1]]))]
    elif kind == "change_patient": R["patient"][1] = str(rng.choice([p for p in pats if p != R["patient"][1]]))
    elif kind == "flip_negation": g["neg"] = not g["neg"]
    elif kind == "flip_attr_negation":
        a = R["patient"][2]; R["patient"][2] = ["negword" if a[0] == "word" else "word", a[1]]
    elif kind == "drop_destination":
        if "destination" not in R: return None
        del R["destination"]
    elif kind == "add_destination":
        if "destination" in R: return None
        R["destination"] = ["word", str(rng.choice(dests))]
    return g

def doc_of(fs):
    d = soplang.Doc(directives=[("world", "commands", None), ("legend", lg.sha, None)]); ids = idgen("g")
    for n, (f, t) in enumerate(zip(fs, texts)):
        ws = rd.frame_wires(dict(f, margins={}), ids, t, n); ws[0].lines.insert(2, soplang.Line("_world", "commands")); d.wires += ws
    return d

reader_doc, _ = formalize_text(rd, texts); open(os.path.join(OUT, "diff_reader.sop"), "w").write(soplang.write(reader_doc))
KINDS = ["none (gold)", "change_action", "change_patient", "flip_negation", "flip_attr_negation", "drop_destination", "add_destination"]
table = {}
for kind in KINDS:
    ms = [f if kind.startswith("none") else mutate(f, kind) for f in frames]; idx = [i for i, m in enumerate(ms) if m is not None]
    prop = doc_of([m if m is not None else f for m, f in zip(ms, frames)])
    pp = os.path.join(OUT, f"diff_proposed_{kind.split()[0]}.sop"); open(pp, "w").write(soplang.write(prop))
    out, _ = diff_docs(reader_doc, prop, "diff_reader.sop", os.path.basename(pp), rd.tau, lg)
    v = [out.wires[i].value("_verdict") for i in idx]; c = collections.Counter(v); n = len(v)
    if kind.startswith("none"):
        table[kind] = (n, c["contradiction"] / n, c["unverifiable"] / n, (c["agreement"] + c["compatible-incomplete"]) / n)
    else:
        table[kind] = (n, (c["contradiction"] + c["compatible-incomplete"]) / n, c["unverifiable"] / n, c["agreement"] / n)
print(f"injected-error verifier table (commands world, smoke reader in {MD}; tau {rd.tau} stated; BUILT by this test, 1 seed, small data)")
print("| error type | n | reader: detected / unverifiable / missed (confident-wrong) |\n|---|---|---|")
for k, (n, a, b, c) in table.items():
    print(f"| {k} | {n} | " + (f"FALSE ALARM {a:.3f} / unverifiable {b:.3f} / confirmed {c:.3f}" if k.startswith("none") else f"{a:.3f} / {b:.3f} / {c:.3f}") + " |")
fa = table["none (gold)"][1]; role = [table[k] for k in ("change_action", "change_patient", "flip_negation")]
det = np.mean([r[1] for r in role]); missed = np.mean([r[3] for r in role])
fails = 0
def ok(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg); fails += 0 if cond else 1
ok(det > fa, f"role errors detected ({det:.3f}) more often than false alarms on gold ({fa:.3f}) — the 10c shape")
ok(missed <= max(fa, 0.05), f"missed / confident-wrong on role errors ({missed:.3f}) at most the gold false-alarm rate ({fa:.3f}) or 0.05")
r = subprocess.run([os.path.join(TOOL, "bin", "sop"), "diff", os.path.join(OUT, "diff_reader.sop"), os.path.join(OUT, "diff_proposed_change_action.sop"), "--model", MD],
                   capture_output=True, text=True)
ok(r.returncode == 0 and "@d1 verdict" in r.stdout and "_verdict=" in r.stdout, "sop diff CLI writes verdict wires (" + r.stderr.strip() + ")")
vp = os.path.join(OUT, "diff_verdicts.sop"); open(vp, "w").write(r.stdout)
r3 = subprocess.run([os.path.join(TOOL, "bin", "sop"), "check", "--legend", os.path.join(MD, "legend.json"), vp], capture_output=True, text=True)
ok(r3.returncode == 0, "the verdict file passes sop check (verdict wires, plain ids)")
# a world mismatch is reported, not scored
other = soplang.write(reader_doc).replace("_world=commands", "_world=reference")
op = os.path.join(OUT, "diff_other_world.sop"); open(op, "w").write(other)
out, cnt = diff_docs(reader_doc, soplang.parse(other), "a", "b", rd.tau, lg)
ok(cnt.get("world mismatch (unverifiable)") == len(texts), f"pairs from two worlds: every pair reported as a world mismatch ({cnt})")
r = subprocess.run([os.path.join(TOOL, "bin", "sop"), "stats", "--model", MD], capture_output=True, text=True)
ok(r.returncode == 0 and "FULL" in r.stdout, "sop stats prints the metrics table")
print(f"\ntest3 (diff/stats): {'PASS' if not fails else f'{fails} FAILED'}"); sys.exit(1 if fails else 0)
