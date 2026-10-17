"""
Test 7 (judgment 934fbae E2, CPU part): WikiSQL through the tool on a small sample.
  - export (300 train / 100 dev questions): the legend with the condition SET value passes the legend checks; every example passes `sop check`;
  - the frames ROUND-TRIP through SOP Lang: write -> check -> read -> write is byte-identical, and the frame read back equals the frame built
    from the SQL (select column, aggregate, the condition set with operators and values);
  - a smoke `sop train` (D = 128, 1 epoch) runs, and `sop formalize` reads a question with its table given after a TAB.
Run: python3 tool/tests/test7_wikisql.py
"""
import json, os, shutil, subprocess, sys, tempfile
TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, TOOL)
from sop import soplang, world
from sop.legend import Legend
from sop.check import check_file
import export_wikisql as ex

fails = 0
def ok(c, m):
    global fails; print(("PASS " if c else "FAIL ") + m); fails += 0 if c else 1

out = tempfile.mkdtemp(prefix="sop_t7_"); data = ex.default_data()
if not os.path.exists(os.path.join(data, "train.jsonl")): print(f"SKIP test 7: no WikiSQL data under {data}"); sys.exit(0)
ex.main(["--out", out, "--n_train", "300", "--n_dev", "100"]); wd = os.path.join(out, "wikisql")
lg = Legend(os.path.join(wd, "legend.json")); ok(not lg.errors, f"the WikiSQL legend loads (condition is a SET role: {lg.reader['set_roles']}; table roles {lg.reader['table_roles']})")
for name in ("train.sop", "dev.sop"):
    p = os.path.join(wd, "examples", name); rep, doc = check_file(p, {"wikisql": lg}); ok(rep.ok, f"{name}: sop check passes ({rep.n_wires} wires)")
    text = open(p).read(); ok(soplang.write(soplang.parse(text)) == text, f"{name}: write -> read -> write is byte-identical")
rows, tabs = ex.load(data, "dev", 100); doc = soplang.load(os.path.join(wd, "examples", "dev.sop")); by = doc.by_id(); mism = 0
for r, top in zip(rows, doc.top_level()):
    f = world.frame_of(top, by, lg); h = tabs[r["table_id"]]; sql = r["sql"]
    conds = sorted([["entity", h[c], ["word", ex.OPS[o]], None] for c, o, v in sql["conds"]], key=json.dumps)
    want = (["word", h[sql["sel"]]], ["word", ex.AGG[sql["agg"]]], (conds[0] if len(conds) == 1 else ["set"] + conds) if conds else None)
    got = (f["roles"].get("select"), f["roles"].get("aggregate"), f["roles"].get("condition"))
    vals = sorted(by[a.text].value("quantity") for a in (top.get("condition").atoms if top.get("condition") else []))
    if json.dumps(want) != json.dumps(got) or vals != sorted(str(v) for c, o, v in sql["conds"]): mism += 1
ok(mism == 0, f"the frames read back equal the SQL (select, aggregate, condition set with operators and values) on {len(rows)} dev questions")
sop = os.path.join(TOOL, "bin", "sop")
r = subprocess.run([sop, "train", wd, "--D", "128", "--epochs", "1", "--batch", "32", "--n_eval", "100", "--log_every", "5000", "--no_ckpt",
                    "--eval_modes", "primary", "--candidate_sets", "catalog"], capture_output=True, text=True)
ok(r.returncode == 0 and "numpy reader vs torch" in r.stdout, "smoke sop train on the WikiSQL world (D = 128, 1 epoch)")
m = json.load(open(os.path.join(wd, "model", "metrics.json")))["per_seed"][0]["dev"]["catalog candidates"]["acc"]
ok(all(k in m for k in ("role select", "role aggregate", "set condition: exact", "set condition: first gold member in the decoded set", "FULL")),
   "metrics: select, aggregate, the condition set (exact; first gold member in the set = 8c's condition-1 analog), FULL")
q = " ".join(ex.tok(rows[0]["question"])) + "\t" + " | ".join(tabs[rows[0]["table_id"]])
r = subprocess.run([sop, "formalize", "--model", os.path.join(wd, "model"), "-"], input=q + "\n", capture_output=True, text=True)
ok(r.returncode == 0 and "select=" in r.stdout and "aggregate=" in r.stdout, "sop formalize reads a question with its table after a TAB (select / aggregate lines written)")
fp = os.path.join(out, "f.sop"); open(fp, "w").write(r.stdout); rep, _ = check_file(fp, {"wikisql": lg}); ok(rep.ok, "the formalize output passes sop check" + ("" if rep.ok else f" {rep.errors[:3]}"))
# --length_buckets trims each batch to its longest sentence: the trimmed batch gives the same states as the padded one (exact)
os.environ["CUDA_VISIBLE_DEVICES"] = ""; sys.path.insert(0, os.path.join(os.path.dirname(TOOL), "experiment"))
import numpy as np, torch, v8_gates as v8
from sop.train import Shim, Trimmed
wk = dict(lg.lexicon); shim = Shim(wk); m = v8.V8(shim, 32, np.random.default_rng(0), ["path", "chainL", "bigram"], "cpu", gate_init="closed")
m.set_F(np.exp(1j * np.random.default_rng(1).uniform(-np.pi, np.pi, (len(shim.vocab), 32))))
with torch.no_grad(): m.logits.uniform_(-3, 3)
short, long_ = ex.tok(rows[0]["question"])[:5], ex.tok(rows[1]["question"])[:12]
short = [t for t in short if t in wk]; long_ = [t for t in long_ if t in wk]
c2, i2, k2 = m.encode([short, long_]); c1, i1, k1 = m.encode([short])
with torch.no_grad(): padded = m.states(c2[:1], i2[:1], k2[:1])[0]; trimmed = Trimmed(m).states(c2[:1], i2[:1], k2[:1])[0]; alone = m.states(c1, i1, k1)[0]
d1 = float((padded - trimmed).abs().max() / padded.abs().max()); d2 = float((alone - trimmed).abs().max() / alone.abs().max())
ok(d1 < 1e-6 and d2 < 1e-12, f"--length_buckets: the trimmed batch gives the padded batch's states (rel. difference {d1:.1e}; trimmed vs unpadded {d2:.1e})")
shutil.rmtree(out, ignore_errors=True)
print(f"\ntest7 (WikiSQL through the tool): {'PASS' if not fails else f'{fails} FAILED'}"); sys.exit(1 if fails else 0)
