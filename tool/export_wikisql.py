"""
export_wikisql.py — the P1 rehearsal on real text (judgment 934fbae E2): WikiSQL as a world folder for the tool.

The world's text = the questions, normalized as experiment/v11_wikisql does (lowercase alphanumeric tokens joined by spaces: the reader's input
is the question only, no table, as RESULTS 8c). The gold frames come from the dataset's own SQL (spec 6.4's second source of annotations).
A hand-written legend: roles select (the asked column), aggregate (NONE / MAX / MIN / COUNT / SUM / AVG, a closed value set), condition (a
SET value, per judgment d8f8a98 Q1: one role whose answer is `$c1 & $c2 ...`, each member an entity wire head=<column>, attr=<operator
EQ / GT / LT>, quantity=<value>), not indexed slots. Column-valued roles take their candidates from the sentence's table (`_table`), as
v11's decoder takes the table's columns as candidates (the table is a CANDIDATE set, not an input of the reader).
Kinds (rule 20, GIVEN): v11's procedure — the 40 most frequent tokens of the training questions and headers are their own kinds, every other
word is one of 48 k-means (cosine) clusters of skip-gram vectors trained on the TRAIN questions and the headers of the first 20,000 train
rows only (experiment/v10_controls.skipgram, imported; the clustering lines are copied from v11_wikisql.py's __main__ and say so); dev-only
words keep an untrained skip-gram vector and so a random cluster, as in v11.
Targets (stated difference from v11): v11's S = SEL (x) bundle(col) + AGG (x) f(agg) + sum_k COND_k (x) [COL (x) bundle(col) + OP (x) f(op)
+ VAL (x) bundle(value)]; here R_select (x) bundle(col) + R_aggregate (x) f(agg) + R_condition (x) sum_k [bundle(col) + R_attr (x) f(op)
+ R_quantity (x) bundle(value)] — the condition terms SUPERPOSED in one role (the set value) instead of indexed slots.
Run: python3 tool/export_wikisql.py [--out tool/worlds_wikisql] [--n_train 56355] [--n_dev 8421] [--data <dir with train.jsonl ...>]
"""
import argparse, collections, json, os, re, subprocess, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); EXP = os.path.join(REPO, "experiment")
sys.path.insert(0, HERE)
import numpy as np
from sop import soplang
from sop.legend import sha256_file

AGG = ["NONE", "MAX", "MIN", "COUNT", "SUM", "AVG"]; OPS = ["EQ", "GT", "LT"]
def tok(s): return re.findall(r"[a-z0-9]+", str(s).lower())           # = experiment/v11_wikisql.tok


def default_data():
    p = os.path.join(EXP, "corp", "wikisql", "data")
    if os.path.exists(p): return p
    try:
        common = subprocess.run(["git", "-C", REPO, "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True).stdout.strip()
        q = os.path.join(os.path.dirname(common), "experiment", "corp", "wikisql", "data")
        if os.path.exists(q): return q
    except Exception: pass
    return p


def load(data, split, n):
    rows = [json.loads(l) for l in open(os.path.join(data, f"{split}.jsonl"))][:n]; tabs = {}
    for l in open(os.path.join(data, f"{split}.tables.jsonl")): t = json.loads(l); tabs[t["id"]] = t["header"]
    return rows, tabs


def kinds(train, ttabs, dev, dtabs, k=48, n_freq=40, seed=0):
    """v11's categories (copied from v11_wikisql.py __main__, unchanged in substance; the skip-gram is experiment/v10_controls.skipgram)"""
    sys.path.insert(0, EXP)
    import torch, v10_controls as v10
    vocab = {}; vid = lambda w: vocab.setdefault(w, len(vocab))
    for rows, tabs in ((train, ttabs), (dev, dtabs)):
        for r in rows:
            for w in tok(r["question"]): vid(w)
            for h in tabs[r["table_id"]]:
                for w in tok(h): vid(w)
            for c in r["sql"]["conds"]:
                for w in tok(c[2]): vid(w)
    freq = collections.Counter(w for r in train for w in tok(r["question"])); freq.update(w for r in train for h in ttabs[r["table_id"]] for w in tok(h))
    singles = [w for w, _ in freq.most_common(n_freq)]
    sents = [tok(r["question"]) for r in train] + [tok(h) for r in train[:20000] for h in ttabs[r["table_id"]]]
    emb = v10.skipgram([s for s in sents if len(s) > 1], vocab, dim=64, window=2, epochs=3, device="cpu", seed=seed)
    X = torch.tensor(emb); X = X / (X.norm(dim=1, keepdim=True) + 1e-8); V = len(vocab)
    gk = torch.Generator(device="cpu").manual_seed(seed); C = X[torch.randperm(V, generator=gk)[:k]].clone()
    for it in range(25):
        asg = (X @ C.T).argmax(1)
        for c in range(k):
            mem = X[asg == c]
            if len(mem): C[c] = mem.mean(0) / (mem.mean(0).norm() + 1e-8)
    asg = (X @ C.T).argmax(1).numpy()
    qwords = {w for rows in (train, dev) for r in rows for w in tok(r["question"])}
    return {w: (w if w in singles else f"C{int(asg[i])}") for w, i in vocab.items() if w in qwords}, singles


def wires_of(r, header, ids):
    q = soplang.Wire(next(ids), "query"); sql = r["sql"]
    q.set("_text", " ".join(tok(r["question"]))); q.set("_source", "wikisql-gold (the dataset's SQL)"); q.set("_world", "wikisql")
    q.lines.append(soplang.line("_table", ("&", list(header)) if len(header) > 1 else header[0]))
    q.set("select", header[sql["sel"]]); q.set("aggregate", AGG[sql["agg"]])
    ents = []
    for c, o, v in sql["conds"]:
        e = soplang.Wire(next(ids), "entity"); e.set("head", header[c]); e.set("attr", OPS[o]); e.set("quantity", str(v)); ents.append(e)
    if ents: q.set("condition", ("&", [("ref", e.id) for e in ents]) if len(ents) > 1 else ("ref", ents[0].id))
    return [q] + ents


LEGEND = {"version": 1,
          "roles": [{"id": "select", "question": "which column is asked for?"}, {"id": "aggregate", "question": "which aggregate of it?"},
                    {"id": "condition", "question": "which columns constrain the rows (a set)?"}],
          "events": [{"id": "query", "verbs": ["what", "which", "how", "who", "when", "where", "name", "tell", "list", "give"]}],
          "operators": [], "moods": ["question"], "kinds": {"source": "lexicon", "lexicon": "kinds.tsv"}, "values": [],
          "reader": {"predicate_role": False, "occupancy_marker": False, "definite_dets": [], "value_identity": "bundle_alnum",
                     "closed_values": {"aggregate": AGG, "attr": OPS}, "table_roles": ["select", "condition"], "set_roles": ["condition"],
                     "value_terms": True, "no_sense_share": 1.0, "full_excludes": ["attr"]}}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="export_wikisql"); ap.add_argument("--out", default=os.path.join(HERE, "worlds_wikisql"))
    ap.add_argument("--n_train", type=int, default=56355); ap.add_argument("--n_dev", type=int, default=8421); ap.add_argument("--data", default=None)
    a = ap.parse_args(argv); data = a.data or default_data()
    train, ttabs = load(data, "train", a.n_train); dev, dtabs = load(data, "dev", a.n_dev)
    wd = os.path.join(a.out, "wikisql")
    for d in ("examples", "raw", "audit", "model", "runs", "versions", "requests"): os.makedirs(os.path.join(wd, d), exist_ok=True)
    lex, singles = kinds(train, ttabs, dev, dtabs)
    with open(os.path.join(wd, "kinds.tsv"), "w") as f:
        f.write("# word\tkind — v11's categories: 40 frequent tokens as their own kinds, 48 skip-gram clusters C<i> (GIVEN, rule 20)\n")
        for w in sorted(lex): f.write(f"{w}\t{lex[w]}\n")
    lp = os.path.join(wd, "legend.json"); json.dump(LEGEND, open(lp, "w"), indent=1); sha = sha256_file(lp)
    json.dump({"name": "wikisql", "language": "en", "description": "WikiSQL questions -> SQL frames (select, aggregate, condition SET); the P1 rehearsal",
               "version": 1, "models": {"pipeline": "claude-haiku-5-5", "annotate": "claude-haiku-5-5", "second_look": "claude-haiku-5-5", "evolve": "claude-opus-5-5"},
               "second_look_escalate": 0.05, "second_look_retrain": 0.05, "min_examples": 500, "max_versions_per_run": 3}, open(os.path.join(wd, "world.json"), "w"), indent=1)
    for name, rows, tabs in (("train.sop", train, ttabs), ("dev.sop", dev, dtabs)):
        doc = soplang.Doc(directives=[("world", "wikisql", None), ("legend", sha, None), ("generated", "export_wikisql.py from the WikiSQL SQL (gold)", None)])
        ids = (f"w{i}" for i in range(1, 10 ** 9))
        for r in rows: doc.wires += wires_of(r, tabs[r["table_id"]], ids)
        with open(os.path.join(wd, "examples", name), "w") as f: f.write(soplang.write(doc))
    json.dump({"heldout_sentence_share": 0.0, "new_identities": "none: real text (the identities are the dataset's words)",
               "heldout_formulations": "none: real text, no templates; RESULTS 8c trains on all train questions and reports dev",
               "file_splits": {"examples/dev.sop": "dev"},
               "protocol": f"RESULTS 8c: question only, train {len(train)}, dev {len(dev)}"}, open(os.path.join(wd, "split_hints.json"), "w"), indent=1)
    if not os.path.exists(os.path.join(wd, "status.json")): json.dump({"label": None}, open(os.path.join(wd, "status.json"), "w"))
    print(f"export_wikisql: {wd} — train {len(train)}, dev {len(dev)}; kinds {len(set(lex.values()))} ({len(singles)} frequent tokens + clusters) over "
          f"{len(lex)} question words; legend {sha[:19]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
