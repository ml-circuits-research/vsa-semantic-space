"""
wikisql_table.py — the P1 rehearsal's table (judgment 934fbae E2): the package on WikiSQL next to RESULTS 8c / 8d, and the SOP round trip.
Run: python3 tool/tests/wikisql_table.py --root tool/runs/wikisql_<commit> [--world <world folder, for the round trip>]
"""
import argparse, json, os, sys
TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, TOOL)


def ms(m, k, split="dev"):
    v = (m or {}).get("mean_over_seeds", {}).get(split, {}).get("catalog candidates", {}).get(k)
    return f"{v['mean']:.3f}±{v['std']:.3f}" if v else "—"


def roundtrip(wd, data):
    from sop import soplang, world
    from sop.legend import Legend
    from sop.check import check_file
    import export_wikisql as ex
    lg = Legend(os.path.join(wd, "legend.json")); out = []
    for name, split in (("train.sop", "train"), ("dev.sop", "dev")):
        p = os.path.join(wd, "examples", name); rep, doc = check_file(p, {"wikisql": lg}); text = open(p).read()
        rows, tabs = ex.load(data, split, 10 ** 9); by = doc.by_id(); mism = 0; tops = doc.top_level()
        for r, top in zip(rows, tops):
            f = world.frame_of(top, by, lg); h = tabs[r["table_id"]]; sql = r["sql"]
            conds = sorted([["entity", h[c], ["word", ex.OPS[o]], None] for c, o, v in sql["conds"]], key=json.dumps)
            want = (["word", h[sql["sel"]]], ["word", ex.AGG[sql["agg"]]], (conds[0] if len(conds) == 1 else ["set"] + conds) if conds else None)
            got = (f["roles"].get("select"), f["roles"].get("aggregate"), f["roles"].get("condition"))
            vals = sorted(by[a.text].value("quantity") for a in (top.get("condition").atoms if top.get("condition") else []))
            mism += json.dumps(want) != json.dumps(got) or vals != sorted(str(v) for c, o, v in sql["conds"])
        out.append(f"| {name} | {len(tops)} | {'yes' if rep.ok else 'NO ' + str(rep.errors[:2])} | {'yes' if soplang.write(soplang.parse(text)) == text else 'NO'} | {mism} |")
    return ["", "SOP round trip (write -> sop check -> read -> write; frames read back vs the SQL):", "| file | questions | sop check | byte-identical | frames differing from the SQL |",
            "|---|---|---|---|---|"] + out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--world"); ap.add_argument("--data"); a = ap.parse_args()
    p = os.path.join(a.root, "wikisql", "metrics.json"); m = json.load(open(p)) if os.path.exists(p) else None
    out = ["WikiSQL through the tool (question only; dev 8,421), mean ± std over the seeds of the run; RESULTS 8c (3 seeds, D = 512, 3 epochs, 56,355 train) "
           "and 8d (56k row) beside it.", "",
           "| axis | package | RESULTS 8c (dev) | RESULTS 8d (56k, dev) |", "|---|---|---|---|",
           f"| selected column | {ms(m, 'role select')} | 0.810±0.004 | 0.816 |",
           f"| aggregator | {ms(m, 'role aggregate')} | 0.814±0.003 | 0.801 |",
           f"| condition column: the first gold condition column in the decoded SET (8c's condition-1 analog) | {ms(m, 'set condition: first gold member in the decoded set')} | 0.632±0.004 (condition-1 column) | 0.626 |",
           f"| condition columns: the decoded set EXACT | {ms(m, 'set condition: exact')} | — (8c: indexed slots) | — |",
           f"| FULL query (select + aggregate + the condition column set; operators and values not read) | {ms(m, 'FULL')} | 0.376±0.003 (all condition columns, indexed) | 0.368 |",
           f"| FULL, pooled-midpoint thresholds (the comparison) | {ms(m, 'FULL [pooled midpoint (comparison)]')} | | |",
           f"| condition presence: present read present / absent read absent | {ms(m, 'presence condition: present read present')} / {ms(m, 'presence condition: absent read absent')} | | |"]
    if m: out.append(f"\nseeds {m['seeds']}; chance {m.get('chance')}")
    if a.world:
        import export_wikisql as ex
        out += roundtrip(a.world, a.data or ex.default_data())
    text = "\n".join(out); print(text); os.makedirs(a.root, exist_ok=True)
    with open(os.path.join(a.root, "wikisql_table.md"), "w") as fh: fh.write(text + "\n")


if __name__ == "__main__":
    main()
