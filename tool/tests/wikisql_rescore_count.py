"""
wikisql_rescore_count.py — one re-scoring of the WikiSQL model (CPU, no retraining): the condition SET read with the NUMBER of conditions
GIVEN (the top-n columns by their condition score, n = the gold number), which is what v11's FULL is given (it decodes COND_k for k < the
gold number of conditions), next to the product readout (the members decided by the member threshold). One change: the count.
Run: python3 tool/tests/wikisql_rescore_count.py --model <world>/model --world <world> --out <run folder>
"""
import argparse, json, os, sys
import numpy as np
TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, TOOL)
from sop import soplang, world
from sop.infer import Reader
from sop.legend import tokenize


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--world", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); rd = Reader(a.model); lg = rd.legend
    doc = soplang.load(os.path.join(a.world, "examples", "dev.sop")); by = doc.by_id()
    acc = {k: [] for k in ("select", "aggregate", "cond1 decided", "cond1 count given", "set decided", "set count given", "FULL decided", "FULL count given")}
    for top in doc.top_level():
        g = world.frame_of(top, by, lg); table = [x.text for x in top.get("_table").atoms]; toks = tokenize(top.value("_text"))
        S, kept, _, _ = rd.chart(toks); f = rd.read(S, toks, table=table)
        gh = world.filler_heads(g["roles"].get("condition")); ph = set(world.filler_heads(f["roles"].get("condition")))
        ratios = f["presence"].get("condition.members", {}); top_n = set(sorted(ratios, key=lambda c: -ratios[c])[:len(gh)]) if gh else set()
        sel = f["roles"].get("select") == g["roles"].get("select"); agg = f["roles"].get("aggregate") == g["roles"].get("aggregate")
        acc["select"].append(sel); acc["aggregate"].append(agg)
        if gh: acc["cond1 decided"].append(gh[0] in ph); acc["cond1 count given"].append(gh[0] in top_n)
        acc["set decided"].append(set(gh) == ph); acc["set count given"].append(set(gh) == top_n)
        acc["FULL decided"].append(sel and agg and set(gh) == ph); acc["FULL count given"].append(sel and agg and set(gh) == top_n)
    out = [f"WikiSQL re-scoring (model {a.model}, seed {rd.cfg['seed']}; dev {len(acc['select'])} questions): the condition set read two ways, one change "
           "(the number of conditions GIVEN, as v11's FULL is given it, vs DECIDED by the member threshold — the product).", "",
           "| axis | product: members decided | the number of conditions given (v11's protocol) | RESULTS 8c (dev, 3 seeds) |", "|---|---|---|---|",
           f"| selected column | {np.mean(acc['select']):.3f} | (same) | 0.810±0.004 |", f"| aggregator | {np.mean(acc['aggregate']):.3f} | (same) | 0.814±0.003 |",
           f"| first gold condition column read | {np.mean(acc['cond1 decided']):.3f} | {np.mean(acc['cond1 count given']):.3f} | 0.632±0.004 |",
           f"| the condition column set exact | {np.mean(acc['set decided']):.3f} | {np.mean(acc['set count given']):.3f} | — |",
           f"| FULL (select + aggregate + condition columns) | {np.mean(acc['FULL decided']):.3f} | {np.mean(acc['FULL count given']):.3f} | 0.376±0.003 |"]
    text = "\n".join(out); print(text)
    with open(os.path.join(a.out, "rescore_count_given_seed0.md"), "w") as fh: fh.write(text + "\n")


if __name__ == "__main__":
    main()
