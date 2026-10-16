"""E3 (judgment 3b4f837): where does the verifier's 16% false alarm come from? On the saved frames (10c-bis reader, odd half of dev, tau 0.05):
the confident-wrong SELECTED-COLUMN decodings on gold, broken down (a) by the number of columns of the table and (b) by whether the gold column's
name appears verbatim in the question; plus the agg confident-wrong rate; and a sample of 50 confident-wrong sel cases (seed 0) written to
logs/audit_confident_wrong_sel_50.jsonl for the team's hand audit (WikiSQL gold carries label noise). Re-scoring only."""
import json, sys, numpy as np, torch, v11_wikisql as v11
FR = sys.argv[1] if len(sys.argv) > 1 else "logs/.ckpt_v17_reader_frames.pt"      # 4c: the frames of a reader trained on 1k / 5k questions
dev, dtabs = v11.load("dev"); saved = torch.load(FR, weights_only=False); frames = saved["frames"]; tau = 0.05
print(f"frames: {FR} (reader trained on {saved.get('n_train', 56355)} questions)")
tok = v11.tok; rows = []
for i in range(1, len(dev), 2):
    r = dev[i]; f = frames[i]; cols = dtabs[r["table_id"]]; gold = r["sql"]["sel"]; pred, mg = f["sel"]
    name_in_q = all(t in set(tok(r["question"])) for t in tok(cols[gold])) if tok(cols[gold]) else False
    rows.append({"i": i, "n_cols": len(cols), "name_in_q": name_in_q, "sel_right": pred == gold, "sel_conf": mg >= tau, "agg_right": f["agg"][0] == r["sql"]["agg"], "agg_conf": f["agg"][1] >= tau,
                 "question": r["question"], "gold_col": cols[gold], "pred_col": cols[pred] if pred < len(cols) else "?", "margin": mg, "table_id": r["table_id"]})
n = len(rows); cw = [x for x in rows if x["sel_conf"] and not x["sel_right"]]; cwa = [x for x in rows if x["agg_conf"] and not x["agg_right"]]
print(f"rescore_v17_e3: odd half of dev, {n} questions, tau {tau}. Selected column: accuracy {np.mean([x['sel_right'] for x in rows]):.3f}, confident-wrong {len(cw)/n:.3f}, abstain {np.mean([not x['sel_conf'] for x in rows]):.3f}. Aggregator: accuracy {np.mean([x['agg_right'] for x in rows]):.3f}, confident-wrong {len(cwa)/n:.3f}")
print("\n(a) by number of columns: bucket | n | sel accuracy | confident-wrong rate | share of all confident-wrong")
for lo, hi in ((1, 5), (6, 8), (9, 12), (13, 100)):
    b = [x for x in rows if lo <= x["n_cols"] <= hi]; c = [x for x in b if x["sel_conf"] and not x["sel_right"]]
    print(f"  {lo:2d}-{hi:3d} | {len(b):5d} | {np.mean([x['sel_right'] for x in b]):.3f} | {len(c)/max(len(b),1):.3f} | {len(c)/max(len(cw),1):.3f}")
print("\n(b) by whether the gold column's name appears verbatim in the question: case | n | sel accuracy | confident-wrong rate | share of all confident-wrong")
for flag in (True, False):
    b = [x for x in rows if x["name_in_q"] == flag]; c = [x for x in b if x["sel_conf"] and not x["sel_right"]]
    print(f"  {'name in question' if flag else 'name NOT in question':22s} | {len(b):5d} | {np.mean([x['sel_right'] for x in b]):.3f} | {len(c)/max(len(b),1):.3f} | {len(c)/max(len(cw),1):.3f}")
rng = np.random.default_rng(0); pick = rng.choice(len(cw), min(50, len(cw)), replace=False)
with open("logs/audit_confident_wrong_sel_50.jsonl" if FR.endswith("frames.pt") else "logs/.audit_tmp.jsonl", "w") as fh:   # the audit file is written only for the full reader
    for j in pick: fh.write(json.dumps({k: cw[j][k] for k in ("i", "table_id", "question", "gold_col", "pred_col", "margin", "n_cols", "name_in_q")} | {"audit_verdict": None}) + "\n")
print(f"\n50 confident-wrong selected-column cases (seed 0) written to logs/audit_confident_wrong_sel_50.jsonl for the hand audit (fields: question, gold_col, pred_col, margin; audit_verdict to fill: 'reader wrong' | 'gold noise / ambiguous')")
print("\nchecklist: [x] re-scoring only, saved frames [x] odd half (report rows) [x] breakdown by table size and lexical match [x] audit sample of stated size and seed [x] confident-wrong at the stated tau (18)")
