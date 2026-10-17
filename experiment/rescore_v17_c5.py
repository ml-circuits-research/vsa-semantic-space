"""C5 re-scoring (judgment e9274a9 Q2): the verifier's CONDITION verdict with the MARKER'S margin as the abstention statistic, on the saved frames
of the C5 reader (logs/.ckpt_v17_reader_frames_occ.pt; RESULTS 10c-sexies). No retraining, no threshold tuning: the presence threshold is the
calibration-half midpoint saved with the frames. Same injected errors as v17 (same rng, same loop over the odd rows).
   old statistic (10c-sexies): a condition-set disagreement is "error" when the minimum COLUMN-identity margin over the slots >= tau, else "unverifiable";
   new statistic: "error" when the minimum over the slots of |marker cosine - threshold| >= tau (the marker's margin), else "unverifiable".
Reported per error type: detected / unverifiable / missed (confident-wrong) under both statistics; false alarm on gold under both. tau = 0.05 (the stated tau)."""
import sys, collections, numpy as np, torch, v11_wikisql as v11
from v17_verifier import propose_with_error

FR = sys.argv[1] if len(sys.argv) > 1 else "logs/.ckpt_v17_reader_frames_occ.pt"; tau = 0.05
dev, dtabs = v11.load("dev"); saved = torch.load(FR, weights_only=False); frames = saved["frames"]; thr = saved["pres_thr"]
print(f"rescore_v17_c5: frames {FR}; presence threshold (calibration half) {thr:.3f}; tau {tau}; odd half of dev ({len(dev)//2} questions)")
def verdict(frame, proposed, stat):
    if proposed["sel"] != frame["sel"][0]: return ("error", "sel") if frame["sel"][1] >= tau else ("unverifiable", "sel")
    if proposed["agg"] != frame["agg"][0]: return ("error", "agg") if frame["agg"][1] >= tau else ("unverifiable", "agg")
    read_cols = [c for (c, p, mg) in frame["conds"] if p >= thr]; prop_cols = [c[0] for c in proposed["conds"]]
    if sorted(read_cols) != sorted(prop_cols):
        conf = min([mg for (c, p, mg) in frame["conds"]] + [1.0]) if stat == "column margin" else min([abs(p - thr) for (c, p, mg) in frame["conds"]] + [1.0])
        return ("error", "conds") if conf >= tau else ("unverifiable", "conds")
    return ("ok", None)
kinds = ["drop_cond", "swap_sel_cond", "change_sel", "flip_agg", "add_cond", "change_value"]; rng = np.random.default_rng(0)
stats = {k: {s: collections.Counter() for s in ("column margin", "marker margin")} for k in kinds + ["no_error"]}
for i, r in enumerate(dev):
    cols = dtabs[r["table_id"]]
    if i % 2 == 0: continue
    for kind in kinds + ["no_error"]:
        prop = {"sel": r["sql"]["sel"], "agg": r["sql"]["agg"], "conds": [list(c) for c in r["sql"]["conds"]]} if kind == "no_error" else propose_with_error(r["sql"], len(cols), rng, kind)
        if prop is None: continue
        for s in ("column margin", "marker margin"): v, _ = verdict(frames[i], prop, s); stats[kind][s][v] += 1
print("\nerror type | n | COLUMN-margin statistic (10c-sexies): detected / unverifiable / missed | MARKER-margin statistic: detected / unverifiable / missed")
for kind in kinds:
    c1, c2 = stats[kind]["column margin"], stats[kind]["marker margin"]; n = sum(c1.values())
    print(f"  {kind:14s} | {n:5d} | {c1['error']/n:.3f} / {c1['unverifiable']/n:.3f} / {c1['ok']/n:.3f} | {c2['error']/n:.3f} / {c2['unverifiable']/n:.3f} / {c2['ok']/n:.3f}")
c1, c2 = stats["no_error"]["column margin"], stats["no_error"]["marker margin"]; n = sum(c1.values())
print(f"  {'no_error (gold)':14s} | {n:5d} | FALSE-ALARM {c1['error']/n:.3f} / unverifiable {c1['unverifiable']/n:.3f} / ok {c1['ok']/n:.3f} | FALSE-ALARM {c2['error']/n:.3f} / unverifiable {c2['unverifiable']/n:.3f} / ok {c2['ok']/n:.3f}")
for s in ("column margin", "marker margin"):
    fa = stats["no_error"][s]["error"] / n; te = sum(stats[k][s]["error"] for k in kinds[:5]); tn = sum(sum(stats[k][s].values()) for k in kinds[:5])
    print(f"  {s}: recall on role errors {te/tn:.3f}, precision vs gold {te / (te + fa * tn / 5):.3f}, false alarm {fa:.3f}")
print("\nchecklist: [x] re-scoring only, saved frames of the C5 reader [x] same injected errors (rng, loop) as v17 [x] threshold = the saved calibration-half midpoint, untuned [x] both statistics on the same rows [x] false alarm on gold under both [x] tau stated (18)")
