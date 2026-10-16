"""Mismatch-aware abstention (hand-over 78d36e8 Q4): re-scoring of the saved reader frames (10c-bis reader by default), no retraining.
The feature available at verification time is NOT "the gold column's name is in the question" (the gold is unknown) but "the name of the
column the reader CHOSE appears verbatim in the question" (the schema is known). Rule M: abstain when margin < tau OR the chosen column's name
is absent from the question. Control: the margin-only rule. For each rule tau is set on the CALIBRATION half of dev (even rows): the smallest
grid tau with confident-wrong < 2% there; the curve (coverage, covered accuracy, confident-wrong) is reported on the REPORT half (odd rows) at
every grid tau and at the chosen tau. Criterion: confident-wrong on sel < 2% at the coverage that results; coverage reported. Rule 18:
calibration reporting, not tuning. Run: python3 rescore_v17_mismatch.py [frames.pt]"""
import sys, numpy as np, torch, v11_wikisql as v11

FR = sys.argv[1] if len(sys.argv) > 1 else "logs/.ckpt_v17_reader_frames.pt"
dev, dtabs = v11.load("dev"); saved = torch.load(FR, weights_only=False); frames = saved["frames"]; tok = v11.tok
print(f"rescore_v17_mismatch: frames {FR} (reader trained on {saved.get('n_train', 56355)} questions); dev {len(dev)} questions, even half = calibration, odd half = report")
rows = []
for i, r in enumerate(dev):
    f = frames[i]; cols = dtabs[r["table_id"]]; gold = r["sql"]["sel"]; pred, mg = f["sel"]; qt = set(tok(r["question"]))
    inq = lambda c: bool(tok(cols[c])) and all(t in qt for t in tok(cols[c]))
    rows.append({"i": i, "right": pred == gold, "mg": mg, "chosen_in": inq(pred) if pred < len(cols) else False, "gold_in": inq(gold)})
cal = [x for x in rows if x["i"] % 2 == 0]; rep = [x for x in rows if x["i"] % 2 == 1]
grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
def stats(rs, tau, rule):
    conf = np.array([(x["mg"] >= tau) and (x["chosen_in"] if rule == "M" else True) for x in rs]); ok = np.array([x["right"] for x in rs]); n = len(rs)
    return float(conf.mean()), (float(ok[conf].mean()) if conf.any() else float("nan")), float((conf & ~ok).sum() / n)
print(f"\nreport half: chosen column's name IN the question on {np.mean([x['chosen_in'] for x in rep]):.3f} of rows (accuracy there {np.mean([x['right'] for x in rep if x['chosen_in']]):.3f}), NOT in on {np.mean([not x['chosen_in'] for x in rep]):.3f} (accuracy {np.mean([x['right'] for x in rep if not x['chosen_in']]):.3f})")
chosen = {}
for rule, name in (("margin", "margin-only (control)"), ("M", "M: margin AND chosen name in the question")):
    cw_cal = {t: stats(cal, t, rule)[2] for t in grid}; tsel = next((t for t in grid if cw_cal[t] < 0.02), None); chosen[rule] = tsel
    print(f"\n== rule {name}: tau set on the calibration half = {tsel} (smallest grid tau with confident-wrong < 2% there; calibration confident-wrong per tau: " + ", ".join(f"{t:.2f}:{cw_cal[t]:.3f}" for t in grid) + ")")
    print("   report half: tau | coverage | covered accuracy | confident-wrong")
    for t in grid:
        cov, acc, cw = stats(rep, t, rule); print(f"   {t:.2f} | {cov:.3f} | {acc:.3f} | {cw:.3f}" + ("   <- chosen on the calibration half" if t == tsel else "") + ("   <- the stated tau of 10c" if t == 0.05 else ""))
    if tsel is not None:
        cov, acc, cw = stats(rep, tsel, rule); print(f"   at the chosen tau {tsel}: coverage {cov:.3f}, covered accuracy {acc:.3f}, confident-wrong {cw:.3f} -> criterion (< 2%) {'MET' if cw < 0.02 else 'NOT met'} at coverage {cov:.3f}")
    else: print("   no grid tau reaches confident-wrong < 2% on the calibration half")
# the 4c split under both rules at tau 0.05: can the 0.169 be turned into abstentions?
print("\n== the 4c split at tau 0.05 (report half): gold column name in the question | n | confident-wrong margin-only | confident-wrong rule M | coverage margin-only | coverage rule M")
for flag in (True, False):
    b = [x for x in rep if x["gold_in"] == flag]; s1 = stats(b, 0.05, "margin"); s2 = stats(b, 0.05, "M")
    print(f"   {'IN' if flag else 'NOT in':6s} | {len(b):5d} | {s1[2]:.3f} | {s2[2]:.3f} | {s1[0]:.3f} | {s2[0]:.3f}")
print("\nchecklist: [x] re-scoring only, saved frames [x] feature available at verification time (chosen column, not gold) [x] tau set on the calibration half, reported on the other half (25) [x] control rule on the same rows [x] acc-vs-coverage curve and confident-wrong (18) [x] criterion stated before the run")
