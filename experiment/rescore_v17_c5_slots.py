"""C5, last re-scoring (judgment 45ccd1d C): PER-SLOT presence thresholds (slot 1, slot 2, slot 3+) set on the calibration half of dev (even rows),
every number on the odd half; saved frames of the C5 reader (logs/.ckpt_v17_reader_frames_occ.pt, marker cosine per slot); no retraining.
Criterion, written before the run: present read as present >= 0.9 on the SECOND AND LATER slots AND the false alarm on gold not above 10c-sexies's
0.152 (verdict with the column-margin statistic, as in 10c-sexies). Met -> C5 met by calibration. Not met -> closed as "NOT met, bounded by the
presence reading of the later slots", with the diagnostic that decides between slot index and capacity: the present rate by SLOT INDEX and by the
NUMBER OF CONDITIONS in the question (if the drop follows the number of conditions rather than the slot, it is superposition — capacity)."""
import sys, collections, numpy as np, torch, v11_wikisql as v11
from v17_verifier import propose_with_error

FR = sys.argv[1] if len(sys.argv) > 1 else "logs/.ckpt_v17_reader_frames_occ.pt"; tau = 0.05
dev, dtabs = v11.load("dev"); saved = torch.load(FR, weights_only=False); frames = saved["frames"]; thr_one = saved["pres_thr"]
slot_of = lambda k: min(k, 2)                                     # slot index 0, 1, 2+ -> three thresholds
cal = {s: {"present": [], "absent": []} for s in (0, 1, 2)}
for i, r in enumerate(dev):
    if i % 2 == 0:
        for k, (c, p, mg) in enumerate(frames[i]["conds"]): cal[slot_of(k)]["present" if k < len(r["sql"]["conds"]) else "absent"].append(p)
thr = {s: 0.5 * (np.mean(cal[s]["present"]) + np.mean(cal[s]["absent"])) for s in (0, 1, 2)}
print(f"rescore_v17_c5_slots: frames {FR}; single threshold of 10c-sexies {thr_one:.3f}; PER-SLOT thresholds (calibration half; midpoint of present / absent): " + "; ".join(f"slot {s+1}{'+' if s == 2 else ''}: {thr[s]:.3f} ({np.mean(cal[s]['present']):.3f} / {np.mean(cal[s]['absent']):.3f}, n present {len(cal[s]['present'])})" for s in (0, 1, 2)))
# occupancy on the odd half, by slot index and by number of conditions, under both threshold schemes
occ = {sch: {"slot": collections.defaultdict(lambda: {"present": [], "absent": []}), "ncond": collections.defaultdict(lambda: {"present": [], "absent": []})} for sch in ("single", "per-slot")}
for i, r in enumerate(dev):
    if i % 2 == 1:
        n = len(r["sql"]["conds"])
        for k, (c, p, mg) in enumerate(frames[i]["conds"]):
            truth = k < n
            for sch in ("single", "per-slot"):
                read = p >= (thr_one if sch == "single" else thr[slot_of(k)]); key = "present" if truth else "absent"
                occ[sch]["slot"][slot_of(k)][key].append(read == truth)
                if truth: occ[sch]["ncond"][min(n, 3)][key].append(read == truth)
print("\n== occupancy on the report half by SLOT INDEX: scheme | slot | present read as present (n) | absent read as absent (n)")
for sch in ("single", "per-slot"):
    for s in (0, 1, 2):
        d = occ[sch]["slot"][s]; print(f"   {sch:8s} | slot {s+1}{'+' if s == 2 else ' '} | {np.mean(d['present']) if d['present'] else float('nan'):.3f} ({len(d['present'])}) | {np.mean(d['absent']) if d['absent'] else float('nan'):.3f} ({len(d['absent'])})")
    later = occ[sch]["slot"][1]["present"] + occ[sch]["slot"][2]["present"]; print(f"   {sch:8s} | slots 2+ present read as present: {np.mean(later):.3f} (n {len(later)})")
print("\n== present slots read as present by the NUMBER OF CONDITIONS in the question (diagnostic: slot index vs capacity): scheme | n conditions | rate (n slots)")
for sch in ("single", "per-slot"):
    for n in (1, 2, 3):
        d = occ[sch]["ncond"][n]["present"]; print(f"   {sch:8s} | {n}{'+' if n == 3 else ' '} | {np.mean(d) if d else float('nan'):.3f} ({len(d)})")
# verdicts with the per-slot thresholds (column-margin statistic as in 10c-sexies), same injected errors
def verdict(frame, proposed, scheme):
    if proposed["sel"] != frame["sel"][0]: return "error" if frame["sel"][1] >= tau else "unverifiable"
    if proposed["agg"] != frame["agg"][0]: return "error" if frame["agg"][1] >= tau else "unverifiable"
    read_cols = [c for k, (c, p, mg) in enumerate(frame["conds"]) if p >= (thr_one if scheme == "single" else thr[slot_of(k)])]; prop_cols = [c[0] for c in proposed["conds"]]
    if sorted(read_cols) != sorted(prop_cols): return "error" if min([mg for (c, p, mg) in frame["conds"]] + [1.0]) >= tau else "unverifiable"
    return "ok"
kinds = ["drop_cond", "swap_sel_cond", "change_sel", "flip_agg", "add_cond", "change_value"]; rng = np.random.default_rng(0)
stats = {k: {s: collections.Counter() for s in ("single", "per-slot")} for k in kinds + ["no_error"]}
for i, r in enumerate(dev):
    cols = dtabs[r["table_id"]]
    if i % 2 == 0: continue
    for kind in kinds + ["no_error"]:
        prop = {"sel": r["sql"]["sel"], "agg": r["sql"]["agg"], "conds": [list(c) for c in r["sql"]["conds"]]} if kind == "no_error" else propose_with_error(r["sql"], len(cols), rng, kind)
        if prop is None: continue
        for s in ("single", "per-slot"): stats[kind][s][verdict(frames[i], prop, s)] += 1
print("\n== verdicts (column-margin statistic, as 10c-sexies): error type | n | single threshold: detected / unverifiable / missed | per-slot thresholds: detected / unverifiable / missed")
for kind in kinds:
    c1, c2 = stats[kind]["single"], stats[kind]["per-slot"]; n = sum(c1.values())
    print(f"  {kind:14s} | {n:5d} | {c1['error']/n:.3f} / {c1['unverifiable']/n:.3f} / {c1['ok']/n:.3f} | {c2['error']/n:.3f} / {c2['unverifiable']/n:.3f} / {c2['ok']/n:.3f}")
c1, c2 = stats["no_error"]["single"], stats["no_error"]["per-slot"]; n = sum(c1.values())
print(f"  {'no_error (gold)':14s} | {n:5d} | FALSE-ALARM {c1['error']/n:.3f} / {c1['unverifiable']/n:.3f} / {c1['ok']/n:.3f} | FALSE-ALARM {c2['error']/n:.3f} / {c2['unverifiable']/n:.3f} / {c2['ok']/n:.3f}")
later = occ["per-slot"]["slot"][1]["present"] + occ["per-slot"]["slot"][2]["present"]; fa = c2["error"] / n
print(f"\n== C5 criterion (per-slot thresholds): present read as present on slots 2+ = {np.mean(later):.3f} (>= 0.9 needed), false alarm on gold = {fa:.3f} (<= 0.152 needed) -> {'MET by calibration' if np.mean(later) >= 0.9 and fa <= 0.152 else 'NOT met: closed as bounded by the presence reading of the later slots'}")
print("\nchecklist: [x] re-scoring only, saved frames [x] per-slot thresholds set on the calibration half, reported on the odd half (25) [x] criterion written before the run [x] diagnostic by slot index and by number of conditions [x] same injected errors as v17 [x] false alarm on gold")
