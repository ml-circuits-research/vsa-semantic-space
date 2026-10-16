"""C4 (judgment a8a8fab E5): re-score the saved verifier frames (logs/.ckpt_v17_reader_frames.pt, logs/.ckpt_v17_lex_frames.pt, written by
v17_verifier.py) with the abstention threshold tau SET on the calibration half of dev (even rows: the smallest tau on a grid whose false-alarm
rate on gold frames is < 0.05) and the WHOLE curve REPORTED on the other half (odd rows): at every tau on the grid, false alarm on gold,
detection / unverifiable / confident-wrong per role error type, plus the detection lost against tau = 0.05 (10c-bis). No retraining; the
reader is not touched. Rule 25: the threshold never comes from the rows it is reported on."""
import numpy as np, torch, collections, v11_wikisql as v11, v17_verifier as v17
dev, dtabs = v11.load("dev"); saved = torch.load("logs/.ckpt_v17_reader_frames.pt", weights_only=False); frames = saved["frames"]; pres_thr = saved["pres_thr"]
lex = torch.load("logs/.ckpt_v17_lex_frames.pt", weights_only=False); dev = dev[:len(frames)]
print(f"rescore_v17_c4: {len(frames)} dev frames; condition presence threshold {pres_thr:.3f} (set on the even half by v17); tau grid set on the EVEN half, reported on the ODD half")
kinds = ["drop_cond", "swap_sel_cond", "change_sel", "flip_agg", "add_cond", "change_value"]; role_kinds = ["swap_sel_cond", "change_sel", "flip_agg"]
def verdict(frame, prop, tau, is_reader):
    if prop["sel"] != frame["sel"][0]: return "error" if frame["sel"][1] >= tau else "unverifiable"
    if prop["agg"] != frame["agg"][0]: return "error" if frame["agg"][1] >= tau else "unverifiable"
    if is_reader:
        read_cols = [c for (c, top, mg) in frame["conds"] if top >= pres_thr]; prop_cols = [c[0] for c in prop["conds"]]
        if sorted(read_cols) != sorted(prop_cols):
            mg = min([mg for (c, top, mg) in frame["conds"]] + [1.0]); return "error" if mg >= tau else "unverifiable"
    return "ok"
# the injected errors are regenerated with the same rng as v17 (seed 0, same row order, every row consumes the generator)
rng = np.random.default_rng(0); props = []
for i, r in enumerate(dev):
    cols = dtabs[r["table_id"]]; row = {}
    for kind in kinds + ["no_error"]:
        row[kind] = {"sel": r["sql"]["sel"], "agg": r["sql"]["agg"], "conds": [list(c) for c in r["sql"]["conds"]]} if kind == "no_error" else v17.propose_with_error(r["sql"], len(cols), rng, kind)
    props.append(row)
grid = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
def table(half, tau, is_reader=True):
    fr = frames if is_reader else lex; st = {k: collections.Counter() for k in kinds + ["no_error"]}
    for i in range(half, len(dev), 2):
        for kind in kinds + ["no_error"]:
            p = props[i][kind]
            if p is None: continue
            st[kind][verdict(fr[i], p, tau, is_reader)] += 1
    out = {}
    for k in st:
        n = sum(st[k].values()); out[k] = (st[k]["error"] / max(n, 1), st[k]["unverifiable"] / max(n, 1), st[k]["ok"] / max(n, 1), n)
    return out
print("\n== calibration half (EVEN rows): false alarm on gold per tau — the first tau with false alarm < 0.05 is chosen")
chosen = None
for tau in grid:
    t = table(0, tau); fa = t["no_error"][0]; print(f"  tau {tau:.2f}: false alarm {fa:.3f}; role errors detected {np.mean([t[k][0] for k in role_kinds]):.3f}")
    if chosen is None and fa < 0.05: chosen = tau
print(f"  chosen tau (even half): {chosen}")
print("\n== REPORT half (ODD rows): tau | gold: false alarm / unverifiable / confirmed | role errors (swap, change_sel, flip_agg): detected / unverifiable / confident-wrong | condition errors (drop, add): detected / unverifiable / confident-wrong | detection lost vs tau 0.05 (role)")
base = table(1, 0.05)
for tau in grid:
    t = table(1, tau); g = t["no_error"]; ro = np.mean([[t[k][j] for j in range(3)] for k in role_kinds], 0); co = np.mean([[t[k][j] for j in range(3)] for k in ("drop_cond", "add_cond")], 0)
    mark = " <- chosen on the even half" if tau == chosen else ""
    print(f"  {tau:.2f} | {g[0]:.3f} / {g[1]:.3f} / {g[2]:.3f} | {ro[0]:.3f} / {ro[1]:.3f} / {ro[2]:.3f} | {co[0]:.3f} / {co[1]:.3f} / {co[2]:.3f} | {np.mean([base[k][0] for k in role_kinds]) - ro[0]:+.3f}{mark}")
t = table(1, chosen); print("\n== per error type at the chosen tau (odd half): type | n | detected / unverifiable / confident-wrong")
for k in kinds: print(f"  {k:14s} | {t[k][3]:5d} | {t[k][0]:.3f} / {t[k][1]:.3f} / {t[k][2]:.3f}")
print(f"  no_error (gold) | {t['no_error'][3]:5d} | false alarm {t['no_error'][0]:.3f} / unverifiable {t['no_error'][1]:.3f} / confirmed {t['no_error'][2]:.3f}")
print("  stated: change_value is invisible to this reader; conditions are read by column set at the saved presence threshold; the lexical baseline has no condition model.")
print("\nchecklist: [x] tau set on the calibration half, curve reported on the other half (25) [x] detection lost at every tau, not one point [x] false alarm on gold at every tau [x] no retraining, saved frames only [x] invisible error types stated")
