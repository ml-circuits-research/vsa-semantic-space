"""
round5_table.py — round 5 (judgment d8f8a98 Q5), from the stored per-sentence presence ratios (presence_ratios_seed<k>.json), CPU only.
Two readouts of the recipient's presence, side by side, on the REPORT rows (present: the reported halves of the held-out formulations;
absent: the held-out sentences without a recipient):
  (i)  the pooled-midpoint threshold — the product today (training sample + calibration half of the held-out formulations, round 4 F);
  (ii) a threshold FITTED on the calibration half of the held-out formulations alone: the threshold that maximises the balanced present /
       absent accuracy on the calibration rows (present: the calibration halves; absent: the training sample's recipient-less sentences — the
       held-out formulations have none; candidates = the midpoints between consecutive distinct calibration ratios; ties: the lowest).
The overlap itself, on the report rows: the share of present-class ratios below the absent class's 90th percentile, and the best achievable
balanced accuracy (any threshold, chosen ON the report rows — a diagnostic of separability, never a readout).
DECISION RULE (judgment d8f8a98 Q5, written before the run): if the present-class ratios on unseen formulations overlap the absent class (no
threshold separates them above 0.9 present / 0.9 absent on the report half), the limit is the READER's and stays recorded; if the fitted
threshold separates them, the limit is the midpoint RULE's and the tool's calibration becomes "threshold fitted on calibration rows that include
held-out formulations, by a fit, not the pooled midpoint".
Run: python3 tool/tests/round5_table.py --root tool/runs/round5_<commit>
"""
import argparse, glob, json, os
import numpy as np
TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = "reference_G_presence_ratios"; UNSEEN = "heldout_formulations_unseen_transitions"


def fit(present, absent):
    v = np.unique(np.concatenate([present, absent]))
    cands = (v[:-1] + v[1:]) / 2 if len(v) > 1 else v
    best = max(cands, key=lambda t: ((present >= t).mean() + (absent < t).mean()) / 2 - 1e-12 * t)
    return float(best)


def scores(t, present, absent):
    p, a = float((present >= t).mean()), float((absent < t).mean()); return p, a, (p + a) / 2


def separable(present, absent):
    """is there a threshold with >= 0.9 present AND >= 0.9 absent correct on these rows? (and the best balanced accuracy)"""
    v = np.unique(np.concatenate([present, absent])); c = np.concatenate([[v[0] - 1], (v[:-1] + v[1:]) / 2, [v[-1] + 1]])
    res = [scores(t, present, absent) for t in c]
    return any(p >= 0.9 and a >= 0.9 for p, a, _ in res), max(b for _, _, b in res)


def ms(xs): return f"{np.mean(xs):.3f}±{np.std(xs):.3f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--role", default="recipient"); a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.root, RUN, "presence_ratios_seed*.json")))
    rows = {k: [] for k in ("thr_pool", "thr_fit", "pool_p", "pool_a", "pool_b", "fit_p", "fit_a", "fit_b", "overlap", "best_b", "sep", "n_p", "n_a", "fit_p_all", "pool_p_all")}
    per = []
    for fp in files:
        d = json.load(open(fp)); r = d["roles"][a.role]
        P_cal, A_cal = np.array(r["present_calibration"]), np.array(r["absent_calibration"])
        P_rep, A_rep = np.array(r["present_report"].get(UNSEEN, [])), np.array(r["absent_report"])
        P_all = np.concatenate([np.array(v) for v in r["present_report"].values()]) if r["present_report"] else np.array([])
        tp = d["thresholds"]["product (pooled midpoint)"][a.role]; tf = fit(P_cal, A_cal)
        sp, sf = scores(tp, P_rep, A_rep), scores(tf, P_rep, A_rep); sep, best = separable(P_rep, A_rep)
        ov = float((P_rep < np.percentile(A_rep, 90)).mean())
        for k, v in (("thr_pool", tp), ("thr_fit", tf), ("pool_p", sp[0]), ("pool_a", sp[1]), ("pool_b", sp[2]), ("fit_p", sf[0]), ("fit_a", sf[1]),
                     ("fit_b", sf[2]), ("overlap", ov), ("best_b", best), ("sep", float(sep)), ("n_p", len(P_rep)), ("n_a", len(A_rep)),
                     ("fit_p_all", float((P_all >= tf).mean()) if len(P_all) else float("nan")), ("pool_p_all", float((P_all >= tp).mean()) if len(P_all) else float("nan"))):
            rows[k].append(v)
        per.append((d["seed"], tp, tf, sp, sf, ov, best, sep, float(np.median(P_rep)) if len(P_rep) else float("nan"), float(np.percentile(A_rep, 90)), float(np.median(P_cal))))
    out = [f"Round 5, reference world (to / by / from given open), role {a.role}; report rows: present = the reported half of {UNSEEN} "
           f"(n {rows['n_p'][0] if rows['n_p'] else '—'}), absent = the held-out sentences without a {a.role} (n {rows['n_a'][0] if rows['n_a'] else '—'}); {len(files)} seeds",
           "The absent class: the held-out formulations contain no recipient-less sentence, so the absent class can only come from the seen recipient-less forms. The fitted threshold therefore uses the calibration half (present) against the training recipient-less rows (absent). The report side uses disjoint rows: the reported halves against the held-out sentences' recipient-less rows.", "",
           "| readout | threshold set by | threshold | present read present | absent read absent | balanced | present read present, all held-out formulations |",
           "|---|---|---|---|---|---|---|"]
    if files:
        out.append(f"| (i) product today: pooled midpoint | training sample + calibration half of the held-out formulations (round 4 F) | {ms(rows['thr_pool'])} | "
                   f"{ms(rows['pool_p'])} | {ms(rows['pool_a'])} | {ms(rows['pool_b'])} | {ms(rows['pool_p_all'])} |")
        out.append(f"| (ii) fitted | max balanced accuracy on the calibration half alone (+ training recipient-less as absent) | {ms(rows['thr_fit'])} | "
                   f"{ms(rows['fit_p'])} | {ms(rows['fit_a'])} | {ms(rows['fit_b'])} | {ms(rows['fit_p_all'])} |")
        out.append("| round 4 F, for reference (tool/runs/round4_39063ed) | product readout: role recipient accuracy on the same split (not the presence-only number) | | 0.630±0.037 | | | |")
        out += ["", "| overlap (report rows) | value |", "|---|---|",
                f"| share of present-class ratios below the absent class's 90th percentile | {ms(rows['overlap'])} |",
                f"| best achievable balanced accuracy (any threshold chosen ON the report rows: separability diagnostic, not a readout) | {ms(rows['best_b'])} |",
                f"| seeds where some threshold gives >= 0.9 present AND >= 0.9 absent on the report rows | {int(sum(rows['sep']))} / {len(files)} |",
                "", "| seed | pooled thr | fitted thr | (i) present / absent | (ii) present / absent | overlap | best balanced | separable | median present (report) | absent p90 (report) | median present (calibration) |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s, tp, tf, sp, sf, ov, best, sep, mp, ap90, mpc in per:
            out.append(f"| {s} | {tp:.3f} | {tf:.3f} | {sp[0]:.3f} / {sp[1]:.3f} | {sf[0]:.3f} / {sf[1]:.3f} | {ov:.3f} | {best:.3f} | {'yes' if sep else 'no'} | {mp:.3f} | {ap90:.3f} | {mpc:.3f} |")
        fit_sep = all(p >= 0.9 and q >= 0.9 for p, q in zip(rows["fit_p"], rows["fit_a"]))
        any_sep = sum(rows["sep"]) == len(files)
        verdict = ("the FITTED threshold separates them (>= 0.9 / >= 0.9 on the report rows in every seed): the limit is the midpoint RULE's; the "
                   "tool's calibration becomes a threshold FITTED on calibration rows that include held-out formulations" if fit_sep else
                   ("no threshold separates them in every seed (the classes overlap): the limit is the READER's and stays recorded" if not any_sep else
                    "a threshold chosen on the report rows would separate them, but the threshold fitted on the calibration half does not reach "
                    "0.9 / 0.9 in every seed: by the rule as written (the FITTED threshold separates) the limit is not shown to be the rule's; "
                    "stated as found"))
        out += ["", "DECISION RULE (judgment d8f8a98 Q5, written before the run): overlap -> the READER's limit, recorded; the fitted threshold "
                "separates -> the midpoint RULE's limit, the calibration becomes a fit.", f"OUTCOME: {verdict}."]
    else: out.append("(no presence_ratios_seed*.json under " + os.path.join(a.root, RUN) + ")")
    text = "\n".join(out); print(text)
    os.makedirs(a.root, exist_ok=True)
    with open(os.path.join(a.root, "round5_table.md"), "w") as fh: fh.write(text + "\n")


if __name__ == "__main__":
    main()
