"""
round5_margins.py — the re-scoring of judgment 752853c §2 from round 5's stored ratios (CPU, no retraining).
Under the FITTED threshold (per seed: argmax balanced accuracy on the calibration rows, as tests/round5_table.py), of the errors on the report
rows — false "present" (absent rows read present) and missed "present" (present rows read absent) — what share lies within the abstention
band of the role's margin: |ratio - threshold| <= tau (0.05, the stated tau of model/config.json) or a role identity margin < tau?
Stored in round 5: the presence ratios only. The per-sentence IDENTITY margins are NOT in presence_ratios_seed<k>.json, so only the
presence-margin part is computed here (a lower bound on the share in the band); round 6 stores both (and the structure score, so that the
presence margin can also be read in cosine units).
Also printed: the band width that WOULD cover half / most (80%) of the errors, as a diagnostic of the scale (the ratio is a presence score
divided by the structure score: present ~1.1, absent ~0.1-0.2 on training rows).
Run: python3 tool/tests/round5_margins.py [--root tool/runs/round5_1ed4306] [--tau 0.05]
"""
import argparse, glob, json, os
import numpy as np
from round5_table import fit, RUN, UNSEEN
TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(TOOL, "runs", "round5_1ed4306")); ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--role", default="recipient"); a = ap.parse_args()
    out = [f"Round 5 re-scoring (judgment 752853c §2): the errors of the FITTED threshold on the report rows, role {a.role}, and the share within "
           f"the presence band |ratio - threshold| <= tau = {a.tau}. The identity margins were not stored in round 5: presence part only "
           "(a LOWER BOUND on the share in the abstention band; round 6 stores both).", "",
           "| seed | fitted thr | false present (n / of absent) | in band | missed present (n / of present) | in band | all errors in band | band covering 50% / 80% of errors |",
           "|---|---|---|---|---|---|---|---|"]
    tot_err, tot_in = 0, 0; shares = []; w50s, w80s = [], []
    for fp in sorted(glob.glob(os.path.join(a.root, RUN, "presence_ratios_seed*.json"))):
        d = json.load(open(fp)); r = d["roles"][a.role]
        tf = fit(np.array(r["present_calibration"]), np.array(r["absent_calibration"]))
        P, A = np.array(r["present_report"].get(UNSEEN, [])), np.array(r["absent_report"])
        fp_ = A[A >= tf]; fn_ = P[P < tf]; dist = np.abs(np.concatenate([fp_, fn_]) - tf)
        inb = lambda x: int((np.abs(x - tf) <= a.tau).sum())
        e = len(dist); i = inb(fp_) + inb(fn_); tot_err += e; tot_in += i; shares.append(i / e if e else float("nan"))
        w50, w80 = (float(np.percentile(dist, 50)), float(np.percentile(dist, 80))) if e else (float("nan"), float("nan")); w50s.append(w50); w80s.append(w80)
        out.append(f"| {d['seed']} | {tf:.3f} | {len(fp_)} / {len(A)} | {inb(fp_)} | {len(fn_)} / {len(P)} | {inb(fn_)} | {i} / {e} = {i / e if e else float('nan'):.3f} | {w50:.3f} / {w80:.3f} |")
    out += ["", f"POOLED over seeds: {tot_in} / {tot_err} errors within |ratio - threshold| <= {a.tau} = {tot_in / max(1, tot_err):.3f} "
            f"(per-seed mean {np.nanmean(shares):.3f}); the band that would cover 50% / 80% of the errors: {np.nanmean(w50s):.3f} / {np.nanmean(w80s):.3f} "
            "(ratio units).",
            "Expectation (judgment 752853c §2): most errors lie in the abstention band. The identity-margin part is missing from round 5's store; "
            "the presence part alone is the number above."]
    text = "\n".join(out); print(text)
    with open(os.path.join(a.root, "round5_margins.md"), "w") as fh: fh.write(text + "\n")


if __name__ == "__main__":
    main()
