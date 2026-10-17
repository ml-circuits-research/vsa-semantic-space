"""
round6_table.py — round 6 (judgment 752853c §2): the FITTED threshold as the calibration rule + the presence margin on the wire.
Product numbers first (presence decided with the fitted rule), the pooled midpoint of the same calibration rows beside it as the comparison.
Per role (roles with a presence threshold): present read present / absent read absent / balanced, and the share of "present" readings that
abstain (the role's margin = min(identity, presence) < tau). From the stored ratios (now with the identity margins): of the product's errors,
the share inside the abstention band (|ratio - threshold| <= tau OR identity margin < tau) — the re-scoring of §2, completed.
Run: python3 tool/tests/round6_table.py --root tool/runs/round6_<commit>
"""
import argparse, glob, json, os
import numpy as np
TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = "reference_H_fitted_rule"; SPLITS = ("heldout_formulations_unseen_transitions", "heldout_sentences", "heldout_depth3")


def load(p): return json.load(open(p)) if os.path.exists(p) else None
def ms(m, split, k):
    v = (m or {}).get("mean_over_seeds", {}).get(split, {}).get("catalog candidates", {}).get(k)
    return f"{v['mean']:.3f}±{v['std']:.3f}" if v else "—"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--tau", type=float, default=0.05); a = ap.parse_args()
    m = load(os.path.join(a.root, RUN, "metrics.json")); out = []
    roles = sorted({k.split(":")[0][len("presence "):] for s in (m or {}).get("mean_over_seeds", {}).values()
                    for k in s.get("catalog candidates", {}) if k.startswith("presence ")})
    out += ["Round 6, reference world (to / by / from given open); PRODUCT first: presence decided with the FITTED thresholds (argmax balanced "
            "accuracy on the calibration rows: training sample + calibration half of the held-out formulations); the pooled midpoint of the same "
            "rows beside it as the comparison (one more round). 5 seeds.", "",
            "| split | role | readout | present read present | absent read absent | 'present' readings that abstain | false 'present' that abstain | true 'present' that abstain |",
            "|---|---|---|---|---|---|---|---|"]
    for sp in SPLITS:
        for r in roles:
            out.append(f"| {sp} | {r} | product: fitted | {ms(m, sp, f'presence {r}: present read present')} | {ms(m, sp, f'presence {r}: absent read absent')} | "
                       f"{ms(m, sp, f'presence {r}: share of ' + chr(39) + 'present' + chr(39) + ' readings that abstain')} | "
                       f"{ms(m, sp, f'presence {r}: share of FALSE ' + chr(39) + 'present' + chr(39) + ' readings that abstain')} | "
                       f"{ms(m, sp, f'presence {r}: share of true ' + chr(39) + 'present' + chr(39) + ' readings that abstain')} |")
            out.append(f"| {sp} | {r} | comparison: pooled midpoint | {ms(m, sp, f'presence {r}: present read present [pooled midpoint (comparison)]')} | "
                       f"{ms(m, sp, f'presence {r}: absent read absent [pooled midpoint (comparison)]')} | | | |")
    sp = SPLITS[0]
    out += ["", f"{sp}: heads FULL (roles) by readout", "| readout | heads FULL | FULL |", "|---|---|---|",
            f"| product: presence decided, fitted thresholds | {ms(m, sp, 'heads FULL (model)')} | {ms(m, sp, 'FULL')} |",
            f"| comparison: presence decided, pooled midpoint | {ms(m, sp, 'heads FULL (model) [pooled midpoint (comparison)]')} | {ms(m, sp, 'FULL [pooled midpoint (comparison)]')} |",
            f"| v16 protocol: every role read, not available to the product | {ms(m, sp, 'heads FULL (model) [presence not decided]')} | {ms(m, sp, 'FULL [presence not decided]')} |",
            "| round 4 F product (tool/runs/round4_39063ed) | 0.418±0.061 | |", "| RESULTS 10b-quinquies (v16 protocol) | 0.627±0.024 | |"]
    # the band, from the stored ratios (identity margins now stored)
    out += ["", f"The product's errors inside the abstention band (|ratio - threshold| <= {a.tau} OR identity margin < {a.tau}), from the stored ratios, role by role:",
            "| seed | role | threshold (fitted) | false present: n / in presence band / in band (either) | missed present (unseen): n / in presence band / in band (either) | all errors in band |",
            "|---|---|---|---|---|---|"]
    tot = {}
    for fp in sorted(glob.glob(os.path.join(a.root, RUN, "presence_ratios_seed*.json"))):
        d = json.load(open(fp))
        for r, x in d["roles"].items():
            t = d["thresholds"]["product"][r]
            A, Ai = np.array(x["absent_report"]), np.array(x["absent_report_identity_margin"])
            P, Pi = np.array(x["present_report"].get(sp, [])), np.array(x["present_report_identity_margin"].get(sp, []))
            fpm = A >= t; fnm = P < t
            pb_f = np.abs(A[fpm] - t) <= a.tau; eb_f = pb_f | (Ai[fpm] < a.tau)
            pb_n = np.abs(P[fnm] - t) <= a.tau; eb_n = pb_n | (Pi[fnm] < a.tau)
            n_e = int(fpm.sum() + fnm.sum()); n_in = int(eb_f.sum() + eb_n.sum()); tt = tot.setdefault(r, [0, 0]); tt[0] += n_in; tt[1] += n_e
            out.append(f"| {d['seed']} | {r} | {t:.3f} | {int(fpm.sum())} / {int(pb_f.sum())} / {int(eb_f.sum())} | {int(fnm.sum())} / {int(pb_n.sum())} / {int(eb_n.sum())} | "
                       f"{n_in} / {n_e} = {n_in / n_e if n_e else float('nan'):.3f} |")
    for r, (i, e) in tot.items(): out.append(f"POOLED {r}: {i} / {e} = {i / max(1, e):.3f} of the product's errors inside the abstention band.")
    out.append("Expectation (judgment 752853c §2): most. If not, the asymmetry option (a legend field 'prefer misses over false alarms for role X') "
               "is reported, not built.")
    text = "\n".join(out); print(text)
    with open(os.path.join(a.root, "round6_table.md"), "w") as fh: fh.write(text + "\n")


if __name__ == "__main__":
    main()
