"""
round4_table.py — round 4 (judgment 45ccd1d D) next to round 3's D rows: the reference world (to / by / from given open), unseen transitions.
The PRODUCT number first (presence decided), each row labelled with its protocol. Round 4 reports the half of each held-out formulation split
that did not set the thresholds; on the same rows it also scores the training-rows-only thresholds (the round-3 product) and the v16 protocol.
Run: python3 tool/tests/round4_table.py --root tool/runs/round4_<commit>
"""
import argparse, json, os
TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R3 = os.path.join(TOOL, "runs", "round3_1d660ee", "reference_D_presence_not_decided")
SP = "heldout_formulations_unseen_transitions"


def load(p): return json.load(open(p)) if os.path.exists(p) else None
def ms(m, k):
    v = (m or {}).get("mean_over_seeds", {}).get(SP, {}).get("catalog candidates", {}).get(k)
    return f"{v['mean']:.3f}±{v['std']:.3f}" if v else "—"
def n(m):
    return (m or {}).get("per_seed", [{}])[0].get(SP, {}).get("catalog candidates", {}).get("n", "—")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    D = load(os.path.join(R3, "metrics.json")); F = load(os.path.join(a.root, "reference_F_calibration_with_heldout_formulations", "metrics.json"))
    cols = ("heads FULL (model)", "role agent", "role object", "role recipient")
    out = [f"Reference world (to / by / from given open), {SP}, 5 seeds; candidates = the role's catalog (18 per person / object role)", "",
           "| run | rows | readout (protocol) | thresholds set on | heads FULL | agent | object | recipient |", "|---|---|---|---|---|---|---|---|"]
    out.append(f"| round 3 D | {n(D)} (all) | product readout: presence decided | training rows | " + " | ".join(ms(D, c) for c in cols) + " |")
    out.append(f"| round 3 D | {n(D)} (all) | v16 protocol: every role read, not available to the product | — | " + " | ".join(ms(D, c + " [presence not decided]") for c in cols) + " |")
    out.append(f"| **round 4 F** | {n(F)} (reported half) | **product readout: presence decided** | training rows + the calibration half of the held-out formulations | "
               + " | ".join(ms(F, c) for c in cols) + " |")
    out.append(f"| round 4 F | {n(F)} (reported half) | presence decided | training rows only (the round-3 product, same rows) | "
               + " | ".join(ms(F, c + " [thresholds from training rows only]") for c in cols) + " |")
    out.append(f"| round 4 F | {n(F)} (reported half) | v16 protocol: every role read, not available to the product | — | "
               + " | ".join(ms(F, c + " [presence not decided]") for c in cols) + " |")
    out.append("| RESULTS 10b-quinquies (v16) | | v16 protocol (C always read) | — | 0.627±0.024 | | | |")
    cfgp = os.path.join(a.root, "reference_F_calibration_with_heldout_formulations", "config.json")
    if os.path.exists(cfgp):
        cfg = json.load(open(cfgp)); new, old = cfg["thresholds"].get("presence", {}), cfg.get("thresholds_training_rows_only", {}).get("presence", {})
        out += ["", "Presence thresholds, seed 0 (ratio to the structure score): " + "; ".join(
            f"{r}: {old.get(r)} -> {new.get(r)}" for r in sorted(set(new) | set(old))) + f" (calibration: {cfg['thresholds'].get('calibration_split')})"]
    out.append("Expectation written before the run: the recipient recovers toward the every-role-read value (0.619 -> toward 0.839) without "
               "reading every role; if not, presence calibration across formulations is a recorded limit of the reader.")
    text = "\n".join(out); print(text)
    with open(os.path.join(a.root, "round4_table.md"), "w") as fh: fh.write(text + "\n")


if __name__ == "__main__":
    main()
