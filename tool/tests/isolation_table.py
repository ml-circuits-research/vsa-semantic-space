"""
isolation_table.py — the reference gap isolated ONE change per run (judgment e9274a9 C2, rule 28), and the commands depth-2 re-run with the
depth from the level gates (Q6).

Rows (reference world, 5 seeds each, same sizes and regime as tool/runs/p0_gpu_5seed_c843f53/reference):
  P0      tool/runs/p0_gpu_5seed_c843f53/reference: legend v1 (gates all learned), the tool's star identities, candidates = every catalog word of
          the role's kinds (18 persons / 18 objects: 12 seen + 6 held out), depth estimated by the fit
  A       + gates GIVEN open for to / by / from (legend v2 reader.gates_given); nothing else changes in the TRAINING (the v2 legend also names
          the level openers: that changes only the depth ESTIMATE, so A is read on estimator-free columns too: depth given)
  B       = A's models re-scored with v16's candidate sets (seen splits: the 12 seen words of the role's kind; new identities: the 6 held-out
          words) — a scoring change, no retraining: the `experiment candidates` column of A's metrics.json, every seed
  C       = B + v16's per-seed identity draw (np.random.default_rng(300 + seed) over the v16 vocabulary) instead of the tool's star function
          (sop train --identities table:identity_vocab_v16.json), read with B's candidate sets
Run: python3 tool/tests/isolation_table.py --root tool/runs/round2_<commit>
"""
import argparse, json, os
TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P0 = os.path.join(TOOL, "runs", "p0_gpu_5seed_c843f53")
AXES = [("seen formulations (roles)", "heldout_sentences", "heads FULL (model)", "1.000±0.000"),
        ("new identities (roles)", "new_identities", "heads FULL (model)", "1.000"),
        ("unseen transitions (roles)", "heldout_formulations_unseen_transitions", "heads FULL (model)", "0.638±0.015"),
        ("depth 3 UNSEEN (roles)", "heldout_depth3", "heads FULL (model)", "0.813±0.030 learned gates (10b); 0.968 to/by/from given open (10b-quinquies)"),
        ("depth 3 UNSEEN FULL, depth given", "heldout_depth3", "FULL [depth given]", "—")]


def cell(m, split, metric, cs):
    v = m.get("mean_over_seeds", {}).get(split, {}).get(cs, {}).get(metric)
    if v is None: return "—"
    return f"{v['mean']:.3f}±{v['std']:.3f}"


def sizes(m, cs):
    ch = m.get("chance", {})
    if cs == "catalog candidates": return " / ".join(f"{r} {round(1 / c)}" for r, c in ch.items() if c and r != "predicate")
    return "seen words of the role's kind (12) / held-out words for new identities (6)"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    load = lambda p: json.load(open(p)) if os.path.exists(p) else None
    runs = [("P0 (c843f53)", load(os.path.join(P0, "reference", "metrics.json")), "catalog candidates", "baseline"),
            ("P0 + v16 candidate sets", load(os.path.join(P0, "reference", "metrics.json")), "experiment candidates", "one change from P0: candidates (re-scoring, free)"),
            ("A: + gates given (to, by, from open)", load(os.path.join(a.root, "reference_A_gates_given", "metrics.json")), "catalog candidates", "one change: gates"),
            ("B: A + v16 candidate sets", load(os.path.join(a.root, "reference_A_gates_given", "metrics.json")), "experiment candidates", "one change: candidates (re-scoring of A)"),
            ("C: B + v16 identity draw", load(os.path.join(a.root, "reference_C_v16_identities", "metrics.json")), "experiment candidates", "one change: identities")]
    out = ["| run | change | candidates (per role) | " + " | ".join(x[0] for x in AXES) + " |", "|---|---|---|" + "---|" * len(AXES)]
    for name, m, cs, change in runs:
        if m is None: out.append(f"| {name} | {change} | (no metrics) |" + " |" * len(AXES)); continue
        out.append(f"| {name} | {change} | {sizes(m, cs)} | " + " | ".join(cell(m, s, k, cs) for _, s, k, _ in AXES) + " |")
    out.append("| RESULTS (5 seeds) | | v16: 12 seen / 6 new | " + " | ".join(r for *_, r in AXES) + " |")
    bdir = os.path.join(a.root, "reference_B_role_catalog"); os.makedirs(bdir, exist_ok=True)
    mA = runs[1][1]
    if mA is not None:
        json.dump({"source": "re-scoring of reference_A_gates_given (no retraining): its 'experiment candidates' column, every seed",
                   "mean_over_seeds": {s: {"experiment candidates": v["experiment candidates"]} for s, v in mA["mean_over_seeds"].items() if "experiment candidates" in v},
                   "chance": mA.get("chance"), "seeds": mA.get("seeds")}, open(os.path.join(bdir, "metrics.json"), "w"), indent=1)
    cm = load(os.path.join(a.root, "commands_depth_from_gates", "metrics.json"))
    out += ["", "Commands world, depth 2 UNSEEN (5 seeds): the depth estimated from the open level gates vs the fit vs given",
            "| run | FULL (level gates, primary) | FULL [fit estimate] | FULL [depth given] | depth estimate: level gates | depth estimate: fit | RESULTS 10d |", "|---|---|---|---|---|---|---|"]
    for name, m in (("P0 (c843f53, fit was the only estimator)", load(os.path.join(P0, "commands", "metrics.json"))), ("re-run, legend v2 (level openers)", cm)):
        if m is None: out.append(f"| {name} | (no metrics) | | | | | |"); continue
        c = lambda k: cell(m, "heldout_depth2", k, "catalog candidates")
        out.append(f"| {name} | {c('FULL')} | {c('FULL [fit estimate]')} | {c('FULL [depth given]')} | {c('depth estimate: level gates (primary)')} | {c('depth estimate: fit (check)')} | 0.854±0.031 |")
    text = "\n".join(out); print(text)
    with open(os.path.join(a.root, "isolation_table.md"), "w") as f: f.write(text + "\n")


if __name__ == "__main__":
    main()
