"""
round3_table.py — the two single changes of round 3 next to their references.
  D: reference, gates given (as round2 A), the unseen-transitions split with presence DECIDED (the tool's readout) and NOT decided (every role
     read, as v16), next to round2 A (0.428) and RESULTS 10b-quinquies (0.627±0.024).
  E: commands, depth 2 unseen, per seed: the depth from the level-opener KIND count (primary in E) vs the open-gate count vs the depth given.
Run: python3 tool/tests/round3_table.py --root tool/runs/round3_<commit>
"""
import argparse, json, os
TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(TOOL, "runs", "round2_815e5f9")


def load(p): return json.load(open(p)) if os.path.exists(p) else None
def ms(m, split, k, cs="catalog candidates"):
    v = (m or {}).get("mean_over_seeds", {}).get(split, {}).get(cs, {}).get(k)
    return f"{v['mean']:.3f}±{v['std']:.3f}" if v else "—"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args(); out = []
    A = load(os.path.join(R2, "reference_A_gates_given", "metrics.json")); D = load(os.path.join(a.root, "reference_D_presence_not_decided", "metrics.json"))
    sp = "heldout_formulations_unseen_transitions"
    out += ["Reference, unseen transitions (5 seeds; candidates = the role's catalog, 18 per person / object role)",
            "| run | readout | heads FULL (roles) | FULL | agent | object | recipient | occupancy: known filler |", "|---|---|---|---|---|---|---|---|"]
    out.append(f"| round2 A (gates given) | presence decided | {ms(A, sp, 'heads FULL (model)')} | {ms(A, sp, 'FULL')} | {ms(A, sp, 'role agent')} | "
               f"{ms(A, sp, 'role object')} | {ms(A, sp, 'role recipient')} | {ms(A, sp, 'occupancy: known filler')} |")
    out.append(f"| D (= A retrained) | presence decided | {ms(D, sp, 'heads FULL (model)')} | {ms(D, sp, 'FULL')} | {ms(D, sp, 'role agent')} | "
               f"{ms(D, sp, 'role object')} | {ms(D, sp, 'role recipient')} | {ms(D, sp, 'occupancy: known filler')} |")
    out.append(f"| D | presence NOT decided (every role read, as v16) | {ms(D, sp, 'heads FULL (model) [presence not decided]')} | "
               f"{ms(D, sp, 'FULL [presence not decided]')} | {ms(D, sp, 'role agent [presence not decided]')} | {ms(D, sp, 'role object [presence not decided]')} | "
               f"{ms(D, sp, 'role recipient [presence not decided]')} | — |")
    out.append("| RESULTS 10b-quinquies (v16, to/by/from given open) | C always read | 0.627±0.024 | | | | | |")
    E = load(os.path.join(a.root, "commands_E_depth_opener_kinds", "metrics.json")); G = load(os.path.join(R2, "commands_depth_from_gates", "metrics.json"))
    out += ["", "Commands, depth 2 UNSEEN, per seed (gates learned, nothing given open)",
            "| run | seed | gate 'that' | depth right: opener kinds | depth right: open gates | FULL [kind count] | FULL [gate count] | FULL [depth given] |",
            "|---|---|---|---|---|---|---|---|"]
    for name, m, prim in (("round2 commands_depth_from_gates", G, "gates"), ("E (primary = opener kinds)", E, "kinds")):
        if m is None: out.append(f"| {name} | (no metrics) | | | | | | |"); continue
        for i, ps in enumerate(m["per_seed"]):
            acc = ps["heldout_depth2"]["catalog candidates"]["acc"]; gate = m["training"][i]["gates"].get("that", float("nan"))
            fk = acc["FULL"] if prim == "kinds" else acc.get("FULL [kind-count estimate]")
            fg = acc["FULL"] if prim == "gates" else acc.get("FULL [gate-count estimate]")
            f = lambda x: "—" if x is None else f"{x:.3f}"
            out.append(f"| {name} | {m['seeds'][i]} | {gate:.2f} | {f(acc.get('depth estimate: opener-kind count'))} | {f(acc.get('depth estimate: open-gate count', acc.get('depth estimate: level gates (primary)')))} | "
                       f"{f(fk)} | {f(fg)} | {f(acc.get('FULL [depth given]'))} |")
        prim_k = "FULL"
        out.append(f"| {name} | mean | | {ms(m, 'heldout_depth2', 'depth estimate: opener-kind count')} | {ms(m, 'heldout_depth2', 'depth estimate: open-gate count') if 'depth estimate: open-gate count' in str(m.get('mean_over_seeds', {}).get('heldout_depth2', {})) else ms(m, 'heldout_depth2', 'depth estimate: level gates (primary)')} | "
                   f"{ms(m, 'heldout_depth2', prim_k if prim == 'kinds' else 'FULL [kind-count estimate]')} | {ms(m, 'heldout_depth2', prim_k if prim == 'gates' else 'FULL [gate-count estimate]')} | "
                   f"{ms(m, 'heldout_depth2', 'FULL [depth given]')} |")
    out.append("RESULTS 10d: depth 2 FULL 0.854±0.031 (depth given).")
    text = "\n".join(out); print(text)
    with open(os.path.join(a.root, "round3_table.md"), "w") as fh: fh.write(text + "\n")


if __name__ == "__main__":
    main()
