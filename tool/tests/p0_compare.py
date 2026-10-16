"""
p0_compare.py — the P0 round-trip table: the package's numbers next to the RESULTS tables' (5 seeds, GPU; cited by section).
Reads <root>/<world>/metrics.json (a committed run folder, tool/runs/...) or <root>/<world>/model/metrics.json (a world folder).
Columns (author C3): the comparable axis; the tool's FULL with the structure (reporting depth, negation) ESTIMATED from the state; FULL with
the depth GIVEN, as the repository scripts. Values are mean ± std over the run's seeds.
Run: python3 tool/tests/p0_compare.py --root tool/runs/p0_cpu_1seed_60f284d   (or a GPU run folder)
"""
import argparse, json, os
TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROWS = {   # world: [(label, split, metric, RESULTS value, source)]
    "reference": [("seen formulations (A, B, C)", "heldout_sentences", "heads FULL (model)", "1.000±0.000", "10b"),
                  ("predicate, surface lemma", "heldout_sentences", "predicate (surface lemma)", "1.000", "10b"),
                  ("new identities", "new_identities", "heads FULL (model)", "1.000", "10b"),
                  ("role cues seen", "heldout_formulations_role_cues_seen", "heads FULL (model)", "1.000", "10b"),
                  ("unseen transitions", "heldout_formulations_unseen_transitions", "heads FULL (model)", "0.638±0.015", "10b"),
                  ("depth 3 UNSEEN", "heldout_depth3", "heads FULL (model)", "0.813±0.030 (10b); 0.968 after E1b (10b-quinquies)", "10b")],
    "commands": [("depth 0: action", "heldout_sentences", "role action", "1.000", "10d"),
                 ("depth 0: patient (with attribute and its negation)", "heldout_sentences", "role patient", "1.000 / 0.999 (patient / attribute)", "10d"),
                 ("depth 0 FULL", "heldout_sentences", "FULL", "0.999", "10d"),
                 ("new patients FULL", "new_identities", "FULL", "1.000", "10d"),
                 ("depth 2 UNSEEN FULL", "heldout_depth2", "FULL", "0.854±0.031", "10d"),
                 ("depth 2 UNSEEN: structure (depth + negation scope)", "heldout_depth2", "structure (depth, negation)", "scope 0.997±0.003 (depth given)", "10d")],
    "quantifiers": [("depth 0: nouns", "heldout_sentences", "head agent|head object|head recipient", "0.999±0.002 (C2c)", "9c-sexies"),
                    ("depth 0: _scope (wide role / both / absent)", "heldout_sentences", "_scope (wide role / both / absent)", "one quantifier 0.995, presence 1.000 (C2c)", "9c-sexies"),
                    ("two-quantifier scope, depth 0", "heldout_sentences", "_scope on two-quantifier sentences", "0.996 (C2c)", "9c-sexies"),
                    ("depth 2 UNSEEN: two-quantifier scope", "heldout_depth2_quantified", "_scope on two-quantifier sentences", "0.966 (C2c, 5/5 seeds)", "9c-sexies"),
                    ("ambiguous (never trained): both", "ambiguous_d0 (two unmarked quantifiers, never trained: _scope=both)", "_scope on two-quantifier sentences", "0.993 (C2c)", "9c-sexies"),
                    ("determiner identity (outside FULL)", "heldout_sentences", "determiner identity (outside FULL when the legend excludes it)", "0.950±0.013 (C2c)", "9c-sexies")],
}


def load(root, w):
    for p in (os.path.join(root, w, "metrics.json"), os.path.join(root, w, "model", "metrics.json")):
        if os.path.exists(p): return json.load(open(p))
    return None


def cell(m, split, metric, cs="catalog candidates"):
    ag = m.get("mean_over_seeds", {}).get(split, {}).get(cs, {})
    keys = metric.split("|"); vals = [ag[k] for k in keys if k in ag]
    if not vals: return "—"
    mean = sum(v["mean"] for v in vals) / len(vals); std = max(v["std"] for v in vals)
    return f"{mean:.3f}" + (f"±{std:.3f}" if len(m.get("seeds", [])) > 1 else "")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(TOOL, "runs", "p0_cpu_1seed_60f284d")); a = ap.parse_args()
    print(f"P0 round trip, {a.root}\n")
    print("| world | axis | package | tool FULL, structure estimated | tool FULL, depth given | RESULTS (5 seeds) | section |\n|---|---|---|---|---|---|---|")
    for w, rows in ROWS.items():
        m = load(a.root, w)
        if m is None: print(f"| {w} | (no metrics.json under {a.root}/{w}) | | | | | |"); continue
        for label, split, metric, ref, src in rows:
            print(f"| {w} | {label} | {cell(m, split, metric)} | {cell(m, split, 'FULL')} | {cell(m, split, 'FULL [depth given]')} | {ref} | {src} |")


if __name__ == "__main__":
    main()
