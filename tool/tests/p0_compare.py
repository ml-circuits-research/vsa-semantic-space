"""
p0_compare.py — the P0 round-trip table: the package's numbers (tool/worlds_p0/<world>/model/metrics.json) next to the RESULTS tables'
(5 seeds, GPU; cited by section). The package estimates the reporting depth (and the negation) from S; the repository scripts are given the
depth — so the comparable column is the roles-only one ('heads FULL' / per-role), and the tool's FULL (roles + predicate + structure) is
printed beside it. Run: python3 tool/tests/p0_compare.py [--root tool/worlds_p0]
"""
import argparse, json, os
TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROWS = {   # world: [(label, split, metric, RESULTS value, source)]
    "reference": [("seen formulations FULL (A, B, C)", "heldout_sentences", "heads FULL (model)", "1.000±0.000", "10b"),
                  ("predicate, surface lemma", "heldout_sentences", "predicate (surface lemma)", "1.000", "10b"),
                  ("new identities FULL", "new_identities", "heads FULL (model)", "1.000", "10b"),
                  ("role cues seen FULL", "heldout_formulations_role_cues_seen", "heads FULL (model)", "1.000", "10b"),
                  ("unseen transitions FULL", "heldout_formulations_unseen_transitions", "heads FULL (model)", "0.638±0.015", "10b"),
                  ("depth 3 UNSEEN FULL", "heldout_depth3", "heads FULL (model)", "0.813±0.030", "10b")],
    "commands": [("depth 0: action", "heldout_sentences", "role action", "1.000", "10d"),
                 ("depth 0: patient (with attribute and its negation)", "heldout_sentences", "role patient", "1.000 / 0.999 (patient / attribute)", "10d"),
                 ("depth 0 FULL", "heldout_sentences", "FULL", "0.999", "10d"),
                 ("new patients FULL", "new_identities", "FULL", "1.000", "10d"),
                 ("depth 2 UNSEEN FULL", "heldout_depth2", "FULL", "0.854±0.031", "10d"),
                 ("depth 2 UNSEEN: structure (depth + negation scope)", "heldout_depth2", "structure (depth, negation)", "scope 0.997±0.003 (depth given)", "10d")],
    "quantifiers": [("depth 0: nouns (agent / object / recipient heads)", "heldout_sentences", "head agent|head object|head recipient", "0.999", "9c-bis"),
                    ("depth 0: noun + determiner per role", "heldout_sentences", "role agent|role object|role recipient", "determiners 0.923±0.009", "9c-bis"),
                    ("depth 2 UNSEEN quantified: nouns", "heldout_depth2_quantified", "head agent|head object|head recipient", "1.00 (seed 0 diagnostic)", "9c-bis"),
                    ("depth 2 UNSEEN quantified: noun + determiner", "heldout_depth2_quantified", "role agent|role object|role recipient", "determiners 0.46-0.64 (seed 0)", "9c-bis"),
                    ("new nouns depth 0: noun + determiner FULL", "new_identities", "FULL", "0.657 (FULL incl. scope)", "9c-bis")],
}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(TOOL, "worlds_p0")); a = ap.parse_args()
    print("| world | axis | package (1 seed, CPU) | tool FULL incl. structure | RESULTS (5 seeds) | section |\n|---|---|---|---|---|---|")
    for w, rows in ROWS.items():
        p = os.path.join(a.root, w, "model", "metrics.json")
        if not os.path.exists(p): print(f"| {w} | (no model: run tests/p0_full_cpu.sh) | | | | |"); continue
        m = json.load(open(p))["per_seed"][0]
        for label, split, metric, ref, src in rows:
            acc = m[split]["catalog candidates"]["acc"]; vals = [acc[k] for k in metric.split("|")]
            v = sum(vals) / len(vals); full = acc["FULL"]
            print(f"| {w} | {label} | {v:.3f} | {full:.3f} | {ref} | {src} |")

if __name__ == "__main__":
    main()
