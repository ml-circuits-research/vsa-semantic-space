# Current status (rewritten at every judgment point; last: 2026-10-15 night, round 9; E1b running)

## Deliverables requested in the post-8d judgment (section F) — status
- EMPIRICAL_NOTE.md: full draft; curve as figure 1 (to draw), same-input table, hard-pair table with n, budget check (gate PASSED: 10x
  transformer 0.649 < 0.75, claim stands at every size), header-at-input row, data-efficiency condition sentence + ablation (categories from
  the 1k/5k subset only: ours 0.654 / 0.751 vs transformer 0.414 / 0.516 — gap unchanged). DONE.
- PAPER_NOTE §7 in the approved tone ("Negative results and what they fixed", reason first, lessons not rounds); §6b round trip. DONE.
- frame()/diff()/to_prolog()/to_smt() in experiment/frame.py with the round-trip test (frame_roundtrip.py): 1.000 full-frame / Prolog /
  SMT(z3) at depths 0-3 (3 unseen), depth estimated from S, 3 seeds. DONE (RESULTS 9a).
- E1 run with its criterion table. DONE (RESULTS 9b), E1b follow-up RUNNING.

## E1 — negation as a role (5 seeds)
| axis | result | criterion |
|---|---|---|
| negated frame at depth 0 (8 forms, scopes S/A/B/C): roles / presence / SCOPE / FULL | 1.000 / 1.000 / **1.000** / **1.000** | > 0.95, > 0.9: MET |
| pair separation cos(affirmed, negated) | model **0.18** (gap 0.82); lexical bag 0.72; world 0.03 | gap > 0.5: MET |
| negation INSIDE reporting (unseen combination), depth 1: scope / FULL | 0.56 / 0.45 | > 0.9: NOT met |
Per form at the unseen combination: argument-scoped negation composes ('nobody', 'to nobody' 1.00; 'it is not X who' 0.85; 'nothing' 0.90),
SENTENTIAL negation does not ('does not V' 0.20, 'never' 0.00: it attaches to the theme instead of the clause). Learned gates: never, not,
nothing, anything, does, who OPEN (operators become emitters like 'that'); nobody closed (the indefinite lives in the filler).
E1b (running): negation seen at depths 0-1, tested at depth 2 — does sentential negation compose one level further once seen inside a
clause, as reporting did (6c)?

## Conclusions relative to the new programme
1. "Operator = role" holds for negation on its own, every scope, no rule, scope right 100%: Prolog/Z3 is a plan, not a promise, for
   single operators. The first limit is specific: an operator that displaces a whole clause does not compose into another such operator
   when never seen inside one (E1b decides whether this is a data or a mechanism limit).
2. The product skeleton exists and is exact on the synthetic world: text -> state -> frame (depth estimated) -> Prolog / SMT -> answer.
3. The short empirical paper is complete as a draft with its gate passed and its condition ablated.

## Questions for judgment
1. E1 criterion "negation inside reported speech > 0.9" was failed for sentential forms only. If E1b shows composition once seen at depth 1
   (as 6c did for reporting), do we accept "operators compose after one exposure inside each other" as the rule and move to E2? If E1b also
   fails, the next step would be a diagnosis of the sentential displacement (trace), before E2.
2. EMPIRICAL_NOTE: ready for a read-through? Figure 1 (the curve) still to draw — as a PNG from the log, or a table only for the findings format?
3. Process: four runs shared the GPU (4-8x slower each). Sequential launches from now on unless told otherwise.

## Files
experiment/RESULTS.md (8d additions, ROUND 9a-9b) | experiment/EMPIRICAL_NOTE.md | experiment/PAPER_NOTE.md | experiment/frame.py, frame_roundtrip.py, v13_negation.py, v12_wikisql_baselines.py
logs: 2026-10-15_1705_frame_roundtrip_a622582, _1709_v13_negation_5b9e933, _1652/_1655_v12 (budget), _1700_v12 (ablation)
