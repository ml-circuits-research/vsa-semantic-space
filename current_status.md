# Current status (rewritten at every judgment point; last: 2026-10-15 night, round 9 incl. E1b)

## Deliverables requested in the post-8d judgment (section F) — status
- EMPIRICAL_NOTE.md: full draft; curve as figure 1 (to draw), same-input table, hard-pair table with n, budget check (gate PASSED: 10x
  transformer 0.649 < 0.75, claim stands at every size), header-at-input row, data-efficiency condition sentence + ablation (categories from
  the 1k/5k subset only: ours 0.654 / 0.751 vs transformer 0.414 / 0.516 — gap unchanged). DONE.
- PAPER_NOTE §7 in the approved tone ("Negative results and what they fixed", reason first, lessons not rounds); §6b round trip. DONE.
- frame()/diff()/to_prolog()/to_smt() in experiment/frame.py with the round-trip test (frame_roundtrip.py): 1.000 full-frame / Prolog /
  SMT(z3) at depths 0-3 (3 unseen), depth estimated from S, 3 seeds. DONE (RESULTS 9a).
- E1 run with its criterion table. DONE (RESULTS 9b); E1b follow-up DONE (RESULTS 9b-bis).

## E1 — negation as a role (5 seeds)
| axis | result | criterion |
|---|---|---|
| negated frame at depth 0 (8 forms, scopes S/A/B/C): roles / presence / SCOPE / FULL | 1.000 / 1.000 / **1.000** / **1.000** | > 0.95, > 0.9: MET |
| pair separation cos(affirmed, negated) | model **0.18** (gap 0.82); lexical bag 0.72; world 0.03 | gap > 0.5: MET |
| negation INSIDE reporting (unseen combination), depth 1: scope / FULL | 0.56 / 0.45 | > 0.9: NOT met |
Per form at the unseen combination: argument-scoped negation composes ('nobody', 'to nobody' 1.00; 'it is not X who' 0.85; 'nothing' 0.90),
SENTENTIAL negation does not ('does not V' 0.20, 'never' 0.00: it attaches to the theme instead of the clause). Learned gates: never, not,
nothing, anything, does, who OPEN (operators become emitters like 'that'); nobody closed (the indefinite lives in the filler).
E1b (done): negation seen at depths 0-1, tested at depth 2 (UNSEEN): FULL frame **0.972±0.049**, all eight forms at scope 1.00 (seed 0),
sentential negation included (E1: 0.0-0.5 -> 1.00). A DATA limit, not a mechanism limit — the same pattern as reporting in 6c: an operator
composes one level further than trained once it has been seen inside another operator once.

## Conclusions relative to the new programme
1. "Operator = role" holds for negation on its own, every scope, no rule, scope right 100%: Prolog/Z3 is a plan, not a promise, for
   single operators AND for negation inside reporting after one exposure (E1b): operators compose one level further than trained, as
   reporting did. Rule for E2-E3: train each operator at depths 0-1, test at depth 2.
2. The product skeleton exists and is exact on the synthetic world: text -> state -> frame (depth estimated) -> Prolog / SMT -> answer.
3. The short empirical paper is complete as a draft with its gate passed and its condition ablated.

## Questions for judgment
1. E1b resolved the E1 failure (0.97 at the unseen depth once negation is seen inside a clause). Proposal: accept "operators compose after one
   exposure inside each other" as the working rule, train E2-E3 operators at depths 0-1 and test at 2, and move to E2 (quantifiers) now.
2. EMPIRICAL_NOTE: ready for a read-through? Figure 1 (the curve) still to draw — as a PNG from the log, or a table only for the findings format?
3. Process: four runs shared the GPU (4-8x slower each). Sequential launches from now on unless told otherwise.

## Files
experiment/RESULTS.md (8d additions, ROUND 9a-9b) | experiment/EMPIRICAL_NOTE.md | experiment/PAPER_NOTE.md | experiment/frame.py, frame_roundtrip.py, v13_negation.py, v12_wikisql_baselines.py
logs: 2026-10-15_1705_frame_roundtrip_a622582, _1709_v13_negation_5b9e933, _1652/_1655_v12 (budget), _1700_v12 (ablation)
