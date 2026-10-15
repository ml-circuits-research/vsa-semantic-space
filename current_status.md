# Current status (rewritten at every judgment point; last: 2026-10-15 night, round 9 incl. E1b and E2 first run)

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

## E2 — quantifiers as roles, first run (5 seeds; RESULTS 9c)
PROTOCOL ERROR found by the diagnostic, stated first: training had quantifier sentences at depths 0-1 and NO sentence at depth 2 at all
(E1b had affirmative sentences at 0-2). The chain parameter 'that:that' only exists at depth >= 2, so at depth 2 everything collapses (nouns
0.31, determiners 0.29) — not the quantifiers. The depth-2 column is void; E2 is being re-run with the E1b protocol (affirmative 0-2,
operator 0-1). The composition rule is NOT yet tested for quantifiers; the reserve stands untouched.
What the run establishes at depths 0-1:
| axis | result | criterion |
|---|---|---|
| nouns / determiners (depth 0) | 1.000 / 0.973 | MET |
| scope with ONE quantifier (depth 0 / 1) | 0.950 / 0.967 | > 0.9 MET |
| scope with TWO quantifiers, marker 'each' / 'a certain' (depth 0 / 1) | **0.811 / 0.744** | > 0.9 NOT met — real weakness |
| ambiguous 'every ... a ...' (never trained): both hypotheses reported | **0.46-0.50**; cos to the superposed world 0.86; forced choice surface-first 0.32 | reported; partial "knows it does not know", by composition |
Learned gates: that, '.', to open; recipient verbs (sees, buys, receives, ...) open, active verbs closed; determiners closed.
Decision taken (rule 10: evaluation -> data -> optimization -> architecture): this is a data/protocol error at the evaluation-data level,
fixed in the data; nothing in the model is changed.

## Conclusions relative to the new programme
1. "Operator = role" holds for negation on its own, every scope, no rule, scope right 100%: Prolog/Z3 is a plan, not a promise, for
   single operators AND for negation inside reporting after one exposure (E1b): operators compose one level further than trained, as
   reporting did. Rule for E2-E3: train each operator at depths 0-1, test at depth 2.
2. The product skeleton exists and is exact on the synthetic world: text -> state -> frame (depth estimated) -> Prolog / SMT -> answer.
3. The short empirical paper is complete as a draft with its gate passed and its condition ablated.

## Questions for judgment
1. E2 two-quantifier scope at 0.81 / 0.74 (one quantifier 0.95): report as a limit of the scope role as defined (one R_sc term per wide
   argument), or allow one world-side change — a per-quantifier scope term R_sc (x) f_det (x) f_role — before E3? My preference: report as is
   now, re-run E2 with the corrected protocol, and decide after the depth-2 result.
2. Ambiguity: "both reported" at ~0.5 by composition, never trained. Is that the number to keep (honest, partial), or do we also report the
   forced-choice distribution (surface-first 0.32) as the default the state would give a reasoner?
3. Figure 1 is drawn (figures/fig1_wikisql_curve.png); EMPIRICAL_NOTE references it. E3 (modality/conditionals, with the right-announced 'Q if P'
   form reported separately) is written and waits for the GPU (sequential launches).
2. EMPIRICAL_NOTE: ready for a read-through? Figure 1 (the curve) still to draw — as a PNG from the log, or a table only for the findings format?
3. Process: four runs shared the GPU (4-8x slower each). Sequential launches from now on unless told otherwise.

## Files
experiment/RESULTS.md (8d additions, ROUND 9a-9b) | experiment/EMPIRICAL_NOTE.md | experiment/PAPER_NOTE.md | experiment/frame.py, frame_roundtrip.py, v13_negation.py, v12_wikisql_baselines.py
logs: 2026-10-15_1705_frame_roundtrip_a622582, _1709_v13_negation_5b9e933, _1652/_1655_v12 (budget), _1700_v12 (ablation)
