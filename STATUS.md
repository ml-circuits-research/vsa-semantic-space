# STATUS — public summary (stable URL; derived from current_status.md at every judgment point; never renamed)

Date: 2026-10-16. Commit: see `git log -1` (this file is committed with the status it describes). Labels (EVALUATION_PRINCIPLES rule 22):
BUILT = code and measurement exist, numbers reported; PLANNED = criterion written, not run; HYPOTHESIS = mechanism described, no criterion.
Every number cites experiment/RESULTS.md and a log in experiment/logs/ whose name carries the commit hash. Calibration (rule 18) and the
predicate role (rule 23) are not yet in these tables: until round 10 B1-B2 land, every row below is a draft under rule 23.

## 1. Claims
| claim | status | where measured | criterion | number |
|---|---|---|---|---|
| Different formulations of one event converge to the same role state (synthetic worlds, no paraphrase labels; roles given, moves learned) | BUILT | RESULTS 3, 5, 6a; logs 2026-10-15_1128, _1246 | role accuracy on seen formulations > 0.95 | 1.000 (5 seeds) |
| New identities decode without retraining (identities never enter the parameters) | BUILT | RESULTS 3-7 | 1.00 by construction, measured against a control | 1.000; same-input transformer 0.215 (chance 0.167) |
| Composition to an unseen nesting depth (reporting, compositional world, prefixed forms) | BUILT | RESULTS 6c; log 2026-10-15_1319 | depth 3 full frame > 0.9 | 1.000 on all six roles (5 seeds) |
| Gates (which categories open a frame) emerge by gradient from the single objective | BUILT | RESULTS 7c; log 2026-10-15_1412 | same set in >= 3 worlds, no loss | identical set in 5/5 seeds: identities closed, relation markers open, verbs closed |
| Same-input transformer / linear associative map fail on new identities and the unseen depth | BUILT | RESULTS 7b, 8a | — | 0.215 / 0.390 (transformer); 0.24 / 0.25-0.39 (bottleneck, any rank) |
| Categories "emerge" from the displacements | RETRACTED | RESULTS 8a | skip-gram control | skip-gram recovers them equally (1.000 = 1.000); the model consumes categories |
| WikiSQL selected column without the table, same input as the baselines | BUILT | RESULTS 8c-8d; logs 2026-10-16_1614, _1628 | > BoW+LR and same-input transformer at every size | 0.82 vs 0.54 (transformer; 0.65 at 10x budget) vs 0.50 (BoW+LR); full query 0.37 vs 0.19 |
| WikiSQL aggregator | BUILT (negative) | RESULTS 8d | — | 0.80 vs 0.89 (transformer): a lexical decision, lost |
| WikiSQL equivalence separation is role-based | BUILT (negative with a nuance) | RESULTS 8d | hard pairs vs the lexical bag | overall lexical (0.29 vs 0.30); H2 same-words/different-SQL 0.62 vs 0.77 (n = 1,353) |
| frame(S) -> Prolog / SMT round trip, depth estimated from the state | BUILT | RESULTS 9a; log 2026-10-15_1705 | full frame > 0.95 | 1.000 at depths 0-3 (3 unseen) |
| Negation as a role: decoded with scope, separated from the affirmed state | BUILT | RESULTS 9b; log 2026-10-15_1709 | scope > 0.9; gap > 0.5 | scope 1.000; cos(aff, neg) 0.18 vs bag 0.72 |
| Operator combination seen at depth d (stated share) is read at depth d+1 — reporting, negation, modality, conditional, modal-in-conditional | BUILT (empirical regularity; counts stated) | RESULTS 6c, 9b-bis, 9d, 9d-bis | full frame at d+1 > 0.9 | 1.000 / 0.972 / 1.000 / 1.000 / 1.000 (5 seeds each) |
| Same regularity for quantifiers | BUILT (negative, in the role-identity world) | RESULTS 9c-bis; log 2026-10-16_0833 | full frame at depth 2 > 0.9 | 0.362 (nouns 1.00, quantifier terms ~0.5) — re-test E2c after a world correction |
| Zero occurrences of a combination in training => not read | BUILT | RESULTS 9b, 9d | — | 0.0-0.5 (E1 sentential), 0.000 (E3 mixed) |
| Scope ambiguity: both hypotheses reported on never-trained ambiguous sentences | BUILT (observed, partial) | RESULTS 9c-bis | reported, no criterion | 0.43 both reported (n = 360/seed); forced choice surface-first 0.26 |
| Right-announced structure (postposed reporting, 'Q if P', comma constructions) is unaddressable by causal codes | BUILT (declared limit, with demonstration) | RESULTS 6b-6c, 9d | — | depth 2 0.88 with the postposed form; 'Q if P' 0.20-0.44 |
| Predicate role; occupancy marker; calibration (confident-wrong, abstention curve); reference table re-run with them | PLANNED | round 10 B1-B2 | B1: three occupancy cases > 0.95 each, predicates > 0.95, no loss; B2: confident-wrong < 2% at coverage > 80% (reported, not tuned) | — |
| Scope of negation on commands; verifier with injected errors on WikiSQL | PLANNED | round 10 B3-B4 | scope > 0.95 (templates), confident-wrong < 1% at coverage > 90%; detection > 0.9 on role errors | — |
| Order collisions and remedies (permutation binding; level-indexed roles); sparse block codes; fourth address code remeasured | PLANNED | round 10 B5-B7 | stated in JUDGMENT_2026-10-16_consolidated.md | — |
| Mood as a role (questions, commands); emotion/stance with scope; fillers as the null jump; real chat corpus | PLANNED / HYPOTHESIS | Part V N1-N4 | stated in the judgment file | — |
| Pronoun resolution over stored frames; instruction-injection detection | HYPOTHESIS | round 10 B8-B9 | — | — |

## 2. Open world-definition corrections
- Quantifier scope term: 9c-bis used R_sc (x) f_role (a role identity as filler — unreachable by any address (x) f_word term). Correction
  approved: R_sc (x) (R_role (x) f_noun), tested in E2c (rule-11 trace first); R_sc (x) f_noun as a secondary column.
- Predicate role missing in all worlds so far (GIVE and SELL map to the same target): R_pred (x) f_verb_lemma, option (a) surface-verb
  identity with a per-world synonym table at readout; reference table re-run (B1).
- Flat world (depth-indexed speaker roles) retired in round 6; the compositional world is the reference.

## 3. Next three experiments, in order
1. E2c — quantifiers with the corrected scope term (both target columns, n per cell, rule-11 trace).
2. Round 10 A2 — stability commit (memory logged every N steps, checkpoints with --resume, log-advance watchdog) + the memory curve of
   one short reference training.
3. Round 10 B1 + B2 — predicate role and occupancy marker (reference table re-run), calibration in evaluate() on every table.
