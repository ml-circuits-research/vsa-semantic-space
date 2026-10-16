# STATUS — public summary (stable URL; derived from current_status.md at every judgment point; never renamed)

Date: 2026-10-16. Commit: see `git log -1` (this file is committed with the status it describes). Labels (EVALUATION_PRINCIPLES rule 22):
BUILT = code and measurement exist, numbers reported; PLANNED = criterion written, not run; HYPOTHESIS = mechanism described, no criterion.
Every number cites experiment/RESULTS.md and a log in experiment/logs/ whose name carries the commit hash. The reference world now has the
predicate role and calibration (round 10 B1-B2, RESULTS 10b); the WikiSQL rows still lack calibration (B2 on WikiSQL dev: PLANNED).

## 1. Claims
| claim | status | where measured | criterion | number |
|---|---|---|---|---|
| Different formulations of one event converge to the same role state (synthetic worlds, no paraphrase labels; roles given, moves learned) | BUILT | RESULTS 3, 5, 6a; logs 2026-10-15_1128, _1246 | role accuracy on seen formulations > 0.95 | 1.000 (5 seeds) |
| New identities decode without retraining (identities never enter the parameters) | BUILT | RESULTS 3-7 | 1.00 by construction, measured against a control | 1.000; same-input transformer 0.215 (chance 0.167) |
| Composition to an unseen nesting depth (reporting, compositional world, prefixed forms) | BUILT | RESULTS 6c; log 2026-10-15_1319 | depth 3 full frame > 0.9 | 1.000 on all six roles (5 seeds), predicate-less world |
| The same with the predicate role and occupancy marker (reference table from now on) | BUILT (criterion not met at depth 3) | RESULTS 10b; log 2026-10-16_1118 | depth 3 > 0.9; confident-wrong < 2% at coverage > 80% | depth 3 UNSEEN 0.813; at tau 0.05 coverage 0.33-0.59 with covered accuracy 0.975-1.000, confident-wrong 0.000-0.008 |
| Predicate role (surface lemma; event type via a readout table); occupancy known / unknown filler / not given read from a marker, not the norm | BUILT | RESULTS 10b | each > 0.95, no loss on seen axes | 1.000 / 1.000; 1.000 / 1.000 / 1.000; seen axes unchanged at 1.000 |
| Calibration on the reference world: confident-wrong at a fixed threshold, abstention split, accuracy-vs-coverage | BUILT | RESULTS 10b | confident-wrong < 2% at coverage > 80% | MET on seen axes and new identities (0.000 at coverage 1.00); NOT met at depth 3 / unseen transitions (coverage 0.3-0.7) |
| Gates (which categories open a frame) emerge by gradient from the single objective | BUILT | RESULTS 7c; log 2026-10-15_1412 | same set in >= 3 worlds, no loss | identical set in 5/5 seeds: identities closed, relation markers open, verbs closed |
| Same-input transformer / linear associative map fail on new identities and the unseen depth | BUILT | RESULTS 7b, 8a | — | 0.215 / 0.390 (transformer); 0.24 / 0.25-0.39 (bottleneck, any rank) |
| Categories "emerge" from the displacements | RETRACTED | RESULTS 8a | skip-gram control | skip-gram recovers them equally (1.000 = 1.000); the model consumes categories |
| WikiSQL selected column without the table, same input as the baselines | BUILT | RESULTS 8c-8d; logs 2026-10-16_1614, _1628 | > BoW+LR and same-input transformer at every size | 0.82 vs 0.54 (transformer; 0.65 at 10x budget) vs 0.50 (BoW+LR); full query 0.37 vs 0.19 |
| WikiSQL aggregator | BUILT (negative) | RESULTS 8d | — | 0.80 vs 0.89 (transformer): a lexical decision, lost |
| WikiSQL equivalence separation is role-based | BUILT (negative with a nuance) | RESULTS 8d | hard pairs vs the lexical bag | overall lexical (0.29 vs 0.30); H2 same-words/different-SQL 0.62 vs 0.77 (n = 1,353) |
| frame(S) -> Prolog / SMT round trip, depth estimated from the state | BUILT | RESULTS 9a; log 2026-10-15_1705 | full frame > 0.95 | 1.000 at depths 0-3 (3 unseen) |
| Negation as a role: decoded with scope, separated from the affirmed state | BUILT | RESULTS 9b; log 2026-10-15_1709 | scope > 0.9; gap > 0.5 | scope 1.000; cos(aff, neg) 0.18 vs bag 0.72 |
| Operator combination seen at depth d (stated share) is read at depth d+1 — reporting, negation, modality, conditional, modal-in-conditional, negation on commands (scope) | BUILT (empirical regularity; counts stated; one named exception: quantifiers) | RESULTS 6c, 9b-bis, 9d, 9d-bis, 10d | full frame at d+1 > 0.9 | 1.000 / 0.972 / 1.000 / 1.000 / 1.000 / 0.997 (5 seeds each) |
| Same regularity for quantifiers (three world definitions: role identity, wide slot, wide noun) | BUILT (negative in all three) | RESULTS 9c-bis, 9c-ter; logs 2026-10-16_0833, _1031, _1052, _1240 | scope > 0.9 at depth 0; full frame at depth 2 > 0.9 | depth 0 scope 0.80 / 0.78 / 0.87; depth 2 scope 0.52 / 0.33 / 0.21 (chance 0.33; nouns 1.00) — the named exception; hypothesis (relation, not envelope) and two PLANNED tests in current_status.md |
| Verifier with injected errors on WikiSQL (controlled: gold frame + one error; no LLM) | BUILT (criterion not met) | RESULTS 10c; log 2026-10-16_1130 | detection > 0.9 on role errors | 0.74-0.81 detected, confident-wrong 0.3-1.3%, false alarm on gold 0.162 (lexical baseline: 0.32-0.62 detected, up to 0.29 confident-wrong); values invisible, stated |
| Negation on commands: prohibition vs negated attribute vs both, scope separate from presence | BUILT | RESULTS 10d; log 2026-10-16_1255 | scope > 0.95 (templates); confident-wrong < 1% at coverage > 90% | scope 1.000 (seven forms), confident-wrong 0.000 at coverage 0.968, depth 2 UNSEEN scope 0.997 (FULL 0.854); separation 0.06 vs bag 0.85; MET on templates, LLM paraphrases PLANNED |
| Memory curve of a reference training (round 10 A2) | BUILT | RESULTS 10a; log 2026-10-16_1115 | flat per step | RSS 2.46 GB and torch 0.64 GB constant after step 500 |
| Zero occurrences of a combination in training => not read | BUILT | RESULTS 9b, 9d | — | 0.0-0.5 (E1 sentential), 0.000 (E3 mixed) |
| Scope ambiguity: both hypotheses reported on never-trained ambiguous sentences | BUILT (observed, partial) | RESULTS 9c-bis | reported, no criterion | 0.43 both reported (n = 360/seed); forced choice surface-first 0.26 |
| Right-announced structure (postposed reporting, 'Q if P', comma constructions) is unaddressable by causal codes | BUILT (declared limit, with demonstration) | RESULTS 6b-6c, 9d | — | depth 2 0.88 with the postposed form; 'Q if P' 0.20-0.44 |
| Calibration on WikiSQL dev; LLM paraphrases (B3) and LLM formalizations (B4) | PLANNED | round 10 B2-B4 follow-ups | as above; no API here, data to be supplied | — |
| Quantifier scope: degree-2 relational term (C1) / determiner gates given open (C2); ambiguity emitted as "both" (C3) | PLANNED (for judgment) | current_status.md C | scope > 0.9 at depth 0 and depth 2, nouns 1.00, no loss elsewhere; both > 0.9 on ambiguous and one reading > 0.9 on unambiguous | — |
| Order collisions and remedies (permutation binding; level-indexed roles); sparse block codes; fourth address code remeasured | PLANNED | round 10 B5-B7 | stated in JUDGMENT_2026-10-16_consolidated.md | — |
| Mood as a role (questions, commands); emotion/stance with scope; fillers as the null jump; real chat corpus | PLANNED / HYPOTHESIS | Part V N1-N4 | stated in the judgment file | — |
| Pronoun resolution over stored frames; instruction-injection detection | HYPOTHESIS | round 10 B8-B9 | — | — |

## 2. Open world-definition corrections
- Quantifier scope term: three definitions tried (role identity 9c-bis; wide slot and wide noun, E2c 9c-ter). The corrected targets are
  reachable and better at the seen depth (0.96-0.99 one-quantifier scope) but carry NO signal at the unseen depth (determiner gates closed;
  the scope difference rides on position-local terms). Open: relation-vs-envelope hypothesis, C1/C2 tests (PLANNED).
- Decoder: the 'no quantifier' hypothesis uses a RELATIVE threshold from now on (an absolute one read 'none' everywhere at depth 2 — the
  evaluation artefact behind the 0.03 in E2c, found and stated in 9c-ter; old numbers not re-run).
- Predicate role: DONE (10b), option (a) surface lemma + synonym table at readout; the depth-3 claim is stated per world (6c predicate-less
  1.000; 10b with predicate 0.813, margin-reported).
- Flat world (depth-indexed speaker roles) retired in round 6; the compositional world is the reference.

## 3. Next three experiments, in order
1. C2 — determiner gates given open (one flag), same regime as 9c-bis, 5 seeds: does the quantifier scope then compose? (pending judgment)
2. C1 — degree-2 relational term, only if C2 fails and the judgment allows a non-word term (pending judgment).
3. B5 — order collisions and remedies (permutation binding; level-indexed roles), per JUDGMENT_2026-10-16_consolidated.md.
