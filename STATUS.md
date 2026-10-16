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
| Same regularity for quantifiers (three world definitions: role identity, wide slot, wide noun) | BUILT (negative in all three) | RESULTS 9c-bis, 9c-ter; logs 2026-10-16_0833, _1031, _1052, _1240 | scope > 0.9 at depth 0; full frame at depth 2 > 0.9 | depth 0 scope 0.80 / 0.78 / 0.87 (decoder: no-none / absolute 0.08 / absolute 0.08); depth 2 scope 0.52 / 0.33 / 0.21 (no-none rule, chance 0.33; nouns 1.00) — the named exception; H-a / H-b recorded; C2 (gates given open, presence/scope split): depth 2 one-quantifier scope 0.67 / 0.63 (slot / noun), two-quantifier 0.53 / 0.56 — signal, no composition (9c-quater); remedies stopped |
| Verifier with injected errors on WikiSQL (controlled: gold frame + one error; no LLM) | BUILT (criterion not met) | RESULTS 10c; log 2026-10-16_1130 | detection > 0.9 on role errors | 10c-bis (threshold on the even half, odd half reported): 0.73-0.80 detected, confident-wrong 0.4-1.0%, false alarm on gold 0.162, on gold confirmed 0.30 / unverifiable 0.54 (a filter, not a judge); lexical 0.32-0.63 detected, up to 0.31 confident-wrong; values invisible. C4 (tau set on the even half for false alarm < 5%): tau 0.25, false alarm 0.033, role detection 0.32 (-0.44); confirmed on gold 0.297 at every tau |
| Negation on commands: prohibition vs negated attribute vs both, scope separate from presence | BUILT | RESULTS 10d; log 2026-10-16_1255 | scope > 0.95 (templates); confident-wrong < 1% at coverage > 90% | scope 1.000 (seven forms), confident-wrong 0.000 at coverage 0.968, depth 2 UNSEEN scope 0.997 (FULL 0.854); separation 0.06 vs bag 0.85; MET on templates, LLM paraphrases PLANNED |
| Memory curve of a reference training (round 10 A2) | BUILT | RESULTS 10a; log 2026-10-16_1115 | flat per step | RSS 2.46 GB and torch 0.64 GB constant after step 500 |
| Zero occurrences of a combination in training => not read | BUILT | RESULTS 9b, 9d | — | 0.0-0.5 (E1 sentential), 0.000 (E3 mixed) |
| Scope ambiguity: both hypotheses reported on never-trained ambiguous sentences | BUILT (observed, partial) | RESULTS 9c-bis | reported, no criterion | 0.43 both reported (n = 360/seed); forced choice surface-first 0.26 |
| Right-announced structure (postposed reporting, 'Q if P', comma constructions) is unaddressable by causal codes | BUILT (declared limit, with demonstration) | RESULTS 6b-6c, 9d | — | depth 2 0.88 with the postposed form; 'Q if P' 0.20-0.44 |
| Calibration on WikiSQL dev; C5 condition count from an occupancy marker (one retraining); LLM paraphrases (B3) and LLM formalizations (B4; input format in experiment/LLM_FORMALIZATIONS_FORMAT.md) | PLANNED | round 10 B2-B4 follow-ups | C5: present/absent > 0.9 on dev; others as above; no API here, data to be supplied | — |
| Quantifier scope with the determiner gates GIVEN open (C2; tests H-a) | BUILT (criterion not met; H-a partly supported) | RESULTS 9c-quater; logs 2026-10-16_1557, _1619; trace logs/trace_c2.txt | scope > 0.9 at depth 0 AND depth 2, one-/two-quantifier separately, nouns 1.00 | depth 0: one-q 0.98 / 0.96 MET, two-q 0.76 / 0.79 not; depth 2: one-q 0.67 / 0.63, two-q 0.53 / 0.56 (chance 0.33) not; nouns 0.99 / 0.98 (cost) |
| Permuted-filler relational term (H-b) | NOT BUILT (judgment: only if C2 composed one quantifier and not two; it composed neither) | — | — | — |
| Ambiguity emitted as "both" (C3) | BUILT (observed, partial) | RESULTS 9c-ter, 9c-quater | both > 0.9 on never-trained ambiguous AND one reading > 0.9 on unambiguous; false-both rate reported | both 0.53 (noun target, C2 decoder) / 0.38 (slot); 0.69 with the absolute-0.08 decoder; one reading on unambiguous one-quantifier 0.96-0.98 |
| Depth-3 regression with the predicate role: breakdown, recipient-less share 0, verb gates given closed, D = 2048 | BUILT (neither branch of the decision rule fires) | RESULTS 10b-bis, 10b-ter; logs 2026-10-16_1351, _1511, _1523, _1535 | gating if (b) or (c) restores > 0.9; capacity if only (d) does | (b) 0.858, (c) 0.745, (d) 0.852; the loss is on C in the PASSIVE forms (T2 C 0.52-0.75) — a form effect, reported as a property of the predicate-role world |
| N5 kind ambiguity; N6 missing-legend residual; N7 world identification by fit | PLANNED (criteria not in the repo: the addendum text was not received; rows added at the judgment's request) | current_status.md D | to be supplied with the addendum | — |
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
  1.000; 10b with predicate 0.813 at full coverage, 1.000 at coverage 0.59 (A) / 0.975 at 0.33 (C)); E4 (b)-(d) show the loss is on the
  passive forms at depth, neither gating nor capacity (10b-ter); a remedy (participle as emitter, given) is proposed, not built.
- Flat world (depth-indexed speaker roles) retired in round 6; the compositional world is the reference.

## 3. Next three experiments, in order
1. B5 — order collisions and remedies (permutation binding; level-indexed roles), per JUDGMENT_2026-10-16_consolidated.md.
2. B6 — sparse block codes; B7 — fourth address code remeasured.
3. C5 — condition count from an occupancy marker in the verifier reader's world (one retraining), then the reader's confident-wrong on sel/agg.
