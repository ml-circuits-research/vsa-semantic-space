# Current status (rewritten whole at every judgment point; last: 2026-10-16 evening — judgment a8a8fab applied and its queue run: C, E4 (a)-(d), C2, C4)

Labels (rule 22): BUILT / PLANNED / HYPOTHESIS. Sources: experiment/JUDGMENT_2026-10-16_a8a8fab.md (the judgment), experiment/RESULTS.md
9c-quater, 10b-bis, 10b-ter, 10c-bis, 10c-ter, STATUS.md, experiment/JOURNAL.md (two entries today, afternoon and evening).

## A. Results of the judgment's order (all BUILT; logs with hash e0cbbbc / ef75946)
| step | result | criterion / decision rule | outcome |
|---|---|---|---|
| C2-threshold re-runs: v16 (training rows), v17 (even half of dev) | every 10b column reproduced to three decimals (threshold 0.21-0.23 either way); 10c numbers move <= 0.013 | report old and new | thresholds were never the issue; rule 25 now satisfied (10b-bis, 10c-bis) |
| E4 (a) depth-3 breakdown | B and predicate 1.000 everywhere; the loss is on C (and A) in the PASSIVE forms (T2 "B is VP to C by A": A 0.81 / C 0.52; T3 0.84 / 0.76) and the C-fronted comma form (C 0.77); verbs with half-open gates are at 1.000; participles (gates closed) carry the loss | find which forms fail | the judgment's first suspect (half-open verb gates) excluded |
| E4 (b) recipient-less share 0 / (c) verb gates given closed / (d) D = 2048 | depth 3: 0.858±0.081 / **0.745**±0.042 (seen axes also drop, 0.986) / 0.852±0.060; passive T2 C 0.75 / 0.56 / 0.60 | gating if (b) or (c) > 0.9; capacity if only (d) | NEITHER fires: a FORM effect (passive + predicate term at depth); reported as a property of the predicate-role world, with the breakdown and the acc-vs-coverage numbers (0.813 full coverage; 1.000 at 0.59 (A), 0.975 at 0.33 (C)) |
| C2: determiner gates GIVEN open, both targets, presence/scope split, depth-2 trace first | depth 0: one-quantifier scope 0.98 / 0.96 (slot / noun), two-quantifier 0.76 / 0.79; depth 2 UNSEEN: one-q **0.67 / 0.63**, two-q 0.53 / 0.56 (chance 0.33; before: 0.33 / 0.21, no signal); presence d2 0.81 / 0.76; nouns 0.99 / 0.98 (cost) | scope > 0.9 at d0 AND d2, one-/two-q separately | NOT met at d2 for both; one-q at d0 MET for the first time. H-a PARTLY supported (signal, no composition); H-b untested. Per the judgment: remedies STOPPED, no permuted-filler term, no third world |
| C3 "emit both" | both 0.53 (noun) / 0.38 (slot) with the C2 decoder; 0.69 with the absolute-0.08 decoder (label kept) | both > 0.9 and one reading > 0.9 on unambiguous | NOT met; observed, partial |
| C4: tau set on the even half (false alarm < 5%), curve on the odd half, no retraining | chosen tau 0.25: false alarm 0.033, role detection **0.32** (from 0.76 at tau 0.05), confirmed on gold 0.297 at every tau | report detection lost at every tau | the 16% false alarm is confident WRONG decoding of sel / agg, not margin noise: the next step is the reader's confident-wrong rate (a reader property), not the verifier threshold |
| LLM formalizations | input format committed (experiment/LLM_FORMALIZATIONS_FORMAT.md: jsonl, two LLMs, disagreement discards, audit of 50, n >= 1,000, seed stated) | — | PLANNED, waits for the data |

## B. Process (BUILT this period)
Rule 25 (thresholds on calibration rows) and rule 26 (queues on the pid lock only; queue.sh; run.sh excludes its own tree); every script saves
final params; v14 decoder = presence + scope|present + joint, one-/two-quantifier separate, no 'none' rule; every quantifier number labelled
with its decoder rule; 9c-ter false sentence corrected; degree-2 term withdrawn; H-a / H-b recorded; N5-N7 rows without criteria (addendum text
not received); run.sh ignores *.md in its dirty-tree check (documents are edited while a queue runs; the hash identifies the code).

## C. Where this leaves the claims
- Composition regularity: six operator cases hold; quantifier scope is the named exception. Carrying the scope term in the re-addressed code
  (C2) turns "no signal" into a weak signal (0.63-0.67 at the unseen depth) without composing; the account is partial and stays HYPOTHESIS.
- Reference table (10b): the depth-3 regression is a form effect (passives with the predicate term); the number is stated in the two-number
  form with the breakdown. The 6c 1.000 is the predicate-less world's number.
- Verifier: a filter on the role part (detects 0.73-0.80 of role errors at 16% false alarm; confirms 0.30 of gold); the false alarm is bounded
  by the reader's confident-wrong rate, not by the threshold.

## D. PLANNED / HYPOTHESIS, in judgment order
B5 (order collisions: permutation binding, level-indexed roles), B6 (sparse block codes), B7 (fourth address code remeasured); C5 (occupancy
marker in the verifier reader's world, one retraining; present/absent > 0.9); the reader's confident-wrong on sel/agg; N3, N1, N2; N5-N7
(criteria not received); B8-B9 (HYPOTHESIS); N4; LLM paraphrases (B3) and LLM formalizations (B4) when data arrives.

## E. Proposals for judgment (not built)
1. Depth-3 passives: the participle as an EMITTER (gate given open, rule 20) so that C after "given to" is addressed through the participle
   rather than through 'to' alone; criterion: T2 / T3 C > 0.9 at depth 3, no loss elsewhere. Or accept the form effect as a stated property.
2. Quantifiers: stop here (as the judgment says) and record the C2 effect size; or one more measurement WITHOUT a new world — the same C2 run
   with the quantified share raised at depth 1 (0.6 -> 0.9) to see whether the depth-2 signal grows with exposure (tests whether 0.65 is a
   data effect or a ceiling). Not started.
3. Verifier false alarm: measure the reader's confident-wrong on sel / agg against training size (8d curve, re-scored at tau 0.05) before any
   change; C5 after.
4. run.sh ignoring *.md — confirm.
5. N5-N7 — the addendum text.
