# Current status (rewritten whole at every judgment point; last: 2026-10-16 midday — E2c both targets + diagnostic, A2, B1+B2, B4, B3 done)

Labels (rule 22): BUILT / PLANNED / HYPOTHESIS. Sources: experiment/RESULTS.md 9c-ter, 10a-10d (numbers, logs with commit hash),
experiment/JOURNAL.md (midday entry), STATUS.md (public summary). The reference table now carries the predicate role and calibration (B1, B2),
so the rule-23 "draft" label is lifted on the reference world; WikiSQL tables still lack calibration (B2 on WikiSQL dev: PLANNED).

## A. What was run since the last judgment (all BUILT; logs in experiment/logs/)
| item | result | where |
|---|---|---|
| E2c slot target (R_sc (x) R_role (x) f_noun), 5 seeds | d0 scope 0.784, FULL 0.667; d2 UNSEEN scope 0.027 (absolute 'none' threshold) | RESULTS 9c-ter; log 2026-10-16_1031 |
| E2c noun target (R_sc (x) f_noun), 5 seeds | d0 scope **0.873**, FULL 0.749 (best seen-depth number); d2 UNSEEN 0.029; ambiguous "both" 0.693 | 9c-ter; log 2026-10-16_1052 |
| diag_e2c (rule 10: evaluation first; one model per target; three 'none' rules) | depth-2 scope under NO 'none' rule: role 0.52, slot 0.33, noun 0.21 (chance 0.33); raw true-wide score = other roles' score for slot/noun (0.024 vs 0.015; 0.035 vs 0.039): no signal at the unseen depth | 9c-ter; log 2026-10-16_1240 |
| decoder fix | 'none' rule made RELATIVE (best scope < 0.25 x best noun) — reads the affirmative case 0.84-1.00, changes quantified cases by <= 0.13 at d0; default from now on, old numbers not re-run | v14_quantifiers.decode_q |
| A2 memory curve (1 seed, 30k) | flat after step 500: RSS 2.46 GB, torch 0.64 GB; verifier 3 epochs: RSS 2.06 GB constant | 10a; log 2026-10-16_1115 |
| B1+B2 reference table with predicate role + occupancy marker + calibration, 5 seeds | seen FULL 1.000; predicate (surface / event type) 1.000 / 1.000; occupancy known / unknown / not given 1.000 / 1.000 / 1.000; confident-wrong 0.000 on every seen axis; **depth 3 UNSEEN 0.813** (6c without predicate: 1.000), margin reports it (coverage 0.33-0.59 at tau 0.05) | 10b; log 2026-10-16_1118 |
| B4 verifier, controlled (gold frame + one injected error; no LLM), dev 8,421 x 6 error types | role errors detected 0.74-0.81, confident-wrong 0.3-1.3%, false alarm on gold 0.162; precision 0.942 / recall 0.527 on role errors; condition errors mostly 'unverifiable'; values INVISIBLE (stated); criterion > 0.9 NOT met | 10c; log 2026-10-16_1130 |
| B3 commands: negation as a role around its scope (prohibition / negated attribute / both), 5 seeds | d0 presence 1.000, SCOPE 1.000 (all seven forms), FULL 0.999; d2 UNSEEN scope **0.997**, FULL 0.854 (fillers, not scope); scope-only pairs confident-wrong **0.000** at coverage 0.968; separation 0.06 vs bag 0.85; criteria MET (templates; LLM paraphrases PLANNED) | 10d; log 2026-10-16_1255 |

## B. The quantifier failure — what it means (for judgment; labels per bullet)
1. BUILT (negative, three worlds). "Seen at depth d in training (share p), tested at depth d+1" holds for reporting, negation, modality,
   conditional, modal-in-conditional, negation-on-commands (1.000 / 0.972 / 1.000 / 1.000 / 1.000 / 0.997) and FAILS for quantifier scope in all three target definitions
   (role 0.52, slot 0.33, noun 0.21 at depth 2, no 'none' rule; nouns 1.00 in all). The world correction approved last time (slot target)
   worked at the SEEN depth (one-quantifier scope 0.96-0.99, up from 0.89) and did not compose at all.
2. BUILT (mechanism visible in the logs). In both corrected worlds the determiner gates stayed CLOSED (open: from, that, to). The chain term,
   the one 'that' re-addresses and the one that makes nouns compose, is then identical for 'each' and 'every'; the scope difference is carried
   by bigram / path terms, which are position-local and do not move with the depth. This is exactly the second branch of the rule-11 trace.
3. HYPOTHESIS. Quantifier scope is a RELATION between two slots (which quantified phrase out-scopes the other), not an envelope operator
   R_op (x) S over a whole state as negation / modality / reporting are. A relation needs a degree-2 term in word identities (address (x) f_A (x)
   f_B) or an operator word that opens a frame; the model has degree-1 word terms only, and the gradient did not open the determiner gates
   (it would cost on the nouns, already at 1.000). Same class as the right-announced limit: the information is in the sentence, no causal
   degree-1 term can hold it at a depth-shifted address.
4. Impact on the claims. The composition regularity stays an empirical regularity with ONE named exception (rule 24 wording kept everywhere);
   the operator-as-role story is unchanged for envelope operators; the frame/verifier programme does not depend on quantifier scope (WikiSQL
   has none). What changes: the paper's §7 now names quantifiers as the negative result, §8 the predicate-role depth-3 regression.

## C. Proposed changes — NOT applied; for judgment, each with a criterion (PLANNED)
- C1. Degree-2 relational term: one per sentence, address (x) f_wide (x) f_narrow bound by the two quantified slots; emitted by the wide
  determiner, inert when no quantifier is present. Criterion: scope > 0.9 at depth 0 AND > 0.9 at depth 2 (regime of 9c-bis, 5 seeds), nouns
  unchanged 1.00, no loss on 6c / 9b / 9d tables. Tests hypothesis B3.
- C2. Determiner gates GIVEN open (rule 20: stated as given, not learned). Same criterion; the cost on nouns and on the other operator tables
  reported. Tests mechanism B2: if C2 already composes, the limit is in the learned gating, not in the degree of the terms. Order: C2 first
  (cheaper, one flag), then C1 only if C2 fails.
- C3. Scope on ambiguous sentences EMITTED as "both" (the product reading): the model produces the superposed state (cos 0.80-0.84 to the
  superposed world), and the two readings are what a reasoner receives; the noun target gives "both reported" 0.69. Criterion to adopt: both
  reported > 0.9 on never-trained ambiguous sentences, while unambiguous sentences get one reading > 0.9 (it must not say "both" everywhere).
- C4. Verifier false alarm on gold (0.162): raise tau on the reader's selected-column margin until false alarm < 0.05 and report the
  detection lost (acc-vs-coverage curve, not a tuned point); then the LLM-produced formalizations (no API here: PLANNED, data to be supplied).
- C5. Condition slot on WikiSQL (0.16 detection, 0.37 full query): a condition-count readout from the occupancy marker (A1 style) before any
  model change; criterion: present/absent condition columns > 0.9 on dev.
- C6. Predicate role depth-3 regression (0.813): report the acc-vs-coverage curve as the headline for unseen depth from now on, and test
  whether a larger D (2048) restores 1.000 with the extra term (capacity, not mechanism) — stated as such.

## D. Still PLANNED / HYPOTHESIS, in judgment order
B5-B7 (order collisions with permutation binding / level-indexed roles; sparse block codes; fourth address code remeasured), N3, N1, N2,
B8-B9 (HYPOTHESIS), N4; LLM paraphrases for B3 and LLM formalizations for B4 (PLANNED, no API).

## E. Questions for the reviewing session
1. Accept the quantifier result as the named exception of the regularity (B1), with the hypothesis B3 recorded as HYPOTHESIS?
2. Order of C2 then C1 — agreed? Or is the degree-2 term a change of principle (the model would then have one non-word term) that should
   not be built?
3. C3: adopt "emit both" as the product reading for ambiguity, with the criterion above?
4. With the predicate role, depth 3 unseen is 0.81 (6c: 1.000). Is the reference table now 10b (predicate role, calibration) with 6c kept
   as the predicate-less number, or should the depth-3 claim be re-stated as "at coverage 0.6, accuracy 1.00"?
5. B4 at 0.74-0.81 detection / 0.16 false alarm: continue to the LLM version (needs formalizations supplied), or C4/C5 first?
6. Process: the queue deadlock (pgrep matching its own command line) cost one hour; rule proposed: queues wait on the pid lock only.
