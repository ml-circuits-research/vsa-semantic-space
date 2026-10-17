# Current status (rewritten whole at every judgment point; last: 2026-10-17 — judgment 45ccd1d applied: F1 recording, per-seed readout, C5 closed; tool round 4 next)

Labels (rule 22): BUILT / PLANNED / HYPOTHESIS. Sources: experiment/JUDGMENT_2026-10-17_45ccd1d.md (saved verbatim), experiment/RESULTS.md
12-quinquies, 10c-nonies; STATUS.md; experiment/JOURNAL.md; tool/README.md.

## A. Done since 45ccd1d, in the instructed order
| step | result | criterion / decision rule (written before) | outcome |
|---|---|---|---|
| F1 recording (one commit) | regularity row: ONE NAMED EXCEPTION (may_neg at the unseen depth 0.855) and the rule-29 (A)/(B) labels on 6c, 10b-quinquies, 10d (seen-hop case stated with its reason: 'the' is an open emitter between the openers), 9c-sexies, 12-quater; the two-limits row carries the trace's cosines (subject's address identical for one pair, closer to the other target for two); B5.1 row "below criterion"; B5d -> HYPOTHESIS ("needed if a legend requires the subject's term as a structure witness"); N8 PLANNED; rule 29; "soft" wording removed everywhere | — | done |
| per-seed readout of 12-quater (rescore_v20, log 2026-10-17_1028; RESULTS 12-quinquies) | the means reproduce the run's digits exactly (deterministic readout); may_neg at depth 2 per seed **0.850, 0.800, 0.950, 0.675, 1.000** (2/5 above 0.9); must 0.925-1.000 (5/5); every other form 0.975-1.000 | rule 27 | the exception is on record per seed |
| **C5 per-slot thresholds** (rescore_v17_c5_slots, log _1029; RESULTS 10c-nonies) | slots 2+ present 0.616 -> **0.760**; slot 1 0.995 -> **0.330** (33 absent examples; their marker cosine is above the present class's, the midpoint is unsettable); false alarm 0.153; by slot index 0.995 / 0.726 / 0.129, by number of conditions 0.995 / 0.835 / 0.664 | present >= 0.9 on slots 2+ AND false alarm <= 0.152 | **NOT met -> C5 CLOSED** as "bounded by the presence reading of the later slots"; the drop follows the slot index more than the number of conditions -> the later slot roles have too few examples (slot 3+ = 295 of 16,840 slots), not superposition |
| Tool (agent) | task D prepared: presence thresholds on calibration rows that include held-out formulations, one run (gpu_round4.sh), product number first in every table, depth estimator stated as a GIVEN | expectation written in the run header: the recipient's presence on unseen formulations recovers toward 0.839 without reading every role | GPU given; result at the next push |

## B. HYPOTHESIS
- Verifier condition axis (C5 closed): the presence reading of the second and later condition slots is bounded by the number of training examples
  of those slot roles (slot 3+ 0.129 vs slot 1 0.995); the diagnostic that would decide it further is a training set re-weighted toward
  multi-condition questions, or one condition role with a count — neither proposed now.
- B5d multi-hop left re-seating: needed only if a legend requires the subject's term as a structure witness (judgment 45ccd1d A2).
- N5 "resolve before drawing": diagnostic to print in the next kinds run.

## C. PLANNED, in the instructed order
Tool round 4 (presence thresholds on held-out formulations; running); B6 sparse block codes; B7 fourth address code remeasured; N6; N8 (verb-final
world, low priority); P1 when the team's data arrives (the annotation path once with a key, wherever the team runs it); N7 after N1; B8-B9.

## D. Questions for the reviewing session
1. C5 is closed as instructed. The diagnostic points at data per slot role, not capacity. Is a re-weighted training set (more multi-condition
   questions) worth one retraining later, or does the verifier stay as it is until P1?
2. 10d's rule-29 label: training at depths 0-1 only, yet every hop at depth 2 occurs at depth 1 because 'the' (an open emitter) sits between the
   openers and also follows 'that' inside the commands — labelled "seen-hop case, stated". Agreed, or should 10d be re-labelled (A) with a
   depth-2 training re-run under the E1b protocol for uniformity?
3. B6 / B7 are next on my side (sparse block codes: phase quantization L in {16, 64, 256}, per-block top-k, capacity curve, bits per vector;
   fourth address code remeasured on the X5 splits with the decision rule "keep only if new-construction does not drop by more than 0.02").
   The reference table for both is 10b-quinquies (relation-marker gates given). Proceed in that order, B6 first?
