# Current status (rewritten whole at every judgment point; last: 2026-10-16 night — indications 013c490 applied: E1b, C2c, N5, 4c, P0)

Labels (rule 22): BUILT / PLANNED / HYPOTHESIS. Sources: experiment/INDICATIONS_2026-10-16_013c490.md (saved verbatim), experiment/RESULTS.md
10b-quinquies, 9c-sexies, 10e, 10c-quinquies, 11; STATUS.md; experiment/JOURNAL.md; tool/README.md. Note: the file
JUDGMENT_2026-10-16_013c490.md the indications refer to ("the answers and the reasoning behind E1b, C2c, N5") was never received; only the
indications file exists in the repository.

## A. Results, in the indications' order (all BUILT)
| step | result | criterion / decision rule (written before the run) | outcome |
|---|---|---|---|
| §0 | 10b-quater's "right-announced structure" sentence on the passive withdrawn | — | done (stays withdrawn after E1b) |
| **§1 E1b** relation-marker gates to / by / from GIVEN open (log 2026-10-16_1905) | depth 3 UNSEEN **0.968**±0.020 (10b: 0.813); passives T2 / T3 A and C 1.000; seen axes and occupancy 1.000; unseen transitions 0.627 (0.638); at tau 0.05 coverage 0.93 / 0.88, confident-wrong 0.000; T9 comma form C 0.84 | depth 3 > 0.9, passives > 0.9, no loss | **MET** — the regression was the gate set, not the form; new reference table (STATUS depth-3 row), learned-gate world kept as history |
| **§2 C2c** category-local quantifier terms, wide/narrow derived at readout (log 2026-10-16_1918; trace logs/trace_c2c.txt) | two-quantifier SCOPE|gold-q d0 **0.996**, **d2 UNSEEN 0.966** (0.988, 0.927, 0.976, 0.951, 0.988); one-quantifier 0.995 / 0.992 (seed 1: 0.974 / 0.980); nouns 0.999; affirmative read absent 1.000; presence d2 0.983, strict 0.924; both on never-trained ambiguous 0.993; FALSE-BOTH 0.000 / 0.000; determiners 0.950, FULL d2 0.60 (cost) | two-q > 0.9 at d0 AND d2; one-q >= 0.99; nouns >= 0.99; strict presence and false-both reported | **MET** on 5/5 seeds for the two-quantifier scope -> quantifiers join the regularity entirely; exception row removed; H-b retired; permuted-filler term NOT built (WITHDRAWN); C3 "both" criterion met under the C2c readout rule |
| **§3 N5** kind ambiguity (v19_kinds, log 2026-10-16_1941) | ref 1.000; (a) majority kind: h-as-noun 0.989, h-as-verb 1.000; (b) superposition + readout by fit: h-as-noun **0.907**±0.026 (0.950, 0.900, 0.909, 0.869, 0.906), h-as-verb 1.000; kind by fit noun 0.967 / verb 0.374, resolved 0.555, both 0.445; bag 0.866 vs state 0.821 | (b) within 0.02 of ref on ambiguous sentences | **NOT met** (gap 0.09 on the noun uses; 0.047 averaged over both uses); (a) is the better baseline; recorded as negative, no remedy without judgment |
| §4a audit | 50 confident-wrong sel cases labelled by me (stated): 34 reader wrong / 7 gold wrong / 9 ambiguous — 0.32 data noise | — | done (10c-quater-bis); C5 is next on the verifier |
| **§4c** data curve (logs 2026-10-16_2033-_2038) | sel confident-wrong at tau 0.05: 0.056 / 0.082 / 0.053 at 1k / 5k / 56k while accuracy 0.38 -> 0.59 -> 0.81 and abstain 0.84 -> 0.24; gold column name NOT in question: confident-wrong 0.069 -> 0.147 -> 0.169 (IN: 0.052 -> 0.023); agg confident-wrong 0.25 -> 0.14; role-error recall 0.21 -> 0.38 -> 0.52, false alarm on gold 0.13 / 0.20 / 0.16 | reported, no criterion | data converts abstentions into right answers, not confident-wrong; the lexical-mismatch confident-wrong GROWS with data |
| **§5 P0** tool (separate agent, worktree; merged; RESULTS 11, tool/README.md) | sop package + server.js + skill + 6 tests PASS (CPU); round trip 1 seed: reference seen 1.000, depth 3 0.792 (0.813±0.030), commands depth 2 0.790 (0.854±0.031), quantifier nouns 0.997 (scope not exported in v1); missing-role recovery PASS; annotator measurement "API unavailable" | reproduce RESULTS within seed variance | acceptable at 1 seed; commands depth 2 two sd low; 5-seed GPU round trip PLANNED (command in README); spec section 6 never received (TODO markers) |
| §6 P1 real text | — | — | waits for the team's data |

## B. Effect-size series recorded (judgment E2 decision rule)
One-quantifier scope at the unseen depth: no signal 0.21-0.33 -> chain-carried 0.63-0.67 -> beacon-relative 0.998 (C2b) -> 0.992 (C2c).
Two-quantifier scope at the unseen depth: 0.33-0.56 (9c-bis..C2) -> 0.48 (C2b, conditional term) -> **0.966** (C2c, two unconditional terms).
The regularity "seen at depth d in training (share p), tested at depth d+1" now holds for every operator case tried: reporting, negation,
modality, conditional, modal-in-conditional, negation on commands, one- and two-quantifier scope. No named exception remains.

## C. HYPOTHESIS
- H-b (wide/narrow is a relation between the two determiners): RETIRED — true as a statement about the legend, false as a requirement on the
  reader: the legend computes the relation from two category-local terms (C2c).
- N5: the kind should be resolved BEFORE the drawing, from the first open emitter's preference, rather than drawn half-way (the half-way
  drawing halves the noun term). Not built; for judgment.

## D. PLANNED, in the indications' order
B5-B7 (order collisions: permutation binding, level-indexed roles; sparse block codes; fourth address code); C5 (occupancy marker in the
verifier's reader world, one retraining); N6 (possible now), N7 (after N1); P0 5-seed GPU round trip; P1 (team data); N3, N1, N2; B8-B9
(HYPOTHESIS); N4; LLM paraphrases / formalizations when data arrives; the annotator measurement (needs the API).

## E. Questions for the reviewing session
1. The indications cite JUDGMENT_2026-10-16_013c490.md; it was never received. Should it be sent so it can be saved verbatim?
2. C2c met the criterion: confirm the removal of the exception row and the retirement of H-b, and that the C2c world (slot target, learned
   gates) is the quantifier reference world. Costs to accept or to work on: determiners 0.950 at depth 0, FULL 0.60 at depth 2.
3. N5 not met with the superposition as defined. Record as negative and stop, or test "resolve before drawing" (C above) with the same
   criterion?
4. 4c: the lexical-mismatch confident-wrong grows with data (0.07 -> 0.17). Is C5 still the next verifier step, or a mismatch-aware
   abstention (name-in-question as a feature of tau) first? C5 is what the indications say; it stays first unless told otherwise.
5. Tool: spec section 6 (splits, skill steps, second look) was cut from the message; the agent left TODO markers. Send section 6?
   The v1 export emits no quantifier scope term — should the exporter emit the C2c terms (new world version) before the 5-seed round trip?
6. The 50-case audit was labelled by me, not by the team (stated in 10c-quater-bis). Does the team want to re-label?
