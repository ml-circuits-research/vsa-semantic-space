# Current status (rewritten whole at every judgment point; last: 2026-10-16 afternoon — judgment at a8a8fab applied: process points C, queue started)

Labels (rule 22): BUILT / PLANNED / HYPOTHESIS. Sources: experiment/JUDGMENT_2026-10-16_a8a8fab.md (the judgment, saved verbatim),
experiment/RESULTS.md 9c-ter / 10a-10d (corrected cells), STATUS.md, experiment/JOURNAL.md.

## A. Judgment a8a8fab — what was applied in this commit (BUILT unless stated)
| point | applied |
|---|---|
| E1 | one hypothesis replaced by two: H-a (addressing: the scope term composes only if carried by the code the frame re-addresses) and H-b (relation, two-quantifier case only); "same class as right-announced" removed — two declared limits; wording "six operator cases; quantifier scope fails at the unseen depth in all three world definitions" in RESULTS / STATUS / PAPER_NOTE |
| E2 | degree-2 term WITHDRAWN (commutative binding: symmetric in the two nouns; information already degree 1). C2 = determiner gates GIVEN open (v14 --det_gates open, v8.fix_gates, rule 20 printed). Depth-2 rule-11 trace done BEFORE the run (trace_c2.py; logs/trace_c2.txt): with the E2c gates the wide noun's chain address is IDENTICAL for 'each' and 'every' (that:that + that:to + L[to:NP]); with the flag it contains the determiner's own category and both 'that' displacements (that:that + that:each + each:a + L[a:NP] vs ... every ...) — the flag puts the scope term into the re-addressed code, so it is a valid test of H-a. One- and two-quantifier scope reported separately |
| E3 | "emit both" adopted with the two-sided criterion + false-both rate; measured on the target C2 ends with; label stays observed/partial until met |
| E4 | 10b = reference table; depth-3 stated with both numbers everywhere; E4 (a) breakdown per template / per verb (with gate value) coded in v16; (b) --p_omit 0, (c) --verb_gates closed (given, rule 20), (d) --D 2048 queued; decision rule as in the judgment |
| E5 | C4/C5 before the LLM version; gold "confirms 0.31 / unverifiable 0.53 — a filter, not a judge" added to 10c and STATUS; v17 now saves the reader and the dev frames (C4 = re-scoring on the saved frames, tau set on the even half, curve reported on the odd half); C5 (condition marker readout) needs a world change for the verifier reader = one retraining, PLANNED after this queue; LLM input format: PLANNED (jsonl spec to be committed with C4) |
| E6 | queue.sh (strict sequence, waits on the pid lock only); run.sh pgrep excludes its own process tree; rule 26 written. Also: run.sh's dirty-tree check now ignores *.md (the hash identifies the CODE; documents are edited while a queue runs) |
| C1 | 9c-ter sentence corrected ("affects the affirmative case only" was false: slot d0 0.978 -> 0.850, d2 0.333 -> 0.033 under the relative rule); NO 'none' rule from now on: PRESENCE axis (threshold on calibration sentences, rng stated) + SCOPE conditional on presence (argmax / both) + FULL joint, one- and two-quantifier separate (v14 rewritten); every quantifier number carries its decoder rule in the cell; the E2c 5-seed models were NOT saved -> numbers kept with labels, not re-scored; all scripts now save final params |
| C2 | v16: occupancy threshold set on depth-0 TRAINING sentences (120 with C, 120 without), the 360 scored readouts untouched, same-rows midpoint printed for comparison; v17: condition presence threshold set on the EVEN half of dev, verdicts reported on the ODD half (old whole-dev numbers kept as 10c); rule 25 written. Both re-runs queued (10b-bis, 10c-bis; old and new side by side) |
| C3 | N5 / N6 / N7 rows added to STATUS.md and section D below WITHOUT criteria: the addendum text was not received (only the three names appear in the judgment). PLANNED, criteria to be supplied |
| C4 | STATUS cells fixed (quantifier row: decoder rule per number; verifier row: confirmed / unverifiable on gold; depth-3 row: both numbers) |

## B. Queue started after this commit (./queue.sh, strict sequence; logs carry the commit hash)
1. v16_reference.py --seeds 5 — 10b-bis: threshold on training rows + E4 (a) breakdown.
2. v17_verifier.py — 10c-bis: threshold on the calibration half, report half; reader + frames saved.
3. v16_reference.py --seeds 5 --p_omit 0 — E4 (b).
4. v16_reference.py --seeds 5 --verb_gates closed — E4 (c).
5. v16_reference.py --seeds 5 --D 2048 — E4 (d), capacity test.
6. v14_quantifiers.py --scope_target slot --det_gates open — C2, slot target.
7. v14_quantifiers.py --scope_target noun --det_gates open — C2, noun target.
Criteria (stated before the runs): E4 — gating if (b) or (c) restores depth 3 > 0.9 with the predicate role, capacity if only (d) does; C2 —
scope > 0.9 at depth 0 AND depth 2 (one- and two-quantifier separately), nouns 1.00 at depth 2, determiners reported, cost on 6c / 9b / 9d /
10d tables reported (those tables are not re-run; the cost is read on this world's own seen axes and, if C2 is adopted, by re-running them).
Then: C4 (re-scoring on the saved frames) + C5 (marker readout, one retraining) + LLM jsonl format; B5-B7; C3 on the world C2 leaves.

## C. BUILT since the previous judgment (unchanged numbers; see RESULTS) — for reference
E2c slot / noun targets (9c-ter, labels per cell); diag_e2c; A2 memory curve (10a); B1+B2 reference table (10b; depth 3 UNSEEN 0.813 at full
coverage, at tau 0.05 coverage 0.59 (A) / 0.33 (C) with covered accuracy 1.000 / 0.975, confident-wrong 0.000 / 0.008); B4 verifier (10c);
B3 commands (10d: scope 1.000, confident-wrong 0.000 at coverage 0.968, depth 2 unseen scope 0.997).

## D. PLANNED / HYPOTHESIS, in judgment order
C4 + C5 + LLM input format (PLANNED); permuted-filler relational term (PLANNED, conditional on C2's outcome); B5-B7; N3, N1, N2; N5 kind
ambiguity, N6 missing-legend residual, N7 world identification by fit (PLANNED, criteria not received); B8-B9 (HYPOTHESIS); N4; LLM paraphrases
for B3 (PLANNED, no API).

## E. Questions for the reviewing session
1. N5-N7: please send the addendum text (criteria); the rows exist without them.
2. run.sh now ignores *.md in its dirty-tree check so that documents can be edited while a queue runs — acceptable, or must documents also be
   committed before every run?
3. C5 requires adding an occupancy marker to the verifier reader's world (one retraining of the WikiSQL reader): confirm, or restrict C5 to a
   readout on the existing state (no marker — then "condition count" is read from the condition slots' cosines only).
