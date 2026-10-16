# Current status (rewritten whole at every judgment point; last: 2026-10-16 night — hand-over 78d36e8 applied; C5, mismatch-aware abstention, B5.0 done; B5c next, trace first)

Labels (rule 22): BUILT / PLANNED / HYPOTHESIS. Sources: experiment/INSTRUCTIONS_2026-10-16_78d36e8.md, experiment/JUDGMENT_2026-10-16_013c490.md,
experiment/INSTRUCTIONS_2026-10-16_B5_after_stop.md (all saved verbatim), experiment/RESULTS.md 9c-septies, 10c-sexies, 10c-septies, 11, 12;
STATUS.md; experiment/JOURNAL.md; tool/README.md.

## A. Done since 78d36e8, in the instructed order (all BUILT unless stated)
| step | result | criterion / decision rule (written before the run) | outcome |
|---|---|---|---|
| C1 | calibration row updated: depth 3 MET with the given relation-marker gates (coverage 0.93 / 0.88, confident-wrong 0.000); unseen transitions NOT met (C coverage 0.74) | — | done |
| C2 | P0 CPU runs committed (tool/runs/p0_cpu_1seed_60f284d/: logs, metrics, splits, commands, model sizes; code f759d93, quantifier v1) | — | done |
| Q1 | the missing judgment saved verbatim; the complete spec (sections 0-10) in tool/ and docs/specs/ | — | done |
| **Q2** FULL redefined over the legend's questions (9c-septies, log 2026-10-16_2125, saved C2c models) | FULL-new d0 0.995, d1 0.996, **d2 UNSEEN 0.967** (0.978, 0.922, 0.978, 0.967, 0.989), strict 0.864, new nouns 0.998, ambiguous (nouns + presence + both) 0.990; FULL-old 0.887 / 0.834 / 0.603 / 0.536 / 0.856; determiner identity own axis 0.890 per sentence | both FULLs once, side by side (rule 16) | done; the old gap was the determiner identity |
| Q3 | N5 row: (a) majority kind is the practical rule, two kind hypotheses at readout when the margin is low; "resolve before drawing" HYPOTHESIS, its diagnostic in the next kinds run | — | done |
| **C5** occupancy marker on the condition slots (10c-sexies, log _2126, one retraining) | present read as present **0.890** (n 5,784), absent 0.992 (n 11,056); marker separation 0.374 vs 0.015 (column cosine was 0.158 vs 0.043); drop_cond / add_cond detection 0.150 / 0.153 (10c-bis 0.159 / 0.162); confident-wrong on drop_cond 0.151 -> 0.025; other role errors within 0.03 | present and absent > 0.9; detection before / after | **NOT met** on the present side (0.890); detection unchanged because the verdict abstains on the column-identity margin (unverifiable 0.83), not on the marker's — re-scoring proposed (E, 2) |
| **Q4** mismatch-aware abstention (10c-septies, log _2235, re-scoring) | margin-only: tau 0.15 (calibration half) -> coverage 0.445, covered accuracy 0.980, confident-wrong 0.009; rule M (chosen name in question): tau 0.10 -> 0.561 / 0.961 / 0.022; at equal tau 0.15 M gives 0.424 / 0.009; NOT-in-question rows: confident-wrong 0.169 -> 0.143 with coverage 0.466 -> 0.143 | confident-wrong < 2% at the resulting coverage | margin-only MET (coverage 0.445); rule M NOT met; the feature adds nothing: the confident wrong choice on mismatched questions is mostly a column whose name IS in the question |
| **B5.0** collision measured as it is (RESULTS 12, log _2236) | 0 rendered sentence pairs differing only in operator order in E1 / E3 / B3; algebra: colliding must/neg and must/may (identical state, no witness), witnessed rep/neg, rep/may, rep/if-then, if-then/may (speaker or a clause under one operator; cos 0.53-0.80); reader charts (seed 0 retrained, stated): rendered order read 1.000 on all three; reversed never-trained order 0.980 (E3, 'perhaps' is a known emitter) / 0.000 (E1, B3: 'never' before 'says' does not move the speaker's star); bags 1.00 / 0.89 / 0.86 | enumeration + traces, no criterion | on record; the measured extent of the chapter-8 limit |
| rule 27 | criterion on the mean, per-seed list, "5/5 seeds" | — | added; AGENTS.md rules 1-27 |
| P0 follow-up (agent, merged) | section 6 implemented (LLM steps report "API unavailable"); quantifier world v2 = C2c with `_scope` in exporter / check / formalize (v1 kept under versions/); FULL with depth estimated vs given vs given+negation; model sizes (per-category-pair keys 96-98%: 11.9 / 15.0 / 3.0 MB); p0_full_gpu.sh prepared with the lock | 6 tests PASS on main | 5-seed GPU round trip DONE (RESULTS 11-bis, tool/runs/p0_gpu_5seed_c843f53/): commands depth 2 0.857 with the depth given (10d 0.854) vs 0.590 estimated — C3 answered: the depth estimator, not the packaging; quantifier v2 reproduces C2c on scope (0.917 vs 0.966 at the unseen depth, both 0.993); reference depth 3 0.742 vs 0.813 (three stated differences, not isolated) |

## B. B5c — the rule-11 trace ran; the result is a STOP (RESULTS 12-bis, logs/trace_b5c.txt)
Method independent of key values and training: with the real reader, random parameters (3 draws) and hard gates, the address of every token is
compared across sentence pairs; two cases with different B5c targets must have different addresses. Result: (a) the operator stars and (b) the
words after the operators get distinct addresses wherever their targets differ — reachable. The SUBJECT that precedes two stacked operators
("p0 must give" vs "p0 must not give", also inside a report) has the IDENTICAL address in both sentences in every draw (cos 1.000) while the
targets differ (cos −0.08): a closed word's address depends on the words before it and on the first open emitter after it only, so it carries
exactly one operator level. With soft gates the two would differ only through the leak past a partially closed operator gate — not a level.
Per the instruction: B5c as specified is not implementable without a model change -> STOP, nothing built. Operator-initial forms satisfy the
condition throughout (which is why every operator so far composed: they all precede what they scope over).

## C. HYPOTHESIS
- Order of two stacked wrapping operators: not representable by commutative binding (B5.0: exact collision); B5c is the candidate (PLANNED).
- N5 "resolve before drawing" (first open emitter's preference): HYPOTHESIS, diagnostic to print in the next kinds run.
- Verifier conditions: the marker's own margin as the verdict's abstention statistic (re-scoring, no retraining) — proposed, not done.

## D. PLANNED, in the instructed order
B5c trace -> B5c run + level-indexed control (or STOP) -> B6 sparse block codes -> B7 fourth address code -> N6 -> P1 when the team's data
arrives; the annotator measurement when the API is available; the 5-seed GPU round trip (running next, agent); N7 after N1; B8-B9 (HYPOTHESIS).

## E. Questions for the reviewing session
1. B5.0's list: the two colliding pairs are not rendered by any grammar. Is a world that renders "A must not give" / "A does not have to give"
   (modal + negation in one clause) to be built for B5c, or should B5c be judged on the witnessed pairs plus the constructed colliding pair?
   The trace can run on the constructed pair either way.
2. C5: present 0.890 misses the criterion by 0.01, on the second and later condition slots; the verdict still abstains on the column margin.
   Re-score the saved marker frames with the marker's margin as the condition verdict's statistic (no retraining), before deciding C5?
3. Q4: the margin alone meets the 2% criterion at coverage 0.445; is that operating point to be written as the verifier's stated one, or is the
   verifier kept at tau 0.05 with its confident-wrong 0.053 and coverage 0.757 (the row as it stands)?
4. Section 6's LLM steps exist but have never run (no API); the tool's annotation path is therefore untested end to end. Does the team
   want a key on this machine, or will annotation happen elsewhere?
6. Tool after the round trip: fix the depth estimator at unseen depths (commands 0.590 -> 0.857 when given) and add the given relation-marker
   gates (to / by / from, 10b-quinquies) as a legend field so the reference world reaches its 0.968 through the package — both engineering,
   no new measurement; proceed without a decision?
5. B5c stopped at the trace (B above). Confirm the recording: "operators that follow their subject" under the declared limit (chapter 8), with
   B5.0's list as its measured extent; B5c not run. If a model change is to be considered at all, the trace names exactly what it would have to
   do: let a closed word be re-seated through ALL open emitters of its clause (a product of hops), not the first one only — that is a change
   of principle in the reader and is NOT proposed here, only named.
