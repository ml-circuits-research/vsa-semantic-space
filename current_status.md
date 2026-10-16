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
| P0 follow-up (agent, merged) | section 6 implemented (LLM steps report "API unavailable"); quantifier world v2 = C2c with `_scope` in exporter / check / formalize (v1 kept under versions/); FULL with depth estimated vs given vs given+negation; model sizes (per-category-pair keys 96-98%: 11.9 / 15.0 / 3.0 MB); p0_full_gpu.sh prepared with the lock | 6 tests PASS on main | 5-seed GPU round trip starts now (GPU free), runs committed under tool/runs/ |

## B. B5c — what the trace must show, and the concern stated before it (instruction after the stop, step 2)
World: op(X) = R_op (x) f_op + R_lvl (x) X for sentential operators (not, must, may, says); 'if' keeps its two roles; word-level negation unchanged;
gates learned, nothing given. Trace (rule 11) on "A must not give B to C" / "A does not have to give B to C" at depth 0 and inside one report:
(a) the operator word's star lands at R_op at its own level; (b) the words after it carry the extra R_lvl.
Concern, stated before the trace: for the words AFTER both operators (a) and (b) can hold with the existing pair keys (hop into the operator
word = R_op at its level; hop out of it = R_lvl, as after 'that'). For the SUBJECT, which precedes the two stacked operators in English, the
chain gives one left re-seating key EL[A <- first open operator]; that key cannot be R_lvl R_A in "A must give" and R_lvl^2 R_A in "A must not
give" at the same time — unless the first operator's own address differs between the two sentences, which it does not (it is first). If the
trace shows this, B5c stops per the instruction and the report says: operators that follow their subject are, for the subject, the
right-announced case inside one clause. The decision is the trace's, not mine.

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
5. B5c (B above): if the trace shows the subject problem, the instruction says STOP. Confirm that "operators after their subject" would then
   be recorded under the declared limit (chapter 8), with B5.0's list as its measured extent, and B5c not run.
