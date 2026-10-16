# Hand-over at 78d36e8 — instructions, the missing judgment, the complete specification (one file, three parts)
(received from the reviewing session on 2026-10-16, saved verbatim: this file holds the preamble and PART 1; PART 2 is saved as
experiment/JUDGMENT_2026-10-16_013c490.md and PART 3 as tool/SPEC_worlds_server_sop_cli.md, as the hand-over asks)

This file has three parts, to be read in this order: PART 1 — the instructions at 78d36e8 (what to do now); PART 2 — the judgment at
013c490, which was never received and which the applied indications referred to (save it verbatim as experiment/JUDGMENT_2026-10-16_013c490.md);
PART 3 — the complete tool specification (save it verbatim as tool/SPEC_worlds_server_sop_cli.md, replacing the partial copy; section 6 was
missing there, and this version adds the concepts page, `_scope=`, the multi-world dispatch and the corrected sizes). Where the tool's code
already took a decision compatible with the specification, do not redo it; note it in the conformance table.

# PART 1 — Instructions at 78d36e8

## 0. Check against the plan: done, in order, with the rules kept
E1b (trace (iii), then the run: MET, depth 3 0.968, passives 1.000, new reference table); C2c (trace, then the run: MET, two-quantifier scope
0.966 at the unseen depth, 5/5 seeds; one-quantifier 0.992; false-both 0.000; both on ambiguous 0.993); N5 (NOT met, recorded as negative, no
remedy built without judgment — correct); 4a audit (50 cases, labeler stated); 4c data curve with the lexical-mismatch split at every point;
P0 (tool built by a separate agent in a worktree, merged, six tests, round trip at one seed; nothing under experiment/ touched); the withdrawn
sentence withdrawn; STATUS rows and history rows kept apart. Two things could not be done because they were not received (0a, 0b above); one
small inconsistency and three process points in section C. Quality of the period: the best so far — two declared exceptions removed in one
evening by traces, with the decision rules written first.

Standing: the composition regularity "seen at depth d in training (share p), tested at depth d+1" now holds for EVERY operator case tried,
with no named exception, on the reference world with the predicate role and the given relation-marker gates. Say it that way, once, in
STATUS section 1 and in PAPER_NOTE; keep the history rows.

## A. Answers

Q1. Yes — the judgment file is attached to this hand-over; save it verbatim next to the indications. Nothing in it contradicts what was run.

Q2. Confirmed on all three: the exception row is removed; H-b is retired as "a relation the legend computes from two category-local terms";
the C2c world (slot target, learned gates) is the quantifier reference world. On the costs: the determiner IDENTITY is not a question of the
legend (the legend asks which phrase is wide, not which word was used), so FULL on the quantifier world is defined over the legend's questions
— nouns, presence, scope-or-both — and the determiner identity is reported as its own axis, outside FULL (rule 16: say that the 9c-* FULL
columns change definition from here and print both the old and the new FULL once). The three-terms-in-one-slot competition (0.950) stays a
reported property; no remedy.

Q3. N5: recorded as negative, as you did. Do NOT build "resolve before drawing" now. The measurement already says what the tool needs: the
word's own kind barely matters because the slot is reached through the emitters around it (variant (a) 0.989); so the practical rule is
(a) majority kind, with the two kind hypotheses tested at readout only when the margin is low, and "both" reported then. "Resolve before
drawing" stays HYPOTHESIS with the diagnostic you named (does the first open emitter develop a preference?) printed in the next N-run that
touches kinds, not as a run of its own. The positioning document's paragraph on kind ambiguity is adjusted on my side to match (a) — the
document, not the repository, carries that change.

Q4. C5 stays first on the verifier, as planned (conditions are the weak slot and independent of the mismatch finding). Right after it, one
re-scoring, no retraining: mismatch-aware abstention. The feature available at verification time is not "gold column name in the question"
(gold is unknown) but "the name of the column the reader CHOSE appears in the question" (the schema is known). Abstain when margin < tau OR
the chosen name is absent; tau set on the calibration half, the curve on the report half; report acc-vs-coverage and the confident-wrong
rate against the margin-only rule. It is calibration reporting (rule 18), not tuning, and it tests whether the 0.169 can be turned into
abstentions. Criterion: confident-wrong on sel < 2% at the coverage that results, coverage reported.

Q5. Section 6 is in the attached specification. Yes: the exporter emits the C2c terms before the 5-seed round trip — the tool's quantifier
world becomes v2 = the C2c world (R_wide / R_q category-local terms, the readout rule in the legend), with its own hash; v1 (no scope) stays
under versions/. SOP Lang gets one reserved line for it, now that scope composes: `_scope=<role>` on the event wire = the role whose quantifier
is wide (`_scope=both` when the legend's rule says both; absent with zero or one quantifier, where `det=` on the entity says which word). Put
it in tool/SPEC as an amendment and in `sop check`. Then the 5-seed GPU round trip, on all three worlds, with the runs committed under
tool/runs/ (a number without a log is not a result — the P0 table's runs are not in the repository).

Q6. The audit labelled by the session is acceptable because it is stated. If the team wants more, a second labeler (another LLM, independent
prompt) labels the same 50 and the agreement is reported; not blocking C5.

## B. One book-facing note (for the record, no action in the repository)
With E1b and C2c, every sentence of the positioning document is covered or exceeded; the book's chapter on kind ambiguity is softened to
the measured mechanism (slot reached through emitters; both kinds reported when the margin is low). STATUS.md remains the only place with
numbers.

## C. Process points (small)
C1. STATUS.md row "Calibration on the reference world" still says "NOT met at depth 3 (coverage 0.3-0.7)"; with 10b-quinquies the depth-3
    coverage is 0.93 / 0.88 at confident-wrong 0.000. Update the row (criterion: confident-wrong < 2% at coverage > 80%: MET at depth 3 now;
    unseen transitions still 0.74-0.84 coverage — say which).
C2. P0 round-trip runs were "not committed". Commit them (tool/runs/ or experiment/logs/ with the hash in the name) or re-run them committed
    before the 5-seed round trip; the P0 row in STATUS cites them.
C3. The tool reads the reporting depth and the negation FROM the state while the experiment scripts are GIVEN the depth; the commands depth-2
    gap (0.790 vs 0.854) may be depth estimation, not packaging. Report FULL with the depth given AND with the depth estimated, side by side,
    in the 5-seed round trip; frame_roundtrip (9a) had depth estimation at 1.000 on the reference world, so a gap here is informative.
C4. Model sizes 12-15 MB (reference, quantifiers) against the spec's "a few MB": state the breakdown (per-category-pair relative keys dominate)
    in tool/README; the spec's sentence is corrected on my side to "a few to ~15 MB at D = 1024". B6 (sparse block codes) is the place where
    this shrinks, if it does.

## D. Order from here
1. C1-C2 (minutes).  2. B5 — order collisions, as prepared (permutation binding; level-indexed roles; the dual criterion: order pairs separated
gap > 0.5 AND unseen-depth composition retained > 0.95 on the new reference table 10b-quinquies).  3. C5, then the mismatch-aware abstention
re-scoring (Q4).  4. Exporter v2 (C2c terms, `_scope`), 5-seed GPU round trip with given vs estimated depth (Q5, C3), runs committed.
5. B6, B7.  6. N6 (the tool's test 5b is the offline analogue; the experiment version with the oracle vector per the addendum).  7. P1 when
the team's data arrives; the annotator measurement when the API is available.
No time estimates; criteria and logs. Documents: the book changes on my side only (kind ambiguity); STATUS.md section 1 gets the one sentence
of section 0 above.
