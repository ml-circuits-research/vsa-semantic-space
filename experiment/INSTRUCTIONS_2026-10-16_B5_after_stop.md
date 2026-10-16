# Instruction after the B5 stop (replaces every earlier B5 text)
(received from the reviewing session on 2026-10-16, after the B5 stop reported at commit a472c4c; saved verbatim)

B5 is not stopped; it changes shape. The permutation route is dropped (your objection stands: a soft gate cannot apply a discrete permutation
a variable number of times; level-indexed roles fail the unseen depth by construction). Nothing given is added to the reader.

Step 1 — B5.0, measure the collision as it is (no training). Enumerate every sentence pair in the E1 / E3 / B3 worlds that differs only in
operator order. For each pair: cos between the world states, cos between the reader charts, and whether the two decode to different frames.
Split into "separated by a witness term" (speaker, predicate, a filler under only one operator) and "colliding" (same frame). Report both lists
with counts and the bag baseline. Rule 11 trace on one colliding pair ("must not P" / "not must P"): the two states term by term, marking the
terms that coincide. Push with current_status.md rewritten, so the collision list is on record before anything is built.

Step 2 — B5c, sentential operators as levels with an identity term (the reporting pattern applied to operators). World correction, rule 16:
   op(X) = R_op (x) f_op + R_lvl (x) X
R_op = one new named star "which operator" (like R_pred); f_op = the operator word's own star; R_lvl = the level star already used for
reporting (R_rep). Applies to sentential operators only (not, must, may, says; 'if' keeps its two roles). Word-level operators unchanged
(negated attribute inside a slot). Gates LEARNED; the level count arises from the gates as it does for 'that'. The regime is restated in
levels: depth = reports + sentential operators above the word; "seen at depth d (share p), tested at d+1", shares printed.
   Rule 11 trace BEFORE the run, on "must not P" / "not must P" at depth 0 and inside one report, must show both: (a) the operator word's star
   lands at R_op at its own level (a learned displacement of its category, like the verb's at R_pred); (b) the words after it carry the extra
   R_lvl (like after 'that'). If (a) and (b) cannot both hold with the existing key structure, B5c is not implementable without a model
   change: STOP and report; do not improvise.
   Criterion (the cosine-gap criterion does NOT apply: the pair shares the inner clause by design, so whole-state cosine stays high):
   (i) operator readout per level ("which operator at which level") > 0.95 at the seen depths AND at the unseen depth;
   (ii) the two sentences of a colliding pair receive different frames in > 0.95 of pairs;
   (iii) unseen-depth composition of the roles retained > 0.95 on the re-run E1 / E3 / B3 tables;
   (iv) presence and scope of every operator unchanged (> 0.95 where it was).
   Whole-state cosine of the pair reported with the bag, not judged. Control: level-indexed roles (expected: separates, fails the unseen depth).
   Decision rule: all four met -> B5c is the operator definition from now on and the book sentence changes on my side ("sentential operators
   open a level and leave their star at it"); (i)-(ii) met but (iii) not -> report the trade-off, limit stays declared, no third remedy;
   not met -> the limit stays as in chapter 8 with B5.0's list as its measured extent.

Everything else in your status is agreed: Q2 re-scoring as 9c-septies (both FULLs printed once side by side); C5 then the mismatch-aware
abstention; the Opus agent's tasks as listed, and the 5-seed GPU round trip starts as soon as exporter v2 is ready, no further decision
needed, runs committed under tool/runs/; add rule 27 (criterion on the mean; per-seed list printed; "5/5 seeds" when every seed is required).

Order: C5 -> mismatch re-scoring -> B5.0 (push) -> B5c trace -> B5c run + control -> B6, B7 -> N6 -> P1 when the data arrives.
