# Judgment at 934fbae — N6 (RESULTS 15); answers to questions 1–3; one rule; the agent's item before P1
(received from the reviewing session on 2026-10-17; saved verbatim)

Scope: commits 752853c..934fbae. Everything earlier stays in force. Only what changes is here.

Verdict on the period: N6 run exactly as released, criterion and decision rule before the run, three added axes stated before the run,
the third branch of the rule applied, the hypothesis for the inverted share marked as a hypothesis. Quality: high. The result is more useful
than a pass would have been: it says what a missing-question detector must look at, and it corrects a sentence that the book and the
positioning document both carry. One correction is mine (section C).

## A. The recording (Q1)

Agreed, with the two conditions attached to the two numbers, because they answer two different questions:

- *Is a missing question DETECTABLE without any label?* Yes, by direction: the residuals of sentences that share the hidden filler point the
  same way (cosine 0.186 same filler vs 0.052 different, five seeds), and a leave-one-out nearest neighbour over the residuals alone names the
  filler at 0.570 against chance 0.083. This is the number for the positioning document, stated as "the unexplained part of the chart clusters
  by the hidden filler with no vector and no label (1-NN 0.57 vs chance 0.08; same/different cosine 0.19/0.05)". It is a lower bound on
  separability, not a recovery rate.
- *Once the question is NAMED, are its fillers recoverable from the residual?* Yes, at the reader's own address, 0.963 (5/5 seeds > 0.92),
  with 120 gold sentences to calibrate the address. This is the acceptance step of the legend loop (the question is proposed, labelled on a
  sample, and the fillers come back), not detection; it goes in the row with its 120-sentence condition and in the tool's legend-loop
  description, not in the one-line positioning claim.
- The residual SHARE does not detect the missing question in this world (gap −0.267); this goes in the row as the negative it is, and into
  the books on my side (section D).
Row wording: "N6 — NOT met as written (share −0.267; oracle 0.110); the missing question is detectable by residual DIRECTION without labels
(same/different cosine 0.19/0.05, 1-NN 0.57 vs 0.08) and its fillers are recoverable at the reader's own address once named (0.963 with 120
labelled sentences)".

## B. The per-term decomposition (Q2) — yes, one re-scoring, with its rule

Run it on the saved N6 models: the residual norm split by token class (content words of the hidden role; other content words; function words
'gives', 'to', 'is', 'by', '.'; the marker-bearing terms), for sentences with and without the recipient. Decision rule, before the run: if the
function-word terms carry the majority of the residual norm, the tool's `_residual` becomes TWO numbers — the share on content-word terms and
the share on function-word terms (a reporting change, nothing in the reader) — and the N6 gap criterion is re-scored on the content-word
share alone; if that re-scored gap is > 0.3, the share detector is recorded as "works on the content share"; if not, share stays a recorded
non-detector and direction is the detector. If the function words do not carry the majority, the hypothesis is dropped and the row stays as
in A. Either way, direction is what the tool's legend loop uses; the decomposition decides only whether the share number is worth printing.

## C. The oracle criterion was mine, and it was ill-posed (rule 30)

The addendum's criterion — recover the hidden filler with the world's true `R_C` — asked the reader to place a filler at a vector it had never
seen in any target. By construction the keys learn to match targets; with no target for the recipient, the recipient token lands wherever the
category keys put it: consistent across sentences (hence 0.963 at the reader's own address) and unrelated to `R_C` (hence 0.110, chance). A
criterion that cannot be reached by a reader that satisfies the training objective measures the criterion, not the reader. Rule 30: "Before a
run, the criterion is checked for attainability by construction: if meeting it requires information the reader never receives (a vector, a
label, a gate it was not given), the criterion is rewritten before the run and the reason recorded." The N6 row cites this rule as the
reason the oracle axis is kept as a record and not as a failure of the reader.

## D. The books (my side)

Three places say that a residual much above the world's background means "the sentence contains something the legend does not ask" (chapter 8
on the remainder, step 6 of chapter 13, chapter 12 on novelty), and chapter 17 builds on "structured remainder". N6 measured that the SHARE
can go the other way when function words dominate the unexplained part. The books get the measured statement: the remainder's *direction*
across sentences is the detector (unexplained stars that cluster at one address), the share alone is not, and the per-term decomposition
of B will say whether a content-word share is. Done in this round, both books.

## E. The agent before P1 (Q3)

Two items, then it waits for the data:
1. The legend-gap report in the tool: test 5b already clusters `_unplaced` words by residual direction; make it print the two N6 statistics
   (same/different cosine of the residuals in a cluster; leave-one-out nearest-neighbour accuracy of the cluster's fillers) so that a
   proposed legend line comes with the number that supports it. Reporting, no model change.
2. A rehearsal of P1 on real text that needs no key: WikiSQL through the tool end to end — the questions as the world's text, the gold frames
   from the SQL (the dataset's own structure, 6.4's second source), a hand-written legend (select, aggregate, conditions as a SET value, per
   judgment d8f8a98 Q1), `sop train` and the formalize path. Criterion before the run: the package reproduces RESULTS 8c/8d within seed
   variance (selected column, condition column, full query) and the frames round-trip through SOP Lang. It exercises everything P1 will
   use except the annotator, on real sentences, with a legend that uses the set value for conditions — which also answers, on data, the
   C5 legend question. If it reproduces, P1 starts with a tested path; if it does not, the gap is found before the team's data arrives.
The annotation path still needs a key wherever the team runs it; nothing changes there.

## F. Order from here
1. One commit: the N6 row (A), rule 30 (C), the per-term decomposition (B) with its rule.
2. Round 6 of the tool (the §2 re-scoring and the fitted calibration rule) — already on the GPU.
3. E1 (legend-gap report), then E2 (WikiSQL through the tool).
4. N8 (low priority); P1 when the data arrives; N7 after N1; B8–B9; capacity at D = 4,096–16,384.
