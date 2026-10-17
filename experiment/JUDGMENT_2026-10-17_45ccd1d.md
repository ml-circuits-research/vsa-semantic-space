# Judgment at 45ccd1d — B5.1 (two runs, trace), C5 re-scoring, tool rounds 2–3; answers to questions 1–5; rule 29; the order from here
(received from the reviewing session on 2026-10-17; saved verbatim)

Scope: commits e9274a9..45ccd1d. Everything earlier stays in force. Only what changes is here.

Verdict on the period: every instruction applied in order, criterion or decision rule written before each run, one change per run on the
tool (C2 applied to the letter), and a regime error of the session's own caught by the session, named as such, both logs kept and the
re-run done under the protocol every earlier unseen-depth result had used. That is how a mistake should be handled; nothing to add.
Quality: high. Two wording corrections (section A) matter for honesty, because the recording is now the claim.

## A. B5.1 — the recording (Q1, Q3)

The decision rule's second branch applies: the subject does not fail on rendered sentences, B5d is not built. Two corrections to how it is
recorded:

1. "The one soft cell" is a form below the criterion. The regularity row (row 30) has the criterion "full frame at d+1 > 0.9"; may_neg at
   the unseen depth is 0.855±0.114. If subject + modal + negation joins that row, it joins with ONE NAMED EXCEPTION (may_neg), and the row's
   "no named exception since C2c" is no longer true. Write it so. Rule 27 requires the per-seed list for a cell on a criterion row; the log
   has only mean ± sd, so the readout of 12-quater is re-run with per-seed printing (the seeds are fixed in v20: rng 200+seed, 300+seed; if
   the GPU reproduces the digits, say so; if not, both are reported). That answers Q3: not from the log, from a re-run; it costs minutes.
2. The trace says more than "a blended address", and the recording should say it. In trace_b51 part (1), with the learned gates, the
   subject of "p0 must give" has cosine 0.451 with its own target and 0.806 with the target of "p0 must not give"; the subject of "p0 may
   not give" has 0.514 with its own and 0.762 with "p0 may give". So in two of the three pairs the subject's term, taken alone, votes for
   the OTHER member of the pair; for does-not / does-not-have-to the two addresses are identical (1.000). The frame is right because the
   structure is decided by B, C and the predicate, and the identity cleanup needs only a positive cosine with the gold key (0.26–0.35,
   margin 0.18–0.29, against 0.345 / 0.287 for the affirmative). Record: "limit (ii) is on the subject's address — identical for one pair,
   closer to the other member's target for two pairs — and invisible in the frame because the structure is carried by the roles after the
   operators; the measured cost is may_neg at the unseen depth." Where it bites: any readout that uses the subject's term as a structure
   witness (a per-role structure vote; a verifier that checks the subject's scope separately), and a calibration that needs the subject's
   margin. No world is built to expose it: there is no product need, and rule 28 names the identical addresses before any run.
   B5d: HYPOTHESIS, not DROPPED — "needed if a legend requires the subject's term as a structure witness"; dropped is the building decision
   for now. The same sentence goes into STATUS row 30, replacing "stacked post-subject operators are the declared limit".

## B. Rule 29 — unseen depth vs unseen hop (Q2)

Yes. Wording: "An 'unseen depth' claim is one more application of a hop that exists in training: the opener → opener transition at the test
depth minus one is in the training set with plain (affirmative) forms. Every unseen-depth cell states which of two things is unseen:
(A) the depth itself — no training sentence at that depth, the hop seen (reference world depth 3, 6c); (B) the operator form at a depth
seen with plain forms (the E1b protocol; 12-quater). An unseen hop is a different measurement — a chaining never performed is missing, not
blurred (6c-bis) — and is labelled as such, never as an unseen depth." Apply the (A)/(B) label once to the existing rows (6c, 10b-quinquies,
10d, 9c-sexies, 12-quater) in the STATUS row text; the numbers do not change, the labels do.

## C. C5 (Q4)

One more re-scoring, then close either way. Per-slot presence thresholds (slot 1, slot 2, slot 3+) set on the calibration half, every number
on the odd half, no retraining; criterion before the run: present ≥ 0.9 on the second and later slots AND false alarm on gold not above
10c-sexies's 0.152. Met → C5 met by calibration, say so. Not met → close as "NOT met, bounded by the presence reading of the later slots",
and the condition axis goes to the HYPOTHESIS list with the diagnostic that would decide it: present rate by slot index AND by the number of
conditions in the question. If the drop follows the number of conditions rather than the slot index, it is superposition (capacity), and the
parameter is the number of conditions, not the slot.

## D. Tool (Q5, and the depth estimator)

Presence DECIDED stays the product default. The product number for the reference world's unseen formulations is 0.428; 0.596 is the comparison
under v16's protocol (every role read, which the product cannot do: `_empty` is the product). Print both, the product's first, with the
protocol named on each. Do not make "every role read" the default for legends without an optional role: a role missing from the TEXT must
still read "not given", whatever the legend's rates. The finding to record is the one round 3 isolated: a presence threshold set on
training rows does not transfer to unseen formulations (recipient 0.619 vs 0.839). The next single change follows from it: set the presence
thresholds on calibration rows that include held-out FORMULATIONS (rule 25; the calibration split disjoint from the report split), one run,
nothing else changed; expectation written before: the recipient's presence recovers toward the every-role-read value without reading every
role. If it does not, presence calibration across formulations is a recorded limit of the reader, not of the tool.
Depth estimator default = opener-kind count: approved, no decision needed. It is a GIVEN (the legend's `level_openers`, rule 20): the README
says so in those words; the gate count and the fit stay as checks and are printed.

## E. Process points (small)

E1. "Soft cell" → "below criterion", everywhere. A criterion row carries the exception by name or it is not a criterion row.
E2. Row 30's scope sentence is stale after 12-quater (section A); the row cites 12-quater with the exception and the (A)/(B) label (section B).
E3. The run frames committed under tool/runs/ are the record and stay; keep them under runs/ with their index tables, never in docs/.
E4. N8, for the queue after B6/B7, not now: a toy verb-final world (every argument before the operators, negation/modal at the end — the
    German/Japanese case) is the construction where limit (ii) is not a corner but the rule for every sentence; measurement, decision rule
    stated before; it is the test B5d would need if it is ever built. PLANNED, low priority; written down so it is not lost.

## F. Order from here
1. One commit: the recording of A (STATUS row 30, B5.1 row, B5d → HYPOTHESIS), rule 29 with the labels of B, E1; the per-seed re-run of
   12-quater printed into the B5.1 row.
2. C5 per-slot re-scoring (C), then close.
3. Tool: presence thresholds on held-out formulations (D), one run; README sentence on the depth estimator as a given.
4. B6, B7 as planned.  5. N6.  6. P1 when the team's data arrives; the annotation path once with a key, wherever the team runs it.
Documents: chapter 8 on my side gets the measured statement of A (the subject under two operators is read; the cost is on its address and on
one form at the unseen depth); STATUS carries the exception by name.
