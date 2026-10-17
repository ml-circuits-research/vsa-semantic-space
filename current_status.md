# Current status (rewritten whole at every judgment point; last: 2026-10-17 morning — judgment e9274a9 applied: C, B5.1 (two runs), C5 re-scoring, tool round 2; round 3 prepared)

Labels (rule 22): BUILT / PLANNED / HYPOTHESIS. Sources: experiment/JUDGMENT_2026-10-16_e9274a9.md (saved verbatim), experiment/RESULTS.md 10c-octies,
11-ter, 12-ter, 12-quater; logs/trace_b51.txt; STATUS.md; experiment/JOURNAL.md; tool/README.md; tool/runs/round2_815e5f9/isolation_table.md.

## A. Done since e9274a9, in the instructed order (all BUILT unless stated)
| step | result | criterion / decision rule (written before the run) | outcome |
|---|---|---|---|
| C (one commit) | rule 28 (address check before a new target; one change per run); STATUS row 26 scope sentence; the two limits as a BUILT row; verifier operating points (tau 0.05: coverage 0.757 / confident-wrong 0.053; tau 0.15: 0.445 / 0.009, covered accuracy 0.980); B5c WITHDRAWN | — | done |
| **B5.1, first run** (log 2026-10-17_0742; RESULTS 12-ter) | depths 0-1: subject 1.000 under one and two stacked operators, structure 1.000; depth 2: subject 0.07-0.18 for EVERY form incl. the affirmative, B / C / pred / structure read | measurement; decision rule on the subject | seen depths: the subject does NOT fail; depth 2: a REGIME ERROR of mine — no depth-2 sentence in training, so depth 2 was also an unseen TRANSITION ('that' -> 'that'; rule 4); kept as the record, re-run with the E1b protocol |
| trace after the run (logs/trace_b51.txt) | learned gates: modal 0.71-0.83 partly open -> the subject's address is a mixture; for "does not" / "does not have to" the two subject addresses are IDENTICAL (cos 1.000) and the subject is still read at 1.000 under both gold keys (cos with the gold key 0.26-0.35, margin 0.18-0.29, no stacked-operator cost); per code at depth 2: the chain carries the subject at depth 1 (0.865) and nothing at 2 (0.009); the path code carries B, C, never the subject | "report how" | the limit (ii) is on the ADDRESS; role accuracy does not see it because the identity cleanup tolerates a blended address and the structure is carried by the words after the operators |
| **B5.1, E1b protocol** (--p_aff2 0.3; log _0840; RESULTS 12-quater) | depth 2 UNSEEN for the operator forms: structure 1.000, B / C / pred 1.000, subject 1.000 (0 ops) / 0.988 (1 op) / **0.948** (2 ops); may_neg 0.855±0.114 the one soft cell; colliding pair frames differ 0.000 / 0.000 / 0.007 | the judgment's decision rule | **the subject does not fail -> B5d DROPPED**; subject + modal + negation joins the composition regularity (seen at depths 0-1, tested at 2) with one soft spot named |
| C5 re-scoring (log _0757; RESULTS 10c-octies) | marker's margin as the condition verdict statistic: drop / add detection 0.15 -> 0.75 / 0.76, unverifiable 0.83 -> 0.23, FALSE ALARM on gold 0.15 -> 0.43, precision 0.943 -> 0.899 | report, no tuning | a one-for-one trade of abstention for false alarms; C5 stays NOT met (present 0.890); the limit is the presence reading of the later slots |
| Tool round 2 (agent; RESULTS 11-ter; tool/runs/round2_815e5f9/) | `gates_given` in legend.json; depth from the gate count; one change per run: A gates given -> depth 3 **0.987**, seen 1.000, unseen transitions **0.428** (v16 0.627); B candidates -> new ids only; C identities -> nothing; commands gate-count depth right on 4/5 seeds, seed 3's 'that' gate 0.24 breaks it | one change per run (C2) | both reference gaps closed by the given gates; one NEW gap (unseen transitions) not isolated; the gate-count estimator depends on a learned gate |
| Tool round 3 (agent; RESULTS 11-quater; tool/runs/round3_1d660ee/) | D: presence NOT decided -> unseen transitions 0.428 -> **0.596** (v16 0.627), recipient 0.619 -> 0.839; the retrain of A is identical at every digit (deterministic); E: the opener-KIND count gets the depth right on 5/5 seeds (the gate count failed on the seed whose 'that' gate learned 0.24), FULL 0.857 = depth given = 10d | one change per run | both gaps explained; the tool reproduces 10b-quinquies and 10d on every axis; defaults: depth = opener kinds; presence decided kept (question 5) |

## B. HYPOTHESIS
- Verifier conditions: the presence reading of the second and later condition slots (0.890) bounds the condition axis; neither the column cosine nor
  the marker's margin as the verdict statistic changes that (10c-sexies, 10c-octies).
- N5 "resolve before drawing": HYPOTHESIS, diagnostic to print in the next kinds run.

## C. PLANNED, in the instructed order
B6 sparse block codes; B7 fourth address code remeasured; tool round 3 (D, E) then the remaining tool gaps; N6; P1 when the team's data arrives
(the annotation path tested once with a key, wherever the team runs it); N7 after N1; B8-B9 (HYPOTHESIS).

## D. Questions for the reviewing session
1. B5.1: the decision rule's second branch applied (the subject does not fail; B5d dropped). The limit (ii) stands on the ADDRESS (12-bis;
   identical addresses for "does not" / "does not have to" even with the learned gates) and is invisible in role accuracy because the identity
   cleanup reads a blended address and the structure is carried by the other roles. Is that the final recording — "a limit of the address, not of
   the frame readout on these forms" — or should a construction where the subject ALONE carries an operator be built to expose it (none exists in
   the repository's worlds)?
2. The first B5.1 run's depth-2 failure came from my regime error (no depth-2 sentence; the 'that' -> 'that' transition unseen). It is kept as
   12-ter with the cause stated. Should rule 4 get the sentence "an unseen depth must not also be an unseen transition: the test depth minus one
   is in training for the affirmatives" explicitly (the E1b protocol as a rule)?
3. may_neg at the unseen depth 0.855±0.114 is the one soft cell of B5.1; per-seed values are not in the log (the script prints mean ± sd per
   form). Add the per-seed list to v20's printout for the record, without re-running?
4. C5: with both verdict statistics tried, the condition axis is bounded by the per-slot presence reading (0.890 on the present side). Close C5
   as "not met, bounded by presence", or try the marker with per-slot thresholds (second and later slots have fewer examples) — a re-scoring?
5. Tool: round 3 closed the last gap (presence decision: unseen transitions 0.596 vs v16's 0.627 with every role read). The tool's default
   readout still DECIDES presence (the spec's `_empty` and the occupancy marker depend on it), which costs 0.17 on unseen formulations where the
   recipient's presence ratio falls under the training-row threshold. Keep the decided readout as the product default (and report both), or make
   "every role read" the default for worlds whose legend has no optional role? Depth estimator default switched to the opener-kind count (no
   decision needed: it is right on 5/5 seeds where the gate count was right on 4/5).
