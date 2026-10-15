# Current status (rewritten at every judgment point; last: 2026-10-15 ~17:40, round 8d; one robustness check still running)

## Decisions applied since the round-8 judgment
- WikiSQL order: (a) 3 seeds + test split + FULL-query accuracy — DONE (8c). (b) tuned neural baselines on the SAME input at 1k/5k/56k and
  (d) hard SQL pairs with the lexical bag on the same row — DONE (8d, below); a longer-budget transformer check (4x, 10x epochs) is RUNNING. (c) aggregator:
  no dedicated head; reported as is, with the reason (lexical decision). Spider after (b). Capacity after WikiSQL.
- Principle note: full prose v1 written (experiment/PAPER_NOTE.md); WikiSQL excluded from it (one sentence in future work).
- Retraction of "emergent categories" propagated to ALL documents in one commit; EVALUATION_PRINCIPLES rule 16 added (a retraction is
  propagated everywhere in the same commit). Wording everywhere: categories are recoverable from distribution (skip-gram gives them equally
  well); the model consumes them as input, it does not produce them; what emerges from the objective are the gates (7c, 8b).
- No-leakage check in v11: skip-gram clusters trained on TRAIN questions + TRAIN headers only, asserted and logged. Dev/test words only get
  fixed random identities; words unseen in train get a random cluster (a handicap for the model, not a leak).

## Round 8 results so far (full tables in experiment/RESULTS.md ROUND 8)
8a. Controls (5 seeds): categories equally recoverable by skip-gram (1.000 = 1.000) => "emergent categories" RETRACTED as a contribution.
    Bottleneck associative map, any rank 8..1024: new ids ~0.24 (chance), unseen depth 0.25-0.39 => "generative with a fixed operation" stands.
8b/8c. WikiSQL, question only (no table at the input), D=512, 3 epochs, 3 seeds:
    | split | model | sel column | aggregator | cond-1 column | FULL query (sel+agg+all cond cols) | separation same/diff skeleton |
    |---|---|---|---|---|---|---|
    | dev | chance / majority | 0.166 | 0.715 | — | — | — |
    | dev | BoW + LR (mandatory baseline) | 0.497 | **0.870** | — | 0.437 (sel+agg only) | — |
    | dev | full model, 3 seeds | **0.810±0.004** | 0.814 | **0.632** | **0.376±0.003** | 0.287 |
    | test (15,878) | full model, 3 seeds | **0.811±0.004** | 0.815 | 0.629 | **0.380±0.002** | 0.275 |
    Dev and test agree to the third decimal. Header at the input HURTS (sel 0.67; table belongs in the decoder or bound to a role).
    Learned gates on real questions, stable across seeds: has, and, than, the, is, of (+ was, which, what) OPEN; content clusters closed.
    Criteria: sel > 0.7 without the table MET; agg > 0.85 NOT met (lexical); separation > 0.3 NOT met (0.28); FULL query 0.38 (state of the
    art ~0.9 with large pretrained models and the table as input) — reported as is.

## 8d. Same-input baselines, data-efficiency curve, hard pairs (v12; 3 seeds; full table in RESULTS 8d)
Identical input for ours and a small transformer (fixed identities + the same 90 categories, no table), identical objective and decoder;
transformer lr chosen once on a validation split held out from TRAIN; no extra tuning for ours. Skip-gram categories come from the WHOLE
train pool at every size, for both models => the claim reads "data efficiency with distributional categories from the whole unlabelled
training corpus". Dev (test at 56k in brackets):
| n | model | sel | agg | cond-1 | FULL query |
|---|---|---|---|---|---|
| 1k | ours / transformer / BoW+LR | **0.632** / 0.415 / 0.475 | 0.766 / **0.819** / 0.762 | 0.387 / 0.360 / — | **0.164** / 0.115 / — |
| 5k | ours / transformer / BoW+LR | **0.760** / 0.510 / 0.502 | 0.800 / **0.864** / 0.810 | 0.555 / 0.415 / — | **0.294** / 0.164 / — |
| 56k | ours / transformer / BoW+LR | **0.816** [0.821] / 0.543 [0.544] / 0.499 | 0.801 / **0.893** / 0.870 | 0.626 / 0.450 / — | **0.368** [0.374] / 0.196 [0.190] / — |
The same-input transformer does not reach us at any size on the selected column or the full query; it wins the aggregator (lexical) even
over BoW+LR. Pending: the same transformer with 4x and 10x the step budget at 56k (its 3-epoch run took 23 s), so that "does not reach us" is
not a budget artefact.
Hard pairs (dev, same table), the question "lexical or role-based?":
| representation | separation same/diff | AUC | H1 different words, same SQL (n=26; want high) | H2 same words, different SQL (n=1,353; want low) |
|---|---|---|---|---|
| bag of identities (lexical only) | 0.298 | 0.858 | 0.337 | 0.772 |
| ours (56k) | 0.291 | **0.914** | **0.462** | **0.622** |
| transformer, same input | 0.293 | 0.900 | 0.859 | 0.744 |
=> The OVERALL separation is lexical-level (0.29 vs 0.30): the headline equivalence metric does not test roles on WikiSQL, as foreseen (only
26 H1 pairs exist). On the hard pairs the role state moves both directions the right way relative to the lexicon (H1 +0.13, weak n; H2
-0.15, solid n), AUC 0.858 -> 0.914; the transformer stays lexical on H2. Kept as a negative with a nuance; metric unchanged.

## Conclusions relative to the goal (interim)
1. Both "emergent" claims are now controlled: gates survive (7c, and the same partition reappears on real text, 8c); categories do not (8a).
2. On real text with an executable consequence: selected column 0.82 vs 0.54 (same-input transformer) vs 0.50 (BoW+LR), dev = test,
   3 seeds, no table at the input, at every data size from 1k; full query 0.37 vs 0.19. The aggregator is lexical and the neural
   classifiers win it. The equivalence separation is lexical overall; the role state helps only on the hard pairs (H2 -0.15).
   => A short empirical paper ("data efficiency on the selected column without the table, with a readable role state", curve as main
   figure) is supported, pending the longer-budget transformer check. A full paper needs Spider (composition).
3. The principle note is in full prose with its two pillars, control, declared limit, trace and caveats; figure 1 still to draw.

## Questions for judgment
1. Short vs full empirical paper: the same-input transformer does NOT reach us even at 56k (0.54 vs 0.82) — stronger than the case
   anticipated in the judgment. Does that change the decision (short paper now; Spider for a full one later), or go straight to Spider?
2. The aggregator: the transformer beats BoW+LR on it (0.89 vs 0.87) and we are at 0.80. Report as is (decided) — but should the short paper
   state the full query (0.37 vs 0.19) as the headline, or the selected column (0.82 vs 0.54)?
3. PAPER_NOTE §7 rewritten as "Negative results and what they fixed" (lessons, reason first) — re-read for tone.

## Files
experiment/RESULTS.md (ROUND 8a-8d) | experiment/PAPER_NOTE.md | experiment/JOURNAL.md | experiment/00_READ_FIRST.md §4-5 | experiment/EVALUATION_PRINCIPLES.md (16)
experiment/v10_controls.py, v11_wikisql.py, v12_wikisql_baselines.py, trace_gates.py
logs: 2026-10-15_1530_v10_controls_727aecb.txt, 2026-10-15_1614_v11_wikisql_9916e38.txt, 2026-10-15_1628_v12_wikisql_baselines_ed24ce4.txt, trace_gates_hand_vs_learned.txt
