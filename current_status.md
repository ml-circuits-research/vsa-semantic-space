# Current status (rewritten at every judgment point; last: 2026-10-15 ~16:50, round 8c; v12 baselines RUNNING)

## Decisions applied since the round-8 judgment
- WikiSQL order: (a) 3 seeds + test split + FULL-query accuracy — DONE (8c). (b) tuned neural baselines on the SAME input at 1k/5k/56k and
  (d) hard SQL pairs with the lexical bag on the same row — RUNNING (v12_wikisql_baselines.py, started ~16:40, ~60 min). (c) aggregator:
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

## What v12 (running) will answer
- Does a small transformer on EXACTLY the same input (fixed identities + same categories, no table, same objective and decoder) reach us at
  56k? Curve 1k / 5k / 56k for both, 3 seeds; lr chosen once on a validation split held out from TRAIN. No extra tuning for ours.
- Is the equivalence separation lexical or role-based? Hard pairs from SQL on the same table — H1 "different words, same SQL" (Jaccard <
  0.3), H2 "same words, different SQL" (Jaccard >= 0.5) — and the AUC same-vs-different, each for the model AND for the bag of identities
  (pure lexical overlap in the same space) on the same row. If the bag separates as well, the separation is lexical: the most important
  thing to learn from WikiSQL.
- FULL-query accuracy for every model and size; test split at 56k.

## Conclusions relative to the goal (interim)
1. Both "emergent" claims are now controlled: gates survive (7c, and the same partition reappears on real text, 8c); categories do not (8a).
2. On real text with an executable consequence the selected-column advantage over the mandatory baseline is large and stable (0.81 vs
   0.50, dev = test, 3 seeds, no table at the input). The aggregator is lexical and the bag wins it; the full query is 0.38.
   Whether an empirical paper exists depends on v12 (same-input neural baseline; lexical vs role-based separation).
3. The principle note is in full prose with its two pillars, control, declared limit, trace and caveats; figure 1 still to draw.

## Questions for judgment (interim; the v12 push will add the answers above)
1. The principle note (PAPER_NOTE.md, full prose v1): read for cut and tone — is the retraction section (§7) an asset or too long?
2. For the empirical paper, if v12 shows the transformer reaching us at 56k but not at 1k/5k: is "data efficiency on the selected column
   without the table, with a readable role state" a sufficient claim, or do we need Spider (nested SQL, where composition matters)?
3. If the lexical bag separates the hard pairs as well as the model: do we drop the equivalence metric from the empirical paper entirely, or
   keep it as a negative?

## Files
experiment/RESULTS.md (ROUND 8a-8c) | experiment/PAPER_NOTE.md | experiment/JOURNAL.md | experiment/00_READ_FIRST.md §4-5 | experiment/EVALUATION_PRINCIPLES.md (16)
experiment/v10_controls.py, v11_wikisql.py, v12_wikisql_baselines.py (running), trace_gates.py
logs: 2026-10-15_1530_v10_controls_727aecb.txt, 2026-10-15_1614_v11_wikisql_9916e38.txt, trace_gates_hand_vs_learned.txt
