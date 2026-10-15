# Current status (rewritten at every judgment point; last: 2026-10-15 evening, round 8)

## Decisions applied since round 7
Learned gates (start closed) are the reference; hand set kept as ablation row. Order: cheap controls -> WikiSQL -> capacity. Principle note
started (experiment/PAPER_NOTE.md: two pillars, control, declared limit, trace hand vs learned, caveats, related work). All on disk in English.

## Round 8 results (full tables in experiment/RESULTS.md ROUND 8)
8a. Controls (5 seeds, reference world):
    - word2vec for categories: learned displacements 1.000 purity/ARI/NN — and skip-gram on the same sentences ALSO 1.000.
      => "emergent categories" RETRACTED as a contribution: categories are recoverable from distribution (skip-gram gives them equally well); the model CONSUMES them as input, it does not produce them; what emerges from the objective are the gates (who opens a frame), 7c and 8b.
    - bottleneck associative map (S = U V x on an order-aware input, rank 8..1024): new ids ~0.24 (chance 0.17), depth 3 unseen 0.25-0.39
      with the transformer's signature (known speakers perfect, inner roles chance); rank 128 = rank 1024 = 0.74 on seen.
      => capacity is not what is missing; "generative with a fixed operation is necessary" stands.
8b. WikiSQL (56k train, 8.4k dev, 1 seed, 3 epochs, D=512; categories = 42 frequent tokens + 48 skip-gram clusters; learned gates):
    | | sel column | aggregator | cond-1 column | separation same/different skeleton |
    |---|---|---|---|---|
    | chance / majority | 0.166 | 0.715 | — | — |
    | token overlap only | 0.484 | — | — | — |
    | BoW + logistic regression (mandatory baseline) | 0.494 | **0.867** | — | — |
    | round 1 (numpy position code) | 0.494 | 0.586 | 0.249 | — |
    | **full model, question only** | **0.824** | 0.813 | **0.631** | 0.28 (hard pairs 0.56 vs 0.76) |
    | full model, header + question | 0.673 | 0.799 | 0.484 | 0.15 |
    Learned gates on real questions: than, and, has, the, what, which, of, is OPEN; all content clusters closed — the synthetic partition again.
    Criteria from README_DGX: sel > 0.7 without the table MET (0.82 vs 0.49 baseline); agg > 0.85 NOT met (0.81 < 0.87 baseline);
    separation > 0.3 NOT met (0.28). Table at the input HURTS (header identities dominate the bundle; the table belongs in the decoder or
    bound to a role, not summed).

## Conclusions relative to the goal
1. The "emergent" claims are now controlled: gates emerge (7c) and survive; categories do NOT count as emergent (8a) — retracted before a
   reviewer had to ask. Generative-with-fixed-operation holds against both a transformer and a linear associative map of any rank.
2. Real text with executable consequence: a large, real advantage on the selected column (0.82 vs 0.49) with no table at the input,
   from a 1-seed run; the aggregator is lexical and the bag of words wins it; equivalence separation is just under the bar.
   This is enough to continue toward an empirical paper, not enough to claim one.
3. The principle note has all its pieces (PAPER_NOTE.md); what it needs is prose and figure 1.

## Questions for judgment
1. WikiSQL next steps, in this order? (a) 3 seeds + the test split (15,878); (b) tuned neural baselines on the same input (small transformer,
   BERT-small) at 1k / 5k / 56k — the empirical paper lives or dies here; (c) aggregator: a separate readout / dedicated code, or report as is;
   (d) hard-negative pairs built from SQL (role swaps) as a separate metric. Spider only after (b).
2. Capacity (D 4096-16384, 64-1000 identities) stays after WikiSQL, as decided?
3. Principle note: proceed to full prose + figure 1 now, or wait for the WikiSQL baselines so that the two papers are cut cleanly?

## Files
experiment/RESULTS.md (ROUND 8) | experiment/PAPER_NOTE.md | experiment/JOURNAL.md | experiment/00_READ_FIRST.md §4-5 | experiment/POSITIONING.md
experiment/v10_controls.py, v11_wikisql.py, trace_gates.py | logs: 2026-10-15_1530_v10_controls_727aecb.txt, 2026-10-15_1534_v11_wikisql_2fb26b9.txt, trace_gates_hand_vs_learned.txt
