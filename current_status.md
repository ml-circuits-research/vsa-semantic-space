# Current status (rewritten at every judgment point; last: 2026-10-15 late afternoon, round 7)

## Decisions applied since round 6
Compositional world = reference; §5.1c postponed and declared a known limit of causal addressing; chainL described as re-addressing;
"seen transitions" axis rebuilt as "role cues seen" and kept; §5.2 gates by gradient; transformer control on 6c. All docs and scripts in English.

## Round 7 results (5 seeds, D=1024, 30k sentences, batch 8; reference world; full tables in experiment/RESULTS.md ROUND 7)
7a. Rebuilt axis "new formulations, ROLE CUES seen" = T8, T13 (T10 'from A , C ...' correctly moves to unseen). Every VSA bundle: 1.000 on it.
7b. Transformer control (0.79M params, same fixed identities, same objective, same decoder, tuned by training-distribution validation):
    | | seen | new ids | role cues seen | depth 2 (seen) | depth 3 (UNSEEN) |
    |---|---|---|---|---|---|
    | transformer | 0.971 | **0.215** (chance 0.167) | 0.968 | 0.797 (A 0.49, C 0.50; B, X1, X2 1.00) | **0.390** (A/B/C at chance; X1, X2 1.00; X3 chance) |
    | VSA path+chainL+bigram | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
    Flat axes comparable (seen trans. 0.77 vs 0.79). It fails exactly where the construction gives the VSA model its generalization.
    Wording allowed: "at this data regime, a small transformer trained identically does not generalize to new identities or to a deeper
    construction" — not "transformers cannot". Not covered: more data/epochs, larger model, learned embeddings.
7c. Gates by gradient (soft chain, 32 categories; hand reference = all but P, O open):
    | variant | seen trans. | UNSEEN trans. | role cues | depth 1 / 2 / 3 (unseen) |
    |---|---|---|---|---|
    | bundle, hand gates | 0.783 | 0.756 | 1.000 | 1.000 / 1.000 / 1.000 |
    | bundle, learned (start closed) | **0.848** | **0.870** | 1.000 | 1.000 / 1.000 / 0.996 |
    | bundle, learned (start open) | 0.790 | 0.736 | 1.000 | 1.000 / 1.000 / 1.000 |
    | chainL alone, hand | 0.059 | 0.469 | 0.221 | 1.000 / 1.000 / 1.000 |
    | chainL alone, learned (closed) | **0.614** | **0.628** | **1.000** | 1.000 / 1.000 / 1.000 |
    Learned gates, IDENTICAL in all 5 seeds: P, O CLOSED; to, from, by, that, ',' (and COP, it, who, says, claims from the open start) OPEN;
    ALL 18 VERB FORMS CLOSED (<= 0.03).
    Criterion ("same set as by search: identities closed, verbs/prepositions open; no loss in 6c") HALF met: agrees on what is closed and
    on the prepositions/connectives; disagrees on the verbs. No loss at depths 1-3; BETTER on new formulations (fewer emitters = fewer
    untrained transitions, the round-2 observation). With left re-addressing a verb has nothing to emit that 'to/by/that' + the left slot
    do not give; the round-2 search opened verbs because, without left expectations, only emitters could place the words after them.

## Conclusions relative to the goal
1. The gating structure EMERGES from the single generative objective, stably across 5 worlds: identities / relation markers / action words.
   The hand set ("verbs + prepositions open") was a linguistic prior, not the optimum; the gradient's partition is better on every axis.
   For the paper: "everything emerges from one objective" is supported for the gates; we state where gradient and search disagree and why.
2. Composition to an unseen depth (6c) + emergent gates (7c) + a control that fails on exactly those two axes (7b) = the two pillars and
   the control of the principle note (POSITIONING article 1). The declared limit: right-announced structure (postposed reporting, comma
   constructions) is unaddressable causally — with the demonstration (RESULTS 6b/6c).
3. Caveats to keep: the world is ours (synthetic, two reporting verbs, prefixed only); the control is small and minimally tuned;
   the soft chain is 5x slower (position loop) — fine for now, to be vectorized before capacity runs.

## Questions for judgment
1. Accept "identities / relation markers / action words" as the emergent partition and REPLACE the hand set by the learned one in all
   subsequent runs (reference gates = learned-from-closed set: that, to, by, '.'; optionally from)?
2. Next, in this order? (a) word2vec control for categories + bottleneck associative-memory control (§5.3-5.4, cheap); (b) capacity on the
   reference world (D 4096-16384, 64-1000 identities; vectorize the soft chain first); (c) WikiSQL with the full model + BoW+LR baseline.
3. Start writing the principle note now (two pillars + control + declared limit), in parallel with (a)-(b)?

## Files
experiment/RESULTS.md (ROUND 7) | experiment/JOURNAL.md | experiment/00_READ_FIRST.md §4-5 | experiment/v8_gates.py, v9_transformer.py, v5_scale.py (split_templates_rolecues)
logs: 2026-10-15_1412_v8_gates_06e12bd.txt, 2026-10-15_1414_v9_transformer_75e7e0d.txt
