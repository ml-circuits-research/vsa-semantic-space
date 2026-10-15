# Current status (rewritten at every judgment point; last: 2026-10-15 afternoon, round 6)

## Since the last judgment (round 5)
- Process: run.sh ignores logs/; depth-3 column reports only roles that were training targets; "0.60 -> 0.82 (model); 0.90 only with the
  position prior" wording applied. Author decisions applied: model = path+chain+bigram, pos = separate prior; §5.1b before §5.2.
- Language: all Markdown translated to English and renamed (00_READ_FIRST, RESULTS, EVALUATION_PRINCIPLES, JOURNAL, POSITIONING,
  INTERPRETATION, INITIAL_ANALYSIS_DGX); number multisets verified identical per file. AGENTS.md: English on disk, Romanian in chat.
  Advanced tests now run on the English synthetic world (v5_scale.set_lang("en")).

## Round 6 results (5 seeds, D=1024, 30k sentences, batch 8; full tables in experiment/RESULTS.md ROUND 6)
6a. Language switch: seen 1.00, new identities 1.00, UNSEEN transitions 0.788 (vs 0.821 ro), depth 1-2 0.935 (vs 0.938) => reproduced.
    "Seen transitions" 0.785 vs 0.998: T10 'from A , C receives B' has C at 0.04 because the only training formulation with a person
    after a comma (T9 'to C , A gives B') has the AGENT there. That axis measures how redundant the formulation set is, not the mechanism.
6b. Left expectations (§5.1b, chainL): depth 2 inner roles A, C 0.96 -> 1.00; speakers X1/X2 0.65 -> 0.77. Criterion X2 > 0.9 NOT met.
    Cause: the FLAT world (depth-indexed speakers, depth-invariant A,B,C) is an algebraic contradiction for any cumulative addressing
    (a+g = 0, b+g = 0, d+b = 0, c+a = 0 => X2 - X1 = a+d = 0). The world was wrong, not the model.
6c. Compositional world (reported speech = R_rep (x) inner state, one speaker role, depth = number of R_rep; no new role at a new depth):
    | outer forms | code | depth 1 | depth 2 (seen) | depth 3 (UNSEEN) |
    |---|---|---|---|---|
    | prefixed only | path+chain+bigram | 0.982 | 0.997 | 0.979 (X3 0.90) |
    | prefixed only | path+chainL+bigram | 1.000 | 1.000 | **1.000 on all six roles** (margin 0.23) |
    | all 3 (incl. postposed '<P> , says X') | path+chainL+bigram | 0.980 | 0.881 | 0.538 |
    => FIRST CLEAN POSITIVE ON AN UNSEEN CONSTRUCTION: depth 3 never trained, decoded at 1.00. The world's algebra (R_rep applied d times)
    and the model's (displacements summed along the chain) coincide, so composition one level further is by construction once training
    lands on the consistent solution. Left expectations close the last gap (X3 0.90 -> 1.00).
    => The precise remaining barrier: structure announced AFTER its scope (postposed reporting; comma constructions). The clause is already
    written when 'says' arrives; no causal address can displace it. This is the exact, data-forced form of "deferred addressing is
    necessary": a displacement applied retroactively to a whole clause (the round-1 zoom-in), and only for right-announced structure.

## Conclusions relative to the goal
1. The thesis gains its strongest evidence so far: different formulations AND a deeper construction than ever seen converge to the same
   decodable state, role by role, from one generative objective, with no rule for 'that' or for verbs.
2. The round-5 negative is now sharper and partly withdrawn: not "nesting >= 2 is unsolvable from the prefix" but "a flat world is
   contradictory, and right-announced structure is unaddressable causally". Our world design produced the first; the second is real.
3. Two evaluation lessons became rules or proposals: a new depth must not add a role (rule 14, satisfied by the compositional world);
   the "seen transitions" axis must be redesigned to test mechanism rather than redundancy (00_READ_FIRST §5.1c).

## Decisions requested
1. Make the compositional world the reference world from now on? (It matches the original "fibre / copy of the space" idea; the flat
   world was our construction.) The flat world stays for reproducing rounds 5-6b.
2. Next: §5.1c (clause-level retroactive displacement for right-announced structure, criterion depth 2 > 0.95 with all 3 outer forms)
   before §5.2 (gates by gradient, now on the compositional world)? Or go to §5.2 directly and treat right-announced structure as a
   declared limit of causal addressing for the first paper?

## Files
experiment/RESULTS.md (ROUND 6) | experiment/JOURNAL.md | experiment/00_READ_FIRST.md §4-5 | experiment/v5_scale.py (set_lang), v6_left.py, v7_nested.py
logs: 2026-10-15_1246_v5_scale_9ea27cd, _1249_v6_left_9ea27cd, _1311_v7_nested_043caf8, _1319_v7_nested_e1e7011
