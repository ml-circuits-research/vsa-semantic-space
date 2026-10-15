# Evaluation principles (to be checked explicitly at every new experiment)

The mistakes already made in this project, and the rule that prevents each:

1. **Decoding addresses come from the WORLD, never from the model.** (Round 1 mistake: addresses from the active form -> penalized
   legitimate overlapping addresses.) The world defines what a role is; the model only has to get there.
2. **The world and the model share the SAME identities (F).** (Round 1 bug.) Check: `model.F is world.F`.
3. **Chance is computed and printed next to every figure.** With N candidates, chance = 1/N; a figure without chance next to it is not reported.
4. **Separate what was seen from what was not seen, on THREE independent axes**: identities (new words), formulations (new templates),
   TRANSITIONS (unseen pairs of adjacent categories). A "new formulation" with seen transitions and one with unseen transitions are
   two different tests and are reported separately.
5. **Report the margin, not just the accuracy**: margin = cos(correct candidate) - max cos(wrong candidate). 1.00 with margin 0.15 and 1.00 with margin
   0.50 are different states.
6. **Minimum 3 seeds, with standard deviation**; the seed also changes the world (identities, role vectors), not just the initialization.
7. **No calibration from the test set.** Everything that is a "reference" (addresses, thresholds) comes from training or from the world.
8. **Every model variation is evaluated with EXACTLY the same evaluation function and the same test data** (same world seeds).
9. **The trivial control is mandatory**: for any result, ask "what would a model that ignores order / ignores identities /
   answers with the majority give?" and print it in the same table.
10. **Do not interpret a failure as an impossibility before checking the evaluation.** (Round 1 mistake: "algebraic impossibility" that
    was an evaluation artifact.) Order of checks: evaluation -> data -> optimization -> only then architecture.
11. **Trace on a concrete example before training**: print the addresses (list of parameters) for every word in every
    formulation and check manually that what I think the model does is what it does.
12. **Training figures (loss, prediction cos) are not results**; they are reported only as diagnostics.

Checklist at the end of every evaluation script (printed in the log):
  [ ] addresses from the world   [ ] shared F   [ ] chance printed   [ ] seen/unseen axes separated   [ ] margin   [ ] >=3 seeds   [ ] trivial control

Added on the DGX (2026-10-15, rounds 4-5):
13. **Model variants are compared at the SAME optimization regime** (batch, number of steps, lr). Changing the batch 1 -> 32 moved
    path from 0.60 to 0.52 without any model change; otherwise we confuse optimization with architecture.
14. **A new construction that introduces a new ROLE (e.g. an unseen nesting depth => one more speaker) is reported per role**:
    the new role is at chance by construction (it was never a target); mixed with the other roles it hides what generalizes and what does not.
15. **A bundle of codes is interpreted only after the breakdown per formulation x role** (diag_v5.py): the mean over formulations showed 0.84 vs
    0.998 without saying that it is ONE role in ONE formulation, with a precise cause (pos: false position regularity). The mean hides the mechanism.
16. **A retracted claim is searched for and corrected in ALL documents in the same commit as the retraction** (grep the key words;
    the round-8a retraction of 'emergent categories' had to be propagated to seven files a commit later).
