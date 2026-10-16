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
   When a MEAN meets a criterion that some seeds do not, the per-seed list is printed in the cell (judgment 013c490 C3; example: the C2b
   noun target, one-quantifier scope at the unseen depth 0.927 with 3/5 seeds above 0.9: "0.927 (0.959, 0.929, 0.898, 0.888, 0.959)").
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
17. **For every logical operator (negation, quantifier, modality, conditional, coordination), report SCOPE correctness — what the operator
    acts on — separately from its presence.** A negation attached to the wrong argument is worse for a reasoner than a missed negation.
    Operators are ROLES in the world; the model stays uniform (same step for every word). If an operator needs a special rule, that is a
    finding to report, not a fix to hide.

Added 2026-10-16 (external review of the positioning document; round 10):
18. **Every accuracy is accompanied by: the margin distribution, the confident-wrong rate at a stated threshold, and the reconstruction
    residual** ||S - sum_r R_r (x) f_best(r)|| / ||S||. "The model knows when it does not know" is never written; "the model reports X and Y,
    calibrated as follows" is.
19. **Any claim of the form "the geometry produces / guarantees ..." is either a proof or is replaced by "we observed ... (round, seeds)".**
20. **The world's roles are an input.** Documents say what is given (roles, identities, categories) and what is learned (moves, relative
    addresses, gates) in the same sentence whenever either is mentioned. Supervision wording: "structured per-sentence targets with
    pre-set roles; the model learns the assignment" — never "no semantic labels" or "unsupervised".
21. **Every run logs memory (RSS and device) and is resumable from a checkpoint; a run without a memory curve in its log is not a result.**
22. **Every experiment in a status carries one of three labels — BUILT (code and measurement exist, numbers reported), PLANNED (criterion
    written, not run), HYPOTHESIS (mechanism described, no criterion yet)** — and the positioning document uses the same three. When a label
    changes, say which paragraph of the document changes with it.
23. **For every table from now on: the predicate role is present, and the calibration numbers of rule 18 are present.** A table without
    them is a draft, not a result.
24. **The regime of a composition test is named exactly**: "seen at depth d in training (share p), tested at depth d+1" — never "one
    exposure" (nothing was shown once; the combination was a stated share of the training sentences at the shallow depth).
25. **A threshold is never set on the rows it is reported on.** Every threshold (presence, occupancy, abstention tau, 'none') is set on a
    calibration split — training sentences, or a named half of dev — and the table header names that split. Setting it on the scored rows
    tunes the number on its own test set, even when it is "only a midpoint".
26. **Queues wait on the pid lock only.** A wait loop never greps for a pattern that occurs in its own command line (two queued launches
    waited on each other for an hour on 2026-10-16); `queue.sh` runs `run.sh` invocations strictly in sequence; the pgrep check inside `run.sh`
    is a secondary guard that excludes its own process tree.
27. **A criterion is judged on the MEAN over seeds, with the per-seed list printed next to it** whenever the mean meets a criterion that some
    seeds do not; a criterion that every seed must meet says so ("5/5 seeds") and is reported as the count of seeds meeting it. (Hand-over
    78d36e8; the C2c two-quantifier scope was met 5/5, the C2b noun target one-quantifier scope 3/5 with the mean met.)
