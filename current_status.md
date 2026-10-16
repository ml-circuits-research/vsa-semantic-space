# Current status (rewritten whole at every judgment point; last: 2026-10-16, process commit applied, E2c launched)

Labels (rule 22): BUILT / PLANNED / HYPOTHESIS. Every table so far is a DRAFT under rule 23 until the predicate role and calibration land (B1, B2).
Source files: experiment/JUDGMENT_2026-10-16_consolidated.md (the programme), experiment/RESULTS.md (numbers), STATUS.md (public summary).

## A. Process commit (Part I-II B) — BUILT (commit 1d598e5)
- P1: external-review corrections A1-A3, A5, A6 applied (PAPER_NOTE: supervision wording, commutative-binding limit, residual instead of
  norm, memory figure; EMPIRICAL_NOTE caveats; 00_READ_FIRST §2); frame.residual() implemented; "knows when it does not know" replaced
  everywhere (grep: 0 left); rules 18-24 appended. A4 (predicate role) is code and comes with B1.
- P2: "one exposure" renamed everywhere to "seen at depth d in training (share p), tested at depth d+1"; shares stated (E1b: negation
  share 0.4 over depths 0-1 = 0.2 at depth 1; E3b: 0.15 at depth 0; E2: quantified share 0.6 over depths 0-1).
- P3: this file rewritten whole. P4: run.sh pid-file lock (logs/.run.lock) in addition to pgrep. N0: STATUS.md at the root.
- Q2: both ambiguity numbers labelled, n per cell (360 sentences per seed) in RESULTS 9c-bis.

## B. BUILT (numbers in RESULTS; logs with commit hash)
| item | result | section |
|---|---|---|
| composition to an unseen depth (reporting) | 1.000 all six roles, 5 seeds | 6c |
| gates emerge by gradient (identities closed, relation markers open, verbs closed) | identical in 5/5 seeds; no loss | 7c |
| same-input transformer / bottleneck map fail on new identities and unseen depth | 0.215 / 0.390 ; 0.24 / 0.25-0.39 | 7b, 8a |
| categories "emerge" | RETRACTED (skip-gram 1.000 = 1.000) | 8a |
| WikiSQL selected column, no table, same input | 0.82 vs 0.54 (transformer; 0.65 at 10x) vs 0.50 (BoW+LR); full query 0.37 vs 0.19 | 8c-8d |
| WikiSQL aggregator (lexical) / equivalence separation | lost (0.80 vs 0.89) / lexical overall; role-based on H2 (0.62 vs 0.77, n 1,353) | 8d |
| frame -> Prolog / SMT round trip, depth estimated from S | 1.000 at depths 0-3 | 9a |
| negation as a role: scope / separation | 1.000 / 0.18 vs bag 0.72 | 9b |
| regularity "seen at depth d (share p), read at d+1": reporting, negation (0.2 at d1), modality, conditional, modal-in-conditional (0.15 at d0) | 1.000 / 0.972 / 1.000 / 1.000 / 1.000 | 6c, 9b-bis, 9d, 9d-bis |
| the same for quantifiers, role-identity world | 0.362 at depth 2 (nouns 1.00; quantifier terms ~0.5) — world error (Q1) | 9c-bis |
| zero occurrences of a combination in training | not read (E1 sentential 0.0-0.5; E3 mixed 0.000) | 9b, 9d |
| ambiguity: both hypotheses reported (never trained) / forced choice surface-first | 0.43 / 0.26 (n = 360 per seed) | 9c-bis |
| right-announced structure | declared limit: postposed reporting 0.88 at depth 2; 'Q if P' 0.20-0.44 | 6c, 9d |
| machine protection after the 2026-10-15 hang | guard.py (allocator cap 50% tested; MemAvailable watchdog; wall-clock), run.sh lock + timeout | README_DGX |

## C. RUNNING / PLANNED, in the order of the judgment
1. E2c (RUNNING) — quantifier world correction. Rule-11 trace written and run before training (experiment/trace_e2c.py, logs/trace_e2c.txt):
   the primary target R_sc (x) (R_role (x) f_noun) IS a product of per-category vectors the model has — the wide determiner's slot
   (L[each:NP], or bg:each:NP when the determiner's gate stays closed) differs from 'every''s; the role-identity target of 9c-bis is not.
   Primary target 'slot', secondary 'noun', 5 seeds each, same regime (--p_aff 0.4). Criterion: scope > 0.9 at depth 0, FULL > 0.9 at the
   unseen depth 2; n per cell. 9c-bis stays as the role-identity-world result.
2. Round 10 A2 (PLANNED, code next) — memory (RSS + device) every N steps in every log; no graph-carrying tensor survives a step;
   expandable_segments; empty_cache between phases; checkpoints every M steps with --resume; log-advance watchdog; the memory curve of
   one short reference training reported before anything larger.
3. B1 (PLANNED) — predicate role R_pred (x) f_verb_lemma, option (a): surface-verb identity with a per-world synonym table at readout
   (decision: (a) first; (b) event-type identity only as a stated change of principle) + occupancy marker m; reference table re-run.
   B2 (PLANNED) — calibration in evaluate(): margin distribution, confident-wrong at a stated threshold, abstention split, accuracy-vs-
   coverage; WikiSQL dev too. Then B3, B4, B5-B7, N3, N1, N2, B8-B9, N4.

## D. HYPOTHESIS
Pronouns by search over stored frames (B8); instruction-injection detection (B9); affect from tone / irony — out of scope and said so (N2).

## E. Open decisions for the reviewing session
None pending. The next status reports E2c (both targets, trace), the A2 memory curve, and the predicate-filler decision (option (a)) in use.
