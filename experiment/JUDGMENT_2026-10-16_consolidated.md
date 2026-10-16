# JUDGMENT 2026-10-16 (consolidated) — received from the reviewing session; the only file to read before the next runs

Saved verbatim into the repository on 2026-10-16 because the files it refers to (MEANING_AS_A_PATH.md, JUDGMENT_round8d_and_next.md,
JUDGMENT_after_external_review.md) were never committed here; the post-8d judgment (R1-R5, E1-E7) is recorded in JOURNAL.md
(2026-10-15 "judgment after 8d") and in 00_READ_FIRST §5.9. Order of application: Part I-II -> Part III -> Part IV -> Part V.

## Part I-II. Answers (Q0-Q3), E2c, process corrections (P1-P4)
Q0. The composition regularity becomes a claim as an EMPIRICAL regularity with its counts, never as a property of the geometry (rule 19),
    with the data regime named exactly. Approved wording: "On constructed worlds, an operator combination that appears in training at
    depth d is read correctly at depth d+1, which was never trained (5 seeds). Combinations with zero occurrences in training are not
    read (E1 sentential forms, E3 mixed). The regularity held for reporting, negation, modality, conditionals and modal-inside-
    conditional (5 of 6 operator cases); for quantifiers it did not hold in the role-identity world (9c-bis) and is re-tested in E2c
    after a world-definition correction." If E2c passes: 6 of 6 and the quantifier line moves to the correction history; if it fails,
    the exception stays in the claim. Zero-occurrence failures stay in the same sentence as the success.
Q1. Quantifiers: (b), world error. The 9c-bis world asserts R_sc (x) f_role (a ROLE identity as filler): unreachable by construction,
    same class as the flat world of round 5. Correction, uniform with negation (R3): the wide-scope term becomes R_sc (x) (R_role (x)
    f_noun) — the scope operator wraps the quantified SLOT. Representable: the noun's star drawn a second time at address R_sc (x) R_role,
    reached with ONE learned displacement for the category "wide-scope determiner" relative to the slot's beacon; unambiguous when the
    same noun fills two roles; no new identity. E2c: this target primary, R_sc (x) f_noun secondary column, same seeds, same regime.
    Before training, rule 11: trace one sentence; write down, for the wide-scope noun, the address the model must produce and the
    per-category vectors it would be a product of; if not such a product, stop and report. 9c-bis stays as "the result for the
    role-identity world"; E2c is a world-definition correction, not a model repair (rule 16: say which earlier sentences change).
Q2. Ambiguity: report both numbers, labelled — "both hypotheses reported" (main; never trained; ~0.43-0.50) and the forced-choice
    distribution (surface-first) as "what a reasoner receives when forced to pick"; n per cell (rule 2); reading stays "observed, partial".
Q3. Sequential launches enforced by run.sh: good. Add round-10 A2: memory (RSS + device) logged every N steps; checkpoints with --resume.
    The guard's hard exit is right for the machine and wrong for the run unless there is a checkpoint.
P1. Apply the external-review corrections A1-A6; append rules 18-20; replace "knows when it does not know" with "reports both hypotheses /
    reports margin and residual" everywhere (rule 16 grep). Predicate role and occupancy marker right after E2c (round 10 B1), with the
    predicate-filler design note decided first.
P2. "One exposure" is the wrong name: the combination was in a stated SHARE of training sentences at the shallow depth. Rename everywhere:
    "seen at depth d in training (share p), tested at depth d+1"; add the share to every such table row.
P3. current_status.md rewritten whole; one numbering; every line current.
P4. run.sh: lock by a pid file written by run.sh (removed at exit), not only pgrep.
Order: B (one commit) -> E2c -> round 10 A2 (stability commit, memory curve of the reference run) -> B1 + B2 -> B3, B4. No Spider; no parallel runs.
Deliverables: the B commit; E2c with both target columns, n per cell, the rule-11 trace; the memory curve; the predicate-filler decision
written before B1; status rewritten whole with BUILT / PLANNED / HYPOTHESIS labels (rule 22).

## Part III. External review of the positioning document — accepted corrections, experiments X1-X6, rules 18-20
A1. Occupancy cannot be read from the norm after unbinding (isometry). "In-map weight" = the RECONSTRUCTION RESIDUAL
    ||S - sum_r R_r (x) f_best(r)|| relative to ||S||; implement frame.residual(); report alongside margins.
A2. Commutative binding records the multiset of roles on a path, not their order: a structural limit (PAPER_NOTE limits, EMPIRICAL_NOTE
    caveats). No claim of arbitrary tree representation.
A3. Supervision wording: "no paraphrase labels" stays; "no semantic labels" / "unsupervised" goes. Correct form: "structured per-sentence
    targets with pre-set roles; the model learns the assignment" (PAPER_NOTE abstract and §1; 00_READ_FIRST §2).
A4. The worlds have no PREDICATE role (GIVE and SELL map to the same target): add R_pred (x) f_verb_lemma; re-run the reference table once.
A5. "Knows when it does not know" -> "reports margins and residual"; calibration reporting (rule 18).
A6. Memory figure: ~D x rows x 4 bytes, "of the order of a megabyte at D~1k". No "sub-megabyte".
X1 order collisions and remedies (permutation binding; level-indexed roles); X2 occupancy marker m: R_r (x) (f_w + m); X3 predicate role
and several events; X4 calibration/abstention (confident-wrong < 2% at coverage > 80%, reported not tuned; WikiSQL dev too); X5 clean
generalisation split (new vocabulary / new construction / new operator combination, three separate held-out sets); X6 same-supervision
parser baselines (per-role BoW+LR; small transformer with one head per role).
Rules 18-20: see EVALUATION_PRINCIPLES.

## Part IV. Round 10 — stability first, then B1-B9
A1 inventory after restart (done in JOURNAL 2026-10-16 morning). A2 stability commit: memory (RSS + device) logged every N steps; no
graph-carrying tensor survives a step; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; empty_cache between phases; checkpoints every M
steps with --resume; nohup + line-flushed logs + a log-advance watchdog; one training at a time; one short reference training with the
memory curve reported before anything larger. A3: reduce D or batch and say so, never let the kernel decide.
B1 predicate role + occupancy marker (design note: option (a) surface verb identity + per-world synonym table at readout, run first;
option (b) event-type identity = a learned verb-to-event map, only as a stated change of principle); B2 calibration in evaluate();
B3 scope of negation on commands (command world, templates + LLM paraphrases with a second LLM check, audited sample of 50);
B4 verifier with injected errors on WikiSQL formalizations (precision/recall per error type vs a lexical baseline); B5 fourth address
code remeasured on the X5 splits (decision rule: keep only if it does not reduce new-construction by more than 0.02); B6 sparse block
codes (phase quantization L in {16, 64, 256}, per-block top-k, capacity curve, bits per vector); B7 = X1; B8 pronouns (after B1-B4);
B9 instruction-injection detection (after B3, B4). Not now: Spider, D 4096-16384, bigger models.
Rules 21-23: see EVALUATION_PRINCIPLES. Deliverables: inventory, memory curve, B1 table, B2 in every table, B3, B4; labels on every item.

## Part V. New validation experiments (after B1-B4 unless stated)
N0 STATUS.md at the repository ROOT, never renamed, one page: (1) date and commit; (2) table claim | status (BUILT/PLANNED/HYPOTHESIS) |
where measured | criterion | number; (3) open world-definition corrections; (4) next three experiments. Public summary derived from
current_status.md. N1 mood as a role (R_decl / R_imp / R_q; f_wh placeholder in the asked role; after B3). N2 emotion/stance with scope
(experiencer / affect / target = nested event; after N1). N3 fillers and disfluencies: the null jump (after B2; cheap). N4 real
conversational text, small (after N1-N3). Order: N0 now; N1 after B3; N3 after B2; N2 after N1; N4 last.
