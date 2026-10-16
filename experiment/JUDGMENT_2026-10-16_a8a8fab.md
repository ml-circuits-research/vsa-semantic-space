# Judgment at a8a8fab — answers to current_status.md E1-E6, one correction to the quantifier hypothesis, four process points
(received from the reviewing session on 2026-10-16, saved verbatim)

Scope: commits 1d598e5..a8a8fab (process commit, E2c both targets + diagnostic, A2, B1+B2, B4, B3). Everything in
JUDGMENT_2026-10-16_consolidated.md and in the N5-N7 addendum stays in force; this file contains only what changes or is new.

Verdict on the period: direction unchanged, programme followed in the agreed order (process commit first, then E2c, A2, B1+B2, B4, B3; no
parallel runs; no Spider; no tuning to pass B3/B4). The process commit applied the consolidated file correctly (rules 18-24, STATUS.md per
N0, trainer.py with memory logging / checkpoints / --resume, pid lock). The E2c diagnostic is the right way to handle a below-chance
number (rule 10 applied before anything else). Quality of the reporting: good. Four points to fix (section C), one of them before the
quantifier numbers are quoted again.

## A. Answers

E1 (quantifiers as the named exception; hypothesis B3 as HYPOTHESIS). Yes to both — with one correction to the hypothesis, which as
written explains less than it claims.
  The relation hypothesis ("scope is a relation between two slots, needs a degree-2 term") can only speak to the TWO-quantifier marked case.
  But the ONE-quantifier case fails at the unseen depth just the same (slot 0.333, noun 0.206 under the no-none rule; raw true-vs-other
  0.024 vs 0.015 — no signal), and "which single slot is quantified" is not a relation between two slots: it is a degree-1 fact, exactly the
  slot target R_sc (x) (R_role (x) f_noun). So the one-quantifier failure is explained by the SECOND bullet of 9c-ter (the scope term rides on
  position-local terms because the determiner gates stayed closed, so it is not re-addressed by the reporting frame), not by the degree of the
  terms. Record two hypotheses, not one:
    H-a (addressing): the scope term composes only if it is carried by the code the reporting frame re-addresses (the chain). Covers both
        the one- and the two-quantifier failures. Tested by C2.
    H-b (relation): the two-quantifier case additionally needs a term that holds both phrases. Covers the two-quantifier case only.
        Testable only AFTER C2, and only if the one-quantifier case then composes while the two-quantifier case still fails.
  Wording for RESULTS/STATUS: "the regularity holds for six operator cases; quantifier scope fails at the unseen depth in all three world
  definitions tried (numbers with the decoder rule stated — see C1 below); hypotheses H-a / H-b recorded, C2 PLANNED". Keep "same class of limit
  as right-announced structure" out of the claim: that limit is about causal order, this one is about addressing; two declared limits, not one.

E2 (C2 then C1; is the degree-2 term a change of principle?). C2 first: agreed. C1 as written is NOT approved, for a mathematical reason
that a rule-11 trace would have caught:
  (i) Binding is commutative. address (x) f_wide (x) f_narrow = address (x) f_narrow (x) f_wide: the term is symmetric in the two nouns and
      cannot say which phrase is wide. "each A ... a B" and "a A ... each B" with the same nouns would produce the SAME term. To carry the
      asymmetry one filler must be permuted (address (x) f_wide (x) rho(f_narrow), rho a fixed permutation) — which is permutation binding, the
      B7 remedy, i.e. a second new mechanism, not one.
  (ii) The information is already degree 1. With two quantified phrases, "A out-scopes B" is fully determined by "which slot is wide", and
      that is what the slot target already encodes. Degree 2 adds no information the world does not already assert; it would only move the
      same information to a different address. So C1 does not test H-b as stated.
  What to do instead, in order, each with its criterion stated before the run:
  C2 as proposed (determiner gates GIVEN open, rule 20 stated), 5 seeds, regime of 9c-bis, both targets. Before training, the rule-11 trace at
      DEPTH 2 on one two-quantifier sentence: write down the address at which the scope term must land inside two reporting frames and which of
      the three codes carries it after the flag. If the flag does not put the scope term into the re-addressed code, the flag is not the test
      of H-a — find the setting that does (the test is "the scope term moves with the frame", whatever flag achieves it), and say so.
      Criterion: scope > 0.9 at depth 0 AND at depth 2, nouns 1.00, cost on 6c / 9b / 9d / 10d tables reported. Report one-quantifier and
      two-quantifier scope SEPARATELY (they test different hypotheses).
  Only if C2 composes the one-quantifier case and not the two-quantifier one: a relational term with a permuted filler, declared in
      00_READ_FIRST and PAPER_NOTE as a change of principle (first term not of the form address (x) one word star; identities still fixed, so
      rule "identities never enter the parameters" is untouched), emitted by the wide determiner's category through a learned displacement, not
      by a hand-coded rule; criterion as C1 plus "inert when no quantifier" measured as no loss on every other table.
  If C2 fails on both: stop building remedies; the quantifier line stays the named exception with H-a / H-b open, and the next experiments are
      B5-B7 as planned. Do not spend a third world definition on quantifiers without a new argument.

E3 (C3, "emit both"). Adopt, with the two-sided criterion exactly as proposed (both > 0.9 on never-trained ambiguous sentences AND one
reading > 0.9 on unambiguous ones), plus two additions: report the "both" rate on unambiguous sentences as a FALSE-BOTH rate with its own
number (it is the false alarm of this verdict); and measure C3 on whichever target C2 ends with, not on the noun target alone (the 0.69 is
one target's number). Until the criterion is met the label stays "observed, partial". In the positioning documents nothing changes: they
already say both readings are kept and handed on. The verdict "both" is the reader's equivalent of the verifier's "compatible, incomplete":
the reasoner receives two frames and the margin between them.

E4 (reference table; depth-3 claim). 10b is the reference table from now on (predicate role, marker, calibration — rule 23). 6c stays as
the predicate-less world's number, named as such. The depth-3 claim is stated with BOTH numbers, never one: "depth 3 unseen: 0.81 at full
coverage; at tau 0.05, coverage 0.59 (A) / 0.33 (C) with covered accuracy 1.000 / 0.975, confident-wrong 0.000 / 0.008". The accuracy-vs-
coverage curve is the headline for every unseen-depth axis from now on (C6 first half: approved).
  Before D = 2048 (C6 second half), three cheaper diagnostics in rule-10 order (evaluation, world, then model), decision rule stated in advance:
  (a) Per-role, per-form breakdown at depth 3: which sentences fail — are they the recipient-less forms (share 0.2) or all forms? The loss is on
      A and C with the predicate correct; find out whether it correlates with the verbs whose gates are now HALF-open (0.55-0.62; in 6c the
      verbs were closed). A gate sitting at the sigmoid midpoint is the first suspect for noise that grows with depth.
  (b) Same world with the recipient-less share set to 0 (one flag): separates "one more term per level" from "half-open verb gates".
  (c) Verb gates GIVEN closed (rule 20 stated), recipient-less forms kept: the other half of the same separation.
  (d) Then D = 2048, stated as the capacity test.
  Decision rule: if (b) or (c) restores > 0.9 at depth 3 with the predicate role, the regression is gating — keep the predicate role and the
  recipient-less forms, and report the depth-3 number with the gate setting named; if only (d) restores it, it is capacity, and the terms-per-
  level cost becomes a stated property (report the capacity curve: pairs per state at depth 3 against D). The 10b reading "first an
  evaluation of the world" is right; (a)-(c) are that evaluation.

E5 (B4: LLM version or C4/C5 first). C4 and C5 first — both are re-scoring or a readout change, no API, no retraining of the reader.
  C4 with one condition: the threshold is set on rows that are NOT the rows reported (split dev in two: set tau on one half, report the whole
  curve on the other), otherwise the false-alarm number is tuned on its own test set. Report detection lost at every tau, not one point.
  C5 approved as written (condition-count readout from the occupancy marker; present/absent > 0.9 on dev); report how much drop_cond / add_cond
  detection moves.
  Add to the stated limits of 10c, in RESULTS and in the STATUS row: on GOLD frames the verifier confirms 0.309 and says "unverifiable" on
  0.529 — it is a filter on the role part of a formalization, not a judge of it; half of correct formalizations get no verdict. That sentence
  belongs next to the 0.162 false alarm.
  The LLM version stays PLANNED and is prepared now so that it is one command when the data arrives: fix the input format in the repo
  (jsonl, one row per question: question, table_id, gold frame, LLM frame, second-LLM agreement flag) and the protocol of B3 (two LLMs, a
  disagreement discards, team audits a sample of 50, size stated); n >= 1,000 rows, sampled from dev with the seed stated. The formalizations
  are produced off the machine, by LLMs, and supplied as that file.

E6 (queue rule). Approved: queues wait on logs/.run.lock only. Written as a rule: "a wait loop never greps for a pattern that occurs in its
own command line; it polls the pid lock (kill -0 on the pid in the file) and nothing else". Two additions: (i) replace ad-hoc waiting with a
queue.sh that takes a list of run.sh invocations and runs them strictly in sequence — then nothing waits by pattern at all; (ii) keep the
pgrep check inside run.sh as a secondary guard (it is not a waiter), but exclude run.sh's own process tree from the match, or it will one
day refuse its own launch.

## B. What was done right (keep)
The diagnostic before any conclusion in 9c-ter; the raw score profile (true vs other roles) as the decisive number; the explicit "world
correction, not model repair" bookkeeping; the trace of rule 11 before E2c; scope separate from presence in 10d, with the scope-only pairs
and the bag control; the 10b design note decided before the run (option (a), stated); calibration present in every new table; the memory
curve flat and logged; STATUS.md in the N0 shape with a row per claim and the label on each; the deadlock written up as a lesson the same
day.

## C. Process points (fix before the quantifier numbers are quoted again)

C1. The decoder change is mis-described and the numbers are unlabelled. RESULTS 9c-ter says the relative 'none' rule "affects the
    affirmative case only"; its own diagnostic table says otherwise: slot one-quantifier d0 0.978 -> 0.850, d2 0.333 -> 0.033; noun d2 0.206
    -> 0.067. The relative rule carries the same depth artefact as the absolute one, weaker (scope scores shrink with depth, noun scores do
    not, so the ratio falls under 0.25 at depth 2 and the decoder says 'none' for quantified sentences too). Meanwhile STATUS.md quotes
    0.52 / 0.33 / 0.21 (the no-none rule) while the default decoder is now the relative rule. Required: (i) correct the sentence; (ii) apply
    rule 17 to the quantifier decoder as it is applied to negation: PRESENCE (quantifier present / absent) is its own axis with its own
    confusion matrix per depth, and SCOPE is reported conditional on presence (and jointly) — then no scope number depends on a threshold
    rule; (iii) every quantifier number in RESULTS, PAPER_NOTE and STATUS carries the decoder rule it was computed under, in the cell;
    (iv) "old numbers not re-run": re-scoring is cheap (diag_e2c already does it) — if the 5-seed models are saved, re-score all seeds under
    the presence/scope split and replace the table; if they are not, keep the numbers with the labels of (iii). No retraining.

C2. Two thresholds are set on the rows they are reported on. v16_reference: the occupancy threshold is the midpoint of the known/absent
    means computed on the same 360 readouts that are then scored. v17_verifier: the condition presence threshold is the midpoint over dev
    using dev's gold labels, and the verifier is scored on the same dev rows. One scalar each, and the separation in 10b (0.38 vs 0.06) means
    the 10b number will not move — but "fixed before scoring" must mean fixed on OTHER rows. Required: set both midpoints on a calibration
    split (training sentences for v16; half of dev for v17), re-score (no retraining), state the split in the table header. The 10c numbers
    may move slightly; report old and new.

C3. The addendum (N5 kind ambiguity, N6 missing-legend residual, N7 world identification by fit) is not in the repo and has no STATUS rows.
    If it was handed over, it was not applied; if not, it comes with this file. Add the three PLANNED rows with their criteria to STATUS.md
    section 1 and to current_status.md D.

C4. STATUS.md: shape matches N0 — good. Three cells to fix with the above: the quantifier row carries the decoder rule; the verifier row
    carries "confirmed 0.31 / unverifiable 0.53 on gold"; the depth-3 row already carries both numbers — keep that form everywhere.

## D. Order from here
1. C (one commit: decoder relabel / presence-scope split and re-score; thresholds on calibration splits and re-score; N5-N7 rows; STATUS cells).
2. E4 diagnostics (a)-(c), then D = 2048 — all on the reference world, cheap, same seeds.
3. C2 with the depth-2 trace first; one- and two-quantifier scope reported separately.
4. C4 + C5 on the verifier; LLM input format committed.
5. Then B5-B7 as in the consolidated file; C3 ("both") measured on the world C2 leaves.
No time estimates; criteria and logs. Nothing in the positioning documents changes with this status: they carry no unseen-depth numbers and
point to STATUS.md, which now carries the depth-3 number in the two-number form.
