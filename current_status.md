# Current status (rewritten whole at every judgment point; last: 2026-10-17 — judgment 752853c applied: decisions recorded, N6 run; tool round 6 next)

Labels (rule 22): BUILT / PLANNED / HYPOTHESIS. Sources: experiment/JUDGMENT_2026-10-17_752853c.md (saved verbatim), experiment/RESULTS.md 15, 14-bis,
11-sexies; STATUS.md; experiment/JOURNAL.md; tool/README.md.

## A. Done since 752853c, in the instructed order
| step | result | criterion / decision rule (written before) | outcome |
|---|---|---|---|
| decisions recorded (§1, §2) | STATUS B7 row: three codes stay the reference, the position code a declared per-world option (reason: a surface-position prior); 00_READ_FIRST §7 says so; P0 row notes the calibration rule decision pending round 6 | — | done |
| tool (agent, in progress) | (1) §2 re-scoring from the stored ratios: share of false / missed "present" inside the abstention band; (2) round 6: fitted threshold as the calibration RULE + presence margin on the wire, `sop train` exposes the code list (default off) | §2's expectation: most errors inside the band | GPU handed over; result at the next push |
| **N6** missing question in the legend (v21_missing_role, log 2026-10-17_1343; RESULTS 15) | remaining roles 1.000; relative residual 0.337 WITH the hidden role vs 0.462 WITHOUT (gap **−0.267**); recovery of the hidden filler from the residual: true R_C **0.110** (chance 0.083), random 0.093, the reader's own calibrated address **0.963** (5/5 seeds > 0.92); residuals cluster by filler (cos 0.19 same / 0.05 different), 1-NN with no vector 0.57 | addendum: gap > 0.3; oracle recovery > 0.8; random at chance | **NOT met as written** (third branch of the rule): the residual SHARE does not expose the missing question, and a reader trained without the role does not put it at the legend's vector; **detectable by residual DIRECTION** — the filler sits at the reader's own address, consistently across sentences (the tool's `_unplaced`-by-direction loop is that detector; test 5b) |

## B. HYPOTHESIS
- Why the residual share goes the wrong way in N6: the relative residual is dominated by the function-word terms the legend never explains, and
  a sentence with the hidden role has more tokens (larger ||S||, smaller share); the recipient-less forms are the rarer templates and fit worse.
  Marked as a hypothesis; the per-term decomposition of the residual would test it (not run).
- Presence calibration across formulations: ~10% class overlap (reader); the midpoint rule added a further third of missed recipients (rule) —
  the fitted rule is adopted (round 6 confirms).
- B5d multi-hop left re-seating: only if a legend requires the subject's term as a structure witness.
- Verifier condition axis (C5 closed): legend choice at P1 (indexed slots vs one condition role with a SET value).
- N5 "resolve before drawing": diagnostic in the next kinds run.

## C. PLANNED, in the instructed order
Tool round 6 (running next); N8 (verb-final world, low priority); P1 when the team's data arrives (the annotation path once with a key, wherever
the team runs it); N7 after N1; B8-B9; capacity at 64-1,000 identities and D = 4,096-16,384 (00_READ_FIRST §7).

## D. Questions for the reviewing session
1. N6's recording: "NOT met as written; detectable by residual direction (0.963 at the reader's address; 1-NN 0.57), not by residual share" —
   agreed? The addendum's oracle test presumes the reader places the missing filler at the legend's vector, which a reader trained without the
   role cannot do; the measurement that succeeds uses the reader's own address, calibrated on 120 gold sentences (a supervised step) or no
   vector at all (1-NN 0.57, unsupervised). Which of the two is the number for the positioning document's "a missing question is detectable"?
2. Should the residual-share hypothesis (function-word terms dominate) be tested with a per-term decomposition (cheap re-scoring on the saved
   N6 models), or is the direction result enough for the record?
3. After round 6 the tool's remaining items are the annotation path (needs a key) and P1 data. Is there anything else for the agent before P1,
   or does it stop until the data arrives?
