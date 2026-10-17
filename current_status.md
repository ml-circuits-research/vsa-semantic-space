# Current status (rewritten whole at every judgment point; last: 2026-10-17 — judgment 752853c applied: decisions recorded, N6 run; tool round 6 next)

Labels (rule 22): BUILT / PLANNED / HYPOTHESIS. Sources: experiment/JUDGMENT_2026-10-17_752853c.md (saved verbatim), experiment/RESULTS.md 15, 14-bis,
11-sexies; STATUS.md; experiment/JOURNAL.md; tool/README.md.

## A. Done since 752853c, in the instructed order
| step | result | criterion / decision rule (written before) | outcome |
|---|---|---|---|
| decisions recorded (§1, §2) | STATUS B7 row: three codes stay the reference, the position code a declared per-world option (reason: a surface-position prior); 00_READ_FIRST §7 says so; P0 row notes the calibration rule decision pending round 6 | — | done |
| Tool §2 re-scoring + round 6 (agent; RESULTS 11-septies; tool/runs/round5_1ed4306/round5_margins.md, tool/runs/round6_051dd33/) | §2: 0.617 of round 5's presence errors inside the band (presence part only); round 6 with the fitted rule: recipient presence on unseen formulations 0.639 -> **0.772**, absent 1.000 -> 0.991; errors inside the abstention band (presence or identity margin) **0.699** pooled (per seed 0.609, 0.550, 0.886, 0.476, 0.750); product heads FULL 0.444 (midpoint 0.418; v16 protocol 0.556; 10b-quinquies 0.627) | §2's expectation: most errors inside the band | rule ADOPTED (judgment 752853c §2); "most" met pooled, not in every seed (0.476 in seed 3); the asymmetry option reported, not built; `sop train` exposes the code list, default off |
| N6 per-term decomposition (rescore_v21_terms, log 2026-10-17_1420; RESULTS 15-bis) | shares along the residual: with the recipient — hidden token 0.30, content fit 0.26, function words 0.44; without — content fit **0.73**, function words 0.27; content-share gap +0.08 | judgment 934fbae B: function words the majority -> two `_residual` numbers and the gap re-scored; not -> hypothesis dropped | **NOT the majority -> the RESULTS 15 hypothesis DROPPED**; the share inverted because the recipient-less forms (the rarer templates) fit worse, not because function words dominate; `_residual` stays one number; direction stays the detector |
| **N6** missing question in the legend (v21_missing_role, log 2026-10-17_1343; RESULTS 15) | remaining roles 1.000; relative residual 0.337 WITH the hidden role vs 0.462 WITHOUT (gap **−0.267**); recovery of the hidden filler from the residual: true R_C **0.110** (chance 0.083), random 0.093, the reader's own calibrated address **0.963** (5/5 seeds > 0.92); residuals cluster by filler (cos 0.19 same / 0.05 different), 1-NN with no vector 0.57 | addendum: gap > 0.3; oracle recovery > 0.8; random at chance | **NOT met as written** (third branch of the rule): the residual SHARE does not expose the missing question, and a reader trained without the role does not put it at the legend's vector; **detectable by residual DIRECTION** — the filler sits at the reader's own address, consistently across sentences (the tool's `_unplaced`-by-direction loop is that detector; test 5b) |

## B. HYPOTHESIS
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
2. (answered by 15-bis) the function-word hypothesis is dropped; the share inverted because the recipient-less forms fit worse. Nothing to decide.
3. (answered by judgment 934fbae E) the agent does E1 (legend-gap statistics) and E2 (WikiSQL through the tool), then waits for the data.
4. Round 6: 0.699 of the product's presence errors abstain pooled, but only 0.476 in seed 3 (its errors are missed "present" with ratios far
   below the threshold). Is "most, pooled" enough for the adoption as written, or does the per-seed spread call for the asymmetry field now?
