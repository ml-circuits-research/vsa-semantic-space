# Current status (rewritten whole at every judgment point; last: 2026-10-17 — judgment d8f8a98 applied: documents, B7 settlement, tool round 5)

Labels (rule 22): BUILT / PLANNED / HYPOTHESIS. Sources: experiment/JUDGMENT_2026-10-17_d8f8a98.md (saved verbatim), experiment/RESULTS.md 14-bis,
11-sexies, 10c-nonies, 14; STATUS.md; experiment/JOURNAL.md; tool/README.md; tool/runs/round5_1ed4306/round5_table.md.

## A. Done since d8f8a98, in the instructed order
| step | result | criterion / expectation (written before) | outcome |
|---|---|---|---|
| documents (Q1, Q2, B1, Q4) | 10c-nonies: for slot 1 the marker carries no presence signal (its cosine is higher on the absent class), the filler cleanup does; 10d labelled (A) — depth unseen, hops seen; reason: 'the' between the openers; RESULTS 14's depth-3 sentence marked as a hypothesis, then withdrawn by the trace; B6 adoption deferred (4-bit route book as an EXPORT option when size is asked for; chart sparsity not adopted) | — | done |
| **B7 settlement** (rescore_v16_b7, log 2026-10-17_1225; RESULTS 14-bis; saved three- and four-code models, no training) | (a) the +0.051 on unseen transitions is a surface-position PRIOR: T11 +0.37 (object at sentence-final position 5, as in training templates T6 / T9) against T4 subject −0.71 and T7 recipient −0.44 — not compositional; (b) omitting the position term beyond the trained range (positions >= 15) gives identical digits on every axis: the untrained positions contribute nothing; depth 3 0.977 -> 0.998 on the same draws comes from the trained positions; (c) terms per sentence +33% (depth 3: 66 vs 49; P1 lengths 80-120 vs 60-90) — the four-code depth-3 bundle already exceeds B6's 64 and reads 0.998, so the synthetic curve is not the reader's binding constraint | (b): no loss, possibly a small gain | (b) met on the no-loss side (identical); the omission rule adopted as the position code's stated behaviour (V8 `pos_max`); **the reference decision is the reviewing session's** (D, 1) |
| **Tool round 5** (agent; RESULTS 11-sexies; tool/runs/round5_1ed4306/) | stored presence ratios, role recipient: pooled midpoint (product) 0.639 present / 1.000 absent; threshold FITTED on the calibration rows 0.871 / 0.912 (balanced 0.820 -> 0.892; ceiling 0.908); overlap ~10% of present ratios below the absent 90th percentile; a 0.9 / 0.9 threshold exists in 2 of 5 seeds | the judgment's rule, verbatim in the header | **OUTCOME by the rule: the READER's limit** (no 0.9 / 0.9 in 3 of 5 seeds); but most of the round-4 gap was the midpoint RULE: the fit takes the product 0.64 -> 0.87 present at 0.91 absent — adopting the fit (a stated trade: ~9% of recipient-less sentences then read as having one) is for the reviewing session (D, 2); nothing changed in the tool |

## B. HYPOTHESIS
- Presence calibration across formulations: ~10% overlap of the present and absent presence-ratio classes on unseen formulations (reader); the
  midpoint rule adds a further third of missed recipients on top (rule). Fit vs midpoint is a product decision (D, 2).
- B5d multi-hop left re-seating: needed only if a legend requires the subject's term as a structure witness.
- Verifier condition axis (C5 closed): the later slot roles have too few examples; the legend choice (indexed slots vs one condition role with a
  SET value) is made when P1's legend is written.
- N5 "resolve before drawing": diagnostic to print in the next kinds run.

## C. PLANNED, in the instructed order
N6 on the settled bundle (after D, 1); N8 (verb-final world, low priority); P1 when the team's data arrives (the annotation path once with a key,
wherever the team runs it); N7 after N1; B8-B9; capacity at 64-1,000 identities and D = 4,096-16,384 (00_READ_FIRST §7); the position code in
`sop train` only if the reference changes; the 4-bit route book as an export option when a deployment asks for size.

## D. Decisions for the reviewing session
1. The reference bundle (B7): the numbers of 14-bis — the unseen-transition gain is a surface-position prior with wins and losses by template
   (T11 +0.37; T4 −0.71 on the subject; T7 −0.44 on the recipient), the depth-3 gain 0.977 -> 0.998 is real and comes from trained positions, the
   untrained positions are inert under the omission rule, terms per sentence +33%. Three codes (10b-quinquies) or four (B7)? N6 waits on this.
2. The tool's presence calibration: keep the pooled midpoint (product today: recipient 0.64 present / 1.00 absent on unseen formulations), or
   adopt the threshold fitted on calibration rows that include held-out formulations (0.87 / 0.91, balanced 0.89, ceiling 0.908) — a stated
   trade of ~9% false "present" on recipient-less sentences for a third more recipients read? By the rule as written the limit is the reader's
   and is recorded so; the fit is a separate product decision.
3. B6 facts for the books (on your side): the chart must keep its magnitudes; 32 terms clean / 64 failing at D = 1024 for synthetic bundles — with
   the caveat from 14-bis (c) that the reader's structured depth-3 bundles read 0.998 at 66 terms.
