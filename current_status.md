# Current status (updated at every judgment point; last: 2026-10-15 evening, commit 965332d)

## What was done today on the DGX
- Environment: .venv with numpy 2.5.4 and torch 2.14.1+cu130 (aarch64 + CUDA, GPU GB10 visible).
- Document cleanup (a1ff20b): alignment to rounds 2-3; "algebraic impossibility" and "deferred addressing" retracted; confidence level corrected.
- Round 4: v4.py numpy on the DGX CPU reproduces round 3 EXACTLY (3 decimals). v4_torch.py (GPU) gives scores IDENTICAL to numpy on all 24 runs.
  Time/variant: old CPU ~77 s -> DGX CPU 24.8 s -> GPU batch=1 28.5 s (no gain: Python) -> GPU encoding once + batch 32: 2.2 s.
  Batch 8-32 preserves the conclusions for "all 4" (+-0.03); the weak codes lose at batch 32.
- Round 5 (§5.1): v5_scale.py, 3x world (6 frames x 16 formulations, 18+18 identities, nesting 0-3), 5 seeds, D=1024, 30k, batch 8.

## Round 5 results (5 seeds)
| code | seen | new id. | new form. seen trans. | UNseen trans. | nesting depth 1-2 | depth 3 |
|---|---|---|---|---|---|---|
| path+chain+bigram | 1.000 | 1.000 | 0.998 | 0.821 | 0.938 | 0.673 |
| all 4 | 1.000 | 1.000 | 0.841 | 0.984 | 0.943 | 0.687 |
| (small world, round 3) path+chain+bigram | 1.000 | 1.000 | 0.987 | 0.824 | — | — |
| (small world, round 3) all 4 | 1.000 | 1.000 | 0.942 | 0.901 | — | — |

## Conclusions relative to the goal
1. REPRODUCED: path+chain+bigram is stable across worlds (0.99 / 0.82). New identities 1.00. This is the model that holds.
2. NOT reproduced: "all 4" inverts. Cause: the pos code = absolute position prior. It learns from T1 "person at position 5 = agent";
   in T10 the recipient is at 5 => pos votes completely wrong (A->C, 0.15) and beats the overlapping addresses of the first word. Its gain on
   unseen transitions comes from new categories ('catre', 'il') where the other codes have nothing. Both belong to the set of formulations, not to the mechanism.
   PROPOSAL: we report the model = path+chain+bigram; pos separately, as a prior.
3. NEGATIVE AND CLEAN (the most important): nesting of depth >= 2 is unsolvable with addresses from the prefix.
   '{X1} spune ca {A} V ...' and '{X1} spune ca {X2} zice ca ...' have the identical prefix 'P spune ca P' => any causal code gives the same place for
   A (depth 1) and X2 (depth 2), the world requires different roles => ~0.5 (X2 0.28-0.53 for any code, confused with A), EVEN on seen data.
   D does not help, data does not help. Round 2 retracted "deferred addressing" too much: there is no need for "the verb writes the buffer", but there IS a need for a
   word to be re-addressable by what follows. The first proof forced by data. It changes figure 1 and the article's contributions.
4. Design flaw caught: at depth 3 the role X3 appears, never a target in training => at chance by construction (0.08 ~ 1/12);
   the "depth 3" figure mixes it with the real degradation of A/X1/X2. Became rule 14 in EVALUATION_PRINCIPLES.
   New rules: 13 (same optimization regime in comparisons), 15 (breakdown per formulation x role before interpretation).

## Questions for the other session's judgment
1. Do we accept path+chain+bigram as "the model" and pos as a separate prior?
2. Next step: §5.1b (retroactive re-addressing: a placed word can be moved when a new emitter appears; criterion X2 > 0.9 at depth 2
   without losses elsewhere) BEFORE §5.2 (gates by gradient)? Argument: the gates depend on the same expectation mechanism.

## Relevant files
experiment/RESULTS.md (rounds 4, 5) | experiment/JOURNAL.md | experiment/EVALUATION_PRINCIPLES.md (13-15) | experiment/00_READ_FIRST.md §4-5
experiment/v4_torch.py, v5_scale.py, diag_v5.py | experiment/logs/2026-10-15_1128_v5_scale_0defffd.txt
