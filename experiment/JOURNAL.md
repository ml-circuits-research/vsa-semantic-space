# Work journal (one entry per session; format in GIT_WORKFLOW.md §3)

## 2026-10-14/15 — initial session (Claude + author), CPU, numpy
Ran: sim_deplasari.py, camp.py, k.py, control_transformer.py, mlp_control.py, sick_test.py, wikisql_test.py, v2.py, v3.py, gates.py, nest2.py, v4.py
Result: see RESULTS.md (3 rounds). In short: factorization + generative objective => aligned roles; learnable gates; redundancy 0.60 -> 0.90.
Surprise / problem: two evaluation mistakes in round 1 (addresses from the active form; different F world/model) produced false "necessary rules".
Changed in code: evaluation redone (v2.evaluate: addresses from the world); EVALUATION_PRINCIPLES.md written.
Next step: torch port; v4 with 5 seeds; WikiSQL with the full model + BoW+LR baseline.

## 2026-10-15 (DGX) — document cleanup: alignment to rounds 2-3, no code, no runs
Ran: nothing (no figure changed; source of truth: 00_READ_FIRST.md and RESULTS.md rounds 2-3; whatever contradicts round 2 is marked RETRACTED/SUPERSEDED, not deleted).
Changed: POSITIONING.md (contribution (i) -> overlapping addresses + argmax; fig.1 (b) -> redundant codes; learned gates = new contribution;
  "hand-set triggers" barrier marked retracted); README_DGX.md (old plan marked SUPERSEDED; pointer + port + data sources remain);
  README_RUN.md (FINAL MODEL = v4.py; camp.py = the world + historical v1); RESULTS.md round 3 (note: all 4 0.94 < bigram 1.00 on seen transitions);
  00_READ_FIRST.md §4 (nesting and learned gates moved from CERTAIN to PROBABLE); logs/README.md (old logs = pre-run.sh); INITIAL_ANALYSIS_DGX.md in the root.
Next step: environment (venv, numpy, torch CUDA aarch64) and the first new run ./run.sh v4.py on the GPU, in a separate commit.

## 2026-10-15 (DGX, afternoon) — environment, reproduction on the DGX, torch port, batch
Ran: run.sh v4.py (numpy CPU)              -> logs/2026-10-15_1101_v4_d26b7db.txt        (10 min; 24.8 s/variant)
       v4_torch.py GPU batch=1              -> logs/2026-10-15_1107_v4_torch_c57d74a.txt  (launched manually with the run.sh format: run.sh refuses a
                                                dirty tree, and the still-uncommitted log of the CPU run was dirtying it; sequential from here on)
       run.sh v4_torch.py --batch 32 / 8    -> logs/2026-10-15_1119_v4_torch_f8fedc5.txt / ..._1120_v4_torch_fc97386.txt
Result: numpy on the DGX reproduces EXACTLY the round 3 table (3 decimals, 8 variants). torch GPU batch=1 gives scores identical to numpy on all
  24 lines (accuracy + margin): the port is numerically equivalent. Batch 8/32: all 4 -> 0.885/0.887 on unseen transitions (vs 0.901),
  within the +-0.03 margin; the weak codes (path, path+chain) lose at batch 32. Full table in RESULTS.md "ROUND 4".
Surprise / problem: at D=512, batch=1, the GPU is not faster than the CPU (28 s vs 25 s/variant): cost = Python + small steps.
  With encoding once + batch 32: 2.2 s/variant (10x), of which 1 s is still Python (the encoding). The torch wheel is the right one (cu130, aarch64).
Changed in code: v4_torch.py NEW (the address structure remains from v2/v4 numpy; phases -> terms -> cos -> autograd in torch; --batch, --lr,
  --seeds, --n_pers, --D, --device); root .gitignore (.venv). Environment: .venv with numpy 2.5.4, torch 2.14.1+cu130.
Next step: 00_READ_FIRST §5.1 — 2-3x world (4-6 frames, 10-20 formulations, nesting), 5 seeds, with v4_torch batch 8-32;
  then §5.2 gates by gradient.

## 2026-10-15 (DGX, evening) — §5.1: 3x world, 5 seeds (v5_scale.py) + diagnostic (diag_v5.py)
Ran: run.sh v5_scale.py (5 seeds, 7 variants, D=1024, 30k, batch 8) -> logs/2026-10-15_1128_v5_scale_0defffd.txt (12.7 min)
       diag_v5.py + breakdown per formulation x role and per code at depth 2 (seed 0; outputs in RESULTS.md round 5)
Result: path+chain+bigram REPRODUCED across worlds (0.998/0.821 vs 0.987/0.824). all 4 NOT: 0.841 / 0.984 (inverted relative to 0.942 / 0.901).
  Nesting: depth 1 1.00; depth 2 (seen!) ~0.85, X2 0.45 confused with A; depth 3: X3 at chance by construction (never a target).
Surprise / problem: (1) pos = absolute position prior: T1 is the only training pattern with a person at pos. 5 (agent) => in T10 (recipient
  at 5) pos votes completely wrong and beats the overlapping addresses of the first word. Its gain on unseen transitions comes from new categories
  ('catre', 'il') where the other codes have nothing. Both are properties of the set of formulations.
  (2) MORE IMPORTANT: '{X1} spune ca {A} V..' and '{X1} spune ca {X2} zice ca..' have an identical prefix ('P spune ca P') => any address from the prefix
  gives the same place for A (depth 1) and X2 (depth 2) => overlap ~0.5. Nesting >= 2 is unsolvable with causal codes, regardless of D/data.
  The "deferred addressing" retracted in round 2 comes back, in another form: the re-addressing of a word by what follows. The first clean proof that it is necessary.
Changed in code: v5_scale.py (BigWorld, BigRoleWorld, V5 with pos up to 40), diag_v5.py. Documents: RESULTS round 5; PRINCIPLES 13-15
  (same optimization regime; new role => report per role; breakdown per formulation x role before interpretation); 00_READ_FIRST §4-5
  (nesting >= 2 NEGATIVE; §5.1 done; §5.1b new: learned deferred addressing); POSITIONING (barrier partially reopened).
Next step: §5.1b (X2 at depth 2 > 0.9 without losses) before §5.2, because the gates by gradient depend on the same expectation mechanism.
  Question for judgment: do we report 'the model' as path+chain+bigram (0.99/0.82) and pos separately as a prior? I say yes.
