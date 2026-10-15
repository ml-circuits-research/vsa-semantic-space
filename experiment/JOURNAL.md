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

## 2026-10-15 (DGX, afternoon) — English world, left expectations (§5.1b), compositional nesting; all documents translated to English
Ran: run.sh v5_scale.py --lang en                 -> logs/2026-10-15_1246_v5_scale_9ea27cd.txt  (language-switch reproduction check)
     v6_left.py (manual run.sh header; run.sh refused the tree dirtied by the translation renames) -> logs/2026-10-15_1249_v6_left_9ea27cd.txt
     run.sh v7_nested.py --variants main           -> logs/2026-10-15_1311_v7_nested_043caf8.txt  (compositional world, 3 outer forms)
     run.sh v7_nested.py --variants main --outer prefixed -> logs/2026-10-15_1319_v7_nested_e1e7011.txt
Result: (6a) English world reproduces seen / new ids / UNSEEN transitions / depth 1-2 within +-0.03; "seen transitions" 0.785 vs 0.998 because
  T10 'from A , C receives B' has C at 0.04: the only training formulation with a person after a comma (T9) has the AGENT there. The axis
  measures formulation-set redundancy. (6b) Left expectations: inner roles at depth 2 -> 1.00, speakers 0.65 -> 0.77; criterion X2 > 0.9
  NOT met; derived the algebraic contradiction of the flat world (a+g = 0 ... => X2 = X1). (6c) Compositional world + prefixed reporting:
  depth 2 = 1.00 all roles (plain chain!), UNSEEN depth 3 = 1.00 all roles with chainL. With the postposed form present: 0.82-0.88 / ~0.5.
Surprise / problem: the round-5 "nesting >= 2 unsolvable from the prefix" was about the WORLD (flat roles), not the model. The precise barrier
  is right-announced structure (a displacement announced after its scope). Both runs via run.sh failed once because run.sh saw the
  translation's git mv as a dirty tree; one run launched manually with the run.sh header (noted in the log).
Changed in code: v5_scale.set_lang(en|ro) (English BigWorld: 16 patterns, prefixed/postposed reporting), v6_left.py (chainL), v7_nested.py
  (NestedRoleWorld, decode by unbinding conj(R_rep)^k, --outer prefixed), diag runs. Docs: all Markdown translated and renamed (AGENTS.md rule:
  English on disk, Romanian in chat); RESULTS ROUND 6 (a, b, c); 00_READ_FIRST §4 (positive on unseen construction; precise negative), §5 (1b done, 1c new).
Next: 1c (clause-level retroactive displacement for right-announced structure), then §5.2 gates by gradient on the compositional world.
  Decision requested: is the compositional world the reference world from now on (it matches the original "fibre" idea; flat was ours)?

## 2026-10-15 (DGX) — decisions after the round-6 judgment
Decided (author + reviewing session): (1) the COMPOSITIONAL world (v7_nested) is the reference world from now on; the flat world is kept only
to reproduce rounds 5-6b. (2) Next is §5.2, gates by gradient, on the compositional world; §5.1c (postposed / right-announced structure)
is postponed and will be declared in the first paper as a known limit of causal addressing, with the demonstration of why.
Also requested, before or alongside §5.2: (a) a control on test 6c — small transformer (torch, minimally tuned: lr sweep, dropout),
trained on depths 1-2, tested on depth 3, same data, same fixed identities, same objective; per role, with margin, 5 seeds;
(b) rebuild the "new formulations, seen transitions" axis with formulations whose ROLE-BEARING transitions were all seen; keep it reported;
(c) describe chainL as the code does (re-addressing), not as union-then-restriction — done in RESULTS 6b and v6_left.py.
Criterion for §5.2: gates learned by gradient converge to the same set as by search (identities closed, verbs/prepositions open) on >= 3
worlds/seeds, with no loss in the round-6c table.

## 2026-10-15 (DGX, late afternoon) — round 7: gates by gradient, transformer control, rebuilt axis
Ran: run.sh v8_gates.py (5 seeds, 6 variants, 134 s/variant soft chain) -> logs/2026-10-15_1412_v8_gates_06e12bd.txt
     run.sh v9_transformer.py (grid on seed 0, then 5 seeds)            -> logs/2026-10-15_1414_v9_transformer_75e7e0d.txt
Result: (7c) gates emerge by gradient, identical sets in every seed: P, O closed; to/from/by/that/comma/COP open; ALL verbs closed; no loss
  at depths 1-3 (1.00), better on new formulations (bundle 0.78/0.76 -> 0.85/0.87; chainL alone role cues 0.22 -> 1.00). Criterion half
  met: agrees with search on what is closed, disagrees on verbs; the gradient's set is better on every axis. (7b) transformer control:
  new ids 0.215 (chance 0.167), depth 3 unseen 0.39 (A/B/C at chance, X1/X2 1.00), depth 2 seen A/C ~0.49; flat axes comparable.
  (7a) role-cues axis: T8, T13 seen; T10 moves to unseen; every bundle at 1.000 on it.
Surprise / problem: verbs closed — with left re-addressing a verb has nothing to emit that the prepositions and 'that' do not give; the round-2
  search needed verbs because only emitters could place the words after them. run.sh refused once more (porcelain paths are repo-relative,
  the 'logs/' filter had a leading space) and the refusal went to /dev/null; fixed, and run.sh output is no longer discarded.
Changed in code: v8_gates.py (soft chain, gates as logits, chainL-alone variants), v9_transformer.py (control), v5_scale.split_templates_rolecues,
  run.sh filter. Docs: RESULTS ROUND 7 (a, b, c), 00_READ_FIRST §4 (gates CERTAIN; control) and §5.2 DONE, current_status rewritten.
Next (proposals for judgment): (1) word2vec control for categories and the bottleneck associative-memory control (§5.3-5.4) — both cheap now;
  (2) capacity (D 4096-16384, 64-1000 identities) on the reference world; (3) then WikiSQL with the full model + BoW+LR baseline (§5.7).
  Writing: the principle note (POSITIONING article 1) now has its two pillars — composition to an unseen depth (6c) and emergent gates (7c) —
  plus the controls (7b) and the declared limit (right-announced structure).

## 2026-10-15 (DGX) — decisions after the round-7 judgment
(1) The partition identities / relation markers / action words is the reference; the LEARNED gates (start closed: that, to, by, '.') replace
the hand set in all subsequent runs; the hand set stays as a table row (the ablation showing the linguistic prior was not optimal).
(2) Order: cheap controls first (word2vec for categories, bottleneck associative memory) — without them "emergent" is attackable; then
WikiSQL (decides whether there is an empirical paper) before capacity (a curve that changes no decision); vectorize the soft chain with WikiSQL.
(3) Start the principle note now, in parallel: two pillars (composition to an unseen depth; emergent gates), a control failing on exactly
those axes, a declared limit with demonstration (postposed structure), the thread of the two retractions; with the caveats (own world,
small control, two reporting verbs). In the gates section: a concrete TRACE — the same sentence with hand gates and with learned gates,
addresses word by word.

## 2026-10-15 (DGX, evening) — round 8: controls on "emergent", WikiSQL with the full model, principle note started
Ran: run.sh v10_controls.py (5 seeds) -> logs/2026-10-15_1530_v10_controls_727aecb.txt ; run.sh v11_wikisql.py -> logs/2026-10-15_1534_v11_wikisql_2fb26b9.txt
     trace_gates.py -> logs/trace_gates_hand_vs_learned.txt ; WikiSQL downloaded to corp/wikisql (gitignored)
Result: (8a) word2vec recovers the categories as perfectly as the learned displacements (both 1.000): "emergent categories" RETRACTED as a
  contribution, kept as a consistency check. Bottleneck associative map: new ids ~0.24 (chance), depth 3 0.25-0.39 with the transformer's
  signature at every rank 8-1024: "generative with a fixed operation is necessary" stands. (8b) WikiSQL, question only: selected column
  0.824 vs 0.494 BoW+LR (and 0.484 overlap-only), condition column 0.631, aggregator 0.813 vs 0.867 BoW+LR (below), separation 0.28;
  header at the input hurts (sel 0.67). Learned gates on real text: than/and/has/the/what/which/of/is open, content clusters closed — the
  synthetic partition reappears on real questions.
Surprise / problem: the aggregator is lexical and the bag of words beats us on it; the table as input dominates the bundle (needs a role, not a sum).
Changed in code: v10_controls.py, v11_wikisql.py, trace_gates.py. Docs: RESULTS ROUND 8 (a, b); PAPER_NOTE.md (draft with §6 filled, trace in §3.1);
  POSITIONING contributions updated; 00_READ_FIRST §4 (retraction), §5.7 started; current_status rewritten.
Next (for judgment): WikiSQL 3 seeds + test split + tuned neural baselines (transformer, BERT-small) at 1k/5k/56k + an aggregator readout;
  write the empirical paper outline only if the advantage survives the tuned baselines. Principle note: tighten prose, add figure 1.

## 2026-10-15 (DGX) — propagation of the 8a retraction
Propagated the 'emergent categories' retraction to INTERPRETATION (point 3 -> gates), 00_READ_FIRST (§3 I, §4, §5.3-5.4), POSITIONING (claim sentence,
contributions, controls), INITIAL_ANALYSIS_DGX, README_RUN, README_DGX, current_status; rule 16 added to EVALUATION_PRINCIPLES. Wording everywhere: categories are recoverable from distribution (skip-gram gives them equally well); the model CONSUMES them as input, it does not produce them; what emerges from the objective are the gates (who opens a frame), 7c and 8b.
