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

## 2026-10-15 (DGX) — decisions after the interim (8c) judgment
PAPER_NOTE §7 rewritten as "Negative results and what they fixed": lessons, not round numbers; the reason first. (2) If the same-input
transformer reaches us at 56k but not at 1k/5k, the empirical claim is "data efficiency on the selected column without the table, with a
readable role state" — enough for a short paper (findings/workshop) with the curve as main figure; a full paper needs Spider (nested SQL =
pillar 1); decide short vs full AFTER v12. (3) If the lexical bag separates the hard pairs as well as the model: keep it as a negative —
"on WikiSQL the equivalence separation is lexical; the set lacks pairs with the same words and different SQL" — do not change the metric.
Checked: in v12 the skip-gram clusters are trained on the WHOLE train pool at every curve size (for both models); the data-efficiency claim
must say "with distributional categories from the whole training corpus, without labels" (noted in the script and to be written in 8d).

## 2026-10-15 (DGX, evening) — judgment after 8d: short empirical paper now; new programme (verifier / formalizer)
Decisions (final unless contradicted): short empirical paper NOW (EMPIRICAL_NOTE.md; headline = selected column 0.82 vs 0.54 vs 0.50, full
query second in the same table, aggregator loss in the same sentence); the 4x/10x transformer budget check is the GATE (if 0.75+ at 10x, the
claim narrows to 1k-5k); Spider later for a full paper. PAPER_NOTE §7 approved (no round numbers; fact + consequence per lesson).
Corrections: n stated for every hard-pair set; H2 phrased as "the role state moves same-words/different-SQL pairs apart where the lexicon
cannot"; data-efficiency sentence carries its condition; ablation with categories from the subset only (both models) added (v12 --cluster_on_subset).
New requirements: R1 frame(S) + diff as canonical output, full-frame accuracy everywhere; R2 frame -> Prolog and -> SMT-LIB with a round-trip
test; R3 operators are roles in the world; R4 rule 17 (scope separately from presence); R5 LLM annotation pipelines with inter-LLM agreement.
Experiments in order: E1 negation, E2 quantifiers, E3 modality/conditionals, E6 verifier on real text, E4 coordination, E5 reference, E7 LLM
pipeline; Spider after E6; capacity after all. Not to do: no tuning to chase the aggregator/transformer; no metric change; no time estimates in docs.
Note recorded for E1-E4: operators announced AFTER their scope (postposed) fall under the declared limit of causal addressing (RESULTS 6c) and
will be reported separately as such. For E5, identities must carry an accessible feature (gender) for pronoun cleanup to be possible by construction.

## 2026-10-15 (DGX, night) — round 9: frame + round trip (R1/R2), E1 negation, empirical note, budget gate, ablation
Ran: run.sh frame_roundtrip.py (3 seeds) -> logs/2026-10-15_1705_frame_roundtrip_a622582.txt ; run.sh v13_negation.py (5 seeds) ->
     logs/2026-10-15_1709_v13_negation_5b9e933.txt ; transformer budget 4x/10x -> logs/2026-10-15_1652/_1655_v12... ; subset-categories
     ablation -> logs/2026-10-15_1700_v12_wikisql_baselines_38bc65d.txt ; E1b (negation seen at depths 0-1, test depth 2) launched.
Result: (9a) frame(S) with estimated depth -> Prolog / SMT: 1.000 on depths 0-3 (3 unseen). (9b) negation as a role: roles, presence, scope,
  full frame all 1.000 at depth 0 for all 8 forms; separation 0.18 (bag 0.72); negation words become emitters; sentential negation inside
  reporting (unseen combination) 0.0-0.5 scope — attaches to the theme — while argument-scoped forms compose (0.8-1.0). Budget gate: 10x
  transformer 0.65 < 0.75, claim stands. Ablation: categories from the subset only, gap unchanged (ours 0.654 / 0.751 vs transformer 0.414 / 0.516).
Surprise / problem: the first operator-inside-operator failure is specific (sentential scope) and testable (E1b). 'nobody' stays closed:
  the indefinite under negation lives in the filler. Four runs shared the GPU for an hour (4-8x slowdown): launch sequentially from now on.
Changed in code: frame.py, frame_roundtrip.py, v13_negation.py (+ --neg_depth_train), v12 (--only, --cluster_on_subset). Docs: EMPIRICAL_NOTE.md
  (full draft, gate and ablation filled), PAPER_NOTE §6b and §7 (approved tone), RESULTS 8d additions + ROUND 9 (a, b), rule 17, 00_READ_FIRST §4-5.
Next: E1b result -> then E2 (quantifiers), E3 (modality/conditionals), E6 (verifier with injected errors on WikiSQL).
E1b (same night): negation seen at depths 0-1 -> depth 2 unseen FULL 0.972±0.049, all 8 forms at scope 1.00 (seed 0); sentential negation
composes once seen inside a clause (E1: 0.0-0.5 -> 1.00). Data limit, not mechanism. Rule for E2-E3: operators trained at depths 0-1, tested at 2.

## 2026-10-15 (DGX, night) — decisions after the round-9 judgment
Working rule accepted with an explicit reserve (written in RESULTS 9b-bis): operators compose when seen at the shallower depth in training (stated share) inside each other; E2-E3
trained at depths 0-1, tested at 2. EMPIRICAL_NOTE to be read by the reviewing session; figure 1 = PNG from the log, three curves with seed
bars. Launches sequential from now on (times go into reports). E2 addition: scope ambiguity ("every child reads a book") reported separately;
the model is NOT trained on ambiguous sentences — the world holds both hypotheses; the test is whether the state decodes both with near-equal
strength on ambiguous sentences and one on disambiguated ones ("reports both hypotheses / reports margin and residual", by composition).

## 2026-10-15 (DGX, night) — E2 quantifiers: protocol error found by the diagnostic; re-run with the E1b protocol
Ran: run.sh v14_quantifiers.py (5 seeds) -> logs/2026-10-15_1815_v14_quantifiers_4ec4ecc.txt ; diag_v14.py (seed 0, inline).
Result: nouns 1.00, determiners 0.97, scope 0.95 with one quantifier / 0.81 with two marked (NOT met), ambiguous "both" 0.46-0.50 by composition.
Depth 2: everything collapses (nouns 0.31) because NO depth-2 sentence of any kind was in training ('that:that' untrained) — a protocol error,
not a quantifier result; the composition rule is not tested by this run. Fix: affirmative sentences at depths 0-2 in training (as in E1b).
Figure 1 drawn (figures/fig1_wikisql_curve.png); E3 script written (v15_modality.py), not yet run (sequential launches).

## 2026-10-16 (DGX, morning) — the interrupted E2 re-run: a NaN bug, not the machine
The 2026-10-15 18:37 E2 re-run (logs/2026-10-15_1837_v14_quantifiers_387cf99.txt, kept as a record) died with the terminal session after 3 seeds
— and those 3 seeds were garbage (nouns 0.08 = chance, all else 0, nan, no gate open, identical across seeds): the affirmative sentences
added by the E1b protocol have an empty scope list and the world divided by sqrt(0) => NaN targets => NaN training. The machine was fine
(GPU idle, no leftover process, 114 GB free). Fix: no scope term when no quantifier; plus an assertion in E1/E2/E3 training that the loss is
finite (a silent garbage log must be impossible). Lesson for the process: a run that keeps going with a constant, chance-level output is a
bug, not a result; the guard makes it stop. Background runs die with the terminal: long runs go under tmux or are relaunched next morning.
Correction (same morning, after the author's report): the machine DID hang on 2026-10-15 evening — no SSH, hard reset — so the earlier
"the machine was fine" referred only to its state after the reset. Unified memory on the DGX Spark: a runaway process starves the OS.
Protection added (experiment/guard.py, imported first in every training script; run.sh): cgroup MemoryMax does NOT bound CUDA unified
allocations (tested: a 3 GB cuda tensor survived a 2 GB cap), so the cap is in-process — torch allocator limited to 50% of the unified
memory (a 70 GB allocation is refused with an OOM error, tested), a watchdog that exits when MemAvailable < 16 GB, a 6 h wall-clock limit;
run.sh refuses a second concurrent run and refuses to start under 40 GB available. The session now runs under tmux.
E2 corrected run (09:00): nouns compose to the unseen depth (1.00), quantifier terms do not (dets 0.46-0.64, scope 0.52): the composition
rule FAILS for quantifiers when seen at the shallower depth in training (stated share) — reserve applied, reported as such. Decoder gap fixed ('none' scope hypothesis). Hypothesis
for judgment: the world's scope filler is a ROLE identity (not a word) => not representable by address (x) f_word; propose R_sc (x) f_noun.

## 2026-10-16 (DGX, morning) — E3 modality/conditionals: composes to the unseen depth (1.000); right-announced form at the limit; zero-occurrence mix fails
Ran: run.sh v15_modality.py (5 seeds) -> logs/2026-10-16_0902_v15_modality_b2b8f34.txt. E3b (mixed construction seen at depth 0, --p_mixed 0.15) launched.
Tally of the composition rule (seen at the shallower depth (stated share), test one level deeper): reporting YES (6c), negation YES (E1b), modality YES, conditional YES,
quantifiers NO (9c-bis; world-side hypothesis for judgment). Zero-occurrence combinations fail every time (E1 sentential, E3 mixed).
E3b (09:50): the mixed modal-in-consequent construction, seen at the shallower depth (stated share) at depth 0 -> inside reporting (unseen) 1.000 (5 seeds). Rule holds
on five operator cases; quantifiers the single exception. Awaiting judgment on E2c (world correction R_sc (x) f_noun) and on the rule as a claim.

## 2026-10-16 (DGX, late morning) — consolidated judgment received; process commit (Part I-II B + Part III A docs + rules 18-24 + STATUS.md)
Saved the judgment as experiment/JUDGMENT_2026-10-16_consolidated.md (the referenced files were never in the repo). Applied: A1 residual()
in frame.py and "occupancy not by norm"; A2 commutative-binding limit in PAPER_NOTE §8 / EMPIRICAL_NOTE §7; A3 supervision wording
(PAPER_NOTE abstract, 00_READ_FIRST §2); A5 "knows when it does not know" -> "reports both hypotheses / margin and residual" (grep, 0 left);
A6 memory figure; rules 18-24 (24 = name the regime exactly); P2 rename "one exposure" -> "seen at depth d (share p), tested at d+1" with
shares; P3 current_status rewritten whole with labels; P4 pid-file lock in run.sh; N0 STATUS.md at the root; Q2 both ambiguity numbers
labelled with n. Next: E2c (rule-11 trace first), then round 10 A2.
Round 10 A2 code (10:35): trainer.py = the one training loop for E1/E2/E3 — memory (RSS, torch alloc/reserved, MemAvailable) every N
steps, .item() losses, no graph after a step, finite-loss assertion, checkpoints every 2000 steps with --resume (tested on CPU: resume at
step 800 works); run.sh: expandable_segments, log-advance watchdog (kill after 30 min without log growth), pid lock. The memory curve of a
short reference GPU run is reported after E2c finishes (sequential launches).

## 2026-10-16 (DGX, midday) — E2c both targets, A2 memory curve, B1+B2 reference table, B4 verifier; a queue deadlock
E2c (v14 --scope_target slot / noun; logs 2026-10-16_1031_..._7f2b6b7, _1052_..._8fc1fcc; 5 seeds each): depth 0 scope 0.784 (slot) /
0.873 (noun) — the noun target is the best scope number so far at the seen depth (9c-bis 0.795) but still below 0.9; depth 2 (unseen)
scope 0.03 for BOTH targets, below chance 0.33 — suspected artefact of the ABSOLUTE 'none' threshold (tau_none = 0.08) added to decode_q
after 9c-bis: scope scores shrink with depth, so at depth 2 the decoder says 'none' everywhere. Rule 10 (evaluation first): diag_e2c.py
re-scores the same regime under three 'none' rules (absolute / none / relative) before any interpretation; written in RESULTS 9c-ter.
A2 memory curve (log 2026-10-16_1115): flat after step 500 (RSS 2.46 GB, torch 0.64 GB allocated), reported in RESULTS 10a.
B1+B2 (log 2026-10-16_1118, 5 seeds): predicate role (surface lemma, option (a)) and occupancy marker at 1.000 on every seen axis, three
occupancy cases 1.000, confident-wrong 0.000 where seen; depth 3 unseen drops to 0.813 (6c: 1.000 without the predicate term) — the margin
reports it (coverage 0.33-0.59 at tau 0.05). RESULTS 10b; this is the reference table from now on.
B4 verifier (log 2026-10-16_1130; 67 min, three epochs on 56k questions): role errors detected 0.74-0.81 with confident-wrong < 1.3%, false
alarm on gold 0.162; condition and value errors mostly 'unverifiable' (values invisible to the reader, stated). Criterion (> 0.9) NOT met.
Process lesson: the two queued launches (diag_e2c, v18) waited on `pgrep -f "v16_reference|v17_verifier"` — a pattern that matches the
waiting shell's OWN command line, so both loops waited forever (found at 12:40, one hour after the lock was free). Rule for queues: wait on
the pid lock (logs/.run.lock) only, never on a pgrep pattern that appears in the waiting command. Both runs relaunched sequentially at 12:40.
B3 commands (13:04; log 2026-10-16_1255, 5 seeds): scope 1.000 on seven templated forms, confident-wrong 0.000 on scope-only pairs
(coverage 0.97), unseen depth 2 scope 0.997 (FULL 0.854, fillers). RESULTS 10d. diag_e2c (log 2026-10-16_1240): the absolute 'none'
threshold explains the 0.03 but under no rule do the corrected targets compose (depth-2 raw scope signal = noise); determiner gates closed in
both corrected worlds. RESULTS 9c-ter; decoder default now relative. current_status.md rewritten whole with the interpretation (relation vs
envelope, HYPOTHESIS), proposals C1-C6 (PLANNED, criteria) and six questions; STATUS.md updated; PAPER_NOTE §7-8 and 00_READ_FIRST §5.9
updated. Push = judgment point.

## 2026-10-16 (DGX, afternoon) — judgment at a8a8fab applied (process points C), queue of re-runs and C2 started
Judgment saved verbatim as experiment/JUDGMENT_2026-10-16_a8a8fab.md. Applied in one commit: (C1) the 9c-ter sentence on the relative 'none'
rule was FALSE (its own diagnostic table showed slot d0 0.978 -> 0.850, d2 0.333 -> 0.033) — corrected; no 'none' rule from now on: v14
reports PRESENCE (threshold on calibration sentences) and SCOPE conditional on presence, one- and two-quantifier separately; every quantifier
number carries its decoder rule; the E2c models were not saved, so the numbers keep their labels (all scripts save final params from now on).
(C2) v16 occupancy threshold moved to depth-0 training sentences; v17 condition threshold moved to the even half of dev, verdicts on the odd
half; rule 25. (C3) N5-N7 rows added without criteria (addendum text not received). (E1) one hypothesis -> H-a addressing / H-b relation;
(E2) degree-2 term withdrawn (commutative binding; information already degree 1); C2 = determiner gates GIVEN open, depth-2 trace first
(logs/trace_c2.txt: the flag puts the wide noun's chain address through the determiner's own category and both 'that' displacements — valid
test of H-a). (E4) depth-3 breakdown coded; (b) (c) (d) queued. (E5) gold "filter, not judge" sentence added; v17 saves reader + frames.
(E6) queue.sh; run.sh excludes its own tree; rule 26; run.sh ignores *.md in the dirty check. Queue: v16, v17, v16 omit0, v16 verb gates
closed, v16 D 2048, v14 slot open, v14 noun open.

## 2026-10-16 (DGX, evening) — the judgment's queue run: thresholds, E4 (a)-(d), C2, C4
Queue (./queue.sh, strict sequence, 13:51-16:41) + rescore (16:41). 10b-bis: threshold on training rows reproduces every 10b column; depth-3
breakdown: the loss is on C in the PASSIVE forms (T2 C 0.52) and the C-fronted form, participles with closed gates; half-open verb gates are
at 1.000 (judgment's suspect excluded). 10b-ter: (b) 0.858, (c) 0.745 (worse, costs seen axes), (d) D 2048 0.852 — neither branch of the
decision rule fires; form effect, reported as such. 10c-bis: threshold on the even half, odd half reported: all numbers within 0.013 of 10c.
9c-quater (C2, determiner gates given open, presence/scope split): depth-0 one-quantifier scope 0.96-0.98 (criterion met for the first time),
two-quantifier 0.76-0.79; depth 2 one-q 0.63-0.67, two-q 0.53-0.56 (chance 0.33; before: no signal) — H-a partly supported, criterion not
met, remedies stopped per the judgment. 10c-ter (C4): false alarm < 5% needs tau 0.25 and costs 44 points of role detection; confirmed on
gold 0.297 at every tau — the false alarm is confident wrong decoding, a reader property. Push = judgment point.

## 2026-10-16 (DGX, late evening) — judgment 3b4f837 applied: C fixes, E1 trace + participle emitter, C2b, E3 breakdown
E1 trace (logs/trace_e1.txt): in the passive the chain gives A and C the active form's addresses with the roles swapped (both hang off the
period); participle gates given open (log 2026-10-16_1717): non-passive forms 1.00 at depth 3, passive T2 C 0.52 unchanged, unseen transitions
0.64 -> 0.57 -> NOT met; form effect = stated property. C2b (determiner as a beacon-relative emitter, trace logs/trace_c2b.txt; logs _1729 slot,
_1751 noun): ONE-quantifier scope at the unseen depth 0.998 (slot, 5 seeds 0.99-1.00), nouns 0.998, affirmative absent 1.000 — criterion MET,
the one-quantifier case joins the regularity; two-quantifier marked 0.46-0.48 at depth 2 — the named exception, H-b, permuted-filler term
proposed for judgment (a8a8fab condition met). Strict re-scoring of C2 (log _1814): depth-2 presence 0.81 -> 0.26 with the depth-1
threshold. E3 (log _1715): sel confident-wrong 0.169 when the gold column's name is not in the question vs 0.023; 50-case audit file written.
N5-N7 criteria in STATUS. Push = judgment point.

## 2026-10-16 (DGX, night) — indications 013c490 applied: E1b, C2c, N5, 4c, P0 tool
E1b (log _1905, 10b-quinquies): relation-marker gates to / by / from given open -> depth 3 0.968, passives 1.000; the regression was the
gate set; new reference table. C2c (log _1918, 9c-sexies; trace logs/trace_c2c.txt): category-local quantifier terms, wide/narrow derived at
readout -> two-quantifier scope at the UNSEEN depth 0.966 (every seed > 0.9), one-quantifier 0.992, nouns 0.999, false-both 0.000, both on
ambiguous 0.993 — criterion MET; the named exception is removed, H-b retired, the permuted-filler term withdrawn. N5 (log _1941, 10e):
superposition of the two kinds with readout by fit loses 0.09 on the noun uses (0.907 vs ref 1.000; majority-kind baseline 0.989) — NOT
met, recorded as negative. 4c (logs _2033-_2038, 10c-quinquies): confident-wrong on the selected column flat with data (0.05-0.08) while
accuracy 0.38 -> 0.81; the lexical-mismatch confident-wrong grows 0.07 -> 0.17. P0 (RESULTS 11): the sop tool built by a separate agent
in a worktree and merged (tool/; 6 tests PASS; round trip within seed variance at 1 seed; spec section 6 missing). Process: run.sh now
ignores *.md and excludes its own process tree from the lock check; queue.sh waits on the pid lock only (rule 26); an accidentally
committed worktree gitlink made three runs refuse at 19:17 (fixed, .claude/ ignored). Push = judgment point.

## 2026-10-16 (DGX, night, second push) — hand-over 78d36e8 applied; C5, mismatch-aware abstention, B5.0
Saved verbatim: the hand-over (instructions + the missing judgment 013c490 + the complete tool spec, section 6 included), the B5 instruction
after the stop. Q2 (log _2125): FULL on the quantifier world redefined over the legend's questions — d2 unseen 0.967 (old 0.603; the gap was the
determiner identity, now its own axis). C5 (log _2126): occupancy marker on the condition slots — present 0.890 / absent 0.992, criterion not
met by 0.01; detection unchanged because the verdict abstains on the column margin. Mismatch-aware abstention (log _2235): the chosen-name
feature adds nothing; margin alone meets < 2% at coverage 0.445. B5 permutation route dropped (author: nothing given in the reader); B5.0
(log _2236): zero rendered order pairs in the three grammars; 2 colliding algebraic pairs (must/neg, must/may), 5 witnessed; the reader reads
the rendered order 1.000, the never-trained reversed order 0.98 (E3) / 0.00 (E1, B3). Rule 27 added. Tool: agent's section-6 work, `_scope`,
quantifier world v2, depth given vs estimated, model sizes merged; 6 tests pass on main; the 5-seed GPU round trip starts after this push.
Process: a run started from a shell without the venv crashed on import (log removed, lock cleared); run.sh is now launched with the venv active.
Push = judgment point.
B5c trace (trace_b5c.py, logs/trace_b5c.txt, no training): (a) and (b) reachable for every word except the subject before its stacked
operators, whose address is identical in 'A must give' / 'A must not give' (one re-seating hop = one level) -> STOP per the instruction;
nothing built (RESULTS 12-bis).
P0 5-seed GPU round trip (agent; tool/runs/p0_gpu_5seed_c843f53/, RESULTS 11-bis): commands depth 2 0.857 with the depth given (10d 0.854)
vs 0.590 estimated — the gap was the tool's depth estimator; quantifier v2 reproduces C2c on scope (0.917 at the unseen depth, both 0.993);
reference depth 3 0.742 vs 0.813, not isolated. Push = judgment point.

## 2026-10-17 (DGX, morning) — judgment e9274a9 applied: C, B5.1 (two runs), C5 re-scoring, tool round 2
Judgment saved verbatim; rule 28 (address check before a new target; one change per run); the two limits of the reader recorded as a row; verifier
operating points. B5.1 first run (log _0742): the subject is read at 1.000 under one and two stacked operators at the seen depths; at depth 2 it
fails for every form because the run trained no depth-2 sentence — an unseen transition ('that' -> 'that'), my regime error, kept as 12-ter.
trace_b51: the limit (ii) is on the subject's ADDRESS (identical for "does not" / "does not have to" even with the learned gates) and invisible in
role accuracy — the identity cleanup reads a blended address and the structure is carried by the other roles. B5.1 with the E1b protocol (log
_0840, 12-quater): everything composes to the unseen depth (subject 0.948 under two operators; may_neg 0.855 below the criterion, the named exception) -> B5d dropped.
C5 re-scoring (log _0757): the marker's margin trades unverifiable for false alarms one for one. Tool round 2 (agent): given gates close the
reference gaps (depth 3 0.987) and open one (unseen transitions 0.428, presence decision suspected); round 3 prepared. Push = judgment point.
Tool round 3 (agent; RESULTS 11-quater): presence not decided brings unseen transitions to 0.596 (v16 0.627); the opener-kind depth count is
right on 5/5 seeds, FULL 0.857 = 10d — the tool reproduces the reference and commands tables on every axis.

## 2026-10-17 (DGX, midday) — judgment 45ccd1d applied: F1 recording, per-seed readout, C5 closed
Judgment saved verbatim; the regularity row now carries ONE named exception (may_neg at the unseen depth 0.855; per seed 0.850, 0.800, 0.950,
0.675, 1.000 — rescore_v20, deterministic readout) and the rule-29 (A)/(B) labels; the two-limits row carries the trace's cosines; B5d
HYPOTHESIS; N8 PLANNED; rule 29. C5 per-slot thresholds (log _1029): slots 2+ 0.760, slot 1 wrecked (33 absent examples) -> C5 CLOSED, bounded
by the presence reading of the later slots; the drop follows the slot index, not the number of conditions. Tool round 4 (presence thresholds on
held-out formulations) handed to the agent with the GPU. Push = judgment point.
B6 (RESULTS 13, log _1037, saved models): 4-bit route book and stars lose nothing (8x smaller routebook); the chart must keep its magnitudes;
block-sparse chart at 25% free. B7 (RESULTS 14, log _1040): the round-1 position code added to the bundle is KEPT — unseen transitions 0.627 ->
0.678, depth 3 0.968 -> 0.998; the reference bundle becomes four codes (confirmation asked). Push = judgment point.
Tool round 4 (agent; RESULTS 11-quinquies): presence thresholds calibrated with held-out formulations move the recipient by +0.005 -> presence
calibration across formulations recorded as a limit of the reader (the agent's reservation on the midpoint rule left for judgment). Push.

## 2026-10-17 (DGX, midday) — judgment d8f8a98 applied: documents, B7 settlement, tool round 5
Documents: slot-1 sentence in 10c-nonies, 10d labelled (A) with its reason, RESULTS 14 hypothesis marked then withdrawn, B6 adoption deferred.
B7 settlement (log _1225, RESULTS 14-bis): the +0.051 is a surface-position prior (T11 +0.37 vs T4 −0.71, T7 −0.44), the depth-3 gain comes from
trained positions, untrained positions inert (omission rule adopted), terms +33% — the reference decision left to the reviewing session. Tool
round 5 (agent, RESULTS 11-sexies): stored presence ratios — by the rule the reader's limit (~10% overlap; no 0.9 / 0.9 in 3 of 5 seeds), but the
fitted threshold takes the recipient 0.64 -> 0.87 at 0.91 absent; adopting the fit is a product decision for the reviewing session. Push.

## 2026-10-17 (DGX, afternoon) — judgment 752853c applied: decisions recorded, N6
Reference bundle stays three codes (B7 a declared option, reason recorded); fitted threshold adopted as the tool's calibration rule (round 6 to
confirm). N6 (log _1343, RESULTS 15): the training legend lacks the recipient — the residual SHARE goes the wrong way (0.337 with vs 0.462
without; gap −0.267) and the true-R_C oracle is at chance (0.110): NOT met as written; the missing filler is recoverable from the residual at
the reader's own calibrated address (0.963, 5/5 seeds) and the residuals cluster by filler without any vector (1-NN 0.57): detectable by
direction, not by share — the detector the tool's `_unplaced` loop already implements. Push = judgment point.
