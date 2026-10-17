# Semantic space as a path: project history, status, outlook
(handover document for a new session; read it in full before running anything)

## 0. The idea we started from (the author's original formulation)
Interpreting text = a trajectory in a space (graph / hypergraph / lattice). The context is a node; each word "develops" the space according
to rules encoded in its structure; different sentences with the same meaning are different paths to the same point. Questions: can a
system be trained to learn word weights so as to obtain the equivalences? Is there a link to "a thousand brains" (Hawkins), to VSA/HDC, to fiber bundles?

## 1. The mathematical framework chosen
Vector Symbolic Architecture with Fractional Power Encoding: complex vectors of modulus 1 (phases), D = 512-1024.
  bind = element-wise multiplication (= addition of phases) = rigid DISPLACEMENT (labelled edge, isometry)
  bundle = addition of vectors = MELTING into a common body (center of mass, hyperedge); the only irreversible operation
  cleanup = argmax cosine over known candidates = "gravity" toward the nearest concept
Geometric reading: the space is a D-dimensional torus; binding with a role moves you into a "fiber" (a copy of the space); fiber bundles of fiber bundles =
chained binds; "algebraic fractal" = contractive bind (IFS) — discussed, NOT implemented (see 7).

## 2. The model that remained standing (after 3 rounds)
A word = a pair (what it is, what it does):
  - identity f_w: fixed, random vector, never learned (one per word)
  - address: computed from the CONTEXT through codes learned per CATEGORY (not per word): persons, objects, copulas = classes; verbs and
    function words = individual categories
Single step, identical for every word:  S <- S + address(context) (x) f_w
The state of a sentence = S (bundle). Two formulations are equivalent if they have deposited the same identities at the same addresses.
Address = BUNDLE OF SEVERAL CODES (round 3, v4.py):
  path   = sum of the displacements of the categories in the prefix (additive, commutative, robust)
  bigram = previous category + own category (local; surprisingly strong)
  chain  = the chain of expectations: a word takes the most recent unconsumed expectation for its category; emitters (gates per
           category) emit expectations relative to their own address (precise, large margin, fragile on new material)
  pos    = absolute position (x) category
World: explicit roles S_world = sum_role R_role (x) f_filler, with the SAME f as the model.
Single objective: maximize cos(S, S_world). Generative, no paraphrase labels, no negatives. Supervision stated exactly (rule 20): structured
per-sentence targets with pre-set roles — roles, identities and categories are GIVEN; moves, relative addresses and gates are LEARNED. Adam, lr 0.02, 10-15k sentences.
Evaluation: decode role r from S with the WORLD's address R_r: argmax_x cos(S (x) conj(R_r), f_x). Report accuracy + margin.

## 3. History, by round, with the mistakes
ROUND 1 (sim_deplasari.py, f.py, camp.py, k.py, control_transformer.py, mlp_control.py)
  - A: word->word prediction: collapse (everything similar). C: predicting the consequence through associative memory: predicts 100% but does NOT align
    the states. D: prototypes: memorizes. E: generative with both sides learned: collapse. => prediction produces equivalence only if it is
    GENERATIVE through a fixed operation and the world is fixed. (CONFIRMED also in rounds 2-3.)
  - F: factorizing the word (displacement per category, fixed identity) was necessary. (CONFIRMED.)
  - G/K: "deferred addressing" (the buffer before the verb) and "zoom-in" at 'ca' set by hand. REFUTED in round 2: they were not necessary;
    the apparent failure came from (a) evaluation with addresses from the active form (penalized legitimate overlapping addresses) and (b) BUG: world and model with
    different identities. All round 1 figures on non-active formulations are underestimates.
  - "Algebraic impossibility of per-word displacement": correct argument for EXACT equality of addresses, irrelevant with argmax decoding.
  - H: new identities zero-shot 0.94: CONFIRMED (1.00 in rounds 2-3), but it is a property of the construction (the identity does not enter the parameters)
    and presupposes that the category is known.
  - I: [RETRACTED as a contribution in round 8a — categories are recoverable from distribution (skip-gram gives them equally well); the model CONSUMES them as input, it does not produce them; what emerges from the objective are the gates (who opens a frame), 7c and 8b] categories discovered unsupervised by clustering the displacements: exact clusters. CONFIRMED on synthetic; on real text (SICK) it gives
    clean classes (man/person/woman; boy/kid/child). The word2vec control is missing.
  - Transformer control (854k params, untuned) and bag-of-words MLP: both 0.25 on new identities (chance) vs VSA 1.00. Valid only as
    "at this data regime"; the transformer was not tuned.
  - SICK (real text, relatedness 1-5): NEGATIVE (Spearman 0.45 vs 0.575 Jaccard). SICK does not measure roles; it does not refute the thesis, but shows that on text
    without a world the signal is noisy. WikiSQL (text->SQL, executable consequence): modest signal (selected col. 0.49 vs chance 0.15, without the table).
ROUND 2 (v2.py, v3.py, gates.py, nest2.py)
  - The simple path over categories, WITHOUT rules, solves perfectly all seen formulations and the nesting 'X spune ca <prop>' (the major correction).
  - v2 (expectations): margin x3 (0.50 vs 0.15), prediction 0.95 vs 0.53, but fragile on new formulations (0.18).
  - GATES ARE LEARNED: discrete search on the prediction criterion opens verbs/prepositions/copula/period and REJECTS persons and objects.
    Learned gates > hand-set gates (0.52 vs 0.46 on the new formulation).
ROUND 3 (v4.py, EVALUATION_PRINCIPLES.md)
  - The real barrier = new CONSTRUCTIONS (unseen category transitions): an untrained phase randomizes a multiplicative address.
  - Redundancy (bundle of codes) raises it from 0.60 to 0.82 (model = path+chain+bigram); 0.90 only with the position prior pos, which costs
    elsewhere (round 5). The rest up to 1.00 is on new categories (no information in the data). bigram alone = 1.00 on seen transitions.

## 4. What the experiments indicate (with confidence level)
  CERTAIN (3-5 seeds, correct evaluation; round 5 on a 3x world): factorization necessary; generative objective necessary; aligned roles
    (11 formulations x 2 frames; 16 x 6); new identities 1.00; redundancy of the STRUCTURAL codes (path+chain+bigram) helps on new constructions:
    0.58 -> 0.82, stable across worlds; nesting of DEPTH 1 = 1.00 without rules.
  CERTAIN (round 6c, 5 seeds, English world): with a COMPOSITIONAL world (reported speech = R_rep (x) inner state, one speaker role) and
    the reporting announced before its clause, nesting of depth 2 is solved exactly (1.00 on every role) by the plain chain code, and the
    learned displacements compose one level further than seen: depth 3, never trained, decodes at 1.00 on all roles (path+chainL+bigram).
    First clean positive on an unseen CONSTRUCTION.
  CERTAIN and NEGATIVE (rounds 5-6): (i) a FLAT world (depth-indexed speakers X1..X3, depth-invariant A,B,C) is unsolvable by any
    cumulative address scheme — an algebraic contradiction (RESULTS 6b), softened by argmax to ~0.77; the world was wrong, not the model.
    (ii) structure announced AFTER its scope (postposed '<P> , says X'; the English 'from A , C receives B') cannot be addressed causally:
    the clause is already written. This is the precise form of "deferred addressing is necessary": a displacement applied retroactively to
    a whole clause (the round-1 zoom-in), forced by data for right-announced structure only. The pos code is an absolute position prior:
    it helps on new categories, hurts on free order; its contribution depends on the set of formulations, not on the mechanism.
    "Seen transitions" as an axis measures the redundancy of the formulation set (RESULTS 6a), not the mechanism.
  CERTAIN (round 7, 5 seeds, reference world): the gates EMERGE by gradient from the single generative objective — identities (P, O)
    closed, relation markers (to, from, by, that, comma, copula) open, VERBS closed — the same set in every seed; no loss at depths 1-3
    (1.00), better than the hand set on new formulations. The search of round 2 had opened the verbs; with left re-addressing they are
    redundant as emitters: the emergent partition is identities / relation markers / action words. Control (round 7b): a small tuned
    transformer on the same task, identities and objective gets 0.22 on new identities (chance 0.17) and 0.39 at the unseen depth
    (inner roles at chance) — "at this data regime", as in round 1.
  RETRACTED as a contribution (round 8a, 5 seeds): "emergent categories" — categories are recoverable from distribution (skip-gram gives them equally well); the model CONSUMES them as input, it does not produce them; what emerges from the objective are the gates (who opens a frame), 7c and 8b. (Both the learned
    displacements and skip-gram on the same sentences cluster the words at 1.000; kept only as a consistency check.) Control B (round 8a): a linear associative map with any rank (8-1024) gets chance on new identities and the
    transformer's signature at the unseen depth; "generative with a fixed operation is necessary" stands against it.
  CERTAIN (round 9, 5 seeds): the state is a frame — frame(S) with the depth estimated from S, Prolog and SMT (z3) round trip at 1.000 on
    depths 0-3 (3 unseen). NEGATION AS A ROLE (E1): decoded, scope-correct and separated from the affirmed state at 1.000 / 0.18 vs the
    lexical bag's 0.72, every form, no rule; negation words become emitters like 'that'. NEGATIVE: sentential negation does not compose
    into reported speech when seen only at depth 0 (scope 0.0-0.5); argument-scoped negation does (0.8-1.0). E1b running.
  CERTAIN (round 8, WikiSQL, 3 seeds, dev = test): selected column 0.82 vs 0.54 (same-input transformer, 10x budget 0.65) vs 0.50 (BoW+LR)
    without the table, at every size from 1k; full query 0.37 vs 0.19; aggregator lost (lexical, 0.80 vs 0.89); equivalence separation
    lexical overall, role-based on hard same-words/different-SQL pairs (-0.15, n = 1,353).
  WEAK / TO BE VALIDATED: any claim about general text; operators beyond negation (E2-E5).
  NEGATIVE: SICK; v2 alone on new constructions; associative prediction for equivalence.

## 5. What must be VALIDATED at scale (in order of importance)
  1. [DONE, round 5, v5_scale.py] Reproduction with 5 seeds, 3x world: path+chain+bigram reproduced (+-0.03); all 4 NOT (pos);
     nesting >= 2 NEGATIVE (identical prefix). See RESULTS.md ROUND 5.
  1b. [DONE, round 6] Left expectations (v6_left.py, chainL): fix inner roles at depth 2 (A, C -> 1.00), speakers only to 0.77 in the FLAT
     world (algebraic limit, RESULTS 6b). Compositional world (v7_nested.py): depth 2 = 1.00, unseen depth 3 = 1.00 with prefixed reporting.
  1c. NEW: retroactive displacement of a whole clause for RIGHT-announced structure (postposed reporting; comma constructions). Candidate:
     a clause-level union-then-restriction — when 'says' arrives after a complete clause, S <- S_outside + R (x) S_clause; the clause
     boundary itself from expectations (',' as emitter). Criterion: depth 2 > 0.95 on all roles with all 3 outer forms, nothing else lower.
     Also: redesign the "seen transitions" axis so that it tests mechanism, not redundancy (hold out formulations whose every LOCAL role
     cue has been seen with the SAME role).
  2. [DONE, round 7, v8_gates.py] Gates by gradient: identities closed and relation markers open on 5/5 worlds (criterion met);
     verbs CLOSED (criterion as written not met — the hand/search set was a prior, the gradient's set is better on every axis).
     "Everything emerges from a single objective" is supported for the gating structure; the partition is identities / relation markers /
     action words. Transformer control (v9_transformer.py) done: fails on new identities and unseen depth at this regime.
  3. [DONE, round 8a] The word2vec control for categories: equally clean clusters (1.000 vs 1.000) => categories are recoverable from distribution (skip-gram gives them equally well); the model CONSUMES them as input, it does not produce them; what emerges from the objective are the gates (who opens a frame), 7c and 8b.
  4. [DONE, round 8a] Control "associative memory with a bottleneck": does NOT align (new ids chance, unseen depth 0.25-0.39 at any rank 8-1024) => "generative necessary" stands.
  5. TUNED transformer (+ dropout, weight decay, lr sweep) and small BERT, at 1k/5k/50k examples: a data-efficiency curve, not a point.
  6. Capacity: 64-1000 identities at D = 4096-16384; sparse block codes; cleanup at large vocabulary.
  7. [STARTED, round 8b, v11_wikisql.py] WikiSQL with the full model + learned gates + distributional categories: selected column 0.824
     without the table vs 0.494 BoW+LR (criterion met); aggregator 0.813 vs 0.867 BoW+LR (NOT met); separation 0.28 (NOT met); table at
     the input HURTS (0.67). Next: 3 seeds, test split, tuned neural baselines at 1k/5k/56k, aggregator readout. Then Spider.
  8. Consequence signal without a synthetic world: next sentence / multiple captions per image (COCO) / translations (OPUS) with hard negatives.
  9. [NEW PROGRAMME, judgment 2026-10-15 after 8d: verifier / formalizer] R1 frame(S)+diff everywhere [DONE, 9a]; R2 Prolog/SMT round trip
     [DONE, 9a]; R3-R4 operators as roles with scope reported separately (rule 17): E1 negation [DONE, 9b: scope 1.00 at depth 0; sentential
     negation inside reporting fails when unseen there; E1b: seen at depth 1 (share 0.2), read at depth 2 = 0.972], E2 quantifiers [DONE,
     9c/9c-bis/9c-ter: the ONE operator for which the regularity fails in all three worlds tried (role / slot / noun targets); hypothesis for
     hypotheses H-a addressing / H-b relation; C2 (gates open) gave a signal at the unseen depth, no composition (9c-quater); C2b (determiner as a beacon-relative emitter) COMPOSES the one-quantifier scope at the unseen depth, 0.998, 5 seeds (9c-quinquies); the two-quantifier marked case was the exception until C2c (9c-sexies): with CATEGORY-LOCAL terms (marked determiner -> R_wide term, unmarked -> R_q term, both unconditional) and wide/narrow derived by the legend at readout, the two-quantifier scope composes at the unseen depth too (0.966, every seed > 0.9) — no exception remains; H-b retired, the permuted-filler term withdrawn], E3
     modality/conditionals [DONE, 9d/9d-bis: 1.000 at the unseen depth; 'Q if P' at the declared limit], E6 verifier on real text [DONE,
     controlled version 10c: role errors 0.74-0.81 detected, confident-wrong < 1.3%, false alarm 0.16; LLM formalizations PLANNED]
     (injected errors, precision/recall), E4 coordination, E5 reference, E7 LLM annotation pipeline; Spider after E6; capacity after all.

## 6. Concrete steps on the DGX (torch)
  Day 1: port (phases as nn.Parameter; torch.polar; cumsum over categories; autograd replaces grad_cos; batch by length);
          check on v4.py that it reproduces the ROUND 3 table with seed 0; then 5 seeds, large world.
  Day 2: gates by gradient; word2vec control; associative memory with a bottleneck.
  Days 3-5: tuned transformer + data curves; capacity.
  Week 2: WikiSQL/Spider with v4; BoW+LR baseline.
  If 2 and 3 come out well: write the principle note (POSITIONING.md, article 1). If 7 comes out well: the empirical article.

## 7. Outlook and unimplemented ideas
  - Contractive bind (IFS, algebraic fractal): addresses with scale, readable depth, non-accumulating noise; cost: fragile inversion.
  - Non-commutative address space (permutations) for native order; commutative only inside a frame.
  - The round-1 position code (one displacement per absolute position) was re-measured in the bundle (RESULTS 14, 14-bis): kept as a per-world
    OPTION, not in the reference — its gain is a prior over absolute positions that wins and loses by template; the reference bundle stays
    path + chainL + bigram (judgment 752853c §1).
  - Expectations with a specific identity (not just category) = top-down lexical prediction; would reduce the overlap of the first word.
  - Multiresolution through clocks (fast level on words, slow on sentences) instead of anchors.
  - Link to TBT/grid cells: motivation and interpretation, NOT a result; stays in the discussion, not in the title.

## 8. Evaluation principles (summary; in full in EVALUATION_PRINCIPLES.md)
  addresses from the world; shared F (assert); chance printed; 3 separate seen/unseen axes (identities / formulations with seen transitions /
  unseen transitions); margin reported; >=3 seeds; trivial control in the same table; no calibration from the test set; trace one example
  before training; a failure is checked first as evaluation, then data, then optimization, only then architecture.

## 9. File map
  GIT_WORKFLOW.md (how we commit and communicate through GitHub) | run.sh (runs + saves a log with date and hash) | JOURNAL.md | logs/
  00_READ_FIRST.md (this one) | README_RUN.md (how to run, which file does what) | README_DGX.md (plan, data sources, porting)
  RESULTS.md (all tables, 3 rounds) | INTERPRETATION.md (how to read each metric) | POSITIONING.md (articles, controls, citations)
  EVALUATION_PRINCIPLES.md | code: camp.py (the world + v1), v2.py (expectations, RoleWorld, evaluate), v4.py (redundant codes; the CURRENT MODEL),
  gates.py (gate search), nest2.py (nesting), control_transformer.py, mlp_control.py, sick_test.py, wikisql_test.py | log_*.txt
  Quick check command (2 min on CPU): python3 v2.py ; main command (25 min CPU, seconds on GPU): python3 v4.py
