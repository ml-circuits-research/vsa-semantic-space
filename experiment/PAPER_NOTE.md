# Principle note (draft, started 2026-10-15) — working title: "Role emergence in a displacement-addressed vector-symbolic sequence model: composition to an unseen depth, emergent gates, and the limit of causal addressing"

Status: DRAFT skeleton with the evidence placed; prose to be tightened. Every number points to RESULTS.md and a log with a commit hash.
Controls pending at the time of writing: word2vec for categories, bottleneck associative memory (v10_controls.py) — section 6 to be filled.

## 0. One-sentence claim
A word is a pair (fixed identity, learned displacement by category); the state of a sentence is the bundle of identities bound to the
addresses reached by integrating displacements along the sentence; training the state to predict, through a fixed operation, the state of
a world with explicit roles makes different formulations converge to the same state role by role, makes the learned displacements compose
one nesting level further than ever trained, and makes the gating structure (which words open a frame) emerge from the same objective.

## 1. Setting (what is fixed, what is learned, what is measured)
- Space: complex unit vectors (phases), D = 1024. bind = elementwise product (phase addition, a rigid displacement); bundle = sum;
  cleanup = argmax cosine over known candidates. (Plate; Kanerva; Gayler; Frady-Kleyko-Sommer fractional power encoding.)
- Word = (identity f_w: fixed random, never learned; address: computed from context by codes learned per CATEGORY, not per word).
  State S = sum_k address_k (x) f_{w_k}. Address = bundle of codes: path (sum of category displacements along the prefix), bigram
  (previous + own category), chain (expectations emitted by gated words; a word takes the most recent unconsumed expectation for its
  category; with LEFT re-addressing: a non-emitter is re-addressed by the first emitter that follows it). [RESULTS rounds 3, 6b, 7c]
- World (reference, compositional): S_world = R_A f_A + R_B f_B + R_C f_C; reported speech S_world(X says that P) = R_X f_X + R_rep (x) S_world(P);
  one speaker role; depth = number of R_rep. Identities SHARED between world and model (asserted). [RESULTS 6c]
- Objective: maximize cos(S, S_world). Generative, no paraphrase labels, no negatives. Adam, lr 0.02, 30k sentences, batch 8.
- Evaluation (EVALUATION_PRINCIPLES 1-15): decode role r by unbinding the WORLD's address: argmax_x cos(S (x) conj(R_rep)^k conj(R_r), f_x);
  accuracy AND margin; chance printed; separate axes: seen / new identities / new formulations (role cues seen) / new formulations (unseen
  transitions) / depth 1 / depth 2 (seen) / depth 3 (UNSEEN construction); >= 5 seeds (the seed changes the world too); same optimization
  regime across compared variants; a new depth must not introduce a new role; per-role breakdown before interpreting any bundle.
- Synthetic English world: 6 frames x 16 formulations (active, passive, dative shift, clefts, topicalization, wh-constructions), 18 + 18
  identities (12 training, 6 new), reporting 'X says/claims that <P>' nested 0-3 levels (training 0-2).

## 2. Pillar 1 — composition to an unseen depth [RESULTS 6c; logs/2026-10-15_1319_v7_nested_e1e7011.txt]
Trained on depths 0-2, tested on depth 3 (never seen; six roles to decode: A, B, C, X1, X2, X3), 5 seeds:
| code | depth 1 | depth 2 (seen) | depth 3 (UNSEEN) | per role at depth 3 |
|---|---|---|---|---|
| path+chain+bigram | 0.982 | 0.997 | 0.979 | A 0.999 B 1.000 C 0.976 X1 1.000 X2 1.000 X3 0.898 |
| path+chainL+bigram | 1.000 | 1.000 | **1.000** (margin 0.23) | all six 1.000 |
Why it works, in one line: the world's algebra (R_rep applied d times) and the model's (displacements summed along the chain) coincide, so
once training lands on the consistent solution (a + g = rho, d + b = rho, ... RESULTS 6b), composition one level further holds by
construction. New identities decode at 1.000 at every depth (identities never enter the parameters).

## 3. Pillar 2 — the gates emerge from the same objective [RESULTS 7c; logs/2026-10-15_1412_v8_gates_06e12bd.txt]
Gates g_c = sigmoid(s_c): probability that category c emits expectations; soft chain differentiable in the gates; 32 categories; 5 seeds.
Learned set, IDENTICAL in every seed and from both starts (all closed / all open): identities P, O CLOSED; relation markers to, from, by,
that, comma, copula OPEN; all 18 verb forms CLOSED. No loss at depths 1-3 (1.000 / 1.000 / 0.996-1.000); better than the hand set on new
formulations (bundle: seen trans. 0.78 -> 0.85, unseen 0.76 -> 0.87; chain alone: role cues 0.22 -> 1.00).
Where gradient and search disagree, and why: the round-2 search (no left expectations) opened the verbs because only emitters could place
the words after them; with left re-addressing, a verb has nothing to emit that 'to / by / that' and the left slot do not already give.
The hand set ("verbs + prepositions open") was a linguistic prior; the emergent partition is identities / relation markers / action words.
Hand set kept as an ablation row.
### 3.1 Trace (the same sentences, word by word; logs/trace_gates_hand_vs_learned.txt)
'p0 gives o0 to p1 .'
| word | hand gates (all but P, O emit) | learned gates (that, to, by, '.') |
|---|---|---|
| p0 | L[gives:P] | L[to:P] |
| gives | root | L[to:gives] |
| o0 | gives:to + L[to:O] | L[to:O] |
| to | gives:to | root |
| p1 | gives:to + to:. + L[.:P] | to:. + L[.:P] |
'p0 says that p1 claims that p2 gives o0 to p3 .' (depth 2)
| word | hand gates | learned gates |
|---|---|---|
| p0 | L[says:P] | L[that:P] |
| p1 | says:that + that:claims + L[claims:P] | that:that + L[that:P] |
| p2 | says:that + that:claims + claims:that + that:gives + L[gives:P] | that:that + that:to + L[to:P] |
| o0 | ... + that:gives + gives:to + L[to:O] | that:that + that:to + L[to:O] |
| p3 | ... + gives:to + to:. + L[.:P] | that:that + that:to + to:. + L[.:P] |
Reading: with the learned gates the sentence becomes a skeleton of relation markers; every content word, the verb included, hangs to the
LEFT of the next marker ("the person before 'to'" = agent; "the thing before 'to'" = theme; "the person before '.' after 'to'" = recipient);
addresses have <= 3 terms instead of 5-7, and the nesting depth is carried by 'that:that'.

## 4. The control that fails on exactly those two axes [RESULTS 7b; logs/2026-10-15_1414_v9_transformer_75e7e0d.txt]
Small transformer (0.79M parameters; 2 pre-norm layers, 4 heads, d_model 128), same fixed identities at the input, same objective, same
decoder, same 30k sentences (12 epochs), tuned by validation on the training distribution (lr 1e-3, dropout 0), 5 seeds:
new identities 0.215 (chance 0.167) vs 1.000; depth 3 (unseen) 0.390 vs 1.000 — inner roles at chance, the two known speakers perfect:
it reproduces the deepest configuration it has seen rather than composing a third; depth 2 (seen) A 0.49 / C 0.50. On the flat axes the two
are comparable (seen trans. 0.77 vs 0.79; role cues 0.97 vs 1.00). Wording: "at this data regime, a small transformer trained identically
does not generalize to new identities or to a deeper construction" — not "transformers cannot". Not covered: more data/epochs, larger
models, learned embeddings (which would remove the fixed-identity constraint and with it the zero-shot test).

## 5. The declared limit, with the demonstration [RESULTS 6b, 6c]
(a) A FLAT world (depth-indexed speaker roles X1..X3, depth-invariant A, B, C) is contradictory for any cumulative address scheme: with
a = says:that, b = claims:that, c = that:says, d = that:claims, g = that:gives, depth-invariance of A gives a+g = 0, b+g = 0, d+b = 0,
c+a = 0, hence X2 - X1 = a + d = 0. Measured: X2 ~0.5-0.77 (argmax softens the contradiction). This was OUR world design, not the model;
the compositional world removes it. (b) Structure announced AFTER its scope — postposed reporting '<P> , says X'; the comma construction
'from A , C receives B' — cannot be addressed causally: the clause is already written when the marker arrives. With the postposed form
present, depth 2 falls to 0.88 and depth 3 to 0.54. This is the precise, data-forced form of "deferred addressing is necessary": a
displacement applied retroactively to a whole clause, and only for right-announced structure. Declared as a limit here; the remedy
(clause-level re-addressing) is future work.

## 6. Controls on "emergent" [RESULTS 8a; logs/2026-10-15_1530_v10_controls_727aecb.txt]
6.1 word2vec for categories: the untied learned displacements cluster the 54 content words into the 5 true classes perfectly (purity, ARI,
    NN-category all 1.000, 5 seeds) — and so does skip-gram on the same sentences (1.000). Categories are trivially distributional in this
    world; "emergent categories" is RETRACTED as a contribution and kept only as a consistency check. For real text the categories can be
    supplied by distributional clustering with no loss of honesty (that is what the WikiSQL experiment does).
6.2 bottleneck associative memory (S = U V x on sum_k pos_k (x) f_{w_k}, complex linear, rank 8-1024): at every rank new identities ~0.24
    (chance 0.17) and depth 3 0.25-0.39 with the transformer's signature (known speakers perfect, inner roles at chance); rank 128 = rank
    1024 = 0.74 on seen data. Capacity is not what is missing; the fixed generative operation is. "Generative necessary" stands.

## 7. The thread of the two retractions (what the negatives taught)
Round 1 argued an "algebraic impossibility" of per-word displacement and hand-coded deferred addressing (verb writes the buffer, zoom-in at
'that'); round 2 retracted both — they were artefacts of evaluating with the model's addresses and of a world/model identity mismatch; argmax
decoding tolerates overlapping addresses. Round 5 then found nesting >= 2 "unsolvable from the prefix"; round 6 sharpened it: the flat
world was contradictory (our construction), and the real limit is right-announced structure (section 5). Each retraction came with a
rule (EVALUATION_PRINCIPLES 1-15) that the next round had to pass.

## 8. Caveats (to state, not to hide)
Own synthetic world (two reporting verbs, prefixed reporting only in the positive result, 6 frames, 18 identities); the "seen transitions"
axis measures formulation-set redundancy (replaced by "role cues seen", kept for comparison); the control is small and minimally tuned;
D = 1024 and 36 identities (capacity not yet measured at scale); real text: WikiSQL selected column 0.82 vs 0.49 BoW+LR without seeing the
table (round 8b, one seed), aggregator below the baseline — reported in the empirical paper, not here; nothing here is a claim about general text.

## 9. Related work to cite (verify each before citing)
Plate 1995/2003 (HRR); Kanerva 2009 (HDC); Gayler 2003 (VSA); Smolensky 1990 (TPR); Eliasmith 2012 (SPA); Frady, Kleyko, Sommer 2018-2022
(FPE, resonator networks, grid cells as FPE); Kleyko et al. 2022 (survey); Hawkins et al. 2017/2019, Lewis et al. 2019 (Thousand Brains,
displacement cells); Elman 1990; Kamp 1981, Heim 1982 (dynamic semantics); Fillmore 1982 (frames); Katharopoulos et al. 2020 (linear
attention = bundle of binds); Schlag & Schmidhuber 2018 (fast weights, TPR); Lake & Baroni 2018 (SCAN).
