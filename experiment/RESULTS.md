# Experiment campaign: words as displacements, state as a bundle of (address ⊗ identity)

Starting point: "different sentences, same meaning = different paths, same point" in an HDC/FPE space
(complex vectors of modulus 1, bind = multiplication, bundle = addition).

## The final model (camp.py / k.py)
- Each word: **identity** f_w (fixed random vector) and **displacement** m_cat(w) (learned, one per category).
- Uniform step: L <- L * m ; S <- S + L * f.  The state of a sentence = S.
- **Deferred addressing**: the words before the first verb are written only at the verb, at address r_verb ⊗ relative_path.
- **Zoom in**: the subordinator "ca" sets an anchor; the inner path starts from zero relative to it.
- Learning: only generative prediction error, cos(S ⊗ T, S_world(consequence)) with T fixed random;
  the world state is "sensory" (fixed code). No paraphrase labels, no negatives. Adam, lr 0.02.
- Evaluation: the roles (agent, object, recipient) are read from the state of any formulation using
  *the addresses of the active form*; correct = the same identity recovered by cleanup.

## Results (mean decoding over all formulations and roles; chance = 1/n identities)

| exp | what was tested | result |
|---|---|---|
| A | local word->word prediction | collapse: everything similar; roles = syntactic positions |
| B | contrastive on paraphrases (labels) | 99.8% paraphrase > role swap; 70% on new pairs |
| C | predicting the consequence through associative memory | prediction 100%, but NO alignment of the states |
| D | prototypes per consequence | separates the object, not the roles; does not generalize |
| E | generative S⊗T, both sides learned | collapse |
| E2 | generative, fixed world, per-word displacement | failure; shown algebraically impossible |
| F | + displacement per category (what it does / what it is) | aligned roles on the passive (recip 1.00, object 1.00, agent 0.64); the initial subject stays positional |
| G2 | + deferred addressing at the verb (5 formulations x 2 frames, 8 id) | **0.86** (G1 without deferral 0.70; G3 without factorization 0.50) |
| H | identities UNSEEN in training | **0.94** (chance 0.25): zero-shot generalization to new words |
| I | categories DISCOVERED unsupervised (iterative clustering of displacements) | exact clusters: persons / objects / finite verbs / participles / copulas; decoding **0.96** |
| J1 | capacity: 8/24/64 identities x D=256/512/1024 | 8: .85/.95/.97; 24: .61/.74/.83; 64: .45/.56/.66 (chance .016) |
| J2 | noise: 0..5 adverbs inserted at random | 0.77 -> 0.70: robust |
| K | hierarchical composition "X spune ca <propozitie>" (3 outer forms x 5 inner x 2 frames, D=1024) | **0.88**; speaker 0.98, object 1.00, recip 0.86, agent 0.68 |

## What was learned
1. Prediction (fast or slow) through associative memory does NOT produce equivalence: it can link many states to the same answer.
   Equivalence appears only when the slow level must GENERATE the next state through a fixed operation.
2. The word must be factorized: the displacement belongs to the category (what it does), the identity is in the feature (what it is).
   With a single displacement per word, the identity of the previous words "leaks" into the addresses of the following ones (algebraic impossibility).
3. Categories need not be given: the generative objective produces them on its own (persons 0.85 similarity among themselves vs -0.03 with objects),
   and iterative clustering recovers them exactly.
4. The subject before the verb requires deferred addressing (the union of locations from the Numenta model / role assignment at the verb).
5. Hierarchy works with the same mechanism: an anchor + relative path (zoom in), without an explicit stack.

## What remains unsolved / to do at scale
- Capacity is the standard HDC one: it drops with the number of identities at fixed D; known remedy: large D (10k+) and sparse block codes.
- The agent in the "de la A" / "de A" formulations remains the weakest (0.6-0.7); probably address collisions.
- "ca" as frame opener and the verb as buffer trigger are set by hand; they must be learned (e.g. from statistics: the words after which prediction "resets").
- The consequence signal is given by the synthetic world; on real text the candidate is the next sentence/event or a paired modality.
- A control is missing: a small transformer on the same task, with the same number of examples.

## CONTROL (control_transformer.py): VSA vs small transformer, same task, same data
Task: from the sentence predict the sensory state of the consequence; role decoding from the PREDICTION with the world's addresses.
Both receive the same fixed feature per word (the shared "what it is" path). Training on formulations 0-3 and 8+8 identities.
Transformer: d=128, 3 layers, 4 heads, 854k parameters, 120k sentences (3x more). VSA: 10k parameters, 40k sentences.

| test | VSA | transformer |
|---|---|---|
| seen identities, seen formulations | 1.000 | 1.000 |
| UNSEEN identities (4 new; chance .25) | **1.000** | 0.258 |
| UNSEEN formulation, seen identities (chance .125) | 0.637 | 0.417 |
| UNSEEN formulation + UNSEEN identities (chance .25) | **0.742** | 0.240 |

Conclusion: at this data regime, the transformer memorizes (perfect on what it has seen, at chance level on new identities),
the VSA composes. Note: the important correction discovered here: the model and the world must share the SAME identity codes
(in the previous experiments they were different, which limited the quality of the prediction, but not the internal evaluations).
Limit of the control: a larger transformer with much more data would generalize better; what matters here is data efficiency.

## FIRST TEST ON REAL TEXT (sick_test.py): SICK, 4927 test pairs, human relatedness score 1-5
Variant G1 of the model (without deferred addressing, without world; equivalence signal = training pairs with score >= 4).
| method | Pearson | Spearman |
|---|---|---|
| Jaccard (word overlap) | 0.582 | 0.575 |
| bag of words (bundle of random features, no displacements) | 0.560 | 0.536 |
| random per-word displacements | 0.223 | 0.197 |
| per-word displacements, trained | 0.425 | 0.433 |
| displacements on discovered categories (clustering), trained | 0.448 | 0.453 |

NEGATIVE RESULT: on lexical relatedness, order/geometry hurts compared to the bag of words. Contradictions (negations) are NOT separated.
What worked: clustering the displacements on real text gives clean semantic classes (man/person/woman/lady; boy/kid/child; a/the/one).
Why it is not decisive: (1) SICK rewards word overlap, not roles (only 29 test pairs have the same words in a different order);
(2) the training signal is "relatedness", not role equivalence; (3) deferred addressing and the world are missing; (4) ~11 min/training on this CPU, hence undertrained.
The right test for the architecture: a set with pairs that differ ONLY in roles/negation, with an equivalence signal given by a world (images: captions of the same image).

## SECOND TEST ON REAL TEXT, with EXECUTABLE consequence (wikisql_test.py): WikiSQL, 56k questions -> SQL
World = the SQL query encoded role-filler (fixed code). Model = the uniform step over the question, generative objective. 2 epochs, CPU, ~70 s.
Decoding from the PREDICTION on dev (3000): selected column (among the table's columns; chance 0.15), aggregator (6; majority 0.72), column of condition 1.
| model | selected col. | aggregator | condition col. | pairs same skeleton / different |
|---|---|---|---|---|
| bag of words (no displacements) | 0.19 | 0.17 | 0.15 | 0.77 / 0.43 |
| a single displacement for all words (position code) | **0.49** | 0.59 | 0.25 | 0.65 / 0.23 |
| per-word displacement | 0.37 | **0.76** | 0.14 | 0.59 / 0.28 |
Reading: real signal, but modest. The model does NOT see the table and has no deferred addressing / categories; rare words remain with untrained
displacements (that is why "per word" loses on the selected column and gains on the aggregator, which depends on a few frequent words).
The obvious next step: displacements on discovered categories + deferred addressing + large D; expect a jump, not a miracle.

## MLP CONTROL (mlp_control.py): orderless bundle -> MLP (2.1M parameters) -> predicted state; same data and evaluation as the transformer
| test | VSA | transformer | MLP on bundle |
|---|---|---|---|
| seen identities, seen formulations | 1.00 | 1.00 | 0.66 |
| UNSEEN identities | 1.00 | 0.26 | 0.25 |
| UNSEEN formulation | 0.64 | 0.42 | 0.67 |
| both unseen | 0.74 | 0.24 | 0.26 |
The MLP plateaus at 0.66 even on seen data: without order it cannot distinguish the agent from the recipient (both persons in the same bag),
but for exactly that reason it is indifferent to the formulation (0.67 on the unseen formulation). It does not generalize to new identities either.
Conclusion: order is necessary (MLP), but not sufficient (transformer); the VSA is the only one that composes on both axes at this data regime.

# ROUND 2 (v2.py, v2b.py, v3.py, gates.py, gates_final.py, nest2.py): redo on the new understanding
Protocol changes: world = explicit roles (R_role (x) identity), identities SHARED between world and model, decoding with the WORLD'S ADDRESSES
(no calibration on the active form), no T, 3 seeds, verbs = individual categories, P/O/COP = classes.
New model v2: addressing through expectations (a word takes the most recent unconsumed expectation for its category; otherwise it stays in the
buffer and moves when one appears); emitters = gates per category; relative addresses on (emitter category, expected category); fixed origin.
v3 (dual) = v2 address (+) path over categories (coarse). No rule for the verb or for 'ca'.

## Main table (3 seeds; training T0-T3 with 8+8 identities) [seen | new identities | new formulation T4 | both]
| model | seen | new id. | T4 | both |
|---|---|---|---|---|
| v1 (path + verb-triggered buffer, by hand) | 1.000 | 1.000 | 0.611±0.059 | 0.716±0.066 |
| path over categories ALONE (no buffer, no verb) | 1.000 | 1.000 | 0.546±0.083 | 0.620±0.069 |
| v2 expectations, hand-set gates | 1.000 | 1.000 | 0.181±0.056 | 0.401±0.135 |
| v2 additive expectations (out+in) | 1.000 | 1.000 | 0.127±0.082 | 0.242±0.096 |
| v2 all gates open (bigram chain) | 0.787 | 0.834 | 0.121 | 0.269 |
| v2 all closed (everything at the origin) | 0.129 | 0.223 | 0.136 | 0.239 |
| v3 dual (expectations + path) | 1.000 | 1.000 | 0.503±0.016 | 0.647±0.059 |
| v3 dual, LEARNED gates | 1.000 | — | 0.524±0.035 | — |

## MAJOR CORRECTION with respect to round 1
The simple path over categories, WITHOUT deferred addressing, solves perfectly all seen formulations (including 'C primeste ...' and 'lui C A ii da B').
The "algebraic impossibility" of round 1 required exact equality of addresses; with argmax decoding, an OVERLAPPING address (half agent,
half recipient, for the first word) is sufficient, because the other role has a term with a complete match. The failure of round 1
(G1: T1 agent 0.00) came from (a) evaluation with the addresses of the active form, which penalizes exactly the overlap, and (b) different features
world/model. So the verb-triggered buffer was NOT necessary for the seen formulations. The cost of the overlap shows up in the MARGIN:
  decoding margin (correct - best wrong): path alone 0.154 | expectations 0.496 | dual 0.480 ; prediction cos: 0.53 | 0.95 | 0.94
Expectations give clean addresses (margin x3, prediction 0.95), the path gives robustness to new constructions. The dual has both.

## Triggers CAN be learned (gates.py): discrete search of the gates on the generative-prediction criterion, without labels
Starting with all closed: accepted open: '.', COP, are, catre, cumpara, da, de, ii, vinde, la. REJECTED every time: P and O
(identities do not open frames), plus dat/lui/nu/primeste/vandut (redundant). The learned gates give T0-T3 1.000 and T4 0.524 vs 0.464 with
the hand-set gates (fewer gates = fewer untrained transitions). The criterion was prediction, not decoding: the structure "who opens
a frame" came out of the same objective as everything else.

## Hierarchy without rules (nest2.py): 'X spune ca <prop>', 3 outer forms x 5 inner x 2 frames, D=1024, no rule for 'ca'
| model | speaker | agent | object | recip | unseen inner T4 |
|---|---|---|---|---|---|
| path alone (commutative) | 1.00 | 1.00 | 1.00 | 0.99 | 0.80 (2 seeds) |
| dual (expectations + path) | 1.00 | 1.00 | 1.00 | 1.00 | 0.78 (1 seed) |
The anchor and the zoom-in set by hand in round 1 (k.py) were not necessary: with explicit roles in the world, both variants solve the nesting.

## What remains true from the "main oversight" and what does not
- True: inferring the motion from context was missing; v2 does it through expectations and gives clean addresses; the gates (who emits expectations) are learned.
- False / corrected: no hand-set rule (verb, 'ca') was needed, not even in v1; the evaluation was to blame.
- New and important: multiplicative addressing (sum of phases) is FRAGILE to a single untrained transition (a random phase ruins the whole
  address). The chain of expectations suffers the most (T4: 0.18); the additive path over categories is more robust (0.55); the dual inherits
  the robustness. Generalization to new CONSTRUCTIONS is the real barrier of the whole family, not the roles and not the identities.

# ROUND 3 (v4.py): REDUNDANT addresses for new constructions — evaluation according to EVALUATION_PRINCIPLES.md
11 formulations x 2 frames; training on 5 formulations (T0,1,2,3,6); separate test: new formulations with ALL transitions seen (T9, T10) vs
new formulations with unseen transitions/categories (T4, T5, T7, T8). Addresses from the world, shared F (assert), 3 seeds, margin, chance 0.125 / 0.25.
| address code | seen | new identities | new form., seen transitions | new form., UNseen transitions |
|---|---|---|---|---|
| path (sum of the categories in the prefix) | 1.00 (m .25) | 1.00 (m .26) | 0.83 (m .11) | 0.60 (m .07) |
| chain (chain of expectations) | 1.00 (m .51) | 1.00 (m .52) | 0.31 (m .04) | 0.51 (m .18) |
| bigram (previous category + own) | 1.00 (m .46) | 1.00 (m .47) | **1.00** (m .31) | 0.71 (m .16) |
| pos (absolute position) | 0.86 | 0.87 | 0.70 | 0.77 |
| path+chain | 1.00 (m .48) | 1.00 | 0.71 | 0.75 |
| path+bigram | 1.00 (m .45) | 1.00 | 0.98 (m .29) | 0.82 |
| path+chain+bigram | 1.00 (m .53) | 1.00 | 0.99 (m .19) | 0.82 |
| all 4 | 1.00 (m .52) | 1.00 | 0.94 (m .19) | **0.90** (m .17) |
Conclusions:
- The redundancy hypothesis is CONFIRMED: on unseen transitions, from 0.60 (path) / 0.51 (chain) to 0.82 (path+chain+bigram = THE MODEL; round 5);
  0.90 only with the position prior (pos), which costs elsewhere (round 5: T10). The rest up to 1.00 is on new categories, where there is no
  information in the data. On seen transitions, from 0.31 (chain) to 0.99.
- Surprise: the BIGRAM code alone (address = previous category + own) is the best on new formulations with seen transitions (1.00, margin .31):
  roles are largely LOCAL (the preceding preposition decides), and the first word stays overlapping and is resolved by argmax.
- Note: on new formulations with SEEN transitions, "all 4" (0.94) is BELOW bigram alone (1.00): redundancy helps on new transitions
  (0.71 -> 0.90) and costs a little on the seen ones (the more fragile codes, chain/pos, add noise where bigram was already sufficient).
- The chain of expectations (v2) has the maximum margin on seen but is the most fragile on new; not worth it alone, worth it in a bundle.
- All codes give 1.00 on new identities: compositionality over identities is a property of the family, not of one code.
- The barrier remains on truly new transitions/categories (0.90, not 1.00): there is no information in the data there; the recovered part comes from
  the codes that do not depend on the new transition (pos, path).

# ROUND 4 (DGX, 2026-10-15): reproduction on the DGX and port to torch/GPU (v4_torch.py) — no model changes
Same world, same seeds, same evaluation as round 3. Logs with commit hash in logs/.

## Reproduction (3 seeds; columns: seen | new id. | new form. seen trans. | new form. UNseen trans.)
| run | all 4 | observation |
|---|---|---|
| v4.py numpy, old CPU (round 3, log 2026-10-15_v4.txt) | 1.000 | 1.000 | 0.942±0.014 | 0.901±0.004 | reference |
| v4.py numpy, DGX CPU (logs/..._1101_v4_d26b7db.txt) | 1.000 | 1.000 | 0.942±0.014 | 0.901±0.004 | IDENTICAL to 3 decimals, all 8 variants |
| v4_torch.py GPU, batch=1 (logs/..._1107_v4_torch_c57d74a.txt) | 1.000 | 1.000 | 0.942±0.014 | 0.901±0.004 | IDENTICAL on all 24 lines (accuracy + margin) |
Conclusion: the world/seeds/evaluation are deterministic and portable; the torch port (autograd + torch Adam, float64) is numerically equivalent
to the hand-written gradient at the reported precision.

## Batch (v4_torch, 3 seeds, N=10000 sentences, lr=0.02; batch>1 = fewer steps, same number of sentences)
| batch | seen | new id. | new form. seen trans. | new form. UNseen trans. | s/variant (training) |
|---|---|---|---|---|---|
| 1 (= numpy) | 1.000 | 1.000 | 0.942±0.014 (m .19) | 0.901±0.004 (m .17) | 28.5 (encoding per step) / ~20 (encoding once) |
| 8 (logs/..._1120_v4_torch_fc97386.txt) | 1.000 | 1.000 | 0.976±0.015 (m .17) | 0.885±0.012 (m .15) | 4.0 (of which 1.0 encoding) |
| 32 (logs/..._1119_v4_torch_f8fedc5.txt) | 1.000 | 1.000 | 0.930±0.005 (m .19) | 0.887±0.016 (m .17) | 2.2 (of which 1.0 encoding) |
| 128 (seed 0 only, informative) | 1.00 | 1.00 | 0.89 | 0.87 | ~3 |
On the other codes, batch 8/32 stays within +-0.03 of batch 1, with two exceptions at batch 32: path 0.52 (vs 0.60) and path+chain 0.59/0.69
(vs 0.71/0.75) — the weak codes lose when the number of steps drops 32x at the same lr. The conclusions of round 3 (ordering of the codes, 0.90 on
unseen transitions with all 4) hold at batch 8-32; batch 128 starts to cost. For the large world: batch 8-32 and/or more sentences.

## Time (D=512, 10k sentences, one variant; 24 variants per full run)
| | s/variant | full run |
|---|---|---|
| numpy, old CPU (round 3) | ~77 | ~30 min |
| numpy, DGX CPU (20 cores, but the code is sequential) | 24.8 | 10 min |
| torch GPU GB10, batch=1, encoding per step | 28.5 | 11.5 min |
| torch GPU, batch=1, encoding once | ~20 | — |
| torch GPU, batch=8 | 4.0 | 1.7 min |
| torch GPU, batch=32 | 2.2 | 1 min |
At D=512 and batch=1, the GPU gains NOTHING: the cost is Python (encoding, 10k small steps), not arithmetic. The gain (10x) comes from encoding
once + batch. Of the 2.2 s at batch 32, 1 s is the encoding in Python (the address structure from v2/v4) — the next place to optimize
if the world grows; the GPU becomes decisive only at D >= 4096 and a large vocabulary.

# ROUND 5 (DGX, v5_scale.py, diag_v5.py): §5.1 — 3x world, 5 seeds, v4 model UNCHANGED
World: 6 frames x 16 formulations (16 shared patterns, different verbs; each verb form = its own category), 18+18 identities (12 training),
nesting 0-3 ('{X} spune ca <P>', '{X} zice ca <P>', '<P> , spune {X}', chosen at random per level; the speakers X1 outer..X3 = roles in the world).
Training: formulations T0,1,2,3,6,12,14; depths 0-2; 30k sentences, batch 8, lr 0.02, D=1024. Log: logs/2026-10-15_1128_v5_scale_0defffd.txt.
New formulations with ALL transitions seen: T9, T10; with UNseen transitions/categories: T4, T5, T7, T8, T11, T13, T15. Chance 1/12 = 0.083 (new id. 1/6).

| code | seen | new id. | new form., seen trans. | UNseen trans. | nesting depth 1-2 (seen) | depth 3 (UNseen) |
|---|---|---|---|---|---|---|
| path | 0.950 (m .23) | 0.952 | 0.829±0.017 | 0.575±0.006 | 0.832 | 0.581 |
| chain | 1.000 (m .48) | 1.000 | 0.232±0.018 | 0.386±0.010 | 0.925 | 0.661 |
| bigram | 0.956 (m .47) | 0.984 | 0.918±0.036 | 0.623±0.022 | 0.907 | 0.651 |
| pos | 0.889 | 0.895 | 0.695 | 0.705 | 0.805 | 0.555 |
| path+bigram | 0.993 | 1.000 | 0.994±0.005 | 0.816±0.009 | 0.920 | 0.652 |
| path+chain+bigram | 1.000 (m .51) | 1.000 | **0.998±0.003** (m .23) | 0.821±0.006 (m .16) | 0.938 | 0.673 |
| all 4 | 1.000 (m .51) | 1.000 | **0.841±0.002** (m .24) | **0.984±0.006** (m .19) | 0.943 | 0.687 |

## The §5.1 criterion (means within +-0.03 of round 3)
- path+chain+bigram: 0.998 / 0.821 (large world) vs 0.987 / 0.824 (small world) => REPRODUCED, stable across worlds.
- all 4: 0.841 / 0.984 vs 0.942 / 0.901 => does NOT reproduce; the two columns are INVERTED. Cause (diag_v5.py, below): the pos code.
- New identities 1.00 for any bundle: confirmed on the large world too. path alone drops to 0.95 on seen (more categories at D=1024).

## Diagnostic 1 (diag_v5.py, seed 0, per formulation x role): why 'all 4' loses on T10 and gains on unseen transitions
T10 '{A} ii V {B} lui {C}': pos alone A 0.00 (all 240 confusions A->C); all 4: A 0.15 (A->C); path+chain+bigram: A 0.96. T9: all 1.00.
Mechanism: in training, the only pattern with a person at position 5 is T1 '{C} VR {B} de la {A}' (there it is the AGENT). pos learns "person at
position 5 = A". In T10 at position 5 is C: pos votes with a COMPLETE match that C is the agent; the first word (A) has in ALL codes an
overlapping A/C address (T0 and T1 both start with a person), hence half a vote. One completely wrong vote beats four correct halves.
On UNseen transitions (per formulation, A/B/C): path+chain+bigram fails exactly where a new CATEGORY or a new transition at the end appears:
  T4 'de catre {A}' A 0.07 ('catre' new) | T11 '{A} il V' A 0.06 ('il' new) | T8 '{A} V lui {C} {B}' B 0.32 (P->O new) | T13 '... {A} {B}' B 0.18 (P->O new);
  all 4 raises them to 0.72 / 0.93 / 1.00 / 1.00 through pos.
Conclusion: pos = ABSOLUTE POSITION PRIOR. It helps when the other codes have zero information (new category) and hurts when the position
regularity from training is false (T10). Both effects are properties of the SET OF FORMULATIONS, not of the mechanism. The figure to report as
"the model" is path+chain+bigram; pos is reported separately, as a prior.

## Diagnostic 2: nesting. Depth 1 = 1.00; depth 2 (SEEN in training) ~0.85; depth 3 UNseen
Per role, seed 0, T0, path+chain+bigram: depth 1: A 1.00 B 1.00 C 1.00 X1 1.00 | depth 2: A 0.84 B 1.00 C 1.00 X1 0.80 X2 0.45 (X2->A 175/240, A->X2, X1->X2)
  | depth 3: A 0.59 B 1.00 C 1.00 X1 0.74 X2 0.30 X3 0.09. No single code separates X2: path 0.28, chain 0.38, bigram 0.33, all 4 0.53.
Cause (logical, confirmed empirically): '{X1} spune ca {A} V ...' (depth 1) and '{X1} spune ca {X2} zice ca {A} V ...' (depth 2) have an IDENTICAL PREFIX
up to the fourth word ('P spune ca P'). Any address computed from the prefix (path, chain, bigram, pos, any bundle) is identical for
A at depth 1 and X2 at depth 2, while the world requires R_A and R_X2 respectively => overlapping address => ~0.5. The role is decided by the NEXT word ('zice' vs verb).
This is, in clean form, the "deferred addressing" problem from round 1, retracted in round 2: for depth 1 and simple frames it is NOT necessary
(confirmed again: depth 1 = 1.00 without rules); for nesting >= 2 the information is NOT in the prefix, so a form of deferral /
retroactive re-addressing / lookahead is necessary. No causal code can solve it, regardless of D or data.
Depth 3: X3 was NEVER a target in training (the maximum training depth is 2) => at chance BY CONSTRUCTION (0.08-0.09 ~ 1/12);
the "depth 3" figure in the table mixes this impossibility with the real degradation of A, X1, X2 (0.59 / 0.74 / 0.30). A new depth = a new
role; it is reported separately (new rule in EVALUATION_PRINCIPLES).

## What remains / what changes with respect to §4
- REMAINS CERTAIN (now 5 seeds, 3x world): factorization + generative objective => aligned roles; new identities 1.00; path+chain+bigram
  0.99 / 0.82 stable across worlds; barrier = truly new transitions/categories (there only a position prior helps further).
- IS CORRECTED: "redundancy 0.60 -> 0.90" becomes "0.58 -> 0.82 (structural codes) + data-dependent position prior";
  "nesting 1.00 without rules" becomes "depth 1 = 1.00; depth >= 2 is NOT solvable with addresses from the prefix".
- NEW: the first clean proof that a form of deferred addressing IS necessary (not for verb/frames, but for identical prefixes with different roles).
Time: 21.7 s/variant (30k sentences, D=1024, batch 8), of which ~13 s Python encoding; full run (7 variants x 5 seeds) 12.7 min.

# ROUND 6 (DGX, English world): language switch check (v5_scale --lang en), left expectations (v6_left), compositional nesting (v7_nested)
From here on advanced tests use the English world (AGENTS.md). Same model, same evaluation, same regime (30k sentences, batch 8, lr 0.02, D=1024, 5 seeds).
English world: 6 frames x 16 formulations; training T0,1,2,3,6,9,12,15; new formulations with all transitions seen: T8 '{B} , {A} V to {C}',
T10 'from {A} , {C} VR {B}'; unseen transitions/categories: T4, T5, T7, T11, T13, T14. Nesting: '{X} says that <P>', '{X} claims that <P>', '<P> , says {X}'.

## 6a. Does the Romanian table reproduce on the English world? (logs/2026-10-15_1246_v5_scale_9ea27cd.txt; flat world as in round 5)
| code | seen | new id. | new form., seen trans. | UNseen trans. | depth 1-2 (seen) | depth 3 (A,B,C,X1,X2 only) |
|---|---|---|---|---|---|---|
| path | 0.970 | 0.984 | 0.501±0.011 | 0.406±0.034 | 0.834 | 0.681 |
| chain | 1.000 | 1.000 | 0.104±0.038 | 0.411±0.008 | 0.930 | 0.784 |
| bigram | 0.973 | 0.989 | 0.768±0.011 | 0.674±0.021 | 0.904 | 0.749 |
| pos | 0.788 | 0.791 | 0.648±0.005 | 0.790±0.017 | 0.705 | 0.556 |
| path+bigram | 0.980 | 0.995 | 0.764±0.022 | 0.755±0.025 | 0.910 | 0.741 |
| path+chain+bigram | 1.000 (m .51) | 1.000 | **0.785±0.016** (m .15) | 0.788±0.014 (m .16) | 0.935 | 0.780 |
| all 4 | 1.000 (m .51) | 1.000 | 0.709±0.019 | **0.993±0.005** | 0.939 | 0.807 |
Reproduces (within +-0.03 of the Romanian world): seen 1.00, new identities 1.00, UNSEEN transitions 0.788 vs 0.821 (model) and 0.993 vs
0.984 (with pos), depth 1-2 0.935 vs 0.938. Does NOT reproduce: "seen transitions" 0.785 vs 0.998.
Per formulation x role (seed 0, path+chain+bigram): T8 A 1.00 B 1.00 C 1.00; T10 A 0.81 B 1.00 **C 0.04** (C decoded as a third person 172/180).
Cause: in training the only formulation with a person after a comma is T9 'to {C} , {A} V {B}' (the AGENT); bigram ',:P' therefore means A,
and the chain/path combination 'from ... , P' is new => C has no correct address anywhere. Every single code fails on T10-C (path 0.10,
bigram 0.06, chain 0.08). The "seen transitions" axis measures how REDUNDANT the formulation set is, not the mechanism: the Romanian T9/T10
were near-duplicates of training formulations ('lui C A da B' vs 'lui C A ii da B'); the English T10 is a genuine role ambiguity after a comma
that only the preceding 'from A' (by exclusion) or the following recipient verb could resolve. Same family as the pos/T10 effect in round 5:
a regularity true in all training formulations and false in a test one.

## 6b. §5.1b — left expectations / re-addressing (v6_left.py; logs/2026-10-15_1249_v6_left_9ea27cd.txt; flat world, English)
Mechanism: an emitter also emits LEFT expectations; the most recent non-emitter of each category before it, not yet re-addressed, is
re-addressed to address(emitter) (x) L[c_emitter, c]; one re-addressing per word, by the first emitter that follows it; emitters anchor.
Trace ('p0 says that p1 claims that p2 gives o0 to p3 .'): p1 = says:that+that:claims+L(claims:P); p2 = ...+that:gives+L(gives:P); at
depth 1 p1 = says:that+that:gives+L(gives:P). The identical prefix no longer matters; what follows decides. Union-then-restriction
(write at all candidates, subtract the losers) equals address replacement on the final state, which is what is trained.
| code | seen | new id. | seen trans. | UNSEEN trans. | depth 1 | depth 2 (seen) | depth 3 (A,B,C,X1,X2) |
|---|---|---|---|---|---|---|---|
| chain | 1.000 | 1.000 | 0.102 | 0.415 | 1.000 | 0.859 | 0.784 |
| chainL | 1.000 | 1.000 | 0.069 | 0.456 | 1.000 | 0.903 | 0.778 |
| path+chain+bigram | 1.000 | 1.000 | 0.784 | 0.791 | 0.998 | 0.869 | 0.779 |
| path+chainL+bigram | 1.000 | 1.000 | 0.745 | 0.798 | 1.000 | 0.903 | 0.788 |
Depth 2 PER ROLE (A | B | C | X1 | X2): chain 0.964 | 1.000 | 0.971 | 0.703 | 0.647 ; chainL 1.000 | 1.000 | 1.000 | 0.772 | 0.773 ;
  path+chain+bigram 0.960 | 1.000 | 0.966 | 0.766 | 0.653 ; path+chainL+bigram 1.000 | 1.000 | 1.000 | 0.782 | 0.760.
Criterion (X2 > 0.9 at depth 2, no per-role loss): NOT met. Left expectations fix the INNER roles completely (A, C 0.96 -> 1.00) and move the
speakers only from 0.65 to 0.77; cost 0.04 on "seen transitions" (T10, see 6a). Why the speakers stay stuck — an algebraic constraint of the
FLAT world: with cumulative addresses, A depth-invariant forces the chain through 'says that' to sum to zero, and then X2's address equals
X1's. Writing a = says:that, b = claims:that, c = that:says, d = that:claims, g = that:gives: A(d1) => a+g = 0 and b+g = 0; A(d2) =>
d+b = 0 and c+a = 0; X2 - X1 = a+d = 0. Exact equality is impossible; argmax decoding softens it to ~0.77 (round-1 lesson: rule 10 —
measured, not only argued). The flat world (depth-indexed speaker roles X1..X3, depth-invariant A,B,C) asks a compositional address
scheme for something no compositional scheme can give. The model is not what is wrong here; the world is.
