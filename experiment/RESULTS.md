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
depth 1 p1 = says:that+that:gives+L(gives:P). The identical prefix no longer matters; what follows decides. As implemented, this is a
RE-ADDRESSING: the word's address list is replaced when the first following emitter arrives (the final state is what is trained). A
union-then-restriction variant (write at all candidates, subtract the losers) is NOT implemented; if it is, this text changes.
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

## 6c. Nesting with a COMPOSITIONAL world (v7_nested.py): S_world(X says that P) = R_X f_X + R_rep (x) S_world(P); one speaker role; depth = number of R_rep
The flat world of rounds 5-6b asked for depth-invariant inner roles and depth-indexed speakers through the same prefix (impossible, 6b). The
compositional world defines reported speech the way the model addresses: entering a clause binds the inner state with a fixed R_rep (a fibre).
Decoding A at depth d: unbind conj(R_rep)^d conj(R_A). No new role at a new depth (rule 14): depth 3 is a fair unseen-construction test.
Model unchanged, no rule for 'that'. Same regime. Logs: all 3 outer forms logs/2026-10-15_1311_v7_nested_043caf8.txt; prefixed forms only
('{X} says that <P>', '{X} claims that <P>'; the postposed '<P> , says {X}' dropped) logs/2026-10-15_1319_v7_nested_e1e7011.txt.
| world / code | seen | new id. | seen trans. | UNSEEN trans. | depth 1 | depth 2 (seen) | depth 3 (UNSEEN) |
|---|---|---|---|---|---|---|---|
| all 3 outer forms, path+chain+bigram | 1.000 | 1.000 | 0.735 | 0.751 | 0.971 | 0.815 | 0.457 |
| all 3 outer forms, path+chainL+bigram | 1.000 | 1.000 | 0.715 | 0.760 | 0.980 | 0.881 | 0.538 |
| PREFIXED only, path+chain+bigram | 1.000 | 1.000 | 0.757 | 0.740 | 0.982 | **0.997** | **0.979** |
| PREFIXED only, path+chainL+bigram | 1.000 | 1.000 | 0.793 | 0.745 | **1.000** | **1.000** | **1.000** |
Per role, prefixed only: depth 2: both variants 1.00 on A, B, C, X1, X2 (chain alone: C 0.987). Depth 3 UNSEEN: path+chain+bigram A 0.999,
B 1.000, C 0.976, X1 1.000, X2 1.000, X3 0.898±0.044; path+chainL+bigram 1.000 on all six roles, margin 0.23.
Per role, all 3 forms: depth 2 X1/X2 0.66-0.78 (as in the flat world); depth 3 0.34-0.67.
Conclusions:
- With a compositional world and reporting announced BEFORE the clause, nesting of depth 2 is solved exactly (1.00 on every role) by the
  plain chain code — no rule, no buffer — and the learned displacements COMPOSE ONE LEVEL FURTHER THAN SEEN: depth 3, never in training,
  decodes at 1.00 on all roles including the third speaker. This is the first clean positive on an unseen CONSTRUCTION (not identities,
  not formulations): the algebra of the world (R_rep applied d times) and the algebra of the model (displacements summed along the chain)
  coincide, so the generalization is by construction once the parameters land on the consistent solution (6b derivation: a+g = rho etc.).
- Left expectations (chainL) are what closes the last gap at the unseen depth (X3 0.90 -> 1.00, C 0.976 -> 1.000) and raise the margin.
- What remains impossible for ANY causal address: a displacement announced AFTER the clause it applies to (postposed '<P> , says X'):
  the inner clause is already written at depth 0 when 'says' arrives. With the postposed form present, depth 2 falls to 0.82-0.88 and depth 3
  to ~0.5. This replaces the round-5 statement "nesting >= 2 is unsolvable from the prefix" by the precise one: nesting is solvable from the
  prefix exactly when the world is compositional and the structure is announced before its scope; right-announced structure (postposed
  reporting, and by the same logic the English T10 comma construction) needs either retroactive re-addressing of a whole clause
  (a displacement applied to a bundle: S <- S_outside + R (x) S_clause, i.e. the zoom-in of round 1, now forced by data for THIS case only)
  or a world that does not demand it.
- Cost of the compositional world on flat axes: none on seen / new identities; "seen transitions" 0.74-0.79 and "unseen transitions" 0.74-0.76
  (vs 0.785 / 0.788 flat): within +-0.05, same T10 failure as 6a.

# ROUND 7 (DGX, reference world = compositional, English, prefixed reporting): control, rebuilt axis, gates by gradient
Decisions applied (judgment 2026-10-15): compositional world is the reference; §5.1c postponed and declared a known limit of causal
addressing; §5.2 gates by gradient; transformer control on test 6c; "seen transitions" axis rebuilt and kept.

## 7a. Rebuilt axis: "new formulations with all ROLE CUES seen" (v5_scale.split_templates_rolecues)
A held-out formulation passes iff for every role slot both local cues — (previous category, role) and (role, next category) — occur in a
training formulation with the SAME role. English world, training T0,1,2,3,6,9,12,15: role cues seen = T8 '{B} , {A} V to {C}', T13 '{B} is what
{A} V to {C}'; unseen role cues = T4, T5, T7, T10, T11, T14. T10 'from {A} , {C} VR {B}' (C 0.04 in 6a) correctly moves to the unseen group
(the cue ', C' was only seen as ', A'). The old "seen transitions" axis is still reported next to it.

## 7b. CONTROL on test 6c: small transformer, same task (v9_transformer.py; logs/2026-10-15_1414_v9_transformer_75e7e0d.txt)
Same fixed identities F at the input (new identities = new random vectors), same objective -cos(S, S_world) with the compositional world,
same decoder (unbinding conj(R_rep)^k conj(R_role)), same 30k sentences (depths 0-2, TRAIN_T), 12 epochs, batch 32, AdamW + cosine.
Architecture: linear(2D -> 128) + sinusoidal positions -> 2-layer pre-norm TransformerEncoder (4 heads, d_ff 256) -> masked mean pooling ->
linear(128 -> 2D) -> complex S; 0.79M parameters (VSA: ~1.6M phases). Minimal tuning on seed 0 by VALIDATION on fresh training-distribution
sentences (never on a test axis): lr 1e-3 / dropout 0 -> val cos 0.963; 3e-4 / 0 -> 0.962; 1e-3 / 0.1 -> 0.867; 3e-4 / 0.1 -> 0.916.
Chosen: lr 1e-3, dropout 0. Then 5 seeds (same world/identity seeds as the VSA runs).
| model | seen | new ids | seen trans. | UNSEEN trans. | ROLE CUES seen | depth 1 | depth 2 (seen) | depth 3 (UNSEEN) |
|---|---|---|---|---|---|---|---|---|
| transformer (tuned) | 0.971±0.057 (m .48) | **0.215±0.012** (m -.02) | 0.770±0.122 | 0.783±0.057 | 0.968±0.050 | 0.839±0.095 | 0.797±0.008 | **0.390±0.024** (m .11) |
| VSA path+chainL+bigram (6c, same regime) | 1.000 | 1.000 | 0.793 | 0.745 | (7c) | 1.000 | 1.000 | **1.000** |
Per role, transformer: depth 2 (seen): A 0.486, B 1.000, C 0.497, X1 1.000, X2 1.000 — it never separates agent from recipient inside a
doubly reported clause even on TRAINING data; depth 3 (unseen): A 0.082, B 0.100, C 0.096 (chance 0.083), X1 1.000, X2 1.000, X3 0.063.
Reading: (i) new identities 0.215 ~ chance 0.167: a fixed random input vector it has never seen is not usable — the VSA model gets 1.00
because identities never enter its parameters (property of the construction; here measured against a control on the same input).
(ii) At the unseen depth the transformer places the two speakers it knows and puts everything else at chance: it reproduces the deepest
configuration it has seen (two R_rep) rather than composing a third; the VSA model composes it by construction. (iii) On flat axes the
two are comparable (seen trans. 0.77 vs 0.79; unseen trans. 0.78 vs 0.75; role cues 0.97). Same caveat as round 1: "at this data regime,
a small transformer trained identically does not generalize to new identities or to a deeper construction" — not "transformers cannot".
What the control does NOT cover: more data / epochs, a larger model, learned input embeddings (which would remove the fixed-identity
constraint and with it the zero-shot test). Parameter counts are comparable (0.79M vs ~1.6M).

## 7c. §5.2 — GATES LEARNED BY GRADIENT (v8_gates.py; logs/2026-10-15_1412_v8_gates_06e12bd.txt; 5 seeds, same regime as 6c)
Soft chain: g_c = sigmoid(s_c) = probability that category c emits; the address of a word is the gate-weighted mixture over "the most
recent open emitter before it" (right) and, for a non-emitter, "the first open emitter after it" (left re-addressing, v6); fully in torch,
differentiable in the gates; no consumed-expectation bookkeeping. Hand gates (fixed at +-6) = reference; learned from an all-closed start
(g = 0.12) and from an all-open start (g = 0.88). Bundle path + soft chainL + bigram, and soft chainL ALONE. 32 categories.
| variant | seen | new ids | seen trans. | UNSEEN trans. | ROLE CUES seen | depth 1 | depth 2 (seen) | depth 3 (UNSEEN) |
|---|---|---|---|---|---|---|---|---|
| bundle, hand gates (fixed) | 1.000 | 1.000 | 0.783±0.031 | 0.756±0.012 | 1.000 | 1.000 | 1.000 | 1.000 (all 6 roles) |
| bundle, learned, start closed | 1.000 | 1.000 | **0.848±0.064** (m .25) | **0.870±0.038** | 1.000 | 1.000 | 1.000 | 0.996 (A 0.986, C 0.991) |
| bundle, learned, start open | 1.000 | 1.000 | 0.790±0.021 | 0.736±0.007 | 1.000 | 1.000 | 1.000 | 1.000 (all 6 roles) |
| chainL ALONE, hand | 1.000 | 1.000 | 0.059 | 0.469 | 0.221 | 1.000 | 1.000 | 1.000 |
| chainL ALONE, learned closed | 1.000 | 1.000 | **0.614** | **0.628** | **1.000** | 1.000 | 1.000 | 1.000 |
| chainL ALONE, learned open | 1.000 | 1.000 | 0.627 | 0.537 | 1.000 | 1.000 | 1.000 | 1.000 |
The soft chain with hand gates reproduces 6c exactly (the consumption bookkeeping costs nothing). Role-cues axis: 1.000 for every bundle.
Learned gates (mean sigmoid over 5 seeds; OPEN if > 0.5), identical sets in every seed:
  bundle, start closed:  OPEN that 1.00, to 1.00, by 0.91, '.' 0.67;  from 0.20;  everything else 0.00-0.03.
  bundle, start open:    OPEN that, to, from, by, COP, who, it, says, ',', claims, what (all 1.00);  CLOSED all 18 verb forms (<= 0.02), O 0.01, P 0.00.
  chainL alone, closed:  OPEN that, to, from, by (0.97-0.99), ',' 0.81;  CLOSED everything else (<= 0.01) including P, O, all verbs, says, claims.
  chainL alone, open:    OPEN from, to, that, by, ',', COP 0.72, it 0.58;  who 0.38;  CLOSED P, O, all verbs, says, claims (<= 0.02).
Criterion ("same set as by search: identities closed, verbs/prepositions open", no loss in 6c): HALF met.
- Met: P and O are closed in every seed and every start (identities never open frames) — the part the search also found; no loss on the
  6c table (depth 1-3 at 1.00; bundle-closed 0.996); on new formulations the learned gates are BETTER than the hand set (seen trans.
  0.78 -> 0.85, unseen 0.76 -> 0.87 in the bundle; chainL alone 0.06 -> 0.61 and role cues 0.22 -> 1.00): fewer emitters = fewer
  untrained transitions, the round-2 observation again.
- Not met: the VERBS are closed in every seed, from both starts, with or without the other codes. The gradient's set is the function
  words that announce a relation — prepositions (to, from, by), the connective 'that', the comma, the copula — plus 'says/claims' only
  from the open start (they are not needed: 'that' carries the clause). With left re-addressing available, a verb has nothing to emit
  that 'to'/'by'/'that' and the left slot do not already give; the search of round 2 (no left expectations, Romanian small world) needed
  the verbs because only emitters could place the words after them.
- Reading for the paper: the gates DO emerge from the single generative objective, stably across 5 worlds, and they separate the vocabulary
  into identities (closed), relation markers (open) and action words (closed: identities of actions, not operators). That is a cleaner
  emergent partition than "verbs + prepositions open"; the hand set was a linguistic prior, not the optimum. The criterion as written
  assumed the search's set was the target; the honest statement is that gradient and search agree on what must be closed and disagree
  on verbs, and the gradient's choice is the better one on every axis measured. Start-dependence (closed start opens 4-5 categories, open
  start 11) means the objective is flat over the extra function words: they are harmless, not needed.
Open issue: the soft chain is ~5x slower than the hard one (134 s vs 22-38 s per variant): the position loop in torch. Acceptable for now.

## Round 7 conclusions
1. Gates emerge by gradient (5 seeds): identities closed, relation markers open, verbs closed; no loss at depths 1-3; better on new formulations.
2. The transformer control at the same regime fails exactly where the construction gives the VSA model its generalization: new identities
   (0.22 ~ chance) and the unseen depth (0.39; inner roles at chance, speakers perfect).
3. The rebuilt "role cues seen" axis is at 1.000 for every bundle; the old "seen transitions" axis stays as a redundancy measure.

# ROUND 8 (DGX): the two cheap controls on "emergent" (v10_controls.py), then WikiSQL (v11_wikisql.py)
## 8a. Controls (logs/2026-10-15_1530_v10_controls_727aecb.txt; reference world, same 30k sentences, 5 seeds)
Control A — word2vec for categories (§5.3). 54 content words, 5 classes (P 18, O 18, V 6, VP 6, VR 6); agglomerative clustering, k = 5;
chance purity 0.333, nearest-neighbour chance 0.245.
| representation | purity | ARI | NN-category |
|---|---|---|---|
| model: untied learned displacements (one per word, generative objective) | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| skip-gram on the same sentences (dim 64, window 2, 5 epochs) | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |
Verdict: the categories ARE recovered from the learned displacements — and equally perfectly by a plain distributional embedding on the
same sentences. On a synthetic world the categories are trivially distributional (persons occupy the same slots). "Emergence of
categories" is therefore NOT a distinctive merit of the architecture and is RETRACTED as a contribution (POSITIONING article 1 (iii));
it stays as a consistency check (the displacements do carry the category). Consequence for real text: the categories the model needs can be
supplied by distributional clustering (that is what v11_wikisql does), with no loss of honesty.
Control B — associative memory with a bottleneck (§5.4). S = U V x, x = sum_k pos_k (x) F[w_k] (fixed random positions, order-aware,
structure-free), complex linear, rank r; same objective, same decoder, 3 epochs.
| rank r | seen | new ids | seen trans. | UNSEEN trans. | ROLE CUES seen | depth 1 | depth 2 (seen) | depth 3 (UNSEEN) | depth 3 per role |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.282 | 0.165 | 0.249 | 0.208 | 0.286 | 0.351 | 0.310 | 0.245 | X1 0.996, rest ~chance |
| 32 | 0.666 | 0.235 | 0.651 | 0.519 | 0.706 | 0.725 | 0.688 | 0.328 | X1 1.00, X2 0.59, A/B/C/X3 chance |
| 128 | 0.738 | 0.238 | 0.833 | 0.369 | 0.853 | 0.827 | 0.864 | 0.392 | X1 1.00, X2 1.00, A/B/C/X3 chance |
| 1024 (= D, no bottleneck) | 0.737 | 0.239 | 0.833 | 0.355 | 0.850 | 0.829 | 0.865 | 0.389 | X1 1.00, X2 1.00, A/B/C/X3 chance |
| VSA bundle, learned gates (7c) | 1.000 | 1.000 | 0.848 | 0.870 | 1.000 | 1.000 | 1.000 | 0.996 | all 1.00 |
Verdict: even with no bottleneck, a linear associative map on an order-aware input does not reach the model on seen data (0.74), gets
nothing on new identities (0.24 ~ chance 0.17) and shows exactly the transformer's signature at the unseen depth (known speakers perfect,
inner roles at chance). Capacity is not what is missing: rank 128 = rank 1024. "A generative construction with a fixed operation is
necessary" stands against this control; it does not narrow to "limited capacity is necessary". (The per-position linear map could in
principle implement 'unbind pos_k, bind R_role' as a diagonal complex map, but only for FIXED positions; free word order and nesting break
the position-role correspondence, which is the point of the displacement codes.)

## 8b. WikiSQL with the full model (v11_wikisql.py; logs/2026-10-15_1534_v11_wikisql_2fb26b9.txt; seed 0; D = 512, 3 epochs, batch 32)
Train 56,355 questions, dev 8,421 (all), vocabulary 33,320 word types. Identities fixed random per word type. Categories: 42 frequent tokens
as singletons + 48 skip-gram clusters (k-means on skip-gram vectors trained on the same questions and headers) = 90 categories (the
round-8a lesson applied: distributional clustering is an honest source of categories). Codes: path + bigram + soft chainL with gates
learned by gradient (start closed). World: S_sql = SEL (x) bundle(selected column) + AGG (x) f(agg) + sum_k COND_k (x) [COL (x) bundle(col)
+ OP (x) f(op) + VAL (x) bundle(value)]. Decoding with the world's addresses; the selected / condition column is chosen among the TABLE's
columns (as in round 1); the table is otherwise NOT seen in input mode 'q'. Mode 'qh' = header tokens + SEP + question at the input.
| input | selected column | aggregator | condition-1 column | cos(S, S_sql) | same-table pairs: same skeleton / different / HARD (same sel) | separation |
|---|---|---|---|---|---|---|
| chance / majority | 0.166 | 0.715 | — | — | — | — |
| trivial: token overlap only | 0.484 | — | — | — | — | — |
| BASELINE BoW + logistic regression | 0.494 | **0.867** | — | — | — | — |
| round 1 (position code, numpy, 2 epochs) | 0.494 | 0.586 | 0.249 | 0.126 | 0.651 / 0.233 | 0.42 (n=121 pairs) |
| **full model, question only (q), epoch 2** | **0.824** | 0.813 | **0.631** | 0.638 | 0.764 / 0.484 / 0.559 (n = 336 / 12,227 / 1,866) | 0.280 |
| full model, header + question (qh), epoch 2 | 0.673 | 0.799 | 0.484 | 0.625 | 0.907 / 0.762 / 0.779 | 0.146 |
Epoch curve (q): sel 0.792 -> 0.813 -> 0.824; cond1 0.574 -> 0.622 -> 0.631; agg 0.798 -> 0.811 -> 0.813. 104 s / epoch (GPU).
Learned gates on real text (q, most open): than 1.00, and 1.00, has 1.00, the 0.96, what 0.91, which 0.79, of 0.71, is 0.69; all 48 content
clusters <= 0.52 (most ~0): the same partition as on the synthetic world — relation markers / wh-words open, content closed — found again
without any supervision on real questions.
Reading:
- Selected column WITHOUT seeing the table: 0.824 vs 0.494 for BoW + LR and for the token-overlap control — the model recovers the column
  from how the question is phrased around the content words, which the bag of words cannot. The round-1 criterion (README_DGX: sel > 0.7
  without the table) is met; condition column 0.63 (round 1: 0.25).
- Aggregator: 0.813 vs 0.867 for BoW + LR — BELOW the baseline. The aggregator is a lexical signal ('how many', 'total', 'average', 'highest')
  that a bag of words captures directly; in our state it is one term among many (AGG (x) f_agg) and competes with the column and condition
  terms. The criterion agg > 0.85 is NOT met. Equivalence separation 0.28 (criterion > 0.3): not met either; the HARD pairs (same selected
  column, different aggregator or condition) at 0.56 vs same-skeleton 0.76 shows the state does separate them, by 0.2.
- Table visible at the INPUT hurts: the header tokens enter the bundle with their own identities and dominate the state (all same-table
  questions share them: same-table cosines rise to 0.76-0.91, separation falls to 0.15), and sel drops to 0.67. The right place for the
  table in this formulation is the decoder (candidate columns), where it already is; a header-aware model would need the header bound to
  a role (zoom-in), not added.
- Honest status for "is there an empirical paper": the selected-column result is a real, large advantage over the mandatory baseline on
  real text with executable consequence, at a single seed and without a tuned neural baseline. Before any claim: 3 seeds; the test split
  (15,878); a tuned small transformer / BERT-small at 1k / 5k / 56k; and the aggregator weakness addressed (a dedicated code or a
  separate readout) or reported as is. Spider (nested SQL) only after that.

## 8c. WikiSQL, 3 seeds, dev AND test, full-query accuracy (logs/2026-10-15_1614_v11_wikisql_9916e38.txt; question only; D = 512, 3 epochs)
No leakage: skip-gram clusters trained on TRAIN questions and TRAIN headers only (asserted and logged from this commit on); dev/test words
have fixed random identities and, if unseen in train, an untrained skip-gram vector => a random cluster (a handicap for the model, not a leak).
| split | model | selected column | aggregator | condition-1 column | FULL query (sel + agg + all condition columns) | cos(S, S_sql) | same / different skeleton | HARD (same sel) | separation |
|---|---|---|---|---|---|---|---|---|---|
| dev (8,421) | chance / majority | 0.166 | 0.715 | — | — | — | — | — | — |
| dev | BoW + LR | 0.497 | **0.870** | — (no condition model) | 0.437 (sel + agg only) | — | — | — | — |
| dev | full model, 3 seeds | **0.810±0.004** | 0.814±0.003 | **0.632±0.004** | **0.376±0.003** | 0.629 | 0.758 / 0.471 | 0.547 | 0.287±0.001 |
| test (15,878) | full model, 3 seeds | **0.811±0.004** | 0.815±0.002 | 0.629±0.001 | **0.380±0.002** | 0.626 | 0.746 / 0.471 | 0.547 | 0.275±0.002 |
Per seed (dev / test): sel 0.815/0.817, 0.811/0.807, 0.805/0.810; FULL 0.380/0.383, 0.375/0.378, 0.372/0.379. 180-234 s per seed (3 epochs).
Learned gates, stable across seeds: has, and, than, the (0.99-1.00), is (0.8-0.9), of (0.75), was, which, what, a (0.2-0.7); content clusters
<= 0.65 (most < 0.3). Same partition as the synthetic world (relation markers open, content closed), now on 56k real questions, 3 seeds.
Reading: dev and test agree to the third decimal; the selected column WITHOUT the table is 0.81 vs 0.50 for the mandatory baseline on both
splits; the aggregator stays below the bag of words (lexical decision, not structural — reported as is, no dedicated head); the FULL query
(the number text-to-SQL papers compare) is 0.38, far below the state of the art (~0.9 with large pretrained models and the table as input)
and above the baseline's sel+agg-only 0.44 is not comparable (no conditions there). The equivalence separation is 0.28-0.29 (criterion 0.3
not met); whether it is lexical or role-based is what round 8d (v12: hard pairs with the lexical bag on the same row) measures.

## 8d. WikiSQL: same-input baselines, data-efficiency curve, hard SQL pairs (v12_wikisql_baselines.py; logs/2026-10-15_1628_v12_wikisql_baselines_ed24ce4.txt)
Setup. Identical input for ours and the transformer: the sequence of FIXED identity vectors of the question tokens + the same 90 category ids
(42 frequent tokens + 48 skip-gram clusters), no table at the input; identical objective -cos(S, S_sql); identical decoder (table columns only
as decoding candidates). Transformer: linear(2D -> 128) + learned category embedding + sinusoidal positions, 2 pre-norm layers, 4 heads,
masked mean pooling, linear -> complex S; lr chosen ONCE on 2,000 questions held out from TRAIN at 56k (1e-3: val cos 0.616 vs 3e-4: 0.575),
then fixed; AdamW + cosine. No extra tuning for ours. Epochs scaled for comparable steps: 60 / 20 / 3 at 1k / 5k / 56k. 3 seeds. The
skip-gram categories are trained on the WHOLE train pool (~54k questions + headers) at every size, for both models: the curve measures data
efficiency "with distributional categories from the whole training corpus, without labels", not from 1k sentences alone.
Curve (dev; test at 56k in brackets; means of 3 seeds, std <= 0.005 everywhere):
| n train | model | sel column | aggregator | cond-1 column | FULL query |
|---|---|---|---|---|---|
| 1k | BoW + LR | 0.475 | 0.762 | — | 0.371 (sel+agg only) |
| 1k | **ours** | **0.632** | 0.766 | **0.387** | **0.164** |
| 1k | transformer, same input | 0.415 | **0.819** | 0.360 | 0.115 |
| 5k | BoW + LR | 0.502 | 0.810 | — | 0.413 (sel+agg only) |
| 5k | **ours** | **0.760** | 0.800 | **0.555** | **0.294** |
| 5k | transformer, same input | 0.510 | **0.864** | 0.415 | 0.164 |
| 56k | BoW + LR | 0.499 | 0.870 | — | 0.441 (sel+agg only) |
| 56k | **ours** | **0.816** [test 0.821] | 0.801 [0.803] | **0.626** [0.629] | **0.368** [0.374] |
| 56k | transformer, same input, 3 epochs (23 s) | 0.543 [0.544] | **0.893** [0.893] | 0.450 [0.433] | 0.196 [0.190] |
Robustness check (same-input transformer at 56k, seed 0; logs/2026-10-15_1652_... and _1655_v12_wikisql_baselines): 3 epochs (23 s) sel 0.543;
12 epochs (72 s) 0.631 dev / 0.626 test, FULL 0.281; 30 epochs (961 s) 0.649 dev / 0.641 test, FULL 0.299, agg 0.883, cond-1 0.558.
The gate set in advance (0.75+ at 10x would narrow the claim to 1k-5k) is NOT reached: the data-efficiency claim stands at every size.
Ablation — categories from the SUBSET only (skip-gram clusters trained on the 1k / 5k training questions and their headers, both models; seed 0;
logs/2026-10-15_1700_v12_wikisql_baselines_38bc65d.txt): ours 1k sel 0.654 (vs 0.632 with whole-corpus categories), FULL 0.174; 5k 0.751 (vs
0.760), FULL 0.288; transformer 1k 0.414 (vs 0.415), 5k 0.516 (vs 0.510); BoW+LR 0.459 / 0.495. The gap is unchanged: the data-efficiency
claim does not depend on categories from the whole corpus; the condition can be stated as "with distributional categories from the training
questions themselves" at every size.
Hard pairs (dev, same table): H1 "different words, same SQL" (Jaccard < 0.3): 26 pairs; H2 "same words, different SQL" (Jaccard >= 0.5): 1,353 pairs.
| representation | same skeleton | different | separation | AUC same-vs-different | H1 (want HIGH) | H2 (want LOW) |
|---|---|---|---|---|---|---|
| bag of identities (lexical overlap only, same space) | 0.760 | 0.462 | 0.298 | 0.858 | 0.337 | 0.772 |
| ours, 56k (3 seeds) | 0.755 | 0.464 | 0.291 | **0.914** | **0.462** | **0.622** |
| transformer, same input, 56k (3 seeds) | 0.910 | 0.616 | 0.293 | 0.900 | 0.859 | 0.744 |
Reading:
- The same-input transformer does NOT reach us at any size on the selected column (0.54 vs 0.82 at 56k) or the full query (0.19 vs 0.37); it
  beats both us and the bag of words on the aggregator (0.89 vs 0.80 vs 0.87) — the aggregator is a lexical decision and the transformer is
  the better lexical classifier; we report it as is, with no dedicated head. The data-efficiency claim holds at every size with the wording
  above (categories from the whole unlabelled corpus). The 56k transformer trained for 23 s (3 epochs, steps matched to ours); a longer
  budget is checked right below so that "does not reach us" is not an artefact of the step budget.
- Equivalence: the OVERALL separation is lexical-level — ours 0.291, the bag 0.298 — so the headline "same skeleton vs different" metric does
  not test roles on WikiSQL (as foreseen in the judgment: the set has only 26 pairs with different words and the same SQL). On the HARD
  pairs, however, the model moves both directions the right way relative to the lexicon: H1 (different words, same SQL) 0.462 vs 0.337 for
  the bag (+0.13, n = 26: weak evidence); H2 (same words, different SQL) 0.622 vs 0.772 (-0.15, n = 1,353: solid), and the AUC rises from
  0.858 to 0.914. The transformer compresses everything (same 0.91 / different 0.62) and keeps H2 at the lexical level (0.744). Kept as a
  negative-with-a-nuance: "on WikiSQL the equivalence separation is lexical; the set lacks pairs with the same words and different SQL to
  test roles; where such pairs exist (H2), the role state separates them 0.15 better than the lexicon". Metric unchanged.
- FULL-query accuracy 0.37 is far below text-to-SQL systems with the table at the input and pretrained encoders (~0.9); this is a model with
  ~2.5M phases, no pretraining, no table, trained in 2.5 minutes — the comparison that matters here is the same-input one.

# ROUND 9 (DGX): the verifier / formalizer programme — canonical frame (R1), mechanical translation (R2), operators as roles (R3-R4)
## 9a. frame(S), diff, to_prolog, to_smt and the round trip (frame.py, frame_roundtrip.py; logs/2026-10-15_1705_frame_roundtrip_a622582.txt)
frame(S): for each depth k = 0..3, unbind conj(R_rep)^k conj(R_role) and clean up over the candidates; the depth is ESTIMATED as the one at
which the three inner roles decode best (not given); speakers read at levels 0..d-1. diff(fa, fb) -> missing / extra / different.
to_prolog: valid SWI-Prolog facts (nested term event(e, says(p4, says(p11, says(p8, gives(p2, o1, p3))))) + role facts), executed by a
minimal built-in evaluator (no SWI-Prolog on the machine). to_smt: SMT-LIB 2 declarations and assertions, checked with z3 5.3.0 (the world's
recipient asserted: sat; its negation: unsat). Model: v8 bundle, learned gates, reference world, trained on depths 0-2; 100 fresh sentences
per depth, 3 seeds.
| depth | full-frame accuracy (all roles + depth at once) | Prolog query recipient(e,R), agent(e,A) correct | SMT consistency (z3) |
|---|---|---|---|
| 0 | 1.000±0.000 | 1.000 | 1.000 |
| 1 | 1.000±0.000 | 1.000 | 1.000 |
| 2 | 1.000±0.000 | 1.000 | 1.000 |
| 3 (UNSEEN) | **1.000±0.000** | 1.000 | 1.000 |
The depth estimate separates cleanly (depth scores e.g. [0.10, 0.04, 0.06, 0.22] for a depth-3 sentence). From now on every evaluation
reports full-frame accuracy (R1). This is the product skeleton in minimal form: text -> state -> frame -> Prolog / SMT -> answer, no learning
anywhere after the state.

## 9b. E1 — NEGATION as a role (v13_negation.py; logs/2026-10-15_1709_v13_negation_5b9e933.txt; 5 seeds, D = 1024, 40k sentences, batch 8)
World: R_neg bound to its scope — sentential (R_neg (x) S), on an argument (R_role (x) R_neg (x) f), negative quantifier (R_role (x) R_neg (x)
f_someone / f_something). Eight negated formulations (two per scope): 'does not V', 'never V3' (S); 'it is not X who', 'nobody' (A); 'nothing',
'does not V anything' (B); 'not to C', 'to nobody' (C). Model unchanged (v8 bundle, gates learned, start closed); every operator word is an
ordinary category. Training: affirmative at depths 0-2 + negated forms at DEPTH 0 only (p = 0.4). Decoding (rule 17): five hypotheses
{none, S, A, B, C}, argmax of the mean best cosine over the three roles under the keys each hypothesis implies; scope = the hypothesis,
presence = hypothesis != none. Chance: fillers 1/13, scope 0.20.
| axis | result (5 seeds) | criterion |
|---|---|---|
| affirmative, depth 0: roles | 1.000 | — |
| negated, depth 0 (seen forms, fresh events): roles | 1.000±0.000 | > 0.95 MET |
| negated, depth 0: negation PRESENCE | 1.000±0.000 | — |
| negated, depth 0: SCOPE (which of S/A/B/C) | **1.000±0.000** | > 0.9 MET |
| negated, depth 0: FULL frame (roles + scope + depth) | **1.000±0.000** | > 0.95 MET |
| pair separation cos(affirmed, negated), same event | model **0.184±0.027** (gap 0.82) ; bag of identities 0.722 ; world 0.034 | gap > 0.5 MET |
| affirmative, depth 3 (UNSEEN): roles | 0.993 | (unchanged by adding negation) |
| negation INSIDE reporting, depth 1 (UNSEEN combination): scope | 0.559±0.065 | > 0.9 NOT met |
| negation inside reporting, depth 1: FULL frame | 0.451±0.070 | — |
| negation inside reporting, depth 2: FULL frame | 0.527±0.060 | — |
Per form at the unseen combination (seed 0, depth 1 / depth 2): argument-scoped forms compose — 'nobody V' 1.00 / 1.00; 'to nobody' 1.00 /
1.00; 'it is not X who' 0.85 / 0.80; 'nothing' 0.90 / 0.37; 'does not V anything' 0.70 / 0.43 — while SENTENTIAL negation does not: 'does
not V' 0.20 / 0.53 and 'never V3' 0.00 / 0.13, decoded as scope B (or C) instead of S: inside a reported clause the negation attaches to the
theme, not to the whole clause. All forms are 1.00 at depth 0.
Learned gates (mean over seeds): that 1.00, to 1.00, never 1.00, by 0.98, anything 0.95, who 0.88, not 0.87, nothing 0.75, does 0.65, '.' 0.56;
nobody 0.22, it 0.05, verbs ~0. The negation words become emitters like 'that' — an operator opens a fibre — except 'nobody', which the
objective treats as an identity ('someone' under negation lives in the filler, not in the structure).
Reading. (i) "Operator = role" holds for negation on its own at every scope: decoded, separated from the affirmed state by construction
(0.18 vs the lexical bag's 0.72), with the scope right in 100% of cases and no rule for any word. (ii) The unseen COMBINATION operator-
inside-operator is not free: argument-scoped negation composes into reporting (0.8-1.0), sentential negation does not (0.0-0.5, attaching
to the theme). Hypothesis to test next (E1b, running): sentential negation displaces the whole clause and was only ever seen at depth 0, where
no 'that' precedes it; seen once inside a reported clause, it should compose one level further, as reporting itself did (6c). (iii) Gates:
the negation operators join the relation markers; the emergent partition is unchanged in kind.
### 9b-bis. E1b — negation seen inside reporting at depth 1, tested at depth 2 (UNSEEN) (logs/2026-10-15_1736_v13_negation_8e2b30e.txt; 5 seeds)
Same as E1 except the negated forms are sampled at depths 0-1 in training (--neg_depth_train 1).
| axis | E1 (negation seen at depth 0 only) | E1b (seen at depths 0-1) | criterion |
|---|---|---|---|
| negated, depth 0: FULL frame | 1.000 | 0.973±0.048 | > 0.95 MET |
| negation inside reporting, depth 1: scope / FULL | 0.559 / 0.451 (unseen) | 0.979 / 0.979 (seen) | — |
| negation inside reporting, depth 2 (UNSEEN in both): FULL | 0.527±0.060 | **0.972±0.049** | > 0.9 MET |
| pair separation cos(aff, neg) | 0.184 | 0.245 (bag 0.723) | gap > 0.5 MET |
Per form at depth 2 (seed 0): all eight forms, including sentential 'does not V' and 'never V3', scope 1.00 (E1: 0.53 / 0.13). One seed (4) is
at 0.88-0.91 on the negated axes (one form unstable), the other four at 0.98-1.00. Gates: that, to, never, anything 1.00; by 0.94; does 0.77; '.'
0.67; who 0.55; not 0.46 (borderline); nobody, verbs ~0.
Reading: the E1 failure was a DATA limit, not a mechanism limit — exactly the pattern of reporting itself (6c): an operator that displaces a
whole clause composes one level further than trained once it has been seen inside another operator at least once. Rule to carry into E2-E3:
train every operator at depths 0-1, test at 2; report the unseen depth as the composition test.
