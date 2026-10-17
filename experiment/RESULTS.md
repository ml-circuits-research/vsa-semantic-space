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
| axis | E1 (negation seen at depth 0 only; share 0.4) | E1b (seen at depths 0-1; share 0.4, i.e. 0.2 at depth 1) | criterion |
|---|---|---|---|
| negated, depth 0: FULL frame | 1.000 | 0.973±0.048 | > 0.95 MET |
| negation inside reporting, depth 1: scope / FULL | 0.559 / 0.451 (unseen) | 0.979 / 0.979 (seen) | — |
| negation inside reporting, depth 2 (UNSEEN in both): FULL | 0.527±0.060 | **0.972±0.049** | > 0.9 MET |
| pair separation cos(aff, neg) | 0.184 | 0.245 (bag 0.723) | gap > 0.5 MET |
Per form at depth 2 (seed 0): all eight forms, including sentential 'does not V' and 'never V3', scope 1.00 (E1: 0.53 / 0.13). One seed (4) is
at 0.88-0.91 on the negated axes (one form unstable), the other four at 0.98-1.00. Gates: that, to, never, anything 1.00; by 0.94; does 0.77; '.'
0.67; who 0.55; not 0.46 (borderline); nobody, verbs ~0.
Reading: the E1 failure was a DATA limit, not a mechanism limit — exactly the pattern of reporting itself (6c): an operator that displaces a
whole clause is read one level deeper than trained once it has been seen inside another operator at the shallower depth (stated share). Regime to carry into E2-E3:
train every operator at depths 0-1, test at 2; report the unseen depth as the composition test.
Reserve (judgment 2026-10-15, written here on purpose): the rule "an operator combination seen at depth d in training (stated share) is read at depth d+1"
is EMPIRICAL, observed on two operators (reporting, 6c; negation, 9b-bis). It becomes a claim only if it also holds on the next two
(quantifiers, modality/conditionals). If one of them is not read at depth d+1 after being seen at depth d (stated share), that is a result of equal weight and is not
"repaired" with more data without saying so.

## 9c. E2 — QUANTIFIERS as roles, scope ambiguity as a superposition (v14_quantifiers.py; logs/2026-10-15_1815_v14_quantifiers_4ec4ecc.txt; 5 seeds)
World: argument slot R_role (x) (f_noun + R_Q (x) f_det); scope role R_sc (x) f_role for the wide-scope argument; 'each' / 'a certain' force
wide scope; two unmarked quantifiers ('every ... a ...') => the world holds BOTH hypotheses (R_sc (x) (f_r1 + f_r2) / sqrt 2). Training on
UNAMBIGUOUS sentences only (one quantifier, or two with a marker). PROTOCOL ERROR, found by the diagnostic (diag_v14.py) and stated first:
training contained quantifier sentences at depths 0-1 and NO sentence of any kind at depth 2 (E1b had affirmative sentences at 0-2); the chain
parameter 'that:that' exists only at depth >= 2, so at depth 2 EVERYTHING collapses (nouns 0.31, determiners 0.29), not the quantifiers in
particular. The depth-2 column therefore does NOT test the composition rule for quantifiers; E2 is re-run with the E1b protocol (affirmative
0-2, operator 0-1). The reserve of 9b-bis is untouched by this run.
What this run does establish (depths 0-1, seen):
| axis | result (5 seeds) | criterion |
|---|---|---|
| nouns / determiners, depth 0 | 1.000 / 0.973±0.019 | > 0.95 MET |
| SCOPE, unambiguous, depth 0 — all | 0.862±0.026 | > 0.9 NOT met |
| SCOPE, one quantifier (diag, seed 0, depth 0 / 1) | 0.950 / 0.967 | MET |
| SCOPE, two quantifiers with a marker (diag, depth 0 / 1) | 0.811 / 0.744 | NOT met |
| FULL frame, unambiguous, depth 0 / 1 | 0.798±0.074 / 0.724±0.074 | — |
| new nouns (6 unseen per class), depth 0, FULL | 0.800±0.081 | — |
| AMBIGUOUS (never trained): "both hypotheses reported" | 0.458±0.046 (diag seed 0: 0.500) | reported, no criterion |
| ambiguous: cos(S, superposed world state) | 0.858±0.011 | — |
| ambiguous: forced choice = surface-first | 0.322±0.077 | — |
Scope-score profile (diag): true wide role 0.36 vs other roles 0.09 (one quantifier); 0.32 vs 0.08 (two marked); ambiguous: the two
quantified roles 0.22 vs the third 0.03 — the superposition is there, but with two quantifiers the wrong one wins in ~20% of cases. Per
determiner (one quantifier): every 0.95, each 1.00, some 0.87, a 1.00, a certain 1.00. Per pattern: 0.80 ('C VR B from A') to 0.975.
Learned gates: that 1.00, '.' 0.87, to 0.80, and — new — the RECIPIENT verbs (sees, buys, accepts, borrows, receives, gets 0.54-0.63) open,
while the active verbs stay closed; determiners (every, each, some, a, the) closed, 'from' 0.40, 'by' 0.32.
Reading: (i) quantifier = role on the argument holds for nouns and determiners (1.00 / 0.97) and for scope with ONE quantifier (0.95);
(ii) with TWO quantifiers the scope decision is the weak point (0.81), a real result to keep; (iii) "reports both hypotheses / reports margin and residual": on never-seen
ambiguous sentences the state reports both scope hypotheses in 46-50% of cases and sits at 0.86 cosine from the superposed world state —
partial, by composition, with no training on ambiguity; (iv) the depth-2 result is void (protocol), re-run pending.

### 9c-bis. E2 re-run with the E1b protocol (affirmative all-definite sentences at depths 0-2, quantified at 0-1; v14 --p_aff 0.4;
logs/2026-10-16_0833_v14_quantifiers_c5333dc.txt; 5 seeds) + diagnostic (diag_v14.py, seed 0, corrected protocol)
| axis | 9c (no depth-2 sentences) | 9c-bis (corrected protocol) | criterion |
|---|---|---|---|
| nouns / determiners, depth 0 | 1.000 / 0.973 | 0.999 / 0.923±0.009 | nouns MET; determiners just below 0.95 |
| SCOPE, unambiguous, depth 0 | 0.862 | 0.795±0.021 | > 0.9 NOT met |
| FULL frame, unambiguous, depth 0 / 1 | 0.798 / 0.724 | 0.665 / 0.562 | — |
| depth 2 (UNSEEN): SCOPE / FULL | 0.469 / 0.047 (void) | **0.603±0.030 / 0.362±0.031** | > 0.9 NOT met |
| new nouns, depth 0, FULL | 0.800 | 0.657 | — |
| AMBIGUOUS (never trained; n = 360 sentences per seed = 6 patterns x 60): "both hypotheses reported" (main) / cos to the superposed world / forced choice = surface-first ("what a reasoner receives when forced to pick") | 0.458 / 0.858 / 0.322 | 0.425±0.031 / 0.843±0.017 / 0.259±0.048 | reported, observed, partial |
Diagnostic per case x depth (seed 0; nouns | determiners | scope): one quantifier d0 1.00 | 0.81 | 0.89; d1 1.00 | 0.84 | 0.87; d2 (UNSEEN)
**1.00** | 0.64 | 0.52. Two marked: d0 1.00 | 0.77 | 0.62; d2 1.00 | 0.46 | 0.52. Affirmative (all 'the'): nouns and determiners 1.00 at every
depth (scope 0.000 by construction: the decoder had no "no quantifier" hypothesis — an EVALUATION gap, fixed in decode_q with a 'none'
threshold, affects only affirmative sentences, not the table above). Per determiner (one quantifier, d0): a 1.00, a certain 0.98, every 0.93,
each 0.86, some 0.75. Per pattern (d0): determiners 0.70-0.95, scope 0.60-0.88.
Reading, with the reserve of 9b-bis APPLIED:
- The composition regularity does NOT hold for quantifiers (quantified sentences seen at depths 0-1, share 0.6; tested at depth 2): at the unseen depth the NOUNS compose (1.00 — reporting itself
  composes, so the 9c protocol error is gone) while the quantifier-specific terms do not (determiners 0.46-0.64, scope 0.52 against chance
  0.33). This is the first operator for which the rule fails, and it is reported as such; it is not repaired with more data.
- The weakness is already visible at SEEN depths: scope 0.80 (one quantifier 0.89, two 0.62) and determiners 0.92 — the quantifier is only
  approximately represented even where it was trained. Adding the affirmative sentences (needed for depth 2) lowered the seen-depth numbers
  (0.86 -> 0.80, 0.97 -> 0.92): the quantifier terms compete with the base.
- Hypothesis on the cause, proposed for judgment (rule 10 order: evaluation fixed, data corrected, optimization unchanged — this is the
  world/model level, as in round 5-6): the world's scope term is R_sc (x) f_role, whose filler is the identity OF A ROLE, not of any word in
  the sentence. The model can only produce terms address (x) f_word, so R_sc (x) f_A is not representable exactly (the same class of
  mismatch as the flat world of round 5: the world asserts something no causal word term can carry). A representable alternative keeps
  "scope as a role": R_sc (x) f_noun_wide — the wide-scope quantified NOUN's identity, which is in the sentence. This is a correction of my
  world definition, not of the model; it is NOT applied here. The determiner term R_Q (x) f_det inside the slot is representable (f_det is a
  word), and its 0.92 / 0.64 suggests a second, smaller issue: two identities bundled in one slot halve each other's strength.
- "Reports both hypotheses": unchanged — ~0.43 of never-trained ambiguous sentences get both hypotheses, cosine 0.84 to the superposed
  world state; partial, by composition.

### 9c-ter. E2c — quantifier world correction, both targets (v14 --scope_target slot: logs/2026-10-16_1031_v14_quantifiers_7f2b6b7.txt; --scope_target noun: logs/2026-10-16_1052_v14_quantifiers_8fc1fcc.txt; 5 seeds each, same regime as 9c-bis: --p_aff 0.4) + evaluation diagnostic (diag_e2c.py; logs/2026-10-16_1240_diag_e2c_c142b57.txt; seed 0, one model per target)
Rule-11 trace first (trace_e2c.py, logs/trace_e2c.txt): the slot target R_sc (x) (R_role (x) f_noun) is a product of per-category vectors the
model has (the wide determiner's emitted slot, L[each:NP] or bg:each:NP), the role-identity target of 9c-bis is not. Regime: quantified
sentences (one quantifier, or two with a wide-scope marker) seen at depths 0-1 (share 0.6), affirmative at 0-2, tested at depth 2.
| axis (5 seeds) | 9c-bis role-identity target | E2c slot target | E2c noun target | criterion |
|---|---|---|---|---|
| depth 0: nouns / determiners | 0.999 / 0.923 | 0.999 / 0.946±0.010 | 0.997 / 0.944±0.011 | nouns MET; determiners just below 0.95 |
| depth 0: SCOPE / FULL | 0.795 / 0.665 | 0.784±0.038 / 0.667±0.027 | **0.873±0.051** / **0.749±0.037** | scope > 0.9 NOT met |
| depth 1 FULL | 0.562 | 0.503±0.051 | 0.578±0.043 | — |
| depth 2 (UNSEEN): SCOPE / FULL, decoder with the ABSOLUTE 'none' threshold 0.08 | (0.603 / 0.362, decoder without 'none') | 0.027 / 0.008 | 0.029 / 0.011 | > 0.9 NOT met; see the diagnostic |
| new nouns d0 FULL | 0.657 | 0.602 | 0.717 | — |
| ambiguous (never trained; n = 360/seed): both reported / cos to superposed world / forced surface-first | 0.425 / 0.843 / 0.259 | 0.389±0.080 / 0.804 / 0.427 | 0.693±0.029 / 0.819 / 0.331 | reported, observed |
Evaluation diagnostic (rule 10: evaluation before data, optimization, architecture). The depth-2 values 0.03 are BELOW chance (0.33); the
suspected cause was the 'none' (no-quantifier) hypothesis added to decode_q after 9c-bis with an ABSOLUTE threshold (tau_none = 0.08) —
all scope scores shrink with depth, so at depth 2 the decoder says 'none' for every sentence. diag_e2c re-scores one model per target under
three rules — absolute 0.08 (as in E2c), none (as in 9c-bis), relative (none if best scope score < 0.25 x best noun score) — and prints the
raw scope-score profile (mean score of the TRUE wide role | mean of the other roles):
| target | case | d0 abs / none / rel | raw true vs other | d2 (UNSEEN) abs / none / rel | raw true vs other |
|---|---|---|---|---|---|
| role (9c-bis) | one quantifier | 0.894 / 0.894 / 0.894 | 0.281 vs 0.059 | 0.483 / **0.522** / 0.478 | 0.093 vs 0.030 |
| role | two, marked | 0.622 / 0.622 / 0.622 | 0.317 vs 0.129 | 0.517 / 0.517 / 0.517 | 0.143 vs 0.062 |
| slot (E2c) | one quantifier | 0.961 / **0.978** / 0.850 | 0.167 vs 0.034 | 0.006 / **0.333** / 0.033 | 0.024 vs 0.015 |
| slot | two, marked | 0.683 / 0.683 / 0.683 | 0.194 vs 0.074 | 0.033 / 0.250 / 0.083 | 0.034 vs 0.029 |
| noun (E2c) | one quantifier | 0.994 / **0.994** / 0.989 | 0.206 vs 0.034 | 0.028 / **0.206** / 0.067 | 0.035 vs 0.039 |
| noun | two, marked | 0.772 / 0.772 / 0.772 | 0.247 vs 0.100 | 0.039 / 0.194 / 0.106 | 0.046 vs 0.044 |
| all three | affirmative (truth = none) | role 0.36 / 0 / 0.72; slot 0.97 / 0 / 1.00; noun 0.91 / 0 / 0.99 | max score 0.05-0.10 | role 0.99 / 0 / 0.96; slot 1.00 / 0 / 0.99; noun 0.95 / 0 / 0.84 | 0.03-0.05 |
Determiners (all targets): one quantifier d0 0.81-0.86, d2 0.64-0.67; two marked d0 0.77-0.82, d2 0.46-0.48. Nouns 1.000 everywhere.
Reading.
- The evaluation artefact is REAL but it is not the whole story. The absolute threshold does turn depth-2 scope into 0.03 for the corrected
  targets (the no-none rule recovers 0.33 / 0.21); it also makes the affirmative case readable (0.91-1.00 where the 9c-bis decoder had 0 by
  construction). CORRECTED (judgment a8a8fab C1; the first version of this paragraph said the relative rule "affects the affirmative case
  only" — the table above contradicts it): the RELATIVE rule carries the same depth artefact, weaker — slot one-quantifier d0 0.978 -> 0.850,
  d2 0.333 -> 0.033; noun d2 0.206 -> 0.067 — because scope scores shrink with depth while noun scores do not. Therefore NO 'none' rule is used
  from now on: rule 17 is applied as for negation — PRESENCE (quantifier present / absent) is its own axis, thresholded on calibration
  sentences (rule 25), and SCOPE is reported conditional on presence as the argmax, with the one- and two-quantifier cases separate
  (v14 decoder from commit after a8a8fab). Every quantifier number in this file, STATUS.md and PAPER_NOTE carries the rule it was computed
  under: 9c-bis = no-none rule; the E2c table = absolute 0.08; the diagnostic table = all three. The 5-seed E2c models were not saved
  (saving is on from now), so these numbers keep their labels and are not re-scored. Under NO rule does a corrected target compose: at depth 2
  the raw score of the true wide role equals the score of the other roles (slot 0.024 vs 0.015, noun 0.035 vs 0.039) — there is NO scope
  signal at the unseen depth, whereas the role-identity target of 9c-bis keeps a weak one (0.093 vs 0.030 -> 0.52 scope under the no-none
  rule, above chance).
- So the world correction worked where the trace said it could — at the SEEN depth the one-quantifier scope is 0.96-0.99 (role target 0.89),
  the best quantifier number so far — and failed at the unseen depth, worse than the target it replaced. The quantifier terms of the corrected
  worlds are carried by terms that are NOT re-addressed by the reporting frame: the diagnostic shows the determiner gates CLOSED in both
  corrected worlds (open: from, that, to; 9c-bis seed 0 opened the verbs), so, exactly as the trace's second branch predicted, the scope
  difference between 'each' and 'every' is carried by bigram / path terms (bg:each:NP), which are position-local and do not move with the
  depth; the chain term (the one 'that' re-addresses, and the one that makes nouns compose at 1.000) is identical for 'each' and 'every' when
  the determiners are closed. The role-identity target of 9c-bis cannot be produced by any word term, so its weak signal lives in the chain by
  default, which is why it survives one level (0.52) without being right.
- Hypotheses recorded (HYPOTHESIS, rule 22; corrected by the judgment at a8a8fab E1 — the first version here had one hypothesis, "scope is a
  relation between two slots, needs a degree-2 term", which cannot explain the ONE-quantifier failure: which single slot is quantified is a
  degree-1 fact, exactly the slot target):
  H-a (addressing): the scope term composes only if it is carried by the code the reporting frame re-addresses (the chain). In both corrected
      worlds the determiner gates stayed closed, so the each/every difference rode on position-local terms. Covers the one- AND two-quantifier
      failures. Tested by C2 (determiner gates GIVEN open; depth-2 trace in logs/trace_c2.txt: with the flag the wide noun's chain address
      contains the determiner's own category and both 'that' displacements; without it, it is identical for 'each' and 'every').
  H-b (relation): the two-quantifier case additionally needs a term that holds both phrases. Covers the two-quantifier case only; testable
      only after C2, and only if the one-quantifier case then composes while the two-quantifier case still fails.
  This is NOT the same limit as right-announced structure (that one is about causal order; this one is about addressing): two declared limits.
- Changes (judgment a8a8fab E2): C2 = determiner gates GIVEN open (rule 20 stated), 5 seeds, regime of 9c-bis, both targets; criterion scope
  > 0.9 at depth 0 AND at depth 2, nouns 1.00, cost on the 6c / 9b / 9d / 10d tables reported; one- and two-quantifier scope separately.
  The degree-2 term as first proposed here (address (x) f_wide (x) f_narrow) is WITHDRAWN: binding is commutative, so the term is symmetric in
  the two nouns and cannot say which is wide; and with two quantified phrases "A out-scopes B" is already determined by "which slot is wide",
  which the slot target encodes — degree 2 adds no information. Only if C2 composes the one-quantifier case and not the two-quantifier one: a
  relational term with a PERMUTED filler (permutation binding, B7), declared as a change of principle. If C2 fails on both: no third world
  definition without a new argument; the quantifier line stays the named exception with H-a / H-b open.
- Ambiguity: the noun target gives "both hypotheses reported" 0.69 (role 0.43, slot 0.39) with a forced surface-first choice 0.33 (absolute-
  0.08 decoder). Adopted (judgment a8a8fab E3): scope on ambiguous sentences is EMITTED as "both" — the reader's equivalent of the verifier's
  "compatible, incomplete". Criterion, two-sided: both > 0.9 on never-trained ambiguous sentences AND one reading > 0.9 on unambiguous ones,
  with the FALSE-BOTH rate on unambiguous sentences reported as its own number; measured on whichever target C2 ends with. Until met: observed, partial.
- Status of the regularity (rule 24): "seen at depth d in training (share p), tested at depth d+1" holds for six operator cases (reporting,
  negation, modality, conditional, modal-in-conditional, negation on commands); quantifier scope FAILS at the unseen depth in all three world
  definitions tried (depth 2, no-none rule: role 0.52, slot 0.33, noun 0.21; nouns 1.00 in all). 9c-bis stays; E2c supersedes nothing — it
  adds the two corrected targets and the diagnostic. The numbers above are not re-run; the decoder from the next run on has no 'none' rule
  (presence / scope split, see the first bullet).

### 9c-quater. C2 — determiner gates GIVEN open (rule 20), presence / scope split (rule 17), thresholds on calibration rows (rule 25) (v14_quantifiers.py --det_gates open; --scope_target slot: logs/2026-10-16_1557_v14_quantifiers_e0cbbbc.txt; --scope_target noun: logs/2026-10-16_1619_v14_quantifiers_e0cbbbc.txt; 5 seeds each; regime of 9c-bis)
Trace first (logs/trace_c2.txt): with the flag the wide noun's chain address contains the determiner's own category and both 'that'
displacements (that:that + that:each + each:a + L[a:NP] vs ... every ...), so the scope term sits in the re-addressed code — a valid test of H-a.
Decoder: PRESENCE = best scope score >= a per-depth threshold set on calibration sentences (rng 1234+depth; scored rows rng 900+; thresholds
d0 ~0.11-0.15, d1 ~0.08-0.12, d2 ~0.03-0.07, quantified vs affirmative means printed per seed); SCOPE|present = argmax (or both), no 'none' rule;
FULL = nouns, determiners, presence and scope all right. Given: determiners 'every, each, some, a, certain' open (logit +8, gradient masked).
| axis (5 seeds) | slot target, C2 | noun target, C2 | E2c (gates learned; decoder absolute 0.08) slot / noun | criterion |
|---|---|---|---|---|
| depth 0: nouns / determiners | 0.988 / 0.968 | 0.977 / 0.966 | 0.999 / 0.946 ; 0.997 / 0.944 | nouns 1.00: slightly NOT met (cost of the open gates) |
| depth 0: PRESENCE (quantified) / ABSENT (affirmative) | 0.887 / 1.000 | 0.933 / 1.000 | — | — |
| depth 0: SCOPE|gold-quantified (argmax on sentences the gold marks as quantified; not conditioned on the predicted presence), one-quantifier / two-quantifier | **0.976 / 0.761** | **0.961 / 0.793** | (scope with 'none' rule: 0.784 ; 0.873) | one-q > 0.9 MET; two-q NOT met |
| depth 0: FULL (joint) | 0.701 | 0.763 | 0.667 / 0.749 | — |
| depth 1 FULL | 0.616 | 0.698 | 0.503 / 0.578 | — |
| depth 2 (UNSEEN): PRESENCE / ABSENT | 0.806 / 0.922 | 0.757 / 0.941 | — | — |
| depth 2 (UNSEEN): SCOPE|gold-quantified, one-quantifier / two-quantifier | **0.667 / 0.529** | **0.631 / 0.556** | (no-none rule, diag: 0.333 / 0.250 ; 0.206 / 0.194) | > 0.9 NOT met (chance 0.33) |
| depth 2 FULL (presence threshold set on LABELLED depth-2 calibration sentences, held out — the unseen depth contributes a labelled statistic to the decoder; the STRICT column with the depth-1 threshold is in 9c-quater-bis) | 0.386 | 0.368 | 0.008 / 0.011 (absolute 0.08) | NOT met |
| new nouns d0 FULL | 0.623 | 0.734 | 0.602 / 0.717 | — |
| ambiguous (never trained): both reported / cos to superposed world / forced surface-first; presence on ambiguous | 0.377 / 0.725 / 0.397 ; 0.78 | **0.533** / 0.742 / 0.327 ; 0.83 | 0.389 / 0.804 ; 0.693 / 0.819 (absolute 0.08) | both > 0.9 NOT met |
Per seed, one-quantifier scope at depth 2: slot 0.65-0.69, noun 0.58-0.69 (all five seeds above chance); two-quantifier 0.50-0.59.
Learned gates besides the given ones: 'that' 1.00, 'to' 0.60, 'from' 0.40; the recipient verbs fall to 0.25-0.38 (slot) / 0.12-0.20 (noun).
Reading, by the criterion stated before the run.
- H-a is PARTLY supported and the criterion is NOT met. Putting the scope term into the re-addressed code produces a scope signal at the unseen
  depth where there was none (one-quantifier 0.63-0.67 against 0.21-0.33 before, five seeds each), but it is far from composing (criterion
  > 0.9), and presence itself degrades at depth 2 (0.76-0.81). At the seen depth the one-quantifier scope now meets its criterion (0.96-0.98)
  for the first time; the two-quantifier (marked) case does not (0.76-0.79) even at depth 0, where the determiner gate is open and the information
  is degree 1 — so the two-quantifier weakness is not (only) addressing either.
- Costs: nouns 0.98-0.99 (from 1.00: every quantified noun's chain now passes through its determiner), determiners up (0.97), ambiguous "both"
  down on the slot target (0.38) and up on the noun target (0.53; still far from the C3 criterion). The 6c / 9b / 9d / 10d tables are not
  affected (their worlds have no determiners and were not re-run).
- Decision per the judgment ("if C2 fails on both: stop building remedies"): C2 fails on both at the unseen depth. The quantifier line stays
  the named exception with H-a (partly supported: signal, no composition) and H-b (untested: the one-quantifier case does not compose either)
  open; no permuted-filler term is built; no third world definition. The relative numbers (0.65 vs 0.33) are recorded as the measured effect
  of carrying the scope term in the chain. Next experiments: B5-B7 as planned.

### 9c-quater-bis. C2 strict columns (judgment 3b4f837 C2): depth 2 with the DEPTH-1 presence threshold, on the saved C2 models (rescore_v14_c2.py; logs/2026-10-16_1814_rescore_v14_c2_edfd146.txt; 5 seeds, same rows)
| target | depth-2 threshold (labelled depth-2 calibration rows, as in 9c-quater): PRESENCE / FULL / aff ABSENT | STRICT, depth-1 threshold (no depth-2 information): PRESENCE / FULL / aff ABSENT |
|---|---|---|
| slot | 0.806 / 0.386 / 0.922 | **0.257 / 0.150 / 0.998** |
| noun | 0.757 / 0.368 / 0.941 | **0.173 / 0.109 / 1.000** |
The scope scores shrink with depth (thresholds d1 0.08-0.12, d2 0.03-0.07): with the depth-1 threshold most depth-2 quantified sentences read
as 'absent'. The 9c-quater depth-2 FULL numbers (0.37-0.39) therefore DEPEND on the labelled depth-2 statistic; the strict numbers are 0.11-0.15.
SCOPE|gold-quantified is unaffected (it does not use the threshold). Both columns are reported from here on.

### 9c-quinquies. C2b — the determiner as a BEACON-RELATIVE emitter (v8.set_rel_emitters; v14 --rel; --scope_target slot: logs/2026-10-16_1729_v14_quantifiers_8faa02b.txt; --scope_target noun: logs/2026-10-16_1751_v14_quantifiers_edfd146.txt; 5 seeds each; regime of 9c-bis; gates LEARNED, determiners closed)
Mechanism (judgment 3b4f837 E2): a quantified determiner (every, each, some, a, certain — a given set) draws the NEXT noun a second time at
(the noun's own address) (x) exp(i REL[det]) (x) f_noun, one learned displacement per determiner category, relative to the noun's slot; the probe's
cumulative position is untouched. Trace first (logs/trace_c2b.txt): (a) the wide noun's second drawing inherits its own address, which contains
both 'that' displacements, plus REL[det]; (b) every other word is IDENTICAL under each / every; (c) with two quantifiers each REL is added to
its own noun's address. Decoder as in 9c-quater (presence on calibration rows per depth; SCOPE|gold-quantified = argmax; strict columns with the
depth-1 threshold).
| axis (5 seeds) | slot target | noun target | C2 (gates open) slot / noun | criterion |
|---|---|---|---|---|
| depth 0: nouns / determiners | **0.998** / 0.919 | 0.993 / 0.867 | 0.988 / 0.968 ; 0.977 / 0.966 | nouns 1.00: slot within 0.002 |
| depth 0: PRESENCE / aff ABSENT | 0.996 / **1.000** | 0.998 / **1.000** | 0.887 / 1.000 ; 0.933 / 1.000 | — |
| depth 0: SCOPE|gold-q one-quantifier / two-quantifier (marked wide by the determiner's category, not the unmarked ambiguous case) | **0.993 / 0.685** | **0.976 / 0.867** | 0.976 / 0.761 ; 0.961 / 0.793 | one-q > 0.9 MET (both); two-q NOT met |
| depth 0 FULL (joint) / depth 1 FULL | 0.686 / 0.599 | 0.633 / 0.520 | 0.701 / 0.616 ; 0.763 / 0.698 | — |
| **depth 2 (UNSEEN): SCOPE|gold-q one-quantifier** (per seed) | **0.998** (1.000, 0.990, 1.000, 1.000, 1.000) | **0.927** (0.959, 0.929, 0.898, 0.888, 0.959) | 0.667 ; 0.631 | > 0.9 **MET** (slot 5/5 seeds; noun 3/5, mean met) |
| depth 2 (UNSEEN): SCOPE|gold-q two-quantifier (marked wide by the determiner's category) | 0.481 | 0.463 | 0.529 ; 0.556 | NOT met (chance 0.33) |
| depth 2: PRESENCE (d2 threshold) / STRICT (d1 threshold) / aff ABSENT (d2 thr; strict) | **0.978 / 0.867** / 1.000 ; 1.000 | 0.912 / **0.032** / 0.987 ; 1.000 | 0.806 / 0.257 ; 0.757 / 0.173 | — |
| depth 2 FULL (d2 threshold) / STRICT | 0.472 / 0.409 | 0.196 / 0.009 | 0.386 / 0.150 ; 0.368 / 0.109 | — (the two-quantifier half of the sentences fails) |
| new nouns d0 FULL | 0.608 | 0.571 | 0.623 ; 0.734 | — |
| ambiguous (never trained): both reported / cos superposed / forced surface-first | 0.655±0.130 / 0.837 / 0.622 | **0.838**±0.062 / 0.829 / 0.398 | 0.377 ; 0.533 | both > 0.9 NOT met (noun target close) |
Gates (learned): that 1.00, to 0.80 / 0.60, from 0.60, by 0.56-0.59, '.' 0.55-0.64; determiners stay closed (0.1-0.2); recipient verbs 0.2-0.4.
Reading, by the criterion and decision rule stated before the run.
- ONE quantifier: the scope term now moves with the frame WITHOUT shifting the rest of the sentence — nouns 0.998 (C2: 0.988), affirmative
  sentences read as absent 1.000 at both depths, and the one-quantifier scope at the UNSEEN depth is 0.998 on the slot target (five seeds at
  0.99-1.00) and 0.927 on the noun target. The one-quantifier case therefore JOINS the regularity "seen at depth d (share 0.6 over depths 0-1),
  tested at depth d+1": the third effect size in the judgment's series is no signal 0.21-0.33 -> chain-carried 0.63-0.67 -> beacon-relative
  0.93-1.00. H-a is supported in the form the judgment restated it: the scope term composes when it is carried by the re-addressed code AND the
  probe is untouched. On the slot target even the strict presence (depth-1 threshold) holds at 0.87; on the noun target it does not (0.03: the
  noun-target scope scores shrink faster with depth), so the slot target is the world definition kept.
- TWO quantifiers (marked): NOT met at any depth (0.69 / 0.87 at depth 0, 0.46-0.48 at depth 2). This is the configuration the a8a8fab
  judgment named: the one-quantifier case composes and the two-quantifier case does not. The trace said why: with two emitters, 'every' must emit
  the scope term in a one-quantifier sentence and must NOT next to 'each' — one displacement per category cannot do both; which phrase is wide
  is a relation between the two determiners. H-b is now TESTABLE and the judgment's conditional applies: a relational term with a PERMUTED filler
  (permutation binding, B7), emitted by the wide determiner's category through a learned displacement, declared as a change of principle, with
  the criterion of C1 plus "inert when no quantifier". It is NOT built here; proposed for judgment (current_status.md E).
- Determiners dropped (0.92 / 0.87 from 0.97): the second drawing competes with the determiner term inside the same slot — a cost to report.
- Status of the regularity (rule 24): six operator cases + the one-quantifier scope hold; the two-quantifier (marked) scope is now the named
  exception, with H-b as its stated hypothesis.

## 9d. E3 — MODALITY and CONDITIONALS as roles (v15_modality.py; logs/2026-10-16_0902_v15_modality_b2b8f34.txt; 5 seeds, D = 1024, 40k)
World: R_may (x) S, R_must (x) S; R_if (x) S_P + R_then (x) S_Q; reporting as before. Forms: "may V", "perhaps V3"; "must V", "has to V";
"if P , then Q", "if P , Q", and "Q if P" (RIGHT-announced: included on purpose). Training: affirmative 0-2, modals and conditionals 0-1; the
MIXED "if P then Q may ..." NEVER trained. Model unchanged (v8 bundle, learned gates). Decoding over six operator hypotheses (rule 17).
| axis | result (5 seeds) | criterion |
|---|---|---|
| affirmative d0 FULL | 1.000 | — |
| modal d0 / d1 / **d2 UNSEEN** FULL | 1.000 / 1.000 / **1.000** | > 0.95; unseen > 0.9: MET |
| conditional 'if P , then Q' d0 / 'if P , Q' d0 | 1.000 / 1.000 | MET |
| conditional d1 / **d2 UNSEEN** FULL | 1.000 / **1.000** | MET |
| conditional **'Q if P' (right-announced)** d0 FULL | **0.203±0.088** | the declared limit (RESULTS 6c), confirmed |
| separation cos(P, may P) / (P, must P) / (may P, must P) | **0.119 / 0.094 / 0.132** ; lexical bag 0.772 / 0.773 / 0.857 | gaps > 0.5: MET |
| MIXED "if P then Q may" — ZERO occurrences in training — d0: operator+scope / FULL ; d1 FULL | 0.035 / 0.000 ; 0.000 | zero occurrences in training: not read, as E1's sentential forms |
Learned gates: that, if, ',', has, perhaps, may, must 1.00; to 0.80; '.' 0.24; **then 0.17** (closed: 'if' and the comma carry the
structure); verbs ~0.12.
Reading. (i) Modality and conditionals are operators-as-roles that compose one level further than trained: 1.000 at the unseen depth, five
seeds, no rule — the third and fourth operators for which the empirical rule holds (reporting, negation, modality, conditional); quantifiers
remain the one failure (9c-bis), with the world-side hypothesis pending judgment. (ii) The right-announced form fails exactly as predicted
(0.20): the consequent is already written when 'if' arrives; this is the limit of causal addressing and is reported as such. (iii) A
combination with ZERO exposures (a modal inside a consequent, never seen in any form) does not compose — the same finding as E1's sentential
negation inside reporting; the regularity is "seen at depth d (stated share), read at d+1", and E3b gives this pair a share of 0.15 at depth 0 and tests it
inside reporting. (iv) Two modals that the world distinguishes (may vs must) are separated by the state (0.13) although the lexical bag
cannot (0.86): the modal word's displacement, not its identity, carries the distinction.
### 9d-bis. E3b — the mixed construction seen at depth 0 in training (share 0.15 of the training sentences, --p_mixed 0.15), tested inside reporting (depth 1) (logs/2026-10-16_0925_v15_modality_f31f4e5.txt; 5 seeds)
| axis | E3 (zero occurrences in training) | E3b (seen at depth 0, share 0.15) |
|---|---|---|
| MIXED "if P then Q may" d0: operator+scope / FULL | 0.035 / 0.000 | **1.000 / 1.000** (seen) |
| MIXED inside reporting, d1 (UNSEEN combination) FULL | 0.000 | **1.000±0.000** |
| everything else (modals, conditionals, unseen depth 2) | 1.000 | 1.000 (unchanged) |
| 'Q if P' (right-announced) | 0.203 | 0.440±0.121 (still at the limit; high variance) |
| separation (P, may P) / (P, must P) / (may, must) | 0.12 / 0.09 / 0.13 | 0.15 / 0.13 / 0.17 (bag 0.77 / 0.77 / 0.86) |
Gates: that, ',', has, may, must, perhaps 1.00; to 0.69; if 0.60; then closed. Same pattern as E1 -> E1b: zero occurrences in training of a combination
fail; seen at depth 0 (share 0.15), the combination is read at depth 1. The empirical regularity now holds on reporting, negation, modality, conditionals and the
modal-in-conditional pair; quantifiers remain the single exception (9c-bis, world-side hypothesis pending).

## 10a. Round 10 A2 — the memory curve of one short reference training (trainer.py; logs/2026-10-16_1115_v16_reference_8fc1fcc.txt; 1 seed, D = 1024, 30k sentences, batch 8)
Memory is now logged every 500 steps in every operator script (process RSS, torch allocated / reserved, MemAvailable), losses are read with
.item(), no graph-carrying tensor survives a step, the loss is asserted finite, checkpoints every 2000 steps with --resume (tested on CPU).
| step | sentences | RSS (GB) | torch allocated / reserved (GB) | MemAvailable (GB) |
|---|---|---|---|---|
| start | 0 | 1.32 | 0.50 / 0.53 | 113.8 |
| 500 | 4,000 | 2.46 | 0.64 / 0.96 | 112.2 |
| 1,000-3,500 | 8,000-28,000 | 2.46 (constant) | 0.64 / 0.96 (constant) | 112.2-112.3 |
| end | 30,000 | 2.46 | 0.64 / 0.69 | 111.9 |
Reading: the curve is flat after the first 500 steps (the first-step jump is the CUDA context and the encoded batch tensors); no growth per
step. The same flat profile holds in the 5-seed run (RSS 2.6 GB at seed 4) and in the WikiSQL verifier (RSS 2.06 GB constant across three
epochs, reserved 3.5-4.2 GB during an epoch, released between epochs; log 2026-10-16_1130). The 2026-10-15 hang is therefore not
reproduced by these scripts; the protection (guard.py cap, watchdog, run.sh lock and timeout) stays in place regardless.

## 10b. B1 + B2 — predicate role, occupancy marker, calibration (v16_reference.py; logs/2026-10-16_1118_v16_reference_c142b57.txt; 5 seeds, D = 1024, 30k)
World: S = R_pred (x) f_lemma + sum_r R_r (x) (f_filler + m), with m a fixed occupancy marker shared by every slot; reporting R_X (x) (f_X + m)
+ R_rep (x) S. Predicate filler = the SURFACE verb lemma (option (a): give / receive / sell / buy ... are 12 distinct identities); the event
type (transfer / sale / shipment / loan / display / proposal) is read at READOUT through a per-world synonym table, not learned. Recipient-less
forms ("A sells B .", "B is sold by A .", "it is A who sells B .") with share 0.2 give the "not given" occupancy case. Model unchanged (v8
bundle, learned gates). Calibration on every readout: margin = top1 - top2 cosine; tau = 0.05 fixed BEFORE the run; confident-wrong = wrong
with margin >= tau; abstention split into right/wrong; accuracy-vs-coverage at thresholds 0, 0.02, 0.05, 0.10, 0.15.
| axis (5 seeds) | result | criterion |
|---|---|---|
| seen formulations FULL (A, B, C + predicate) | 1.000±0.000 | > 0.95 MET |
| predicate, surface lemma / event type (via table) | 1.000 / 1.000 | > 0.95 MET (23) |
| new identities FULL / role cues FULL | 1.000 / 1.000 | MET |
| new formulations, UNSEEN transitions FULL | 0.638±0.015 | — (as before: 0.6-0.7 in rounds 3-6; no loss from the predicate role) |
| depth 2 (seen) FULL / depth 3 UNSEEN FULL | 1.000 / 0.813±0.030 | depth 3 > 0.9 NOT met with the predicate role (6c without it: 1.000) |
| occupancy: known filler / UNKNOWN filler / NOT GIVEN | 1.000 / 1.000 / 1.000 | three cases > 0.95 MET (A1: by marker, not by norm) |
| presence cosine on the marker: known 0.38, unknown 0.38, absent 0.06 (threshold 0.22 = midpoint, per seed; set on the SAME 360 scored readouts — rule 25 violation, re-run with the threshold set on training rows in 10b-bis) | separation 0.32 | — |
Calibration (seed 0, tau = 0.05; the same shape on the other seeds): seen roles and predicate: accuracy 1.000, margin mean 0.22-0.33, p10
0.13-0.29, coverage 1.00, confident-wrong 0.000. New identities: 1.000, confident-wrong 0.000. Depth 3 (unseen): accuracy 0.90 (A) /
0.875 (C) at full coverage, margins mean 0.06 / 0.04 — at tau 0.05 coverage drops to 0.59 / 0.33 and the covered accuracy is 1.000 / 0.975,
confident-wrong 0.000 / 0.008; abstention right/wrong 0.31/0.10 and 0.55/0.12. Unseen transitions: accuracy 0.87 (A) / 0.75 (C), at tau 0.05
coverage 0.69 / 0.67 with covered accuracy 1.000 / 0.938, confident-wrong 0.000 / 0.042. B2 criterion (confident-wrong < 2% at coverage >
80%): MET on every seen axis and on new identities; at depth 3 and on unseen transitions the model reaches < 1% confident-wrong (A) only by
abstaining on 30-40% — the margin is informative (wrong answers have small margins), the coverage criterion is NOT met there; reported, not tuned.
Learned gates: 'that' and '.' 1.00; the recipient-less verbs (accepts, gets, sees, buys, receives, borrows) 0.55-0.62 (open: they must emit
the next slot when no 'to' follows); object marker and participles closed (0.17-0.20).
Reading. (i) The predicate role costs nothing on the seen axes and gives 12 distinguishable events where GIVE and SELL used to share a target;
the event type is a readout convention (option (a)), stated as such. (ii) Occupancy is read from the marker (0.38 vs 0.06), not from the
norm — the A1 correction holds at 1.000 in all three cases. (iii) The one regression: depth 3 unseen is 0.81 instead of 1.000 (6c). The
predicate term R_pred (x) f_lemma at depth k is addressed by R_rep^k (x) R_pred; the verb's displacement must now land on a role address of
its own, one more term per level that competes with the two or three role terms — the reduction is on A and C (0.90 / 0.875 at full
coverage), the predicate itself stays correct. Under rule 10 this is first an evaluation of the world (one more term per level), not of the
model; the margin reports it (abstention 0.3-0.5 at depth 3) rather than hiding it. The 6c table (1.000 at depth 3) stands for the
predicate-less world; this one is the reference table from here on.

## 10c. B4 — verifier with injected errors on WikiSQL (v17_verifier.py; logs/2026-10-16_1130_v17_verifier_c142b57.txt; seed 0; D = 512; 3 epochs on the 56,355 training questions; dev 8,421)
Controlled version: the "formalization to verify" is the GOLD frame with ONE injected error (drop a condition column, swap the selected and a
condition column, change the selected column, flip the aggregator, add a condition column, change a value); the LLM-produced formalizations
of the judgment are PLANNED (no API). Reader = the v11 full model (question only, no table) decoding selected column, aggregator and the
condition-column set (presence threshold 0.101 = midpoint of present 0.158 / absent 0.043, fixed before scoring); 'unverifiable' = the
reader's margin on the disputed slot below tau = 0.05; otherwise 'detected' (reader disagrees with the injected frame) or 'missed'
(reader agrees with the error, with margin >= tau = confident-wrong). Lexical baseline (BoW + LR for the selected column and the aggregator
on the same rows; no condition model) scored identically.
| error type | n | reader: detected / unverifiable / missed | lexical: detected / unverifiable / missed |
|---|---|---|---|
| change_sel (selected column) | 8,393 | **0.764** / 0.232 / **0.004** | 0.400 / 0.585 / 0.015 |
| swap_sel_cond | 8,357 | **0.740** / 0.257 / **0.003** | 0.322 / 0.386 / 0.292 |
| flip_agg | 8,421 | **0.805** / 0.183 / 0.013 | 0.623 / 0.362 / 0.014 |
| drop_cond | 8,357 | 0.159 / 0.690 / 0.151 | 0.214 / 0.361 / 0.425 |
| add_cond | 8,393 | 0.162 / 0.837 / 0.001 | 0.214 / 0.358 / 0.428 |
| change_value | 8,357 | 0.159 / 0.532 / 0.310 | 0.214 / 0.361 / 0.425 |
| no error (gold) | 8,421 | FALSE ALARM **0.162** / unverifiable 0.529 / confirmed 0.309 | 0.213 / 0.359 / 0.427 |
Precision of the 'error' verdict on ROLE errors (change_sel, swap, flip_agg) against the gold false-alarm rate: 0.942; recall on role errors
0.527 (the rest is 'unverifiable', not 'missed'). Criterion (judgment B4: detection > 0.9 on role errors): NOT met — 0.74-0.81 detected,
0.18-0.26 unverifiable, confident-wrong 0.3-1.3%. On GOLD frames the verifier confirms 0.309 and says 'unverifiable' on 0.529: it is a
FILTER on the role part of a formalization, not a judge of it — half of the correct formalizations get no verdict (judgment a8a8fab E5).
Caveat (rule 25, judgment C2): the condition presence threshold of this run was the midpoint over the SAME dev rows that are scored; the
re-run sets it on the even half of dev and reports the odd half (10c-bis, below, old and new side by side).
Stated limits. change_value is INVISIBLE to this reader (it decodes columns and the aggregator, not values): 0.159 "detected" there is the
false-alarm rate, as on gold. Condition errors (drop / add) are read only through the column SET at a presence threshold, and the condition
model is weak (0.16 detected, 0.5-0.8 unverifiable): the condition slot was already the weak part of the full-query number (8c, 0.37).
Reading. The reader catches the errors in the roles it decodes well (which column, which aggregator, which column is the selected one) with
a confident-wrong rate below 1.3% and a false-alarm rate of 16% on gold; the lexical baseline on the same rows either misses them with
confidence (swap: 0.29 missed) or cannot decide (change_sel: 0.59 unverifiable). What the model does not decode (values) it cannot verify,
and it says so through the margin rather than through a wrong verdict — in B4's terms, the verifier is useful on the role part of a
formalization and silent on the lexical part. The 16% false alarm on gold (the reader's own wrong decodings with margin >= tau) is the number
to reduce before the LLM version.

## 10d. B3 — negation on COMMANDS: scope as a role (v18_commands.py; logs/2026-10-16_1255_v18_commands_1540118.txt; 5 seeds, D = 1024, 40k, batch 8)
World: command = R_act (x) f_verb + R_pat (x) f_noun [+ R_attr (x) f_adj] [+ R_dest (x) f_place]; negation = a role around its scope: prohibition
R_neg (x) S (the whole command), negated attribute R_pat (x) (f_noun + R_neg (x) f_adj) (only the attribute), or both. Seven templated forms:
affirmative, with destination, "do not V the N", "never V the N", "V the un-ADJ N" (affixal), "V the N that is not ADJ" (relative), and
prohibition + negated attribute. Training: all forms at depth 0 and inside one report ("the admin says that ...", share 0.3 at depth 1); test
on fresh events at depth 0, 3 UNSEEN patients, and depth 2 (unseen). Decoding over four scope hypotheses (none / S / attr / S+attr), scope
reported separately from presence (rule 17); scope-only pairs = "do not delete the saved files" vs "delete the unsaved files" (same words
except the negation site), 240 readouts per seed at tau = 0.05; lexical bag on the same pairs. LLM paraphrases: PLANNED (no API).
| axis (5 seeds) | result | criterion |
|---|---|---|
| depth 0: action / patient / attribute | 1.000 / 1.000 / 0.999 | — |
| depth 0: PRESENCE of negation / SCOPE (4 hypotheses) / FULL | 1.000 / **1.000** / 0.999 | scope > 0.95 MET (templates) |
| scope per form (d0): aff, aff_dest, prohib, never, affixal, relative, prohib+attr | 1.00 on all seven | — |
| new patients FULL / depth 1 FULL | 1.000 / 0.999 | — |
| depth 2 (UNSEEN; negation forms seen at depths 0-1, share 0.3 at depth 1): SCOPE / FULL | **0.997±0.003** / 0.854±0.031 | scope: regularity holds; FULL < 0.9 (the loss is on the fillers at depth 2, not on the scope) |
| scope-only pairs: confident-wrong / coverage at tau 0.05 | **0.000±0.000** / 0.968±0.049 | confident-wrong < 1% at coverage > 90%: MET |
| separation cos(aff, prohibition) / cos(prohibition, attr-negated); lexical bag | 0.058 / 0.174 ; bag 0.845 / 0.676 | gaps > 0.5 MET |
Learned gates (5 seeds): 'never', 'the', DEST open in 5/5; 'do', 'that' in 4/5; the negated-attribute category (NATTR, "un-ADJ") in 3/5;
'not' in 1/5 (its scope is carried by 'do' / 'never' and the attribute category, the same as E1's partition).
Reading. (i) On templated commands the three readings of a negation (forbid the action, negate the attribute, both) are decoded at 1.000 with
scope separate from presence, and the two sentences that differ only in the negation site are never confused with confidence (0.000 at
coverage 0.97). (ii) The regularity "seen at depth d (share p), tested at d+1" holds for the scope (0.997) — the sixth operator case; the
FULL frame at depth 2 is 0.85 because the fillers (patient, attribute) lose at the unseen depth, not the scope. (iii) The lexical bag puts
"delete the saved files" and "do not delete the saved files" at cos 0.85: the separation (0.06) is structural, as in E1. (iv) Everything here
is templated; whether the reading survives paraphrase (the B3 criterion as written) is the LLM-paraphrase follow-up, PLANNED.

## 10b-bis. Reference table re-run with the occupancy threshold set on TRAINING rows (rule 25) + depth-3 breakdown, E4 (a) (v16_reference.py; logs/2026-10-16_1351_v16_reference_e0cbbbc.txt; 5 seeds, same seeds and regime as 10b)
Threshold: midpoint of the marker cosine on 120 depth-0 TRAINING sentences with C and 120 without (per seed: 0.21-0.23; the same-rows midpoint
of 10b would have been the same value to two decimals). Every column of 10b is reproduced to three decimals (seen 1.000, predicate 1.000 /
1.000, new ids 1.000, cues 1.000, unseen transitions 0.638±0.015, depth 2 1.000, depth 3 UNSEEN 0.813±0.030, occupancy 1.000 / 1.000 / 1.000;
calibration of seed 0 identical). 10b's numbers stand; the threshold is now set where rule 25 requires.
E4 (a) — where the depth-3 loss is (all 5 seeds, 100 sentences per template; roles A | B | C | predicate):
| template (depth 3, inside three reports) | A | B | C | pred |
|---|---|---|---|---|
| T0 "A V B to C ." | 0.990 | 1.000 | 0.950 | 1.000 |
| T1 "C VR B from A ." | 0.940 | 1.000 | 0.920 | 1.000 |
| T6 "it is A who V B to C ." | 1.000 | 1.000 | 0.990 | 1.000 |
| T12 "what A V to C is B ." | 0.990 | 1.000 | 0.960 | 1.000 |
| T15 "A V B , to C ." | 1.000 | 1.000 | 0.910 | 1.000 |
| T9 "to C , A V B ." | 0.990 | 1.000 | **0.770** | 1.000 |
| T3 "B is VP by A to C ." (passive) | 0.840 | 1.000 | **0.760** | 1.000 |
| T2 "B is VP to C by A ." (passive) | **0.810** | 1.000 | **0.520** | 1.000 |
Per verb: the recipient-less verbs whose gates sit at the midpoint (accepts, gets, sees, buys, receives, borrows; 0.42-0.81) are at 1.000 on A
and C at depth 3 in almost every seed; the PARTICIPLES (given, lent, offered, sold, shown; gates CLOSED, 0.00-0.33) carry the C loss
(0.29-0.86); 'gives' / 'lends' (active, gate 0.00) have C 0.85 / 0.91.
Reading. B and the predicate are 1.000 everywhere; the loss is on C (and A) in the two PASSIVE forms and in the C-fronted comma form — the
forms where the recipient is not introduced by the verb that precedes it. The judgment's first suspect (half-open verb gates) is NOT
supported: those verbs are at 1.000; the failing verbs are participles with closed gates. The predicate term is addressed by the same prefix
as the slots at every level and costs cosine exactly where the margin was already smallest (6c without the predicate: 1.000). Decision rule
of E4 applied after (b)-(d) below; whatever they give, the depth-3 number is reported with this breakdown and the acc-vs-coverage curve.

## 10c-bis. Verifier re-run with the condition presence threshold set on the calibration half of dev (rule 25) (v17_verifier.py; logs/2026-10-16_1403_v17_verifier_e0cbbbc.txt; same seed, D, epochs as 10c; threshold on the EVEN rows, verdicts on the ODD rows, 4,210 questions)
Threshold 0.101 (midpoint of present 0.159 / absent 0.044 on the even half; the same-rows midpoint of 10c would also be 0.101).
| error type | n (odd half) | reader 10c-bis: detected / unverifiable / confident-wrong | reader 10c (whole dev, same-rows threshold) | lexical 10c-bis |
|---|---|---|---|---|
| change_sel | 4,195 | 0.759 / 0.238 / 0.004 | 0.764 / 0.232 / 0.004 | 0.405 / 0.582 / 0.012 |
| swap_sel_cond | 4,177 | 0.733 / 0.263 / 0.004 | 0.740 / 0.257 / 0.003 | 0.317 / 0.376 / 0.306 |
| flip_agg | 4,210 | 0.795 / 0.195 / 0.010 | 0.805 / 0.183 / 0.013 | 0.628 / 0.359 / 0.013 |
| drop_cond | 4,177 | 0.158 / 0.688 / 0.153 | 0.159 / 0.690 / 0.151 | 0.216 / 0.360 / 0.424 |
| add_cond | 4,195 | 0.162 / 0.836 / 0.001 | 0.162 / 0.837 / 0.001 | 0.216 / 0.356 / 0.428 |
| change_value (invisible) | 4,177 | 0.158 / 0.543 / 0.298 | 0.159 / 0.532 / 0.310 | 0.216 / 0.360 / 0.424 |
| gold: false alarm / unverifiable / confirmed | 4,210 | **0.162 / 0.541 / 0.297** | 0.162 / 0.529 / 0.309 | 0.215 / 0.358 / 0.427 |
Precision on role errors 0.942, recall 0.522 (10c: 0.942 / 0.527). Every number moves by at most 0.013: the threshold was never the issue, the
reader's own 16% confident-wrong on gold is. On gold the verifier confirms 0.30 and abstains on 0.54 — a filter on the role part, not a judge.
This is the 10c table from now on; the reader and the dev frames are saved for C4 (re-scoring with tau set on the even half, curve on the odd half).

## 10b-ter. E4 (b)-(d): is the depth-3 regression gating or capacity? (v16_reference.py; logs/2026-10-16_1511 (--p_omit 0), _1523 (--verb_gates closed), _1535 (--D 2048), all _v16_reference_e0cbbbc.txt; 5 seeds each, same seeds as 10b)
Decision rule stated before the runs (judgment a8a8fab E4): gating if (b) or (c) restores depth 3 > 0.9 with the predicate role; capacity if
only (d) does.
| setting | seen FULL | unseen trans. | depth 2 | **depth 3 UNSEEN** | T2 passive A / C | T3 passive A / C | T9 fronted C | T1 "C receives B from A" A / C |
|---|---|---|---|---|---|---|---|---|
| 10b / 10b-bis (reference: recipient-less share 0.2, gates learned, D 1024) | 1.000 | 0.638 | 1.000 | **0.813±0.030** | 0.81 / 0.52 | 0.84 / 0.76 | 0.77 | 0.94 / 0.92 |
| (b) recipient-less share 0 (the occupancy columns are void here: no 'not given' case exists) | 1.000 | 0.670 | 1.000 | **0.858±0.081** | 0.86 / 0.75 | 0.78 / 0.89 | 0.87 | 0.78 / 0.87 |
| (c) verb gates GIVEN closed (rule 20), recipient-less kept | 0.986 | 0.597 | 0.968 | **0.745±0.042** | 0.86 / 0.56 | 0.79 / 0.81 | 0.81 | 0.34 / 0.61 |
| (d) D = 2048 (capacity test) | 1.000 | 0.650 | 0.995 | **0.852±0.060** | 0.85 / 0.60 | 0.80 / 0.91 | 0.86 | 0.90 / 0.90 |
Gates: (b) the recipient-less verbs fall to the midpoint (0.41-0.50) and 'to' rises to 0.40; (c) everything closed except 'that' and '.'; (d) as 10b.
Reading, by the decision rule. NEITHER branch fires: no setting restores depth 3 > 0.9. (c) is WORSE (0.745) and costs the seen axes
(0.986, depth 2 0.968): closing the verbs removes the gate that the recipient-less forms need ("C receives B from A", T1, drops to A 0.34),
so the half-open verb gates were doing work, not adding noise — the judgment's first suspect is excluded twice (10b-bis breakdown, and (c)).
(b) and (d) move depth 3 to 0.85-0.86 with large seed variance (±0.06-0.08) and leave the passive T2 at C 0.60-0.75: doubling D does not
repair it, so it is not capacity; removing the recipient-less forms does not repair it, so it is not the extra occupancy case. What all four
settings share is the loss on the PASSIVE forms (and the C-fronted comma form) at depth 3, with the predicate term present — the forms where
C follows a participle whose gate is closed, so C's address is reached through 'to' only, and three 'that' displacements plus the predicate
term at the same prefix leave C with the smallest margin (10b: d3:C margin mean 0.04). Conclusion for the reference table: the depth-3 number
stays 0.813 (full coverage) / 1.000 at coverage 0.59 (A) and 0.975 at 0.33 (C), stated with this breakdown; the regression is a FORM effect
(passive + predicate term at depth), neither gating nor capacity, and it is reported as a property of the predicate-role world. A remedy would
be a change of world or model (e.g. the participle as an emitter — rule 20, given) and is NOT built here; proposed for judgment.

## 10c-ter. C4 — abstention threshold set on the calibration half, curve on the report half (rescore_v17_c4.py; logs/2026-10-16_1641_rescore_v17_c4_ef75946.txt; saved frames of 10c-bis, no retraining)
Rule: the smallest tau on a grid whose false-alarm rate on GOLD frames is < 0.05 on the EVEN half of dev; everything below is on the ODD half.
| tau | gold: false alarm / unverifiable / confirmed | role errors: detected / unverifiable / confident-wrong | condition errors: detected / unverifiable / confident-wrong | role detection lost vs tau 0.05 |
|---|---|---|---|---|
| 0.00 | 0.703 / 0.000 / 0.297 | 0.993 / 0.000 / 0.007 | 0.921 / 0.000 / 0.079 | -0.231 |
| 0.05 (10c) | 0.162 / 0.541 / 0.297 | 0.762 / 0.231 / 0.007 | 0.160 / 0.761 / 0.079 | 0 |
| 0.10 | 0.109 / 0.594 / 0.297 | 0.642 / 0.351 / 0.007 | 0.107 / 0.814 / 0.079 | +0.120 |
| 0.15 | 0.072 / 0.630 / 0.297 | 0.520 / 0.473 / 0.007 | 0.071 / 0.850 / 0.079 | +0.242 |
| 0.20 | 0.049 / 0.654 / 0.297 | 0.419 / 0.574 / 0.007 | 0.048 / 0.873 / 0.079 | +0.343 |
| **0.25 (chosen on the even half)** | **0.033 / 0.669 / 0.297** | **0.323 / 0.670 / 0.007** | 0.033 / 0.889 / 0.079 | **+0.439** |
| 0.30 | 0.019 / 0.683 / 0.297 | 0.235 / 0.757 / 0.007 | 0.019 / 0.902 / 0.079 | +0.527 |
At the chosen tau, per type: flip_agg 0.558 detected, change_sel 0.205, swap 0.206, drop_cond 0.032, add_cond 0.033; confident-wrong 0.004-0.014
on role errors, 0.156 on drop_cond (the missing condition is read as absent with confidence), 0.298 on change_value (invisible).
Reading. The trade-off is steep: bringing the false alarm on gold under 5% costs 44 points of role detection (0.76 -> 0.32), and the confirmed
rate on gold stays 0.297 at every tau — it is the fraction of gold frames the reader decodes exactly, and no threshold changes it. The reader's
margin separates right from wrong only partly on gold (false alarm at tau 0.05 = 0.16 of confident disagreements with gold): the 16% are
confident wrong decodings of the selected column or the aggregator, not noise the margin can filter. This fixes what the next step must be:
the reader's confident-wrong rate on sel / agg (a reader property, 8c-8d), not the verifier's threshold. C5 (condition count from an occupancy
marker, one retraining) stays PLANNED and is independent of this.

## 10c-quater. E3 — where the verifier's false alarm comes from (rescore_v17_e3.py; logs/2026-10-16_1715_rescore_v17_e3_8faa02b.txt; saved frames of 10c-bis, odd half of dev, tau 0.05; re-scoring only)
Selected column on the odd half: accuracy 0.814, confident-wrong 0.053, abstention 0.243; aggregator: accuracy 0.824, confident-wrong 0.137.
| breakdown of the confident-wrong SELECTED-COLUMN decodings | n | sel accuracy | confident-wrong rate | share of all confident-wrong |
|---|---|---|---|---|
| tables with 1-5 columns | 1,657 | 0.855 | 0.048 | 0.36 |
| 6-8 columns | 2,114 | 0.795 | 0.058 | 0.55 |
| 9-12 columns | 409 | 0.756 | 0.049 | 0.09 |
| 13+ columns | 30 | 0.700 | 0.033 | 0.00 |
| gold column's name appears verbatim in the question | 3,328 | **0.910** | **0.023** | 0.34 |
| gold column's name NOT in the question | 882 | **0.449** | **0.169** | **0.67** |
Reading. The confident-wrong rate is flat across table sizes (0.03-0.06) and seven times higher when the gold column's name is not in the
question (0.169 vs 0.023): two thirds of the confident-wrong selected columns come from the 21% of questions whose answer column is named by
other words — a LEXICAL-MISMATCH effect, not a hard-table effect. On those questions the reader is at 0.45 and still confident; the margin
does not know that the column was never named. The aggregator's confident-wrong (0.137) is the larger contributor to the 16% false alarm on
gold and was already the lexical decision the model loses (8d). The 50-case audit sample (seed 0) is in logs/audit_confident_wrong_sel_50.jsonl
with an empty audit_verdict per row, for the team: the share that is gold noise / ambiguous questions is the number needed before touching the reader.

## 10b-quater. E1 — the depth-3 passives: rule-11 trace, then the participle as an emitter (trace_e1.py, logs/trace_e1.txt; v16_reference.py --part_gates open; logs/2026-10-16_1717_v16_reference_8faa02b.txt; 5 seeds, same seeds as 10b)
Trace (i), with the learned gate set of 10b (that, '.', recipient verbs open; 'to' and the participles CLOSED), inside three reports:
| form at depth 3 | A | C | predicate carrier |
|---|---|---|---|
| T0 "A gives B to C ." | that:that + that:that + that:. + **.:P** | that:that + that:that + that:. + **L[.:P]** | ... + L[.:gives] |
| T2 "B is given to C by A ." | ... + **L[.:P]** | ... + **.:P** | ... + L[.:given] |
| T3 "B is given by A to C ." | ... + .:P | ... + L[.:P] | ... + L[.:given] |
With 'to' and the participle closed, the only emitter inside the clause is the period: both persons hang off '.' through the two keys the
category P has there (.:P and L[.:P]), and the PASSIVE T2 gives them the SAME two addresses as T0 WITH THE ROLES SWAPPED. The chain code is
therefore form-blind on T2 (it says "first person = agent"); the roles can only be told apart by the path / bigram codes, which do not move with
the depth. C and the predicate share only the 'that' prefix (2 of 3 keys) in every form — the predicate crosstalk blamed in 10b-ter is not what
the trace shows; the trace shows an A/C address collision specific to the passive. The emitter test is therefore the right one: with the
participle open, T2 gives C <- ... + that:given + given:P and A <- ... + given:. + L[.:P] — distinct keys, and C now hangs off the participle.
Run (ii), participle gates GIVEN open (given, sold, sent, lent, shown, offered; rule 20):
| axis (5 seeds) | 10b (participles learned, closed) | E1 (ii) participles given open | criterion |
|---|---|---|---|
| seen / predicate / new ids / cues / depth 2 / occupancy | 1.000 everywhere | 1.000 everywhere | no loss: MET |
| unseen transitions FULL | 0.638 | **0.570**±0.029 | loss |
| depth 3 UNSEEN FULL | 0.813±0.030 | 0.850±0.052 | > 0.9 NOT met |
| depth 3 per form, A / C: T0 | 0.99 / 0.95 | 1.00 / 1.00 | — |
| T1 "C receives B from A" | 0.94 / 0.92 | 0.96 / 1.00 | — |
| **T2 passive** | 0.81 / 0.52 | **0.77 / 0.52** | C > 0.9 NOT met |
| **T3 passive** | 0.84 / 0.76 | 0.81 / 0.79 | C > 0.9 NOT met |
| T9 fronted C | 0.99 / 0.77 | 1.00 / 0.75 | — |
Gates: the participles at 1.00 (given); the recipient verbs rise to 0.87-0.89 (learned).
Reading, by the decision rule stated before the run: NOT met -> the form effect is a stated property of the predicate-role world; the 0.813 /
two-number form with the breakdown stays; no further remedy is built. The emitter repaired what the trace predicted it could (the non-passive
forms reach 1.00 at depth 3) and did not touch the passives: with the participle open C does hang off 'given' (given:P) but so does the
by-phrase's A through 'given:.' + L[.:P], and the learned displacements did not separate the two at depth 3. [WITHDRAWN (indications 013c490
§0): the sentence that classified the passive's by-phrase agent as right-announced structure — the relation markers announce their fillers before
them; the classification waits for E1b (relation-marker gates given open), 10b-quinquies.] The cost on unseen transitions (0.64 -> 0.57) is the
price of one more emitter.

### 10c-quater-bis. The 50-case audit (indications 013c490 §4a) — logs/audit_confident_wrong_sel_50.jsonl, verdicts and one-line reasons filled in
Labeler: an LLM (Claude Fable 5.1, the session running this repository; not an independent model — stated; anyone can check rows by hand).
Of the 50 confident-wrong selected-column cases (odd half of dev, tau 0.05, seed 0): **gold-correct (the reader is wrong) 34 = 0.68; gold-wrong 7
= 0.14; ambiguous question 9 = 0.18**. Share of the 16% that is gold noise or ambiguity: **0.32** (16/50). Typical reader errors: the reader
picks the CONDITION column (the one the question names a value for) instead of the asked column — 'what days were the rounds of ...' -> Rounds,
'which score has a location attendance of ...' -> Location Attendance (20 of the 34); synonyms the reader does not have ('driver' -> Rider,
'selection' -> Pick #, 'writers' -> Written by). Typical gold noise: the asked entity vs its score column ('what away team played at ...' ->
gold 'Away team score'), garbled questions. Consequence for the operating point: of the 0.162 false alarm on gold, about 0.11 is the reader's
and about 0.05 is the data's; the reader's part is mostly the asked-vs-condition column confusion, a decision the question's syntax carries
('what X has a Y of ...'): a reader property, as 10c-ter said. No operating point is chosen on this; C5 and 4c follow.

## 10b-quinquies. E1b — relation-marker gates GIVEN open: the reference table (v16_reference.py --rel_gates open; logs/2026-10-16_1905_v16_reference_1713d10.txt; 5 seeds, same seeds and regime as 10b; trace logs/trace_e1.txt part (iii))
Why (indications 013c490 §1): 7c learned 'to' 1.00 and 'by' 0.91 open and had the passives at 1.000 at depth 3; 10b learned 'to' and 'by'
closed (the predicate term and the recipient-less forms pulled them shut), and the E1 trace showed the A/C address collision on T2 comes from
both persons hanging off the period. Given (rule 20): 'to', 'by', 'from' open; participles and verbs learned; recipient-less forms kept.
Trace (iii), depth 3: T0 A <- ... that:to + L[to:P], C <- ... that:to + to:. + L[.:P]; T2 C <- ... that:to + to:by + L[by:P], A <- ... that:to +
to:by + by:. + L[.:P]; T3 A <- ... that:by + by:to + L[to:P], C <- ... by:to + to:. + L[.:P] — A and C land at DISTINCT addresses that both carry
the 'that' prefix, in every form. The emitter test was therefore the right one with the right emitters.
| axis (5 seeds) | 10b (gates learned: to, by closed) | **E1b (to, by, from given open)** | criterion |
|---|---|---|---|
| seen FULL / predicate / new ids / cues / depth 2 | 1.000 everywhere | 1.000 everywhere | seen axes 1.000: MET |
| occupancy known / unknown / not given | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | unchanged: MET |
| unseen transitions FULL | 0.638±0.015 | 0.627±0.024 | reported; no loss beyond seed noise (E1 (ii) had lost 0.07) |
| **depth 3 UNSEEN FULL** | 0.813±0.030 | **0.968±0.020** | > 0.9: MET |
| depth 3 per form A / C: T2 passive | 0.81 / 0.52 | **1.00 / 1.00** | C > 0.9: MET |
| T3 passive | 0.84 / 0.76 | **1.00 / 1.00** | MET |
| T0, T1, T6, T12, T15 | 0.94-1.00 / 0.91-0.99 | 1.00 / 1.00 | — |
| T9 "to C , A V B ." (fronted recipient) | 0.99 / 0.77 | 0.99 / **0.84** | the one form below 0.9 (the comma construction: a declared limit of 6b) |
| calibration, depth 3 (seed 0): A / C accuracy at full coverage; at tau 0.05 coverage, confident-wrong | 0.90 / 0.875; 0.59 / 0.33, 0.000 / 0.008 | **1.000 / 0.967; 0.93 / 0.88, 0.000 / 0.000** | — |
| unseen transitions, calibration (seed 0): A / C accuracy; coverage at tau 0.05; confident-wrong | 0.867 / 0.750; 0.69 / 0.67; 0.000 / 0.042 | 0.850 / 0.842; 0.84 / 0.74; 0.008 / 0.000 | — |
Gates: that, by, from, to 1.00 (given for the last three), '.' 0.53; everything else closed (the recipient verbs no longer need to open).
Reading, by the decision rule stated before the run: MET. The depth-3 regression of the predicate-role world was the GATE SET, not the form:
with the relation markers open, the passives compose at depth 3 like the active forms (1.000), the margins at depth 3 triple (A 0.06 -> 0.12,
C 0.04 -> 0.10) and the abstention falls from 0.4-0.7 to 0.07-0.12. The remaining loss is the comma construction T9 (C 0.84), consistent with
6b. This table is the REFERENCE TABLE from now on (10b-quinquies); the depth-3 claim is restated without "but the passives": "depth 3 unseen
0.968 at full coverage; at tau 0.05 coverage 0.93 (A) / 0.88 (C) with covered accuracy 1.000, confident-wrong 0.000", gate choice recorded as
given (like 'that'). 10b, 10b-bis, 10b-ter, 10b-quater stay as the history of the learned-gate world. The 10b-quater sentence on right-announced
structure stays withdrawn: the markers announce their fillers before them, and with them open the passive is read.

### 9c-sexies. C2c — quantifier scope as CATEGORY-LOCAL terms, wide/narrow derived at readout (v14_quantifiers.py --scope_target c2c --rel; logs/2026-10-16_1918_v14_quantifiers_baf4703.txt; 5 seeds, regime of 9c-bis; gates LEARNED, determiners closed; trace logs/trace_c2c.txt)
Why (indications 013c490 §2): C2b composes the one-quantifier scope because the term is category-local and relative to the slot; the
two-quantifier case failed because its target was CONDITIONAL ('every' emits the scope term alone, not next to 'each') — a relation. C2c
removes the relation from the reader. World correction (rule 16, rows that change): 'a' and 'the' draw nothing; marked determiners
('each', 'a certain') draw their noun a second time at R_wide (x) R_role (x) f_noun; unmarked quantifiers ('every', 'some') at R_q (x) R_role
(x) f_noun; both unconditional, one learned relative displacement per determiner category (the C2b emitter). Readout rule (printed next to
every number): an R_wide term present -> that slot is wide and the R_q slot narrow; one R_q term alone -> the only quantifier; two R_q and no
R_wide -> "both". Term threshold per depth from calibration rows (rule 25; d0 0.16, d1 0.14-0.15, d2 0.09-0.11 across seeds); presence as
in 9c-quater; STRICT depth-2 columns with the depth-1 threshold. Trace (rule 11, one two-quantifier sentence at depth 2): the two second
drawings sit at distinct addresses, both carrying the 'that' prefix. Chance: noun 1/12, determiner 1/6, scope 1/3.
| axis (5 seeds; mean ± sd, per-seed list where a criterion applies) | C2c | C2b (slot target, 9c-quinquies) | criterion |
|---|---|---|---|
| depth 0: nouns / determiners | **0.999**±0.002 / 0.950±0.013 | 0.998 / 0.919 | nouns >= 0.99: MET |
| depth 0: PRESENCE / affirmative read ABSENT | 1.000 / 1.000 | 0.996 / 1.000 | — |
| depth 0: SCOPE|gold-q ONE quantifier (per seed) | 0.995 (1.000, 0.974, 1.000, 1.000, 1.000) | 0.993 | >= 0.99: MET on the mean; seed 1 at 0.974 |
| **depth 0: SCOPE|gold-q TWO quantifiers (marked)** (per seed) | **0.996** (1.000, 1.000, 0.994, 0.988, 1.000) | 0.685 | > 0.9: **MET, 5/5 seeds** |
| depth 1: SCOPE one / two | 0.998 / 1.000 | — | — |
| depth 0 FULL (joint) / depth 1 FULL | 0.887±0.027 / 0.834±0.027 | 0.686 / 0.599 | — |
| **depth 2 (UNSEEN): SCOPE|gold-q ONE quantifier** (per seed) | **0.992** (0.990, 0.980, 1.000, 1.000, 0.990) | 0.998 | >= 0.99: MET on the mean; seeds 0, 1, 4 at 0.98-0.99 |
| **depth 2 (UNSEEN): SCOPE|gold-q TWO quantifiers (marked)** (per seed) | **0.966** (0.988, 0.927, 0.976, 0.951, 0.988) | 0.481 | > 0.9: **MET, 5/5 seeds** |
| depth 2: PRESENCE (d2 threshold) / STRICT (d1 threshold, per seed) / aff ABSENT | 0.983 / **0.924** (0.967, 0.850, 0.967, 0.889, 0.950) / 1.000 | 0.978 / 0.867 / 1.000 | reported |
| depth 2 FULL (d2 threshold) / STRICT | 0.603±0.076 / 0.536±0.102 | 0.472 / 0.409 | — (the determiner identity is the bottleneck; not a criterion axis) |
| new nouns d0 FULL | 0.856 | 0.608 | — |
| ambiguous (never trained): both reported (rule: two R_q, no R_wide) / cos superposed / forced surface-first | **0.993**±0.010 / 0.866 / 0.517 | 0.655 / 0.837 / 0.622 | C3: both > 0.9 AND one reading > 0.9 on unambiguous: MET |
| FALSE-BOTH on unambiguous sentences, d0 / d2 | **0.000 / 0.000** (every seed) | not printed | reported (C3) |
Gates (learned): that 1.00, to 0.80, '.' 0.64, from / by 0.60; three seeds open the relation markers (to, by, from), two seeds (1, 3) open the
recipient verbs instead — seed 1 is the one below 0.99 on the one-quantifier scope. Determiners stay closed (0.1-0.2).
Reading, by the criterion and decision rule stated before the run (indications §2): **MET**. With the relation removed from the reader — each
determiner category emits its own term, and wide/narrow is a rule the legend applies to the two terms — the two-quantifier scope composes at
the unseen depth (0.966, every seed above 0.9) as the one-quantifier scope did in C2b, with no false "both" on unambiguous sentences and the
never-trained ambiguous sentences read as "both" at 0.993. Consequences: (i) quantifiers join the regularity ENTIRELY ("seen at depths 0-1,
share 0.6, tested at depth 2"); (ii) the two-quantifier exception row is REMOVED from STATUS; (iii) H-b is RETIRED as "a relation the legend
computes from category-local terms" — the reader never needed to represent it; (iv) the permuted-filler term (current_status E1 of 3b4f837) is
NOT built: the condition for it no longer holds; (v) the correction history role (9c-bis) -> slot / noun (9c-ter) -> C2 (9c-quater) -> C2b
(9c-quinquies) -> C2c stays here as the record; (vi) the C2c world is the quantifier reference world from now on (slot target, judgment Q2).
Costs to report: determiners 0.950 at depth 0 (the three terms in one slot compete), FULL at depth 2 0.60 for the same reason; the strict
presence at depth 2 (0.924) is now above the labelled one of C2 (0.81) with no depth-2 information.

## 10e. N5 — kind ambiguity: noun/verb homographs in the reference world (v19_kinds.py --seeds 5; logs/2026-10-16_1941_v19_kinds_90c44df.txt; 5 seeds, D = 1024, 30k sentences, predicate role present)
Definition (judgment 013c490 Q3, addendum N5): three homographs h0-h2 are objects o15-o17 AND the V3 form of GIVE / SELL / SEND; the context
decides the kind. Variants, same regime: ref = unambiguous world (distinct words); (a) one category per token = the MAJORITY kind in training
(verb, 9,000 vs 5,500 uses); (b) SUPERPOSITION: the token's state is the mean of the two kind assignments' states (every category-dependent key
becomes the half-way key, in all codes); the kind is resolved at READOUT by fit — noun hypothesis cos with R_B (x) (f_h + marker), verb
hypothesis with R_pred (x) f_lemma — resolved if margin > tau = 0.05, else both reported. Bag control: cos between the bag of a noun use and
the bag of a verb use of the same word vs the same cosine between states. Chance: objects 1/18, verbs 1/12. Kinds are an input (rule 20);
the half-way key is a definition, not learned.
| variant (5 seeds) | FULL h-as-noun (per seed) | FULL h-as-verb | FULL no homograph | kind by fit: acc noun / verb | margin | resolved (margin > tau) / both | cos(noun use, verb use): bag / state |
|---|---|---|---|---|---|---|---|
| ref (unambiguous) | 1.000 | 1.000 | 1.000 | — | — | — | — |
| (a) majority kind (verb) | **0.989**±0.014 (0.991, 1.000, 0.963, 1.000, 0.994) | 1.000 | 1.000 | 0.923 / 0.535 | 0.135 | 0.540 / 0.460 | 0.866 / 0.807 |
| (b) superposition, readout by fit | **0.907**±0.026 (0.950, 0.900, 0.909, 0.869, 0.906) | 1.000 | 0.999 | 0.967 / 0.374 | 0.149 | 0.555 / 0.445 | 0.866 / 0.821 |
Criterion (addendum N5): (b) FULL on sentences with an ambiguous word within 0.02 of the reference. (b) averages 0.953 over the two uses vs
1.000 (gap 0.047); on the noun uses alone 0.907 (gap 0.093, every seed below 0.98): **NOT met**. Reading: the verb use is read perfectly under
both variants (the predicate slot takes the lemma; the half-way drawing costs nothing there), the NOUN use loses 0.09 under the superposition and
0.01 under the majority-kind baseline — the baseline (a) is the better of the two, because the noun slot is reached through the determiner's
and the verb's beacons regardless of the token's own category key, while the half-way drawing halves the noun term in the slot. The kind by fit
reads the noun correctly (0.97) but the verb hypothesis fits only 0.37 of the verb uses with margin > tau: the lemma readout competes with the
noun readout and leaves 0.45 of the tokens as "both". The state separates the two uses barely more than the bag (0.82 vs 0.87). Decision rule
stated before the run (not met -> report, propose nothing without judgment): (a) stays the status quo; (b) as defined is recorded as a
negative result; no remedy proposed here (candidates for judgment: resolve the kind BEFORE drawing from the first open emitter's preference,
the diagnostic the indications named, which this run did not print — PLANNED).

## 10c-quinquies. 4c — the reader's confident-wrong against training size, with the lexical-mismatch split at every point (v17_verifier.py --n_train 1000 / 5000: logs/2026-10-16_2033 and _2035_v17_verifier_90c44df.txt; rescore_v17_e3.py on the saved frames: logs/2026-10-16_2034 and _2038_rescore_v17_e3_90c44df.txt; the 56,355 point is 10c-bis / 10c-quater; seed 0, D = 512, 3 epochs; presence threshold on the even half of dev, every number on the odd half, 4,210 questions, tau 0.05)
| training questions | sel accuracy / confident-wrong / abstain | agg accuracy / confident-wrong | sel confident-wrong, gold column name IN the question / NOT in (sel accuracy) | verifier: role-error recall / precision / false alarm on gold | reader training (s) |
|---|---|---|---|---|---|
| 1,000 | 0.384 / 0.056 / 0.837 | 0.742 / 0.247 | 0.052 (0.415) / 0.069 (0.266) | 0.207 / 0.891 / 0.127 | 18 |
| 5,000 | 0.587 / 0.082 / 0.569 | 0.756 / 0.227 | 0.065 (0.654) / 0.147 (0.336) | 0.378 / 0.906 / 0.197 | 161 |
| 56,355 (10c-bis) | 0.814 / 0.053 / 0.243 | 0.824 / 0.137 | 0.023 (0.910) / 0.169 (0.449) | 0.522 / 0.942 / 0.162 | 4,030 |
Reading (no criterion; the curve the indications asked for): the selected-column confident-wrong rate at tau 0.05 does NOT fall with data
(0.056 -> 0.082 -> 0.053) while accuracy rises 0.38 -> 0.81 and abstention falls 0.84 -> 0.24: more data converts abstentions into right
answers, not confident-wrong into right. The lexical-mismatch split explains where the confident-wrong stays: with the column name in the
question it falls 0.052 -> 0.023, without it it RISES 0.069 -> 0.169 — the full reader is confident on mismatched questions it still gets
wrong 0.55 of the time. The aggregator's confident-wrong falls with data (0.25 -> 0.14) but stays the larger of the two (8d: a lexical
decision). The verifier's role-error recall grows with the reader (0.21 -> 0.52); its false alarm on gold does not improve (0.13-0.20), as
10c-ter said: a reader property. The 8d data curve itself (other sizes) was not re-scored; these are three new points.

## 11. P0 — the reader packaged for the tool (tool/; built by a separate agent in a worktree from docs/specs/SPEC_worlds_server_sop_cli.md; merged at commit ed73f41; tests tool/tests/run_all.sh; results in tool/README.md "P0 results")
Engineering, no new measurement (indications §5). Delivered: `sop` package (SOP Lang reader/writer, check with per-world legends and `_world`,
numpy formalize, diff with four verdicts, stats, train on V8 + trainer.train_loop, export of the three constructed worlds), `server.js`
(stdlib only), skill `skills/train-world/`, six tests (all PASS on CPU), README with the spec-v2 conformance table and the choices where the
spec was silent. Nothing under experiment/ was modified. Acceptance round trip (1 seed, CPU, full sizes; runs not committed):
| world | axis | package (1 seed) | RESULTS (5 seeds) | reading |
|---|---|---|---|---|
| reference | seen / predicate / new ids / cues | 1.000 | 1.000 (10b) | reproduced |
| reference | unseen transitions (roles) / depth 3 | 0.633 / 0.792 | 0.638±0.015 / 0.813±0.030 (10b, learned gates) | within seed variance |
| commands | depth 0 FULL / new patients / depth 2 FULL | 0.998 / 1.000 / 0.790 | 0.999 / 1.000 / 0.854±0.031 (10d) | depth 2 about two sd low (1 seed) |
| quantifiers | nouns d0 / d2; noun+determiner d0 / d2 | 0.997 / 0.993; 0.878 / 0.726 | 0.999 / 1.00; determiners 0.923 / 0.46-0.64 (9c-bis) | nouns reproduced; the v1 export emits NO scope term, so this is not the 9c-bis world (scope not compared) |
The tool's own FULL is lower where it must also estimate the reporting depth and negation from the state (the scripts are given the depth).
Missing-role recovery (test 5b, offline, oracle annotator): with `destination` removed from the commands legend the only novel supported
`_unplaced` cluster is {archive, cloud, folder, trash} (225 sentences), proposed, migrated, retrained, accepted (FULL 0.860 -> 0.871). Small vs
large annotator: "API unavailable" (PLANNED, as expected). Model sizes: reference 12 MB, quantifiers 15 MB (above the spec's "a few MB"),
commands 2.9 MB. Not done: spec section 6 (never received; marked TODO), LLM steps (behind `llm_call`), decoding of if / must / may / and,
mood as a role, the "two legends with the same questions are one world" check. The 5-seed GPU round trip is a command in tool/README.md,
not run (PLANNED).

### 9c-septies. FULL on the quantifier world redefined over the legend's questions (hand-over 78d36e8 Q2, rule 16) — both definitions once, on the saved C2c models (rescore_v14_c2c_full.py; logs/2026-10-16_2125_rescore_v14_c2c_full_bba9497.txt; 5 seeds, same calibration and scored rows as 9c-sexies; no retraining)
Definition change (rule 16): the legend asks which phrase is wide, not which determiner word was used, so from here FULL on the quantifier
world = nouns + presence + scope-or-both; the determiner IDENTITY is its own axis outside FULL. FULL-old (the 9c-* columns up to 9c-sexies)
= nouns + determiners + presence + scope. On affirmative rows presence means ABSENT; on ambiguous rows "both" by the C2c readout rule.
| axis (5 seeds) | FULL-old | **FULL-new** |
|---|---|---|
| unambiguous d0 | 0.887±0.027 | **0.995**±0.006 |
| unambiguous d1 | 0.834 | **0.996** |
| unambiguous d2 UNSEEN (d2 threshold) | 0.603±0.076 | **0.967**±0.023 (0.978, 0.922, 0.978, 0.967, 0.989) |
| STRICT d2 (d1 threshold) | 0.536±0.102 | **0.864**±0.065 (0.911, 0.756, 0.922, 0.822, 0.911) |
| affirmative d0 / d2 (ABSENT) | 1.000 / 0.999 | 1.000 / 0.999 |
| new nouns d0 | 0.856 | **0.998** |
| ambiguous d0: both / FULL-new (nouns + presence + both) | 0.993 / — | 0.993 / **0.990** |
| determiner identity, own axis (all three right per sentence), d0 | 0.890±0.024 | (outside FULL) |
Reading: the whole gap between the old FULL columns and the scope columns of 9c-sexies was the determiner identity (three terms competing in
one slot: 0.890 per sentence, 0.950 per determiner). Over the legend's questions the C2c world reads the unseen depth at 0.967 (strict 0.864)
and new nouns at 0.998. The 9c-* FULL numbers before this section keep the old definition and are not restated.

## 10c-sexies. C5 — condition count from an OCCUPANCY MARKER in the verifier's reader world (v17_verifier.py --occ_marker; logs/2026-10-16_2126_v17_verifier_bba9497.txt; one retraining, seed 0, D = 512, 3 epochs, 56,355 questions; presence threshold on the even half of dev, every number on the odd half, 4,210 questions)
World rows that change (rule 16): a FILLED condition slot k adds R_COND[k] (x) R_OCC (x) M to the target (X2 applied to the verifier world;
R_OCC and M fixed, given). Presence of a condition = cosine with M after unbinding R_COND[k] R_OCC, thresholded at the midpoint of the
calibration half (present 0.374 / absent 0.015 -> 0.194; 10c-bis read presence from the column cosine: present 0.158 / absent 0.043 -> 0.101).
| axis (odd half) | 10c-bis (presence from the column cosine) | **C5 (presence from the marker)** | criterion |
|---|---|---|---|
| condition-slot occupancy: present read as present / absent read as absent | not printed then (the column cosine; separation 0.158 vs 0.043) | **0.890** (n 5,784) / **0.992** (n 11,056) | both > 0.9: **NOT met** (present 0.890) |
| drop_cond detected / unverifiable / confident-wrong | 0.159 / 0.690 / 0.151 | 0.150 / 0.825 / 0.025 | before / after reported |
| add_cond | 0.162 / 0.837 / 0.001 | 0.153 / 0.846 / 0.000 | — |
| swap_sel_cond / change_sel / flip_agg detected | 0.740 / 0.764 / 0.805 | 0.707 / 0.732 / 0.781 | no loss beyond 0.03 |
| gold: false alarm / unverifiable / confirmed | 0.162 / 0.541 / 0.297 | 0.152 / 0.495 / 0.353 | — |
| role-error recall / precision | 0.522 / 0.942 | 0.505 / 0.943 | — |
Reading: the marker separates the two cases far better than the column cosine (0.374 vs 0.015 against 0.158 vs 0.043) and reads absence at
0.992, but presence stays at 0.890, under the criterion: the slots the reader misses are the second and later conditions (the first slot is
the one most questions fill). The drop_cond / add_cond DETECTION did not move (0.15), because the verdict code is unchanged: a condition-set
disagreement is called "error" only when the minimum COLUMN-identity margin over the slots clears tau, and that margin is small on empty slots —
so 0.83 of the condition errors are "unverifiable", not missed (confident-wrong on drop_cond fell 0.151 -> 0.025). The marker's own margin
(distance from the presence threshold) is not used by the verdict yet; using it is a re-scoring of the saved frames
(logs/.ckpt_v17_reader_frames_occ.pt), no retraining — proposed, not done (current_status.md).

## 10c-septies. Mismatch-aware abstention (hand-over 78d36e8 Q4; rescore_v17_mismatch.py; logs/2026-10-16_2235_rescore_v17_mismatch_e8be5cf.txt; saved frames of 10c-bis, no retraining; tau per rule set on the even half, curve on the odd half)
Feature available at verification time: the name of the column the reader CHOSE appears verbatim in the question (the schema is known, the gold
is not). Rule M = abstain when margin < tau OR the chosen name is absent; control = margin only. Criterion: confident-wrong on sel < 2% at the
coverage that results, coverage reported (rule 18: calibration reporting, not tuning).
| rule | tau chosen on the calibration half (first grid value with confident-wrong < 2% there) | report half: coverage / covered accuracy / confident-wrong | criterion |
|---|---|---|---|
| margin only (control) | 0.15 | **0.445** / 0.980 / **0.009** | MET at coverage 0.445 |
| M: margin AND chosen name in the question | 0.10 | 0.561 / 0.961 / 0.022 | NOT met (0.022) |
| M at tau 0.15 (for comparison, same tau as the control) | — | 0.424 / 0.980 / 0.009 | the name rule costs 0.02 coverage for no gain |
| at the stated tau 0.05: margin only / M | — | 0.757 / 0.930 / 0.053 ; 0.689 / 0.932 / 0.047 | — |
The 4c split at tau 0.05 (gold column name in the question or not): IN — confident-wrong 0.023 / 0.022, coverage 0.834 / 0.834 (control / M);
NOT in — confident-wrong 0.169 -> 0.143, coverage 0.466 -> 0.143. Chosen name in the question on 0.837 of the report rows (accuracy 0.860
there) vs 0.163 (accuracy 0.576).
Reading: the 0.169 cannot be turned into abstentions by this feature: on the mismatched questions the reader's confident wrong choice is
mostly a column whose name IS in the question (the wrong one of the mentioned columns), so rule M removes coverage (0.466 -> 0.143) while the
confident-wrong rate there falls only 0.169 -> 0.143. The margin alone reaches the criterion at tau 0.15 with coverage 0.445; the chosen-name
rule adds nothing at equal tau. The mismatch finding of 10c-quater stays a property of the reader, not a feature to abstain on.

## 12. B5.0 — the operator-ORDER collision of commutative binding, measured as it is (b5_0_collisions.py; logs/2026-10-16_2236_b5_0_collisions_e8be5cf.txt; instruction after the B5 stop, step 1; no remedy, nothing added to the reader)
Part A, algebra (world states, 100 fresh events per pair). Rendered sentence pairs differing ONLY in operator order: **0** in all three grammars
(E1, E3, B3 each nest their operators in one fixed order). Pairs whose two orders are expressible as world states:
| world | pair | cos(world states) | class | rendered? |
|---|---|---|---|---|
| E1 | rep o neg_S vs neg_S o rep ("X says that A never gives" / "X never says that A gives") | 0.762 | SEPARATED: witness = the speaker term | first yes, second no |
| E3 | rep o may vs may o rep | 0.750 | SEPARATED: speaker | first yes, second no |
| E3 | if-then(may on Q) vs may o if-then | 0.525 | SEPARATED: the P clause under one operator only | first yes (mixed), second no |
| E3 | rep o if-then vs if-then o rep (both clauses reported) | 0.802 | SEPARATED: speaker | first yes, second no |
| E3 | must o may vs may o must | **1.000** | **COLLIDING** (identical state, same frame) | neither (the grammar never stacks two modals) |
| E3 + neg | must o neg vs neg o must ("A must not give" / "A does not have to give") | **1.000** | **COLLIDING** | neither (no world has both operators; built from E3's R_must and a R_neg drawn like E1's) |
| B3 | rep o neg_S vs neg_S o rep ("the user says that do not send" / "the user never says that send") | 0.752 | SEPARATED: speaker | first yes, second no |
Counts: colliding 2, witnessed 5. Word-level operators (constituent negation, the negated attribute) are not order operators (listed as such).
Rule 11 traces, term by term: must o neg vs neg o must — every term coincides (cos 1.000 on A, B, C), the bag of identities is identical too:
colliding. rep o neg vs neg o rep — the three inner terms coincide (1.000 each), the speaker term differs (cos 0.047): the witness; whole-state
cos 0.766 (3 of 4 terms shared).
Part B, reader charts (seed 0 of each world RETRAINED with the original regime — E1b, E3b, 10d — because those runs saved no checkpoints;
the reversed sentence was never in training; n = 100 pairs each):
| pair | cos world states | cos reader charts | cos bags (control) | order hypothesis right: rendered sentence / reversed (unseen) | frames DIFFER |
|---|---|---|---|---|---|
| E1 rep o neg vs neg o rep | 0.762 | 0.364 | 1.000 | 1.000 (margin +0.115) / **0.000** (margin -0.027) | 0.000 |
| E3 if-then(may on Q) vs may o if-then | 0.525 | 0.502 | 0.888 | 1.000 (+0.171) / **0.980** (+0.027) | 0.980 |
| B3 rep o neg vs neg o rep | 0.758 | 0.719 | 0.857 | 1.000 (+0.168) / **0.000** (-0.138) | 0.000 |
Reading. (i) The collision is real and exact where two wrapping operators stack with nothing between them (must/neg, must/may): the frame is
the same and no reader of these worlds can tell the orders apart; no grammar of the repository renders such a pair, so it is a measured
property of the algebra, not of a tested sentence. (ii) Where a term sits under only one of the two operators (the speaker; a clause of the
conditional) the two orders are different states (cos 0.53-0.80) and are decodable through that witness — the reader reads the rendered order
right at 1.000 on all three worlds. (iii) On the REVERSED, never-trained order the reader's chart carries the witness only in E3 (0.980: 'perhaps'
is an open emitter the reader already knows at depth 0, so "perhaps if P , then Q" is read as may over the whole conditional); in E1 and B3 the
reversed sentence ("X never says that ...") is read as the trained order (0.000): the reader's chart separates the two sentences (cos 0.36 /
0.72 against bags 1.00 / 0.86) but not along the witness — the 'never' before 'says' does not move the speaker's star under R_neg, because that
displacement was never trained. The bag control cannot tell E1's pair at all (1.000). This list is the measured extent of the chapter-8 limit
and the record B5c is judged against.

### 12-bis. B5c — rule 11 trace before the run: operators as levels with an identity term (trace_b5c.py; logs/trace_b5c.txt; no training; real V8 reader, random parameters, 3 draws, HARD gates: operators / auxiliaries / 'to' / 'that' open, content words closed)
World under test (instruction after the stop, step 2): op(X) = R_op (x) f_op + R_lvl (x) X for sentential operators; targets per token from the
sentence's operator structure. Method, independent of key values and of training: the ADDRESS of each token (the reader's state with that
word as the only non-zero identity) is compared across two sentences; a necessary condition for any learnable match is that two cases with
DIFFERENT targets have DIFFERENT addresses.
| pair | token | cos(addresses) | cos(targets) | reading |
|---|---|---|---|---|
| "p0 must give o0 to p1" vs "p0 must not give o0 to p1" | must | < 1 where targets differ | — | (a) reachable |
| same | give, o0, p1 | distinct (0.3-0.4 and below) | distinct | (b) reachable: the extra R_lvl can be carried by the hop out of the operator |
| same | **p0 (the subject)** | **1.000, every draw** | −0.084 | **unlearnable**: identical address, different targets |
| "p0 does not give" vs "p0 does not have to give" | p0 | **1.000** | −0.084 | unlearnable |
| "p2 says that p0 must give" vs "... p0 must not give" | p0 | **1.000** | −0.084 | unlearnable (inside a report too) |
Why: a closed word's address is a function of the words BEFORE it (path, bigram) and of the FIRST open emitter after it (one left re-seating
key); it never sees the second operator. With hard gates "p0 must give" and "p0 must not give" give the subject the same address in every code.
With SOFT gates the subject's address would differ only through the leak past the first operator (the share 1 − g_must that re-seats to 'not'
instead), i.e. a separation that rests on a partially closed operator gate, not on a level — not the mechanism B5c names.
Verdict, by the instruction's rule: (a) and (b) hold for every word except the subject that PRECEDES its stacked operators; for that subject the
existing key structure gives exactly one level. B5c as specified is not fully implementable without a model change -> **STOP, report, nothing
built**. Operator-initial material (an operator before the whole clause: 'perhaps', 'if', a report) satisfies the condition throughout, which is
why the composition regularity held for every operator so far: those operators all precede what they scope over. The post-subject auxiliary
("A must not ...") is the right-announced case inside one clause (chapter 8), with B5.0's list as its measured extent.

### 11-bis. P0 — the 5-seed GPU round trip through the package (tool/tests/p0_full_gpu.sh at commit c843f53; runs committed under tool/runs/p0_gpu_5seed_c843f53/<world>/ with p0_table.md; built and run by the tool agent; quantifier world v2 = C2c)
| world | axis | package, 5 seeds (roles) | tool FULL, depth ESTIMATED | tool FULL, depth GIVEN | RESULTS (5 seeds) |
|---|---|---|---|---|---|
| reference | seen formulations / new identities / role cues | 0.985±0.012 / 0.994±0.007 / 1.000 | same | same | 1.000 / 1.000 / 1.000 (10b) |
| reference | unseen transitions | 0.643±0.017 | 0.564±0.079 | 0.567±0.077 | 0.638±0.015 |
| reference | depth 3 UNSEEN | 0.742±0.040 | 0.742 | 0.742 | 0.813±0.030 (10b, learned gates); 0.968 (10b-quinquies, relation-marker gates given open — not expressed in the tool's legend) |
| commands | depth 0 FULL / new patients | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.999 / 1.000 (10d) |
| commands | **depth 2 UNSEEN FULL** | — | **0.590**±0.059 (structure right on 0.684) | **0.857**±0.025 | 0.854±0.031 |
| quantifiers v2 | nouns d0 / two-quantifier scope d0 / **d2 UNSEEN** / ambiguous "both" | 0.999 / 0.986±0.017 / **0.917**±0.063 / 0.993 | — | 0.995 / 0.995 / 0.962±0.029 / 0.987 | 0.999 / 0.996 / 0.966 / 0.993 (9c-sexies) |
| quantifiers v2 | determiner word (own axis, outside FULL) | 0.744±0.006 | — | 0.995 | 0.950 (9c-sexies / 9c-septies) |
Reading (C3 answered): the commands depth-2 gap of the 1-seed CPU run (0.790) was the tool's DEPTH ESTIMATION at an unseen depth, not the
packaging — with the depth given the package reproduces 10d (0.857 vs 0.854); with the depth estimated FULL falls to 0.590 (structure read right
on 0.684 of rows). Quantifier v2 reproduces C2c on scope (two-quantifier at the unseen depth 0.917±0.063 vs 0.966: lower mean, larger spread;
"both" 0.993 = 0.993) with a lower determiner-word axis (0.744 vs 0.950: the tool decides the word with a presence threshold, v14 with its
relative rule). Reference: role cues and unseen transitions reproduced; seen formulations, new identities and depth 3 (0.742 vs 0.813) not
within seed variance — the run differs from v16 in three stated ways (per-seed identities from the tool's star function, depth estimated,
candidates from the whole catalog), not isolated. Open for the tool: the depth estimator at unseen depths; the given relation-marker gates
(to / by / from open, 10b-quinquies) as a legend field.

### 10c-octies. C5 re-scoring — the condition verdict with the MARKER'S margin as the abstention statistic (rescore_v17_c5.py; logs/2026-10-17_0757_rescore_v17_c5_a34973a.txt; saved frames of the C5 reader, no retraining; threshold = the saved calibration-half midpoint 0.194, untuned; tau 0.05; same injected errors as 10c-sexies)
| error type (odd half) | column-margin statistic (10c-sexies): detected / unverifiable / missed | **marker-margin statistic**: detected / unverifiable / missed |
|---|---|---|
| drop_cond | 0.150 / 0.825 / 0.025 | **0.745** / 0.230 / 0.025 |
| add_cond | 0.153 / 0.846 / 0.000 | **0.763** / 0.236 / 0.000 |
| swap_sel_cond / change_sel / flip_agg | 0.707 / 0.732 / 0.781 | 0.767 / 0.740 / 0.793 |
| change_value (invisible) | 0.149 / 0.495 / 0.356 | 0.423 / 0.221 / 0.356 |
| gold: FALSE ALARM / unverifiable / confirmed | 0.152 / 0.495 / 0.353 | **0.426** / 0.221 / 0.353 |
| role-error recall / precision vs gold | 0.505 / 0.943 | 0.762 / 0.899 |
Reading: the marker's margin turns the condition errors from "unverifiable" into "detected" (drop 0.15 -> 0.75, add 0.15 -> 0.76) — and turns the
gold frames into false alarms at the same rate (0.15 -> 0.43): the reader's condition SET disagrees with the gold set, confidently by the marker, on
0.43 of the gold frames (per slot the present side is read at 0.890, 10c-sexies; a frame with two or three conditions compounds it). The marker
statistic therefore trades abstention for false alarms one for one; precision falls 0.943 -> 0.899. C5's row stays NOT met (present 0.890); the
condition axis of the verifier is limited by the presence reading of the second and later slots, not by the verdict statistic. Reported, no
threshold tuning.

## 12-ter. B5.1 — modal and/or negation AFTER the subject, current operator definition (v20_operators.py; logs/2026-10-17_0742_v20_operators_a34973a.txt; 5 seeds, D = 1024, 40k, all forms at depth 0 and inside one report (share 0.4); trace after the run: trace_b51.py, logs/trace_b51.txt)
World: S = R_pred f_lemma + R_A f_A + R_B f_B + R_C f_C; op(S) = R_op (x) S; forms aff / must ("must", "has to") / may / neg ("does not", "never") /
must_neg ("must not", "must never") / neg_must ("does not have to", "need not"; the SAME target as must_neg by construction) / may_neg ("may not",
"might not"); reporting prefixed. Rule 28 address check before training (printed): the subject has the IDENTICAL address in "must give" /
"must not give" and in "does not give" / "does not have to give" under hard gates — the run was the declared measurement of that limit.
Readout: structure = argmax over 6 hypotheses of the mean best cosine of the four roles; roles under the GOLD structure; FULL under the decoded one.
| depth | form | A | B | C | pred | structure | FULL |
|---|---|---|---|---|---|---|---|
| 0 | every form (7) | 0.990-1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.990-1.000 |
| 1 (seen) | every form | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 (UNSEEN) | aff | **0.130** | 0.995 | 0.990 | 0.590 | 1.000 | 0.100 |
| 2 (UNSEEN) | must / may / neg | 0.145 / 0.070 / 0.115 | 1.000 / 1.000 / 0.990 | 1.000 | 1.000 / 1.000 / 0.785 | 1.000 | 0.07-0.15 |
| 2 (UNSEEN) | must_neg / neg_must / may_neg | 0.175 / 0.150 / 0.095 | 0.99 | 1.000 | 1.000 | 1.000 | 0.10-0.18 |
Colliding pair must_neg / neg_must, same event: decoded frames DIFFER 0.000 at depths 0 and 1 (identical target, as constructed), 0.610 at depth 2
(both wrong). Subject A by the number of operators after it: depth 0 — 0 ops 1.000, 1 op 0.997, 2 ops 1.000; depth 1 — 1.000 everywhere.
Gates learned: that, not, never, has, have 1.00; might 0.92, need 0.84, does 0.84, may 0.80, must 0.77, to 0.69; verbs 0.33.
Reading, by the decision rule stated before the run.
- At the seen depths the subject does NOT fail: 1.000 under one and under two stacked operators. The trace on the trained seed-0 reader says how
  (logs/trace_b51.txt): (i) the modal gates are PARTLY open (must 0.71, may 0.72, does 0.83), so the subject's left re-seating is a MIXTURE
  (a share through the modal, a share past it to the negation when there is one), and the two addresses differ (cos 0.95 / 0.94 for must and may)
  — hardened to 0/1 they coincide, as 12-bis said; (ii) for "does not" / "does not have to" the two subject addresses are IDENTICAL even with the
  learned gates (cos 1.000: 'does' 0.83 and 'not' 1.00 sit before the difference), yet the subject is read at 1.000 under both gold keys — the
  identity CLEANUP tolerates a blended address: the subject's term keeps a cosine of 0.26-0.35 with the gold key in every form (margin 0.18-0.29,
  no stacked-operator cost visible), the other 11 candidates stay near zero, and the operator STRUCTURE is carried by B, C and the predicate. So
  the limit (ii) is real on the ADDRESS and invisible in role accuracy on these sentences: the frame is right because the other roles carry the
  operators and the subject's identity survives at an address shared by two targets. Where it would bite: a construction in which the subject
  alone carries the operator (none in this world), or a calibration that needs the subject's margin. B5d is NOT needed for the frame readout on
  these forms; the judgment's "report how" is the two mechanisms above.
- At the UNSEEN depth 2 the subject fails for EVERY form, the affirmative included (0.07-0.18), while B, C (0.99) and the predicate (1.00 with an
  operator; 0.59 / 0.79 aff / neg) and the structure (1.000) are read. The trace (part 2): per code, the chain carries the subject at depth 1
  (0.865) and NOTHING at depth 2 (0.009), the path code carries B and C at both depths (0.72 -> 0.60) and never the subject (0.06). Cause stated:
  this run trained NO sentence at depth 2, so depth 2 is not only an unseen depth but an UNSEEN TRANSITION (the hop 'that' -> 'that' never
  occurred; rule 4) — the chain cannot compose a hop it never learned, and the subject is the word the chain carries. Every earlier unseen-depth
  result (6c, 9b-bis, 9d-bis, 10d, 9c-sexies, 10b-quinquies) had affirmatives at the test depth minus one AND the deeper transition seen (the E1b
  protocol: affirmatives at depths 0-2, operators at 0-1). This run departed from that protocol by my mistake; it is re-run with it
  (--p_aff2 0.3, affirmatives at depths 0-2; operator forms at 0-1; 12-quater), both logs kept.

### 12-quater. B5.1 re-run with the E1b protocol (v20_operators.py --p_aff2 0.3; logs/2026-10-17_0840_v20_operators_ec0aa03.txt; 5 seeds; affirmatives at depths 0-2 (share 0.3), the operator forms at depths 0-1; depth 2 is now an unseen DEPTH for the operator forms, not an unseen transition)
| depth | form | A | B | C | pred | structure | FULL |
|---|---|---|---|---|---|---|---|
| 0 / 1 (seen) | every form | 0.980-1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.980-1.000 |
| 2 (UNSEEN for the operator forms) | aff / may / neg / must_neg | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 (UNSEEN) | must / neg_must | 0.965±0.034 / 0.990±0.012 | 1.000 | 1.000 | 1.000 | 1.000 | 0.965 / 0.990 |
| 2 (UNSEEN) | may_neg ("may not", "might not") | **0.855**±0.114 | 1.000 | 1.000 | 1.000 | 1.000 | 0.855 |
Subject A by the number of operators after it: depth 0 — 1.000 / 0.998 / 0.995 (0 / 1 / 2 ops); depth 1 — 1.000 / 1.000 / 0.993; depth 2 UNSEEN — 1.000 /
0.988 / **0.948**. Colliding pair must_neg / neg_must: frames DIFFER 0.000 / 0.000 / 0.007 (identical target, as constructed). Gates learned:
that, to, has, not, never, might 1.00; need 0.95, does 0.87, have 0.85, may 0.85, must 0.83 — every operator word opens.
Reading, by the decision rule stated before the run (judgment e9274a9 Q1): the subject does NOT fail. With the regime right (12-ter's depth-2
failure was the unseen 'that' -> 'that' transition of a run that trained no depth-2 sentence), modal + negation after the subject compose to the
unseen depth like every operator before them: structure 1.000 everywhere, B, C and the predicate 1.000, the subject 0.95-1.00 under two stacked
operators at the unseen depth, with ONE form BELOW THE CRITERION — may_neg at depth 2 (0.855±0.114; per-seed list in 12-quinquies) — the only place where the stacked post-subject
operators cost anything. The "how" stands as 12-ter / trace_b51 said: the operator gates learn OPEN (0.83-1.00), the subject's re-seating is a
mixture through and past the first operator, and the identity cleanup reads the filler from a blended address whose cosine with the gold key
stays positive; the operator STRUCTURE is carried by the words after the operators. Consequences: limit (ii) is recorded as a property of the
subject's ADDRESS (12-bis: identical under hard gates; identical with the learned gates for "does not" / "does not have to"), with NO measured
cost on role accuracy on these forms except may_neg at the unseen depth; **B5d is dropped** (the decision rule's second branch); the composition
regularity now also covers subject + modal + negation (the prohibition form of regulations), seen at depths 0-1, tested at depth 2. The first
run (12-ter) is kept as the record of the regime error: an unseen depth must not also be an unseen transition (rule 4).

### 11-ter. Tool round 2 — one change per run on the reference and commands worlds (tool/tests/gpu_round2.sh at commit 815e5f9; tool/runs/round2_815e5f9/ with isolation_table.md; 5 seeds each; built and run by the tool agent; judgment e9274a9 Q6 / C2)
| reference, one change per row | candidates | seen formulations | new ids | unseen transitions | depth 3 UNSEEN |
|---|---|---|---|---|---|
| P0 (c843f53) baseline | catalog 18 per role | 0.985±0.012 | 0.994±0.007 | 0.643±0.017 | 0.742±0.040 |
| P0 re-scored with v16's candidate sets | 12 seen / 6 held out | 0.985±0.011 | 1.000 | 0.647±0.016 | 0.742 |
| **A: to / by / from GIVEN open (legend `gates_given`)** | catalog 18 | **1.000** | 1.000 | **0.428**±0.047 | **0.987**±0.007 |
| B: A re-scored with v16's candidate sets | 12 / 6 | 1.000 | 1.000 | 0.434±0.049 | 0.987 |
| C: B + v16's identity draw | 12 / 6 | 1.000 | 1.000 | 0.436±0.035 | 0.973±0.017 |
| RESULTS 10b-quinquies (v16, gates given) | 12 / 6 | 1.000 | 1.000 | 0.627±0.024 | 0.968±0.020 |
Commands, depth 2 UNSEEN: FULL with the depth from the gate count 0.682±0.342 (seeds 0, 1, 2, 4: depth right 1.000 and FULL = the depth-given
value 0.82-0.89; seed 3: the 'that' gate learned 0.24, the count reads depth 0 everywhere, FULL 0.000), with the fit 0.590±0.059, with the depth
given 0.857±0.025 (10d 0.854).
Reading: the given gates close both reference gaps (depth 3 0.742 -> 0.987, seen formulations 0.985 -> 1.000); the candidate set accounts for the
new identities only; the identity draw changes nothing beyond seed noise. One NEW gap, not isolated: unseen transitions 0.643 -> 0.428 with the
gates given (v16: 0.627). The per-role numbers point at the tool's PRESENCE decision (a filled role read as filled 0.927 -> 0.874; recipient 0.619;
v16 never decides presence on that split) — round 3 run D tests it as the single change. The gate-count depth estimator holds only when the
complementizer's gate opens — round 3 run E counts the level-opener KIND itself (deterministic from the legend), no gate involved.

### 11-quater. Tool round 3 — two single changes (tool/tests/gpu_round3.sh at commit 1d660ee; tool/runs/round3_1d660ee/ with round3_table.md; 5 seeds; built and run by the tool agent)
| D — reference, to / by / from given open, unseen transitions | readout | heads FULL | agent | object | recipient |
|---|---|---|---|---|---|
| round 2 A | presence decided | 0.428±0.047 | 0.833 | 0.811±0.063 | 0.619±0.047 |
| D (= A retrained: identical at every digit — training is deterministic) | presence decided | 0.428±0.047 | 0.833 | 0.811 | 0.619 |
| **D** | **presence NOT decided (every role read, as v16)** | **0.596**±0.048 | 0.833 | 0.811 | **0.839**±0.004 |
| RESULTS 10b-quinquies (v16, C always read) | — | 0.627±0.024 | | | |
| E — commands, depth 2 UNSEEN, per seed | learned 'that' gate | depth right: opener KINDS / open gates | FULL kind count / gate count / depth given |
|---|---|---|---|
| seeds 0, 1, 2, 4 | 1.00 / 0.70 / 1.00 / 1.00 | 1.000 / 1.000 | 0.838, 0.819, 0.867, 0.886 — identical under the three |
| seed 3 | **0.24** | **1.000** / 0.000 | **0.876** / 0.000 / 0.876 |
| mean | | 1.000 / 0.800±0.400 | **0.857**±0.025 / 0.682±0.342 / 0.857±0.025 (10d 0.854±0.031) |
Reading: (D) the round-2 drop on unseen transitions was the tool's PRESENCE decision: with every role read the recipient goes 0.619 -> 0.839 and the
split 0.428 -> 0.596, within about one standard deviation of v16's 0.627; the decided readout fails because the unseen formulations put the
recipient's presence ratio under the threshold set on training rows. (E) counting the level-opener KIND (deterministic from the legend's
level_openers on the token sequence) gets the depth right on every sentence of every seed, including the seed whose 'that' gate learned 0.24
and broke the gate count; FULL with the kind count equals FULL with the depth given (0.857 = 10d). The tool's reference world now reproduces
10b-quinquies on every axis (seen 1.000, new ids 1.000, depth 3 0.987, unseen transitions 0.596 vs 0.627 with every role read), and the commands
world reproduces 10d with its own depth estimate. Defaults after this round: depth estimator = opener kinds (gate count and fit as checks);
presence decided (the spec's `_empty` depends on it) — whether the reference world should read every role is a question for the reviewing session.

### 12-quinquies. 12-quater per seed (rescore_v20.py; logs/2026-10-17_1028_rescore_v20_b8b61dc.txt; the saved models of the E1b-protocol run, same worlds, seeds and evaluation rngs; no retraining)
The means reproduce the digits of the 12-quater log exactly: the readout is deterministic. Depth 2 (UNSEEN for the operator forms, rule 29 (B)),
subject A = FULL on every form (B, C, pred, structure are 1.000):
| form | per seed (0-4) | mean ± sd | seeds above 0.9 |
|---|---|---|---|
| aff / may / neg / must_neg | 1.000 ×5 | 1.000 | 5/5 |
| must | 0.925, 1.000, 0.925, 1.000, 0.975 | 0.965±0.034 | 5/5 |
| neg_must | 1.000, 0.975, 1.000, 1.000, 0.975 | 0.990±0.012 | 5/5 |
| **may_neg** ("may not", "might not") | **0.850, 0.800, 0.950, 0.675, 1.000** | **0.855**±0.114 | **2/5** — below the criterion 0.9 on the mean: the NAMED EXCEPTION of the regularity row |

## 10c-nonies. C5 closed — per-slot presence thresholds (rescore_v17_c5_slots.py; logs/2026-10-17_1029_rescore_v17_c5_slots_b8b61dc.txt; saved frames of the C5 reader; thresholds per slot index (1, 2, 3+) set on the calibration half, every number on the odd half; no retraining)
Criterion written before the run: present read as present >= 0.9 on the second and later slots AND false alarm on gold <= 0.152 (10c-sexies).
| scheme | slot 1 present / absent (n) | slot 2 present / absent | slot 3+ present / absent | slots 2+ present | gold false alarm (column-margin verdict) |
|---|---|---|---|---|---|
| single threshold 0.194 (10c-sexies) | 0.995 / 0.000 (4,177 / 33) | 0.726 / 0.981 (1,312 / 2,898) | 0.129 / 1.000 (295 / 8,125) | 0.616 | 0.152 |
| per-slot thresholds (0.452 / 0.156 / 0.097) | **0.330** / 0.182 | 0.758 / 0.964 | 0.773 / 0.930 | **0.760** | 0.153 |
Present slots read as present by the NUMBER of conditions in the question (single threshold): 1 condition 0.995 (n 2,865), 2 conditions 0.835
(2,094), 3+ conditions 0.664 (825). By SLOT INDEX: 0.995 / 0.726 / 0.129.
Reading: NOT met — closed as "bounded by the presence reading of the later slots". The per-slot thresholds lift slots 2+ from 0.616 to 0.760
(still under 0.9) and wreck slot 1 (0.995 -> 0.330): the absent class of slot 1 is tiny (33 odd-half questions without conditions) and its
marker cosine is HIGHER than the present class's (0.484 vs 0.421 on the calibration half), so the midpoint falls above most present scores — a
calibration that cannot be set on so few examples, not a tuning choice; for slot 1 the MARKER does not carry the presence signal at all (its
cosine is higher on the absent class), the filler cleanup does — so the 0.330 is not a calibration accident but the absence of a marker signal on
that slot (judgment d8f8a98 Q1). Diagnostic (judgment 45ccd1d C): the drop follows the SLOT INDEX more than
the number of conditions — slot 3+ reads 0.129 while 3-condition questions average 0.664 over their three slots (slot 1 stays near 1.0 inside them,
inferred from the two tables) — so the parameter is the slot's own role R_COND[k] with few training examples (slot 3+ is 295 of 16,840 report
slots), not superposition of the conditions; a capacity effect would lower all slots of a 3-condition question together. The condition axis of
the verifier goes to the HYPOTHESIS list with this diagnostic.

## 13. B6 — sparse block codes and phase quantization: how many bits does a vector need? (rescore_v16_b6.py; logs/2026-10-17_1037_rescore_v16_b6_3d395ff.txt; the saved 10b-quinquies models, 5 seeds, D = 1024; no training; one format changed per variant)
Criterion written before the run: a format is RECOMMENDED when every reference axis stays within 0.02 of the dense value on the 5-seed mean; the
recommendation is the smallest L / the sparsest k meeting it. The dense variant is within seed noise of 10b-quinquies (unseen transitions 0.630 vs
0.627, depth 3 0.977 vs 0.968) but not digit-identical: the world's own rng was consumed differently (the calibration loop stops early), so the
evaluation events differ — same distribution, stated.
| format (bits per coordinate) | seen | new ids | unseen transitions | depth 2 | depth 3 UNSEEN | occupancy known / unknown / absent | within 0.02 of dense? |
|---|---|---|---|---|---|---|---|
| dense (64: complex64) | 1.000 | 1.000 | 0.630±0.037 | 1.000 | 0.977±0.018 | 1.000 / 1.000 / 1.000 | — |
| **routebook phases at 4 bits (L = 16)** | 1.000 | 1.000 | 0.615±0.025 | 1.000 | 0.962±0.025 | 1.000 ×3 | **YES** |
| routebook at 6 / 8 bits | 1.000 | 1.000 | 0.622 / 0.615 | 1.000 | 0.983 / 0.972 | 1.000 | YES / YES |
| **identities + roles (catalog, legend) at 4 bits** | 1.000 | 1.000 | 0.622±0.026 | 1.000 | 0.985±0.010 | 1.000 | **YES** |
| identities + roles at 6 / 8 bits | 1.000 | 1.000 | 0.628 / 0.628 | 1.000 | 0.968 / 0.968 | 1.000 | YES / YES |
| state as unit PHASORS at 4 / 6 / 8 bits (magnitudes dropped) | 1.000 | 1.000 | 0.575 / 0.568 / 0.580 | 1.000 | 0.948 / 0.953 / 0.950 | 1.000 | **no** (unseen transitions −0.05, depth 3 −0.03) |
| state block-sparse, top-16 / 32 (50%) | 1.000 | 1.000 | 0.637±0.034 | 1.000 | 0.978±0.011 | 1.000 | YES |
| **state block-sparse, top-8 / 32 (25%)** | 1.000 | 1.000 | 0.633±0.022 | 1.000 | 0.980±0.019 | 1.000 | **YES** |
| state block-sparse, top-4 / 32 (12.5%) | 0.999 | 1.000 | 0.622±0.026 | 0.998 | 0.958±0.042 | 1.000 | YES (depth 3 −0.019, at the edge) |
Capacity curve (synthetic, D = 1024, 1,000 candidate fillers, 200 trials; cleanup accuracy per filler vs the number of bundled role-filler
terms): n = 4 / 8 / 16 / 32 — 1.000 for every format (top-8/32 at n = 32: 0.997); n = 64 — dense 0.989, phasor L = 16 / 64 0.950 / 0.953, block
top-16 0.969, top-8 0.850.
Reading, by the criterion: (i) the LEARNED PHASES (the route book) and the GIVEN STARS (catalog, legend) survive 4-bit quantization with no axis
moving more than 0.02 — the tool's routebook.npz (11.9 MB at float32 phases for the reference world, 11-bis / C4) can be stored at 4 bits per
phase, an 8-fold reduction, with the catalog regenerated from seeds as before; (ii) the CHART must keep its magnitudes: the unit-phasor projection
loses 0.05 on unseen transitions and 0.03 at depth 3 at every L, because a bundle's magnitudes carry the vote counts of its terms; (iii) block
sparsity of the chart to 25% (top-8 of 32) costs nothing on the reference axes and 12.5% is at the edge (depth 3 −0.019) — a sparse block code
for the chart at a quarter of its coordinates is the recommendation; (iv) capacity is the standard HDC curve: 32 terms clean at D = 1024 in every
format, 64 terms start to fail (dense 0.989) and sparsity accelerates the failure (top-8: 0.850). Nothing in the reader changes; this is a storage
and transmission result (RESULTS 11-bis sizes; the tool's model and the wire). B6 is BUILT with its recommendation; the capacity at 64-1000
identities and D = 4096-16384 (00_READ_FIRST §7 item 6) stays PLANNED.

## 14. B7 — the fourth address code remeasured: the round-1 POSITION code added to the bundle (v16_reference.py --rel_gates open --codes path,chainL,bigram,pos; logs/2026-10-17_1040_v16_reference_60fd964.txt; 5 seeds, same seeds, regime and axes as 10b-quinquies)
Code (round 1, RESULTS 1 / 8d history; brought into V8 as an optional code): term_i = exp(i (pos[i] + pc[c_i])) (x) f_i — one learned displacement
per ABSOLUTE position (positions 0-39) plus one per category, added to the path + chainL + bigram bundle; nothing given beyond the relation-marker
gates of 10b-quinquies. Decision rule written before the run (judgment consolidated Part IV): keep only if the new-construction axis (unseen
transitions FULL) does not drop by more than 0.02 against 10b-quinquies and no other axis loses.
| axis (5 seeds) | 10b-quinquies (3 codes) | **+ position code (4 codes)** | change |
|---|---|---|---|
| seen FULL / predicate / new ids / role cues / depth 2 | 1.000 | 1.000 | — |
| unseen transitions FULL (new construction) | 0.627±0.024 | **0.678**±0.007 (0.675, 0.675, 0.675, 0.675, 0.692) | +0.051 |
| depth 3 UNSEEN FULL | 0.968±0.020 | **0.998**±0.003 (1.000, 0.992, 1.000, 1.000, 1.000) | +0.030 |
| occupancy known / unknown / absent | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | — |
Gates learned: that 1.00 (to, by, from given 1.00), '.' 0.48, who 0.31; participles and verbs closed (0.00-0.04).
Reading, by the decision rule: **KEPT** — the fourth code does not reduce the new-construction axis; it raises it by 0.05 (five seeds at 0.675-0.692,
against 0.627±0.024) and takes the unseen depth 3 to 0.998 (one seed at 0.992). The positions beyond the training lengths (depth-3 sentences reach
positions never seen) carry untrained displacements, so the position term is noise there and the other three codes decide — the gain at depth 3
comes from the shorter forms — a HYPOTHESIS written as a reading, marked as such until the trace of RESULTS 14-bis (judgment d8f8a98 B1).
Recording (rule 16; corrected by judgment d8f8a98 Q3): B7 is KEPT as a measured VARIANT available by flag; the change of the REFERENCE bundle is
a separate decision, pending the trace (14-bis), the out-of-range rule and the terms-per-sentence table; 10b-quinquies stays the reference
meanwhile; the tool's `sop train` should expose the code list so the reference world's model carries it
(engineering, for the agent; not done in this run).

### 11-quinquies. Tool round 4 — presence thresholds calibrated with held-out FORMULATIONS (one change; tool/tests/gpu_round4.sh at commit 39063ed; tool/runs/round4_39063ed/ with round4_table.md; 5 seeds; reference world, to / by / from given open; judgment 45ccd1d D)
Expectation written before the run (round4.log line 2): the recipient's presence on unseen formulations recovers toward the every-role-read value
without reading every role. Each held-out-formulation split cut in two halves by hash before training; the calibration half joins the training
sample for the midpoint thresholds; the other half (87 rows) is reported, with the round-3 (training-rows-only) thresholds scored on the same rows.
| rows | readout (protocol) | thresholds set on | heads FULL | agent | object | recipient |
|---|---|---|---|---|---|---|
| round 3 D, 180 | product: presence decided | training rows | 0.428±0.047 | 0.833 | 0.811 | 0.619±0.047 |
| round 3 D, 180 | v16 protocol: every role read (not available to the product) | — | 0.596±0.048 | 0.833 | 0.811 | 0.839±0.004 |
| **round 4 F, 87** | **product: presence decided** | **training rows + the calibration half of the held-out formulations** | 0.418±0.061 | 0.839 | 0.779 | **0.630**±0.037 |
| round 4 F, same 87 | presence decided | training rows only (round-3 product) | 0.418±0.061 | 0.839 | 0.779 | 0.625±0.042 |
| round 4 F, same 87 | v16 protocol: every role read | — | 0.556±0.061 | 0.839 | 0.779 | 0.809±0.006 |
Reading: NOT met — on identical rows the held-out calibration moves the recipient by +0.005. The midpoint rule barely moves (seed 0: 0.636 -> 0.622;
new thresholds 0.61-0.70) because the 156 calibration-half sentences sit among about 480 present training rows whose present mean (1.11-1.20) and
absent mean (0.11-0.20) are far apart; about one recipient in five on unseen formulations has a presence ratio below a threshold that is already
far below the training present mean. No cost elsewhere (held-out sentences 1.000, depth 3 0.987, new identities 1.000). Per the judgment's rule:
**presence calibration across formulations is a recorded limit of the READER, not of the tool** — the product reads the reference world's unseen
formulations at 0.418-0.428 with presence decided and 0.556-0.596 under v16's every-role-read protocol. Reservation stated by the agent and left
open: this run tested the pooled-midpoint rule only, not a threshold fitted on the held-out formulations alone, and the per-sentence presence
ratios of the reported rows were not stored (the run keeps no models), so whether those ratios overlap the absent class is unknown.
