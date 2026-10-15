# Role emergence in a displacement-addressed vector-symbolic sequence model: composition to an unseen depth, emergent gates, and the limit of causal addressing

Principle note, full-prose draft v1 (2026-10-15). Scope: the synthetic reference world only. WikiSQL is NOT part of this note (one sentence
in future work); it belongs to the empirical paper, after its baselines. Every number cites RESULTS.md and a log with a commit hash.
Figure 1 to be drawn (three panels, section 2). The caveats of section 8 are part of the claims, not an appendix.

## Abstract
We study a sequence model in which a word is a pair: a fixed random identity vector and a displacement computed from context by codes
learned per category. The state of a sentence is the bundle of identities bound to the addresses reached by integrating displacements
along the sentence (a vector-symbolic architecture with fractional power encoding: complex unit vectors, bind = phase addition, bundle =
sum). The only training signal is generative: the state must match, under cosine, the state of a world with explicit roles, built from
the same identities. We report three results on a synthetic English world with reported-speech nesting. (1) Trained on nesting depths
0-2, the model decodes every role at depth 3 — a construction it has never seen — at 1.000 on all six roles over 5 seeds; the learned
displacements compose one level further than trained because the world's algebra (a fixed role vector applied once per clause) and the
model's (displacements summed along the chain) coincide. (2) The gates — which categories emit expectations, i.e. open a frame — are
learned by gradient from the same objective and converge, identically in 5 seeds, to: identities closed, relation markers open, verbs
closed; the hand-set linguistic prior (verbs + prepositions) was not the optimum. (3) A small transformer tuned on the same task, with the
same fixed identities and objective, fails on exactly the two axes where the construction gives the model its generalization: new
identities (0.22, chance 0.17) and the unseen depth (0.39, inner roles at chance); a linear associative map of any rank fails the same way.
We also state a limit with its demonstration — structure announced after its scope (postposed reporting) cannot be addressed by any
causal code — and we retract a claim we had made, that categories emerge, since a skip-gram on the same sentences recovers them equally.

## 1. Setting
**Space and operations.** D = 1024 complex unit vectors (phases). bind = elementwise product = phase addition = a rigid displacement;
bundle = sum (the only irreversible operation); cleanup = argmax cosine over known candidates. This is the fractional-power-encoding
reading of vector-symbolic architectures (Plate; Kanerva; Gayler; Frady, Kleyko and Sommer).

**Word = (identity, address).** The identity f_w is a fixed random unit vector, one per word type, never learned. The address is computed
from the context by codes whose parameters are per CATEGORY (persons, objects, each function word, each verb form), not per word. The state
is S = sum_k address_k (x) f_{w_k}. We use a bundle of three codes: path (sum of the category displacements along the prefix), bigram
(previous + own category) and chain (expectations: a gated word emits, for each category, an expectation at its own address plus a
relative displacement; a later word of that category takes the most recent unconsumed one; with LEFT re-addressing, a non-emitter is
re-addressed by the first emitter that follows it). Everything is phases; everything is trained by Adam on one loss.

**World.** S_world = R_A f_A + R_B f_B + R_C f_C for an event (agent, theme, recipient); reported speech is a fibre: S_world(X says that P)
= R_X f_X + R_rep (x) S_world(P), with one speaker role R_X and depth = the number of R_rep bindings. The world and the model share the
identities (asserted in code). Role vectors are random and fixed.

**Objective.** Maximize cos(S, S_world). No paraphrase labels, no negatives, no rule for 'that' or for any verb. 30,000 sentences,
batch 8, lr 0.02.

**Evaluation.** Role r at level k is decoded with the WORLD's address: argmax_x cos(S (x) conj(R_rep)^k (x) conj(R_r), f_x), accuracy and
margin (cos of the correct candidate minus the best wrong one), chance printed. Axes kept separate: seen identities; new identities (random
vectors never in training); new formulations whose role cues (the categories before and after each role) were all seen with the same role;
new formulations with unseen transitions; depth 1; depth 2 (seen); depth 3 (unseen). Five seeds, each seed a new world. Compared variants
share the optimization regime. A new depth must not introduce a new role (the compositional world guarantees it). Per-role breakdown
precedes any interpretation of a bundle. (EVALUATION_PRINCIPLES.md, rules 1-16.)

**Synthetic English world.** Six frames (give, sell, send, lend, show, offer) x 16 formulations each — active, passive, dative shift,
clefts, topicalization, wh-constructions, comma-separated adjuncts; 18 persons and 18 objects (12 of each in training); reporting 'X says
that <P>' / 'X claims that <P>', nested 0-3 levels, training on 0-2.

## 2. Figure 1 (to draw)
Panel (a): the uniform step — two formulations of one event drawn as two paths on a grid that touch the same (address, identity) pairs
in a different order. Panel (b): the chain with learned gates — the sentence as a skeleton of relation markers with content words hung to
the left of the next marker (section 4, trace). Panel (c): the objective — S must match R_rep-nested role states; the same event at depths
0, 1, 2 and the unseen 3 as nested copies of the space (fibres).

## 3. Pillar 1 — composition to an unseen depth
RESULTS 6c, logs/2026-10-15_1319_v7_nested_e1e7011.txt. Trained on depths 0-2, tested on depth 3, five seeds, two bundles:

| code | depth 1 | depth 2 (seen) | depth 3 (UNSEEN) | per role at depth 3 |
|---|---|---|---|---|
| path+chain+bigram | 0.982 | 0.997 | 0.979 | A 0.999 B 1.000 C 0.976 X1 1.000 X2 1.000 X3 0.898 |
| path+chainL+bigram | 1.000 | 1.000 | **1.000** (margin 0.23) | all six 1.000 |

New identities decode at 1.000 at every depth (they never enter the parameters). The generalization is not statistical. Write a =
says:that, b = claims:that, c = that:says, d = that:claims, g = that:gives, rho = the phase of R_rep. Depth-0 and depth-1 agents give
a + g = rho and b + g = rho; depth-2 agents give d + b = rho and c + a = rho; the inner speaker at depth 2 gives a + d = rho; all are
consistent, and the depth-3 agent then sums to 3 rho by the same identities. Training finds that solution (per-role 1.000 at depth 3 with
the left re-addressing; without it the third speaker is at 0.90 and the recipient at 0.976). The world's algebra and the model's coincide,
so composition one level further holds by construction once the parameters land on the consistent solution.

## 4. Pillar 2 — the gates emerge from the same objective
RESULTS 7c, logs/2026-10-15_1412_v8_gates_06e12bd.txt. A gate g_c = sigmoid(s_c) is the probability that category c emits expectations.
The chain is made differentiable in the gates: a word's right address is the gate-weighted mixture over "the most recent open emitter
before it" (p(q,k) = g_q prod_{q<i<k}(1-g_i)) and the root; a non-emitter's left address is the mixture over "the first open emitter after
it"; the final address is g_k times the right one plus (1-g_k) times the left one. Gate logits are trained with the phases by the same
Adam on the same loss. 32 categories, five seeds, two starts (all closed, all open), with and without the path and bigram codes.

The learned set is IDENTICAL in every seed: identities P and O closed; the relation markers to, from, by, that, the comma and the copula
open; all 18 verb forms closed (sigmoid <= 0.03). From the open start the reporting verbs 'says' and 'claims' stay open as well — they are
harmless: 'that' already carries the clause. Nothing is lost at depths 1-3 (1.000 / 1.000 / 0.996-1.000), and on new formulations the
learned gates beat the hand set (bundle: unseen transitions 0.76 -> 0.87; chain alone: role cues 0.22 -> 1.00) — fewer emitters means
fewer untrained transitions.

Where the gradient and our earlier discrete search disagree, and why. The search (round 2, no left expectations) had opened the verbs,
because only emitters could place the words after them. With left re-addressing available, a verb has nothing to emit that 'to', 'by',
'that' and the left slot do not already provide. The hand set "verbs + prepositions open" was a linguistic prior; the emergent partition
is identities / relation markers / action words, and verbs behave as identities of actions, not as operators. We keep the hand set as
an ablation row.

**Trace** (logs/trace_gates_hand_vs_learned.txt; the address of each word as the list of its parameters; no training involved):

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

With the learned gates the sentence becomes a skeleton of relation markers; every content word, the verb included, hangs to the left of
the next marker: "the person before 'to'" is the agent, "the thing before 'to'" the theme, "the person before '.' after 'to'" the
recipient; addresses have at most three terms instead of five to seven, and the nesting depth is carried by 'that:that'. In the passive
'o0 is given to p1 by p0 .' the agent is 'to:by + by:. + L[.:P]': a different address for the same role, which the objective maps to R_A.

## 5. The control that fails on exactly those two axes
RESULTS 7b, logs/2026-10-15_1414_v9_transformer_75e7e0d.txt. A small transformer (0.79M parameters: linear 2D -> 128, sinusoidal
positions, two pre-norm encoder layers with four heads, masked mean pooling, linear 128 -> 2D, complex output) is trained on the same
30,000 sentences (12 epochs), with the same fixed identities at the input, the same -cos objective against the same world, and decoded by
the same decoder. Learning rate and dropout are chosen on seed 0 by validation on fresh training-distribution sentences, never on a test
axis (lr 1e-3, dropout 0). Five seeds.

| | seen | new identities | role cues seen | depth 2 (seen) | depth 3 (UNSEEN) |
|---|---|---|---|---|---|
| transformer | 0.971 | **0.215** (chance 0.167) | 0.968 | 0.797 (A 0.49, C 0.50; B, X1, X2 1.00) | **0.390** (A, B, C at chance; X1, X2 1.00; X3 chance) |
| this model | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

On the flat axes the two are comparable (0.77 vs 0.79 on the old "seen transitions" axis). The transformer cannot use an input vector it
has never seen, and at the unseen depth it places the two speakers it knows and puts everything else at chance: it reproduces the deepest
configuration seen rather than composing a third. A linear associative map S = U V x on an order-aware structure-free input (fixed random
position vectors bound to identities), at every rank from 8 to 1024, shows the same signature (new identities 0.24, depth 3 0.25-0.39,
RESULTS 8a); capacity is not what is missing. We state this as "at this data regime, a small model trained identically does not
generalize to new identities or to a deeper construction", not as "transformers cannot".

## 6. The limit, with its demonstration
RESULTS 6b and 6c. (a) A FLAT world — depth-indexed speaker roles X1, X2, X3 with depth-invariant A, B, C — is contradictory for any
cumulative address scheme: depth-invariance of A forces a + g = 0, b + g = 0, d + b = 0, c + a = 0, hence X2 - X1 = a + d = 0. Measured
on that world, the inner speaker sits at 0.5-0.77 (argmax softens the contradiction). This was our world design, not the model; the
compositional world of section 1 removes it. (b) Structure announced AFTER its scope cannot be addressed causally: in the postposed
'<P> , says X' the clause is already written at depth 0 when the marker arrives; likewise 'from A , C receives B' puts the recipient
after a comma that in every training formulation preceded an agent. With the postposed form in the data, depth 2 falls to 0.88 and depth 3
to 0.54. This is the precise, data-forced form of "deferred addressing is necessary": a displacement applied retroactively to a whole
clause, and only for right-announced structure. We declare it as a limit here; the remedy (clause-level re-addressing) is future work.

## 7. What was retracted, and what each retraction taught
Round 1 argued an "algebraic impossibility" of per-word displacement and hand-coded two triggers (the verb writes a buffer; 'that' opens a
frame). Round 2 retracted both: they were artefacts of decoding with the model's own addresses and of a world/model identity mismatch;
argmax decoding tolerates overlapping addresses. Round 5 found nesting of depth 2 "unsolvable from the prefix"; round 6 sharpened it into
section 6: the flat world was contradictory (our construction), the real limit is right-announced structure. Round 8 retracted "categories
emerge": the untied learned displacements do cluster the 54 content words into the five true classes (purity, ARI and nearest-neighbour
category all 1.000, five seeds) — and so does a skip-gram trained on the same sentences (1.000). Categories are recoverable from
distribution; the model consumes them as input, it does not produce them; what emerges from the objective are the gates. Each retraction
became a rule (EVALUATION_PRINCIPLES 1-16) that the next round had to pass; we report them because they are what makes the remaining
claims credible.

## 8. Caveats (part of the claims)
The world is ours: synthetic, six frames, two reporting verbs, 36 identities, D = 1024; the positive nesting result holds for reporting
announced before its clause. The control is small (0.79M parameters) and minimally tuned; larger models, more data and learned embeddings
(which would remove the fixed-identity constraint and with it the zero-shot test) are not covered. The old "seen transitions" axis measures
the redundancy of the formulation set, not the mechanism; we replaced it by "role cues seen" and keep it for comparison. Capacity at
scale (thousands of identities, D up to 16k) is not measured. Nothing here is a claim about natural text; a first experiment on text with an
executable consequence belongs to a separate, empirical paper and is not used here.

## 9. Related work (verify each before citing)
Plate 1995/2003 (holographic reduced representations); Kanerva 2009 (hyperdimensional computing); Gayler 2003 (VSA); Smolensky 1990
(tensor product representations); Eliasmith 2012 (semantic pointer architecture); Frady, Kleyko, Sommer 2018-2022 (fractional power
encoding; resonator networks; grid cells as FPE); Kleyko et al. 2022 (survey); Hawkins et al. 2017/2019, Lewis et al. 2019 (Thousand
Brains; displacement cells); Elman 1990 (sequences as trajectories); Kamp 1981, Heim 1982 (dynamic semantics); Fillmore 1982 (frame
semantics); Katharopoulos et al. 2020 (linear attention as a bundle of binds); Schlag and Schmidhuber 2018 (fast weights, TPR); Lake and
Baroni 2018 (SCAN, compositional generalization).

## 10. Future work (one sentence each)
Clause-level retroactive re-addressing for right-announced structure; capacity at scale with sparse block codes; real text with an
executable consequence (text-to-SQL), reported separately with its own baselines.
