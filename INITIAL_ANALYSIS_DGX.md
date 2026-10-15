# Analysis of `experiment/` — what it is, what is established, what is unconfirmed, discrepancies

## What the project is

Starting hypothesis: understanding a sentence means traversing a path in a space, and two sentences with the same
meaning are different paths that reach the same point. The space is built with VSA/FPE (complex vectors of
modulus 1, D = 512-1024). Three operations:
- bind: a displacement (element-wise multiplication = addition of phases);
- bundle: a summation that melts the terms into a single one (the only irreversible operation);
- cleanup: choosing the nearest known concept (argmax on cosine).

The current model (v4.py): each word has a fixed, random identity, plus an address computed from context.
The address is a bundle of 4 codes learned per CATEGORY, not per word:
- path:   sum of the displacements of the categories in the prefix;
- bigram: the previous category + its own;
- chain:  the chain of expectations;
- pos:    absolute position (x) category.

Sentence state: S = sum address (x) f_w. Single, generative objective: cos(S, S_world), where the world is given by explicit
roles and uses the SAME identities as the model. The evaluation decodes the roles with the world's addresses and
reports accuracy + margin.

## What is established (3 rounds, numpy, CPU)

- Word factorization (identity + displacement per category) and the generative objective are necessary.
  Associative prediction and the variant with both sides learned collapse.
- On seen formulations the roles align perfectly. On new identities it gives 1.00, but that follows from
  construction (the identity does not enter the parameters).
- The nesting "X spune ca ..." works WITHOUT any hand-set rule. The major correction from round 2: the rules
  "the verb empties the buffer" and "zoom-in at 'ca'" from round 1 came from two evaluation errors
  (addresses from the active form; different F world/model), they were not necessary.
- The gates (which categories open a frame) can be learned by discrete search: verbs and prepositions
  open, persons and objects do not.
- The real barrier is new CONSTRUCTIONS (unseen category transitions). The redundant bundle of codes
  raises the accuracy on them from 0.60 to 0.90.

## What is still unconfirmed

- Nesting on the dual has 1 seed; "path alone" has 2. The categories on SICK and the learned gates were seen
  on a single world.
- The key controls are missing: word2vec for categories, associative memory with a bottleneck, TUNED
  transformer (the current one is untuned, 854k params).
- WikiSQL: 0.49 a single time, without the BoW+LR baseline. SICK: negative (0.45 vs 0.575 Jaccard).
- Detail from v4: on new formulations with SEEN transitions, "all 4" gives 0.94, below bigram alone (1.00).
  Redundancy helps where the transitions are new and costs something where they are seen. To be noted in RESULTS.md.

## Discrepancies in the documents

- POSITIONING.md still proposes the "algebraic impossibility" as the first contribution of article 1 and
  "deferred addressing" (buffer + zoom-in) in panel (b) of figure 1. Both were refuted in round 2.
- README_DGX.md mixes the old plan (from after round 1: "deferred addressing with a learned trigger", k.py
  zoom-in on Spider) with the one from after round 2 (the section at the end). The first sections are superseded.
- The logs in logs/ do not have the run.sh format (the commit hash is missing from the name): they were produced before
  run.sh existed.
- README_RUN.md calls camp.py "THE FINAL MODEL" and lists "deferred addressing, zoom-in" as parts of it;
  00_READ_FIRST.md says the current model is v4.py. 00_READ_FIRST has priority.

## Working environment

- On the machine neither numpy nor torch is installed. Python 3.12.3. GPU: NVIDIA GB10 (DGX Spark, aarch64).
- A venv with numpy and torch with CUDA for aarch64 is needed.

## What is to be done (the plan from 00_READ_FIRST.md §5-6)

1. Environment: venv + numpy + torch (CUDA aarch64).
2. Day 1: port v4.py to torch (phases as nn.Parameter, torch.polar, autograd instead of the manual gradient,
   batch by length). Check that seed 0 reproduces the round 3 table within +-0.03. Then 5 seeds on a world
   2-3x larger (4-6 frames, 10-20 formulations, 3-4 nesting levels).
3. Day 2: gates learned by gradient (sigmoid + soft mix), on >=3 worlds. Then the word2vec control and
   the associative memory with a bottleneck.
4. Further: tuned transformer + data curves 1k/5k/50k; capacity (D 4096-16384, sparse block
   codes); WikiSQL with full v4 + BoW+LR baseline; then Spider.
5. Discipline: commit before every run -> ./run.sh -> entry in JOURNAL.md -> table in RESULTS.md.
   The evaluation follows the checklist in EVALUATION_PRINCIPLES.md.

Proposed start: steps 1-2 (environment, v4 port, seed 0 reproduction). The corrections to POSITIONING.md and
README_DGX.md separately, in a small commit.
