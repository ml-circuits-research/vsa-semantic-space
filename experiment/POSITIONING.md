# Positioning, presenting the architecture, controls, credibility

## The architecture's name and the one-line sentence
Proposed name: "Path-Integrated Semantic State" (PISS is unfortunate; alternatively "Displacement Semantics" or "Role-Emergent VSA" / REVSA).
Sentence: *A word is a pair (fixed identity, learned displacement per category); the state of a sentence is the bundle of the identities
bound to the locations reached by integrating the displacements; training it to predict generatively, through a fixed operation, the state of the consequence makes
different formulations converge to the same state, role by role, and syntactic categories emerge without supervision.*

## How the architecture is presented (figure 1)
Three panels, not one:
 (a) the uniform step: L <- L * m_cat(w); S <- S + L * f_w, with an active and a passive sentence drawn as two paths on a grid that
     touch the same (address, identity) in a different order;
 (b) REDUNDANT address codes (path, bigram, chain, pos) in a bundle: an untrained category transition does not randomize the whole address
     (round 3: 0.60 -> 0.90 on new constructions). [RETRACTED: "deferred addressing" (buffer + zoom-in at "ca") from round 1; refuted in round 2.]
 (c) the objective: S * T ≈ S_world, with T fixed, versus associative memory (which does NOT align). This panel is the key of the argument.
Table 1 = the negative results (A, C, D, E2) with the algebraic explanation of each. It is unusual, but it is what gives credibility:
it shows that the ingredients are not chosen arbitrarily, but forced.

## Three possible articles, from the safest to the most ambitious
1. **Principle note** (neuro-symbolic workshop at NeurIPS/ICLR, or Neural Computation / Frontiers in Comp. Neurosci.):
   "Necessary conditions for role emergence in vector-symbolic sequence models". Contributions: (i) overlapping addresses + argmax decoding:
   role alignment does not require exact equality of addresses [RETRACTED: "the algebraic impossibility of per-word displacement" — correct only for
   exact equality, irrelevant with argmax decoding; see round 2]; (ii) associative vs generative (confirmed against a bottleneck associative map, round 8a); (iii) [RETRACTED round 8a: emergence of categories — skip-gram on the same sentences recovers them equally; kept as a consistency check]; (iii-bis) composition to an unseen nesting depth (round 6c);
   (iv) the gates (who opens a frame) are learned by search on the prediction criterion; (v) zero-shot generalization to new identities vs
   small transformer.
   Minimum requirements: 5 seeds, shared F, world with 4-6 frames and 8+ formulations, the word2vec control for categories, the "associative memory
   with a bottleneck" control. Achievable in 2-3 weeks on the DGX. Credibility: good, if the claims stay at the synthetic level.
2. **Empirical article** (ACL/EMNLP findings, or CoNLL): text->SQL with readable state. Contributions: data efficiency (curves 1k/5k/56k),
   role-by-role interpretability, new vocabulary without retraining. MANDATORY controls: BoW + logistic regression; small tuned
   transformer; a fine-tuned BERT-small; all on the same data. Achievable in 1-2 months. Credibility: depends entirely on the controls.
3. **Theory/neuro article** (link to TBT / grid cells / predictive coding): only if the triggers learn themselves
   (experiment 3 from README_DGX). Without that, the link to the cortex stays in the discussion section of article 1, not in the title.

## Where the barrier is, honestly
- On text without a world (SICK) we lost to the bag of words. Do not publish anything on general text before having a set where equivalence
  is given by execution/image and where hard negatives (inverted roles) are measured separately.
- [RETRACTED after round 2, PARTIALLY REOPENED in round 5] The hand-set triggers (verb -> flush, "ca" -> frame) were not necessary for
  the seen formulations and depth-1 nesting. BUT at depth >= 2 two sentences with an identical prefix require different roles for the same
  word; no address from the prefix can separate them (RESULTS.md round 5, diag. 2). So A form of deferred addressing is necessary — not
  "the verb writes the buffer", but "a word can be re-addressed by what follows". The correct formulation for the article: the deferral is forced by the data
  (identical prefixes), not chosen; and it must be shown that it is learned (00_READ_FIRST.md §5.1b, §5.2).
- Capacity: 64 identities -> 0.66 at D=1024. Standard for HDC; solved with large D and sparse codes, but it must be shown.
- The transformer control is weak (untuned). Until it is tuned, the permitted formulation is "at this data regime, a small transformer trained
  identically does not generalize", not "transformers cannot".

## Which comparisons are done / planned
Done: small transformer (numpy, 854k params, gradient-check) on synthetic; bag of words and Jaccard on SICK; bag of words and position code
on WikiSQL. Planned (README_DGX): BoW+LR, tuned transformer, BERT-small, word2vec for categories, associative memory with a bottleneck,
data curves, 5 seeds. A simple MLP (bundle -> MLP -> state) is worth adding as the "no order, with capacity" baseline.

## What credibility should NOT rest on
Not on the analogy with the brain, not on the small number of parameters in itself (HDC is always small), not on a single benchmark. On: explained negative
results, ablations (each ingredient removed in turn: factorization, deferral, fixed operation), and controls on the same data.

## Works to cite (check each before citing; the list is from memory)
Plate 1995/2003 (HRR); Kanerva 2009 (HDC); Gayler 2003 (VSA); Smolensky 1990 (tensor product); Eliasmith 2012 (SPA);
Frady, Kleyko, Sommer 2018-2022 (fractional power encoding; resonator networks; grid cells as FPE);
Kleyko et al. 2022 (HDC/VSA survey); Hawkins et al. 2017/2019, Lewis et al. 2019 (Thousand Brains, displacement cells);
Rao & Ballard 1999, Friston 2010 (predictive coding); Elman 1990 (RNN as trajectory); Kamp 1981, Heim 1982, Veltman 1996 (dynamic semantics);
Fillmore 1982 (frame semantics); Hasson et al. 2008 (temporal receptive windows); Katharopoulos et al. 2020 (linear attention = bundle of binds);
Schlag & Schmidhuber 2018 (fast weights, TPR); Lake & Baroni 2018 (SCAN, compositional generalization); Zhong et al. 2017 (WikiSQL); Marelli et al. 2014 (SICK).
