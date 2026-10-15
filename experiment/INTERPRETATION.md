# How to read the results (and what would refute them)

## The metrics, one by one

**Role decoding (camp.py, k.py, control_transformer.py).**
Take the state S of an arbitrary formulation (passive, "primeste de la", with the dative in front...). Unbind it with the role's ADDRESS
taken from the ACTIVE FORM (template 0) and do cleanup among the candidates (persons or objects). Correct = the real identity comes out.
- Why this way: it measures directly the thesis "different paths, same point": if the agent of the passive is readable at the agent's address from the active,
  the two paths have reached the same place FOR THAT ROLE. It is stricter than the global cosine between states (which can be high
  because of shared words) and more informative (it says WHICH role failed).
- Chance = 1/number of candidates (0.125 at 8, 0.25 at 4, 0.083 at 12). Anything below ~2x chance means "nothing".
- Template 0 always gives 1.00 by construction (the addresses come from there); it is NOT a result, it is the calibration. The figures that matter are T1-T4.
- "Mean over all formulations and roles" includes T0, so it is slightly inflated; it drops by ~0.1 if you exclude T0.

**Cos(pred, S_world) and prediction quality.** How well the slow level generates the consequence state. It is the trained objective; it is not a
measure of equivalence. Experiment C showed it can be perfect (100% recovery) with zero equivalence.

**Similarity paraphrases / swapped roles / random (sim_deplasari.py).** Cosine between final states. The figure alone says nothing:
what matters is the DIFFERENCE paraphrases minus swapped-roles (hard negative) and "paraphrase beats swap" (the fraction of pairs where the paraphrase is
closer than the version with the roles inverted). High random (>0.3) = collapse: the model made everything similar.

**Mean displacement per class** (1 - mean cos of the phases): 1.0 = maximum displacement (random), 0 = identity. Tested H1 (function words
small, verbs large). It was confirmed only weakly; do not insist on it.

**Spearman with the human score (sick_test.py).** Rank correlation between the cosine of the states and the 1-5 relatedness. The decisive baseline is Jaccard
(word overlap): 0.575. Any model below it has lost lexical information without gaining structure. SICK does NOT measure roles
(29 of 4927 pairs have the same words in a different order); a good result on SICK would not have confirmed the thesis, a weak one does not refute it.

**WikiSQL (wikisql_test.py).** Decoding from the prediction: selected column (candidates = the table's columns, chance ~0.15), aggregator
(6 classes, majority 0.72: anything below 0.72 on the aggregator is WORSE than always answering "none"), condition column. The model does not see
the table, so the column is guessed only from the words of the question; this caps the result. "Same skeleton / different skeleton" = cosine
between predictions for questions from the same table with the same (sel, agg, condition columns) vs different; the difference matters, not the values.

## The three results that support the thesis and what would bring them down
1. Factorization is necessary (G3 vs G2: 0.50 vs 0.86; E2 shown algebraically). Brought down by: a model with per-word displacement that reaches
   >0.8 with another learning rule.
2. Generative prediction (fixed operation) produces equivalence, associative does not (C,D vs F,G). Brought down by: an associative memory with a
   bottleneck (limited capacity) that aligns the states equally well. (A useful experiment to do, actually.)
3. Categories emerge (I: exact clusters; SICK: man/person/woman). Brought down by: equally clean clusters obtained from simple co-occurrence
   (word2vec on the same corpus). Do this control; if word2vec gives them just as well, the emergence is not the architecture's merit.

## Known traps in the figures here
- Single seed everywhere. The variation between seeds was NOT measured; on synthetic expect +-0.05, on WikiSQL +-0.02. Run 5 seeds.
- The world and the model must share the same features F (bug found late; experiments G-K ran with different F, the control with shared F).
  The G-K figures are therefore underestimates; re-run them with shared F before citing them.
- The evaluation on synthetic uses the same world in training and test (only the identities/formulations are held out). There is no "test set"
  in the classical sense for frames: both frames were seen.
- The transformer control: no hyperparameter tuning, 3 epoch-equivalents, no dropout/weight decay. A well-tuned transformer at 40k
  examples would generalize better to new formulations (not to new identities, where the learned embedding is structurally disadvantaged; but with
  fixed features at the input, as I gave it, it is not excluded there either with more data).

## Added after round 2
- Decoding with the world's addresses (not the active form's) is the only correct one: the active form's addresses penalize overlapping addresses, which are
  LEGITIMATE (argmax resolves them). All round 1 figures on non-active formulations are underestimates for this reason.
- Report the MARGIN alongside accuracy: 1.00 with margin 0.15 (path) and 1.00 with margin 0.50 (expectations) are very different states; at
  large vocabulary a small margin becomes an error.
- "New formulation" tests untrained transitions; in multiplicative addressing a single untrained phase randomizes the address. Separate
  in the report: new formulations with seen transitions vs with unseen transitions.
