# Validation plan on the DGX Spark

THE VALID PLAN is the one in 00_READ_FIRST.md §5-6 (order of experiments, criteria, day-by-day steps). This file keeps only:
the port to torch and the data sources. The sections marked SUPERSEDED are the plan from after round 1 and contradict round 2 (deferred addressing and
zoom-in are not necessary; the gates are learned); they remain as history, they are not to be followed.

## Port to torch (first day)
Everything is numpy with hand-written gradients so that it runs here without torch. In torch:
  theta = nn.Parameter(...); m = torch.polar(torch.ones_like(theta), theta)   # displacements as complex units
  L = torch.cumprod(m[cats], dim=0); S = (L * F[ids]).sum(0)                  # the uniform step, on GPU, in batch
  loss = -cosine(S * T, S_lume)                                                # autograd does the rest
Batch: group the sentences by length (or pad + mask). D=4096-16384 fits easily in unified memory.

## Data sources with equivalence "given by a world" (all public)
- WikiSQL: https://raw.githubusercontent.com/salesforce/WikiSQL/master/data.tar.bz2 (26 MB; jsonl + sqlite)
- Spider (text->SQL nested): https://github.com/taoyds/spider (link to Google Drive in the README)
- CoSQL / SParC: conversations -> SQL (yale-lily)
- Code + tests: APPS, CodeContests, MBPP (HuggingFace datasets); equivalence = the same tests passed
- Proofs: Lean mathlib (github.com/leanprover-community/mathlib4), miniF2F; derivation lines = equivalent by construction
- Instructions with state: ProPara (github.com/allenai/propara), ALFRED (askforalfred.com), TextWorld (Microsoft)
- Multiple captions per image: COCO captions (images.cocodataset.org/annotations/annotations_trainval2017.zip), Flickr30k (Kaggle)
- Parallel translations: OPUS (opus.nlpl.eu), Tatoeba (tatoeba.org/downloads) — adds hard negatives automatically
- Paraphrases with ready-made hard negatives: PAWS (github.com/google-research-datasets/paws), PAWS-X (6 languages, including ... not Romanian; but OPUS has it)
- Reformulation markers from textbooks/Wikipedia ("adica", "cu alte cuvinte", "i.e."): own extraction, large volume, large noise


## SUPERSEDED (plan from after round 1; kept as history, not to be followed — see 00_READ_FIRST.md §5-6)
> ## What is in the zip
> - camp.py, k.py: the complete synthetic model (uniform step, deferred addressing, zoom-in, generative learning) + evaluation on roles
> - control_transformer.py: the control (numpy transformer) on the same task
> - sick_test.py, wikisql_test.py: the first tests on real text (negative on SICK, modestly positive on WikiSQL)
> - RESULTS.md: all numbers and conclusions; log_*.txt: raw outputs; corp/: SICK (small) — WikiSQL is downloaded (see below)
>
> ## The experiments, in order of value (each with its "works / does not work" criterion)
> 1. Full WikiSQL (56k; 15k test) with: discovered categories (iterative clustering as in i2.py), deferred addressing with a learned
>    trigger (see 3), D=8192. Works if: selected col. > 0.7 and aggregator > 0.85 WITHOUT seeing the table; and same-skeleton pairs
>    separated from different-skeleton by > 0.3 cosine. Does not work if it stays below bag of words + logistic regression on the same tokens (make it the baseline!).
> 2. Spider (github.com/taoyds/spider; harder, nested SQL): tests the zoom-in (k.py) on sub-queries.
> 3. Learning the triggers: instead of "the verb writes the buffer", learn a score per word/category that decides the flush (sigmoid, gradient through
>    a weighted bundle: S += sigma(s_w) * deferred_term + (1 - sigma(s_w)) * immediate_term). Works if the learned score aligns with the verbs/
>    linking words without having been told.
> 4. Equivalence with hard negatives: pairs (question, question with roles/values swapped) built automatically from the SQL; report separately
>    the accuracy on these pairs, not just the global correlation (SICK showed that the mean hides exactly what matters).
> 5. Strong control: a small transformer in torch (or a small BERT) on the same task and the same data; compare at little data (1k, 5k, 56k).
> 6. Capacity: sparse block codes (Laiho/Frady/Kleyko) instead of dense phases, for a vocabulary of 50k+.
>
> ## What to expect, realistically
> - Not a model that beats LLMs on text->SQL (those are at ~90%; here, without the table and with 10k parameters, I have 0.49 after 70 s of CPU).
> - A layer of readable semantic state (who-what-to-whom), trainable with little data, which generalizes to new words without retraining
>   and which can VERIFY the equivalence of two formulations role by role. Useful as a verifier/interpreter next to an LLM, in closed domains.
> - The most valuable scientific result possible: if the triggers (3) learn themselves on WikiSQL/Spider, then the whole mechanism
>   (categories, frames, roles, hierarchy) is emergent from a single generative-prediction objective. That would be worth publishing.
>
> ## After round 2: what to run first on the DGX
> 0. Reproduce the main table from RESULTS.md "ROUND 2" with 5 seeds (v2.py, v3.py), then gates.py with large N.
> 1. Generalization to new CONSTRUCTIONS (the real barrier): world with 10-20 formulations; hold out formulations whose transitions have all been seen
>    separately vs formulations with new transitions. Test REDUNDANT codes for addresses (bundle of several paths / blocks), so that an untrained
>    transition does not randomize the whole address.
> 2. Gates learned by gradient (sigmoid + soft mix between the provisional and the expected address) instead of discrete search; check
>    whether they converge to the same set (identities closed, verbs/prepositions open).
> 3. WikiSQL with v3 (dual) + learned gates + discovered categories; report the margin, not just the accuracy.
