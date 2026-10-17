# REPORT — world `reference`, legend sha256:e221118d4315… (version 2)

Labels (rule 22): every number below is BUILT (code and measurement exist, computed by this training run). Nothing here is PLANNED or HYPOTHESIS except where a line says so.

## Given and learned (rule 20)
GIVEN: the legend's roles and their vectors, the identities (stars regenerated from the seed), the kinds (lexicon), the event synonym table, the reader block of the legend. LEARNED: the route book only — per-kind displacements of the path / bigram / chain codes and the gates (sigmoid per kind, from an all-closed start), by the single objective max cos(S, S_world) with structured per-sentence targets with pre-set roles; the model learns the assignment.

## Data
30660 annotated sentences from 6 files; provenance: {'generator': 30660}. Audit: none in this world folder (audit/ empty) — agreement not measured. Shapes the v1 decoder cannot read (trained as targets, scored as FULL-wrong): none.

## Splits (written to runs/reference_D_presence_not_decided/splits.json before training; source: split_hints.json)
- heldout_depth3: 120 sentences
- heldout_formulations_role_cues_seen: 60 sentences
- heldout_formulations_seen_transitions: 60 sentences
- heldout_formulations_unseen_transitions: 180 sentences
- heldout_sentences: 3072 sentences
- new_identities: 240 sentences
- train: 26928 sentences

The four splits of spec 6.4: seen formulations = `train` (+ `heldout_sentences`, the held-out SENTENCES of the seen formulations, 10% by hash); held-out FORMULATIONS (by template / file when the examples are templated, else by signature = the sentence with its fillers replaced by their kind; splits named heldout_formulations*); new identities (fillers never in training: ['o12', 'o13', 'o14', 'o15', 'o16', 'o17', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17']); plus held-out depth from None when the world says so. The lexical bag baseline and the lexical order baseline are trained on the same training rows and scored on the same rows as the reader.

## Kinds (spec 6.3)
307205 tokens in the examples; 0 with no kind in the lexicon (share 0.000), given the kind `unknown` (placed in the chart, never a filler, listed in `_unplaced`). Word -> kind table: model/catalog.json.

## Training
D=1024, seeds [0, 1, 2, 3, 4], batch 8, lr 0.02, epochs 1, device cuda, codes path+chainL+bigram, gates learned (closed start). Code commit 1d660ee. Memory curve: runs/reference_D_presence_not_decided/train.log (trainer.train_loop, rule 21).

## Thresholds (rule 25)
Abstention tau = 0.05: stated, not fitted. Presence / attribute / determiner thresholds: midpoint of the mean present / absent ratio on CALIBRATION rows = a sample of the TRAINING split (never the scored rows):
- presence:agent: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 400, absent 0)
- presence:object: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 400, absent 0)
- presence:recipient: threshold 0.636 (present 1.161, n 326 / absent 0.111, n 74)

## Results (seed 0 model in model/; mean ± std over seeds when more than one)

Chance per role (1 / catalog candidates of the role's kinds): agent 0.056, object 0.056, recipient 0.056, predicate 0.083

Two candidate sets: `catalog` = every catalog word of the role's kinds (what `sop formalize` does); `experiment` = as the repository scripts (seen splits: the words seen in training; new identities: the held-out words), for the P0 comparison with RESULTS.

### heldout_depth3 — catalog candidates (n = 120 of 120)
| axis | accuracy |
|---|---|
| FULL | 0.987±0.007 |
| FULL [fit estimate] | 0.987±0.007 |
| FULL [kind-count estimate] | 0.987±0.007 |
| FULL [depth given] | 0.987±0.007 |
| FULL [depth and negation given] | 0.987±0.007 |
| FULL [presence not decided] | 0.987±0.007 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (gates) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 1.000±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.000±0.000 |
| lexical bag head agent | 0.233±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.987±0.007 |
| head recipient | 0.987±0.007 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.200±0.000 |
| speakers | 1.000±0.000 |
| heads FULL (model) | 0.987±0.007 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.075±0.000 |
| role agent [presence not decided] | 1.000±0.000 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 0.987±0.007 |
| heads FULL (model) [presence not decided] | 0.987±0.007 |

residual mean 0.629±0.006; status {'ok': 102, 'abstain': 18}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.124 / 0.094 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.114 / 0.063 | 0.93 | 0.000 | 0.075 / 0.000 |
| roles (pooled) | 0.997 | 0.111 / 0.066 | 0.94 | 0.000 | 0.061 / 0.003 |
| role object | 1.000 | 0.125 / 0.094 | 0.99 | 0.000 | 0.008 / 0.000 |
| role recipient | 0.992 | 0.093 / 0.048 | 0.89 | 0.000 | 0.100 / 0.008 |
| speakers | 1.000 | 0.179 / 0.104 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_depth3 — experiment candidates (n = 120 of 120)
| axis | accuracy |
|---|---|
| FULL | 0.987±0.007 |
| FULL [fit estimate] | 0.987±0.007 |
| FULL [kind-count estimate] | 0.987±0.007 |
| FULL [depth given] | 0.987±0.007 |
| FULL [depth and negation given] | 0.987±0.007 |
| FULL [presence not decided] | 0.987±0.007 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (gates) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 1.000±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.000±0.000 |
| lexical bag head agent | 0.233±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.987±0.007 |
| head recipient | 0.987±0.007 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.200±0.000 |
| speakers | 1.000±0.000 |
| heads FULL (model) | 0.987±0.007 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.075±0.000 |
| role agent [presence not decided] | 1.000±0.000 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 0.987±0.007 |
| heads FULL (model) [presence not decided] | 0.987±0.007 |

residual mean 0.629±0.006; status {'ok': 102, 'abstain': 18}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.124 / 0.094 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.115 / 0.063 | 0.93 | 0.000 | 0.075 / 0.000 |
| roles (pooled) | 0.997 | 0.112 / 0.068 | 0.94 | 0.000 | 0.061 / 0.003 |
| role object | 1.000 | 0.126 / 0.094 | 0.99 | 0.000 | 0.008 / 0.000 |
| role recipient | 0.992 | 0.095 / 0.048 | 0.89 | 0.000 | 0.100 / 0.008 |
| speakers | 1.000 | 0.181 / 0.108 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_formulations_role_cues_seen — catalog candidates (n = 60 of 60)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [fit estimate] | 1.000±0.000 |
| FULL [kind-count estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| FULL [presence not decided] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (gates) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 1.000±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.000±0.000 |
| lexical bag head agent | 0.500±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 1.000±0.000 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.467±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.317±0.000 |
| role agent [presence not decided] | 1.000±0.000 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 1.000±0.000 |
| heads FULL (model) [presence not decided] | 1.000±0.000 |

residual mean 0.550±0.002; status {'ok': 60}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.223 / 0.184 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.215 / 0.172 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.262 / 0.195 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.313 / 0.284 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.256 / 0.208 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_formulations_role_cues_seen — experiment candidates (n = 60 of 60)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [fit estimate] | 1.000±0.000 |
| FULL [kind-count estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| FULL [presence not decided] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (gates) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 1.000±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.000±0.000 |
| lexical bag head agent | 0.500±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 1.000±0.000 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.467±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.317±0.000 |
| role agent [presence not decided] | 1.000±0.000 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 1.000±0.000 |
| heads FULL (model) [presence not decided] | 1.000±0.000 |

residual mean 0.550±0.002; status {'ok': 60}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.223 / 0.184 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.216 / 0.177 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.265 / 0.195 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.318 / 0.284 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.261 / 0.209 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_formulations_seen_transitions — catalog candidates (n = 60 of 60)
| axis | accuracy |
|---|---|
| FULL | 0.503±0.007 |
| FULL [fit estimate] | 0.503±0.007 |
| FULL [kind-count estimate] | 0.503±0.007 |
| FULL [depth given] | 0.503±0.007 |
| FULL [depth and negation given] | 0.503±0.007 |
| FULL [presence not decided] | 0.503±0.007 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (gates) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 0.987±0.012 |
| event type (synonym table) | 0.990±0.008 |
| role agent | 0.763±0.103 |
| head agent | 0.763±0.103 |
| lexical order head agent | 0.000±0.000 |
| lexical bag head agent | 0.533±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.503±0.007 |
| head recipient | 0.503±0.007 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.567±0.000 |
| heads FULL (model) | 0.503±0.007 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.400±0.000 |
| role agent [presence not decided] | 0.763±0.103 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 0.503±0.007 |
| heads FULL (model) [presence not decided] | 0.503±0.007 |

residual mean 0.696±0.002; status {'ok': 30, 'abstain': 30}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 0.967 | 0.138 / 0.010 | 0.72 | 0.000 | 0.250 / 0.033 |
| role agent | 0.783 | 0.131 / 0.014 | 0.58 | 0.033 | 0.233 / 0.183 |
| roles (pooled) | 0.761 | 0.163 / 0.009 | 0.67 | 0.011 | 0.100 / 0.228 |
| role object | 1.000 | 0.208 / 0.062 | 0.93 | 0.000 | 0.067 / 0.000 |
| role recipient | 0.500 | 0.150 / 0.002 | 0.50 | 0.000 | 0.000 / 0.500 |

### heldout_formulations_seen_transitions — experiment candidates (n = 60 of 60)
| axis | accuracy |
|---|---|
| FULL | 0.503±0.007 |
| FULL [fit estimate] | 0.503±0.007 |
| FULL [kind-count estimate] | 0.503±0.007 |
| FULL [depth given] | 0.503±0.007 |
| FULL [depth and negation given] | 0.503±0.007 |
| FULL [presence not decided] | 0.503±0.007 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (gates) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 0.987±0.012 |
| event type (synonym table) | 0.990±0.008 |
| role agent | 0.763±0.103 |
| head agent | 0.763±0.103 |
| lexical order head agent | 0.000±0.000 |
| lexical bag head agent | 0.533±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.507±0.013 |
| head recipient | 0.507±0.013 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.567±0.000 |
| heads FULL (model) | 0.503±0.007 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.400±0.000 |
| role agent [presence not decided] | 0.763±0.103 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 0.507±0.013 |
| heads FULL (model) [presence not decided] | 0.503±0.007 |

residual mean 0.696±0.002; status {'ok': 30, 'abstain': 30}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 0.967 | 0.138 / 0.010 | 0.72 | 0.000 | 0.250 / 0.033 |
| role agent | 0.783 | 0.132 / 0.014 | 0.58 | 0.033 | 0.233 / 0.183 |
| roles (pooled) | 0.761 | 0.167 / 0.009 | 0.68 | 0.011 | 0.089 / 0.228 |
| role object | 1.000 | 0.214 / 0.064 | 0.97 | 0.000 | 0.033 / 0.000 |
| role recipient | 0.500 | 0.155 / 0.002 | 0.50 | 0.000 | 0.000 / 0.500 |

### heldout_formulations_unseen_transitions — catalog candidates (n = 180 of 180)
| axis | accuracy |
|---|---|
| FULL | 0.428±0.047 |
| FULL [fit estimate] | 0.428±0.047 |
| FULL [kind-count estimate] | 0.428±0.047 |
| FULL [depth given] | 0.428±0.047 |
| FULL [depth and negation given] | 0.428±0.047 |
| FULL [presence not decided] | 0.596±0.048 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (gates) | 1.000±0.000 |
| depth estimate: fit (check) | 0.833±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 0.986±0.016 |
| event type (synonym table) | 0.986±0.016 |
| role agent | 0.833±0.000 |
| head agent | 0.833±0.000 |
| lexical order head agent | 0.333±0.000 |
| lexical bag head agent | 0.450±0.000 |
| occupancy: known filler | 0.874±0.016 |
| role object | 0.811±0.063 |
| head object | 0.811±0.063 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.619±0.047 |
| head recipient | 0.619±0.047 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.322±0.000 |
| heads FULL (model) | 0.428±0.047 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.211±0.000 |
| role agent [presence not decided] | 0.833±0.000 |
| role object [presence not decided] | 0.811±0.063 |
| role recipient [presence not decided] | 0.839±0.004 |
| heads FULL (model) [presence not decided] | 0.596±0.048 |

residual mean 0.785±0.002; status {'abstain': 90, 'ok': 90}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 0.983 | 0.170 / 0.030 | 0.88 | 0.000 | 0.106 / 0.017 |
| role agent | 0.833 | 0.158 / 0.011 | 0.83 | 0.000 | 0.000 / 0.167 |
| roles (pooled) | 0.728 | 0.128 / 0.000 | 0.69 | 0.000 | 0.037 / 0.272 |
| role object | 0.739 | 0.112 / 0.004 | 0.67 | 0.000 | 0.072 / 0.261 |
| role recipient | 0.611 | 0.114 / 0.000 | 0.57 | 0.000 | 0.039 / 0.389 |

### heldout_formulations_unseen_transitions — experiment candidates (n = 180 of 180)
| axis | accuracy |
|---|---|
| FULL | 0.434±0.049 |
| FULL [fit estimate] | 0.434±0.049 |
| FULL [kind-count estimate] | 0.434±0.049 |
| FULL [depth given] | 0.434±0.049 |
| FULL [depth and negation given] | 0.434±0.049 |
| FULL [presence not decided] | 0.603±0.051 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (gates) | 1.000±0.000 |
| depth estimate: fit (check) | 0.833±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 0.986±0.016 |
| event type (synonym table) | 0.986±0.016 |
| role agent | 0.833±0.000 |
| head agent | 0.833±0.000 |
| lexical order head agent | 0.333±0.000 |
| lexical bag head agent | 0.450±0.000 |
| occupancy: known filler | 0.876±0.015 |
| role object | 0.823±0.061 |
| head object | 0.823±0.061 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.623±0.045 |
| head recipient | 0.623±0.045 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.322±0.000 |
| heads FULL (model) | 0.434±0.049 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.211±0.000 |
| role agent [presence not decided] | 0.833±0.000 |
| role object [presence not decided] | 0.823±0.061 |
| role recipient [presence not decided] | 0.841±0.004 |
| heads FULL (model) [presence not decided] | 0.603±0.051 |

residual mean 0.785±0.002; status {'abstain': 90, 'ok': 90}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 0.983 | 0.170 / 0.030 | 0.88 | 0.000 | 0.106 / 0.017 |
| role agent | 0.833 | 0.159 / 0.009 | 0.83 | 0.000 | 0.000 / 0.167 |
| roles (pooled) | 0.733 | 0.130 / 0.000 | 0.69 | 0.000 | 0.039 / 0.267 |
| role object | 0.744 | 0.115 / 0.004 | 0.67 | 0.000 | 0.078 / 0.256 |
| role recipient | 0.622 | 0.116 / 0.000 | 0.58 | 0.000 | 0.039 / 0.378 |

### heldout_sentences — catalog candidates (n = 600 of 3072)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [fit estimate] | 1.000±0.000 |
| FULL [kind-count estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| FULL [presence not decided] | 0.817±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (gates) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 1.000±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.348±0.000 |
| lexical bag head agent | 0.420±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 1.000±0.000 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.392±0.000 |
| lexical bag head recipient | 0.510±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.257±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.283±0.000 |
| role agent [presence not decided] | 1.000±0.000 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 0.817±0.000 |
| heads FULL (model) [presence not decided] | 0.817±0.000 |
| speakers | 1.000±0.000 |
| occupancy: not given | 1.000±0.000 |

residual mean 0.348±0.001; status {'ok': 600}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.242 / 0.191 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.235 / 0.155 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.249 / 0.171 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.281 / 0.214 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.225 / 0.160 | 1.00 | 0.000 | 0.000 / 0.000 |
| speakers | 1.000 | 0.254 / 0.203 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_sentences — experiment candidates (n = 600 of 3072)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [fit estimate] | 1.000±0.000 |
| FULL [kind-count estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| FULL [presence not decided] | 0.817±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (gates) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 1.000±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.348±0.000 |
| lexical bag head agent | 0.420±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 1.000±0.000 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.392±0.000 |
| lexical bag head recipient | 0.510±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.257±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.283±0.000 |
| role agent [presence not decided] | 1.000±0.000 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 0.817±0.000 |
| heads FULL (model) [presence not decided] | 0.817±0.000 |
| speakers | 1.000±0.000 |
| occupancy: not given | 1.000±0.000 |

residual mean 0.348±0.001; status {'ok': 600}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.242 / 0.191 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.235 / 0.155 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.252 / 0.174 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.284 / 0.215 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.232 / 0.162 | 1.00 | 0.000 | 0.000 / 0.000 |
| speakers | 1.000 | 0.258 / 0.206 | 1.00 | 0.000 | 0.000 / 0.000 |

### new_identities — catalog candidates (n = 240 of 240)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [fit estimate] | 1.000±0.000 |
| FULL [kind-count estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| FULL [presence not decided] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (gates) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 1.000±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.375±0.000 |
| lexical bag head agent | 0.525±0.000 |
| occupancy: unknown filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 1.000±0.000 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.475±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.000±0.000 |
| role agent [presence not decided] | 1.000±0.000 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 1.000±0.000 |
| heads FULL (model) [presence not decided] | 1.000±0.000 |

residual mean 0.340±0.004; status {'ok': 240}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.267 / 0.221 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.271 / 0.199 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.300 / 0.212 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.341 / 0.302 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.287 / 0.206 | 1.00 | 0.000 | 0.000 / 0.000 |

### new_identities — experiment candidates (n = 240 of 240)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [fit estimate] | 1.000±0.000 |
| FULL [kind-count estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| FULL [presence not decided] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (gates) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 1.000±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.375±0.000 |
| lexical bag head agent | 0.525±0.000 |
| occupancy: unknown filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 1.000±0.000 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.475±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.000±0.000 |
| role agent [presence not decided] | 1.000±0.000 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 1.000±0.000 |
| heads FULL (model) [presence not decided] | 1.000±0.000 |

residual mean 0.340±0.004; status {'ok': 240}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.267 / 0.221 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.290 / 0.225 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.312 / 0.235 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.353 / 0.318 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.294 / 0.220 | 1.00 | 0.000 | 0.000 / 0.000 |

### train (sample) — catalog candidates (n = 300 of 26928)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [fit estimate] | 1.000±0.000 |
| FULL [kind-count estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| FULL [presence not decided] | 0.827±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (gates) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 1.000±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.327±0.000 |
| lexical bag head agent | 0.410±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 1.000±0.000 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.413±0.000 |
| lexical bag head recipient | 0.500±0.000 |
| speakers | 1.000±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.233±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.277±0.000 |
| role agent [presence not decided] | 1.000±0.000 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 0.827±0.000 |
| heads FULL (model) [presence not decided] | 0.827±0.000 |
| occupancy: not given | 1.000±0.000 |

residual mean 0.340±0.001; status {'ok': 300}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.246 / 0.186 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.242 / 0.154 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.255 / 0.171 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.288 / 0.216 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.231 / 0.150 | 1.00 | 0.000 | 0.000 / 0.000 |
| speakers | 1.000 | 0.256 / 0.207 | 1.00 | 0.000 | 0.000 / 0.000 |

## Gates (seed 0; OPEN > 0.5)
that=1.00 by=1.00 from=1.00 to=1.00 who=0.77 .=0.26 unknown=0.12 lent=0.00 given=0.00 shown=0.00 what=0.00 sold=0.00 sees=0.00 receives=0.00 buys=0.00 borrows=0.00 accepts=0.00 offered=0.00 sent=0.00 gets=0.00 offers=0.00 sends=0.00 it=0.00 gives=0.00 shows=0.00 sells=0.00 claims=0.00 lends=0.00 ,=0.00 says=0.00 COP=0.00 O=0.00 P=0.00

## What is weak, what to add
- heldout_depth3: heads FULL (lexical order baseline) 0.000
- heldout_depth3: heads FULL (lexical bag baseline, trained on the same training rows) 0.075
- heldout_formulations_role_cues_seen: heads FULL (lexical order baseline) 0.000
- heldout_formulations_role_cues_seen: heads FULL (lexical bag baseline, trained on the same training rows) 0.317
- heldout_formulations_seen_transitions: FULL 0.500
- heldout_formulations_seen_transitions: role agent 0.783
- heldout_formulations_seen_transitions: head agent 0.783
- heldout_formulations_seen_transitions: role recipient 0.500
- heldout_formulations_seen_transitions: head recipient 0.500
- heldout_formulations_seen_transitions: heads FULL (model) 0.500
- heldout_formulations_seen_transitions: heads FULL (lexical order baseline) 0.000
- heldout_formulations_seen_transitions: heads FULL (lexical bag baseline, trained on the same training rows) 0.400
- heldout_formulations_seen_transitions: FULL [fit estimate] 0.500
- heldout_formulations_seen_transitions: FULL [kind-count estimate] 0.500
- heldout_formulations_seen_transitions: FULL [depth given] 0.500
- heldout_formulations_seen_transitions: FULL [depth and negation given] 0.500
- heldout_formulations_seen_transitions: FULL [presence not decided] 0.500
- heldout_formulations_seen_transitions: role agent [presence not decided] 0.783
- heldout_formulations_seen_transitions: role recipient [presence not decided] 0.500
- heldout_formulations_seen_transitions: heads FULL (model) [presence not decided] 0.500
- heldout_formulations_unseen_transitions: FULL 0.356
- heldout_formulations_unseen_transitions: depth estimate: fit (check) 0.833
- heldout_formulations_unseen_transitions: role agent 0.833
- heldout_formulations_unseen_transitions: head agent 0.833
- heldout_formulations_unseen_transitions: occupancy: known filler 0.874
- heldout_formulations_unseen_transitions: role object 0.739
- heldout_formulations_unseen_transitions: head object 0.739
- heldout_formulations_unseen_transitions: role recipient 0.611
- heldout_formulations_unseen_transitions: head recipient 0.611
- heldout_formulations_unseen_transitions: heads FULL (model) 0.356
- heldout_formulations_unseen_transitions: heads FULL (lexical order baseline) 0.000
- heldout_formulations_unseen_transitions: heads FULL (lexical bag baseline, trained on the same training rows) 0.211
- heldout_formulations_unseen_transitions: FULL [fit estimate] 0.356
- heldout_formulations_unseen_transitions: FULL [kind-count estimate] 0.356
- heldout_formulations_unseen_transitions: FULL [depth given] 0.356
- heldout_formulations_unseen_transitions: FULL [depth and negation given] 0.356
- heldout_formulations_unseen_transitions: FULL [presence not decided] 0.522
- heldout_formulations_unseen_transitions: role agent [presence not decided] 0.833
- heldout_formulations_unseen_transitions: role object [presence not decided] 0.739
- heldout_formulations_unseen_transitions: role recipient [presence not decided] 0.839
- heldout_formulations_unseen_transitions: heads FULL (model) [presence not decided] 0.522
- heldout_sentences: heads FULL (lexical order baseline) 0.257
- heldout_sentences: heads FULL (lexical bag baseline, trained on the same training rows) 0.283
- heldout_sentences: FULL [presence not decided] 0.817
- heldout_sentences: role recipient [presence not decided] 0.817
- heldout_sentences: heads FULL (model) [presence not decided] 0.817
- new_identities: heads FULL (lexical order baseline) 0.000
- new_identities: heads FULL (lexical bag baseline, trained on the same training rows) 0.000
- train (sample): heads FULL (lexical order baseline) 0.233
- train (sample): heads FULL (lexical bag baseline, trained on the same training rows) 0.277
- train (sample): FULL [presence not decided] 0.827
- train (sample): role recipient [presence not decided] 0.827
- train (sample): heads FULL (model) [presence not decided] 0.827

Formulations with the lowest FULL on the held-out splits (add examples of these formulations):
- heldout_formulations_unseen_transitions: `P is offered O by P .` FULL 0.000 (n 12)
- heldout_formulations_seen_transitions: `from P , P gets O .` FULL 0.000 (n 8)
- heldout_formulations_seen_transitions: `from P , P accepts O .` FULL 0.000 (n 7)
- heldout_formulations_unseen_transitions: `P shows P O .` FULL 0.000 (n 7)
- heldout_formulations_unseen_transitions: `it is O that P offers to P .` FULL 0.000 (n 7)
- heldout_formulations_unseen_transitions: `P offers P O .` FULL 0.000 (n 6)
- heldout_formulations_unseen_transitions: `P gives P O .` FULL 0.000 (n 6)
- heldout_formulations_seen_transitions: `from P , P receives O .` FULL 0.000 (n 5)

## Structure given vs estimated (author C3)
`FULL` = the reporting depth and the event negation ESTIMATED from the state (what `sop formalize` does); `FULL [depth given]` = the depth GIVEN (as the repository scripts), negation estimated; `FULL [depth and negation given]`.

## Checklist
[x] addresses from the world [x] shared identities (one star function) [x] chance printed [x] seen / unseen axes separated [x] margin [x] >=3 seeds [x] trivial controls (lexical bag and lexical order baselines) [x] calibration (rule 18) [x] memory logged (rule 21) [x] thresholds on calibration rows (rule 25) [x] given vs learned stated (rule 20)
