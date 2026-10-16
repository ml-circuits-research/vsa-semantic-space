# REPORT — world `reference`, legend sha256:a90d0a429f56… (version 1)

Labels (rule 22): every number below is BUILT (code and measurement exist, computed by this training run). Nothing here is PLANNED or HYPOTHESIS except where a line says so.

## Given and learned (rule 20)
GIVEN: the legend's roles and their vectors, the identities (stars regenerated from the seed), the kinds (lexicon), the event synonym table, the reader block of the legend. LEARNED: the route book only — per-kind displacements of the path / bigram / chain codes and the gates (sigmoid per kind, from an all-closed start), by the single objective max cos(S, S_world) with structured per-sentence targets with pre-set roles; the model learns the assignment.

## Data
30660 annotated sentences from 6 files; provenance: {'generator': 30660}. Audit: none in this world folder (audit/ empty) — agreement not measured. Shapes the v1 decoder cannot read (trained as targets, scored as FULL-wrong): none.

## Splits (written to runs/reference/splits.json before training; source: split_hints.json)
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
D=1024, seeds [0, 1, 2, 3, 4], batch 8, lr 0.02, epochs 1, device cuda, codes path+chainL+bigram, gates learned (closed start). Code commit c843f53. Memory curve: runs/reference/train.log (trainer.train_loop, rule 21).

## Thresholds (rule 25)
Abstention tau = 0.05: stated, not fitted. Presence / attribute / determiner thresholds: midpoint of the mean present / absent ratio on CALIBRATION rows = a sample of the TRAINING split (never the scored rows):
- presence:agent: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 400, absent 0)
- presence:object: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 400, absent 0)
- presence:recipient: threshold 0.686 (present 1.193, n 326 / absent 0.178, n 74)

## Results (seed 0 model in model/; mean ± std over seeds when more than one)

Chance per role (1 / catalog candidates of the role's kinds): agent 0.056, object 0.056, recipient 0.056, predicate 0.083

Two candidate sets: `catalog` = every catalog word of the role's kinds (what `sop formalize` does); `experiment` = as the repository scripts (seen splits: the words seen in training; new identities: the held-out words), for the P0 comparison with RESULTS.

### heldout_depth3 — catalog candidates (n = 120 of 120)
| axis | accuracy |
|---|---|
| FULL | 0.742±0.040 |
| FULL [depth given] | 0.742±0.040 |
| FULL [depth and negation given] | 0.742±0.040 |
| structure (depth, negation) | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 0.903±0.011 |
| head agent | 0.903±0.011 |
| lexical order head agent | 0.000±0.000 |
| lexical bag head agent | 0.233±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.810±0.040 |
| head recipient | 0.810±0.040 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.200±0.000 |
| speakers | 1.000±0.000 |
| heads FULL (model) | 0.742±0.040 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.075±0.000 |

residual mean 0.653±0.013; status {'abstain': 101, 'ok': 19}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.118 / 0.087 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 0.892 | 0.060 / 0.017 | 0.55 | 0.017 | 0.358 / 0.092 |
| roles (pooled) | 0.897 | 0.070 / 0.012 | 0.59 | 0.008 | 0.311 / 0.094 |
| role object | 1.000 | 0.115 / 0.086 | 0.98 | 0.000 | 0.017 / 0.000 |
| role recipient | 0.800 | 0.034 / 0.004 | 0.25 | 0.008 | 0.558 / 0.192 |
| speakers | 1.000 | 0.181 / 0.102 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_depth3 — experiment candidates (n = 120 of 120)
| axis | accuracy |
|---|---|
| FULL | 0.742±0.040 |
| FULL [depth given] | 0.742±0.040 |
| FULL [depth and negation given] | 0.742±0.040 |
| structure (depth, negation) | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 0.903±0.011 |
| head agent | 0.903±0.011 |
| lexical order head agent | 0.000±0.000 |
| lexical bag head agent | 0.233±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.810±0.040 |
| head recipient | 0.810±0.040 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.200±0.000 |
| speakers | 1.000±0.000 |
| heads FULL (model) | 0.742±0.040 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.075±0.000 |

residual mean 0.653±0.013; status {'abstain': 100, 'ok': 20}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.118 / 0.087 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 0.892 | 0.061 / 0.017 | 0.56 | 0.017 | 0.350 / 0.092 |
| roles (pooled) | 0.897 | 0.070 / 0.012 | 0.60 | 0.008 | 0.303 / 0.094 |
| role object | 1.000 | 0.115 / 0.086 | 0.98 | 0.000 | 0.017 / 0.000 |
| role recipient | 0.800 | 0.035 / 0.005 | 0.27 | 0.008 | 0.542 / 0.192 |
| speakers | 1.000 | 0.183 / 0.105 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_formulations_role_cues_seen — catalog candidates (n = 60 of 60)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
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

residual mean 0.596±0.010; status {'ok': 60}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.228 / 0.194 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.142 / 0.092 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.212 / 0.113 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.325 / 0.295 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.170 / 0.125 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_formulations_role_cues_seen — experiment candidates (n = 60 of 60)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
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

residual mean 0.596±0.010; status {'ok': 60}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.228 / 0.194 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.142 / 0.092 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.215 / 0.113 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.331 / 0.299 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.172 / 0.125 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_formulations_seen_transitions — catalog candidates (n = 60 of 60)
| axis | accuracy |
|---|---|
| FULL | 0.660±0.053 |
| FULL [depth given] | 0.660±0.053 |
| FULL [depth and negation given] | 0.660±0.053 |
| structure (depth, negation) | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 0.820±0.029 |
| head agent | 0.820±0.029 |
| lexical order head agent | 0.000±0.000 |
| lexical bag head agent | 0.533±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.760±0.078 |
| head recipient | 0.760±0.078 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.567±0.000 |
| heads FULL (model) | 0.660±0.053 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.400±0.000 |

residual mean 0.678±0.022; status {'ok': 30, 'abstain': 30}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.192 / 0.111 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 0.783 | 0.101 / 0.012 | 0.57 | 0.017 | 0.233 / 0.200 |
| roles (pooled) | 0.883 | 0.166 / 0.014 | 0.70 | 0.011 | 0.194 / 0.106 |
| role object | 1.000 | 0.282 / 0.206 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 0.867 | 0.114 / 0.006 | 0.53 | 0.017 | 0.350 / 0.117 |

### heldout_formulations_seen_transitions — experiment candidates (n = 60 of 60)
| axis | accuracy |
|---|---|
| FULL | 0.660±0.053 |
| FULL [depth given] | 0.660±0.053 |
| FULL [depth and negation given] | 0.660±0.053 |
| structure (depth, negation) | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 0.820±0.029 |
| head agent | 0.820±0.029 |
| lexical order head agent | 0.000±0.000 |
| lexical bag head agent | 0.533±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.760±0.078 |
| head recipient | 0.760±0.078 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.567±0.000 |
| heads FULL (model) | 0.660±0.053 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.400±0.000 |

residual mean 0.678±0.022; status {'ok': 30, 'abstain': 30}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.192 / 0.111 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 0.783 | 0.101 / 0.012 | 0.57 | 0.017 | 0.233 / 0.200 |
| roles (pooled) | 0.883 | 0.168 / 0.014 | 0.70 | 0.011 | 0.194 / 0.106 |
| role object | 1.000 | 0.289 / 0.207 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 0.867 | 0.115 / 0.006 | 0.53 | 0.017 | 0.350 / 0.117 |

### heldout_formulations_unseen_transitions — catalog candidates (n = 180 of 180)
| axis | accuracy |
|---|---|
| FULL | 0.564±0.079 |
| FULL [depth given] | 0.567±0.077 |
| FULL [depth and negation given] | 0.567±0.077 |
| structure (depth, negation) | 0.900±0.082 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 0.943±0.046 |
| head agent | 0.943±0.046 |
| lexical order head agent | 0.333±0.000 |
| lexical bag head agent | 0.450±0.000 |
| occupancy: known filler | 0.927±0.017 |
| role object | 0.978±0.028 |
| head object | 0.978±0.028 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.719±0.041 |
| head recipient | 0.719±0.041 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.322±0.000 |
| heads FULL (model) | 0.643±0.017 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.211±0.000 |

residual mean 0.765±0.016; status {'ok': 118, 'abstain': 62}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.226 / 0.181 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 0.983 | 0.089 / 0.041 | 0.84 | 0.000 | 0.139 / 0.017 |
| roles (pooled) | 0.876 | 0.103 / 0.000 | 0.74 | 0.000 | 0.135 / 0.124 |
| role object | 0.956 | 0.152 / 0.019 | 0.83 | 0.000 | 0.122 / 0.044 |
| role recipient | 0.689 | 0.066 / 0.000 | 0.54 | 0.000 | 0.144 / 0.311 |

### heldout_formulations_unseen_transitions — experiment candidates (n = 180 of 180)
| axis | accuracy |
|---|---|
| FULL | 0.566±0.081 |
| FULL [depth given] | 0.571±0.076 |
| FULL [depth and negation given] | 0.571±0.076 |
| structure (depth, negation) | 0.900±0.082 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 0.943±0.046 |
| head agent | 0.943±0.046 |
| lexical order head agent | 0.333±0.000 |
| lexical bag head agent | 0.450±0.000 |
| occupancy: known filler | 0.927±0.017 |
| role object | 0.980±0.027 |
| head object | 0.980±0.027 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.720±0.042 |
| head recipient | 0.720±0.042 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.322±0.000 |
| heads FULL (model) | 0.647±0.016 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.211±0.000 |

residual mean 0.765±0.016; status {'ok': 118, 'abstain': 62}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.226 / 0.181 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 0.983 | 0.089 / 0.041 | 0.84 | 0.000 | 0.139 / 0.017 |
| roles (pooled) | 0.880 | 0.105 / 0.000 | 0.74 | 0.000 | 0.139 / 0.120 |
| role object | 0.967 | 0.159 / 0.022 | 0.83 | 0.000 | 0.133 / 0.033 |
| role recipient | 0.689 | 0.067 / 0.000 | 0.54 | 0.000 | 0.144 / 0.311 |

### heldout_sentences — catalog candidates (n = 600 of 3072)
| axis | accuracy |
|---|---|
| FULL | 0.985±0.012 |
| FULL [depth given] | 0.985±0.012 |
| FULL [depth and negation given] | 0.985±0.012 |
| structure (depth, negation) | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 0.998±0.003 |
| head agent | 0.998±0.003 |
| lexical order head agent | 0.348±0.000 |
| lexical bag head agent | 0.420±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.987±0.010 |
| head recipient | 0.987±0.010 |
| lexical order head recipient | 0.392±0.000 |
| lexical bag head recipient | 0.510±0.000 |
| heads FULL (model) | 0.985±0.012 |
| heads FULL (lexical order baseline) | 0.257±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.283±0.000 |
| speakers | 1.000±0.000 |
| occupancy: not given | 1.000±0.000 |

residual mean 0.392±0.002; status {'ok': 563, 'abstain': 37}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.245 / 0.187 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.184 / 0.100 | 0.98 | 0.000 | 0.020 / 0.000 |
| roles (pooled) | 0.999 | 0.208 / 0.101 | 0.98 | 0.000 | 0.021 / 0.001 |
| role object | 1.000 | 0.283 / 0.217 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 0.998 | 0.146 / 0.072 | 0.95 | 0.000 | 0.049 / 0.002 |
| speakers | 1.000 | 0.255 / 0.202 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_sentences — experiment candidates (n = 600 of 3072)
| axis | accuracy |
|---|---|
| FULL | 0.985±0.011 |
| FULL [depth given] | 0.985±0.011 |
| FULL [depth and negation given] | 0.985±0.011 |
| structure (depth, negation) | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 0.998±0.003 |
| head agent | 0.998±0.003 |
| lexical order head agent | 0.348±0.000 |
| lexical bag head agent | 0.420±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.987±0.010 |
| head recipient | 0.987±0.010 |
| lexical order head recipient | 0.392±0.000 |
| lexical bag head recipient | 0.510±0.000 |
| heads FULL (model) | 0.985±0.011 |
| heads FULL (lexical order baseline) | 0.257±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.283±0.000 |
| speakers | 1.000±0.000 |
| occupancy: not given | 1.000±0.000 |

residual mean 0.392±0.002; status {'ok': 563, 'abstain': 37}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.245 / 0.187 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.184 / 0.100 | 0.98 | 0.000 | 0.020 / 0.000 |
| roles (pooled) | 0.999 | 0.210 / 0.101 | 0.98 | 0.000 | 0.021 / 0.001 |
| role object | 1.000 | 0.286 / 0.217 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 0.998 | 0.148 / 0.072 | 0.95 | 0.000 | 0.049 / 0.002 |
| speakers | 1.000 | 0.258 / 0.203 | 1.00 | 0.000 | 0.000 / 0.000 |

### new_identities — catalog candidates (n = 240 of 240)
| axis | accuracy |
|---|---|
| FULL | 0.994±0.007 |
| FULL [depth given] | 0.994±0.007 |
| FULL [depth and negation given] | 0.994±0.007 |
| structure (depth, negation) | 1.000±0.000 |
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
| role recipient | 0.994±0.007 |
| head recipient | 0.994±0.007 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.475±0.000 |
| heads FULL (model) | 0.994±0.007 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.000±0.000 |

residual mean 0.396±0.003; status {'ok': 236, 'abstain': 4}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.266 / 0.225 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.220 / 0.163 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.251 / 0.164 | 0.99 | 0.000 | 0.006 / 0.000 |
| role object | 1.000 | 0.328 / 0.293 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.207 / 0.110 | 0.98 | 0.000 | 0.017 / 0.000 |

### new_identities — experiment candidates (n = 240 of 240)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
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

residual mean 0.396±0.003; status {'ok': 237, 'abstain': 3}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.266 / 0.225 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.220 / 0.163 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.257 / 0.164 | 1.00 | 0.000 | 0.004 / 0.000 |
| role object | 1.000 | 0.339 / 0.306 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.211 / 0.123 | 0.99 | 0.000 | 0.013 / 0.000 |

### train (sample) — catalog candidates (n = 300 of 26928)
| axis | accuracy |
|---|---|
| FULL | 0.985±0.008 |
| FULL [depth given] | 0.985±0.008 |
| FULL [depth and negation given] | 0.985±0.008 |
| structure (depth, negation) | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 0.998±0.002 |
| head agent | 0.998±0.002 |
| lexical order head agent | 0.327±0.000 |
| lexical bag head agent | 0.410±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.987±0.008 |
| head recipient | 0.987±0.008 |
| lexical order head recipient | 0.413±0.000 |
| lexical bag head recipient | 0.500±0.000 |
| speakers | 1.000±0.000 |
| heads FULL (model) | 0.985±0.008 |
| heads FULL (lexical order baseline) | 0.233±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.277±0.000 |
| occupancy: not given | 1.000±0.000 |

residual mean 0.384±0.002; status {'ok': 282, 'abstain': 18}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.248 / 0.193 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 0.997 | 0.188 / 0.096 | 0.98 | 0.000 | 0.013 / 0.003 |
| roles (pooled) | 0.998 | 0.214 / 0.102 | 0.98 | 0.000 | 0.019 / 0.002 |
| role object | 1.000 | 0.289 / 0.221 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 0.996 | 0.155 / 0.071 | 0.95 | 0.000 | 0.048 / 0.004 |
| speakers | 1.000 | 0.257 / 0.205 | 1.00 | 0.000 | 0.000 / 0.000 |

## Gates (seed 0; OPEN > 0.5)
that=1.00 .=1.00 sees=0.49 gets=0.46 accepts=0.45 buys=0.42 receives=0.38 borrows=0.28 unknown=0.12 gives=0.07 O=0.03 lent=0.01 who=0.00 sent=0.00 shown=0.00 given=0.00 sold=0.00 sends=0.00 lends=0.00 offers=0.00 offered=0.00 what=0.00 it=0.00 shows=0.00 sells=0.00 says=0.00 to=0.00 P=0.00 claims=0.00 COP=0.00 by=0.00 ,=0.00 from=0.00

## What is weak, what to add
- heldout_depth3: FULL 0.700
- heldout_depth3: role agent 0.892
- heldout_depth3: head agent 0.892
- heldout_depth3: role recipient 0.800
- heldout_depth3: head recipient 0.800
- heldout_depth3: heads FULL (model) 0.700
- heldout_depth3: heads FULL (lexical order baseline) 0.000
- heldout_depth3: heads FULL (lexical bag baseline, trained on the same training rows) 0.075
- heldout_depth3: FULL [depth given] 0.700
- heldout_depth3: FULL [depth and negation given] 0.700
- heldout_formulations_role_cues_seen: heads FULL (lexical order baseline) 0.000
- heldout_formulations_role_cues_seen: heads FULL (lexical bag baseline, trained on the same training rows) 0.317
- heldout_formulations_seen_transitions: FULL 0.683
- heldout_formulations_seen_transitions: role agent 0.783
- heldout_formulations_seen_transitions: head agent 0.783
- heldout_formulations_seen_transitions: role recipient 0.867
- heldout_formulations_seen_transitions: head recipient 0.867
- heldout_formulations_seen_transitions: heads FULL (model) 0.683
- heldout_formulations_seen_transitions: heads FULL (lexical order baseline) 0.000
- heldout_formulations_seen_transitions: heads FULL (lexical bag baseline, trained on the same training rows) 0.400
- heldout_formulations_seen_transitions: FULL [depth given] 0.683
- heldout_formulations_seen_transitions: FULL [depth and negation given] 0.683
- heldout_formulations_unseen_transitions: FULL 0.506
- heldout_formulations_unseen_transitions: structure (depth, negation) 0.833
- heldout_formulations_unseen_transitions: occupancy: known filler 0.898
- heldout_formulations_unseen_transitions: role recipient 0.689
- heldout_formulations_unseen_transitions: head recipient 0.689
- heldout_formulations_unseen_transitions: heads FULL (model) 0.628
- heldout_formulations_unseen_transitions: heads FULL (lexical order baseline) 0.000
- heldout_formulations_unseen_transitions: heads FULL (lexical bag baseline, trained on the same training rows) 0.211
- heldout_formulations_unseen_transitions: FULL [depth given] 0.506
- heldout_formulations_unseen_transitions: FULL [depth and negation given] 0.506
- heldout_sentences: heads FULL (lexical order baseline) 0.257
- heldout_sentences: heads FULL (lexical bag baseline, trained on the same training rows) 0.283
- new_identities: heads FULL (lexical order baseline) 0.000
- new_identities: heads FULL (lexical bag baseline, trained on the same training rows) 0.000
- train (sample): heads FULL (lexical order baseline) 0.233
- train (sample): heads FULL (lexical bag baseline, trained on the same training rows) 0.277

Formulations with the lowest FULL on the held-out splits (add examples of these formulations):
- heldout_formulations_unseen_transitions: `P shows P O .` FULL 0.000 (n 7)
- heldout_formulations_unseen_transitions: `it is O that P offers to P .` FULL 0.000 (n 7)
- heldout_formulations_unseen_transitions: `P offers P O .` FULL 0.000 (n 6)
- heldout_formulations_unseen_transitions: `P gives P O .` FULL 0.000 (n 6)
- heldout_formulations_unseen_transitions: `it is O that P sends to P .` FULL 0.000 (n 6)
- heldout_depth3: `P claims that P claims that P claims that O is sold to P by P .` FULL 0.000 (n 2)
- heldout_depth3: `P says that P claims that P claims that O is offered to P by P .` FULL 0.000 (n 2)
- heldout_depth3: `P claims that P says that P claims that to P , P sends O .` FULL 0.000 (n 2)

## Structure given vs estimated (author C3)
`FULL` = the reporting depth and the event negation ESTIMATED from the state (what `sop formalize` does); `FULL [depth given]` = the depth GIVEN (as the repository scripts), negation estimated; `FULL [depth and negation given]`.

## Checklist
[x] addresses from the world [x] shared identities (one star function) [x] chance printed [x] seen / unseen axes separated [x] margin [x] >=3 seeds [x] trivial controls (lexical bag and lexical order baselines) [x] calibration (rule 18) [x] memory logged (rule 21) [x] thresholds on calibration rows (rule 25) [x] given vs learned stated (rule 20)
