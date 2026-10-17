# REPORT — world `reference`, legend sha256:e221118d4315… (version 2)

Labels (rule 22): every number below is BUILT (code and measurement exist, computed by this training run). Nothing here is PLANNED or HYPOTHESIS except where a line says so.

## Given and learned (rule 20)
GIVEN: the legend's roles and their vectors, the identities (stars regenerated from the seed), the kinds (lexicon), the event synonym table, the reader block of the legend. LEARNED: the route book only — per-kind displacements of the path / bigram / chain codes and the gates (sigmoid per kind, from an all-closed start), by the single objective max cos(S, S_world) with structured per-sentence targets with pre-set roles; the model learns the assignment.

## Data
30660 annotated sentences from 6 files; provenance: {'generator': 30660}. Audit: none in this world folder (audit/ empty) — agreement not measured. Shapes the v1 decoder cannot read (trained as targets, scored as FULL-wrong): none.

## Splits (written to runs/reference_F_calibration_with_heldout_formulations/splits.json before training; source: split_hints.json)
- calibration_heldout_formulations_role_cues_seen: 27 sentences
- calibration_heldout_formulations_seen_transitions: 36 sentences
- calibration_heldout_formulations_unseen_transitions: 93 sentences
- heldout_depth3: 120 sentences
- heldout_formulations_role_cues_seen: 33 sentences
- heldout_formulations_seen_transitions: 24 sentences
- heldout_formulations_unseen_transitions: 87 sentences
- heldout_sentences: 3072 sentences
- new_identities: 240 sentences
- train: 26928 sentences

The four splits of spec 6.4: seen formulations = `train` (+ `heldout_sentences`, the held-out SENTENCES of the seen formulations, 10% by hash); held-out FORMULATIONS (by template / file when the examples are templated, else by signature = the sentence with its fillers replaced by their kind; splits named heldout_formulations*); new identities (fillers never in training: ['o12', 'o13', 'o14', 'o15', 'o16', 'o17', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17']); plus held-out depth from None when the world says so. The lexical bag baseline and the lexical order baseline are trained on the same training rows and scored on the same rows as the reader.

## Kinds (spec 6.3)
307205 tokens in the examples; 0 with no kind in the lexicon (share 0.000), given the kind `unknown` (placed in the chart, never a filler, listed in `_unplaced`). Word -> kind table: model/catalog.json.

## Training
D=1024, seeds [0, 1, 2, 3, 4], batch 8, lr 0.02, epochs 1, device cuda, codes path+chainL+bigram, gates learned (closed start). Code commit 39063ed. Memory curve: runs/reference_F_calibration_with_heldout_formulations/train.log (trainer.train_loop, rule 21).

## Thresholds (rule 25)
Abstention tau = 0.05: stated, not fitted. Presence / attribute / determiner thresholds: midpoint of the mean present / absent ratio on CALIBRATION rows = a sample of the TRAINING split (never the scored rows):
- presence:agent: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 556, absent 0)
- presence:object: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 556, absent 0)
- presence:recipient: threshold 0.622 (present 1.133, n 482 / absent 0.111, n 74)

## Results (seed 0 model in model/; mean ± std over seeds when more than one)

Chance per role (1 / catalog candidates of the role's kinds): agent 0.056, object 0.056, recipient 0.056, predicate 0.083

Two candidate sets: `catalog` = every catalog word of the role's kinds (what `sop formalize` does); `experiment` = as the repository scripts (seen splits: the words seen in training; new identities: the held-out words), for the P0 comparison with RESULTS.

### heldout_depth3 — catalog candidates (n = 120 of 120)
| axis | accuracy |
|---|---|
| FULL | 0.987±0.007 |
| FULL [fit estimate] | 0.987±0.007 |
| FULL [gate-count estimate] | 0.987±0.007 |
| FULL [depth given] | 0.987±0.007 |
| FULL [depth and negation given] | 0.987±0.007 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (kinds) | 1.000±0.000 |
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
| FULL [gate-count estimate] | 0.987±0.007 |
| FULL [depth given] | 0.987±0.007 |
| FULL [depth and negation given] | 0.987±0.007 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (kinds) | 1.000±0.000 |
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

residual mean 0.629±0.006; status {'ok': 102, 'abstain': 18}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.124 / 0.094 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.115 / 0.063 | 0.93 | 0.000 | 0.075 / 0.000 |
| roles (pooled) | 0.997 | 0.112 / 0.068 | 0.94 | 0.000 | 0.061 / 0.003 |
| role object | 1.000 | 0.126 / 0.094 | 0.99 | 0.000 | 0.008 / 0.000 |
| role recipient | 0.992 | 0.095 / 0.048 | 0.89 | 0.000 | 0.100 / 0.008 |
| speakers | 1.000 | 0.181 / 0.108 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_formulations_role_cues_seen — catalog candidates (n = 33 of 33)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [fit estimate] | 1.000±0.000 |
| FULL [gate-count estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| FULL [presence not decided] | 1.000±0.000 |
| FULL [thresholds from training rows only] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (kinds) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 1.000±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.000±0.000 |
| lexical bag head agent | 0.545±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 1.000±0.000 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.576±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.394±0.000 |
| role agent [presence not decided] | 1.000±0.000 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 1.000±0.000 |
| heads FULL (model) [presence not decided] | 1.000±0.000 |
| role agent [thresholds from training rows only] | 1.000±0.000 |
| role object [thresholds from training rows only] | 1.000±0.000 |
| role recipient [thresholds from training rows only] | 1.000±0.000 |
| heads FULL (model) [thresholds from training rows only] | 1.000±0.000 |

residual mean 0.553±0.003; status {'ok': 33}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.221 / 0.177 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.213 / 0.173 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.261 / 0.196 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.314 / 0.285 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.254 / 0.206 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_formulations_role_cues_seen — experiment candidates (n = 33 of 33)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [fit estimate] | 1.000±0.000 |
| FULL [gate-count estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| FULL [presence not decided] | 1.000±0.000 |
| FULL [thresholds from training rows only] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (kinds) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 1.000±0.000 |
| event type (synonym table) | 1.000±0.000 |
| role agent | 1.000±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.000±0.000 |
| lexical bag head agent | 0.545±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 1.000±0.000 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.576±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.394±0.000 |
| role agent [presence not decided] | 1.000±0.000 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 1.000±0.000 |
| heads FULL (model) [presence not decided] | 1.000±0.000 |
| role agent [thresholds from training rows only] | 1.000±0.000 |
| role object [thresholds from training rows only] | 1.000±0.000 |
| role recipient [thresholds from training rows only] | 1.000±0.000 |
| heads FULL (model) [thresholds from training rows only] | 1.000±0.000 |

residual mean 0.553±0.003; status {'ok': 33}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.221 / 0.177 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.214 / 0.179 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.264 / 0.196 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.318 / 0.285 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.259 / 0.209 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_formulations_seen_transitions — catalog candidates (n = 24 of 24)
| axis | accuracy |
|---|---|
| FULL | 0.550±0.017 |
| FULL [fit estimate] | 0.550±0.017 |
| FULL [gate-count estimate] | 0.550±0.017 |
| FULL [depth given] | 0.550±0.017 |
| FULL [depth and negation given] | 0.550±0.017 |
| FULL [presence not decided] | 0.550±0.017 |
| FULL [thresholds from training rows only] | 0.550±0.017 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (kinds) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 0.992±0.017 |
| event type (synonym table) | 0.992±0.017 |
| role agent | 0.800±0.110 |
| head agent | 0.800±0.110 |
| lexical order head agent | 0.000±0.000 |
| lexical bag head agent | 0.375±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.550±0.017 |
| head recipient | 0.550±0.017 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.625±0.000 |
| heads FULL (model) | 0.550±0.017 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.333±0.000 |
| role agent [presence not decided] | 0.800±0.110 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 0.550±0.017 |
| heads FULL (model) [presence not decided] | 0.550±0.017 |
| role agent [thresholds from training rows only] | 0.800±0.110 |
| role object [thresholds from training rows only] | 1.000±0.000 |
| role recipient [thresholds from training rows only] | 0.550±0.017 |
| heads FULL (model) [thresholds from training rows only] | 0.550±0.017 |

residual mean 0.681±0.002; status {'ok': 13, 'abstain': 11}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.155 / 0.009 | 0.75 | 0.000 | 0.250 / 0.000 |
| role agent | 0.875 | 0.138 / 0.016 | 0.58 | 0.000 | 0.292 / 0.125 |
| roles (pooled) | 0.806 | 0.174 / 0.013 | 0.68 | 0.000 | 0.125 / 0.194 |
| role object | 1.000 | 0.219 / 0.059 | 0.92 | 0.000 | 0.083 / 0.000 |
| role recipient | 0.542 | 0.164 / 0.009 | 0.54 | 0.000 | 0.000 / 0.458 |

### heldout_formulations_seen_transitions — experiment candidates (n = 24 of 24)
| axis | accuracy |
|---|---|
| FULL | 0.550±0.017 |
| FULL [fit estimate] | 0.550±0.017 |
| FULL [gate-count estimate] | 0.550±0.017 |
| FULL [depth given] | 0.550±0.017 |
| FULL [depth and negation given] | 0.550±0.017 |
| FULL [presence not decided] | 0.550±0.017 |
| FULL [thresholds from training rows only] | 0.550±0.017 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (kinds) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 0.992±0.017 |
| event type (synonym table) | 0.992±0.017 |
| role agent | 0.800±0.110 |
| head agent | 0.800±0.110 |
| lexical order head agent | 0.000±0.000 |
| lexical bag head agent | 0.375±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.550±0.017 |
| head recipient | 0.550±0.017 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.625±0.000 |
| heads FULL (model) | 0.550±0.017 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.333±0.000 |
| role agent [presence not decided] | 0.800±0.110 |
| role object [presence not decided] | 1.000±0.000 |
| role recipient [presence not decided] | 0.550±0.017 |
| heads FULL (model) [presence not decided] | 0.550±0.017 |
| role agent [thresholds from training rows only] | 0.800±0.110 |
| role object [thresholds from training rows only] | 1.000±0.000 |
| role recipient [thresholds from training rows only] | 0.550±0.017 |
| heads FULL (model) [thresholds from training rows only] | 0.550±0.017 |

residual mean 0.681±0.002; status {'ok': 13, 'abstain': 11}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 1.000 | 0.155 / 0.009 | 0.75 | 0.000 | 0.250 / 0.000 |
| role agent | 0.875 | 0.139 / 0.016 | 0.58 | 0.000 | 0.292 / 0.125 |
| roles (pooled) | 0.806 | 0.178 / 0.013 | 0.68 | 0.000 | 0.125 / 0.194 |
| role object | 1.000 | 0.223 / 0.059 | 0.92 | 0.000 | 0.083 / 0.000 |
| role recipient | 0.542 | 0.171 / 0.009 | 0.54 | 0.000 | 0.000 / 0.458 |

### heldout_formulations_unseen_transitions — catalog candidates (n = 87 of 87)
| axis | accuracy |
|---|---|
| FULL | 0.418±0.061 |
| FULL [fit estimate] | 0.418±0.061 |
| FULL [gate-count estimate] | 0.418±0.061 |
| FULL [depth given] | 0.418±0.061 |
| FULL [depth and negation given] | 0.418±0.061 |
| FULL [presence not decided] | 0.556±0.061 |
| FULL [thresholds from training rows only] | 0.418±0.061 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (kinds) | 1.000±0.000 |
| depth estimate: fit (check) | 0.839±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 0.989±0.013 |
| event type (synonym table) | 0.989±0.013 |
| role agent | 0.839±0.000 |
| head agent | 0.839±0.000 |
| lexical order head agent | 0.333±0.000 |
| lexical bag head agent | 0.471±0.000 |
| occupancy: known filler | 0.880±0.013 |
| role object | 0.779±0.086 |
| head object | 0.779±0.086 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.630±0.037 |
| head recipient | 0.630±0.037 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.310±0.000 |
| heads FULL (model) | 0.418±0.061 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.207±0.000 |
| role agent [presence not decided] | 0.839±0.000 |
| role object [presence not decided] | 0.779±0.086 |
| role recipient [presence not decided] | 0.809±0.006 |
| heads FULL (model) [presence not decided] | 0.556±0.061 |
| role agent [thresholds from training rows only] | 0.839±0.000 |
| role object [thresholds from training rows only] | 0.779±0.086 |
| role recipient [thresholds from training rows only] | 0.625±0.042 |
| heads FULL (model) [thresholds from training rows only] | 0.418±0.061 |

residual mean 0.790±0.002; status {'abstain': 48, 'ok': 39}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 0.989 | 0.172 / 0.031 | 0.89 | 0.000 | 0.103 / 0.011 |
| role agent | 0.839 | 0.160 / 0.016 | 0.84 | 0.000 | 0.000 / 0.161 |
| roles (pooled) | 0.716 | 0.126 / 0.000 | 0.68 | 0.000 | 0.034 / 0.284 |
| role object | 0.690 | 0.102 / 0.003 | 0.61 | 0.000 | 0.080 / 0.310 |
| role recipient | 0.621 | 0.115 / 0.000 | 0.60 | 0.000 | 0.023 / 0.379 |

### heldout_formulations_unseen_transitions — experiment candidates (n = 87 of 87)
| axis | accuracy |
|---|---|
| FULL | 0.425±0.065 |
| FULL [fit estimate] | 0.425±0.065 |
| FULL [gate-count estimate] | 0.425±0.065 |
| FULL [depth given] | 0.425±0.065 |
| FULL [depth and negation given] | 0.425±0.065 |
| FULL [presence not decided] | 0.563±0.065 |
| FULL [thresholds from training rows only] | 0.425±0.065 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (kinds) | 1.000±0.000 |
| depth estimate: fit (check) | 0.839±0.000 |
| depth estimate: opener-kind count | 1.000±0.000 |
| predicate (surface lemma) | 0.989±0.013 |
| event type (synonym table) | 0.989±0.013 |
| role agent | 0.839±0.000 |
| head agent | 0.839±0.000 |
| lexical order head agent | 0.333±0.000 |
| lexical bag head agent | 0.471±0.000 |
| occupancy: known filler | 0.880±0.014 |
| role object | 0.795±0.082 |
| head object | 0.795±0.082 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.632±0.038 |
| head recipient | 0.632±0.038 |
| lexical order head recipient | 0.000±0.000 |
| lexical bag head recipient | 0.310±0.000 |
| heads FULL (model) | 0.425±0.065 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.207±0.000 |
| role agent [presence not decided] | 0.839±0.000 |
| role object [presence not decided] | 0.795±0.082 |
| role recipient [presence not decided] | 0.811±0.006 |
| heads FULL (model) [presence not decided] | 0.563±0.065 |
| role agent [thresholds from training rows only] | 0.839±0.000 |
| role object [thresholds from training rows only] | 0.795±0.082 |
| role recipient [thresholds from training rows only] | 0.632±0.038 |
| heads FULL (model) [thresholds from training rows only] | 0.425±0.065 |

residual mean 0.790±0.002; status {'abstain': 48, 'ok': 39}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| predicate | 0.989 | 0.172 / 0.031 | 0.89 | 0.000 | 0.103 / 0.011 |
| role agent | 0.839 | 0.161 / 0.013 | 0.84 | 0.000 | 0.000 / 0.161 |
| roles (pooled) | 0.720 | 0.127 / 0.000 | 0.68 | 0.000 | 0.038 / 0.280 |
| role object | 0.701 | 0.105 / 0.003 | 0.61 | 0.000 | 0.092 / 0.299 |
| role recipient | 0.621 | 0.117 / 0.000 | 0.60 | 0.000 | 0.023 / 0.379 |

### heldout_sentences — catalog candidates (n = 600 of 3072)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [fit estimate] | 1.000±0.000 |
| FULL [gate-count estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (kinds) | 1.000±0.000 |
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
| FULL [gate-count estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (kinds) | 1.000±0.000 |
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
| FULL [gate-count estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (kinds) | 1.000±0.000 |
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
| FULL [gate-count estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (kinds) | 1.000±0.000 |
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
| FULL [gate-count estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: open-gate count | 1.000±0.000 |
| depth estimate: primary (kinds) | 1.000±0.000 |
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
- heldout_formulations_role_cues_seen: heads FULL (lexical bag baseline, trained on the same training rows) 0.394
- heldout_formulations_seen_transitions: FULL 0.542
- heldout_formulations_seen_transitions: role agent 0.875
- heldout_formulations_seen_transitions: head agent 0.875
- heldout_formulations_seen_transitions: role recipient 0.542
- heldout_formulations_seen_transitions: head recipient 0.542
- heldout_formulations_seen_transitions: heads FULL (model) 0.542
- heldout_formulations_seen_transitions: heads FULL (lexical order baseline) 0.000
- heldout_formulations_seen_transitions: heads FULL (lexical bag baseline, trained on the same training rows) 0.333
- heldout_formulations_seen_transitions: FULL [fit estimate] 0.542
- heldout_formulations_seen_transitions: FULL [gate-count estimate] 0.542
- heldout_formulations_seen_transitions: FULL [depth given] 0.542
- heldout_formulations_seen_transitions: FULL [depth and negation given] 0.542
- heldout_formulations_seen_transitions: FULL [presence not decided] 0.542
- heldout_formulations_seen_transitions: role agent [presence not decided] 0.875
- heldout_formulations_seen_transitions: role recipient [presence not decided] 0.542
- heldout_formulations_seen_transitions: heads FULL (model) [presence not decided] 0.542
- heldout_formulations_seen_transitions: FULL [thresholds from training rows only] 0.542
- heldout_formulations_seen_transitions: role agent [thresholds from training rows only] 0.875
- heldout_formulations_seen_transitions: role recipient [thresholds from training rows only] 0.542
- heldout_formulations_seen_transitions: heads FULL (model) [thresholds from training rows only] 0.542
- heldout_formulations_unseen_transitions: FULL 0.333
- heldout_formulations_unseen_transitions: depth estimate: fit (check) 0.839
- heldout_formulations_unseen_transitions: role agent 0.839
- heldout_formulations_unseen_transitions: head agent 0.839
- heldout_formulations_unseen_transitions: occupancy: known filler 0.881
- heldout_formulations_unseen_transitions: role object 0.690
- heldout_formulations_unseen_transitions: head object 0.690
- heldout_formulations_unseen_transitions: role recipient 0.621
- heldout_formulations_unseen_transitions: head recipient 0.621
- heldout_formulations_unseen_transitions: heads FULL (model) 0.333
- heldout_formulations_unseen_transitions: heads FULL (lexical order baseline) 0.000
- heldout_formulations_unseen_transitions: heads FULL (lexical bag baseline, trained on the same training rows) 0.207
- heldout_formulations_unseen_transitions: FULL [fit estimate] 0.333
- heldout_formulations_unseen_transitions: FULL [gate-count estimate] 0.333
- heldout_formulations_unseen_transitions: FULL [depth given] 0.333
- heldout_formulations_unseen_transitions: FULL [depth and negation given] 0.333
- heldout_formulations_unseen_transitions: FULL [presence not decided] 0.471
- heldout_formulations_unseen_transitions: role agent [presence not decided] 0.839
- heldout_formulations_unseen_transitions: role object [presence not decided] 0.690
- heldout_formulations_unseen_transitions: role recipient [presence not decided] 0.805
- heldout_formulations_unseen_transitions: heads FULL (model) [presence not decided] 0.471
- heldout_formulations_unseen_transitions: FULL [thresholds from training rows only] 0.333
- heldout_formulations_unseen_transitions: role agent [thresholds from training rows only] 0.839
- heldout_formulations_unseen_transitions: role object [thresholds from training rows only] 0.690
- heldout_formulations_unseen_transitions: role recipient [thresholds from training rows only] 0.609
- heldout_formulations_unseen_transitions: heads FULL (model) [thresholds from training rows only] 0.333
- heldout_sentences: heads FULL (lexical order baseline) 0.257
- heldout_sentences: heads FULL (lexical bag baseline, trained on the same training rows) 0.283
- new_identities: heads FULL (lexical order baseline) 0.000
- new_identities: heads FULL (lexical bag baseline, trained on the same training rows) 0.000
- train (sample): heads FULL (lexical order baseline) 0.233
- train (sample): heads FULL (lexical bag baseline, trained on the same training rows) 0.277

Formulations with the lowest FULL on the held-out splits (add examples of these formulations):
- heldout_formulations_unseen_transitions: `P is offered O by P .` FULL 0.000 (n 6)
- heldout_formulations_unseen_transitions: `P offers P O .` FULL 0.000 (n 5)
- heldout_formulations_unseen_transitions: `P buys from P O .` FULL 0.000 (n 5)
- heldout_formulations_seen_transitions: `from P , P accepts O .` FULL 0.000 (n 4)
- heldout_formulations_unseen_transitions: `P gives P O .` FULL 0.000 (n 4)
- heldout_formulations_unseen_transitions: `it is O that P offers to P .` FULL 0.000 (n 4)
- heldout_formulations_seen_transitions: `from P , P gets O .` FULL 0.000 (n 3)
- heldout_formulations_seen_transitions: `from P , P buys O .` FULL 0.000 (n 2)

## Unseen formulations: the readouts (rounds 3-4; judgment 45ccd1d D)
The PRODUCT number comes first: the product readout decides presence (a role missing from the text reads "not given"; `_empty` is the product), with the thresholds of model/config.json (set on training rows + the calibration half of the held-out formulations). The v16-protocol row reads every role of the schema (v16 always reads C): a comparison, not available to the product, never a default. Catalog candidates.

| split | readout | heads FULL | FULL | role agent | role object | role recipient |
|---|---|---|---|---|---|---|
| heldout_formulations_role_cues_seen | product readout: presence decided | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| heldout_formulations_role_cues_seen | presence decided, thresholds from training rows only (the round-3 product) | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| heldout_formulations_role_cues_seen | v16 protocol: every role read, not available to the product | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| heldout_formulations_seen_transitions | product readout: presence decided | 0.550±0.017 | 0.550±0.017 | 0.800±0.110 | 1.000±0.000 | 0.550±0.017 |
| heldout_formulations_seen_transitions | presence decided, thresholds from training rows only (the round-3 product) | 0.550±0.017 | 0.550±0.017 | 0.800±0.110 | 1.000±0.000 | 0.550±0.017 |
| heldout_formulations_seen_transitions | v16 protocol: every role read, not available to the product | 0.550±0.017 | 0.550±0.017 | 0.800±0.110 | 1.000±0.000 | 0.550±0.017 |
| heldout_formulations_unseen_transitions | product readout: presence decided | 0.418±0.061 | 0.418±0.061 | 0.839±0.000 | 0.779±0.086 | 0.630±0.037 |
| heldout_formulations_unseen_transitions | presence decided, thresholds from training rows only (the round-3 product) | 0.418±0.061 | 0.418±0.061 | 0.839±0.000 | 0.779±0.086 | 0.625±0.042 |
| heldout_formulations_unseen_transitions | v16 protocol: every role read, not available to the product | 0.556±0.061 | 0.556±0.061 | 0.839±0.000 | 0.779±0.086 | 0.809±0.006 |

## Structure given vs estimated (author C3)
`FULL` = the reporting depth ESTIMATED by the primary estimator of model/config.json (`kinds`: the count of level-opener kinds by default; the open-gate count and the fit are the checks, `FULL [gate-count estimate]` / `FULL [fit estimate]`) and the event negation by the fit (what `sop formalize` does); `FULL [depth given]` = the depth GIVEN (as the repository scripts), negation estimated; `FULL [depth and negation given]`.

## Checklist
[x] addresses from the world [x] shared identities (one star function) [x] chance printed [x] seen / unseen axes separated [x] margin [x] >=3 seeds [x] trivial controls (lexical bag and lexical order baselines) [x] calibration (rule 18) [x] memory logged (rule 21) [x] thresholds on calibration rows (rule 25) [x] given vs learned stated (rule 20)
