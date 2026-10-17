# REPORT — world `commands`, legend sha256:4e0f52081284… (version 2)

Labels (rule 22): every number below is BUILT (code and measurement exist, computed by this training run). Nothing here is PLANNED or HYPOTHESIS except where a line says so.

## Given and learned (rule 20)
GIVEN: the legend's roles and their vectors, the identities (stars regenerated from the seed), the kinds (lexicon), the event synonym table, the reader block of the legend. LEARNED: the route book only — per-kind displacements of the path / bigram / chain codes and the gates (sigmoid per kind, from an all-closed start), by the single objective max cos(S, S_world) with structured per-sentence targets with pre-set roles; the model learns the assignment.

## Data
40315 annotated sentences from 3 files; provenance: {'generator': 40315}. Audit: none in this world folder (audit/ empty) — agreement not measured. Shapes the v1 decoder cannot read (trained as targets, scored as FULL-wrong): none.

## Splits (written to runs/commands_depth_from_gates/splits.json before training; source: split_hints.json)
- heldout_depth2: 105 sentences
- heldout_sentences: 4660 sentences
- new_identities: 210 sentences
- train: 35340 sentences

The four splits of spec 6.4: seen formulations = `train` (+ `heldout_sentences`, the held-out SENTENCES of the seen formulations, 10% by hash); held-out FORMULATIONS (by template / file when the examples are templated, else by signature = the sentence with its fillers replaced by their kind; splits named heldout_formulations*); new identities (fillers never in training: ['backups', 'notes', 'tickets']); plus held-out depth from None when the world says so. The lexical bag baseline and the lexical order baseline are trained on the same training rows and scored on the same rows as the reader.

## Kinds (spec 6.3)
312906 tokens in the examples; 0 with no kind in the lexicon (share 0.000), given the kind `unknown` (placed in the chart, never a filler, listed in `_unplaced`). Word -> kind table: model/catalog.json.

## Training
D=1024, seeds [0, 1, 2, 3, 4], batch 8, lr 0.02, epochs 1, device cuda, codes path+chainL+bigram, gates learned (closed start). Code commit 815e5f9. Memory curve: runs/commands_depth_from_gates/train.log (trainer.train_loop, rule 21).

## Thresholds (rule 25)
Abstention tau = 0.05: stated, not fitted. Presence / attribute / determiner thresholds: midpoint of the mean present / absent ratio on CALIBRATION rows = a sample of the TRAINING split (never the scored rows):
- presence:action: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 400, absent 0)
- presence:patient: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 400, absent 0)
- attr_presence:patient: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 400, absent 0)
- presence:destination: threshold 0.497 (present 0.967, n 57 / absent 0.028, n 343)

## Results (seed 0 model in model/; mean ± std over seeds when more than one)

Chance per role (1 / catalog candidates of the role's kinds): action 0.200, patient 0.111, destination 0.250

Two candidate sets: `catalog` = every catalog word of the role's kinds (what `sop formalize` does); `experiment` = as the repository scripts (seen splits: the words seen in training; new identities: the held-out words), for the P0 comparison with RESULTS.

### heldout_depth2 — catalog candidates (n = 105 of 105)
| axis | accuracy |
|---|---|
| FULL | 0.682±0.342 |
| FULL [fit estimate] | 0.590±0.059 |
| FULL [depth given] | 0.857±0.025 |
| FULL [depth and negation given] | 0.857±0.025 |
| structure (depth, negation) | 0.800±0.400 |
| depth estimate: level gates (primary) | 0.800±0.400 |
| depth estimate: fit (check) | 0.684±0.065 |
| role action | 0.990±0.010 |
| head action | 0.990±0.010 |
| lexical order head action | 1.000±0.000 |
| lexical bag head action | 1.000±0.000 |
| occupancy: known filler | 0.996±0.009 |
| role patient | 0.848±0.029 |
| head patient | 1.000±0.000 |
| lexical order head patient | 1.000±0.000 |
| lexical bag head patient | 1.000±0.000 |
| role destination | 0.971±0.057 |
| head destination | 0.971±0.057 |
| lexical order head destination | 0.857±0.000 |
| lexical bag head destination | 0.905±0.000 |
| occupancy: not given | 0.978±0.044 |
| speakers | 0.800±0.400 |
| heads FULL (model) | 0.966±0.059 |
| heads FULL (lexical order baseline) | 0.857±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.905±0.000 |

residual mean 0.918±0.034; status {'abstain': 103, 'ok': 2}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| role action | 1.000 | 0.097 / 0.053 | 0.92 | 0.000 | 0.076 / 0.000 |
| roles (pooled) | 0.924 | 0.100 / 0.056 | 0.95 | 0.076 | 0.053 / 0.000 |
| role patient | 0.838 | 0.105 / 0.061 | 0.96 | 0.162 | 0.038 / 0.000 |
| speakers | 1.000 | 0.249 / 0.102 | 1.00 | 0.000 | 0.000 / 0.000 |
| role destination | 1.000 | 0.085 / 0.074 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_depth2 — experiment candidates (n = 105 of 105)
| axis | accuracy |
|---|---|
| FULL | 0.682±0.342 |
| FULL [fit estimate] | 0.590±0.059 |
| FULL [depth given] | 0.857±0.025 |
| FULL [depth and negation given] | 0.857±0.025 |
| structure (depth, negation) | 0.800±0.400 |
| depth estimate: level gates (primary) | 0.800±0.400 |
| depth estimate: fit (check) | 0.684±0.065 |
| role action | 0.990±0.010 |
| head action | 0.990±0.010 |
| lexical order head action | 1.000±0.000 |
| lexical bag head action | 1.000±0.000 |
| occupancy: known filler | 0.996±0.009 |
| role patient | 0.848±0.029 |
| head patient | 1.000±0.000 |
| lexical order head patient | 1.000±0.000 |
| lexical bag head patient | 1.000±0.000 |
| role destination | 0.971±0.057 |
| head destination | 0.971±0.057 |
| lexical order head destination | 0.857±0.000 |
| lexical bag head destination | 0.905±0.000 |
| occupancy: not given | 0.978±0.044 |
| speakers | 0.800±0.400 |
| heads FULL (model) | 0.966±0.059 |
| heads FULL (lexical order baseline) | 0.857±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.905±0.000 |

residual mean 0.918±0.034; status {'abstain': 103, 'ok': 2}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| role action | 1.000 | 0.097 / 0.053 | 0.92 | 0.000 | 0.076 / 0.000 |
| roles (pooled) | 0.924 | 0.101 / 0.056 | 0.95 | 0.076 | 0.053 / 0.000 |
| role patient | 0.838 | 0.107 / 0.061 | 0.96 | 0.162 | 0.038 / 0.000 |
| speakers | 1.000 | 0.249 / 0.102 | 1.00 | 0.000 | 0.000 / 0.000 |
| role destination | 1.000 | 0.085 / 0.074 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_sentences — catalog candidates (n = 600 of 4660)
| axis | accuracy |
|---|---|
| FULL | 0.947±0.105 |
| FULL [fit estimate] | 1.000±0.001 |
| FULL [depth given] | 1.000±0.001 |
| FULL [depth and negation given] | 1.000±0.001 |
| structure (depth, negation) | 0.948±0.105 |
| depth estimate: level gates (primary) | 0.948±0.105 |
| depth estimate: fit (check) | 1.000±0.000 |
| role action | 0.970±0.060 |
| head action | 0.970±0.060 |
| lexical order head action | 1.000±0.000 |
| lexical bag head action | 1.000±0.000 |
| occupancy: known filler | 1.000±0.001 |
| role patient | 0.957±0.085 |
| head patient | 0.974±0.051 |
| lexical order head patient | 1.000±0.000 |
| lexical bag head patient | 1.000±0.000 |
| role destination | 0.964±0.073 |
| head destination | 0.964±0.073 |
| lexical order head destination | 0.872±0.000 |
| lexical bag head destination | 1.000±0.000 |
| occupancy: not given | 0.963±0.073 |
| heads FULL (model) | 0.950±0.100 |
| heads FULL (lexical order baseline) | 0.872±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 1.000±0.000 |
| speakers | 0.800±0.400 |

residual mean 0.279±0.076; status {'ok': 589, 'abstain': 11}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| role action | 1.000 | 0.547 / 0.460 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 0.999 | 0.525 / 0.430 | 1.00 | 0.001 | 0.000 / 0.000 |
| role patient | 0.998 | 0.510 / 0.410 | 1.00 | 0.002 | 0.000 / 0.000 |
| role destination | 1.000 | 0.457 / 0.395 | 1.00 | 0.000 | 0.000 / 0.000 |
| speakers | 1.000 | 0.508 / 0.460 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_sentences — experiment candidates (n = 600 of 4660)
| axis | accuracy |
|---|---|
| FULL | 0.947±0.105 |
| FULL [fit estimate] | 1.000±0.001 |
| FULL [depth given] | 1.000±0.001 |
| FULL [depth and negation given] | 1.000±0.001 |
| structure (depth, negation) | 0.948±0.105 |
| depth estimate: level gates (primary) | 0.948±0.105 |
| depth estimate: fit (check) | 1.000±0.000 |
| role action | 0.971±0.058 |
| head action | 0.971±0.058 |
| lexical order head action | 1.000±0.000 |
| lexical bag head action | 1.000±0.000 |
| occupancy: known filler | 1.000±0.001 |
| role patient | 0.958±0.084 |
| head patient | 0.978±0.043 |
| lexical order head patient | 1.000±0.000 |
| lexical bag head patient | 1.000±0.000 |
| role destination | 0.964±0.071 |
| head destination | 0.964±0.071 |
| lexical order head destination | 0.872±0.000 |
| lexical bag head destination | 1.000±0.000 |
| occupancy: not given | 0.964±0.072 |
| heads FULL (model) | 0.951±0.097 |
| heads FULL (lexical order baseline) | 0.872±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 1.000±0.000 |
| speakers | 0.800±0.400 |

residual mean 0.279±0.076; status {'ok': 589, 'abstain': 11}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| role action | 1.000 | 0.547 / 0.460 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 0.999 | 0.525 / 0.432 | 1.00 | 0.001 | 0.000 / 0.000 |
| role patient | 0.998 | 0.512 / 0.414 | 1.00 | 0.002 | 0.000 / 0.000 |
| role destination | 1.000 | 0.457 / 0.395 | 1.00 | 0.000 | 0.000 / 0.000 |
| speakers | 1.000 | 0.508 / 0.460 | 1.00 | 0.000 | 0.000 / 0.000 |

### new_identities — catalog candidates (n = 210 of 210)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [fit estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: level gates (primary) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| role action | 1.000±0.000 |
| head action | 1.000±0.000 |
| lexical order head action | 1.000±0.000 |
| lexical bag head action | 1.000±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role patient | 1.000±0.000 |
| head patient | 1.000±0.000 |
| lexical order head patient | 1.000±0.000 |
| lexical bag head patient | 1.000±0.000 |
| occupancy: unknown filler | 1.000±0.000 |
| role destination | 1.000±0.000 |
| head destination | 1.000±0.000 |
| lexical order head destination | 0.857±0.000 |
| lexical bag head destination | 1.000±0.000 |
| occupancy: not given | 1.000±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.857±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 1.000±0.000 |

residual mean 0.236±0.001; status {'ok': 207, 'abstain': 3}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| role action | 1.000 | 0.566 / 0.504 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.546 / 0.475 | 1.00 | 0.000 | 0.000 / 0.000 |
| role patient | 1.000 | 0.534 / 0.465 | 1.00 | 0.000 | 0.000 / 0.000 |
| role destination | 1.000 | 0.483 / 0.450 | 1.00 | 0.000 | 0.000 / 0.000 |

### new_identities — experiment candidates (n = 210 of 210)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL [fit estimate] | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: level gates (primary) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| role action | 1.000±0.000 |
| head action | 1.000±0.000 |
| lexical order head action | 1.000±0.000 |
| lexical bag head action | 1.000±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role patient | 1.000±0.000 |
| head patient | 1.000±0.000 |
| lexical order head patient | 1.000±0.000 |
| lexical bag head patient | 1.000±0.000 |
| occupancy: unknown filler | 1.000±0.000 |
| role destination | 1.000±0.000 |
| head destination | 1.000±0.000 |
| lexical order head destination | 0.857±0.000 |
| lexical bag head destination | 1.000±0.000 |
| occupancy: not given | 1.000±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.857±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 1.000±0.000 |

residual mean 0.236±0.001; status {'ok': 207, 'abstain': 3}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| role action | 1.000 | 0.566 / 0.504 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.559 / 0.483 | 1.00 | 0.000 | 0.000 / 0.000 |
| role patient | 1.000 | 0.564 / 0.488 | 1.00 | 0.000 | 0.000 / 0.000 |
| role destination | 1.000 | 0.483 / 0.450 | 1.00 | 0.000 | 0.000 / 0.000 |

### train (sample) — catalog candidates (n = 300 of 35340)
| axis | accuracy |
|---|---|
| FULL | 0.937±0.123 |
| FULL [fit estimate] | 0.999±0.002 |
| FULL [depth given] | 0.999±0.002 |
| FULL [depth and negation given] | 0.999±0.002 |
| structure (depth, negation) | 0.938±0.124 |
| depth estimate: level gates (primary) | 0.938±0.124 |
| depth estimate: fit (check) | 1.000±0.000 |
| role action | 0.965±0.071 |
| head action | 0.965±0.071 |
| lexical order head action | 1.000±0.000 |
| lexical bag head action | 1.000±0.000 |
| occupancy: known filler | 0.999±0.002 |
| role patient | 0.951±0.095 |
| head patient | 0.969±0.063 |
| lexical order head patient | 1.000±0.000 |
| lexical bag head patient | 1.000±0.000 |
| role destination | 0.957±0.085 |
| head destination | 0.957±0.085 |
| lexical order head destination | 0.880±0.000 |
| lexical bag head destination | 1.000±0.000 |
| occupancy: not given | 0.957±0.086 |
| speakers | 0.800±0.400 |
| heads FULL (model) | 0.944±0.112 |
| heads FULL (lexical order baseline) | 0.880±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 1.000±0.000 |

residual mean 0.290±0.088; status {'ok': 290, 'abstain': 10}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| role action | 1.000 | 0.545 / 0.465 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 0.998 | 0.521 / 0.426 | 1.00 | 0.002 | 0.000 / 0.000 |
| role patient | 0.997 | 0.505 / 0.408 | 1.00 | 0.003 | 0.000 / 0.000 |
| speakers | 1.000 | 0.511 / 0.462 | 1.00 | 0.000 | 0.000 / 0.000 |
| role destination | 1.000 | 0.447 / 0.400 | 1.00 | 0.000 | 0.000 / 0.000 |

## Gates (seed 0; OPEN > 0.5)
do=1.00 the=1.00 that=1.00 never=1.00 DEST=0.82 ATTR=0.74 PAT=0.42 not=0.32 unknown=0.12 PRINC=0.02 says=0.01 asks=0.01 are=0.00 NATTR=0.00 .=0.00 ACT=0.00 to=0.00

## What is weak, what to add
- heldout_depth2: FULL 0.838
- heldout_depth2: depth estimate: fit (check) 0.638
- heldout_depth2: role patient 0.838
- heldout_depth2: heads FULL (lexical order baseline) 0.857
- heldout_depth2: FULL [fit estimate] 0.552
- heldout_depth2: FULL [depth given] 0.838
- heldout_depth2: FULL [depth and negation given] 0.838
- heldout_sentences: heads FULL (lexical order baseline) 0.872
- new_identities: heads FULL (lexical order baseline) 0.857
- train (sample): heads FULL (lexical order baseline) 0.880

Formulations with the lowest FULL on the held-out splits (add examples of these formulations):
- heldout_depth2: `the PRINC asks that the PRINC asks that ACT the NATTR PAT .` FULL 0.000 (n 4)
- heldout_depth2: `the PRINC says that the PRINC says that do not ACT the NATTR PAT .` FULL 0.200 (n 5)
- heldout_depth2: `the PRINC asks that the PRINC asks that do not ACT the NATTR PAT .` FULL 0.400 (n 5)
- heldout_depth2: `the PRINC asks that the PRINC says that ACT the NATTR PAT .` FULL 0.500 (n 6)
- heldout_depth2: `the PRINC says that the PRINC says that ACT the NATTR PAT .` FULL 0.500 (n 4)
- heldout_sentences: `the PRINC says that do not ACT the NATTR PAT .` FULL 0.889 (n 9)

## Structure given vs estimated (author C3)
`FULL` = the reporting depth and the event negation ESTIMATED from the state (what `sop formalize` does); `FULL [depth given]` = the depth GIVEN (as the repository scripts), negation estimated; `FULL [depth and negation given]`.

## Checklist
[x] addresses from the world [x] shared identities (one star function) [x] chance printed [x] seen / unseen axes separated [x] margin [x] >=3 seeds [x] trivial controls (lexical bag and lexical order baselines) [x] calibration (rule 18) [x] memory logged (rule 21) [x] thresholds on calibration rows (rule 25) [x] given vs learned stated (rule 20)
