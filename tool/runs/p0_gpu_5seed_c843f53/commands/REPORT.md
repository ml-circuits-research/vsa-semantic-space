# REPORT — world `commands`, legend sha256:0782b57ac430… (version 1)

Labels (rule 22): every number below is BUILT (code and measurement exist, computed by this training run). Nothing here is PLANNED or HYPOTHESIS except where a line says so.

## Given and learned (rule 20)
GIVEN: the legend's roles and their vectors, the identities (stars regenerated from the seed), the kinds (lexicon), the event synonym table, the reader block of the legend. LEARNED: the route book only — per-kind displacements of the path / bigram / chain codes and the gates (sigmoid per kind, from an all-closed start), by the single objective max cos(S, S_world) with structured per-sentence targets with pre-set roles; the model learns the assignment.

## Data
40315 annotated sentences from 3 files; provenance: {'generator': 40315}. Audit: none in this world folder (audit/ empty) — agreement not measured. Shapes the v1 decoder cannot read (trained as targets, scored as FULL-wrong): none.

## Splits (written to runs/commands/splits.json before training; source: split_hints.json)
- heldout_depth2: 105 sentences
- heldout_sentences: 4660 sentences
- new_identities: 210 sentences
- train: 35340 sentences

The four splits of spec 6.4: seen formulations = `train` (+ `heldout_sentences`, the held-out SENTENCES of the seen formulations, 10% by hash); held-out FORMULATIONS (by template / file when the examples are templated, else by signature = the sentence with its fillers replaced by their kind; splits named heldout_formulations*); new identities (fillers never in training: ['backups', 'notes', 'tickets']); plus held-out depth from None when the world says so. The lexical bag baseline and the lexical order baseline are trained on the same training rows and scored on the same rows as the reader.

## Kinds (spec 6.3)
312906 tokens in the examples; 0 with no kind in the lexicon (share 0.000), given the kind `unknown` (placed in the chart, never a filler, listed in `_unplaced`). Word -> kind table: model/catalog.json.

## Training
D=1024, seeds [0, 1, 2, 3, 4], batch 8, lr 0.02, epochs 1, device cuda, codes path+chainL+bigram, gates learned (closed start). Code commit c843f53. Memory curve: runs/commands/train.log (trainer.train_loop, rule 21).

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
| FULL | 0.590±0.059 |
| FULL [depth given] | 0.857±0.025 |
| FULL [depth and negation given] | 0.857±0.025 |
| structure (depth, negation) | 0.684±0.065 |
| role action | 0.998±0.004 |
| head action | 0.998±0.004 |
| lexical order head action | 1.000±0.000 |
| lexical bag head action | 1.000±0.000 |
| occupancy: known filler | 0.998±0.002 |
| role patient | 0.779±0.051 |
| head patient | 0.971±0.031 |
| lexical order head patient | 1.000±0.000 |
| lexical bag head patient | 1.000±0.000 |
| role destination | 0.987±0.018 |
| head destination | 0.987±0.018 |
| lexical order head destination | 0.857±0.000 |
| lexical bag head destination | 0.905±0.000 |
| occupancy: not given | 0.989±0.022 |
| speakers | 0.792±0.033 |
| heads FULL (model) | 0.958±0.039 |
| heads FULL (lexical order baseline) | 0.857±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.905±0.000 |

residual mean 0.910±0.005; status {'abstain': 103, 'ok': 2}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| role action | 1.000 | 0.113 / 0.073 | 0.97 | 0.000 | 0.029 / 0.000 |
| roles (pooled) | 0.898 | 0.104 / 0.054 | 0.92 | 0.071 | 0.044 / 0.031 |
| role patient | 0.781 | 0.098 / 0.048 | 0.87 | 0.152 | 0.067 / 0.067 |
| speakers | 0.762 | 0.205 / 0.000 | 0.76 | 0.000 | 0.000 / 0.238 |
| role destination | 1.000 | 0.087 / 0.072 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_depth2 — experiment candidates (n = 105 of 105)
| axis | accuracy |
|---|---|
| FULL | 0.590±0.059 |
| FULL [depth given] | 0.857±0.025 |
| FULL [depth and negation given] | 0.857±0.025 |
| structure (depth, negation) | 0.684±0.065 |
| role action | 0.998±0.004 |
| head action | 0.998±0.004 |
| lexical order head action | 1.000±0.000 |
| lexical bag head action | 1.000±0.000 |
| occupancy: known filler | 0.998±0.002 |
| role patient | 0.779±0.051 |
| head patient | 0.971±0.031 |
| lexical order head patient | 1.000±0.000 |
| lexical bag head patient | 1.000±0.000 |
| role destination | 0.987±0.018 |
| head destination | 0.987±0.018 |
| lexical order head destination | 0.857±0.000 |
| lexical bag head destination | 0.905±0.000 |
| occupancy: not given | 0.989±0.022 |
| speakers | 0.792±0.033 |
| heads FULL (model) | 0.958±0.039 |
| heads FULL (lexical order baseline) | 0.857±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.905±0.000 |

residual mean 0.910±0.005; status {'abstain': 103, 'ok': 2}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| role action | 1.000 | 0.113 / 0.073 | 0.97 | 0.000 | 0.029 / 0.000 |
| roles (pooled) | 0.898 | 0.106 / 0.054 | 0.93 | 0.071 | 0.036 / 0.031 |
| role patient | 0.781 | 0.101 / 0.050 | 0.89 | 0.152 | 0.048 / 0.067 |
| speakers | 0.762 | 0.205 / 0.000 | 0.76 | 0.000 | 0.000 / 0.238 |
| role destination | 1.000 | 0.087 / 0.072 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_sentences — catalog candidates (n = 600 of 4660)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.001 |
| FULL [depth given] | 1.000±0.001 |
| FULL [depth and negation given] | 1.000±0.001 |
| structure (depth, negation) | 1.000±0.000 |
| role action | 1.000±0.000 |
| head action | 1.000±0.000 |
| lexical order head action | 1.000±0.000 |
| lexical bag head action | 1.000±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role patient | 1.000±0.001 |
| head patient | 1.000±0.000 |
| lexical order head patient | 1.000±0.000 |
| lexical bag head patient | 1.000±0.000 |
| role destination | 1.000±0.000 |
| head destination | 1.000±0.000 |
| lexical order head destination | 0.872±0.000 |
| lexical bag head destination | 1.000±0.000 |
| occupancy: not given | 1.000±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.872±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 1.000±0.000 |
| speakers | 1.000±0.000 |

residual mean 0.241±0.003; status {'ok': 589, 'abstain': 11}

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
| FULL | 1.000±0.001 |
| FULL [depth given] | 1.000±0.001 |
| FULL [depth and negation given] | 1.000±0.001 |
| structure (depth, negation) | 1.000±0.000 |
| role action | 1.000±0.000 |
| head action | 1.000±0.000 |
| lexical order head action | 1.000±0.000 |
| lexical bag head action | 1.000±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role patient | 1.000±0.001 |
| head patient | 1.000±0.000 |
| lexical order head patient | 1.000±0.000 |
| lexical bag head patient | 1.000±0.000 |
| role destination | 1.000±0.000 |
| head destination | 1.000±0.000 |
| lexical order head destination | 0.872±0.000 |
| lexical bag head destination | 1.000±0.000 |
| occupancy: not given | 1.000±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.872±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 1.000±0.000 |
| speakers | 1.000±0.000 |

residual mean 0.241±0.003; status {'ok': 589, 'abstain': 11}

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
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
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
| FULL [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
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
| FULL | 0.999±0.002 |
| FULL [depth given] | 0.999±0.002 |
| FULL [depth and negation given] | 0.999±0.002 |
| structure (depth, negation) | 1.000±0.000 |
| role action | 1.000±0.000 |
| head action | 1.000±0.000 |
| lexical order head action | 1.000±0.000 |
| lexical bag head action | 1.000±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role patient | 0.999±0.002 |
| head patient | 1.000±0.000 |
| lexical order head patient | 1.000±0.000 |
| lexical bag head patient | 1.000±0.000 |
| role destination | 1.000±0.000 |
| head destination | 1.000±0.000 |
| lexical order head destination | 0.880±0.000 |
| lexical bag head destination | 1.000±0.000 |
| occupancy: not given | 1.000±0.000 |
| speakers | 1.000±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.880±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 1.000±0.000 |

residual mean 0.246±0.003; status {'ok': 290, 'abstain': 10}

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
- heldout_depth2: FULL 0.552
- heldout_depth2: structure (depth, negation) 0.638
- heldout_depth2: role patient 0.781
- heldout_depth2: speakers 0.762
- heldout_depth2: heads FULL (lexical order baseline) 0.857
- heldout_depth2: FULL [depth given] 0.838
- heldout_depth2: FULL [depth and negation given] 0.838
- heldout_sentences: heads FULL (lexical order baseline) 0.872
- new_identities: heads FULL (lexical order baseline) 0.857
- train (sample): heads FULL (lexical order baseline) 0.880

Formulations with the lowest FULL on the held-out splits (add examples of these formulations):
- heldout_depth2: `the PRINC asks that the PRINC says that ACT the PAT that are not ATTR .` FULL 0.000 (n 7)
- heldout_depth2: `the PRINC asks that the PRINC asks that ACT the NATTR PAT .` FULL 0.000 (n 4)
- heldout_depth2: `the PRINC asks that the PRINC asks that ACT the PAT that are not ATTR .` FULL 0.000 (n 2)
- heldout_depth2: `the PRINC says that the PRINC asks that ACT the PAT that are not ATTR .` FULL 0.000 (n 2)
- heldout_depth2: `the PRINC asks that the PRINC says that ACT the NATTR PAT .` FULL 0.167 (n 6)
- heldout_sentences: `the PRINC says that do not ACT the NATTR PAT .` FULL 0.889 (n 9)

## Structure given vs estimated (author C3)
`FULL` = the reporting depth and the event negation ESTIMATED from the state (what `sop formalize` does); `FULL [depth given]` = the depth GIVEN (as the repository scripts), negation estimated; `FULL [depth and negation given]`.

## Checklist
[x] addresses from the world [x] shared identities (one star function) [x] chance printed [x] seen / unseen axes separated [x] margin [x] >=3 seeds [x] trivial controls (lexical bag and lexical order baselines) [x] calibration (rule 18) [x] memory logged (rule 21) [x] thresholds on calibration rows (rule 25) [x] given vs learned stated (rule 20)
