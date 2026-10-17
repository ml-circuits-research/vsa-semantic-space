# REPORT — world `wikisql`, legend sha256:9b7ddb62bd18… (version 1)

Labels (rule 22): every number below is BUILT (code and measurement exist, computed by this training run). Nothing here is PLANNED or HYPOTHESIS except where a line says so.

## Given and learned (rule 20)
GIVEN: the legend's roles and their vectors, the identities (stars regenerated from the seed), the kinds (lexicon), the event synonym table, the reader block of the legend. LEARNED: the route book only — per-kind displacements of the path / bigram / chain codes and the gates (sigmoid per kind, from an all-closed start), by the single objective max cos(S, S_world) with structured per-sentence targets with pre-set roles; the model learns the assignment.

## Data
64776 annotated sentences from 2 files; provenance: {"wikisql-gold (the dataset's SQL)": 64776}. Audit: none in this world folder (audit/ empty) — agreement not measured. Shapes the v1 decoder cannot read (trained as targets, scored as FULL-wrong): none.

## Splits (written to runs/wikisql/splits.json before training; source: split_hints.json)
- dev: 8421 sentences
- train: 56355 sentences

The four splits of spec 6.4: seen formulations = `train` (+ `heldout_sentences`, the held-out SENTENCES of the seen formulations, 0% by hash); held-out FORMULATIONS (by template / file when the examples are templated, else by signature = the sentence with its fillers replaced by their kind; splits named heldout_formulations*); new identities (fillers never in training: []); plus held-out depth from None when the world says so. The lexical bag baseline and the lexical order baseline are trained on the same training rows and scored on the same rows as the reader.

## Kinds (spec 6.3)
782849 tokens in the examples; 0 with no kind in the lexicon (share 0.000), given the kind `unknown` (placed in the chart, never a filler, listed in `_unplaced`). Word -> kind table: model/catalog.json.

## Training
D=512, seeds [0, 1, 2], batch 32, lr 0.02, epochs 3, device cuda, codes path+chainL+bigram, gates learned (closed start). Code commit 1528450. Memory curve: runs/wikisql/train.log (trainer.train_loop, rule 21).

## Thresholds (rule 25)
Abstention tau = 0.05: stated, not fitted. Presence / attribute / determiner thresholds: midpoint of the mean present / absent ratio on CALIBRATION rows = a sample of the TRAINING split (never the scored rows):
- set_member:condition: threshold 0.341 (present 0.727, n 528 / absent 0.139, n 2031)
- presence:select: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 400, absent 0)
- presence:aggregate: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 400, absent 0)
- presence:condition: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 398, absent 2)

## Results (seed 0 model in model/; mean ± std over seeds when more than one)

Chance per role (1 / catalog candidates of the role's kinds): select 0.000, aggregate 0.000, condition 0.000

Two candidate sets: `catalog` = every catalog word of the role's kinds (what `sop formalize` does); `experiment` = as the repository scripts (seen splits: the words seen in training; new identities: the held-out words), for the P0 comparison with RESULTS.

### dev — catalog candidates (n = 8421 of 8421)
| axis | accuracy |
|---|---|
| FULL | 0.217±0.006 |
| FULL incl. attr (previous definition) | 0.001±0.000 |
| FULL [pooled midpoint (comparison)] | 0.233±0.003 |
| FULL incl. attr (previous definition) [pooled midpoint (comparison)] | 0.001±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| set condition: exact | 0.307±0.013 |
| set condition: first gold member in the decoded set | 0.682±0.032 |
| role select | 0.816±0.003 |
| head select | 0.816±0.003 |
| lexical order head select | 0.000±0.000 |
| lexical bag head select | 0.004±0.000 |
| occupancy: known filler | 0.951±0.010 |
| role aggregate | 0.827±0.004 |
| head aggregate | 0.827±0.004 |
| lexical order head aggregate | 0.000±0.000 |
| lexical bag head aggregate | 0.000±0.000 |
| role condition | 0.307±0.013 |
| head condition | 0.307±0.013 |
| lexical order head condition | 0.008±0.000 |
| lexical bag head condition | 0.003±0.000 |
| heads FULL (model) | 0.217±0.006 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.000±0.000 |
| role select [pooled midpoint (comparison)] | 0.816±0.003 |
| role aggregate [pooled midpoint (comparison)] | 0.827±0.004 |
| role condition [pooled midpoint (comparison)] | 0.336±0.003 |
| heads FULL (model) [pooled midpoint (comparison)] | 0.233±0.003 |
| occupancy: unknown filler | 0.913±0.022 |
| occupancy: not given | 0.510±0.097 |

residual mean 0.884±0.002; status {'abstain': 4391, 'ok': 4030}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| role select | 0.811 | 0.146 / 0.015 | 0.75 | 0.052 | 0.116 / 0.137 |
| roles (pooled) | 0.650 | 0.169 / 0.007 | 0.73 | 0.157 | 0.072 / 0.193 |
| role aggregate | 0.823 | 0.287 / 0.072 | 0.93 | 0.135 | 0.030 / 0.042 |
| role condition | 0.314 | 0.072 / 0.000 | 0.53 | 0.285 | 0.071 / 0.401 |

### train (sample) — catalog candidates (n = 300 of 56355)
| axis | accuracy |
|---|---|
| FULL | 0.303±0.028 |
| FULL incl. attr (previous definition) | 0.001±0.002 |
| structure (depth, negation) | 1.000±0.000 |
| depth estimate: fit (check) | 1.000±0.000 |
| set condition: exact | 0.381±0.032 |
| set condition: first gold member in the decoded set | 0.776±0.023 |
| role select | 0.902±0.007 |
| head select | 0.902±0.007 |
| lexical order head select | 0.000±0.000 |
| lexical bag head select | 0.010±0.000 |
| occupancy: known filler | 0.967±0.009 |
| role aggregate | 0.836±0.002 |
| head aggregate | 0.836±0.002 |
| lexical order head aggregate | 0.000±0.000 |
| lexical bag head aggregate | 0.000±0.000 |
| role condition | 0.381±0.032 |
| head condition | 0.381±0.032 |
| lexical order head condition | 0.007±0.000 |
| lexical bag head condition | 0.010±0.000 |
| heads FULL (model) | 0.303±0.028 |
| heads FULL (lexical order baseline) | 0.000±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.000±0.000 |
| occupancy: unknown filler | 0.940±0.027 |
| occupancy: not given | 0.833±0.236 |

residual mean 0.879±0.003; status {'abstain': 144, 'ok': 156}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| role select | 0.910 | 0.158 / 0.021 | 0.77 | 0.033 | 0.170 / 0.057 |
| roles (pooled) | 0.710 | 0.176 / 0.012 | 0.76 | 0.141 | 0.090 / 0.148 |
| role aggregate | 0.837 | 0.287 / 0.066 | 0.92 | 0.123 | 0.037 / 0.040 |
| role condition | 0.383 | 0.082 / 0.000 | 0.59 | 0.268 | 0.064 / 0.349 |

## Gates (seed 0; OPEN > 0.5)
which=1.00 what=1.00 the=0.93 than=0.67 is=0.66 of=0.61 C35=0.50 C42=0.34 who=0.32 C31=0.28 C47=0.27 C22=0.20 C0=0.19 C12=0.12 C39=0.12 C43=0.12 C5=0.12 unknown=0.12 C10=0.08 C44=0.07 away=0.00 C1=0.00 C15=0.00 C45=0.00 C40=0.00 home=0.00 C23=0.00 team=0.00 venue=0.00 country=0.00 C13=0.00 1=0.00 C14=0.00 C34=0.00 C11=0.00 location=0.00 has=0.00 result=0.00 name=0.00 for=0.00 a=0.00 C24=0.00 C6=0.00 C26=0.00 C4=0.00 C46=0.00 opponent=0.00 2=0.00 C8=0.00 record=0.00 C3=0.00 points=0.00 C17=0.00 C18=0.00 rank=0.00 C38=0.00 C25=0.00 was=0.00 C7=0.00 time=0.00 C20=0.00 when=0.00 score=0.00 s=0.00 position=0.00 attendance=0.00 C19=0.00 C33=0.00 C30=0.00 C41=0.00 with=0.00 game=0.00 C37=0.00 C29=0.00 and=0.00 C27=0.00 date=0.00 in=0.00 C28=0.00 year=0.00 C36=0.00 player=0.00 C2=0.00 number=0.00 C32=0.00 C9=0.00 C21=0.00 total=0.00 C16=0.00

## What is weak, what to add
- dev: FULL 0.220
- dev: FULL incl. attr (previous definition) 0.001
- dev: set condition: exact 0.316
- dev: set condition: first gold member in the decoded set 0.667
- dev: role select 0.811
- dev: head select 0.811
- dev: role aggregate 0.823
- dev: head aggregate 0.823
- dev: role condition 0.316
- dev: head condition 0.316
- dev: heads FULL (model) 0.220
- dev: heads FULL (lexical order baseline) 0.000
- dev: heads FULL (lexical bag baseline, trained on the same training rows) 0.000
- dev: FULL [pooled midpoint (comparison)] 0.230
- dev: role select [pooled midpoint (comparison)] 0.811
- dev: role aggregate [pooled midpoint (comparison)] 0.823
- dev: role condition [pooled midpoint (comparison)] 0.336
- dev: heads FULL (model) [pooled midpoint (comparison)] 0.230
- dev: FULL incl. attr (previous definition) [pooled midpoint (comparison)] 0.002
- dev: occupancy: not given 0.594
- train (sample): FULL 0.323
- train (sample): FULL incl. attr (previous definition) 0.003
- train (sample): set condition: exact 0.387
- train (sample): set condition: first gold member in the decoded set 0.758
- train (sample): role aggregate 0.837
- train (sample): head aggregate 0.837
- train (sample): role condition 0.387
- train (sample): head condition 0.387
- train (sample): heads FULL (model) 0.323
- train (sample): heads FULL (lexical order baseline) 0.000
- train (sample): heads FULL (lexical bag baseline, trained on the same training rows) 0.000

Formulations with the lowest FULL on the held-out splits (add examples of these formulations):
- dev: `what is pronunciation spelled free when pronunciation spelled checked is` FULL 0.000 (n 3)
- dev: `what is the party for south carolina 3` FULL 0.000 (n 2)
- dev: `name the most played` FULL 0.000 (n 2)
- dev: `what is the worst highest score` FULL 0.000 (n 2)
- dev: `what is the sexual abuse rate where the conflict is the burundi civil war` FULL 0.000 (n 2)

## Structure given vs estimated (author C3)
`FULL` = the reporting depth ESTIMATED by the primary estimator of model/config.json (`kinds`: the count of level-opener kinds by default; the open-gate count and the fit are the checks, `FULL [gate-count estimate]` / `FULL [fit estimate]`) and the event negation by the fit (what `sop formalize` does); `FULL [depth given]` = the depth GIVEN (as the repository scripts), negation estimated; `FULL [depth and negation given]`.

FULL is defined over the legend's questions: ['attr'] excluded (reported as its own axis); the previous definition (everything included) is printed once next to it (rule 16).

## Checklist
[x] addresses from the world [x] shared identities (one star function) [x] chance printed [x] seen / unseen axes separated [x] margin [x] >=3 seeds [x] trivial controls (lexical bag and lexical order baselines) [x] calibration (rule 18) [x] memory logged (rule 21) [x] thresholds on calibration rows (rule 25) [x] given vs learned stated (rule 20)
