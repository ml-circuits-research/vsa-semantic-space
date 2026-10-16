# REPORT — world `quantifiers`, legend sha256:c52ccef3fc2e… (version 2)

Labels (rule 22): every number below is BUILT (code and measurement exist, computed by this training run). Nothing here is PLANNED or HYPOTHESIS except where a line says so.

## Given and learned (rule 20)
GIVEN: the legend's roles and their vectors, the identities (stars regenerated from the seed), the kinds (lexicon), the event synonym table, the reader block of the legend. LEARNED: the route book only — per-kind displacements of the path / bigram / chain codes and the gates (sigmoid per kind, from an all-closed start), by the single objective max cos(S, S_world) with structured per-sentence targets with pre-set roles; the model learns the assignment.

## Data
40450 annotated sentences from 5 files; provenance: {'generator': 40450}. Audit: none in this world folder (audit/ empty) — agreement not measured. Shapes the v1 decoder cannot read (trained as targets, scored as FULL-wrong): none.

## Splits (written to runs/quantifiers/splits.json before training; source: split_hints.json)
- ambiguous_d0 (two unmarked quantifiers, never trained: _scope=both): 90 sentences
- depth2_affirmative (seen depth, fresh sentences): 90 sentences
- heldout_depth2_quantified: 90 sentences
- heldout_sentences: 3942 sentences
- new_identities: 180 sentences
- train: 36058 sentences

The four splits of spec 6.4: seen formulations = `train` (+ `heldout_sentences`, the held-out SENTENCES of the seen formulations, 10% by hash); held-out FORMULATIONS (by template / file when the examples are templated, else by signature = the sentence with its fillers replaced by their kind; splits named heldout_formulations*); new identities (fillers never in training: ['baker', 'drum', 'flag', 'kite', 'miner', 'monk', 'rope', 'sailor', 'scout', 'shell', 'stone', 'tailor']); plus held-out depth from None when the world says so. The lexical bag baseline and the lexical order baseline are trained on the same training rows and scored on the same rows as the reader.

## Kinds (spec 6.3)
498382 tokens in the examples; 0 with no kind in the lexicon (share 0.000), given the kind `unknown` (placed in the chart, never a filler, listed in `_unplaced`). Word -> kind table: model/catalog.json.

## Training
D=1024, seeds [0, 1, 2, 3, 4], batch 8, lr 0.02, epochs 1, device cuda, codes path+chainL+bigram, gates learned (closed start). Code commit c843f53. Memory curve: runs/quantifiers/train.log (trainer.train_loop, rule 21).

## Thresholds (rule 25)
Abstention tau = 0.05: stated, not fitted. Presence / attribute / determiner thresholds: midpoint of the mean present / absent ratio on CALIBRATION rows = a sample of the TRAINING split (never the scored rows):
- scope_term: threshold 0.460 (present 0.907, n 389 / absent 0.012, n 2011)
- presence:agent: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 400, absent 0)
- det_presence:agent: threshold 0.350 (present 0.587, n 142 / absent 0.112, n 258)
- presence:object: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 400, absent 0)
- det_presence:object: threshold 0.310 (present 0.500, n 121 / absent 0.120, n 279)
- presence:recipient: no threshold (one class missing in the calibration rows: read by the training rate (always / never); n present 400, absent 0)
- det_presence:recipient: threshold 0.404 (present 0.697, n 126 / absent 0.110, n 274)

## Results (seed 0 model in model/; mean ± std over seeds when more than one)

Chance per role (1 / catalog candidates of the role's kinds): agent 0.056, object 0.056, recipient 0.056

Two candidate sets: `catalog` = every catalog word of the role's kinds (what `sop formalize` does); `experiment` = as the repository scripts (seen splits: the words seen in training; new identities: the held-out words), for the P0 comparison with RESULTS.

### ambiguous_d0 (two unmarked quantifiers, never trained: _scope=both) — catalog candidates (n = 90 of 90)
| axis | accuracy |
|---|---|
| FULL | 0.987±0.016 |
| FULL incl. det (previous definition) | 0.987±0.016 |
| FULL [depth given] | 0.987±0.016 |
| FULL incl. det (previous definition) [depth given] | 0.987±0.016 |
| FULL [depth and negation given] | 0.987±0.016 |
| FULL incl. det (previous definition) [depth and negation given] | 0.987±0.016 |
| structure (depth, negation) | 1.000±0.000 |
| _scope (wide role / both / absent) | 0.993±0.013 |
| _scope on two-quantifier sentences | 0.993±0.013 |
| role agent | 0.987±0.016 |
| head agent | 0.993±0.013 |
| lexical order head agent | 0.500±0.000 |
| lexical bag head agent | 0.544±0.000 |
| determiner identity (outside FULL when the legend excludes it) | 0.994±0.007 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 1.000±0.000 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.500±0.000 |
| lexical bag head recipient | 0.544±0.000 |
| heads FULL (model) | 0.993±0.013 |
| heads FULL (lexical order baseline) | 0.500±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.544±0.000 |

residual mean 0.498±0.019; status {'ok': 89, 'abstain': 1}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| scope | 1.000 | 0.136 / 0.081 | 0.99 | 0.000 | 0.011 / 0.000 |
| role agent | 1.000 | 0.262 / 0.193 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.293 / 0.196 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.366 / 0.320 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.249 / 0.173 | 1.00 | 0.000 | 0.000 / 0.000 |

### ambiguous_d0 (two unmarked quantifiers, never trained: _scope=both) — experiment candidates (n = 90 of 90)
| axis | accuracy |
|---|---|
| FULL | 0.987±0.016 |
| FULL incl. det (previous definition) | 0.987±0.016 |
| FULL [depth given] | 0.987±0.016 |
| FULL incl. det (previous definition) [depth given] | 0.987±0.016 |
| FULL [depth and negation given] | 0.987±0.016 |
| FULL incl. det (previous definition) [depth and negation given] | 0.987±0.016 |
| structure (depth, negation) | 1.000±0.000 |
| _scope (wide role / both / absent) | 0.993±0.013 |
| _scope on two-quantifier sentences | 0.993±0.013 |
| role agent | 0.987±0.016 |
| head agent | 0.993±0.013 |
| lexical order head agent | 0.500±0.000 |
| lexical bag head agent | 0.544±0.000 |
| determiner identity (outside FULL when the legend excludes it) | 0.994±0.007 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 1.000±0.000 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.500±0.000 |
| lexical bag head recipient | 0.544±0.000 |
| heads FULL (model) | 0.993±0.013 |
| heads FULL (lexical order baseline) | 0.500±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.544±0.000 |

residual mean 0.498±0.019; status {'ok': 89, 'abstain': 1}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| scope | 1.000 | 0.136 / 0.081 | 0.99 | 0.000 | 0.011 / 0.000 |
| role agent | 1.000 | 0.264 / 0.193 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.294 / 0.196 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.367 / 0.320 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.250 / 0.173 | 1.00 | 0.000 | 0.000 / 0.000 |

### depth2_affirmative (seen depth, fresh sentences) — catalog candidates (n = 90 of 90)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL incl. det (previous definition) | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL incl. det (previous definition) [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| FULL incl. det (previous definition) [depth and negation given] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| _scope (wide role / both / absent) | 1.000±0.000 |
| role agent | 1.000±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.500±0.000 |
| lexical bag head agent | 0.456±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 1.000±0.000 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.500±0.000 |
| lexical bag head recipient | 0.456±0.000 |
| speakers | 1.000±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.500±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.456±0.000 |

residual mean 0.484±0.018; status {'ok': 90}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| scope | 1.000 | 0.115 / 0.090 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.257 / 0.166 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.255 / 0.162 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.288 / 0.250 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.220 / 0.141 | 1.00 | 0.000 | 0.000 / 0.000 |
| speakers | 1.000 | 0.472 / 0.445 | 1.00 | 0.000 | 0.000 / 0.000 |

### depth2_affirmative (seen depth, fresh sentences) — experiment candidates (n = 90 of 90)
| axis | accuracy |
|---|---|
| FULL | 1.000±0.000 |
| FULL incl. det (previous definition) | 1.000±0.000 |
| FULL [depth given] | 1.000±0.000 |
| FULL incl. det (previous definition) [depth given] | 1.000±0.000 |
| FULL [depth and negation given] | 1.000±0.000 |
| FULL incl. det (previous definition) [depth and negation given] | 1.000±0.000 |
| structure (depth, negation) | 1.000±0.000 |
| _scope (wide role / both / absent) | 1.000±0.000 |
| role agent | 1.000±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.500±0.000 |
| lexical bag head agent | 0.456±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 1.000±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 1.000±0.000 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.500±0.000 |
| lexical bag head recipient | 0.456±0.000 |
| speakers | 1.000±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.500±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.456±0.000 |

residual mean 0.484±0.018; status {'ok': 90}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| scope | 1.000 | 0.115 / 0.090 | 1.00 | 0.000 | 0.000 / 0.000 |
| role agent | 1.000 | 0.258 / 0.166 | 1.00 | 0.000 | 0.000 / 0.000 |
| roles (pooled) | 1.000 | 0.256 / 0.162 | 1.00 | 0.000 | 0.000 / 0.000 |
| role object | 1.000 | 0.288 / 0.250 | 1.00 | 0.000 | 0.000 / 0.000 |
| role recipient | 1.000 | 0.221 / 0.141 | 1.00 | 0.000 | 0.000 / 0.000 |
| speakers | 1.000 | 0.472 / 0.445 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_depth2_quantified — catalog candidates (n = 90 of 90)
| axis | accuracy |
|---|---|
| FULL | 0.962±0.029 |
| FULL incl. det (previous definition) | 0.658±0.033 |
| FULL [depth given] | 0.962±0.029 |
| FULL incl. det (previous definition) [depth given] | 0.658±0.033 |
| FULL [depth and negation given] | 0.962±0.029 |
| FULL incl. det (previous definition) [depth and negation given] | 0.658±0.033 |
| structure (depth, negation) | 1.000±0.000 |
| _scope (wide role / both / absent) | 0.962±0.029 |
| role agent | 0.842±0.022 |
| head agent | 0.996±0.009 |
| lexical order head agent | 0.500±0.000 |
| lexical bag head agent | 0.600±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 0.933±0.010 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.869±0.016 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.500±0.000 |
| lexical bag head recipient | 0.600±0.000 |
| determiner identity (outside FULL when the legend excludes it) | 0.756±0.029 |
| speakers | 1.000±0.000 |
| heads FULL (model) | 0.996±0.009 |
| heads FULL (lexical order baseline) | 0.500±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.600±0.000 |
| _scope on two-quantifier sentences | 0.917±0.063 |

residual mean 0.639±0.013; status {'abstain': 36, 'ok': 54}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| scope | 0.989 | 0.086 / 0.042 | 0.89 | 0.000 | 0.100 / 0.011 |
| role agent | 0.867 | 0.206 / 0.095 | 0.99 | 0.133 | 0.011 / 0.000 |
| roles (pooled) | 0.900 | 0.207 / 0.125 | 1.00 | 0.100 | 0.004 / 0.000 |
| role object | 0.944 | 0.236 / 0.190 | 1.00 | 0.056 | 0.000 / 0.000 |
| role recipient | 0.889 | 0.180 / 0.113 | 1.00 | 0.111 | 0.000 / 0.000 |
| speakers | 1.000 | 0.392 / 0.353 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_depth2_quantified — experiment candidates (n = 90 of 90)
| axis | accuracy |
|---|---|
| FULL | 0.962±0.029 |
| FULL incl. det (previous definition) | 0.658±0.033 |
| FULL [depth given] | 0.962±0.029 |
| FULL incl. det (previous definition) [depth given] | 0.658±0.033 |
| FULL [depth and negation given] | 0.962±0.029 |
| FULL incl. det (previous definition) [depth and negation given] | 0.658±0.033 |
| structure (depth, negation) | 1.000±0.000 |
| _scope (wide role / both / absent) | 0.962±0.029 |
| role agent | 0.842±0.022 |
| head agent | 0.996±0.009 |
| lexical order head agent | 0.500±0.000 |
| lexical bag head agent | 0.600±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 0.933±0.010 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.869±0.016 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.500±0.000 |
| lexical bag head recipient | 0.600±0.000 |
| determiner identity (outside FULL when the legend excludes it) | 0.756±0.029 |
| speakers | 1.000±0.000 |
| heads FULL (model) | 0.996±0.009 |
| heads FULL (lexical order baseline) | 0.500±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.600±0.000 |
| _scope on two-quantifier sentences | 0.917±0.063 |

residual mean 0.639±0.013; status {'abstain': 36, 'ok': 54}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| scope | 0.989 | 0.086 / 0.042 | 0.89 | 0.000 | 0.100 / 0.011 |
| role agent | 0.867 | 0.206 / 0.095 | 0.99 | 0.133 | 0.011 / 0.000 |
| roles (pooled) | 0.900 | 0.208 / 0.125 | 1.00 | 0.100 | 0.004 / 0.000 |
| role object | 0.944 | 0.236 / 0.190 | 1.00 | 0.056 | 0.000 / 0.000 |
| role recipient | 0.889 | 0.180 / 0.113 | 1.00 | 0.111 | 0.000 / 0.000 |
| speakers | 1.000 | 0.392 / 0.353 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_sentences — catalog candidates (n = 600 of 3942)
| axis | accuracy |
|---|---|
| FULL | 0.995±0.006 |
| FULL incl. det (previous definition) | 0.785±0.005 |
| FULL [depth given] | 0.995±0.006 |
| FULL incl. det (previous definition) [depth given] | 0.785±0.005 |
| FULL [depth and negation given] | 0.995±0.006 |
| FULL incl. det (previous definition) [depth and negation given] | 0.785±0.005 |
| structure (depth, negation) | 1.000±0.000 |
| _scope (wide role / both / absent) | 0.996±0.005 |
| role agent | 0.905±0.005 |
| head agent | 0.997±0.004 |
| lexical order head agent | 0.498±0.000 |
| lexical bag head agent | 0.475±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 0.940±0.001 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| determiner identity (outside FULL when the legend excludes it) | 0.744±0.006 |
| role recipient | 0.939±0.003 |
| head recipient | 0.999±0.001 |
| lexical order head recipient | 0.498±0.000 |
| lexical bag head recipient | 0.475±0.000 |
| heads FULL (model) | 0.996±0.005 |
| heads FULL (lexical order baseline) | 0.498±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.475±0.000 |
| speakers | 1.000±0.000 |
| _scope on two-quantifier sentences | 0.986±0.017 |

residual mean 0.519±0.018; status {'ok': 492, 'abstain': 108}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| scope | 1.000 | 0.149 / 0.100 | 0.98 | 0.000 | 0.020 / 0.000 |
| role agent | 0.908 | 0.290 / 0.202 | 1.00 | 0.092 | 0.000 / 0.000 |
| roles (pooled) | 0.930 | 0.321 / 0.203 | 1.00 | 0.070 | 0.000 / 0.000 |
| role object | 0.940 | 0.395 / 0.292 | 1.00 | 0.060 | 0.000 / 0.000 |
| role recipient | 0.942 | 0.277 / 0.177 | 1.00 | 0.058 | 0.000 / 0.000 |
| speakers | 1.000 | 0.448 / 0.362 | 1.00 | 0.000 | 0.000 / 0.000 |

### heldout_sentences — experiment candidates (n = 600 of 3942)
| axis | accuracy |
|---|---|
| FULL | 0.995±0.006 |
| FULL incl. det (previous definition) | 0.785±0.005 |
| FULL [depth given] | 0.995±0.006 |
| FULL incl. det (previous definition) [depth given] | 0.785±0.005 |
| FULL [depth and negation given] | 0.995±0.006 |
| FULL incl. det (previous definition) [depth and negation given] | 0.785±0.005 |
| structure (depth, negation) | 1.000±0.000 |
| _scope (wide role / both / absent) | 0.996±0.005 |
| role agent | 0.905±0.005 |
| head agent | 0.997±0.004 |
| lexical order head agent | 0.498±0.000 |
| lexical bag head agent | 0.475±0.000 |
| occupancy: known filler | 1.000±0.000 |
| role object | 0.940±0.001 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| determiner identity (outside FULL when the legend excludes it) | 0.744±0.006 |
| role recipient | 0.939±0.003 |
| head recipient | 0.999±0.001 |
| lexical order head recipient | 0.498±0.000 |
| lexical bag head recipient | 0.475±0.000 |
| heads FULL (model) | 0.996±0.005 |
| heads FULL (lexical order baseline) | 0.498±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.475±0.000 |
| speakers | 1.000±0.000 |
| _scope on two-quantifier sentences | 0.986±0.017 |

residual mean 0.519±0.018; status {'ok': 492, 'abstain': 108}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| scope | 1.000 | 0.149 / 0.100 | 0.98 | 0.000 | 0.020 / 0.000 |
| role agent | 0.908 | 0.291 / 0.202 | 1.00 | 0.092 | 0.000 / 0.000 |
| roles (pooled) | 0.930 | 0.322 / 0.203 | 1.00 | 0.070 | 0.000 / 0.000 |
| role object | 0.940 | 0.396 / 0.292 | 1.00 | 0.060 | 0.000 / 0.000 |
| role recipient | 0.942 | 0.277 / 0.177 | 1.00 | 0.058 | 0.000 / 0.000 |
| speakers | 1.000 | 0.448 / 0.362 | 1.00 | 0.000 | 0.000 / 0.000 |

### new_identities — catalog candidates (n = 180 of 180)
| axis | accuracy |
|---|---|
| FULL | 0.993±0.006 |
| FULL incl. det (previous definition) | 0.654±0.006 |
| FULL [depth given] | 0.993±0.006 |
| FULL incl. det (previous definition) [depth given] | 0.654±0.006 |
| FULL [depth and negation given] | 0.993±0.006 |
| FULL incl. det (previous definition) [depth and negation given] | 0.654±0.006 |
| structure (depth, negation) | 1.000±0.000 |
| _scope (wide role / both / absent) | 0.994±0.005 |
| _scope on two-quantifier sentences | 0.989±0.010 |
| role agent | 0.883±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.500±0.000 |
| lexical bag head agent | 0.478±0.000 |
| determiner identity (outside FULL when the legend excludes it) | 0.771±0.004 |
| occupancy: unknown filler | 1.000±0.000 |
| role object | 0.867±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.904±0.006 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.500±0.000 |
| lexical bag head recipient | 0.522±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.500±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.000±0.000 |

residual mean 0.585±0.017; status {'ok': 132, 'abstain': 48}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| scope | 0.994 | 0.143 / 0.101 | 0.97 | 0.000 | 0.022 / 0.006 |
| role agent | 0.883 | 0.281 / 0.206 | 1.00 | 0.117 | 0.000 / 0.000 |
| roles (pooled) | 0.885 | 0.314 / 0.202 | 1.00 | 0.115 | 0.000 / 0.000 |
| role object | 0.867 | 0.391 / 0.325 | 1.00 | 0.133 | 0.000 / 0.000 |
| role recipient | 0.906 | 0.270 / 0.171 | 1.00 | 0.094 | 0.000 / 0.000 |

### new_identities — experiment candidates (n = 180 of 180)
| axis | accuracy |
|---|---|
| FULL | 0.993±0.006 |
| FULL incl. det (previous definition) | 0.654±0.006 |
| FULL [depth given] | 0.993±0.006 |
| FULL incl. det (previous definition) [depth given] | 0.654±0.006 |
| FULL [depth and negation given] | 0.993±0.006 |
| FULL incl. det (previous definition) [depth and negation given] | 0.654±0.006 |
| structure (depth, negation) | 1.000±0.000 |
| _scope (wide role / both / absent) | 0.994±0.005 |
| _scope on two-quantifier sentences | 0.989±0.010 |
| role agent | 0.883±0.000 |
| head agent | 1.000±0.000 |
| lexical order head agent | 0.500±0.000 |
| lexical bag head agent | 0.478±0.000 |
| determiner identity (outside FULL when the legend excludes it) | 0.771±0.004 |
| occupancy: unknown filler | 1.000±0.000 |
| role object | 0.867±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.904±0.006 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.500±0.000 |
| lexical bag head recipient | 0.522±0.000 |
| heads FULL (model) | 1.000±0.000 |
| heads FULL (lexical order baseline) | 0.500±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.000±0.000 |

residual mean 0.585±0.017; status {'ok': 132, 'abstain': 48}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| scope | 0.994 | 0.143 / 0.101 | 0.97 | 0.000 | 0.022 / 0.006 |
| role agent | 0.883 | 0.288 / 0.223 | 1.00 | 0.117 | 0.000 / 0.000 |
| roles (pooled) | 0.885 | 0.327 / 0.222 | 1.00 | 0.115 | 0.000 / 0.000 |
| role object | 0.867 | 0.414 / 0.353 | 1.00 | 0.133 | 0.000 / 0.000 |
| role recipient | 0.906 | 0.278 / 0.172 | 1.00 | 0.094 | 0.000 / 0.000 |

### train (sample) — catalog candidates (n = 300 of 36058)
| axis | accuracy |
|---|---|
| FULL | 0.995±0.005 |
| FULL incl. det (previous definition) | 0.774±0.004 |
| FULL [depth given] | 0.995±0.005 |
| FULL incl. det (previous definition) [depth given] | 0.774±0.004 |
| FULL [depth and negation given] | 0.995±0.005 |
| FULL incl. det (previous definition) [depth and negation given] | 0.774±0.004 |
| structure (depth, negation) | 1.000±0.000 |
| _scope (wide role / both / absent) | 0.997±0.004 |
| role agent | 0.934±0.004 |
| head agent | 0.998±0.003 |
| lexical order head agent | 0.497±0.000 |
| lexical bag head agent | 0.530±0.000 |
| determiner identity (outside FULL when the legend excludes it) | 0.774±0.005 |
| occupancy: known filler | 1.000±0.000 |
| role object | 0.913±0.000 |
| head object | 1.000±0.000 |
| lexical order head object | 1.000±0.000 |
| lexical bag head object | 1.000±0.000 |
| role recipient | 0.925±0.002 |
| head recipient | 1.000±0.000 |
| lexical order head recipient | 0.497±0.000 |
| lexical bag head recipient | 0.530±0.000 |
| heads FULL (model) | 0.998±0.003 |
| heads FULL (lexical order baseline) | 0.497±0.000 |
| heads FULL (lexical bag baseline, trained on the same training rows) | 0.530±0.000 |
| speakers | 1.000±0.000 |
| _scope on two-quantifier sentences | 0.993±0.010 |

residual mean 0.521±0.017; status {'ok': 248, 'abstain': 52}

| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |
|---|---|---|---|---|---|
| scope | 1.000 | 0.146 / 0.098 | 0.99 | 0.000 | 0.013 / 0.000 |
| role agent | 0.937 | 0.286 / 0.194 | 1.00 | 0.063 | 0.000 / 0.000 |
| roles (pooled) | 0.926 | 0.316 / 0.198 | 1.00 | 0.074 | 0.000 / 0.000 |
| role object | 0.913 | 0.390 / 0.299 | 1.00 | 0.087 | 0.000 / 0.000 |
| role recipient | 0.927 | 0.272 / 0.168 | 1.00 | 0.073 | 0.000 / 0.000 |
| speakers | 1.000 | 0.440 / 0.351 | 1.00 | 0.000 | 0.000 / 0.000 |

## Gates (seed 0; OPEN > 0.5)
that=1.00 to=1.00 from=1.00 by=0.99 .=0.37 unknown=0.12 claims=0.00 P=0.00 NP=0.00 given=0.00 sells=0.00 sends=0.00 shown=0.00 lends=0.00 accepts=0.00 gives=0.00 offers=0.00 sold=0.00 shows=0.00 sent=0.00 buys=0.00 offered=0.00 gets=0.00 says=0.00 sees=0.00 receives=0.00 lent=0.00 borrows=0.00 ,=0.00 certain=0.00 each=0.00 a=0.00 some=0.00 every=0.00 the=0.00 COP=0.00 NO=0.00

## What is weak, what to add
- ambiguous_d0 (two unmarked quantifiers, never trained: _scope=both): heads FULL (lexical order baseline) 0.500
- ambiguous_d0 (two unmarked quantifiers, never trained: _scope=both): heads FULL (lexical bag baseline, trained on the same training rows) 0.544
- depth2_affirmative (seen depth, fresh sentences): heads FULL (lexical order baseline) 0.500
- depth2_affirmative (seen depth, fresh sentences): heads FULL (lexical bag baseline, trained on the same training rows) 0.456
- heldout_depth2_quantified: FULL incl. det (previous definition) 0.700
- heldout_depth2_quantified: role agent 0.867
- heldout_depth2_quantified: role recipient 0.889
- heldout_depth2_quantified: determiner identity (outside FULL when the legend excludes it) 0.794
- heldout_depth2_quantified: heads FULL (lexical order baseline) 0.500
- heldout_depth2_quantified: heads FULL (lexical bag baseline, trained on the same training rows) 0.600
- heldout_depth2_quantified: FULL incl. det (previous definition) [depth given] 0.700
- heldout_depth2_quantified: FULL incl. det (previous definition) [depth and negation given] 0.700
- heldout_sentences: FULL incl. det (previous definition) 0.790
- heldout_sentences: determiner identity (outside FULL when the legend excludes it) 0.750
- heldout_sentences: heads FULL (lexical order baseline) 0.498
- heldout_sentences: heads FULL (lexical bag baseline, trained on the same training rows) 0.475
- heldout_sentences: FULL incl. det (previous definition) [depth given] 0.790
- heldout_sentences: FULL incl. det (previous definition) [depth and negation given] 0.790
- new_identities: FULL incl. det (previous definition) 0.656
- new_identities: role agent 0.883
- new_identities: determiner identity (outside FULL when the legend excludes it) 0.772
- new_identities: role object 0.867
- new_identities: heads FULL (lexical order baseline) 0.500
- new_identities: heads FULL (lexical bag baseline, trained on the same training rows) 0.000
- new_identities: FULL incl. det (previous definition) [depth given] 0.656
- new_identities: FULL incl. det (previous definition) [depth and negation given] 0.656
- train (sample): FULL incl. det (previous definition) 0.777
- train (sample): determiner identity (outside FULL when the legend excludes it) 0.778
- train (sample): heads FULL (lexical order baseline) 0.497
- train (sample): heads FULL (lexical bag baseline, trained on the same training rows) 0.530
- train (sample): FULL incl. det (previous definition) [depth given] 0.777
- train (sample): FULL incl. det (previous definition) [depth and negation given] 0.777

Formulations with the lowest FULL on the held-out splits (add examples of these formulations):
- heldout_depth2_quantified: `P claims that P claims that a certain NP accepts the NO from some NP .` FULL 0.000 (n 1)
- new_identities: `each NP receives the NO from every NP .` FULL 0.000 (n 1)

## Structure given vs estimated (author C3)
`FULL` = the reporting depth and the event negation ESTIMATED from the state (what `sop formalize` does); `FULL [depth given]` = the depth GIVEN (as the repository scripts), negation estimated; `FULL [depth and negation given]`.

FULL is defined over the legend's questions: ['det'] excluded (reported as its own axis); the previous definition (everything included) is printed once next to it (rule 16).

## Checklist
[x] addresses from the world [x] shared identities (one star function) [x] chance printed [x] seen / unseen axes separated [x] margin [x] >=3 seeds [x] trivial controls (lexical bag and lexical order baselines) [x] calibration (rule 18) [x] memory logged (rule 21) [x] thresholds on calibration rows (rule 25) [x] given vs learned stated (rule 20)
