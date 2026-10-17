Round 6, reference world (to / by / from given open); PRODUCT first: presence decided with the FITTED thresholds (argmax balanced accuracy on the calibration rows: training sample + calibration half of the held-out formulations); the pooled midpoint of the same rows beside it as the comparison (one more round). 5 seeds.

| split | role | readout | present read present | absent read absent | 'present' readings that abstain | false 'present' that abstain | true 'present' that abstain |
|---|---|---|---|---|---|---|---|
| heldout_formulations_unseen_transitions | recipient | product: fitted | 0.772±0.035 | — | 0.198±0.045 | — | 0.198±0.045 |
| heldout_formulations_unseen_transitions | recipient | comparison: pooled midpoint | 0.639±0.040 | — | | | |
| heldout_sentences | recipient | product: fitted | 1.000±0.000 | 0.991±0.014 | 0.002±0.003 | — | 0.000±0.000 |
| heldout_sentences | recipient | comparison: pooled midpoint | 1.000±0.000 | 1.000±0.000 | | | |
| heldout_depth3 | recipient | product: fitted | 1.000±0.000 | — | 0.100±0.025 | — | 0.100±0.025 |
| heldout_depth3 | recipient | comparison: pooled midpoint | 1.000±0.000 | — | | | |

heldout_formulations_unseen_transitions: heads FULL (roles) by readout
| readout | heads FULL | FULL |
|---|---|---|
| product: presence decided, fitted thresholds | 0.444±0.073 | 0.444±0.073 |
| comparison: presence decided, pooled midpoint | 0.418±0.061 | 0.418±0.061 |
| v16 protocol: every role read, not available to the product | 0.556±0.061 | 0.556±0.061 |
| round 4 F product (tool/runs/round4_39063ed) | 0.418±0.061 | |
| RESULTS 10b-quinquies (v16 protocol) | 0.627±0.024 | |

The product's errors inside the abstention band (|ratio - threshold| <= 0.05 OR identity margin < 0.05), from the stored ratios, role by role:
| seed | role | threshold (fitted) | false present: n / in presence band / in band (either) | missed present (unseen): n / in presence band / in band (either) | all errors in band |
|---|---|---|---|---|---|
| 0 | recipient | 0.409 | 2 / 2 / 2 | 21 / 5 / 12 | 14 / 23 = 0.609 |
| 1 | recipient | 0.371 | 2 / 2 / 2 | 18 / 4 / 9 | 11 / 20 = 0.550 |
| 2 | recipient | 0.416 | 28 / 21 / 28 | 16 / 8 / 11 | 39 / 44 = 0.886 |
| 3 | recipient | 0.488 | 2 / 2 / 2 | 19 / 1 / 8 | 10 / 21 = 0.476 |
| 4 | recipient | 0.356 | 3 / 3 / 3 | 25 / 10 / 18 | 21 / 28 = 0.750 |
POOLED recipient: 95 / 136 = 0.699 of the product's errors inside the abstention band.
Expectation (judgment 752853c §2): most. If not, the asymmetry option (a legend field 'prefer misses over false alarms for role X') is reported, not built.
