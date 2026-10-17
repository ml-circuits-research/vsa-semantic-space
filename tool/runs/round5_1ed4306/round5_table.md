Round 5, reference world (to / by / from given open), role recipient; report rows: present = the reported half of heldout_formulations_unseen_transitions (n 87), absent = the held-out sentences without a recipient (n 603); 5 seeds
The absent class: the held-out formulations contain no recipient-less sentence, so the absent class can only come from the seen recipient-less forms. The fitted threshold therefore uses the calibration half (present) against the training recipient-less rows (absent). The report side uses disjoint rows: the reported halves against the held-out sentences' recipient-less rows.

| readout | threshold set by | threshold | present read present | absent read absent | balanced | present read present, all held-out formulations |
|---|---|---|---|---|---|---|
| (i) product today: pooled midpoint | training sample + calibration half of the held-out formulations (round 4 F) | 0.651±0.036 | 0.639±0.040 | 1.000±0.000 | 0.820±0.020 | 0.782±0.024 |
| (ii) fitted | max balanced accuracy on the calibration half alone (+ training recipient-less as absent) | 0.336±0.047 | 0.871±0.064 | 0.912±0.061 | 0.892±0.011 | 0.922±0.039 |
| round 4 F, for reference (tool/runs/round4_39063ed) | product readout: role recipient accuracy on the same split (not the presence-only number) | | 0.630±0.037 | | | |

| overlap (report rows) | value |
|---|---|
| share of present-class ratios below the absent class's 90th percentile | 0.103±0.016 |
| best achievable balanced accuracy (any threshold chosen ON the report rows: separability diagnostic, not a readout) | 0.908±0.008 |
| seeds where some threshold gives >= 0.9 present AND >= 0.9 absent on the report rows | 2 / 5 |

| seed | pooled thr | fitted thr | (i) present / absent | (ii) present / absent | overlap | best balanced | separable | median present (report) | absent p90 (report) | median present (calibration) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.622 | 0.409 | 0.644 / 1.000 | 0.759 / 0.997 | 0.126 | 0.897 | no | 1.022 | 0.314 | 1.132 |
| 1 | 0.608 | 0.280 | 0.644 / 1.000 | 0.885 / 0.922 | 0.080 | 0.916 | yes | 0.984 | 0.262 | 1.084 |
| 2 | 0.696 | 0.356 | 0.563 / 1.000 | 0.931 / 0.859 | 0.103 | 0.904 | no | 1.078 | 0.376 | 1.233 |
| 3 | 0.692 | 0.343 | 0.678 / 1.000 | 0.931 / 0.829 | 0.092 | 0.918 | yes | 1.079 | 0.373 | 1.230 |
| 4 | 0.639 | 0.292 | 0.667 / 1.000 | 0.851 / 0.954 | 0.115 | 0.904 | no | 1.103 | 0.260 | 1.182 |

DECISION RULE (judgment d8f8a98 Q5, written before the run): overlap -> the READER's limit, recorded; the fitted threshold separates -> the midpoint RULE's limit, the calibration becomes a fit.
OUTCOME: no threshold separates them in every seed (the classes overlap): the limit is the READER's and stays recorded.
