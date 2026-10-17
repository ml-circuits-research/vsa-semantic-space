Reference world (to / by / from given open), heldout_formulations_unseen_transitions, 5 seeds; candidates = the role's catalog (18 per person / object role)

| run | rows | readout (protocol) | thresholds set on | heads FULL | agent | object | recipient |
|---|---|---|---|---|---|---|---|
| round 3 D | 180 (all) | product readout: presence decided | training rows | 0.428±0.047 | 0.833±0.000 | 0.811±0.063 | 0.619±0.047 |
| round 3 D | 180 (all) | v16 protocol: every role read, not available to the product | — | 0.596±0.048 | 0.833±0.000 | 0.811±0.063 | 0.839±0.004 |
| **round 4 F** | 87 (reported half) | **product readout: presence decided** | training rows + the calibration half of the held-out formulations | 0.418±0.061 | 0.839±0.000 | 0.779±0.086 | 0.630±0.037 |
| round 4 F | 87 (reported half) | presence decided | training rows only (the round-3 product, same rows) | 0.418±0.061 | 0.839±0.000 | 0.779±0.086 | 0.625±0.042 |
| round 4 F | 87 (reported half) | v16 protocol: every role read, not available to the product | — | 0.556±0.061 | 0.839±0.000 | 0.779±0.086 | 0.809±0.006 |
| RESULTS 10b-quinquies (v16) | | v16 protocol (C always read) | — | 0.627±0.024 | | | |

Presence thresholds, seed 0 (ratio to the structure score): agent: None -> None; object: None -> None; recipient: 0.636042517031197 -> 0.6222416271573876 (calibration: train (a sample of 400 training sentences, rng 1234+seed) + the CALIBRATION half of the held-out formulations (156 sentences, splits calibration_*), disjoint from the reported half (rule 25))
Expectation written before the run: the recipient recovers toward the every-role-read value (0.619 -> toward 0.839) without reading every role; if not, presence calibration across formulations is a recorded limit of the reader.
