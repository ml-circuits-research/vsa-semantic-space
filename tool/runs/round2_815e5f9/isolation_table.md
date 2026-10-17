| run | change | candidates (per role) | seen formulations (roles) | new identities (roles) | unseen transitions (roles) | depth 3 UNSEEN (roles) | depth 3 UNSEEN FULL, depth given |
|---|---|---|---|---|---|---|---|
| P0 (c843f53) | baseline | agent 18 / object 18 / recipient 18 | 0.985±0.012 | 0.994±0.007 | 0.643±0.017 | 0.742±0.040 | 0.742±0.040 |
| P0 + v16 candidate sets | one change from P0: candidates (re-scoring, free) | seen words of the role's kind (12) / held-out words for new identities (6) | 0.985±0.011 | 1.000±0.000 | 0.647±0.016 | 0.742±0.040 | 0.742±0.040 |
| A: + gates given (to, by, from open) | one change: gates | agent 18 / object 18 / recipient 18 | 1.000±0.000 | 1.000±0.000 | 0.428±0.047 | 0.987±0.007 | 0.987±0.007 |
| B: A + v16 candidate sets | one change: candidates (re-scoring of A) | seen words of the role's kind (12) / held-out words for new identities (6) | 1.000±0.000 | 1.000±0.000 | 0.434±0.049 | 0.987±0.007 | 0.987±0.007 |
| C: B + v16 identity draw | one change: identities | seen words of the role's kind (12) / held-out words for new identities (6) | 1.000±0.000 | 1.000±0.000 | 0.436±0.035 | 0.973±0.017 | 0.973±0.017 |
| RESULTS (5 seeds) | | v16: 12 seen / 6 new | 1.000±0.000 | 1.000 | 0.638±0.015 | 0.813±0.030 learned gates (10b); 0.968 to/by/from given open (10b-quinquies) | — |

Commands world, depth 2 UNSEEN (5 seeds): the depth estimated from the open level gates vs the fit vs given
| run | FULL (level gates, primary) | FULL [fit estimate] | FULL [depth given] | depth estimate: level gates | depth estimate: fit | RESULTS 10d |
|---|---|---|---|---|---|---|
| P0 (c843f53, fit was the only estimator) | 0.590±0.059 | — | 0.857±0.025 | — | — | 0.854±0.031 |
| re-run, legend v2 (level openers) | 0.682±0.342 | 0.590±0.059 | 0.857±0.025 | 0.800±0.400 | 0.684±0.065 | 0.854±0.031 |
