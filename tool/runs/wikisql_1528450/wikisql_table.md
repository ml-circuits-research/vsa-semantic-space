WikiSQL through the tool (question only; dev 8,421), mean ± std over the seeds of the run; RESULTS 8c (3 seeds, D = 512, 3 epochs, 56,355 train) and 8d (56k row) beside it.

| axis | package | RESULTS 8c (dev) | RESULTS 8d (56k, dev) |
|---|---|---|---|
| selected column | 0.816±0.003 | 0.810±0.004 | 0.816 |
| aggregator | 0.827±0.004 | 0.814±0.003 | 0.801 |
| condition column: the first gold condition column in the decoded SET (8c's condition-1 analog) | 0.682±0.032 | 0.632±0.004 (condition-1 column) | 0.626 |
| condition columns: the decoded set EXACT | 0.307±0.013 | — (8c: indexed slots) | — |
| FULL query (select + aggregate + the condition column set; operators and values not read) | 0.217±0.006 | 0.376±0.003 (all condition columns, indexed) | 0.368 |
| FULL, pooled-midpoint thresholds (the comparison) | 0.233±0.003 | | |
| condition presence: present read present / absent read absent | — / — | | |

seeds [0, 1, 2]; chance {'select': 3.1849162367029746e-05, 'aggregate': 0.00025940337224383917, 'condition': 3.1849162367029746e-05}

SOP round trip (write -> sop check -> read -> write; frames read back vs the SQL):
| file | questions | sop check | byte-identical | frames differing from the SQL |
|---|---|---|---|---|
| train.sop | 56355 | yes | yes | 0 |
| dev.sop | 8421 | yes | yes | 0 |
