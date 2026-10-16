P0 round trip, /home/salboaie/work/ml-circuits-research/vsa-semantic-space/.claude/worktrees/agent-a42b50d396b003d0e/tool/runs/p0_gpu_5seed_c843f53

| world | axis | package | tool FULL, structure estimated | tool FULL, depth given | RESULTS (5 seeds) | section |
|---|---|---|---|---|---|---|
| reference | seen formulations (A, B, C) | 0.985±0.012 | 0.985±0.012 | 0.985±0.012 | 1.000±0.000 | 10b |
| reference | predicate, surface lemma | 1.000±0.000 | 0.985±0.012 | 0.985±0.012 | 1.000 | 10b |
| reference | new identities | 0.994±0.007 | 0.994±0.007 | 0.994±0.007 | 1.000 | 10b |
| reference | role cues seen | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000 | 10b |
| reference | unseen transitions | 0.643±0.017 | 0.564±0.079 | 0.567±0.077 | 0.638±0.015 | 10b |
| reference | depth 3 UNSEEN | 0.742±0.040 | 0.742±0.040 | 0.742±0.040 | 0.813±0.030 (10b); 0.968 after E1b (10b-quinquies) | 10b |
| commands | depth 0: action | 1.000±0.000 | 1.000±0.001 | 1.000±0.001 | 1.000 | 10d |
| commands | depth 0: patient (with attribute and its negation) | 1.000±0.001 | 1.000±0.001 | 1.000±0.001 | 1.000 / 0.999 (patient / attribute) | 10d |
| commands | depth 0 FULL | 1.000±0.001 | 1.000±0.001 | 1.000±0.001 | 0.999 | 10d |
| commands | new patients FULL | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000 | 10d |
| commands | depth 2 UNSEEN FULL | 0.590±0.059 | 0.590±0.059 | 0.857±0.025 | 0.854±0.031 | 10d |
| commands | depth 2 UNSEEN: structure (depth + negation scope) | 0.684±0.065 | 0.590±0.059 | 0.857±0.025 | scope 0.997±0.003 (depth given) | 10d |
| quantifiers | depth 0: nouns | 0.999±0.004 | 0.995±0.006 | 0.995±0.006 | 0.999±0.002 (C2c) | 9c-sexies |
| quantifiers | depth 0: _scope (wide role / both / absent) | 0.996±0.005 | 0.995±0.006 | 0.995±0.006 | one quantifier 0.995, presence 1.000 (C2c) | 9c-sexies |
| quantifiers | two-quantifier scope, depth 0 | 0.986±0.017 | 0.995±0.006 | 0.995±0.006 | 0.996 (C2c) | 9c-sexies |
| quantifiers | depth 2 UNSEEN: two-quantifier scope | 0.917±0.063 | 0.962±0.029 | 0.962±0.029 | 0.966 (C2c, 5/5 seeds) | 9c-sexies |
| quantifiers | ambiguous (never trained): both | 0.993±0.013 | 0.987±0.016 | 0.987±0.016 | 0.993 (C2c) | 9c-sexies |
| quantifiers | determiner identity (outside FULL) | 0.744±0.006 | 0.995±0.006 | 0.995±0.006 | 0.950±0.013 (C2c) | 9c-sexies |
