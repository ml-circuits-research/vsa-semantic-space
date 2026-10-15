# How to run

Requirements: Python 3.10+, numpy. That is all. NO GPU is needed: all the experiments here run in 0.5-5 minutes on an ordinary CPU.
(The GPU becomes useful only at D >= 4096, a vocabulary of thousands of words and real text; then the port to torch with complex tensors is direct:
 bind = element-wise multiplication, bundle = addition, and the gradients come from autograd, no longer written by hand.)

Files, in historical order (each is self-contained or imports the previous one):
- sim_deplasari.py      the first experiments (A prediction, B contrastive, C consequence, D prototypes, E generative). `python3 sim_deplasari.py`
- f.py, f2.py           word factorization (displacement per category); report per role. `python3 f2.py`
- camp.py               the synthetic world + v1 (historical; its buffer and zoom-in are NOT necessary, see round 2). `python3 camp.py`
- gh.py                 G (deferred addressing) + H (unseen identities)
- i2.py                 I: categories discovered unsupervised (iterative clustering)
- j.py                  J: capacity (D, number of identities, adjuncts)
- k.py                  K: hierarchical composition "X spune ca <propozitie>"
- mlp_control.py        CONTROL: MLP on an orderless bundle (baseline 'no order, with capacity')
- control_transformer.py  CONTROL: small transformer (numpy, manual backprop, gradient-check included) vs VSA on the same task
- v2.py, v3.py, gates.py, nest2.py   ROUND 2: expectations, dual, learned gates, nesting without rules
- v4.py                 THE FINAL MODEL (round 3): redundant addresses (path, bigram, chain, pos); evaluation according to EVALUATION_PRINCIPLES.md. `python3 v4.py`
- EVALUATION_PRINCIPLES.md  evaluation rules (to be read before any new experiment)
- RESULTS.md            all results and conclusions
- INTERPRETATION.md     how to read each metric, what would refute the thesis, traps in the figures
- POSITIONING.md        name, figure 1, three possible articles, the barrier, controls, what to cite
- README_DGX.md         the validation plan on the DGX: experiments with works/does-not-work criteria, data sources, torch port
- log_*.txt             the raw outputs of the runs

Main parameters: D (hypervector dimension), n_sent (training sentences), lr (Adam, 0.02 is good), persons/objects.

Suggested next steps (in order of value):
1. [SUPERSEDED after round 2: the hand-set triggers were not necessary; the gates are learned by search (gates.py).] Next come the gates by
   GRADIENT, on >=3 worlds — see 00_READ_FIRST.md §5.2. The valid plan is in 00_READ_FIRST.md §5-6.
2. Real text in a closed domain; consequence = the state of the next sentence (self-supervised) instead of the synthetic world.
3. Capacity: D=4096-16384 and sparse block codes for cleaner cleanup at large vocabulary.
4. Stronger control: larger transformer + more data, to see where the efficiency gap closes.
