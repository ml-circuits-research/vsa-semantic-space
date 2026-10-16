# tool/ — the reader as a tool: SOP Lang, the `sop` CLI, the worlds server, the train-world skill

P0 of the programme. Specification: `tool/SPEC_worlds_server_sop_cli.md` (complete, sections 0-10, with the `_scope=` reserved line and the
multi-world dispatch of section 9) and `docs/specs/INDICATIONS_tool_agent.md` (models by role, build order, the six tests); the author's
instructions at 78d36e8: `experiment/INSTRUCTIONS_2026-10-16_78d36e8.md` (Q5, C2-C4). Nothing here changes the model or the measurement rules
of `experiment/`: the package imports `experiment/v8_gates.V8` and `experiment/trainer.train_loop` unchanged (nothing under `experiment/` was
modified or copied).

**Given and learned (rule 20).** GIVEN: the legend's roles (and their vectors), the word identities (stars), the kinds of words (lexicon),
the event synonym table, the legend's reader block (readout rules). LEARNED: the route book only — per-kind displacements of the path /
bigram / chain codes, the gates, and the beacon-relative displacements where the legend names the emitters — by the single generative
objective max cos(S, S_world), with structured per-sentence targets with pre-set roles; the model learns the assignment.

Every number in this file points to a committed run folder under `tool/runs/` or to a test named here; labels per rule 22.

## Install and run

Python >= 3.10 with numpy (inference) and torch (training only); Node.js >= 18 (server; standard library only). No other packages.
```
tool/bin/sop --help                               # uses the repository .venv when present ($SOP_PYTHON overrides); or: pip install -e tool/
tool/bin/sop export --out tool/worlds             # the three constructed worlds as world folders (small: 2000 sentences each)
tool/bin/sop check --legend tool/worlds/commands/legend.json tool/worlds/commands/examples/*.sop
tool/bin/sop train tool/worlds/commands --D 256 --epochs 3            # CPU smoke; writes model/, runs/<ts>/, status.json
echo "do not delete the unsaved files ." | tool/bin/sop formalize --model tool/worlds/commands/model -
tool/bin/sop formalize --worlds tool/worlds sentences.txt            # every world's reader reads; the dispatcher picks by fit (section 9)
tool/bin/sop diff a.sop b.sop [--model <dir>]     # verdict wires
tool/bin/sop stats --model tool/worlds/commands/model
node tool/server.js --root <DATA_ROOT> --port 8080 [--host 0.0.0.0]  # NO authentication; localhost by default
tool/tests/run_all.sh                              # the six tests (CPU)
tool/test_sop.sh                                   # the CPU smoke test: export, check, smoke train, formalize 3, diff 2, stats
python3 tool/skills/train-world/scripts/loop.py run <world> [--annotator llm|oracle:<gold dir>]   # the skill's loop (spec 6)
```
Exit codes (spec 5): 0 ok; 1 input / validation error; 2 model missing or legend mismatch. Output is deterministic for the same model and input
(tested byte for byte).

## What is implemented (labels: BUILT = code and test exist)

| piece | where | label |
|---|---|---|
| SOP Lang reader / writer (directives, wires, `$ref`, `&`, `|`, `~`, `?`, quoting; byte-for-byte round trip of the canonical layout) | `sop/soplang.py` | BUILT (test 1) |
| `sop check` (every check of 3.6 with line and reason; several worlds per file via `_world`; `_scope` only on event wires, a legend role or `both`; legend consistency: reserved-name collisions, events without verbs) | `sop/check.py`, `sop/legend.py` | BUILT (test 1) |
| target states of wires (events, entity, not / must / may / if / says / and, sets, alternatives, `~`; the quantifier world v2 terms R_wide / R_q) | `sop/world.py` | BUILT |
| `sop train` (spec 6.3-6.6: kinds with `unknown`, the four splits before training, V8 + trainer.train_loop, thresholds on calibration rows, metrics.json with every field of 6.6, REPORT.md, `--eval_only`, `--run_dir`, `--no_ckpt`) | `sop/train.py` | BUILT (tests 2, 5) |
| `sop formalize` (numpy only; every wire with `_margin`, `_status`, `_empty`, `_depth`; `_residual`, `_unplaced`, `_world`, `_verb`, `_scope`; `--json`; `--worlds` / several `--model` = dispatcher by fit with `_world_margin`, `_world=none`) | `sop/infer.py`, `sop/cli.py` | BUILT (test 2) |
| `sop diff` (four verdicts with margins; per world) and `sop stats` | `sop/diff.py`, `sop/cli.py` | BUILT (test 3) |
| the server (all routes of section 2, atomic writes, path normalization, two shell-outs, one page) | `server.js` | BUILT (tests 4, 6) |
| the three constructed worlds as world folders; the quantifier world VERSION 2 (C2c) with v1 under `versions/v1/` | `export_worlds.py` (`sop export`) | BUILT (test 2) |
| the train-world skill, spec 6 steps 1-8 and the acceptance of a training | `skills/train-world/` | BUILT offline (test 5); LLM steps PLANNED (no API) |
| the annotator measurement of 6b (small vs large, 500 commands sentences) | `skills/train-world/scripts/measure_annotators.py` | PLANNED: "API unavailable" on this machine |
| the 5-seed GPU round trip | `tests/p0_full_gpu.sh` | PLANNED: prepared, not run (the GPU is not free) |

### Choices where the spec leaves room (each stated in the code too)
- **Several legends per file** (amendment of 2026-10-16): `!legend sha256:<hex>` for the default world and `!legend <world> sha256:<hex>` per
  named world; `sop formalize` writes the named form and `_world` on every top-level wire; `sop check --legend name=path` (repeatable) or
  `--legends <dir>`; with ONE legend and at most one world named in the file, that legend is the file's legend. `sop diff` compares only
  wires of the same world (a pair from two worlds: `_verdict=unverifiable`, `_role=_world`).
- **Dispatcher by fit** (section 9): every reader reads; the smallest RELATIVE residual wins (residual / the reader's no-fit residual), the
  mean margin second; `_world_margin` = the relative-residual gap to the second reader; when the best reader cannot place the sentence or its
  residual is at or above the no-fit value (stated 0.9: fitting it needs out-of-world sentences a world folder does not have; unseen depths
  of a world reach 0.6-0.8 in the P0 runs, random text 0.99) the output is `unknown` with `_world=none`.
- **Word stars are global** (section 0): identity of a word = star(GLOBAL seed `sop-stars-v1`, "id:"+word), the same in every world; role /
  operator vectors and the occupancy marker come from the world's seed. Rule 6 (the seed also changes the identities): `--vary_stars`.
- **The `reader` block in legend.json** (it changes the targets, so it is under the legend hash): `predicate_role`, `occupancy_marker`,
  `definite_dets`; for the quantifier world v2 also `scope_terms` (`wide`: each, a certain; `q`: every, some), `scope_rule` (the readout rule
  as text), `rel_emitters` (the beacon-relative emitters, given) and `full_excludes` (`det`: the determiner WORD is outside FULL, its presence
  stays in).
- **Lexicon** `kinds.tsv`: `word<TAB>kind[<TAB>lemma]`. A token with no kind gets the kind `unknown` (spec 6.3): it is placed in the chart,
  never a filler, and listed in `_unplaced`; REPORT.md gives the share. The tagger source is a limit of v1 (no tagger installed).
- **Multi-word values** (`det="a certain"`) are read with the identity of their LAST token, the token the sentence carries.
- **Reserved-line extensions**: `_verb=<lemma>`, `_template` (generator template id, used by the splits), `_reading` (in
  audit/disagreements.sop: annotation / reader), `_margin.<slot>`, `_margin.structure` (gap between the best and the second structure
  hypothesis, depth x negation).
- **Every wire** written by formalize has `_margin` (the minimum of its own `_margin.*` lines; `none` when it carries none), `_status` (the
  sentence's), `_empty` (the keys of its type not filled; `-` when none) and `_depth` (distance from the top-level wire).
- **`_unplaced`**: tokens with no kind, plus tokens of filler kinds whose star is in no decoded slot (neither the token nor its lemma).
  `no-sense` = more than half of the tokens unplaced (stated 0.5); `unknown-event` = the structure is unreadable, or (predicate role) no
  token's lemma is a legend verb — a lexical test, stated.
- **`_scope`** (spec 3.3-3.4, quantifier world v2): read by the legend's rule from the two category-local terms; term presence = ratio of the
  term's cosine to the structure score >= a threshold set on CALIBRATION rows (midpoint of true terms vs the others; `thresholds.scope_term`).
  The presence of a quantifier on a slot is read from its term (as v14's C2c decoder); the determiner word from the determiner channel.
  Written only with two or more quantified slots (`both` when two plain quantifiers and no marker).
- **Structure given vs estimated** (author C3): the tool estimates the reporting depth and the event negation from the state; metrics.json and
  REPORT.md give FULL three ways: estimated, depth given, depth and negation given.
- **`|` alternatives** are compiled into targets (superposition / sqrt(n)) but the reader never emits them: a margin under tau gives
  `_status=abstain` with the top candidate.
- **Decoder reads** says* -> [not] -> event, with fillers word | negated word | entity(head, attr [negated], det) and `_scope`. `if`, `must`,
  `may`, `and`, `~` are accepted in examples and compiled into training targets, but not decoded (their sentences score FULL-wrong).
- **Thresholds** (rule 25): tau 0.05, no-sense share 0.5 and the dispatcher's no-fit residual 0.9 are STATED, not fitted; presence /
  attribute / determiner / scope-term thresholds are midpoints on CALIBRATION rows = a sample of the TRAINING split. All in `model/config.json`.
- **Splits** (spec 6.4), written to `runs/<ts>/splits.json` before training: seen formulations (`train`) and their held-out SENTENCES
  (`heldout_sentences`, 10% by hash); held-out FORMULATIONS by template when the examples are templated (or by file, `split_hints.json`),
  else by role order (the order of the roles' fillers, with depth); new identities; held-out depth when the world says so. A world may state
  `"heldout_formulations": "none: <reason>"` in `split_hints.json` — the commands and quantifier exports do, because the repository scripts
  train every form; the acceptance of a training then accepts the stated absence. Two candidate sets are scored: `catalog` (what formalize
  does) and `experiment` (as the repository scripts).
- **Baselines** (spec 6.4, rule 9): the LEXICAL BAG baseline, trained on the training rows (role presence from the bag of non-filler words,
  filler = the candidate with the highest training rate of filling the role, no position information), and the lexical ORDER baseline (the
  rank among tokens of the role's kinds); both on the same rows as the reader.
- **Seeds** (spec 6.5): the skill trains 5 seeds when the world has at least `min_examples` annotated sentences, 1 otherwise, and says so;
  `--seeds` in its train arguments wins (the tests give `--seeds 1`, stated).
- **Second look** (6b): reads the TRAINING split only; confident = every margin >= tau; cases with both readings in
  `audit/disagreements.sop`; the kept reading replaces the annotation in place with `_changed=yes`; retrain once when the changed share
  exceeds `world.json second_look_retrain` (default 0.05; a separate key from `second_look_escalate`, the disagreement-rate threshold of the
  model escalation, also 0.05).
- **Server writes**: top-level `legend.json`, `world.json`, `split_hints.json`, `*.tsv`; anything under `examples/`, `raw/`, `audit/`;
  `requests/` through its own route. `model/`, `runs/`, `versions/`, `status.json` are refused (403); `world.json` cannot be deleted.
- **Verifier test** (test 3): `experiment/v17_verifier.propose_with_error` works on WikiSQL SQL dicts; the same one-error-per-instance scheme
  is applied to the commands world with the error types mapped by function. The test checks the 10c shape, not its numbers.

## Conformance with the complete specification (sections 0-10, the `_world` amendment, the 78d36e8 instructions)
"unchanged" = the code's earlier decision was already compatible with the specification and was kept (the author's rule).

| statement | status | where |
|---|---|---|
| §0 one world = one folder = one versioned legend = one reader | done (unchanged) | world folders; `model/` bound to the legend hash (exit 2 on mismatch) |
| §0 / §9 dispatch: explicit by channel; by fit (smallest relative residual, mean margin second, `_world=none` when no reader fits) | done (by fit CHANGED: relative residual, mean margin, `_world_margin`, `_world=none`, `--worlds <root>`) | `infer.formalize_text`, `readers_of_root`; test 2 |
| §0 shared: package, server, skill, word-star catalog, kinds lexicon per language; NOT shared: the route book | done (unchanged); the lexicon is a file per world (per-language sharing = the same `kinds.tsv`) | `world.GLOBAL_STAR_SEED` |
| §0 two legends with the same question list are one world | not enforced | stated here |
| §0 the vocabulary word for a role's region does not appear in the tool | done (unchanged) | grep-clean |
| §0 sizes: "a few to ~15 MB at D = 1024" | done (CHANGED: breakdown below) | `tests/model_sizes.py`; `tool/runs/p0_cpu_1seed_60f284d/model_sizes.json` |
| §1 world.json: `models` per step, `min_examples`, `max_versions_per_run`, `second_look_escalate` | done (unchanged); `second_look_retrain` ADDED (6b) | `server.js`, `export_worlds.py`, `loop.py` |
| §3.3 `_scope=<role>|both` on the event wire; absent with zero or one quantifier | done (NEW) | `world.py` (gold, targets), `infer.read_scope`, `check.py`; tests 1, 2 |
| §3.3 `_world`, `_world_margin` | done (`_world_margin` NEW) | `infer.py` |
| §4 model/ folder | done (unchanged); `acceptance.json`, `second_look.json` added by the skill | |
| §5 every wire has `_margin`, `_status`, `_empty`, `_depth`; nothing silently dropped | done (CHANGED: was top-level / event wires only) | `Reader.wire_lines` |
| §5 `sop formalize --worlds <DATA_ROOT>` | done (NEW) | `cli.py` |
| §6.1 inventory; stop on legend inconsistency or examples with unknown roles | done (NEW) | `loop.py inventory`; test 5a |
| §6.2 annotate; invented keys / types rejected and re-asked once; `examples/annotated_<date>.sop`, `_source=llm:<model>` | done (re-ask NEW); needs the API | `loop.py annotate` |
| §6.3 kinds; `unknown` kind; share reported; table in catalog.json | done (`unknown` NEW) | `train.py`, `infer.encode` |
| §6.4 the four splits before training; formulations by template or role-order clustering; lexical bag baseline on the same rows | done (role order and bag baseline NEW; splits.json unchanged) | `train.make_splits`, `bag_baseline`; test 5a |
| §6.5 reference trainer, memory, checkpoints, 5 seeds when the data allow it; thresholds on calibration rows | done (seeds rule NEW) | `train.py`, `loop.seeds_for` |
| §6.6 evaluation fields (per role, FULL, three occupancy cases, calibration, held-out formulations, new identities, lexical baseline, agreement with the annotation on a held-out slice, what is weak and which formulations) | done (occupancy, bag baseline, worst formulations NEW; agreement = `heldout_sentences` FULL and `audit/heldout_verdicts.sop`) | `train.evaluate`, `write_report` |
| §6 6b second look on the training sentences, `audit/disagreements.sop`, label noise, retrain once over a stated share; the annotator measurement | done offline (retrain, in-place replacement, `disagreements.sop` NEW); the measurement needs the API | `loop.second_look`, `measure_annotators.py` |
| §6.7 package: formalize a held-out slice next to the annotation | done (NEW) | `loop.package` |
| §6.8 evolve (1.1 steps 3-5) | done (unchanged); LLM re-annotation of the migration's sentences NEW (needs the API) | `loop.step`, `llm_reannotate` |
| §6 acceptance of a training: check, metrics fields, REPORT fields, confident-wrong, ten examples reproduced | done (NEW); label BUILT / FAILED | `loop.accept_training`; test 5a |
| §9 strategy 3 (sticky context) | not done (HYPOTHESIS in the spec) | |
| §10 out of scope | as stated | |
| amendment `_world`: several legends per file, per-world hash, diff within a world | done (unchanged) | test 1 |
| Q5 quantifier world v2 = C2c with its own hash; v1 under versions/ | done (NEW) | `export_worlds.export_quantifiers`; test 2 |
| C2 P0 runs committed | done by the coordinator | `tool/runs/p0_cpu_1seed_60f284d/` |
| C3 FULL with depth given and estimated, side by side | done (NEW) | `train.evaluate`, `tests/p0_compare.py` |
| C4 size breakdown | done (NEW) | below |

## Model size (author C4)
`tests/model_sizes.py` on the P0 CPU models, output committed as `tool/runs/p0_cpu_1seed_60f284d/model_sizes.json` (D = 1024, float32 phases):

| world | kinds K | routebook.npz on disk | bg + A + L (K x K x D each) | path + bg0 (K x D each) | rest of model/ |
|---|---|---|---|---|---|
| reference | 32 | 11.9 MB | 3 x 4.19 MB = 12.58 MB (97.9% of the arrays) | 2 x 0.13 MB | ~0.11 MB (REPORT, metrics, catalog, config, background) |
| commands | 16 | 3.0 MB | 3 x 1.05 MB = 3.15 MB (96.0%) | 2 x 0.07 MB | ~0.07 MB |
| quantifiers (v1) | 36 | 15.0 MB | 3 x 5.31 MB = 15.93 MB (98.2%) | 2 x 0.15 MB | ~0.10 MB |

The per-category-PAIR relative keys (bigram bg[c1, c2], chain A[c1, c2], left re-addressing L[c1, c2]) are ~3 K^2 D x 4 bytes and dominate;
K is large in the constructed worlds because every function word and verb form is its own kind. The quantifier world v2 adds the beacon-relative
array rel (K x D, ~0.15 MB). Where this shrinks, if it does: B6 (sparse block codes), not the packaging.

## P0 round trip
Criterion (indications step 2): training through the package on the exported examples reproduces the RESULTS tables' role accuracies within
seed variance.

**CPU, 1 seed, full sizes** (BUILT: `tool/runs/p0_cpu_1seed_60f284d/<world>/` — train.log, metrics.json, splits.json, command.txt; the code
commit that ran is in each command.txt: f759d93; the quantifier world there is VERSION 1, no scope; these runs predate the depth-given column):

| world | axis | package (roles) | tool FULL, structure estimated | RESULTS (5 seeds) |
|---|---|---|---|---|
| reference | seen / predicate / new identities / role cues | 1.000 each | 1.000 | 1.000 (10b) |
| reference | unseen transitions | 0.633 | 0.511 | 0.638±0.015 (10b) |
| reference | depth 3 UNSEEN | 0.792 | 0.792 | 0.813±0.030 (10b) |
| commands | depth 0 FULL / new patients FULL | 0.998 / 1.000 | 0.998 / 1.000 | 0.999 / 1.000 (10d) |
| commands | depth 2 UNSEEN FULL | 0.790 | 0.790 (structure 0.905) | 0.854±0.031 (10d) |
| quantifiers v1 | nouns depth 0 / depth 2 | 0.997 / 0.993 | 0.660 / 0.356 | 0.999 / 1.00 (9c-bis) |

Reference reproduced within seed variance on every roles column; commands depth 2 about two standard deviations low at one seed (the tool
estimates depth and negation, the script is given the depth — C3: the 5-seed run reports both); quantifier v1 nouns reproduced, determiners
not comparable (no scope term in v1, see the REPORT of that run).

**The two quantifier rows are not comparable on scope.** The committed CPU runs (`tool/runs/p0_cpu_1seed_60f284d/`) were made by code
commit f759d93 with the quantifier world VERSION 1 (no scope term in the targets, no `_scope`); the GPU round trip below uses VERSION 2 (the
C2c world, `_scope`). Only the noun rows of the two runs measure the same thing.

**GPU, 5 seeds** (PLANNED, prepared, not run): `tool/tests/p0_full_gpu.sh` — reference, commands, quantifiers VERSION 2 (C2c); full sizes,
`--seeds 5 --vary_stars --device cuda --no_ckpt`; it takes the repository's GPU lock (`experiment/logs/.run.lock`, the pid lock of run.sh /
queue.sh; `--wait` polls it, rule 26), refuses an unclean code tree, and writes `tool/runs/p0_gpu_5seed_<commit>/<world>/` (train.log,
metrics.json, splits.json, command.txt, REPORT.md, config.json, legend.json; no checkpoints) and `p0_table.md` (`tests/p0_compare.py`:
structure estimated and depth given side by side, against 10b, 10d and 9c-sexies).

## Test results (BUILT; `tests/run_all.sh` on CPU; smoke readers D = 256; outputs in `tool/tests/out/`, not committed; all six PASS)
- test 1 (SOP Lang + check): the 3.5 example parses and round-trips byte for byte; every 3.6 negative case rejected with line and reason;
  `_scope` accepted on an event wire (`agent`, `both`), rejected on an entity wire and with a non-role value; an event without verbs is a
  legend error; a file mixing two worlds passes with two legends and fails with one.
- test 2 (formalize): deterministic (byte-identical twice) on the three worlds; output passes `sop check`; exit 2 on a missing model / a legend
  mismatch; the dispatcher by fit sends 3 + 3 sentences to the right worlds and the two-world file passes `sop check --legends`; quantifier
  world: legend v2 current and v1 under versions/v1/; formalize emits `_scope` (130 wires on 300 eval sentences, 59 `both`); metrics give FULL
  with the structure estimated and with the depth given, `_scope` accuracy, and the previous FULL (incl. the determiner word) once (smoke
  reader at 2000 sentences: `_scope` 0.800 on the never-trained ambiguous sentences, 0.805 on held-out sentences — a pipeline check).
- test 3 (injected-error verifier, commands smoke reader, new patients, 140 sentences, tau 0.05): gold: false alarm 0.007 / confirmed 0.993;
  change_action, change_patient, flip_negation detected 1.000; flip_attr_negation 0.964 (0.036 unverifiable); drop / add destination 1.000;
  missed (confident-wrong) 0.000 on every type: the 10c shape.
- test 4 + 6 (server and page): every route, world.json with the section 1 fields (and `second_look_retrain`), 503 without a model,
  traversal rejected, model/ and runs/ refused, byte-identical downloads, the page. PASS.
- test 5 (skill, spec 6): (a) one version on the commands world from raw/ — inventory clean; the four splits (train 880, held-out sentences
  101, held-out formulations 153 = one template held out, new identities 97, held-out depth 42); seeds stated; second look on the training
  split (disagreement rate 0.008, 7 of 880; oracle annotator, nothing changed); audit/disagreements.sop; the held-out slice packaged next to
  the annotation (50: agreement 49, unverifiable 1); the acceptance of the training met on all five checks (ten random seen examples
  reproduced) -> BUILT; REPORT.md has every section. (b) legend without `destination`: the only novel supported `_unplaced` cluster is
  {archive, cloud, folder, trash} (kind DEST, 225 sentences, 3 routes); proposed (placeholder name: API unavailable), migrated (225 examples),
  retrained and ACCEPTED: FULL 0.852 -> 0.852 (tolerance 0.02), residual 0.458 -> 0.456, unknown 0 -> 0, confident-wrong 0.028 -> 0.025.

## The train-world skill and the models
The coding agent of the loop is Claude; the models are named by role in `world.json` `models`: `pipeline`, `annotate`, `second_look` =
claude-haiku-5-5 (small class), `evolve` (legend proposal and evolution) = claude-opus-5-5 (large class); escalation of annotate / second_look to
the large class when the second-look disagreement rate exceeds `second_look_escalate` (0.05). Nothing requires a model above the Opus class.
All LLM calls go through `skills/train-world/scripts/llm.py:llm_call(role, prompt, world_dir)` (the official `anthropic` SDK). **There is no
API key (and no `anthropic` package) on this machine**: every LLM step reports "API unavailable" cleanly; the offline loop uses the generator
gold as annotator (`--annotator oracle:<dir>`) and placeholder names for new roles. See `skills/train-world/SKILL.md` (the steps of spec 6,
one script each).

**The one measurement** (`skills/train-world/scripts/measure_annotators.py --n 500`): annotate 500 generated commands sentences with the small
and with the large model, FULL-frame agreement with the gold for each; the small default stands if it is within 0.02 of the large. Status:
PLANNED — "API unavailable" (exit 3); the small default is a stated default, not a measured one.

## Limits of v1
- No LLM calls were made (no API): annotation (with its re-ask), the second look's choice between readings, role naming, the LLM
  re-annotation of a migration and the annotator measurement are PLANNED; the offline loop is BUILT.
- The decoder reads one event per sentence (says* -> [not] -> event); `if`, `must`, `may`, `and` are not decoded.
- Mood is not a role of the reader: `_mood` comes from a normalizer heuristic (the legend's only mood, or '?' -> question).
- Values (numbers, dates) are invisible to the reader; `values` patterns only give their kind.
- No tagger for kinds; a lexicon is required (unknown words get the kind `unknown`).
- Two legends with the same question list are not detected as one world; dispatch strategy 3 (sticky context) is not built.
- The dispatcher's no-fit residual (0.9) is stated, not calibrated (no out-of-world sentences in a world folder).
- Smoke models (D = 256, 2000 sentences) are pipeline checks, not results.
