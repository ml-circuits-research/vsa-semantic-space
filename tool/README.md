# tool/ — the reader as a tool: SOP Lang, the `sop` CLI, the worlds server, the train-world skill

P0 of the programme. Specification: `docs/specs/SPEC_worlds_server_sop_cli.md` (version 2; sections 0-5 received, section 6 — the training
skill — NOT received) and `docs/specs/INDICATIONS_tool_agent.md` (models by role, build order, the six tests). Nothing here changes the model
or the measurement rules of `experiment/`: the package imports `experiment/v8_gates.V8` and `experiment/trainer.train_loop` unchanged
(nothing under `experiment/` was modified or copied).

**Given and learned (rule 20).** GIVEN: the legend's roles (and their vectors), the word identities (stars), the kinds of words (lexicon),
the event synonym table. LEARNED: the route book only — per-kind displacements of the path / bigram / chain codes and the gates — by the single
generative objective max cos(S, S_world), with structured per-sentence targets with pre-set roles; the model learns the assignment.

## Install and run

Python >= 3.10 with numpy (inference) and torch (training only); Node.js >= 18 (server; standard library only). No other packages.
```
tool/bin/sop --help                               # uses the repository .venv when present ($SOP_PYTHON overrides); or: pip install -e tool/
tool/bin/sop export --out tool/worlds             # the three constructed worlds as world folders (small: 2000 sentences each)
tool/bin/sop check --legend tool/worlds/commands/legend.json tool/worlds/commands/examples/*.sop
tool/bin/sop train tool/worlds/commands --D 256 --epochs 3            # CPU smoke; writes model/, runs/<ts>/, status.json
echo "do not delete the unsaved files ." | tool/bin/sop formalize --model tool/worlds/commands/model -
tool/bin/sop diff a.sop b.sop [--model <dir>]     # verdict wires
tool/bin/sop stats --model tool/worlds/commands/model
node tool/server.js --root <DATA_ROOT> --port 8080 [--host 0.0.0.0]  # NO authentication; localhost by default
tool/tests/run_all.sh                              # the six tests (CPU)
tool/test_sop.sh                                   # the CPU smoke test: export, check, smoke train, formalize 3, diff 2, stats
```
Exit codes (spec 5): 0 ok; 1 input / validation error; 2 model missing or legend mismatch. Output is deterministic for the same model and input
(tested byte for byte).

## What is implemented (labels: BUILT = code and test exist)

| piece | where | label |
|---|---|---|
| SOP Lang reader / writer (directives, wires, values with `$ref`, `&`, `|`, `~`, `?`, quoting; byte-for-byte round trip of the canonical layout) | `sop/soplang.py` | BUILT (test 1) |
| `sop check` (all checks of 3.6 with line number and reason; several worlds per file via `_world`) | `sop/check.py` | BUILT (test 1) |
| target states of wires (events, entity, not / must / may / if / says / and, sets, alternatives, `~`) | `sop/world.py` | BUILT |
| `sop train` (V8 + trainer.train_loop, splits before training, calibration on training rows, REPORT.md, metrics.json, status.json, `--eval_only`) | `sop/train.py` | BUILT (tests 2, 5) |
| `sop formalize` (numpy only; `_margin`, `_status`, `_empty`, `_residual`, `_unplaced`, `_depth`, `_world`, `_verb`; `--json`; dispatcher by fit) | `sop/infer.py`, `sop/cli.py` | BUILT (test 2) |
| `sop diff` (four verdicts with margins; per world) and `sop stats` | `sop/diff.py`, `sop/cli.py` | BUILT (test 3) |
| the server (all routes of section 2, atomic writes, path normalization, two shell-outs, one page) | `server.js` | BUILT (tests 4, 6) |
| the three constructed worlds as world folders | `export_worlds.py` (`sop export`) | BUILT |
| the train-world skill: annotate, train, second look + escalation rule, feedback (residual-direction clusters), evolve, migrate, accept, promote / reject | `skills/train-world/` | BUILT offline (test 5); LLM steps PLANNED (no API) |
| the annotator measurement (small vs large, 500 commands sentences) | `skills/train-world/scripts/measure_annotators.py` | PLANNED: "API unavailable" on this machine |

### Choices where the spec is silent or ambiguous (each stated in the code too)
- **`_world` and several legends per file** (amendment of 2026-10-16). Hash directives: `!legend sha256:<hex>` for the default world (the
  single-legend form) and `!legend <world> sha256:<hex>` per named world; `sop formalize` writes the named form and `_world` on every
  top-level wire. `sop check --legend name=path` (repeatable) or `--legends <dir>`; with ONE legend and at most one world named in the file,
  that legend is the file's legend (the single-legend form keeps working). `sop diff` compares only wires of the same world and reports a
  pair from two worlds as `_verdict=unverifiable`, `_role=_world`.
- **Dispatcher by fit** (spec v2 section 0): implemented — `sop formalize --model A --model B ...`: every reader reads, the formalization with
  the smallest `_residual` wins (a status other than no-sense / unknown-event first), and `_world` names the winner. Explicit dispatch "by
  channel" is the single `--model` form.
- **Word stars are global** (spec v2 section 0): identity of a word = star(GLOBAL seed `sop-stars-v1`, "id:"+word), the same in every world;
  role / operator vectors and the occupancy marker come from the world's training seed (not shared). Rule 6 (the seed also changes the
  identities) is available with `sop train --vary_stars` for research reproduction.
- **The `reader` block in legend.json** (an extension of the informal schema): `predicate_role` (R_pred (x) f_lemma, the v16 world),
  `occupancy_marker` (+ m in every filled slot, v16), `definite_dets` (determiners that add no term, v14). It is part of the legend because it
  changes the targets, so it is covered by the legend hash.
- **Lexicon** `kinds.tsv`: `word<TAB>kind[<TAB>lemma]`; the lemma maps a surface verb to the legend's verb (gives -> give) and an affixal
  negative to its positive (unsaved -> saved). The tagger source of the schema is not implemented (TODO).
- **Multi-word values** (`det="a certain"`, `place="New York"`) are read with the identity of their LAST token, the token the sentence carries.
- **Reserved-line extensions**: `_verb=<lemma>` (the predicate filler, written by formalize, used for `_residual` and audits), `_template`
  (generator template id, used by splits), `_world`. All begin with `_`, so `sop check` accepts them.
- **`_empty`** is written on event wires when non-empty (the spec line listing what every wire carries was cut in the paste); `_margin` and
  `_status` are on the top-level wire, `_margin.<slot>` on the wire that carries the slot; `_margin.structure` = the gap between the best
  and the second structure hypothesis (depth x negation).
- **`_unplaced`**: tokens with no kind (no displacement: they cannot be placed; never dropped silently) plus tokens of filler kinds whose star
  is in no decoded slot (neither the token nor its lemma). `no-sense` = more than half of the tokens unplaced (stated 0.5);
  `unknown-event` = the structure is unreadable, or (predicate role) no token's lemma is a legend verb — a lexical test, stated.
- **`|` alternatives** are compiled into targets (superposition / sqrt(n)) but the v1 reader never emits them: a margin under tau gives
  `_status=abstain` with the top candidate. Quantifier scope is not emitted (spec): the determiner stays in `det=`.
- **Decoder v1 reads** says* -> [not] -> event, with fillers word | negated word | entity(head, attr [negated], det). `if`, `must`, `may`,
  `and`, `~` are accepted in examples and compiled into training targets, but not decoded (their sentences score FULL-wrong; REPORT lists them).
- **Thresholds** (rule 25): the abstention tau 0.05 is STATED (the repository's v16 value), not fitted; presence / attribute / determiner
  thresholds are midpoints of present / absent ratios on CALIBRATION rows = a sample of the TRAINING split. All are in `model/config.json`.
- **Splits** (section 6 missing): `split_hints.json` in the world folder (written by `sop export`: `file_splits`, `new_identities`,
  `heldout_depth_min`, `heldout_sentence_share`); without it: automatic (held-out formulation signatures, new identities per kind, 10% held-out
  sentences by hash). Written to `runs/<ts>/splits.json` before training. Two candidate sets are scored: `catalog` (what `formalize` does:
  every catalog word of the role's kinds) and `experiment` (as the repository scripts: seen words; new identities: the held-out words).
- **Server writes**: top-level `legend.json`, `world.json`, `split_hints.json`, `*.tsv`; anything under `examples/`, `raw/`, `audit/`;
  `requests/` through its own route. `model/`, `runs/`, `versions/`, `status.json` are refused (403); `world.json` cannot be deleted.
  `POST /worlds/{w}` creates world.json with `models` per step, `min_examples` (500), `max_versions_per_run` (3), `second_look_escalate` (0.05).
- **Verifier test** (test 3): `experiment/v17_verifier.propose_with_error` injects errors into WikiSQL SQL dicts and needs the WikiSQL data
  and the v11 reader; it cannot be applied to SOP frames as is. The same one-error-per-instance scheme is applied to the commands world with the
  error types mapped by function (change_action / change_patient ~ change_sel, flip_negation ~ flip_agg, drop / add destination ~ drop /
  add_cond). The test checks the 10c shape (role errors detected more often than false alarms on gold; missed <= false alarms), not its numbers.

## Conformance with spec v2 and the `_world` amendment
| statement | status | where |
|---|---|---|
| one world = one folder = one versioned legend = one reader | done | world folders (`sop export`, server `POST /worlds/{w}`); `model/` bound to the legend hash (`config.json`, CLI exit 2 on mismatch) |
| readers are per world; many worlds = many readers + a dispatcher (by channel, or by fit: smallest residual wins) | done | `sop formalize --model A [--model B ...]` (`infer.formalize_text`); by channel = one `--model` |
| shared across worlds: the package, the server, the skill, the word-star catalog (a word's star is the same everywhere), the kinds lexicon per language | done | `world.GLOBAL_STAR_SEED`; one `sop` package, `server.js`, `skills/train-world`; the lexicon is a file per world folder (per-language sharing = copying the same `kinds.tsv`; no central per-language store yet) |
| NOT shared: the route book | done | `model/routebook.npz` per world |
| two legends with the same question list are one world | not enforced | stated here; no check compares question lists across worlds |
| the word "domain" does not appear in the tool | done | grep-clean in code and docs of `tool/` |
| world.json: `models` {pipeline, annotate, second_look, evolve}, `min_examples`, `max_versions_per_run`, `second_look_escalate` (0.05) | done | `server.js` (POST /worlds), `export_worlds.py`; read by `skills/train-world/scripts/{llm,loop}.py` |
| `_world` on top-level wires; a .sop file may hold several worlds; per-world legend hash | done | `soplang` (reserved line), `check.py` (`!legend <world> sha256:...`), `infer.py` writes `_world`, `diff.py` compares within a world; test 1 (mixed file passes with two legends, fails with one) |
| expected sizes stated (5-20 roles, tens of event types, a few thousand sentences, a model under a few MB); no time estimates | done | below |

## Concepts (spec v2 section 0) and expected sizes
One world = one folder = one versioned legend = one reader; readers are per world (the route book targets the world's named stars and is not
shared); the `sop` package, the server, the skill, the catalog of word stars and the kinds lexicon per language are shared. Two legends with the
same question list are one world. Expected sizes (the spec's expectations, not measurements): 5-20 roles and tens of event types per legend; a
few thousand annotated sentences to start; a model under a few MB.

## P0 round trip
Criterion (spec / indications step 2): training through the package on the exported examples reproduces the RESULTS tables' role accuracies
within seed variance. The full-size command (the repository's sizes: reference 30,000 sentences, commands and quantifiers 40,000, D = 1024,
batch 8, lr 0.02, gates learned from a closed start; 5 seeds on the GPU, through `experiment/queue.sh` when the GPU is free):
```
tool/bin/sop export --out tool/worlds_p0 --n 30000 --n_eval 30 --worlds reference
tool/bin/sop export --out tool/worlds_p0 --n 40000 --n_eval 30 --worlds commands,quantifiers
tool/bin/sop train tool/worlds_p0/reference   --D 1024 --seeds 5 --vary_stars --device cuda --n_eval 600
tool/bin/sop train tool/worlds_p0/commands    --D 1024 --seeds 5 --vary_stars --device cuda --n_eval 600
tool/bin/sop train tool/worlds_p0/quantifiers --D 1024 --seeds 5 --vary_stars --device cuda --n_eval 600
```
`tests/p0_full_cpu.sh` runs the same at one seed on the CPU. Results: see "P0 results" below.

P0_RESULTS_PLACEHOLDER

## The train-world skill and the models
The coding agent of the loop is Claude; the models are named by role in `world.json` `models`: `pipeline`, `annotate`, `second_look` =
claude-haiku-5-5 (small class), `evolve` (legend proposal and evolution) = claude-opus-5-5 (large class); escalation of annotate / second_look to
the large class when the second-look disagreement rate exceeds `second_look_escalate` (0.05). Nothing requires a model above the Opus class.
All LLM calls go through `skills/train-world/scripts/llm.py:llm_call(role, prompt, world_dir)` (the official `anthropic` SDK). **There is no
API key (and no `anthropic` package) on this machine**: every LLM step reports "API unavailable" cleanly; the offline loop uses the generator
gold as annotator (`--annotator oracle:<dir>`) and placeholder names for new roles. See `skills/train-world/SKILL.md`.

**The one measurement** (`skills/train-world/scripts/measure_annotators.py --n 500`): annotate 500 generated commands sentences with the small
and with the large model, FULL-frame agreement with the gold for each; the small default stands if it is within 0.02 of the large. Status:
PLANNED — "API unavailable" (exit 3); the small default is a stated default, not a measured one.

## Limits of v1 (spec 9 was not received; these are the limits known from the build)
- Section 6 of the spec (the training skill steps and splits) was not received: `TODO(section 6 missing)` marks the minimum implementations
  (splits, the loop steps, the second look's in-place replacement of annotations).
- The decoder reads one event per sentence (says* -> [not] -> event); `if`, `must`, `may`, `and` are not decoded.
- Mood is not a role of the reader yet: `_mood` comes from a normalizer heuristic (the legend's only mood, or '?' -> question).
- Values (numbers, dates) are invisible to the reader; `values` patterns only give their kind.
- The kinds tagger (`kinds.source = tagger`) is not implemented; a lexicon is required.
- No LLM calls were made (no API): annotation, second look, role naming and the annotator measurement are PLANNED; the offline loop is BUILT.
- Smoke models (D = 256, 2000 sentences) are pipeline checks, not results.
