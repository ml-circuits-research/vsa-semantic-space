---
name: train-world
description: Run the learning loop of one world folder (legend -> annotations -> reader -> feedback -> next legend version), with written acceptance criteria. Use when a world folder has raw/ sentences to annotate, examples to train on, or a legend version to evolve.
---

# train-world — the loop of spec 1.1, for a coding agent

You are the coding agent of the loop. You do not decide by taste: every step below is a script with a written rule, and you read its
JSON output and go on. The models are named by ROLE in `<world>/world.json` `models`, never chosen by you:

| step | role key | default | |
|---|---|---|---|
| inventory, kinds, splits, training, evaluation, packaging (you) | `pipeline` | claude-haiku-5-5 | tool calls over scripts with written criteria |
| annotation of raw/ | `annotate` | claude-haiku-5-5 | closed labelling task, guarded by `sop check` and the second look |
| second look | `second_look` | claude-haiku-5-5 | escalates by rule (below) |
| legend proposal (v1) and evolution (names, migrations) | `evolve` | claude-opus-5-5 | once per version |

The implementing agent of this tool is Claude; the agent that RUNS this skill is Claude with the `pipeline` model (Haiku class). Nothing here
needs a model above the Opus class; when a step cannot be done at that class, report it as a limit. All LLM calls go through
`scripts/llm.py:llm_call(role, prompt, world_dir)`; with no API reachable it raises `LLMUnavailable` and the step says "API unavailable".

Scripts: `scripts/loop.py <cmd> <world>` (cmd: inventory, propose-legend, annotate, train, second-look, package, accept-training, feedback,
evolve, step, run), `scripts/llm.py`,
`scripts/measure_annotators.py`. The CLI is `tool/bin/sop` (check, formalize, diff, stats, train).

## The loop (spec 1.1 steps 1-6)

1. **Propose.** If `legend.json` is missing: `loop.py propose-legend <world>` (evolve model, from a sample of raw/). Freeze it: `version`,
   sha256 (the hash is the file's; every `.sop` carries it in `!legend`). Roles, event ids and operators are then CLOSED for this version.
2. **Annotate, train, evaluate.** `loop.py annotate <world>` (annotate model; every output goes through `sop check`; a sentence the model
   cannot annotate validly becomes an `unknown` wire, never dropped). If fewer than `min_examples` annotated sentences: stop and say so.
   `loop.py train <world> --train-args "..."` runs `sop train` (splits written before training; thresholds on calibration rows; REPORT.md).
   `loop.py second-look <world>`: the reader re-reads the training sentences; where it is confident and disagrees with the annotation, the
   second_look model takes a second look with both readings in front of it; the disagreement rate and the number of annotations changed (the
   measured label noise) are written to `model/second_look.json` and REPORT.md.
   **Escalation rule:** if the disagreement rate > `second_look_escalate` (world.json, default 0.05), `annotate` and `second_look` move to the
   large class for this world (`llm.escalate`), raw/ is re-annotated ONCE, and REPORT.md states the switch and both rates.
3. **Feedback from the reader** (`loop.py feedback <world>`, all measured): `_unplaced` words clustered by residual direction (the address
   each unplaced star was deposited at; |cos| >= 0.5 joins a cluster; clusters sharing a word are one question reached by several routes),
   with their sentences and the share of the residual they explain; `unknown` wires grouped by their words; per-sentence `_residual`; roles
   empty across the corpus; per-split / per-role accuracy and confident-wrong (metrics.json); `requests/*.md` as one more signal.
4. **Decide a change set** (`loop.py evolve <world>`): new roles from supported `_unplaced` clusters (support >= 2% of the sentences read and
   >= 5 sentences) whose words are NOVEL — a cluster whose words (or lemmas) already fill a legend role in the annotations is the reader's
   error on an existing question and is reported, not proposed; new synonyms / events from `unknown` clusters. The evolve model NAMES them (offline: placeholder `role_<k>`, stated in
   CHANGES.md). A MIGRATION (`versions/v<N+1>/migration.json`) says which sentences to re-annotate; examples are migrated automatically.
   Constraints: add freely; remove or merge a role ONLY if it is empty across the whole corpus; never propose a change set equal to a
   version marked `rejected`.
5. **Accept by criterion, on the held-out set fixed before the change** (the v run's `splits.json`, reused for v+1 with `--splits`):
   - FULL frame of v+1 >= FULL of v within seed variance (tolerance max(0.02, 2 x std over seeds));
   - `_residual` mean down;
   - `unknown` share down (both 0 counts as not up);
   - confident-wrong at tau not up (tolerance 0.005);
   - stable across the seeds run (vacuous with one seed — say so).
   Met -> v+1 becomes current (`legend.json`, `model/`, `status.json` name it; v stays under `versions/v<N>/` with its REPORT). Not met ->
   `versions/v<N+1>/` is kept with `mark.json` = rejected and its acceptance numbers; the loop continues from v.
   `loop.py step <world>` does 3-5 once; `loop.py run <world>` does 1-5 with at most `max_versions_per_run` evolution steps.
6. **People** at any time read any version's REPORT, drop a request into `requests/`, mark a version `rejected` or `pinned`
   (server: `POST /worlds/{w}/versions/{n}/reject|pin`). A pinned version stays the serving one: an accepted v+1 is recorded, not served.
   None of this blocks the loop.

Rules that carry over from the research repository: thresholds on calibration rows only (rule 25); every number labelled BUILT / PLANNED /
HYPOTHESIS (rule 22); the reader learns the route book only — roles, identities, kinds and event synonyms are inputs (rule 20); versions are
explicit and hash-checked; no time estimates anywhere.

## Offline operation (this machine, 2026-10-16: no API key)

`--annotator oracle:<dir of gold .sop>` replaces the annotate model on constructed worlds (the generators' gold wires for the same _text;
provenance `_source=oracle:generator-gold ...`). The second look by the oracle changes nothing (the gold is the annotation). The evolve model's
naming falls back to placeholders. `scripts/measure_annotators.py` (small vs large annotator, 500 commands sentences, FULL agreement with
gold; the small default stands if within 0.02) prints "API unavailable" and exits 3.

## The steps of spec section 6, one script each (`loop.py run` does them in this order)

| spec 6 step | what the script does | command |
|---|---|---|
| 1 Inventory | world.json, legend.json, the files of examples/ raw/ audit/; `sop check` on examples; STOP and report on a legend inconsistency (role ids colliding with reserved names, events without verbs) or on examples with unknown roles | `loop.py inventory` |
| 2 Annotate raw/ | the annotate model writes the wires; `sop check` validates every wire; a rejected one is RE-ASKED ONCE with the errors; then an `unknown` wire; `examples/annotated_<date>.sop`, `_source=llm:<model>` | `loop.py annotate` |
| 3 Kinds | word -> kind from the lexicon over all tokens; a word with no kind gets the kind `unknown`; the share of `unknown` tokens in REPORT.md; the table in model/catalog.json | `sop train` |
| 4 Splits | runs/<ts>/splits.json BEFORE training: seen formulations (train), held-out sentences, held-out FORMULATIONS (by template when templated, else by role order), new identities; the lexical bag baseline (and the order baseline) trained on the same rows | `sop train` |
| 5 Train | experiment/trainer.train_loop + V8 (memory logged, checkpoints); 5 seeds when the data allow it (>= min_examples sentences), 1 otherwise and said so; thresholds on calibration rows | `sop train` (seeds chosen by `loop.py run`) |
| 6 Evaluate | metrics.json + REPORT.md: per role, FULL (structure estimated, depth given, depth and negation given), the three occupancy cases, calibration, held-out formulations, new identities, the baselines on the same rows, agreement with the annotation on the held-out sentences; what is weak and which formulations to add | `sop train` |
| 6b Second look | the reader re-reads the TRAINING sentences; confident and disagreeing -> the second_look model keeps one reading (in place, `_changed=yes`); audit/disagreements.sop; disagreement rate and changes (label noise) in REPORT.md; ESCALATION (rate > second_look_escalate: annotate / second_look to the large class, raw/ re-annotated once); RETRAIN once when the changed share > second_look_retrain (default 0.05) | `loop.py second-look` |
| 7 Package | model/ complete; status.json; runs/<ts>/ with log and command; `sop formalize` on a held-out slice next to the annotation, with the verdicts (audit/heldout_*.sop) | `loop.py package` |
| acceptance of a training | `sop check` on model/legend.json + examples; metrics.json has every field of step 6; REPORT.md states n, provenance, the 6b rate; confident-wrong at tau reported; ten random seen examples reproduced role line for role line -> BUILT, else FAILED with the reason | `loop.py accept-training` |
| 8 Evolve | steps 3-5 of 1.1 above; with an API, the annotate model re-annotates the sentences the migration lists under the new legend; offline the migration's rule | `loop.py step` |

The annotator measurement of 6b (small vs large, 500 commands sentences): `scripts/measure_annotators.py`.
