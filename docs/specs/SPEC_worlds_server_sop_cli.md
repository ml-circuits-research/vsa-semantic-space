# PART 3 — Specification — worlds server, training skill, SOP Lang, and the `sop` CLI
(received from the reviewing session on 2026-10-16 as PART 3 of the hand-over at 78d36e8; saved verbatim, replacing the partial copy of
2026-10-16 that lacked section 6; this version adds the concepts page, `_scope=`, the multi-world dispatch and the corrected sizes)

Purpose: a minimal, real tool around the reader. A person puts a world's files (legend, examples, raw sentences) on a small server; a
coding agent, following a skill, annotates what needs annotating and trains the reader; the result is a self-contained `model/` folder;
a CLI reads sentences and writes their formalizations in SOP Lang. Nothing here changes the model or the measurement rules; it packages them.

Vocabulary (same as the book): a **world** is an activity with a fixed list of questions (a ledger, a warehouse, a command system). Its
**legend** lists the questions (roles) and the kinds of events. A **domain** is a role's region and is never used for a world here.

## 0. Concepts in one page

| object | what it is | where it lives | size | lifecycle |
|---|---|---|---|---|
| **world** | an activity with its fixed list of questions: commands to a system, a lending ledger, a warehouse | one folder on the server | — | created once; lives through legend versions |
| **legend** | the questions (roles) + the kinds of events with their synonyms + the operators admitted + the kinds of words | `legend.json`, versioned, hashed | a few KB | v1 proposed by the large model from a sample; then evolves in the loop (1.1), version by version |
| **examples** | sentences with their SOP wires | `examples/*.sop` | thousands of lines | appended; re-annotated automatically on a migration |
| **reader (model)** | what was learned: the route book (one jump per kind of word, the gates, the relative beacons), thresholds, the residual background, the catalog seed | `model/`, bound to a legend hash | tens of KB to a few MB | retrained at every accepted legend version; old ones kept in `versions/` |
| **domain** | the region of the sky of ONE role ("the recipient's domain") | nowhere: a geometric consequence of the role's named star, not a stored or trained thing | — | none; the word does not appear in the tool |

**One world = one folder = one (versioned) legend = one reader.** Readers are per world by design: the reader's job is "where does each
word go, for THIS list of questions"; merging worlds would read every sentence against questions that do not apply and lose "which world is
this". Many worlds = many small readers + a dispatcher (explicit by channel, or by fit: every reader reads, the smallest residual wins). Shared
across worlds: the `sop` package, the server, the skill, the catalog of word stars (a word's star is the same everywhere), the kinds lexicon per
language. Not shared: the route book (its jumps target the world's named stars). Two legends with the same question list are one world.
Expected sizes: 5-20 roles and tens of event types per legend; a few thousand annotated sentences to start (the held-out-formulation number in
REPORT.md says when to add more); a model of a few to ~15 MB at D = 1024 (the per-category-pair relative keys dominate); training in minutes on one small accelerator; reading in microseconds.

## 1. Folder layout — one world = one folder

```
<DATA_ROOT>/<world>/
  world.json             manifest: name, language, description, created, version; models per step: {pipeline, annotate, second_look, evolve}
                         (a small model by default for pipeline / annotate / second_look, a large one for evolve and the first legend proposal);
                         min_examples; max_versions_per_run; second_look_escalate (default 0.05: above this disagreement rate, annotation moves to the large model)
  legend.json            roles, event types with their synonym table, operators allowed, kinds (word categories) with lexicon or tagger
  examples/*.sop         annotated sentences (SOP Lang, with _text): the training pairs. Human- or LLM-annotated; provenance in the header
  raw/*.txt              unannotated sentences, one per line: what the agent must annotate before training
  audit/                 optional: a sampled human audit (same .sop format, plus verdicts) and the reader-vs-annotation disagreements
  model/                 what was learned (section 4). Rebuilt by training; never edited by hand
  runs/<timestamp>/      training logs (memory curve included), split definitions, the exact command and commit of the training code
  versions/v<N>/         legend history: legend.json, migration.json, CHANGES.md (feedback acted on, change set, acceptance numbers), metrics
  requests/*.md          free-text requests from people to the loop; each is answered by a line in the next CHANGES.md
  status.json            last training: when, on which examples (hash), legend version, metrics summary, label BUILT / TRAINING / FAILED
```

`legend.json` (schema, informal):
```
{ "version": 1,
  "roles": [ {"id":"agent","question":"who gives?"}, {"id":"object","question":"what?"}, {"id":"recipient","question":"to whom?"},
             {"id":"time","question":"when?"} ],
  "events": [ {"id":"transfer","verbs":["give","receive","hand"]}, {"id":"sale","verbs":["sell","buy"]} ],
  "operators": ["not","must","may","if","says"],            # the operator wires this world admits (section 3)
  "moods": ["statement","command","question"],
  "kinds": { "source":"lexicon|tagger", "lexicon":"kinds.tsv", "tagger":"spacy:en_core_web_sm" },
  "values": [ {"kind":"number","pattern":"\\d+(\\.\\d+)?"}, {"kind":"date","pattern":"..."} ]   # normalized value fillers (section 3.4)
}
```
Role ids and event ids are the closed vocabulary of SOP Lang for this world; `sop check` rejects anything else.

### 1.1 The legend is a closed vocabulary — and it evolves in a learning loop

Within one legend version, wire types (event ids), question keys (role ids) and operators are FIXED by `legend.json`. Nobody discovers them
during annotation or formalization: an LLM annotator that invents a type or a key is rejected by `sop check`; the reader can only produce terms addressed by the legend's named stars (roles are given, rule 20), and the event type is the predicate role
read through the legend's synonym table. Two formalizations are comparable only because they share this vocabulary.

Between versions the legend is decided by the LOOP, not by a person. The coding agent (its skill) and the reader's feedback drive it; people
can read, request and veto at any time, and the loop does not wait for them.

1. **Propose.** The agent drafts `legend.json` v1 from a sample of `raw/` (roles, event types, synonyms, kinds), freezes it with `version` and
   sha256, and names the questions itself.
2. **Annotate, train, evaluate** (sections 6.1-6.7): one LLM annotates; the reader is trained and re-reads the training sentences; where it is
   confident and disagrees with the annotation, the LLM takes a second look with the disagreement in front of it; `model/metrics.json` and
   `REPORT.md` are written, with the disagreement rate and the number of annotations changed (the measured label noise).
3. **Feedback from the reader**, all measured: `_unplaced` words clustered by residual direction (with the sentences they come from); `unknown`
   wires grouped by their verbs; per-sentence `_residual`; crowded domains (`|` answers); roles that are empty across the corpus; per-role
   accuracy and confident-wrong. Human requests in `requests/*.md` are read as one more feedback signal.
4. **Decide a change set** for version v+1: new synonyms for existing events; new event types (from `unknown` clusters); new roles (from
   `_unplaced` clusters); split or merge of roles. The agent names new roles/events (LLM) and writes a MIGRATION (old key -> new key, or
   "re-annotate these sentences") so that `examples/` are re-annotated automatically for v+1.
5. **Accept by criterion, on a held-out set fixed before the change** (the acceptance rule replaces the human gate): FULL frame >= v (within
   seed variance); `_residual` mean down; `unknown` share down; confident-wrong at tau not up; stable across the seeds run. Met -> v+1 becomes
   current (`status.json`, `legend.json`, `model/` all name it; the previous version stays under `versions/v<N>/` with its REPORT). Not met ->
   v+1 is kept under `versions/` as `rejected` with its metrics, and the loop continues from v.
6. **People** at any time: read any version's REPORT; drop a request into `requests/`; mark a version `rejected` (the agent never returns to
   it) or `pinned` (the serving version stays until unpinned). None of this blocks the loop.

Two constraints stay, because they are structural, not caution:
- **Add freely, remove only what is empty.** A metric-driven agent has a cheap path: merge or drop questions so that frames are easier to fill.
  Roles, events and synonyms may be ADDED under the criterion above; a role may be REMOVED or MERGED only if it is empty across the whole
  corpus (never filled in any example of the current version). A question people answer stays.
- **Versions are explicit**, for comparability: every `.sop` and every `model/` names the legend hash it was made with, and the CLI refuses a
  mismatch (`--force` only for inspection), so a model trained on v3 is never scored against wires written for v4. What the agent decided, and
  why, is in `versions/v<N>/CHANGES.md` (feedback it acted on, the change set, the migration, the acceptance numbers).

Folder additions: `versions/v<N>/` (legend, migration, CHANGES.md, metrics of that version), `requests/*.md` (free-text human requests, each
answered by a line in the next CHANGES.md).

## 2. The server (Node.js, standard library only, no authentication)

Single file `server.js`; `node server.js --root <DATA_ROOT> --port 8080`. JSON over HTTP; file bodies raw. Binds to localhost by default
(`--host 0.0.0.0` to expose; there is no auth, so say so on startup). No framework, no npm packages.

| method | path | effect |
|---|---|---|
| GET | /worlds | list worlds with their status.json summary |
| POST | /worlds/{w} | create the folder layout (409 if it exists) |
| GET | /worlds/{w} | world.json + legend.json + file inventory (names, sizes, mtimes) + status.json |
| DELETE | /worlds/{w} | remove the world (requires `?confirm={w}`) |
| GET | /worlds/{w}/files/{path} | download a file (any of the subfolders; `model/` included, read-only) |
| PUT | /worlds/{w}/files/{path} | upload / overwrite a file (raw body; creates subfolders under examples/, raw/, audit/ only; `model/` and `runs/` are refused: they are written by training) |
| DELETE | /worlds/{w}/files/{path} | delete a file (same restriction) |
| GET | /worlds/{w}/files?dir=examples | list a subfolder |
| POST | /worlds/{w}/check | run `sop check` on all examples against the legend; returns the report (section 5) |
| POST | /worlds/{w}/formalize | body: text (one sentence per line) -> SOP Lang, by shelling out to `sop formalize --model <w>/model` (503 if no model) |
| PUT | /worlds/{w}/requests/{name}.md | a human request to the loop (free text); listed and answered in the next versions/v<N>/CHANGES.md |
| POST | /worlds/{w}/versions/{n}/reject or /pin | mark a legend version rejected (never returned to) or pinned (stays the serving version) |
| GET | /worlds/{w}/versions | the version history with each version's acceptance numbers |
| GET | / | one static HTML page: list of worlds, upload form per folder, version history, a textarea that calls /formalize and shows the result |

Rules: paths are normalized and must stay inside the world folder (reject `..`); every write is atomic (temp file + rename); the server
never trains and never runs anything but the two `sop` subcommands above; errors are JSON `{error, detail}` with the right status code.
Logging: one line per request to stdout. Tests: a shell script that creates a world, uploads a legend and two examples, checks, and
downloads them back byte-identical.

## 3. SOP Lang — the formalization format (one file = a sequence of wires)

A **wire** is one formalized event, entity or operator. A sentence yields one top-level wire and, if it nests (reporting, negation,
conditions, complex noun phrases), auxiliary wires that the top-level one refers to. Two sigils, never confused: `@` DECLARES a wire
(header line), `$` REFERS to one (inside a value). Lines are UTF-8, `\n`-terminated; blank lines and `#` comments are ignored.

```
!world ledger                      directives (file header): world, legend hash, model, generated-by, date
!legend sha256:…
@<wireId> <wireType>               header: declaration
<question>=<answer>                one line per filled role
...
@<nextWireId> <wireType>
```

3.0 Wire types at a glance (a world's legend decides which of these it admits; `sop check` rejects the rest)

| family | wireType | lines it carries | what it is in the reader | example |
|---|---|---|---|---|
| event | one per event id in the legend: `transfer`, `sale`, `delete`, `send`… | the legend's roles (`agent=`, `object=`…); `_text`, `_mood`, `_margin`, `_empty`… | the predicate role read through the synonym table + the filled role slots | `@w3 transfer` / `agent=Paul` / `object=book` / `recipient=Anna` |
| entity | `entity` | `head=`, `attr=` (word or `$wire`), `quantity=`, `unit=`, `det=` | a filler that is itself a small constellation inside a slot (noun + attribute, negated or not) | `@w4 entity` / `head=files` / `attr=$w5` |
| operator | `not` | `scope=$wire` or `scope=word` | R_neg around a state or around a word | `@w2 not` / `scope=$w3` ; `@w5 not` / `scope=saved` |
| operator | `must`, `may` | `scope=$wire` | modality as a role around its scope | `@w6 must` / `scope=$w3` |
| operator | `if` | `condition=$wire`, `consequence=$wire` | the conditional as a role around two states | `@w7 if` / `condition=$w8` / `consequence=$w9` |
| operator | `says` | `speaker=`, `scope=$wire` | reporting: the moved origin ("reported"), one level down | `@w1 says` / `speaker=Maria` / `scope=$w2` |
| operator | `and` | `items=$w1 & $w2` | two events coordinated in one sentence (annotation and verification; the v1 reader reads one event per sentence) | `@w0 and` / `items=$w1 & $w2` |
| fallback | `unknown` | `_text`, `_unplaced`, `_status` | the reader found no event it knows | `@w1 unknown` / `_status=no-sense` |
| verifier | `verdict` (only in `sop diff` output) | `left=`, `right=`, `_verdict`, `_role`, `_left`, `_right`, margins | the four verdicts of the verifier | 3.7 |

Mood is not a wire type: `_mood=statement|command|question` is a line on the event wire (in the reader, mood is a role). Quantifier scope is a
`_scope=` line on the event wire (3.3), not a new type; the determiner word itself stays in `det=` on the entity wire.

3.1 Header line. `@` + id + space + type. Ids are unique in the file (`w1`, `w2`, … any `[A-Za-z0-9_-]+`). `wireType` is: an event id from the
legend (`transfer`); `entity` (a complex filler, 3.4); an operator (`not`, `must`, `may`, `if`, `says`, `and`); or `unknown` (the reader found no
event it knows).

3.2 Question lines. `question` is a role id from the legend, exactly; `answer` runs to the end of the line. Values:
- a word or normalized value: `recipient=Anna`, `quantity=40`. A value with spaces or reserved characters is quoted: `place="aisle B"`,
  `recipient="New York"`; inside quotes `\"` and `\\` escape. A bare value is one token without spaces.
- a reference to another wire: `scope=$w2`, `object=$w4`
- a SET (both, coordination): `agent="Paul" & "Peter"` — "Paul and Peter give…" (bare tokens also allowed: `agent=Paul & Peter`)
- ALTERNATIVES (one of them, unresolved by the reader): `agent="Paul" | "Peter"` — the crowded domain; `_status=ambiguous`. Spaces around
  `&` and `|` are optional; the two never mix in one value.
- `~` : the role is filled but the filler is not recognized (the occupancy marker says "someone is here"); `_raw.<role>=<surface token>`
- `?` : the asked role of a question (`_mood=question`); a yes/no question has no `?` and `_asked=whether`
- the role is not given: the line is absent (and the role appears in `_empty`).
Escaping: a bare value that would start with `$ ~ ?` or contain `& |` is quoted instead; newlines in `_text` become spaces.

3.3 Reserved lines (begin with `_`; a legend may not define roles starting with `_`). Written by the CLI; allowed in examples:
- `_text=<the sentence>` — first line after the header of every top-level wire; the training pairs are read from here.
- `_sentence=3` — index of the sentence in the input (several top-level wires may share it: "A gives B a book and C sells D a car").
- `_mood=statement|command|question`; `_asked=<role>|whether` for questions.
- `_margin.<role>=0.21` (top1−top2 cosine of that role's readout), `_margin.event=…` (the predicate), `_margin=<min over the wire>`.
- `_status=ok|abstain|ambiguous|unknown-event|no-sense` — `abstain`: a margin under the tau in `model/config.json`; `no-sense`: most stars
  outside the legend.
- `_residual=0.12` — unexplained share of the chart after reconstruction, against the world's background.
- `_unplaced=word1,word2` — words whose stars landed in no domain of the legend (what `no-sense` and `_residual` are made of).
- `_empty=time,place` — roles the legend has and the sentence did not fill.
- `_span.<role>=5-6` — token positions (0-based, inclusive) the filler came from; for highlighting and audits. Optional.
- `_depth=1` — nesting depth of this wire. `_tense=past|present|future` — from the normalizer, not the reader; optional, stated as such.
- `_source=reader|llm:<model>|human` — provenance on annotated examples; `_changed=yes` when the second look replaced the first annotation.
- `_scope=<role>` on an event wire — the role whose quantifier is wide when the sentence has two quantified phrases (`_scope=both` when the
  legend's rule says both; absent with zero or one quantifier, where `det=` on the entity wire says which word). Emitted from world version 2 of
  the quantifier world on (the C2c category-local terms); v1 exporters emit none.
- `_world=<world>` and `_world_margin=0.31` — on every top-level wire of a file written by the multi-world dispatcher (section 9): which
  reader read this sentence and how clearly the fit chose it (`_world=none` on an `unknown` wire that no reader could place).

3.4 Operators and entities are wires; an operator wraps a wire OR a word (the reader has both: R_op (x) S and R_op (x) f_word).
- `not`: `scope=$w3` (prohibition / sentential negation) or `scope=saved` (a negated word).
- `must` / `may`: `scope=$wN`.   `if`: `condition=$wA`, `consequence=$wB`.   `says`: `speaker=Maria`, `scope=$wN`.
- `and` (events coordinated in one sentence): `items=$w1&$w2`.
- `entity` (a complex filler): `head=files`, `attr=$w5` or `attr=red`, `quantity=40`, `unit=boxes`, `det=the|a|each|every` (the determiner
  word kept as text; which quantified phrase is wide is the event wire's `_scope=` line, 3.3). "the unsaved files" = `object=$w4`; `@w4 entity`, `head=files`, `attr=$w5`;
  `@w5 not`, `scope=saved`.
- Values: numbers, dates, identifiers normalized by the world's `values` patterns (`time=2026-10-16`, `invoice=4471`); the reader addresses the
  slot, the normalizer supplies the value (values are invisible to the reader — stated limit).

3.5 Example — "Maria says that Paul does not give Anna the unsaved files."
```
!world commands
!legend sha256:3f9a…
!generated sop 0.1, model commands/model (2026-10-16)

@w1 says
_text=Maria says that Paul does not give Anna the unsaved files.
_sentence=0
speaker=Maria
scope=$w2
_margin=0.19
_status=ok
_depth=0

@w2 not
scope=$w3
_depth=1

@w3 transfer
agent=Paul
object=$w4
recipient=Anna
_empty=time
_margin=0.24
_depth=2

@w4 entity
head=files
attr=$w5

@w5 not
scope=saved
```

3.6 Grammar (for `sop check`): file := (directive)* (wire)*; directive := "!" name " " value; wire := header (line)*; header := "@" id " "
type; line := key "=" value; key := roleId | "_" name ("." roleId)?; value := atom (sep atom)*; sep := spaces? ("&"|"|") spaces?; atom := bare | quoted | "~" | "?" | "$" id; bare := one token
without spaces or reserved characters; quoted := '"' (chars with \" and \\ escapes) '"'.
Checks: ids unique; every `$id` resolves and no reference cycle; every key is a legend role or reserved; every type is a legend event, `entity`,
an allowed operator, or `unknown`; a `&` and a `|` never mix in one value; exactly one `_text` per top-level wire in examples; `_asked` only
with `_mood=question`; the legend hash in `!legend` matches the legend given to `check`.

3.7 `sop diff a.sop b.sop` writes its verdicts in SOP too. The two files are named once in directives; a verdict wire names the compared
wires by their plain ids (not `$`, which only resolves inside one file), one verdict wire per compared top-level wire:
```
!left a.sop
!right b.sop

@d1 verdict
left=w1
right=w1
_verdict=contradiction
_role=recipient
_left=Anna
_right=Maria
_margin.left=0.24
_margin.right=0.09
```
`_verdict` is one of `agreement | compatible-incomplete | contradiction | unverifiable`; `_role` names the disputed role for a contradiction;
`_margin.left/right` are the margins the verdict rests on (unverifiable = the margin on the disputed slot under tau).

## 4. The `model/` folder — everything the CLI needs, nothing else

```
model/config.json      D (dials), seed, language, thresholds (abstention tau, occupancy, presence), training-code commit, legend hash
model/legend.json      a copy of the legend the model was trained with (the CLI refuses a different legend hash unless --force)
model/catalog.json     word -> star seed (or the list of words; stars are regenerated from the seed), kinds lexicon as used
model/routebook.npz    learned displacements per kind, gates (sigmoid values), relative vectors (beacons), scale constants
model/background.npz   the world's residual background (per-role statistics) for _residual
model/metrics.json     the evaluation table of this training: per role, FULL, calibration (confident-wrong at tau, coverage, acc-vs-coverage),
                       unseen-formulation split, lexical baseline on the same rows; labels per rule 22
model/REPORT.md        the human-readable report: data (n, provenance, audit size and agreement), splits, metrics, what is weak, what to add
```
Inference needs numpy only (phases, bind = add angles, bundle = complex sum, cleanup = cosine against the catalog). Training needs torch.

## 5. The `sop` CLI (Python, numpy at inference)

```
sop formalize --model <dir> [file|-] [-o out.sop] [--tau 0.05] [--json]     sentences (one per line, or paragraphs split by the sentence splitter) -> SOP Lang
sop check     --legend <legend.json> <files...>                                 validate SOP files against a legend; report per file; exit 1 on errors
sop diff      <a.sop> <b.sop>                                                   the verifier: agreement / compatible-incomplete / contradiction (with the role) / unverifiable per wire
sop stats     --model <dir>                                                     print metrics.json as a table
sop formalize --worlds <DATA_ROOT> [file|-] [-o out.sop]                        multi-world: every world's reader reads each sentence; the dispatcher picks by fit (section 9)
```
Exit codes: 0 ok; 1 input/validation error; 2 model missing or legend mismatch. Output is deterministic for the same model and input.
`--json` emits the same content as JSON (one object per wire) for programs. Every wire written by `formalize` has `_margin`, `_status`,
`_empty`, `_depth`; nothing is silently dropped: a sentence the reader cannot place still yields a wire (`unknown`, `_status=no-sense`).

## 6. The training skill (for the coding agent)

`skills/train-world/SKILL.md` + `scripts/`. The agent is given a world folder (local path or server URL) and follows:

1. **Inventory.** Read world.json, legend.json, list examples/, raw/, audit/. Run `sop check` on examples. Stop and report if the legend is
   inconsistent (role ids colliding with reserved names, events without verbs) or if examples reference unknown roles.
2. **Annotate raw/** (only if raw/ is non-empty). One LLM, given the legend and the SOP Lang rules, writes the wire(s) of each sentence;
   `sop check` validates every wire (invented keys or types are rejected and re-asked once); write `examples/annotated_<date>.sop` with
   `_source=llm:<model>`. The control is the reader, in step 6b, not a second annotator.
3. **Kinds.** Build the word -> kind table from the legend's `kinds` source (lexicon or tagger) over all tokens in examples; unknown words get
   a kind from the tagger or `unknown`; write it to model/catalog.json. Report the share of tokens with kind `unknown`.
4. **Splits**, written to runs/<ts>/splits.json before training: seen formulations; held-out sentences; held-out FORMULATIONS (sentence
   patterns never seen, by template signature when examples are templated, by clustering of role-order otherwise); new identities (fillers
   never seen). A lexical bag baseline is trained on the same rows.
5. **Train** with the reference trainer (memory logged, checkpoints, 5 seeds when the data allow it, 1 seed otherwise and say so) on
   the world's constellations built from the examples' wires; D from config (default 1024); thresholds set on calibration rows, never on the
   reported rows (rule 25).
6. **Evaluate** and write model/metrics.json and model/REPORT.md: per-role accuracy, FULL frame, the three occupancy cases, calibration
   (confident-wrong at tau, coverage, acc-vs-coverage), the held-out-formulation split (the number that says how far the world's language is
   covered), new identities, the lexical baseline on the same rows, agreement with the annotation on a held-out slice. Label everything
   BUILT; list what is weak and what kind of examples would help (which formulations, which roles).
6b. **Second look.** (The model choice is measured, not assumed: on a constructed world with gold frames, annotate 500 generated sentences with the
   small and the large model and report both against the gold; the result decides the default in world.json.) The trained reader re-reads the training sentences. Where its margin is above tau and its frame disagrees with the
   annotation, the LLM re-annotates that sentence with both readings in front of it and keeps one; write the cases to
   `audit/disagreements.sop`. REPORT.md states the disagreement rate and how many annotations changed — the measured label noise. If more than
   a stated share changed (world.json, default 5%), retrain once on the corrected examples. A human audit (`audit/sample_*.sop`) is optional.
7. **Package**: model/ complete per section 4; status.json updated; runs/<ts>/ has the log and the exact command. Then run
   `sop formalize` on a held-out slice and append the reader's wires next to the LLM's, for anyone who wants to look.

8. **Evolve the legend** (section 1.1, steps 3-5): read the feedback of this training and `requests/`; decide a change set; write the
   migration; re-annotate, retrain and evaluate v+1 on the held-out set fixed before the change; accept or reject by the criterion; write
   `versions/v<N+1>/CHANGES.md`. Repeat while a change set is found and the budget of the run allows (budget stated in world.json:
   max versions per run). Removals/merges only of roles empty across the corpus.

Acceptance of a training (the skill checks before declaring BUILT): `sop check` passes on model/legend.json + examples; metrics.json has
every field of step 6; REPORT.md states n, provenance, the disagreement rate of step 6b; confident-wrong at the stated tau is reported (no threshold on it: a high
value is a finding, not a failure); the CLI formalizes `_text` of ten random examples and reproduces their role lines on the seen split.

## 7. First world to build with this (proposal)

The command world of the alignment chapter: actions (delete, save, send, move, copy, notify) as event types, with roles (patient, destination,
condition, principal), operators `not`, `if`, `says`; moods command / statement / question. It exists as a constructed world (scope 1.000 on
templates, 0.997 at the unseen depth); the pilot replaces its templates with real requests written by people (or LLM paraphrases of them),
annotated as in steps 2 and 6b. Criterion for "real": on held-out real sentences, confident-wrong < 2% at whatever coverage results, coverage
reported; the held-out-formulation split reported separately. The numbers go to STATUS.md as the first real-text row.

## 8. Worked example, end to end (the commands world)

8.1 `legend.json`
```
{
  "version": 1,
  "roles": [
    {"id":"patient",     "question":"to what?"},
    {"id":"destination", "question":"where to / to whom?"},
    {"id":"condition",   "question":"under what condition?"},
    {"id":"principal",   "question":"who asks / on whose behalf?"}
  ],
  "events": [
    {"id":"delete", "verbs":["delete","remove","erase","drop"]},
    {"id":"save",   "verbs":["save","store","keep"]},
    {"id":"send",   "verbs":["send","forward","mail"]},
    {"id":"move",   "verbs":["move","transfer","relocate"]},
    {"id":"copy",   "verbs":["copy","duplicate"]},
    {"id":"notify", "verbs":["notify","tell","inform","alert"]}
  ],
  "operators": ["not","if","says","and"],
  "moods": ["statement","command","question"],
  "kinds": {"source":"lexicon","lexicon":"kinds.tsv","fallback":"tagger:spacy:en_core_web_sm"},
  "values": [
    {"kind":"id",   "pattern":"#?\\d{3,}"},
    {"kind":"date", "pattern":"\\d{4}-\\d{2}-\\d{2}|tomorrow|today|on (Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day"},
    {"kind":"path", "pattern":"[\\w./-]+\\.(pdf|docx|csv|txt)"}
  ]
}
```
The action itself is not a role: it is the wire type (the predicate role of the reader, read through the synonym table), so "what is to be
done?" is answered by the header line. `kinds.tsv` (word TAB kind; the kinds are the reader's categories, one learned key per kind):
```
delete	VERB
files	NOUN
the	DET
unsaved	ADJ
to	PREP
archive	NOUN
please	FILLER
if	COND
not	NEG
Maria	PERSON
```

8.2 `raw/requests_2026-10.txt` (what the team supplies; one sentence per line)
```
Please delete the unsaved files.
Do not send the invoice to the client before Friday.
If the backup finished, move the logs to the archive.
Maria says that the report should not be forwarded to legal.
Which files did you move to the archive?
Copy the report and notify Paul.
```

8.3 `examples/annotated_2026-10-16.sop` (what the agent writes after annotation; the same format the CLI emits)
```
!world commands
!legend sha256:3f9a7c…
!source llm:<model>; reader second look in audit/disagreements.sop

@w1 delete
_text=Please delete the unsaved files.
_sentence=0
_mood=command
patient=$w2
_empty=destination,condition,principal
_source=llm:<model>

@w2 entity
head=files
det=the
attr=$w3

@w3 not
scope=saved

@w4 not
_text=Do not send the invoice to the client before Friday.
_sentence=1
_mood=command
scope=$w5

@w5 send
patient=invoice
destination=client
condition="before Friday"
_empty=principal

@w6 if
_text=If the backup finished, move the logs to the archive.
_sentence=2
_mood=command
condition=$w7
consequence=$w8

@w7 unknown
_unplaced=backup,finished
# the legend has no event for "backup finished": the condition is kept as an opaque wire; stated, not hidden

@w8 move
patient=logs
destination=archive

@w9 says
_text=Maria says that the report should not be forwarded to legal.
_sentence=3
_mood=statement
speaker=Maria
scope=$w10

@w10 not
scope=$w11

@w11 send
patient=report
destination=legal
_empty=condition,principal

@w12 move
_text=Which files did you move to the archive?
_sentence=4
_mood=question
_asked=patient
patient=?
destination=archive

@w13 and
_text=Copy the report and notify Paul.
_sentence=5
_mood=command
items=$w14 & $w15

@w14 copy
patient=report

@w15 notify
destination=Paul
```

8.4 Training (the coding agent, following the skill) — the shell it ends up running
```
sop check --legend worlds/commands/legend.json worlds/commands/examples/*.sop
python -m sop.train --world worlds/commands --seeds 5 --D 1024      # writes runs/<ts>/, model/, status.json
sop stats --model worlds/commands/model
```
`model/REPORT.md` then states: n sentences, provenance (how many human, how many LLM), the second-look disagreement rate and changes, the
splits, the metrics table (per role, FULL, calibration, held-out formulations, new identities, lexical baseline), and "what would help".

8.5 Formalizing new text with the CLI
```
$ echo "Do not move the unsaved logs to the archive before Friday." | sop formalize --model worlds/commands/model
!world commands
!legend sha256:3f9a7c…
!generated sop 0.1, model worlds/commands/model (2026-10-16T18:40Z)

@w1 not
_text=Do not move the unsaved logs to the archive before Friday.
_sentence=0
_mood=command
scope=$w2
_margin=0.27
_status=ok
_depth=0

@w2 move
patient=$w3
destination=archive
condition="before Friday"
_empty=principal
_margin.patient=0.31
_margin.destination=0.22
_margin.condition=0.09
_margin=0.09
_status=ok
_depth=1

@w3 entity
head=logs
det=the
attr=$w4

@w4 not
scope=saved
```
With `--tau 0.10` the same input gives `_status=abstain` on w2 (the condition margin 0.09 is under tau) and the consumer knows exactly which
question is uncertain. A sentence outside the world:
```
$ echo "The weather in Lisbon is lovely." | sop formalize --model worlds/commands/model
@w1 unknown
_text=The weather in Lisbon is lovely.
_sentence=0
_unplaced=weather,Lisbon,lovely
_residual=0.91
_status=no-sense
```

8.6 Verifying a formalization made by someone else (an LLM, a person)
```
$ sop formalize --model worlds/commands/model request.txt -o reader.sop
$ sop diff llm.sop reader.sop
!left llm.sop
!right reader.sop

@d1 verdict
left=w1
right=w1
_verdict=contradiction
_role=destination
_left=client
_right=legal
_margin.right=0.22
```

8.7 The server, from a shell
```
node server.js --root ./worlds --port 8080
curl -X POST localhost:8080/worlds/commands
curl -T legend.json localhost:8080/worlds/commands/files/legend.json
curl -T requests.txt localhost:8080/worlds/commands/files/raw/requests_2026-10.txt
curl -X POST localhost:8080/worlds/commands/check
curl -X POST --data-binary @new_requests.txt localhost:8080/worlds/commands/formalize     # 503 until model/ exists
```

8.8 `skills/train-world/SKILL.md` — skeleton
```
---
name: train-world
description: Train the reader on a world folder (legend.json + examples/*.sop), annotating raw/ first; the reader's second look measures label noise; write model/ and REPORT.md.
---
Inputs: a world folder path or server URL. Never edit model/ by hand. Steps 1-7 of the specification, each with its check. Stop and report
when: the legend is inconsistent; examples fail `sop check`; fewer than N annotated sentences exist (N stated in world.json); the memory
curve is not flat. Output: model/, runs/<ts>/, status.json, and a five-line summary (n, FULL, confident-wrong at tau,
coverage, held-out-formulation FULL) in the chat.
scripts/: inventory.py, annotate.py, kinds.py, splits.py, train.py (the reference trainer), evaluate.py, second_look.py, evolve_legend.py, package.py
```

## 9. Several worlds: dispatch of mixed text

A complex system holds many worlds, each with its own reader. Mixing happens at the level of SENTENCES (a document or a conversation carries
sentences from several worlds; a sentence is, as a rule, from one). Three dispatch strategies, cheapest first; the labels are those of the
research status:
1. **Explicit, by channel or context** (BUILT by construction): the command channel goes to the commands world, the invoice import to the
   invoices world. `sop formalize --model <that world>`.
2. **By fit** (N7, PLANNED with its criterion: identification > 0.95 on single-world sentences with the vocabulary overlap stated): every
   world's reader reads the sentence; the dispatcher picks the smallest relative residual, with the mean margin as the second rule; if every
   residual is high the sentence is from no known world (`unknown`, `_world=none`). Readers cost microseconds, so a hundred worlds cost nothing.
   The identification's confusion matrix and confident-wrong rate are reported like any other reading. Output: one file, several `!legend
   <world> sha256:…` directives, `_world=` and `_world_margin=` on every top-level wire.
3. **Sticky context** (HYPOTHESIS, not measured): a conversation stays in its current world until the fit says otherwise, with hysteresis,
   so that one ambiguous sentence does not switch worlds; entity memory across sentences (pronouns, B8) is per world.
What stays with the LLM in such a system: cutting free text into units; everything no reader places (`unknown` wires); the dialogue. The
semantic layer says what was understood, in which world, with what certainty; the rest flows on. Sentences that mix worlds inside themselves
(literature, metaphor) are out of scope and stated so.

## 10. Out of scope for v1 (stated)
Pronoun resolution across sentences; right-announced structure ("Q if P" with the consequence first); mixed worlds in one
sentence; authentication, multi-user, HTTPS on the server; training triggered by the server.
