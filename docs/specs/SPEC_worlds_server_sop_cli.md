# Specification — worlds server, training skill, SOP Lang, and the `sop` CLI
(version 2, received from the author on 2026-10-16, saved verbatim; this paste ended inside 3.7 — 3.7 to 5 below are taken from the first
version received the same day (tool/SPEC_worlds_server_sop_cli.md), marked; section 6 (training skill) and section 8/9 referenced by the
indications were not received in either version.)

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
REPORT.md says when to add more); a model under a few MB; training in minutes on one small accelerator; reading in microseconds.

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

Mood is not a wire type: `_mood=statement|command|question` is a line on the event wire (in the reader, mood is a role). Quantifier scope is not
emitted in v1 (the determiner stays in `det=`); when the two-quantifier case composes it will be a `_scope=` line, not a new type.

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

3.4 Operators and entities are wires; an operator wraps a wire OR a word (the reader has both: R_op (x) S and R_op (x) f_word).
- `not`: `scope=$w3` (prohibition / sentential negation) or `scope=saved` (a negated word).
- `must` / `may`: `scope=$wN`.   `if`: `condition=$wA`, `consequence=$wB`.   `says`: `speaker=Maria`, `scope=$wN`.
- `and` (events coordinated in one sentence): `items=$w1&$w2`.
- `entity` (a complex filler): `head=files`, `attr=$w5` or `attr=red`, `quantity=40`, `unit=boxes`, `det=the|a` (determiner kept as text;
  quantifier scope is NOT emitted in v1 — the named exception). "the unsaved files" = `object=$w4`; `@w4 entity`, `head=files`, `attr=$w5`;
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

[from here on: text of the FIRST version received (identical structure); the second paste ended at the start of 3.7]

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
```
Exit codes: 0 ok; 1 input/validation error; 2 model missing or legend mismatch. Output is deterministic for the same model and input.
`--json` emits the same content as JSON (one object per wire) for programs. Every wire written by `formalize` has `_margin`, `_status`,
`_empty` [... the paste ended here; section 6 (training skill) not received]
