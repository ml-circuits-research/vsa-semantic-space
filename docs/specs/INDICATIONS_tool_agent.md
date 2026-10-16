# Indications for the agent building the worlds server, the `sop` CLI and the train-world skill (SPEC_worlds_server_sop_cli.md)
(received from the author on 2026-10-16, saved verbatim)

## 1. Models — which one does what (the loop must never require a model above the Opus class)

The skill names models by ROLE, read from world.json `models`, never hard-coded. Defaults:

| step | role key | default class | why |
|---|---|---|---|
| inventory, kinds, splits, training, evaluation, packaging | `pipeline` | small (Haiku class) | tool calls over scripts with written criteria; a larger model adds nothing |
| annotation of raw/ (sentence -> SOP wires, legend in front) | `annotate` | small (Haiku class) | a closed, specified labelling task, guarded by `sop check` and by the reader's second look |
| second look (re-annotate where the reader confidently disagrees) | `second_look` | small (Haiku class) | same task with two readings in front; escalates by rule, below |
| legend proposal (v1 from a raw sample) and legend evolution (name new roles/events, write migrations, decide change sets) | `evolve` | large (Opus class) | judgement-heavy, runs once per version; cost negligible |
| the implementing agent itself (writing the server, CLI, skill) | — | Opus class is enough | nothing here needs more |

Escalation rule (in the skill, not in a person's head): if the second-look disagreement rate of a training exceeds `second_look_escalate`
(world.json, default 0.05), `annotate` and `second_look` move to the large class for that world and the raw/ set is re-annotated once; the
REPORT states the switch and both rates. Nothing in the loop ever requires a model above the Opus class; if a step cannot be done at that
class, the skill reports it as a limit instead of asking for a larger model.

The default for `annotate` is a MEASUREMENT, done once while building (spec 6b): on the constructed commands world (gold frames exist),
annotate 500 generated sentences with the small and with the large model; FULL-frame agreement with the gold for each; if the small model is
within 0.02 of the large one, the small default stands; otherwise the default becomes the large class and the number is written in the
skill's README. Report both numbers whatever they are.

## 2. Build order (each step has a test before the next starts)

1. `sop check` + the SOP Lang parser/writer (spec 3, 3.6): tests = the worked example of spec 8.3 parses, round-trips byte-for-byte, and
   every negative case in 3.6 (duplicate id, dangling `$ref`, cycle, unknown key, `&`/`|` mixed, `_asked` without question mood, legend hash
   mismatch) is rejected with a line number and a reason.
2. `sop formalize` on the repository's constructed worlds (reference, commands, quantifiers): numpy-only inference from a `model/` folder
   exported by the repository's trainers; test = the generated test sentences of each world, formalized, reproduce the RESULTS tables' role
   accuracies within seed variance (the "P0 round trip" row). Deterministic output for the same model and input.
3. `sop diff` (= frame.py's diff, four verdicts) and `sop stats`; test = injected-error pairs from v17 give the 10c verdict distribution.
4. The server (spec 2): standard library only; test = the shell script of spec 2 (create, upload, check, download byte-identical, formalize
   503 without a model, path traversal rejected, writes to model/ and runs/ refused).
5. The skill (spec 6, steps 1-8) with its scripts; test = the full loop on the constructed commands world with its templates as raw/ (the
   legend given), one version: model/ complete, REPORT.md with every field, second-look rate reported; then one evolution step with a
   deliberately incomplete legend (one role removed from legend.json, sentences kept): the loop must PROPOSE that role back from `_unplaced`
   / residual clusters and accept it by the criterion — this is the test that the loop finds a missing question.
6. The demo page (spec 2, `GET /`): worlds, uploads, version history, a textarea -> formalize.

## 3. Rules that carry over from the research repository
- Every threshold set on calibration rows, never on reported rows (rule 25); every number with its label (BUILT / PLANNED / HYPOTHESIS).
- The reader learns the route book only; roles, identities, kinds and event synonyms are inputs (rule 20) and the REPORT says which is which.
- Nothing is silently dropped: a sentence the reader cannot place yields an `unknown` wire with `_unplaced`.
- Add freely, remove only what is empty across the corpus (spec 1.1); versions explicit, hash-checked; CHANGES.md per version.
- No time estimates in READMEs or reports; criteria and logs.

## 4. Deliverables
`server.js` (one file), `sop/` (package with `formalize`, `check`, `diff`, `stats`, `train` entry point), `skills/train-world/` (SKILL.md +
scripts), `tests/` (the six tests above, runnable in one command), `README.md` (install, the one measurement of section 1, the limits of v1
per spec 9). Report back: the P0 round-trip numbers, the small-vs-large annotation numbers, and the missing-role recovery result of test 5.
