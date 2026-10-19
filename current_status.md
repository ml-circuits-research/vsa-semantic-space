# Current status — the freeze (tag paper-2026-10) lifted for ONE run: 7b-bis, the transformer control with categories (2026-10-19)

Labels (rule 22): BUILT / PLANNED / HYPOTHESIS. Sources: experiment/RESULTS.md 7b-bis (this judgment point), 11-octies, 15, 15-bis;
experiment/JUDGMENT_2026-10-17_8d335f8.md and _753287f.md; STATUS.md; experiment/JOURNAL.md; tool/README.md.

## A. Done since the freeze (judgment point after 6dca504)
| step | result | outcome |
|---|---|---|
| PR #1 reviewed and merged (9ea8adc) | experiment/v22_transformer_categories.py (external review, Astra, 2026-10-19): the 7b control with ONE change, the reader's 32 word categories at the transformer's input; criterion in the header before the run; v9_transformer.py unchanged. Follow-up 18dc7fd: import guard, depth-2 per-role print, checklist (no change to the computation) | merged |
| run 7b-bis (DGX, CUDA, 5 seeds; logs/2026-10-19_1525_v22_transformer_categories_18dc7fd.txt) | chosen lr 3e-4 / dropout 0 by seed-0 validation; new ids 0.253±0.034 (per seed 0.213-0.304; 7b 0.215; reader 1.000); depth 3 0.393±0.023 (per seed 0.365-0.434; 7b 0.390; reader 0.996); inner roles at depth 3 at chance, the two known speakers placed (7b's signature); categories lift the seen / flat axes (seen 1.000, depth 1 0.976, role cues 1.000) | decision rule as written: < 0.5 on both axes -> the 7b gap is the mechanism, not the category input. STATUS transformer-control row updated. The paper is NOT changed here; the reviewing session updates it from RESULTS 7b-bis |
| earlier at the freeze (judgment 753287f) | see JOURNAL 2026-10-17 (evening) and tag paper-2026-10 | unchanged |

## B. PLANNED after submission, in order (judgment 753287f section F)
1. Set-size reading of a set-valued role on the WikiSQL world: (a) a relative member rule on the stored scores (re-scoring; CPU; permitted before
   submission, not required), then (b) a count role (one retraining). Criterion: WikiSQL dev, 3 seeds, decided set exact >= 0.55 with the
   count-given reading re-measured beside it (0.628 on seed 0); FULL decided >= 0.38; the false-size rate on the 0-condition questions printed.
2. P1 when the team's data arrives: the legend written first; the annotator path once with a key (the 500-sentence annotator measurement); a
   300-sentence audited gold set before any number is reported; B2 calibration on the real dev.
3. N6-bis (the stronger omitted-role test), low priority unless P1's legend loop needs it.
4. Capacity: D = 4,096-16,384, 64-1,000 identities, with B6's storage measurement repeated at the larger D.
5. N7 after N1; N8 (verb-final world) low; B8-B9 HYPOTHESIS.
Before F, after submission: the repository consolidation of judgment 753287f section E — front-page README, the script index (00_READ_FIRST §9 and
README_RUN.md), the judgment files to experiment/judgments/, run and log indexes, the regression gate (tool/tests/run_all.sh + one CPU
reproduction), requirements pinned, the policy in GIT_WORKFLOW.md; scripts not moved; nothing of provenance deleted.
What the tool can demonstrate now (judgment 753287f section G): WikiSQL behind the server, within the stated bounds (values invisible; one event
per sentence; set size open; abstention shown as a feature); open text not credible now; the first real demonstration is P1's domain.

## C. HYPOTHESIS (unchanged)
Presence calibration across formulations (~10% class overlap, the reader's); B5d multi-hop left re-seating (only if a legend needs the subject's
term as a structure witness); the verifier's condition axis (legend choice settled on data by E2: a set-valued role; size open); N5 "resolve
before drawing" (diagnostic in the next kinds run).

## D. Open questions
1. For the reviewing session: carry 7b-bis into the paper's control paragraph (the comparison now isolates the mechanism on new identities
   and on the unseen depth). Experiments are frozen again after this run; nothing else runs.
