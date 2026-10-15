# Data-efficient text-to-SQL role states without the table: a displacement-addressed vector-symbolic model against same-input baselines

Short empirical paper (findings / workshop format), draft v1 (2026-10-15). Claim, exactly: on WikiSQL, with the SAME input (fixed random word
identities + distributional categories from the whole unlabelled training corpus; no table at the input), the same objective and the same
decoder, a displacement-addressed vector-symbolic model recovers the selected column far better than a small transformer and a bag-of-words
baseline at every training size from 1k to 56k, while exposing a readable role state; it loses the aggregator, a lexical decision, to the
neural classifiers; its equivalence separation is lexical overall and role-based only on hard pairs. Every number cites RESULTS.md 8b-8d
and a log with a commit hash. [BUDGET CHECK: fill from logs/2026-10-15_1652 and _1655 — gate for the claim, section 4.]

## 1. Task and world
WikiSQL (Zhong et al. 2017): 56,355 training questions, 8,421 dev, 15,878 test, each with a table and a SQL query (SELECT agg(col) WHERE
conditions). We treat the query as a role-filler state in a complex vector space of dimension D = 512: S_sql = SEL (x) bundle(selected column
tokens) + AGG (x) f_agg + sum_k COND_k (x) [COL (x) bundle(column) + OP (x) f_op + VAL (x) bundle(value)], with fixed random role vectors and
fixed random identities f_w per word type (shared by the model). The model maps the question to a state S and is trained to maximize
cos(S, S_sql). Decoding uses the world's addresses: the selected column is the table column whose bundle is closest to S (x) conj(SEL), the
aggregator the closest of six, condition columns likewise; the table enters ONLY as the set of decoding candidates. We report selected
column, aggregator, first-condition column, FULL query (sel + agg + all condition columns correct at once) and cos(S, S_sql).

## 2. Models on the same input
Input for every sequence model: the sequence of fixed identity vectors of the question tokens and the same 90 category ids — the 42 most
frequent tokens as singletons plus 48 k-means clusters of skip-gram vectors trained on the training questions and headers only (no dev/test).
- Ours: three address codes per word, learned per category — path (sum of category displacements along the prefix), bigram (previous + own
  category), and a chain of expectations whose gates (which categories emit) are learned by gradient; state = sum of address (x) identity.
  ~2.5M phase parameters. Adam, lr 0.02, batch 32.
- Transformer, same input: linear(2D -> 128) + learned category embedding + sinusoidal positions, two pre-norm encoder layers (4 heads,
  d_ff 256), masked mean pooling, linear -> complex S; 0.8M parameters; lr chosen once among {1e-3, 3e-4} on 2,000 questions held out from
  TRAIN at 56k (1e-3), AdamW with cosine schedule.
- Bag of words + logistic regression: aggregator from a hashed bag of the question; selected column from per-(question, column) features
  (token overlap, overlap fraction, header length, position).
Epochs scaled for comparable gradient steps: 60 / 20 / 3 at 1k / 5k / 56k. Three seeds. No tuning of ours beyond the values used on the
synthetic worlds. The skip-gram categories are trained on the whole training pool at every size, for both sequence models — the data-
efficiency claim therefore reads "with distributional categories from the whole unlabelled training corpus" [ABLATION: categories from the
1k/5k subset only — fill from the ablation log].

## 3. Results (RESULTS 8d; logs/2026-10-15_1628_v12_wikisql_baselines_ed24ce4.txt)
Figure 1 (to draw): selected-column accuracy vs training size, three curves. Table 1 (dev; test at 56k in brackets; std <= 0.005):
| n train | model | selected column | aggregator | condition-1 column | FULL query |
|---|---|---|---|---|---|
| 1k | ours | **0.632** | 0.766 | **0.387** | **0.164** |
| 1k | transformer, same input | 0.415 | **0.819** | 0.360 | 0.115 |
| 1k | BoW + LR | 0.475 | 0.762 | — | 0.371 (sel + agg only) |
| 5k | ours | **0.760** | 0.800 | **0.555** | **0.294** |
| 5k | transformer, same input | 0.510 | **0.864** | 0.415 | 0.164 |
| 5k | BoW + LR | 0.502 | 0.810 | — | 0.413 (sel + agg only) |
| 56k | ours | **0.816** [0.821] | 0.801 [0.803] | **0.626** [0.629] | **0.368** [0.374] |
| 56k | transformer, same input | 0.543 [0.544] | **0.893** [0.893] | 0.450 [0.433] | 0.196 [0.190] |
| 56k | BoW + LR | 0.499 | 0.870 | — | 0.441 (sel + agg only) |
Chance: selected column 0.166, majority aggregator 0.715; a pure token-overlap column pick gives 0.484.
The selected column is the headline: 0.82 vs 0.54 vs 0.50 at 56k, and the gap is present at 1k (0.63 vs 0.42 vs 0.48). The full query is
0.37 vs 0.19 under these conditions; the aggregator is a lexical decision and the neural classifiers win it (0.89). A table-at-the-input
variant of ours (header tokens + question) drops to 0.67 on the selected column: the header's identities dominate the bundle; the table
belongs in the decoder (as here) or bound to a role, not summed (RESULTS 8b).

## 4. Budget check for the transformer (gate for the claim)
With steps matched to ours (3 epochs at 56k, 23 s), the transformer reaches 0.543. With 4x the budget (12 epochs): 0.631 dev / 0.626 test,
FULL 0.281 / 0.267. With 10x (30 epochs): [FILL]. Decision rule stated in advance: if the transformer reaches 0.75+ at 10x, the claim narrows
to data efficiency at 1k-5k; otherwise it stands at every size.

## 5. Equivalence and hard pairs (RESULTS 8d)
Pairs of dev questions on the same table, same SQL skeleton (sel, agg, condition columns) vs different; and two hard sets built from SQL and
token overlap: H1 "different words, same SQL" (Jaccard < 0.3, n = 26) and H2 "same words, different SQL" (Jaccard >= 0.5, n = 1,353).
Each row is a representation; the bag of identities is the lexical control in the same space.
| representation | same | different | separation | AUC | H1 (n = 26; want high) | H2 (n = 1,353; want low) |
|---|---|---|---|---|---|---|
| bag of identities (lexical) | 0.760 | 0.462 | 0.298 | 0.858 | 0.337 | 0.772 |
| ours (56k, 3 seeds) | 0.755 | 0.464 | 0.291 | 0.914 | 0.462 | **0.622** |
| transformer, same input | 0.910 | 0.616 | 0.293 | 0.900 | 0.859 | 0.744 |
The overall separation is lexical: ours and the bag separate same-vs-different skeletons equally (0.29 vs 0.30), so WikiSQL's headline
equivalence metric does not test roles — the set has only 26 pairs with different words and the same SQL, too few for a conclusion. Where
such pairs exist in number (H2), the role state moves same-words/different-SQL pairs apart where the lexicon cannot (0.62 vs 0.77; the
transformer stays at 0.74). We state this as a negative with that nuance, not as "equivalence works".

## 6. What the state exposes
frame(S): unbinding with the world's roles and cleanup over the table's columns returns {sel, agg, conds: [(col, op, val)]} directly — the
decoder of section 1 is already that function; the same frame is what a verifier compares (future work, FRAME module).

## 7. Caveats
Single dataset; the table is not used at the input by any model here (the state of the art, ~0.9 full query, uses it with pretrained
encoders); three seeds; the transformer is small and tuned minimally [budget check in section 4]; distributional categories from the whole
training corpus [ablation]; dev and test agree to the third decimal.

## 8. Related work
Zhong et al. 2017 (WikiSQL, Seq2SQL); Xu et al. 2017 (SQLNet); Hwang et al. 2019 (SQLova); Kleyko et al. 2022 (VSA survey); Plate 2003;
Frady, Kleyko, Sommer (fractional power encoding).
