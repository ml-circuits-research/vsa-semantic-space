# Input format for the LLM-produced formalizations (round 10 B4, LLM version — PLANNED; judgment a8a8fab E5)

The verifier of RESULTS 10c reads a formalization and says error / unverifiable / ok on the role part (selected column, aggregator,
condition columns). The LLM version replaces the gold-frame-with-one-injected-error by formalizations produced OFF this machine by LLMs and
supplied as one file. Nothing here is run until that file exists; this document fixes the format so that the run is one command.

File: `experiment/data/llm_formalizations_dev.jsonl` (not in the repo until supplied; gitignored by size if large). One JSON object per line:
```
{"question": "...",                       # the WikiSQL dev question, verbatim
 "table_id": "1-10015132-11",             # WikiSQL dev table id (the columns come from the table file, never from the LLM)
 "gold":   {"sel": 3, "agg": 0, "conds": [[5, 0, "value"]]},        # WikiSQL gold frame (sel = column index, agg 0-5, conds [col, op, value])
 "llm":    {"sel": 3, "agg": 0, "conds": [[5, 0, "value"]]},        # the formalization to verify, same shape; column NAMES mapped to indices by the producer
 "llm_model": "name-and-version",         # which model produced "llm"
 "second_llm_agrees": true,               # protocol of B3: two LLMs; a disagreement DISCARDS the row (rows with false are kept in the file, skipped at scoring, counted)
 "audited": false,                        # true for the rows in the team's audit sample
 "audit_verdict": null}                   # "correct" | "wrong" | null — the team's judgment on the audited rows
```
Protocol (stated with the data): n >= 1,000 dev rows sampled with a stated seed (`numpy.random.default_rng(<seed>).choice(8421, n, replace=False)`),
two LLMs named with versions, disagreement discards (the discard rate is reported), audit sample of 50 rows by the team (agreement of the audit
with the gold-vs-llm comparison reported), every number labelled with the model names and n.
Scoring (v17 verifier, saved reader of logs/.ckpt_v17_reader_frames.pt or a retrained one with its log): per row, the reader's verdict on
"llm" vs the truth "llm == gold on the role part"; detection = verdict 'error' when llm differs from gold on sel / agg / condition columns;
false alarm = 'error' when llm equals gold on the role part; values (condition values) are stated INVISIBLE; abstention tau set on the
calibration half (rule 25).
