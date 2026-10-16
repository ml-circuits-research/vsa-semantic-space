# Current status (rewritten whole at every judgment point; last: 2026-10-16 evening — judgment 3b4f837 applied: C fixes, E1 (i)-(ii), C2b, E3, N5-N7 rows)

Labels (rule 22): BUILT / PLANNED / HYPOTHESIS. Sources: experiment/JUDGMENT_2026-10-16_3b4f837.md and _addendum_N5-N7.md (saved verbatim),
experiment/RESULTS.md 9c-quater-bis, 9c-quinquies, 10b-quater, 10c-quater; STATUS.md; experiment/JOURNAL.md.

## A. Results, in the judgment's order (all BUILT; logs with hash 8faa02b / edfd146)
| step | result | criterion / decision rule | outcome |
|---|---|---|---|
| C1-C3 | STATUS decoder bullet rewritten; 9c-quater headers say "SCOPE|gold-quantified" and state the labelled depth-2 threshold; STRICT depth-2 columns (depth-1 threshold) added to v14 and re-scored on the saved C2 models; queue.sh evals quoted specs | — | done. Strict: C2 depth-2 presence 0.81 / 0.76 -> **0.26 / 0.17**, FULL 0.39 / 0.37 -> 0.15 / 0.11 — the C2 depth-2 FULL numbers depended on the labelled statistic (9c-quater-bis) |
| E1 (i) trace T2 vs T0 at depth 3 | with 'to' and the participles closed, both persons hang off the period through the two keys P has there; the PASSIVE gives A and C the active form's addresses with the ROLES SWAPPED — the chain is form-blind on T2; C and the predicate share only the 'that' prefix (the 10b-ter crosstalk explanation is not what the trace shows) | emitter test is the right one if C is reached through 'to' alone / the addresses collide | the emitter test was the right one (addresses collide) |
| E1 (ii) participle gates GIVEN open, 5 seeds | depth 3 0.850±0.052; non-passive forms reach 1.00; **passive T2 C 0.52 unchanged**, T3 0.79; unseen transitions 0.64 -> 0.57 | T2 / T3 C > 0.9 at depth 3, no loss | NOT met -> form effect = stated property of the predicate-role world; 0.813 / two-number form + breakdown stays; "on all forms but the passives" written in STATUS and PAPER_NOTE. The by-phrase agent after the recipient is right-announced structure inside one clause |
| C2b trace | (a) wide noun's 2nd drawing = own address (with both 'that') + REL[det]; (b) all other words identical under each / every; (c) each REL relative to its own noun | three checks | all three pass (logs/trace_c2b.txt) |
| **C2b run, slot target, 5 seeds** | nouns 0.998, affirmative read ABSENT 1.000 (d0 and d2), presence d2 0.978 (strict 0.867); **one-quantifier SCOPE|gold-q: d0 0.993, d2 UNSEEN 0.998 (1.000, 0.990, 1.000, 1.000, 1.000)**; two-quantifier (marked) d0 0.685, d2 0.481; determiners 0.919 (cost); both 0.655 | scope > 0.9 at d0 AND d2, one-/two-q separately, nouns 1.00 | **one-quantifier MET** (joins the regularity); two-quantifier NOT met |
| C2b run, noun target | one-q d0 0.976, d2 0.927 (3/5 seeds > 0.9); two-q 0.867 / 0.463; strict presence d2 **0.032**; both 0.838 | same | one-q met on the mean, not strict; the slot target is kept |
| E3 re-scoring (saved frames) | sel confident-wrong 0.053 flat across table sizes (0.03-0.06), **0.169 when the gold column's name is NOT in the question vs 0.023 when it is**; 2/3 of confident-wrong sel from the 21% such questions (reader at 0.45 there); agg confident-wrong 0.137; 50-case audit sample written (logs/audit_confident_wrong_sel_50.jsonl, audit_verdict empty) | breakdown + audit | lexical-mismatch effect, not hard-table; the audit is the team's step |
| N5-N7 | rows filled with the addendum's criteria; literature (worlds mixed inside one sentence) out of scope stated | — | PLANNED |

## B. Effect sizes recorded (judgment E2 decision rule): one-quantifier scope at the unseen depth
no signal (role / slot / noun targets, gates learned) 0.21-0.33 (chance 0.33) -> chain-carried (gates given open) 0.63-0.67 -> beacon-relative
emitter **0.998 (slot) / 0.927 (noun)**. The regularity now holds for seven cases; the exception is the two-quantifier marked scope.

## C. HYPOTHESIS
H-b (relation): with two quantified phrases, which is wide is a relation between the two determiners; one displacement per category cannot
mark 'every' wide alone and narrow next to 'each' (the C2b two-quantifier numbers: 0.69 / 0.87 at depth 0, 0.46-0.48 at depth 2).

## D. PLANNED, in judgment order
Permuted-filler relational term (H-b test; pending the decision in E1 below); N5 (design question E3 below); B5-B7; C5 after the audit; the
reader's confident-wrong on sel/agg vs training size; N6 (possible now), N7 (after N1); N3, N1, N2; B8-B9 (HYPOTHESIS); N4; LLM paraphrases /
formalizations when data arrives; C3 false-both rate (to add to the v14 printout).

## E. Questions for the reviewing session
1. The a8a8fab condition is met (one quantifier composes, two do not). Build the permuted-filler relational term now — emitted by the WIDE
   marker's category ('each', 'certain') as a second drawing of the narrow noun at (its own address) (x) exp(i REL2) (x) rho(f_wide)? Concretely:
   the marker's term holds the OTHER quantified noun under a fixed permutation rho, so "each A ... every B" and "every A ... each B" differ.
   Criterion: two-quantifier scope > 0.9 at depth 0 and 2, one-quantifier unchanged, inert without a quantifier (no loss on 6c/9b/9d/10d
   tables). Declared as a change of principle in 00_READ_FIRST / PAPER_NOTE if built. Or stop with the exception as stated.
2. Keep the slot target as the quantifier world (strict presence 0.87 vs 0.03 for the noun target) — agreed? The noun target gives the better
   "both" (0.84) and the worse strict presence; the two cannot both be the reference.
3. N5 design: in the soft model every category pair has a key, so "a beacon for one of its kinds" needs a definition. Proposal: the ambiguous
   word's term = 0.5 x (term under kind 1 + term under kind 2) for all codes (drawn half-way) UNLESS the first open emitter after it has a
   learned preference (|L[q, k1]| vs |L[q, k2]| in the cosine to the world's slot) — measured, not given — in which case the term is drawn
   under that kind; "resolved by beacon" share = the share where the preference exceeds a stated margin. Confirm or correct before building.
4. E1: the passive's by-phrase agent is right-announced for the chain. Is "on all forms but the passives" the final wording, or should the
   passive forms be moved to the declared-limit row (6c) explicitly?
5. The 50-case audit file is ready for the team.
