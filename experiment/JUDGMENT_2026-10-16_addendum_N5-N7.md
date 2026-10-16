# Addendum to JUDGMENT_2026-10-16_consolidated.md — three experiments added to Part V (N5-N7). Nothing else changes.
(received from the reviewing session on 2026-10-16 together with the judgment at 3b4f837, saved verbatim)

Context: the positioning document gets a chapter on polysemy, several worlds and ontologies. Three claims in it need numbers. Same
rules as Part V (constructed worlds, 5 seeds, criterion first, rule 17/18/22). Order: N5 after B2 (cheap), N6 after B1 (needs the
predicate role and the marker in place), N7 after N1. Add them to STATUS.md as PLANNED with these criteria.

## N5. Kind ambiguity: one word, two categories (the held-aside mechanism applied to lexical ambiguity)
Today every word type has ONE category. Take the reference world and make a set of words that belong to two categories with different
keys (noun/verb homographs: "run", "charge", "book"; in the Romanian world "cer", "car", "vin"), with sentences in which the context
decides the kind. Two variants, same regime: (a) status quo, one category per word (the categorizer picks the majority kind);
(b) category superposition: the word's address is the bundle of the two categories' keys, and the word is HELD ASIDE until a beacon for
one of its kinds appears (the existing left re-seating), drawn half-way if none appears.
Measure: FULL frame on sentences with an ambiguous word; margin on the ambiguous word's domain; share of ambiguous words resolved by a
beacon vs drawn half-way; bag-of-words control. Criterion: (b) within 0.02 of the unambiguous reference on FULL frame; resolved-by-
beacon share reported; (a) reported as the baseline it is.

## N6. A missing question in the legend: can the residual point to it?
Train on a world whose legend LACKS one role that the sentences systematically express (generate with the full world; drop the role
from the target legend, e.g. the time role in the warehouse world). Measure (i) FULL frame on the remaining roles (should hold), (ii) the
reconstruction residual with vs without the missing role in the data (gap stated), (iii) whether the unexplained stars CLUSTER: unbind
the residual with the hidden role's true vector (oracle, for measurement only) and with random vectors; report the cleanup accuracy of
the hidden role's fillers from the residual alone, and the cosine structure of the residuals across sentences (are they one direction or
noise?). Criterion: residual gap > 0.3 relative; fillers of the hidden role recoverable from the residual > 0.8 with the oracle vector
(this says the information is THERE, addressable), 0 with random vectors. This is the number behind "a missing question is detectable";
the step from detectable to proposed (clustering the residual directions without the oracle) is reported if cheap, otherwise HYPOTHESIS.

## N7. Which world is this sentence from? Identification by fit among K legends
Train one reader per world on three worlds that share vocabulary (lending ledger, warehouse, timetable share names, numbers, days; make
the overlap explicit and report it). At test, read each sentence with all three readers; pick the world by the smallest relative
residual (and, as a second rule, by the largest mean margin over its filled domains). Measure: world identification accuracy; the two
rules compared; confusion matrix; the confident-wrong rate of the identification (rule 18). Then a MIXED paragraph (sentences from two
worlds interleaved): per-sentence identification. Criterion: identification > 0.95 on single-world sentences with the stated overlap;
mixed paragraphs reported, no criterion. Say in STATUS.md that texts which mix worlds inside one sentence (literature) are out of scope.
