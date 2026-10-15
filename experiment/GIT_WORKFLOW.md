# How we work through GitHub (so that a new Claude session picks up the context without losses)

## 1. Initialization (once, on the DGX)
    unzip spatiu_semantic_vsa.zip -d spatiu-semantic && cd spatiu-semantic
    git init && git add -A && git commit -m "import: starea de la 2026-10-15 (3 runde sintetic, SICK, WikiSQL)"
    # repo on GitHub (private is ok), then:
    git remote add origin git@github.com:<user>/spatiu-semantic.git && git push -u origin main

## 2. What gets committed (EVERYTHING relevant; missing results are the only real loss)
  - code: *.py (any change, however small)
  - raw results: logs/<date>_<script>_<hash>.txt  (produced automatically by run.sh, see 4)
  - RESULTS.md: the new tables, in the existing format (model | metric | seeds); JOURNAL.md: one entry per session (see 3)
  - EVALUATION_PRINCIPLES.md if the evaluation changes (and re-run an old experiment with the new evaluation, as a link point)
  - small data (< 50 MB) in corp/; large data NO (put the download command in corp/README.md)
  - NO: large checkpoints, __pycache__, temporary files (.gitignore included)

## 3. JOURNAL.md: one entry per work session, 3-8 lines, fixed format
    ## 2026-10-16 (evening) — <short title>
    Ran: run.sh v4 (5 seeds, D=4096)           -> logs/2026-10-16_v4_a1b2c3d.txt
    Result: new form. unseen trans. 0.91±0.02 (confirms the 0.90 from the CPU)
    Surprise / problem: ...
    Changed in code: v4.py: ... (why: ...)
    Next step: ...
  Write also when it did NOT work. A recorded failure is worth as much as a success.

## 4. run.sh: runs a script and saves the log with date + commit hash in the name (does not depend on discipline)
    ./run.sh v4.py            # -> logs/2026-10-16_v4_a1b2c3d.txt  (+ also prints to the screen)
    ./run.sh gates.py
  run.sh refuses to run if there are uncommitted changes (so that the log corresponds EXACTLY to a commit). Commit first, then run.

## 5. Commit messages: say WHY, not WHAT
    good:  "evaluation: addresses from the world instead of the active form (penalized overlapping addresses)"
           "v4: bigram code added; hypothesis: roles are local"
    bad:   "update", "fix", "wip"
  One commit = one idea. The git history is the history of the reasoning.

## 6. How to hand the context to a new Claude session
  Give it the link to the repo (or add the repo through the GitHub connector) and say:
    "Read in this order: 00_READ_FIRST.md, JOURNAL.md (all of it), EVALUATION_PRINCIPLES.md, then RESULTS.md and logs/ from the last week.
     Do not propose experiments before telling me what you understood from JOURNAL.md and which results seem unconfirmed to you."
  If you want to communicate asynchronously: open a GitHub Issue per question/experiment; the answers and decisions stay there, not in chat.
  Useful labels for Issues: `to-validate`, `evaluation`, `idea`, `blocked`.

## 7. Branches
  main = everything that is verified (evaluation according to the principles, >=3 seeds). Speculative experiments on branches `exp/<name>`;
  they are merged into main only with an entry in JOURNAL.md and a table in RESULTS.md.
