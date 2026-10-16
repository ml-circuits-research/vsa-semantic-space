#!/usr/bin/env bash
# queue.sh — runs ./run.sh invocations strictly in sequence (judgment 2026-10-16 E6: a wait loop never greps for a pattern that occurs in its
# own command line; it polls the pid lock and nothing else). Usage: ./queue.sh "v16_reference.py --seeds 5" "v17_verifier.py" ...
# Each item is one run.sh invocation (script + arguments, quoted as one word). A refused run (dirty code tree, low memory) is reported and
# the queue continues with the next item. Output per item: the last QUEUE_TAIL lines (default 14), cut at QUEUE_CUT columns (default 300).
set -u
cd "$(dirname "$0")"
for spec in "$@"; do
  while [ -f logs/.run.lock ] && kill -0 "$(cat logs/.run.lock 2>/dev/null)" 2>/dev/null; do sleep 30; done
  echo "# queue $(date +%H:%M) start: $spec"
  ./run.sh $spec 2>&1 | grep -v "^\[train" | tail -n "${QUEUE_TAIL:-14}" | cut -c1-"${QUEUE_CUT:-300}"
  echo "# queue $(date +%H:%M) end: $spec"
done
