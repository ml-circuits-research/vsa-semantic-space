#!/usr/bin/env bash
# Runs a python script and saves the log with the date and the current commit hash in its name.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: ./run.sh script.py [arguments]"; exit 1; }
# logs/ is ignored by the check: new logs are always uncommitted at run time
if [ -n "$(git status --porcelain 2>/dev/null | grep -v "logs/")" ]; then
  echo "There are uncommitted changes. Commit first (git add -A && git commit), so that the log corresponds to a commit."; exit 1
fi
mkdir -p logs
hash=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
name=$(basename "$1" .py)
out="logs/$(date +%Y-%m-%d_%H%M)_${name}_${hash}.txt"
{ echo "# $(date -Iseconds)  commit=$hash  cmd: python3 $*"; echo "# $(python3 --version 2>&1)  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"; } > "$out"
python3 "$@" 2>&1 | tee -a "$out"
echo "log saved: $out"
