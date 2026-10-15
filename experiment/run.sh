#!/usr/bin/env bash
# Ruleaza un script python si salveaza logul cu data si hash-ul commit-ului curent in nume.
set -euo pipefail
[ $# -ge 1 ] || { echo "folosire: ./run.sh script.py [argumente]"; exit 1; }
# logs/ e ignorat la verificare: logurile noi sunt mereu necomise in momentul rularii
if [ -n "$(git status --porcelain 2>/dev/null | grep -v " logs/")" ]; then
  echo "Exista modificari necomise. Comite intai (git add -A && git commit), ca logul sa corespunda unui commit."; exit 1
fi
mkdir -p logs
hash=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
name=$(basename "$1" .py)
out="logs/$(date +%Y-%m-%d_%H%M)_${name}_${hash}.txt"
{ echo "# $(date -Iseconds)  commit=$hash  cmd: python3 $*"; echo "# $(python3 --version 2>&1)  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"; } > "$out"
python3 "$@" 2>&1 | tee -a "$out"
echo "log salvat: $out"
