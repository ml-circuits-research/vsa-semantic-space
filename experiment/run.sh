#!/usr/bin/env bash
# Runs a python script and saves the log with the date and the current commit hash in its name.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: ./run.sh script.py [arguments]"; exit 1; }
# logs/ is ignored by the check: new logs are always uncommitted at run time
if [ -n "$(git status --porcelain 2>/dev/null | grep -v "logs/")" ]; then
  echo "There are uncommitted changes. Commit first (git add -A && git commit), so that the log corresponds to a commit."; exit 1
fi
# protection (2026-10-16, after the machine hung on 2026-10-15): one run at a time, enough free memory, a wall-clock cap
LOCK=logs/.run.lock
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "Another run.sh is active (pid $(cat "$LOCK"), lock $LOCK). Runs are sequential; wait for it or stop it first."; exit 1
fi
if pgrep -f "python3? v[0-9]+_|python3? frame_roundtrip|python3? diag_" >/dev/null; then
  echo "Another experiment process is running (pgrep). Runs are sequential on this machine; wait for it or stop it first."; pgrep -af "python3? v[0-9]+_" | head -3; exit 1
fi
avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "$avail_gb" -lt 40 ]; then echo "Only ${avail_gb} GB available; refusing to start (need 40)."; exit 1; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VSA_MEM_FRACTION=${VSA_MEM_FRACTION:-0.5} VSA_MIN_AVAILABLE_GB=${VSA_MIN_AVAILABLE_GB:-16} VSA_MAX_HOURS=${VSA_MAX_HOURS:-6}
mkdir -p logs
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
hash=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
name=$(basename "$1" .py)
out="logs/$(date +%Y-%m-%d_%H%M)_${name}_${hash}.txt"
{ echo "# $(date -Iseconds)  commit=$hash  cmd: python3 $*  (guard: mem fraction $VSA_MEM_FRACTION, min available $VSA_MIN_AVAILABLE_GB GB, max $VSA_MAX_HOURS h; MemAvailable at start ${avail_gb} GB)"; echo "# $(python3 --version 2>&1)  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"; } > "$out"
( timeout --signal=TERM $((${VSA_MAX_HOURS%.*}*3600+600)) python3 -u "$@" 2>&1 | tee -a "$out" ) &
pid=$!
# log-advance watchdog: if the log has not grown for STALL_MIN minutes, kill the run (default 30)
STALL_MIN=${STALL_MIN:-30}; last=$(stat -c %s "$out"); stalled=0
while kill -0 $pid 2>/dev/null; do
  sleep 60; now=$(stat -c %s "$out")
  if [ "$now" -eq "$last" ]; then stalled=$((stalled+1)); else stalled=0; last=$now; fi
  if [ $stalled -ge $STALL_MIN ]; then echo "# watchdog: log did not advance for $STALL_MIN min; killing the run" | tee -a "$out"; pkill -TERM -P $pid; kill -TERM $pid; break; fi
done
wait $pid
echo "log saved: $out"
