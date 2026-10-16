"""
guard.py — process-level protection for runs on the DGX Spark (unified memory: a runaway process starves the OS; on 2026-10-15 the
machine stopped accepting SSH and had to be reset). cgroup MemoryMax does NOT bound CUDA unified-memory allocations here (tested:
a 3 GB cuda tensor survived a 2 GB cap), so the protection is in-process:
  1. torch.cuda.set_per_process_memory_fraction(FRACTION): the caching allocator raises an OOM error inside the process instead of
     taking the machine down (default 0.5 of the unified total ~ 60 GB; env VSA_MEM_FRACTION overrides).
  2. a watchdog thread reads /proc/meminfo every 2 s and hard-exits the process (os._exit(3), after printing) when MemAvailable drops
     below MIN_AVAILABLE_GB (default 16; env VSA_MIN_AVAILABLE_GB), whatever the cause.
  3. a wall-clock limit (env VSA_MAX_HOURS, default 6) that exits the process.
Import it first in every training script: `import guard` (it installs itself on import). run.sh also refuses to start a second run.
"""
import os, sys, threading, time

FRACTION = float(os.environ.get("VSA_MEM_FRACTION", "0.5"))
MIN_AVAILABLE_GB = float(os.environ.get("VSA_MIN_AVAILABLE_GB", "16"))
MAX_HOURS = float(os.environ.get("VSA_MAX_HOURS", "6"))

def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"): return int(line.split()[1]) / 1024 ** 2
    return float("inf")

def _watchdog(t_start):
    while True:
        time.sleep(2)
        avail = mem_available_gb()
        if avail < MIN_AVAILABLE_GB:
            print(f"\n[guard] MemAvailable {avail:.1f} GB < {MIN_AVAILABLE_GB} GB: exiting to keep the machine alive", flush=True); os._exit(3)
        if time.time() - t_start > MAX_HOURS * 3600:
            print(f"\n[guard] wall-clock limit {MAX_HOURS} h reached: exiting", flush=True); os._exit(4)

def install():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(FRACTION)
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"[guard] cuda allocator capped at {FRACTION:.0%} of {total:.0f} GB; watchdog: exit if MemAvailable < {MIN_AVAILABLE_GB} GB or after {MAX_HOURS} h", flush=True)
    except Exception as e:
        print(f"[guard] torch cap not set ({e}); watchdog only", flush=True)
    th = threading.Thread(target=_watchdog, args=(time.time(),), daemon=True); th.start()

install()
