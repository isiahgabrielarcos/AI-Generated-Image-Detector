"""
queue_rebuild.py
────────────────
Watcher that waits for the quickgelu re-check to finish, then automatically
launches the full CLIP cache rebuild (equalized training data + per-gen
test data, with the corrected ViT-L-14-quickgelu model).

It polls for the completion marker in recheck_quickgelu_result.txt, so it
costs ~nothing while the re-check runs (the CPU stays on the re-check).
Once the re-check is done, the CPU is free and the rebuild starts.

Output:
  cache_rebuild.log   - full rebuild log (tail this to watch progress)
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RECHECK_RESULT = Path("recheck_quickgelu_result.txt")
MARKER = "<-- this run"          # printed on the final line of the re-check
REBUILD_LOG = Path("cache_rebuild.log")
POLL_SECS = 30


def log(msg):
    print(f"[queue {datetime.now():%H:%M:%S}] {msg}", flush=True)


def recheck_finished() -> bool:
    if not RECHECK_RESULT.exists():
        return False
    try:
        return MARKER in RECHECK_RESULT.read_text(encoding="utf-8",
                                                  errors="ignore")
    except Exception:
        return False


def main():
    log("watcher started; waiting for re-check to complete ...")
    waited = 0
    while not recheck_finished():
        time.sleep(POLL_SECS)
        waited += POLL_SECS
        if waited % 600 == 0:           # heartbeat every 10 min
            log(f"still waiting ({waited//60} min elapsed) ...")

    log("re-check finished. Starting full cache rebuild "
        "(quickgelu + equalized + per-gen).")

    # Limit threads to avoid the CPU thread-contention stall seen earlier.
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "8"
    env["MKL_NUM_THREADS"] = "8"

    t0 = time.time()
    with open(REBUILD_LOG, "w", encoding="utf-8") as f:
        rc = subprocess.run(
            [sys.executable, "cache_clip_features.py", "--overwrite"],
            stdout=f, stderr=subprocess.STDOUT, env=env,
        ).returncode
    mins = (time.time() - t0) / 60
    log(f"cache rebuild finished rc={rc} in {mins:.0f} min. "
        f"See {REBUILD_LOG}.")
    if rc == 0:
        log("Caches ready under datasets_eq/clip_cache. "
            "Next: re-run diagnose probe / start retrain.")


if __name__ == "__main__":
    main()
