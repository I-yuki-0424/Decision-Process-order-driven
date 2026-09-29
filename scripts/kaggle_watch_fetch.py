"""Light Kaggle watcher: poll a kernel's status (no output downloads while it runs), then fetch ONLY result files
(summary/index/result/jsonl/log; never checkpoints, which are GBs for long runs) into output_remote/<run-id>/.
  python scripts/kaggle_watch_fetch.py --slug arc-latency-b --run-id arc_latency_10m_d24
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PATTERN = r"(summary|index|result|greedy1_reference|ar_result)\.json$|\.jsonl$|\.log$"


def kaggle(*args):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout or "") + (r.stderr or "")


def watch(slug, run_id, max_wait_h=11.0, poll_s=60):
    kid = f"bfloat16/{slug}"
    t0 = time.time()
    while time.time() - t0 < max_wait_h * 3600:
        st = kaggle("kernels", "status", kid).lower()
        if "complete" in st or "error" in st or "cancel" in st:
            print(f"[WATCH] {datetime.now():%H:%M:%S} {st.strip()[:120]}", flush=True)
            break
        print(f"[WATCH] {datetime.now():%H:%M:%S} {st.strip()[:100]} elapsed={(time.time()-t0)/60:.0f}min", flush=True)
        time.sleep(poll_s)
    out = Path("output_remote") / run_id
    out.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", kid, "-p", str(out), "--file-pattern", PATTERN, "-o", "-q")[-500:], flush=True)
    print(f"[FETCH] Retrieved {len([f for f in out.rglob('*') if f.is_file()])} files -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-wait-hours", type=float, default=11.0)
    a = ap.parse_args()
    watch(a.slug, a.run_id, a.max_wait_hours)
