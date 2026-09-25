"""
Kaggle Automation Orchestrator for the DPOD Candidate Smoke-Verification Run.

A small, separate sibling of scripts/kaggle_run.py, pointed at the new
bfloat16/dpod-candidates-smoke-verify kernel instead of the existing
production kernel. It reuses scripts/kaggle_run.py's push/poll/fetch helper
functions (log streaming, JSONL parsing, ERROR-vs-COMPLETE handling) by
importing them -- scripts/kaggle_run.py itself is never modified, so the
existing production workflow (and anything that depends on its
KERNEL_ID/OUTPUT_DIR constants) is untouched.

This is a SMALL verification run: order of tens-to-low-hundreds of training
steps per candidate, well under an hour of GPU wall-clock. It is not a Phase
II-scale run and does not need ADR-002 authorization.

Usage:
    python scripts/kaggle_run_candidates_smoke.py \\
        --candidates transformer_branch variant_5_1 mdp_branch \\
        --train-episodes 15 --eval-episodes 5 --max-steps-per-ep 15 --d-model 32
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Reuse scripts/kaggle_run.py's helpers without modifying that file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_run as _base  # noqa: E402

KERNEL_ID = "bfloat16/dpod-candidates-smoke-verify"
KERNEL_SLUG = "dpod-candidates-smoke-verify"
KERNEL_DIR = "kaggle_kernel_candidates_smoke"
NOTEBOOK_SRC = f"{KERNEL_DIR}/candidates_smoke_verify.ipynb"
OUTPUT_DIR = Path("output_remote")
RUN_TAG = "candidates_smoke"


def build_notebook(cfg: dict) -> None:
    print(f"\n[BUILD] Generating candidate smoke-verify notebook (candidates={cfg['candidates']})...")
    cmd = [
        sys.executable, "build_kaggle_kernel_candidates_smoke.py",
        "--candidates", *cfg["candidates"],
        "--train-episodes", str(cfg["train_episodes"]),
        "--eval-episodes", str(cfg["eval_episodes"]),
        "--max-steps-per-ep", str(cfg["max_steps_per_ep"]),
        "--d-model", str(cfg["d_model"]),
        "--checkpoint-every", str(cfg["checkpoint_every"]),
        "--seed", str(cfg["seed"]),
    ]
    _base.run_cmd(cmd)
    print(f"[BUILD] Notebook ready: {NOTEBOOK_SRC}")


def push_kernel() -> None:
    print(f"\n[PUSH] Pushing kernel: {KERNEL_ID}")
    out = _base.kaggle("kernels", "push", "-p", KERNEL_DIR)
    print(f"[PUSH] {out}")


def poll_with_live_logs(log_dir: Path, max_wait_s: int = 2 * 3600) -> bool:
    """Same JSONL log-streaming / ERROR-vs-COMPLETE logic as
    scripts/kaggle_run.py's poll_with_live_logs, but against KERNEL_ID above.
    A short max_wait_s (2h default) matches this being a small verification
    run, not a multi-hour Phase II run."""
    print(f"\n[POLL] Monitoring kernel: {KERNEL_ID}")
    print(f"       Log sync every {_base.LOG_INTERVAL}s | Status check every {_base.POLL_INTERVAL}s")
    print(f"       Max wait: {max_wait_s / 3600:.1f}h\n")

    log_dir.mkdir(parents=True, exist_ok=True)
    import time
    start = time.time()
    last_log_sync = 0.0
    last_printed = 0
    poll_count = 0

    def fetch_log():
        try:
            _base.run_cmd(["kaggle", "kernels", "output", KERNEL_ID,
                            "-p", str(log_dir), "--force"], check=False, capture=True)
        except Exception:
            pass
        return log_dir / f"{KERNEL_SLUG}.log"

    while time.time() - start < max_wait_s:
        now = time.time()
        if now - last_log_sync >= _base.LOG_INTERVAL:
            log_path = fetch_log()
            entries = _base._parse_log_jsonl(log_path)
            last_printed = _base._print_new_log_entries(entries, last_printed)
            last_log_sync = now

        raw = _base.kaggle("kernels", "status", KERNEL_ID, check=False)
        status = raw.lower()
        poll_count += 1
        elapsed = now - start
        if poll_count % 4 == 0:
            print(f"[POLL] {datetime.now().strftime('%H:%M:%S')} "
                  f"status={status.strip()!r} elapsed={elapsed / 60:.1f}min", flush=True)

        if "error" in status:
            time.sleep(5)
            log_path = fetch_log()
            entries = _base._parse_log_jsonl(log_path)
            last_printed = _base._print_new_log_entries(entries, last_printed)
            print("\n[POLL] ===== KERNEL FAILED - full error log =====")
            for e in entries:
                if e.get("stream_name") == "stderr":
                    _base._safe_print(e.get("data", "").rstrip())
            print("[POLL] ================================================\n")
            return False

        if "complete" in status:
            time.sleep(3)
            log_path = fetch_log()
            entries = _base._parse_log_jsonl(log_path)
            _base._print_new_log_entries(entries, last_printed)
            elapsed = time.time() - start
            print(f"\n[POLL] COMPLETE in {elapsed / 60:.1f} min.", flush=True)
            return True

        time.sleep(_base.POLL_INTERVAL)

    print(f"\n[POLL] WARNING: Timeout after {max_wait_s / 3600:.1f}h.", flush=True)
    return False


def fetch_outputs(run_id: str) -> Path:
    out_dir = OUTPUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[FETCH] Downloading outputs -> {out_dir}/")
    _base.run_cmd(["kaggle", "kernels", "output", KERNEL_ID,
                    "-p", str(out_dir), "--force"], check=False)
    files = [f for f in out_dir.rglob("*") if f.is_file()]
    print(f"[FETCH] Retrieved {len(files)} files:")
    for f in sorted(files):
        size_kb = f.stat().st_size / 1024
        print(f"        {f.relative_to(out_dir)}  ({size_kb:.1f} KB)")
    if not files:
        print("[FETCH] WARNING: 0 files. Kernel may still be running or had no outputs.")
    return out_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="DPOD Candidate Smoke-Verification Kaggle Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--candidates", nargs="+",
                    default=["transformer_branch", "variant_5_1", "mdp_branch"])
    p.add_argument("--train-episodes", type=int, default=15)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--max-steps-per-ep", type=int, default=15)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--run-id", default=None)
    p.add_argument("--fetch-only", action="store_true")
    p.add_argument("--max-wait-hours", type=float, default=1.0)
    return p.parse_args(argv)


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    args = parse_args()
    run_id = args.run_id or f"candidates_smoke_{_ts()}"
    cfg = {
        "candidates": args.candidates,
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "max_steps_per_ep": args.max_steps_per_ep,
        "d_model": args.d_model,
        "checkpoint_every": args.checkpoint_every,
        "seed": args.seed,
    }
    log_dir = OUTPUT_DIR / run_id

    if not args.fetch_only:
        build_notebook(cfg)
        push_kernel()
        success = poll_with_live_logs(log_dir, max_wait_s=int(args.max_wait_hours * 3600))
        if not success:
            print("[RUN] Kernel errored. Fetching outputs for post-mortem.")

    out_dir = fetch_outputs(run_id)

    results_files = list(out_dir.rglob("candidate_smoke_summary.json"))
    if results_files:
        with open(results_files[0], encoding="utf-8") as f:
            data = json.load(f)
        print(f"\n{'=' * 60}")
        print(f"  RESULTS: {run_id}")
        print(f"{'=' * 60}")
        print(f"  Backend: {data.get('hardware', {}).get('jax_backend', '?').upper()}")
        for r in data.get("results", []):
            print(f"  [{r['model_name']}] crafter_score={r['crafter_score']:.4f} "
                  f"avg_unlocked={r['avg_unlocked_count']:.2f} avg_steps={r['avg_steps']:.1f}")
        print(f"{'=' * 60}\n")
    else:
        print(f"[RESULTS] WARNING: no candidate_smoke_summary.json found under {out_dir}")

    print(f"\n[DONE] {run_id}")
    print(f"  Outputs : {out_dir}/")
