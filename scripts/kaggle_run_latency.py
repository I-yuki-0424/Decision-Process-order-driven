"""Kaggle orchestrator for the latency/goal-chunking experiment (TASK-20260928-013), kernel
bfloat16/dpod-latency-chunking-calibration. Reuses kaggle_run_candidates_smoke.py's push/poll/fetch helpers by
re-pointing its module-level kernel constants at a NEW, separate kernel, same pattern as kaggle_run_experiment.py.

  python scripts/kaggle_run_latency.py --run-id latency_calibration_v1 --script-args --calibrate --seeds 0,1
"""
import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, ".")
import kaggle_run_candidates_smoke as sm  # noqa: E402
import build_kaggle_kernel_candidates_smoke as bk  # noqa: E402

KDIR = "kaggle_kernel_latency_chunking"
sm.KERNEL_ID = "bfloat16/dpod-latency-chunking-calibration"
sm.KERNEL_SLUG = "dpod-latency-chunking-calibration"
sm.KERNEL_DIR = KDIR
sm.NOTEBOOK_SRC = f"{KDIR}/latency_calibration.ipynb"


def build(args):
    src_lines = bk.build_src_decode_cell()
    script_path = args.script
    script = Path(script_path).read_text(encoding="utf-8")
    cli = list(args.script_args) + ["--out", "/kaggle/working/exp"]
    cells = [
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import subprocess, sys\n",
            "r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'craftax', 'optax'], capture_output=True, text=True)\n",
            "print(r.stdout[-1500:], r.stderr[-1500:] if r.returncode else '')\n",
            "import jax; print('JAX Backend:', jax.default_backend(), jax.devices(), jax.__version__)\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src_lines},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import os, subprocess, sys\nos.makedirs('scripts', exist_ok=True)\n",
            f"open({script_path!r},'w',encoding='utf-8').write({script!r})\n",
            f"r = subprocess.run([sys.executable, '-u', {script_path!r}] + {cli!r})\n",
            "print('exit', r.returncode)\n"]},
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    Path(KDIR, "latency_calibration.ipynb").write_text(json.dumps(nb), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--script", default="scripts/run_latency_experiment.py")
    p.add_argument("--script-args", nargs=argparse.REMAINDER, default=[])
    p.add_argument("--max-wait-hours", type=float, default=4.0)
    p.add_argument("--fetch-only", action="store_true")
    a = p.parse_args()
    log_dir = sm.OUTPUT_DIR / a.run_id
    if not a.fetch_only:
        build(a)
        sm.push_kernel()
        ok = sm.poll_with_live_logs(log_dir, max_wait_s=int(a.max_wait_hours * 3600))
        print("[RUN] kernel ok" if ok else "[RUN] kernel failed/timeout; fetching anyway")
    sm.fetch_outputs(a.run_id)
