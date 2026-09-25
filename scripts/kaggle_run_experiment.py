"""Kaggle orchestrator for the candidate convergence experiment (kernel
bfloat16/dpod-candidates-convergence). Reuses the smoke orchestrator's
poll/fetch helpers by re-pointing its module-level kernel constants.

  python scripts/kaggle_run_experiment.py --run-id kaggle_a --candidates variant_5_1 mdp_branch \
      --updates 150 --batch 8
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, ".")
import kaggle_run_candidates_smoke as sm  # noqa: E402
import build_kaggle_kernel_candidates_smoke as bk  # noqa: E402

KDIR = "kaggle_kernel_candidates_experiment"
sm.KERNEL_ID = "bfloat16/dpod-candidates-convergence"
sm.KERNEL_SLUG = "dpod-candidates-convergence"
sm.KERNEL_DIR = KDIR


def build(args):
    src_lines = bk.build_src_decode_cell()
    script_path = args.script
    script = Path(script_path).read_text(encoding="utf-8")
    if args.script_args:  # generic mode: pass CLI through verbatim (Idea-6 policy study)
        cli = list(args.script_args)
    else:
        cli = ["--out", "/kaggle/working/exp", "--candidates", *args.candidates, "--modes", *args.modes,
               "--updates", str(args.updates), "--batch", str(args.batch), "--T", str(args.T),
               "--d-model", str(args.d_model), "--eval-every", str(args.eval_every), "--seed", str(args.seed)]
    import base64
    extra = ["import base64, os\n"]
    for fp in args.extra_files:
        extra.append(f"os.makedirs({os.path.dirname(fp)!r}, exist_ok=True)\n")
        extra.append(f"open({fp!r},'wb').write(base64.b64decode({base64.b64encode(Path(fp).read_bytes()).decode()!r}))\n")
    cells = [
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import subprocess, sys\n",
            "r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'craftax', 'optax'], capture_output=True, text=True)\n",
            "print(r.stdout[-1500:], r.stderr[-1500:] if r.returncode else '')\n",
            "import jax; print('JAX Backend:', jax.default_backend(), jax.devices(), jax.__version__)\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src_lines},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import os\nos.makedirs('scripts', exist_ok=True)\n",
            *extra,
            f"open({script_path!r},'w',encoding='utf-8').write({script!r})\n",
            f"r = subprocess.run([sys.executable, '-u', {script_path!r}] + {cli!r})\n",
            "print('exit', r.returncode)\n"]},
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    Path(KDIR, "candidates_convergence.ipynb").write_text(json.dumps(nb), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--candidates", nargs="+", default=["transformer_branch"])
    p.add_argument("--script", default="scripts/run_candidate_experiment.py")
    p.add_argument("--script-args", nargs=argparse.REMAINDER, default=[])
    p.add_argument("--extra-files", nargs="*", default=[])
    p.add_argument("--modes", nargs="+", default=["reinforce"])
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=2026)
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
