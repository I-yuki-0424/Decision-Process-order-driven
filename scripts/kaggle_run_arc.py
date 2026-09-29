"""Generic Kaggle launcher for the arc.tex design-proposal experiments (arc-proposals TASK-20260929-016).

Pushes a private GPU kernel named bfloat16/<slug> that embeds src/ plus the listed scripts, then runs several jobs
concurrently, job i pinned to GPU (i % n_gpu) via CUDA_VISIBLE_DEVICES (Kaggle offers 2x T4). Outputs land in
/kaggle/working/job<i>/ and are fetched into output_remote/<run-id>/.

  python scripts/kaggle_run_arc.py --slug arc-ac-litmus --run-id arc_ac_litmus_v1 \
      --scripts scripts/run_actor_critic_litmus.py \
      --job "scripts/run_actor_critic_litmus.py --arms base oracle_act --seeds 0 1 --lr 3e-4" \
      --job "scripts/run_actor_critic_litmus.py --arms base oracle_act --seeds 0 1 --lr 1e-3"
Each --job is 'script args...' (the launcher appends --out /kaggle/working/job<i>). --no-wait pushes and returns.
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, ".")
import build_kaggle_kernel_candidates_smoke as bk  # noqa: E402
import kaggle_run_candidates_smoke as sm  # noqa: E402


def build(args, kdir: Path):
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"bfloat16/{args.slug}", "title": args.slug.replace("-", " ").title(),
        "code_file": "arc_job.ipynb", "language": "python", "kernel_type": "notebook", "is_private": "true",
        "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": []}, indent=2), encoding="utf-8")
    write_files = ["import os\nos.makedirs('scripts', exist_ok=True)\n"]
    for s in args.scripts:
        write_files.append(f"open({s!r},'w',encoding='utf-8').write({Path(s).read_text(encoding='utf-8')!r})\n")
    for extra in args.data:  # small data files (e.g. pickles); base64 to stay binary-safe
        import base64
        write_files.append(f"os.makedirs(os.path.dirname({extra!r}) or '.', exist_ok=True)\n"
                           f"import base64; open({extra!r},'wb').write(base64.b64decode({base64.b64encode(Path(extra).read_bytes()).decode()!r}))\n")
    jobs = []
    for i, j in enumerate(args.job):
        toks = shlex.split(j)
        jobs.append(toks + ["--out", f"/kaggle/working/job{i}"])
    run_cell = [
        "import os, subprocess, sys\n",
        f"jobs = {jobs!r}\n",
        "import jax\n",
        "ng = max(1, len(jax.devices()))\n",
        "procs = []\n",
        "for i, j in enumerate(jobs):\n",
        "    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % ng), XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1')\n",
        "    procs.append(subprocess.Popen([sys.executable, '-u'] + j, env=env, stdout=open(f'/kaggle/working/job{i}.log', 'w'), stderr=subprocess.STDOUT))\n",
        "import time\n",
        "while any(p.poll() is None for p in procs):\n",
        "    time.sleep(60)\n",
        "    for i in range(len(procs)):\n",
        "        os.system(f'tail -n 1 /kaggle/working/job{i}.log | cut -c1-200')\n",
        "print('exit codes', [p.returncode for p in procs])\n",
    ]
    cells = [
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import subprocess, sys\n",
            "r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'craftax', 'optax'], capture_output=True, text=True)\n",
            "print(r.stdout[-1500:], r.stderr[-1500:] if r.returncode else '')\n",
            "import jax; print('JAX Backend:', jax.default_backend(), jax.devices(), jax.__version__)\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": bk.build_src_decode_cell()},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": write_files},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": run_cell},
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    (kdir / "arc_job.ipynb").write_text(json.dumps(nb), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--slug", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--scripts", nargs="+", required=True)
    p.add_argument("--data", nargs="*", default=[])
    p.add_argument("--job", action="append", required=True)
    p.add_argument("--max-wait-hours", type=float, default=6.0)
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--fetch-only", action="store_true")
    a = p.parse_args()
    kdir = Path(f"kaggle_kernel_arc_{a.slug}")
    sm.KERNEL_ID, sm.KERNEL_SLUG, sm.KERNEL_DIR = f"bfloat16/{a.slug}", a.slug, str(kdir)
    if not a.fetch_only:
        build(a, kdir)
        import time
        while True:  # Kaggle allows 2 concurrent GPU sessions: retry until a slot frees up
            out = sm._base.kaggle("kernels", "push", "-p", sm.KERNEL_DIR)
            print(f"[PUSH] {out}", flush=True)
            if "Maximum batch GPU session count" not in out and "error" not in out.lower():
                break
            time.sleep(300)
        if a.no_wait:
            sys.exit(0)
        import kaggle_watch_fetch as kw
        kw.watch(a.slug, a.run_id, a.max_wait_hours)
        sys.exit(0)
    import kaggle_watch_fetch as kw
    kw.watch(a.slug, a.run_id, 0.0)
