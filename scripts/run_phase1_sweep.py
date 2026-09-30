"""Phase-1 sweep driver: runs (arm x config x seeds) jobs in parallel Docker containers on the local GPU, resumable.

  python scripts/run_phase1_sweep.py tune  --steps 100000 --workers 3          # tuning seeds 1000-1002, equal grid for every arm
  python scripts/select_phase1_config.py output/phase1/tune_100k               # -> selection.json (reads tuning outputs only)
  python scripts/run_phase1_sweep.py final --steps 100000 --workers 3          # evaluation seeds 0-9 with the selected configs

Every arm gets the same optimiser/rollout grid (P9): rollout shape {(16 envs x 32 steps), (64 x 64)} x lr {3e-4, 1e-3, 3e-3}.
Architecture sizes are fixed defaults (not tuned). Existing outputs are skipped, so an interrupted sweep can be restarted.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROLLOUTS = [(16, 32), (64, 64)]
LRS = [3e-4, 1e-3, 3e-3]
ARMS = {   # name -> (script, extra args, name of the rollout-length flag)
    "ppo_mlp": ("run_epa_mini", ["--arm", "ppo_mlp"], "--num-steps"),
    "ppo_gru": ("run_epa_mini", ["--arm", "ppo_gru"], "--num-steps"),
    "tf": ("run_epa_mini", ["--arm", "tf"], "--num-steps"),
    "tf_wm": ("run_epa_mini", ["--arm", "tf_wm"], "--num-steps"),
    "tf_wm_random": ("run_epa_mini", ["--arm", "tf_wm_random"], "--num-steps"),
    "tf_sc": ("run_epa_mini", ["--arm", "tf", "--arm-kwargs", '{"critic": "mlp"}'], "--num-steps"),
    "tf_wm_sc": ("run_epa_mini", ["--arm", "tf_wm", "--arm-kwargs", '{"critic": "mlp"}'], "--num-steps"),
    "tf_wm_random_sc": ("run_epa_mini", ["--arm", "tf_wm_random", "--arm-kwargs", '{"critic": "mlp"}'], "--num-steps"),
    "chunkppo_k1": ("run_epa_chunkppo", ["--k", "1", "--hist", "4"], "--cycles"),
    "chunkppo_k4": ("run_epa_chunkppo", ["--k", "4", "--hist", "4"], "--cycles"),   # informational (Phase 3)
}


def cfg_name(n, t, lr):
    return f"n{n}_t{t}_lr{lr:g}"


def build_jobs(a):
    root = f"output/phase1/{a.stage}_{a.steps // 1000}k"
    arms = a.arms.split(",") if a.arms else [x for x in ARMS if x not in ("chunkppo_k4", "tf_sc", "tf_wm_sc", "tf_wm_random_sc")]
    seeds = a.seeds
    jobs, sel = [], None
    if a.stage == "final":
        sel = json.load(open(f"output/phase1/tune_{a.steps // 1000}k/selection.json"))
    for arm in arms:
        script, extra, tflag = ARMS[arm]
        grid = [(n, t, lr) for (n, t) in ROLLOUTS for lr in LRS] if a.stage == "tune" else [tuple(sel[arm]["config"])]
        for n, t, lr in grid:
            k = 4 if arm == "chunkppo_k4" else 1
            t_eff = t // k if script == "run_epa_chunkppo" and k > 1 else t   # keep steps/update equal across k
            name = f"{arm}__{cfg_name(n, t, lr)}"
            out = f"{root}/{name}.json"
            cmd = ["python", f"scripts/{script}.py", *extra, "--seeds", seeds, "--steps", str(a.steps),
                   "--num-envs", str(n), tflag, str(t_eff), "--lr", str(lr), "--role", "tune" if a.stage == "tune" else "final",
                   "--out", out, "--tuning-budget", "6 configs x 3 tuning seeds (seeds 1000-1002), equal grid for every arm; "
                   "selection metric = mean reward_pct on tuning seeds",
                   "--protocol", a.protocol]
            jobs.append((name, out, cmd, root))
    return jobs


def run_job(job, memfrac, commit):
    name, out, cmd, root = job
    if os.path.exists(out) and json.load(open(out)).get("seeds") is not None and job_done(out, cmd):
        return name, "skip", 0.0
    os.makedirs(f"{root}/logs", exist_ok=True)
    t0 = time.time()
    dcmd = ["docker", "run", "--rm", "--gpus", "all", "-e", f"GIT_COMMIT={commit}", "-e", "JAX_COMPILATION_CACHE_DIR=/workspace/.jaxcache",
            "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false", "-e", f"XLA_PYTHON_CLIENT_MEM_FRACTION={memfrac}", "-e", "PYTHONUNBUFFERED=1",
            "-v", f"{os.getcwd()}:/workspace", "-w", "/workspace", "dpod-local:latest", *cmd]
    for attempt in range(4):   # containers occasionally fail CUDA init when several start at once
        with open(f"{root}/logs/{name}.log", "w") as lf:
            rc = subprocess.run(dcmd, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, "MSYS_NO_PATHCONV": "1"}).returncode
        if rc == 0:
            break
        time.sleep(20 * (attempt + 1))
    return name, "ok" if rc == 0 else f"FAIL rc={rc}", time.time() - t0


def job_done(out, cmd):
    want = [int(s) for s in cmd[cmd.index("--seeds") + 1].split(",") if s]
    return json.load(open(out)).get("seeds") == want


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["tune", "final"])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--arms", default="")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--protocol", default="EP-A-mini")
    ap.add_argument("--memfrac", type=float, default=0.25)
    a = ap.parse_args()
    a.seeds = a.seeds or ("1000,1001,1002" if a.stage == "tune" else "0,1,2,3,4,5,6,7,8,9")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    jobs = build_jobs(a)
    print(f"{len(jobs)} jobs, {a.workers} workers, commit {commit[:8]}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for name, status, dt in ex.map(lambda j: run_job(j, a.memfrac, commit), jobs):
            print(f"{time.strftime('%H:%M:%S')} {name}: {status} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
