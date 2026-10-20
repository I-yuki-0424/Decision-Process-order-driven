"""TASK-20261019-026: G1.2 last-mile tuning of the GRU-TP arm on CPU (no GPU / Docker / Kaggle in the cloud session).

  python scripts/run_t026_cpu_sweep.py list
  python scripts/run_t026_cpu_sweep.py run --stage R1 --configs R0,R1,R2 --seeds 4000-4001 --workers 2 --deadline 2026-10-19T23:00
  python scripts/run_t026_cpu_sweep.py report --stage R1
  python scripts/run_t026_cpu_sweep.py run --stage D4M --configs R0 --seeds 4000-4001 --steps 4000000 --protocol diag-4M --workers 2 --deadline ...   # diagnostic, never EP-A

Configs are one-factor changes around G4TP1 (Base_O02_D04_S00) built with the TASK-025 helpers; commands are the TASK-025 commands.
Pre-registration (seeds, selection rule, stop rule): docs/experiments/2026-10-19_phase1_g12_tuning/README.md, written before the runs it covers.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_t025_sweep as t25  # noqa: E402  (base(), command(), parse_seeds(), report())

ROOT = "output/phase1/g12_t026"
t25.ROOT = ROOT
TP = dict(tp_coef=1.0)

CONFIGS = {
    "R0": t25.base(arm=TP),                                  # reference = G4TP1
    "R1": t25.base(arm=TP, lr=2e-3),
    "R2": t25.base(arm=TP, lr=5e-4),
    "R3": t25.base(arm=TP, ent=0.02),
    "R4": t25.base(arm=TP, gamma=0.99, lam=0.8),
    "R5": t25.base(arm=dict(tp_coef=1.0, width=512)),
    "R6": t25.base(arm=TP, n=32, t=128),
    "R7": t25.base(arm=TP, epochs=8),
    # round R3 (amendment 5): more optimiser updates per env step through smaller batches (the data-scaling diagnostic says the gap is sample efficiency)
    "R8": t25.base(arm=TP, n=32),                  # 488 updates of 2048 transitions
    "R9": t25.base(arm=TP, n=16),                  # 976 updates of 1024
    "R10": t25.base(arm=TP, n=32, t=32),           # 976 updates of 1024, shorter BPTT window
}
t25.CONFIGS.update(CONFIGS)


def job(name, seed, stage, role, offset, protocol, budget, steps=1_000_000):
    out = f"{ROOT}/{stage}/{name}__s{seed}.json"
    cmd = t25.command(name, seed, out, role, steps=steps, offset=offset, protocol=protocol, budget=budget)
    cmd[0] = sys.executable
    return dict(name=f"{name}__s{seed}", out=out, cmd=cmd)


def is_done(j):
    try:
        return json.load(open(j["out"])).get("per_seed") is not None
    except (OSError, ValueError):
        return False


def run(jobs, workers, deadline, sec_per_job, prov):
    todo = [j for j in jobs if not is_done(j)]
    print(f"{len(jobs)} jobs, {len(todo)} to do, {workers} workers", flush=True)
    lock, queue = threading.Lock(), list(todo)

    def worker(w):
        cores = ",".join(str(c) for c in range(w * (4 // workers), (w + 1) * (4 // workers)))
        while True:
            with lock:
                if not queue:
                    return
                j = queue.pop(0)
                if time.time() + sec_per_job > deadline.timestamp():
                    print(f"deadline: not starting {j['name']}", flush=True)
                    continue
            os.makedirs(os.path.dirname(j["out"]) + "/logs", exist_ok=True)
            lp = os.path.dirname(j["out"]) + f"/logs/{j['name']}.log"
            t0 = time.time()
            with open(lp, "a") as lf:
                rc = subprocess.run(["taskset", "-c", cores, *j["cmd"]], stdout=lf, stderr=subprocess.STDOUT,
                                    env={**os.environ, **prov, "PYTHONUNBUFFERED": "1"}).returncode
            print(f"{time.strftime('%m-%d %H:%M:%S')} {j['name']}: {'ok' if rc == 0 else f'FAIL rc={rc}'} {time.time() - t0:.0f}s", flush=True)

    ths = [threading.Thread(target=worker, args=(w,)) for w in range(workers)]
    for i, t in enumerate(ths):
        t.start()
        time.sleep(5)
    for t in ths:
        t.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "run", "report", "final"])
    ap.add_argument("--stage", default="R1")
    ap.add_argument("--configs", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--deadline", default="")
    ap.add_argument("--sec-per-job", type=float, default=3600)
    ap.add_argument("--steps", type=int, default=1_000_000, help="training env steps (tune stages only; != 1M is a diagnostic, label it with --protocol)")
    ap.add_argument("--protocol", default="EP-A-tuning")
    a = ap.parse_args()
    if a.cmd == "list":
        for k, v in CONFIGS.items():
            print(k, v["arm_key"], v["arm"], {kk: vv for kk, vv in v["point"].items() if t25.POINT.get(kk) != vv})
        return
    if a.cmd == "report":
        return t25.report(a.stage)
    seeds = t25.parse_seeds(a.seeds)
    git = lambda *x: subprocess.check_output(["git", *x]).decode().strip()
    dirty = git("status", "--porcelain", "--", "src", "scripts", "tests", "docker", "requirements.txt", "requirements-cuda.txt")
    prov = {"GIT_COMMIT": git("rev-parse", "HEAD"), "GIT_DIRTY": "1" if dirty else "0",
            "EVALUATOR_COMMIT": git("log", "-1", "--format=%H", "--", "src/environment/craftax_env_adapter.py", "src/pipeline/epa_harness.py")}
    if a.cmd == "final":
        if prov["GIT_DIRTY"] == "1":
            raise SystemExit("uncommitted changes under src/ scripts/ tests/ docker/: commit before the final stage")
        if max(seeds) >= 1000:
            raise SystemExit("final seeds must be < 1000")
        jobs = [job(n, s, a.stage, "final", t25.TEST_SEED_OFFSET, "EP-A", f"TASK-026 screening on tuning seeds 4000-4999 ({a.configs})")
                for s in seeds for n in a.configs.split(",")]
    else:
        if min(seeds) < 4000:
            raise SystemExit("TASK-026 tuning seeds are 4000-4999 (disjoint from every earlier seed)")
        jobs = [job(n, s, a.stage, "tune", 0, a.protocol, "TASK-026 CPU screening", steps=a.steps) for s in seeds for n in a.configs.split(",")]
    run(jobs, a.workers, dt.datetime.fromisoformat(a.deadline), a.sec_per_job, prov)


if __name__ == "__main__":
    main()
