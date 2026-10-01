"""TASK-20261001-020: equal-budget random-search tuning at the full 1M-step budget, then the 10-seed EP-A ladder (local GPU, Docker).

  python scripts/run_phase1_random_search.py plan   --trials 12                       # print the sampled points (same for every arm)
  python scripts/run_phase1_random_search.py tune   --trials 12 --deadline 2026-10-02T09:00
  python scripts/run_phase1_random_search.py select                                   # -> selection.json (tuning outputs only)
  python scripts/run_phase1_random_search.py final  --deadline 2026-10-02T11:00

Rules fixed BEFORE any run (P7/P9/P8):
  * every arm sees the same trial points (fixed RNG), 2 tuning seeds per point (1000, 1001), 1M steps, same job order (trial-major),
    so the budget is equal at any cut-off; only trials finished for ALL arms are eligible for selection.
  * selection metric = mean over tuning seeds of (reward_pct + score_pct); evaluation seeds are never read by `select`.
  * final = seeds 42..51 (test seed = train seed + 382 -> 424..433), role=final, protocol EP-A, selected config of each arm.
Jobs are one (arm, trial, seed) each, resumable (finished outputs are skipped). A job is only started if it can finish before the
deadline, so an interruption never loses an in-flight run. Docker VRAM caps are summed against a budget so the card never overflows
into shared system memory (that caused the stutter of 2026-09-30).
"""
import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import threading
import time

import numpy as np

ARMS = {   # name -> run_epa_mini args
    "ppo_gru": ["--arm", "ppo_gru"],
    "cnn_gru": ["--arm", "cnn_gru"],
    "tf_gru": ["--arm", "tf_gru"],   # = Truck (Idea4_O01_D01_S00)
}
SHAPES = [(32, 64), (64, 64)]   # (16,32) is 2x slower; >=128 envs x 64 steps OOMs tf_gru on the 8 GiB card (jit_stack 4 GiB)
TUNE_SEEDS = [1000, 1001]
FINAL_SEEDS = list(range(42, 52))
TEST_SEED_OFFSET = 382
TUNE_DIR = "output/phase1/tune_1000k_t020"
FINAL_DIR = "output/phase1/final_1000k_t020"
RNG_SEED = 20261001
INCUMBENT = dict(n=64, t=64, lr=1e-3, epochs=4, minibatches=4, ent=0.01, lam=0.8, gamma=0.99)   # previous default, trial 0 for every arm
# from the VRAM/throughput probe (STATE.yaml TASK-020 notes): (arm, n_envs) -> MiB of VRAM, arm -> seconds per 1M steps with the GPU to itself
MEM_MIB = {("tf_gru", 64): 4300, ("tf_gru", 32): 2300, ("ppo_gru", 64): 900, ("ppo_gru", 32): 800, ("cnn_gru", 64): 1000, ("cnn_gru", 32): 900}
SEC_1M = {"ppo_gru": 450, "cnn_gru": 600, "tf_gru": 950}
MEM_DEFAULT = 2500
VRAM_BUDGET_MIB = 6000
GPU_MIB = 8192


def sample_points(n_trials):
    rs = np.random.RandomState(RNG_SEED)
    pts = [dict(INCUMBENT)]
    while len(pts) < n_trials:
        n, t = SHAPES[rs.randint(len(SHAPES))]
        rs.choice([4, 8])   # epochs draw kept in the stream but fixed to 4 (cost); keeps points reproducible
        pts.append(dict(n=n, t=t, lr=float(np.exp(rs.uniform(np.log(3e-4), np.log(3e-3)))), epochs=4,
                        minibatches=int(rs.choice([4, 8])), ent=float(rs.choice([0.003, 0.01, 0.02])),
                        lam=float(rs.choice([0.8, 0.95])), gamma=float(rs.choice([0.99, 0.995]))))
    return pts


def est_seconds(arm, p, workers=1):
    # the GPU is saturated by one job (probe: 3 concurrent jobs ~ sequential total), so wall time scales with the number of workers
    return SEC_1M[arm] * (p["epochs"] / 4) * workers


def est_mem(arm, p):
    return MEM_MIB.get((arm, p["n"]), MEM_DEFAULT)


def cli(arm, p, seeds, role, out, steps, protocol, budget, final):
    c = ["python", "scripts/run_epa_mini.py", *ARMS[arm], "--seeds", ",".join(map(str, seeds)), "--role", role, "--steps", str(steps),
         "--num-envs", str(p["n"]), "--num-steps", str(p["t"]), "--lr", repr(p["lr"]), "--epochs", str(p["epochs"]),
         "--minibatches", str(p["minibatches"]), "--ent", str(p["ent"]), "--lam", str(p["lam"]), "--gamma", str(p["gamma"]),
         "--out", out, "--protocol", protocol, "--tuning-budget", budget]
    if final:
        c += ["--eval-seed-offset", str(TEST_SEED_OFFSET), "--seed-convention", "mlflow_v11"]
    return c


def build_jobs(stage, trials, steps):
    budget = (f"random search, {trials} points (trial 0 = previous default) x 2 tuning seeds (1000,1001) at {steps} steps, identical points "
              f"and budget for ppo_gru/cnn_gru/tf_gru; selection = mean(reward_pct+score_pct) over tuning seeds (TASK-20261001-020)")
    jobs = []
    if stage == "tune":
        pts = sample_points(trials)
        for i, p in enumerate(pts):
            for arm in ARMS:
                for s in TUNE_SEEDS:
                    out = f"{TUNE_DIR}/{arm}__t{i:02d}__s{s}.json"
                    jobs.append(dict(name=f"{arm}__t{i:02d}__s{s}", out=out, arm=arm, p=p, seed=s, trial=i,
                                     cmd=cli(arm, p, [s], "tune", out, steps, "EP-A-tuning", budget, False)))
    else:
        sel = json.load(open(f"{TUNE_DIR}/selection.json"))
        for s in FINAL_SEEDS:
            for arm in ARMS:
                p = sel["arms"][arm]["point"]
                out = f"{FINAL_DIR}/{arm}__s{s}.json"
                jobs.append(dict(name=f"{arm}__s{s}", out=out, arm=arm, p=p, seed=s, trial=-1,
                                 cmd=cli(arm, p, [s], "final", out, steps, "EP-A", sel["tuning_budget"], True)))
    return jobs


def done(job):
    try:
        d = json.load(open(job["out"]))
        return d.get("seeds") == [job["seed"]]
    except Exception:
        return False


class Scheduler:
    def __init__(self, jobs, workers, deadline, root, commit):
        self.jobs, self.workers, self.deadline, self.root, self.commit = jobs, workers, deadline, root, commit
        self.cv, self.used, self.running = threading.Condition(), 0, 0
        self.stop = False

    def run(self):
        threads = []
        for j in self.jobs:
            if done(j):
                continue
            need = int(est_mem(j["arm"], j["p"]) * 1.25)
            with self.cv:
                while (self.running >= self.workers or self.used + need > VRAM_BUDGET_MIB) and not self.stop:
                    self.cv.wait(30)
                if time.time() + est_seconds(j["arm"], j["p"], self.workers) * 1.3 > self.deadline.timestamp():
                    self.log(f"deadline: not starting {j['name']} (est {est_seconds(j['arm'], j['p'], self.workers):.0f}s)")
                    continue   # cheaper later jobs may still fit
                self.used += need
                self.running += 1
            t = threading.Thread(target=self.exec, args=(j, need))
            t.start()
            threads.append(t)
            time.sleep(15)   # staggered starts: CUDA init failures when many containers start at once
        for t in threads:
            t.join()

    def log(self, msg):
        print(f"{time.strftime('%m-%d %H:%M:%S')} {msg}", flush=True)

    def exec(self, j, need):
        os.makedirs(f"{self.root}/logs", exist_ok=True)
        t0 = time.time()
        rc = -1
        for attempt in range(3):
            mf = min(0.75, need * (1.0 + 0.4 * attempt) / GPU_MIB)
            dcmd = ["docker", "run", "--rm", "--gpus", "all", "-e", f"GIT_COMMIT={self.commit}", "-e",
                    "JAX_COMPILATION_CACHE_DIR=/workspace/.jaxcache", "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false", "-e",
                    f"XLA_PYTHON_CLIENT_MEM_FRACTION={mf:.3f}", "-e", "PYTHONUNBUFFERED=1", "-v", f"{os.getcwd()}:/workspace", "-w",
                    "/workspace", "dpod-local:latest", *j["cmd"]]
            with open(f"{self.root}/logs/{j['name']}.log", "w") as lf:
                rc = subprocess.run(dcmd, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, "MSYS_NO_PATHCONV": "1"}).returncode
            if rc == 0 or time.time() > self.deadline.timestamp():
                break
            time.sleep(20 * (attempt + 1))
        with self.cv:
            self.used -= need
            self.running -= 1
            self.cv.notify_all()
        self.log(f"{j['name']}: {'ok' if rc == 0 else f'FAIL rc={rc}'} {time.time() - t0:.0f}s")


def select():
    per = {}
    for f in sorted(glob.glob(f"{TUNE_DIR}/*__t*__s*.json")):
        d = json.load(open(f))
        if d["role"] != "tune" or min(d["seeds"]) < 1000:
            raise SystemExit(f"{f}: not a tuning file (P8)")
        arm, trial, _ = os.path.basename(f)[:-5].split("__")
        per.setdefault((arm, int(trial[1:])), []).append(d)
    complete = {k: v for k, v in per.items() if len(v) == len(TUNE_SEEDS)}
    trials = [set(t for (a, t) in complete if a == arm) for arm in ARMS]
    common = sorted(set.intersection(*trials)) if all(trials) else []
    if not common:
        raise SystemExit("no trial is complete for all arms")
    sel = {"rule": "max over common trials of mean_over_tuning_seeds(reward_pct + score_pct)", "common_trials": common, "arms": {}}
    for arm in ARMS:
        rows = []
        for t in common:
            ds = complete[(arm, t)]
            r = float(np.mean([x["per_seed"][0]["reward_pct"] for x in ds]))
            s = float(np.mean([x["per_seed"][0]["score_pct"] for x in ds]))
            c = ds[0]["config"]
            rows.append(dict(trial=t, reward_pct=r, score_pct=s, metric=r + s,
                             point=dict(n=c["num_envs"], t=c["num_steps"], lr=c["lr"], epochs=c["epochs"], minibatches=c["minibatches"],
                                        ent=c["ent"], lam=c["lam"], gamma=c["gamma"])))
        rows.sort(key=lambda x: -x["metric"])
        print(f"\n{arm}  (trials compared: {len(common)})")
        for x in rows:
            print(f"  t{x['trial']:02d} reward={x['reward_pct']:.2f} score={x['score_pct']:.2f} sum={x['metric']:.2f} {x['point']}")
        sel["arms"][arm] = dict(rows[0], n_trials=len(common), table=rows)
    first = next(iter(complete.values()))[0]
    sel["tuning_budget"] = (f"random search, {len(common)} trials x 2 tuning seeds (1000,1001) x {first['env_steps_total']} steps per arm; identical points "
                            f"and budget for ppo_gru/cnn_gru/tf_gru; selection = mean(reward_pct+score_pct) over tuning seeds")
    json.dump(sel, open(f"{TUNE_DIR}/selection.json", "w"), indent=1)
    print("\nselected:", {a: v["point"] for a, v in sel["arms"].items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["plan", "tune", "select", "final"])
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--deadline", default="2026-10-02T11:00", help="local time; no job is started that cannot finish before it")
    a = ap.parse_args()
    if a.stage == "plan":
        for i, p in enumerate(sample_points(a.trials)):
            print(i, p)
        return
    if a.stage == "select":
        return select()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    jobs = build_jobs(a.stage, a.trials, a.steps)
    root = TUNE_DIR if a.stage == "tune" else FINAL_DIR
    todo = [j for j in jobs if not done(j)]
    print(f"{len(jobs)} jobs ({len(todo)} to do), {a.workers} workers, VRAM budget {VRAM_BUDGET_MIB} MiB, commit {commit[:8]}", flush=True)
    Scheduler(jobs, a.workers, dt.datetime.fromisoformat(a.deadline), root, commit).run()


if __name__ == "__main__":
    main()
