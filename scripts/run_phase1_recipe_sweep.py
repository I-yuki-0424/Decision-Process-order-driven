"""TASK-20261004-023 (+ the TASK-022 replication checks): reference-recipe sweep with equal budget for every arm, Truck and
GRU-baseline variants, fresh-seed finals. Local GPU via Docker (dpod-local), resumable, NOTHING starts on its own: every
stage is launched by hand, in order, after the operator's go-ahead.

  python scripts/run_phase1_recipe_sweep.py plan                         # every stage's jobs + estimated GPU hours (no GPU use)
  python scripts/run_phase1_recipe_sweep.py env                          # pip freeze + GPU name of the image -> environment.txt
  python scripts/run_phase1_recipe_sweep.py replicate --deadline ...     # S0 determinism (100k) + replication of TASK-017 finals
  python scripts/run_phase1_recipe_sweep.py diag --deadline ...          # S1 Truck + GRU with saved params, then the death diagnostic
  python scripts/run_phase1_recipe_sweep.py tune --deadline ...          # S2a anchors x 3 tuning seeds + random points x seed 1000
  python scripts/run_phase1_recipe_sweep.py tune2 --deadline ...         # S2b top-2 random points per arm -> seeds 1001, 1002
  python scripts/run_phase1_recipe_sweep.py select                       # -> recipe_selection.json (tuning outputs only)
  python scripts/run_phase1_recipe_sweep.py arch --deadline ...          # S3 5 Truck variants + 5 baseline variants x 3 tuning seeds
  python scripts/run_phase1_recipe_sweep.py select-arch                  # -> arch_selection.json
  python scripts/run_phase1_recipe_sweep.py final --deadline ...         # S4 fresh seeds 52-61 (test 434-443) for the selected arms

Rules fixed BEFORE any run (P7/P8/P9; recorded in STATE.yaml TASK-023):
  * S2: every arm sees the same points (fixed RNG): 3 anchors (Dedieu 2025 Table 3 recipe, Moon et al. 2023 recipe, the
    TASK-017/020 default) on 3 tuning seeds (1000-1002) + 6 random points on seed 1000; the 2 best random points of each arm
    (by reward_pct + score_pct on seed 1000) get seeds 1001 and 1002. 19 jobs per arm. Selection = max mean(reward_pct + score_pct)
    over the 3 tuning seeds among points that have all 3.
  * S3: Truck variants at tf_gru's selected point; the SAME number of baseline variants at the best baseline's point; 3 tuning
    seeds each; selection per family by the same metric (the S2 winner of the family competes with its 3 seeds).
  * S4: replicate k = 10..19 of the MLflow seed convention (train 52-61, test 434-443): seeds 42-51 were already used and looked
    at for the TASK-020 finals. Final runs refuse a dirty tree (run_epa_mini.py).
  * S0 replication re-runs the TASK-017 default config on seeds 0-9 (role replication, never a result). Determinism runs repeat
    one seed twice with and without XLA's deterministic GPU ops at 100k steps and compare bit for bit.
Jobs are one (arm, point, seed) each; finished outputs are skipped; a job only starts if it can finish before --deadline.
"""
import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = "output/phase1/recipe_t023"
TUNE_SEEDS = [1000, 1001]   # 10 h profile (TASK-023 amendment 2026-10-05): 2 tuning seeds instead of 3
FINAL_SEEDS = list(range(52, 62))
TEST_SEED_OFFSET = 382
RNG_SEED = 20261004
N_RANDOM, N_PROMOTE = 4, 2   # 10 h profile, extended 2026-10-05 after S2a: 4 random points (first 4 of the registered 6, same RNG), top-2 promoted
DET_FLAGS = "--xla_gpu_deterministic_ops=true"
# arms: name -> (run_epa_mini --arm, arm kwargs, VRAM MiB at 4096-4608 steps/rollout, seconds per 1M steps with the GPU to itself)
ARMS = {
    "ppo_gru": ("ppo_gru", {}, 900, 450),
    "ppo_gru_ln": ("ppo_gru", {"ln": True, "skip": True}, 1000, 500),
    "tf_gru": ("tf_gru", {}, 4800, 950),   # = Truck (Idea4_O01_D01_S00)
}
TRUCK_VARIANTS = {
    "tf_gru_mem": {"mem_token": True}, "tf_gru_pa": {"prev_act": True}, "tf_gru_hs": {"head_skip": True},
    "tf_gru_mem_pa_hs": {"mem_token": True, "prev_act": True, "head_skip": True},
    "tf_gru_hs_nocand": {"head_skip": True, "cand": False},   # attribution control: no candidate-action tokens
}
BASELINE_VARIANTS = {   # same count as TRUCK_VARIANTS (P9)
    "gru256_ln": {"ln": True}, "gru256_skip": {"skip": True}, "gru384_ln_skip": {"width": 384, "ln": True, "skip": True},
    "gru512_ln_skip": {"width": 512, "ln": True, "skip": True}, "gru512": {"width": 512},
}
VARIANT_COST = {"tf_gru": (5200, 1050), "ppo_gru": (1600, 800)}
# Anchors. lr/gamma/lambda/epochs/minibatches/clip/vf/grad-norm/value-norm/advantage standardization from the papers:
#   ref  = Dedieu et al. 2025 (arXiv 2502.01591) Table 3: 48 envs x 96, 4 epochs x 8 minibatches, lr 4.5e-4 annealed,
#          gamma 0.925, lambda 0.625, clip 0.2, TD coef 1.0, entropy 0.01, max grad norm 0.5, target EMA alpha 0.95, GAE standardized
#          across the batch.
#   moon = Moon et al. 2023 (arXiv 2307.03486): 4096 steps per rollout (env count not taken from the paper: 64 x 64), 3 epochs x
#          8 minibatches, lr 3e-4, gamma 0.95, lambda 0.65, clip 0.2, vf 0.5, entropy 0.01, max grad norm 0.5, value EMA 0.99.
#   t00  = the TASK-017 default = TASK-020 trial 0 (continuity point).
ANCHORS = {
    "ref": dict(n=48, t=96, lr=4.5e-4, epochs=4, minibatches=8, ent=0.01, lam=0.625, gamma=0.925, clip=0.2, vf=1.0,
                max_grad_norm=0.5, value_norm=0.95, adv_norm="batch"),
    "moon": dict(n=64, t=64, lr=3e-4, epochs=3, minibatches=8, ent=0.01, lam=0.65, gamma=0.95, clip=0.2, vf=0.5,
                 max_grad_norm=0.5, value_norm=0.99, adv_norm="minibatch"),
}   # t00 (the TASK-017/020 default) is not re-run in the 10 h profile: its numbers already exist (TASK-020)
SHAPES = [(48, 96), (64, 64), (32, 128)]   # 4096-4608 steps per rollout; >= 8192 OOMs tf_gru on the 8 GiB card
VRAM_BUDGET_MIB, GPU_MIB = 6000, 8192


def sample_points():
    rs = np.random.RandomState(RNG_SEED)
    pts = {}
    for i in range(N_RANDOM):
        n, t = SHAPES[rs.randint(len(SHAPES))]
        pts[f"r{i:02d}"] = dict(n=n, t=t, lr=float(np.exp(rs.uniform(np.log(2e-4), np.log(2e-3)))), epochs=int(rs.choice([3, 4])),
                                minibatches=int(rs.choice([4, 8])), ent=float(rs.choice([0.003, 0.01])),
                                lam=float(rs.choice([0.625, 0.65, 0.8])), gamma=float(rs.choice([0.925, 0.95, 0.99])), clip=0.2,
                                vf=float(rs.choice([0.5, 1.0])), max_grad_norm=0.5, value_norm=float(rs.choice([0.95, 0.99])),
                                adv_norm=str(rs.choice(["minibatch", "batch"])))
    return pts


def all_points():
    return {**ANCHORS, **sample_points()}


def cli(arm_key, kwargs, p, seed, role, out, steps, protocol, budget, offset=0, save_params=""):
    c = ["python", "scripts/run_epa_mini.py", "--arm", arm_key, "--arm-kwargs", json.dumps(kwargs), "--seeds", str(seed),
         "--role", role, "--steps", str(steps), "--num-envs", str(p["n"]), "--num-steps", str(p["t"]), "--lr", repr(p["lr"]),
         "--epochs", str(p["epochs"]), "--minibatches", str(p["minibatches"]), "--ent", str(p["ent"]), "--lam", str(p["lam"]),
         "--gamma", str(p["gamma"]), "--clip", str(p["clip"]), "--vf", str(p["vf"]), "--max-grad-norm", str(p["max_grad_norm"]),
         "--value-norm", str(p["value_norm"]), "--adv-norm", p["adv_norm"], "--out", out, "--protocol", protocol,
         "--tuning-budget", budget]
    if offset:
        c += ["--eval-seed-offset", str(offset), "--seed-convention", "mlflow_v11"]
    if save_params:
        c += ["--save-params", save_params]
    return c


def job(name, out, arm_key, kwargs, p, seed, role, protocol, budget, mem, sec, steps=1_000_000, env=None, offset=0, save_params="",
        extra=None):
    return dict(name=name, out=out, seed=seed, mem=mem, sec=sec * steps / 1_000_000 * (p["epochs"] / 4), env=env or {}, extra=extra or [],
                cmd=cli(arm_key, kwargs, p, seed, role, out, steps, protocol, budget, offset, save_params))


TUNE_BUDGET = (f"TASK-023 10 h profile: {len(ANCHORS)} anchors (ref, moon) x {len(TUNE_SEEDS)} tuning seeds + {N_RANDOM} random points x seed 1000 + top-{N_PROMOTE} x seed 1001 = "
               f"{len(ANCHORS) * len(TUNE_SEEDS) + N_RANDOM + N_PROMOTE} jobs per arm at 1M steps, identical points for every arm; "
               f"selection = mean(reward_pct + score_pct) over {len(TUNE_SEEDS)} tuning seeds")


def stage_jobs(stage):
    d = f"{ROOT}/{stage}"
    pts = all_points()
    jobs = []
    if stage == "replicate":
        for arm in ("tf_gru", "ppo_gru"):   # determinism: same seed twice, with and without deterministic GPU ops (100k steps)
            a_key, kw, mem, sec = ARMS[arm]
            for mode, env in (("nondet", {}), ("det", {"XLA_FLAGS": DET_FLAGS})):
                if arm == "ppo_gru" and mode == "det":
                    continue
                for rep in ("a", "b"):
                    jobs.append(job(f"{arm}__determinism_{mode}_{rep}", f"{d}/{arm}__determinism_{mode}_{rep}.json", a_key, kw, ANCHORS["t00"],
                                    1000, "diag", "EP-A-determinism-check", "determinism check, not a tuning trial", mem, sec,
                                    steps=100_000, env=env))
        for s in range(10):   # replication of the TASK-017 1M finals (default config, seeds 0-9, test seed = train seed)
            for arm in ("tf_gru", "ppo_gru"):
                a_key, kw, mem, sec = ARMS[arm]
                jobs.append(job(f"{arm}__replication__s{s}", f"{d}/{arm}__replication__s{s}.json", a_key, kw, ANCHORS["t00"], s,
                                "replication", "EP-A-replication", "replication of TASK-017 final (no tuning)", mem, sec))
    elif stage == "diag":
        for arm in ("tf_gru", "ppo_gru"):
            a_key, kw, mem, sec = ARMS[arm]
            out = f"{d}/{arm}__diag__s1002.json"
            jobs.append(job(f"{arm}__diag__s1002", out, a_key, kw, ANCHORS["t00"], 1002, "diag", "EP-A-diagnostic",
                            "diagnostic run, not a tuning trial", mem, sec, save_params=f"{d}/params",
                            extra=[["python", "scripts/diagnose_phase1_deaths.py", f"{d}/params/{arm}__diag__s1002__s1002.pkl",
                                    "--out", f"{d}/{arm}__s1002_deaths.json"]]))
    elif stage in ("tune", "tune2"):
        promoted = promotions() if stage == "tune2" else {}
        for arm, (a_key, kw, mem, sec) in ARMS.items():
            for pname, p in pts.items():
                if stage == "tune":
                    seeds = TUNE_SEEDS if pname in ANCHORS else TUNE_SEEDS[:1]
                else:
                    seeds = TUNE_SEEDS[1:] if pname in promoted.get(arm, []) else []
                for s in seeds:
                    jobs.append(job(f"{arm}__{pname}__s{s}", f"{ROOT}/tune/{arm}__{pname}__s{s}.json", a_key, kw, p, s, "tune",
                                    "EP-A-tuning", TUNE_BUDGET, mem, sec))
    elif stage == "arch":
        sel = json.load(open(f"{ROOT}/tune/recipe_selection.json"))
        base = sel["best_baseline"]
        for fam, variants, src_arm, cost_key in (("truck", TRUCK_VARIANTS, "tf_gru", "tf_gru"), ("baseline", BASELINE_VARIANTS, base, "ppo_gru")):
            p = sel["arms"][src_arm]["point"]
            a_key, kw0 = ARMS[src_arm][0], ARMS[src_arm][1]
            for vname, kw in variants.items():
                kwv = dict(kw) if fam == "baseline" else {**kw0, **kw}
                mem, sec = VARIANT_COST[cost_key]
                for s in TUNE_SEEDS:
                    jobs.append(job(f"{vname}__s{s}", f"{ROOT}/arch/{vname}__s{s}.json", a_key, kwv, p, s, "tune", "EP-A-tuning",
                                    f"TASK-023 10 h profile variants: {len(TRUCK_VARIANTS)} per family x {len(TUNE_SEEDS)} tuning seeds at the family's selected recipe point", mem, sec))
    elif stage == "final":
        sel = json.load(open(f"{ROOT}/arch/arch_selection.json"))
        for s in FINAL_SEEDS:
            for name, spec in sel["finalists"].items():
                jobs.append(job(f"{name}__s{s}", f"{ROOT}/final/{name}__s{s}.json", spec["arm_key"], spec["arm_kwargs"], spec["point"], s,
                                "final", "EP-A", spec["tuning_budget"], spec["mem"], spec["sec"], offset=TEST_SEED_OFFSET))
    return jobs


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def metric(path):
    d = load(path)
    return None if d is None else d["per_seed"][0]["reward_pct"] + d["per_seed"][0]["score_pct"]


def promotions():
    """Top-N_PROMOTE random points of every arm by seed-1000 reward_pct + score_pct (tuning outputs only)."""
    out = {}
    for arm in ARMS:
        rows = [(metric(f"{ROOT}/tune/{arm}__{p}__s1000.json"), p) for p in sample_points()]
        if any(m is None for m, _ in rows):
            raise SystemExit(f"{arm}: rung-1 tuning not complete")
        out[arm] = [p for _, p in sorted(rows, reverse=True)[:N_PROMOTE]]
    return out


def check_tuning_file(d, path):
    if d["role"] != "tune" or min(d["seeds"]) < 1000:
        raise SystemExit(f"{path}: not a tuning file (P8)")


def select():
    pts, sel = all_points(), {"rule": f"max over points with all {len(TUNE_SEEDS)} tuning seeds of mean(reward_pct + score_pct)", "arms": {}}
    for arm in ARMS:
        rows = []
        for pname, p in pts.items():
            ds = [load(f"{ROOT}/tune/{arm}__{pname}__s{s}.json") for s in TUNE_SEEDS]
            if any(x is None for x in ds):
                continue
            for x, s in zip(ds, TUNE_SEEDS):
                check_tuning_file(x, f"{arm}__{pname}__s{s}")
            r = float(np.mean([x["per_seed"][0]["reward_pct"] for x in ds]))
            sc = float(np.mean([x["per_seed"][0]["score_pct"] for x in ds]))
            rows.append(dict(point_name=pname, point=p, reward_pct=r, score_pct=sc, metric=r + sc, n_seeds=len(TUNE_SEEDS)))
        if not rows:
            raise SystemExit(f"{arm}: no point with {len(TUNE_SEEDS)} tuning seeds")
        rows.sort(key=lambda x: -x["metric"])
        sel["arms"][arm] = dict(rows[0], table=rows)
        print(f"\n{arm}")
        for x in rows:
            print(f"  {x['point_name']:5s} reward={x['reward_pct']:.2f} score={x['score_pct']:.2f} sum={x['metric']:.2f} {x['point']}")
    sel["best_baseline"] = max(("ppo_gru", "ppo_gru_ln"), key=lambda a: sel["arms"][a]["metric"])
    sel["tuning_budget"] = TUNE_BUDGET
    json.dump(sel, open(f"{ROOT}/tune/recipe_selection.json", "w"), indent=1)
    print("\nselected:", {a: v["point_name"] for a, v in sel["arms"].items()}, "best baseline:", sel["best_baseline"])


def select_arch():
    rsel = json.load(open(f"{ROOT}/tune/recipe_selection.json"))
    fin = {}
    for fam, variants, src_arm in (("truck", TRUCK_VARIANTS, "tf_gru"), ("baseline", BASELINE_VARIANTS, rsel["best_baseline"])):
        src = rsel["arms"][src_arm]
        cands = [dict(name=src_arm, arm_key=ARMS[src_arm][0], arm_kwargs=ARMS[src_arm][1], metric=src["metric"],
                      reward_pct=src["reward_pct"], score_pct=src["score_pct"], cost=ARMS[src_arm][2:])]
        for vname, kw in variants.items():
            ds = [load(f"{ROOT}/arch/{vname}__s{s}.json") for s in TUNE_SEEDS]
            if any(x is None for x in ds):
                raise SystemExit(f"{vname}: incomplete")
            for x, s in zip(ds, TUNE_SEEDS):
                check_tuning_file(x, f"{vname}__s{s}")
            r = float(np.mean([x["per_seed"][0]["reward_pct"] for x in ds]))
            sc = float(np.mean([x["per_seed"][0]["score_pct"] for x in ds]))
            kwv = dict(kw) if fam == "baseline" else {**ARMS["tf_gru"][1], **kw}
            cands.append(dict(name=vname, arm_key=ARMS[src_arm][0], arm_kwargs=kwv, metric=r + sc, reward_pct=r, score_pct=sc,
                              cost=VARIANT_COST["tf_gru" if fam == "truck" else "ppo_gru"]))
        cands.sort(key=lambda x: -x["metric"])
        print(f"\n{fam}")
        for c in cands:
            print(f"  {c['name']:18s} reward={c['reward_pct']:.2f} score={c['score_pct']:.2f} sum={c['metric']:.2f}")
        w = cands[0]
        fin[w["name"]] = dict(arm_key=w["arm_key"], arm_kwargs=w["arm_kwargs"], point=src["point"], mem=w["cost"][0], sec=w["cost"][1],
                              family=fam, tuning_budget=TUNE_BUDGET + f"; + {len(TRUCK_VARIANTS)} variants per family x {len(TUNE_SEEDS)} tuning seeds (TASK-023 S3, 10 h profile)")
    json.dump(dict(rule="max mean(reward_pct + score_pct) over the tuning seeds per family", finalists=fin),
              open(f"{ROOT}/arch/arch_selection.json", "w"), indent=1)
    print("\nfinalists:", list(fin))


class Scheduler:
    """VRAM-budgeted, deadline-aware, resumable job runner (same rules as run_phase1_random_search.py, per-job env and costs)."""

    def __init__(self, jobs, workers, deadline, root, prov_env):
        self.jobs, self.workers, self.deadline, self.root, self.prov_env = jobs, workers, deadline, root, prov_env
        self.cv, self.used, self.running = threading.Condition(), 0, 0

    def run(self):
        threads = []
        for j in self.jobs:
            if done(j):
                continue
            need = min(int(j["mem"] * 1.25), VRAM_BUDGET_MIB)   # a job above the budget runs alone
            with self.cv:
                while self.running >= self.workers or self.used + need > VRAM_BUDGET_MIB:
                    self.cv.wait(30)
                if time.time() + j["sec"] * self.workers * 1.3 > self.deadline.timestamp():
                    log(f"deadline: not starting {j['name']} (est {j['sec'] * self.workers:.0f}s)")
                    continue
                self.used += need
                self.running += 1
            t = threading.Thread(target=self.exec, args=(j, need))
            t.start()
            threads.append(t)
            time.sleep(15)   # staggered starts: CUDA init fails when several containers start at once
        for t in threads:
            t.join()

    def docker(self, j, cmd, need, log_path):
        rc = -1
        for attempt in range(3):
            mf = min(0.75, need * (1.0 + 0.4 * attempt) / GPU_MIB)
            env = {**self.prov_env, "JAX_COMPILATION_CACHE_DIR": "/workspace/.jaxcache", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                   "XLA_PYTHON_CLIENT_MEM_FRACTION": f"{mf:.3f}", "PYTHONUNBUFFERED": "1", **j["env"]}
            dcmd = ["docker", "run", "--rm", "--gpus", "all", *[x for k, v in env.items() for x in ("-e", f"{k}={v}")],
                    "-v", f"{os.getcwd()}:/workspace", "-w", "/workspace", "dpod-local:latest", *cmd]
            with open(log_path, "a") as lf:
                rc = subprocess.run(dcmd, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, "MSYS_NO_PATHCONV": "1"}).returncode
            if rc == 0 or time.time() > self.deadline.timestamp():
                break
            time.sleep(20 * (attempt + 1))
        return rc

    def exec(self, j, need):
        os.makedirs(f"{self.root}/logs", exist_ok=True)
        t0, lp = time.time(), f"{self.root}/logs/{j['name']}.log"
        rc = self.docker(j, j["cmd"], need, lp)
        for extra in j["extra"]:
            if rc == 0:
                rc = self.docker(j, extra, need, lp)
        with self.cv:
            self.used -= need
            self.running -= 1
            self.cv.notify_all()
        log(f"{j['name']}: {'ok' if rc == 0 else f'FAIL rc={rc}'} {time.time() - t0:.0f}s"
            + ("" if rc == 0 else f"  -> record it: python scripts/mlflow_ingest.py --record-failure ... (see log {lp})"))


def done(j):
    d = load(j["out"])
    ok = d is not None and d.get("seeds") == [j["seed"]]
    for extra in j["extra"]:
        ok = ok and os.path.exists(extra[extra.index("--out") + 1])
    return ok


def log(msg):
    print(f"{time.strftime('%m-%d %H:%M:%S')} {msg}", flush=True)


def host_provenance():
    """Commit, dirty flag and evaluator commit seen on the host (git inside the container may refuse the mounted repo)."""
    git = lambda *a: subprocess.check_output(["git", *a]).decode().strip()
    dirty = git("status", "--porcelain", "--", "src", "scripts", "tests", "docker", "requirements.txt", "requirements-cuda.txt")
    return {"GIT_COMMIT": git("rev-parse", "HEAD"), "GIT_DIRTY": "1" if dirty else "0",
            "EVALUATOR_COMMIT": git("log", "-1", "--format=%H", "--", "src/environment/craftax_env_adapter.py", "src/pipeline/epa_harness.py")}


def plan():
    total = 0.0
    print(f"anchor and random points (identical for every arm, RNG {RNG_SEED}):")
    for k, p in all_points().items():
        print(f"  {k:5s} {p}")
    for stage in ("tune",):
        js = stage_jobs(stage)
        h = sum(j["sec"] for j in js) / 3600
        total += h
        print(f"{stage:9s}: {len(js):3d} jobs, ~{h:.1f} GPU-h")
    est = {
           "arch": len(TUNE_SEEDS) * (len(TRUCK_VARIANTS) * VARIANT_COST["tf_gru"][1] + len(BASELINE_VARIANTS) * VARIANT_COST["ppo_gru"][1]),
           "final": len(FINAL_SEEDS) * (VARIANT_COST["tf_gru"][1] + VARIANT_COST["ppo_gru"][1])}
    for stage, s in est.items():
        total += s / 3600
        print(f"{stage:9s}: ~{s / 3600:.1f} GPU-h (job list depends on the previous selection)")
    print(f"total ~{total:.0f} GPU-h on the RTX 3060 Ti (one GPU-saturating job at a time; more workers do not add throughput)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["plan", "env", "replicate", "diag", "tune", "tune2", "select", "arch", "select-arch", "final"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--deadline", default="", help="local time, e.g. 2026-10-06T08:00; no job starts that cannot finish before it")
    a = ap.parse_args()
    if a.stage == "plan":
        return plan()
    if a.stage == "select":
        return select()
    if a.stage == "select-arch":
        return select_arch()
    if subprocess.run(["docker", "image", "inspect", "dpod-local:latest"], capture_output=True).returncode != 0:
        raise SystemExit("image dpod-local:latest not found: docker build -t dpod-local:latest -f docker/Dockerfile .   (then run the 'env' stage)")
    if a.stage == "env":
        os.makedirs(ROOT, exist_ok=True)
        freeze = subprocess.check_output(["docker", "run", "--rm", "dpod-local:latest", "python", "-m", "pip", "freeze"]).decode()
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        with open(f"{ROOT}/environment.txt", "w") as f:
            f.write(f"# recorded {dt.datetime.now().isoformat(timespec='seconds')}\n# gpu: {gpu}\n{freeze}")
        return print(f"wrote {ROOT}/environment.txt")
    if not a.deadline:
        raise SystemExit("--deadline is required for stages that start GPU jobs")
    prov = host_provenance()
    if prov["GIT_DIRTY"] == "1" and a.stage == "final":
        raise SystemExit("uncommitted changes under src/ scripts/ tests/ docker/: commit before the final stage")
    jobs = stage_jobs(a.stage)
    root = f"{ROOT}/{a.stage if a.stage not in ('tune2',) else 'tune'}"
    todo = [j for j in jobs if not done(j)]
    log(f"{a.stage}: {len(jobs)} jobs ({len(todo)} to do), {a.workers} workers, commit {prov['GIT_COMMIT'][:8]} dirty={prov['GIT_DIRTY']}")
    Scheduler(jobs, a.workers, dt.datetime.fromisoformat(a.deadline), root, prov).run()


if __name__ == "__main__":
    main()
