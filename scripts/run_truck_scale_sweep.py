"""TASK-20261005-024: Truck seed-variance / parameter-scale study. Local GPU via Docker (dpod-local), resumable.

  python scripts/run_truck_scale_sweep.py list                                        # configs, parameter counts are in the report
  python scripts/run_truck_scale_sweep.py run --stage A --configs T0,lr7e-4 --seeds 2000-2005 --deadline 2026-10-06T08:00
  python scripts/run_truck_scale_sweep.py report --stage A                            # mean / sd / seed table per config
  python scripts/run_truck_scale_sweep.py final --config NAME --deadline ...          # EP-A finals, seeds 62-71 / test 444-453

Config registry below: every config is the Truck-improved (tfgru + mem + pa + hs, TASK-023 winner) with named deviations. All shared
rules (tuning seeds >= 1000, final seeds 62-71 never used before, one config per name) are fixed in
docs/experiments/2026-10-05_truck_scale_study/README.md BEFORE the runs they cover.
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_phase1_recipe_sweep as rs  # noqa: E402  (Scheduler, done, host_provenance, log)

ROOT = "output/phase1/truck_scale_t024"
FINAL_SEEDS = list(range(62, 72))
TEST_SEED_OFFSET = 382
rs.VRAM_BUDGET_MIB = 7000

T0_POINT = dict(n=64, t=64, lr=0.0014954284394328645, epochs=3, minibatches=8, ent=0.003, lam=0.625, gamma=0.99, clip=0.2, vf=1.0,
                max_grad_norm=0.5, value_norm=0.99, adv_norm="batch", warmup=0.0, remat=False)
T0_ARM = dict(mem_token=True, prev_act=True, head_skip=True)   # d=64, layers=2, heads=4, width=256 (0.467M parameters)


def cfg(arm=None, **point):
    return dict(arm={**T0_ARM, **(arm or {})}, point={**T0_POINT, **point})


CONFIGS = {
    "T0": cfg(),
    # --- A: stability factors at the TASK-023 size ---------------------------------------------------------------------------------
    "lr7e-4": cfg(lr=7e-4),
    "lr3e-4": cfg(lr=3e-4),
    "warm5": cfg(warmup=0.05),
    "ent01": cfg(ent=0.01),
    "env128": cfg(n=128, remat=True),                      # 16 envs per minibatch (T0: 8), same 8 minibatches, half the updates
    "env256": cfg(n=256, remat=True),                      # 32 envs per minibatch
    # --- B: hidden-dimension scale at the T0 recipe --------------------------------------------------------------------------------
    "d32w128": cfg(arm=dict(d=32, width=128)),
    "d32w256": cfg(arm=dict(d=32, width=256)),
    "d64w128": cfg(arm=dict(width=128)),
    "d64w512": cfg(arm=dict(width=512)),
    "d96w256": cfg(arm=dict(d=96)),
    "d128w256": cfg(remat=True, arm=dict(d=128)),
    "d128w512": cfg(remat=True, arm=dict(d=128, width=512)),
    "d192w384": cfg(remat=True, arm=dict(d=192, heads=6, width=384)),
    "d256w512": cfg(remat=True, arm=dict(d=256, heads=8, width=512)),
    # --- C: combinations chosen from the (incomplete) stage AB, see README amendment ----------------------------------------------
    "C1": cfg(warmup=0.05, ent=0.01),
    "C2": cfg(warmup=0.05, ent=0.01, arm=dict(width=128)),
    "C3": cfg(warmup=0.05, ent=0.01, lr=7e-4, remat=True, arm=dict(d=128)),
    "C4": cfg(warmup=0.05, ent=0.01, arm=dict(d=32, width=128)),
    # --- D: recipe refinement around C2 (d 64, GRU width 128, warm-up 5 %, entropy 0.01) -------------------------------------------
    "D1": cfg(warmup=0.15, ent=0.01, arm=dict(width=128)),
    "D2": cfg(warmup=0.05, ent=0.02, arm=dict(width=128)),
    "D3": cfg(warmup=0.05, ent=0.01, lr=1e-3, arm=dict(width=128)),
    "D4": cfg(warmup=0.05, ent=0.01, gamma=0.97, arm=dict(width=128)),
    "D5": cfg(warmup=0.05, ent=0.01, epochs=2, arm=dict(width=128)),
    # --- E: combinations around D3 (lr 1e-3) ---------------------------------------------------------------------------------------------
    "E1": cfg(warmup=0.05, ent=0.01, lr=1e-3, gamma=0.97, arm=dict(width=128)),
    "E2": cfg(warmup=0.05, ent=0.01, lr=1e-3, epochs=4, arm=dict(width=128)),
    "E3": cfg(warmup=0.05, ent=0.01, lr=1e-3, gamma=0.97, epochs=4, arm=dict(width=128)),
    "E4": cfg(warmup=0.05, ent=0.01, lr=7e-4, arm=dict(width=128)),
}


def register(name, **kw):
    CONFIGS[name] = kw


def cost(name):
    """(VRAM MiB cap, seconds per 1M steps with the GPU shared) -- rough, only used for scheduling."""
    a = CONFIGS[name]["arm"]
    d, w = a.get("d", 64), a.get("width", 256)
    scale = (d / 64) * 0.6 + 0.4 + (w / 256 - 1) * 0.2
    return 4400, int(500 * max(scale, 0.7))   # one job at a time (workers=1): 2 concurrent jobs ran 2.5x slower each in stage AB


def command(name, seed, out, role, steps=1_000_000, offset=0, protocol="EP-A-tuning", budget="TASK-024 screening, 6 tuning seeds per config"):
    c, p = CONFIGS[name], CONFIGS[name]["point"]
    cmd = ["python", "scripts/run_epa_mini.py", "--arm", "tf_gru", "--arm-kwargs", json.dumps(c["arm"]), "--seeds", str(seed), "--role", role,
           "--steps", str(steps), "--num-envs", str(p["n"]), "--num-steps", str(p["t"]), "--lr", repr(p["lr"]), "--epochs", str(p["epochs"]),
           "--minibatches", str(p["minibatches"]), "--ent", str(p["ent"]), "--lam", str(p["lam"]), "--gamma", str(p["gamma"]),
           "--clip", str(p["clip"]), "--vf", str(p["vf"]), "--max-grad-norm", str(p["max_grad_norm"]), "--value-norm", str(p["value_norm"]),
           "--adv-norm", p["adv_norm"], "--out", out, "--protocol", protocol, "--tuning-budget", budget]
    if p["warmup"]:
        cmd += ["--warmup", str(p["warmup"])]
    if p["remat"]:
        cmd += ["--remat"]
    if offset:
        cmd += ["--eval-seed-offset", str(offset), "--seed-convention", "mlflow_v11"]
    return cmd


def jobs_for(stage, names, seeds, role="tune", offset=0, protocol="EP-A-tuning", budget="TASK-024 screening, 6 tuning seeds per config"):
    jobs = []
    for s in seeds:   # seed-major: every config advances together
        for n in names:
            mem, sec = cost(n)
            out = f"{ROOT}/{stage}/{n}__s{s}.json"
            jobs.append(dict(name=f"{n}__s{s}", out=out, seed=s, mem=mem, sec=sec, env={}, extra=[],
                             cmd=command(n, s, out, role, offset=offset, protocol=protocol, budget=budget)))
    return jobs


def parse_seeds(txt):
    a, _, b = txt.partition("-")
    return list(range(int(a), int(b) + 1)) if b else [int(a)]


def report(stage):
    rows = {}
    for f in sorted(glob.glob(f"{ROOT}/{stage}/*.json")):
        d = json.load(open(f))
        n = os.path.basename(f).split("__s")[0]
        r = d["per_seed"][0]
        rows.setdefault(n, []).append((d["seeds"][0], r["reward_pct"], r["score_pct"], d["params_total"], r.get("train_diag", {}), r["train_curve_return"]))
    print(f"{'config':10s} {'n':>2s} {'params':>8s} {'reward':>13s} {'sd':>5s} {'score':>12s} {'sd':>5s} {'r+s':>6s} | per-seed reward | clip(early/late) kl(early) gnorm ent(early/late)")
    for n, v in rows.items():
        rw, sc = np.array([x[1] for x in v]), np.array([x[2] for x in v])
        se = lambda a: a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
        dg = [x[4] for x in v if x[4]]
        dm = lambda k, i: float(np.mean([g[k][i] for g in dg])) if dg else float("nan")
        print(f"{n:10s} {len(v):2d} {v[0][3]:8d} {rw.mean():6.2f}+-{se(rw):5.2f} {rw.std(ddof=1) if len(rw) > 1 else float('nan'):5.2f} "
              f"{sc.mean():5.2f}+-{se(sc):4.2f} {sc.std(ddof=1) if len(sc) > 1 else float('nan'):5.2f} {rw.mean() + sc.mean():6.2f} | "
              f"{np.round(rw, 0).astype(int).tolist()} | clip {dm('clip_frac', 1):.2f}/{dm('clip_frac', 9):.2f} kl {dm('approx_kl', 1):.3f} "
              f"gn {dm('grad_norm', 5):.2f} ent {dm('entropy', 1):.2f}/{dm('entropy', 9):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "run", "report", "final"])
    ap.add_argument("--stage", default="A")
    ap.add_argument("--configs", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--seeds", default="2000-2005")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--deadline", default="")
    ap.add_argument("--extra-configs", default="", help="JSON file with additional configs {name: {arm: {...}, point: {...}}}")
    a = ap.parse_args()
    if a.extra_configs:
        for k, v in json.load(open(a.extra_configs)).items():
            CONFIGS[k] = cfg(arm=v.get("arm"), **v.get("point", {}))
    if a.cmd == "list":
        for k, v in CONFIGS.items():
            print(k, v["arm"], {kk: vv for kk, vv in v["point"].items() if T0_POINT[kk] != vv})
        return
    if a.cmd == "report":
        return report(a.stage)
    if not a.deadline:
        raise SystemExit("--deadline is required")
    prov = rs.host_provenance()
    if a.cmd == "final":
        if prov["GIT_DIRTY"] == "1":
            raise SystemExit("uncommitted changes under src/ scripts/ tests/ docker/: commit before the final stage")
        jobs = jobs_for("final", [a.config], FINAL_SEEDS, role="final", offset=TEST_SEED_OFFSET, protocol="EP-A",
                        budget=f"TASK-024: config chosen from screening on tuning seeds 2000-2999 ({a.config})")
        root = f"{ROOT}/final"
    else:
        jobs = jobs_for(a.stage, a.configs.split(","), parse_seeds(a.seeds))
        root = f"{ROOT}/{a.stage}"
    todo = [j for j in jobs if not rs.done(j)]
    rs.log(f"{a.cmd} {a.stage}: {len(jobs)} jobs ({len(todo)} to do), {a.workers} workers, commit {prov['GIT_COMMIT'][:8]} dirty={prov['GIT_DIRTY']}")
    os.makedirs(root, exist_ok=True)
    rs.Scheduler(jobs, a.workers, dt.datetime.fromisoformat(a.deadline), root, prov).run()


if __name__ == "__main__":
    main()
