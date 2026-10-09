"""TASK-20261008-025: Phase-1 push (Truck + generic sample-efficiency methods, applied to the baseline as well). Local GPU via Docker, resumable.

  python scripts/run_t025_sweep.py list
  python scripts/run_t025_sweep.py run --stage K --configs E1,K1 --seeds 3000-3003 --deadline 2026-10-09T08:00
  python scripts/run_t025_sweep.py report --stage K
  python scripts/run_t025_sweep.py final --config NAME --seeds 82-91 --deadline ...     # EP-A finals, test seed = seed + 382

Every config names its arm (tf_gru = Truck, ppo_gru = baseline) and its deviations from the E1 / G4 recipe point of TASK-024.
Pre-registration (seeds, selection rule, final blocks): docs/experiments/2026-10-08_phase1_push/README.md, written before the runs it covers.
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

ROOT = "output/phase1/push_t025"
TEST_SEED_OFFSET = 382
rs.VRAM_BUDGET_MIB = 7000

# TASK-024 recipe point shared by E1 (Truck) and G4 (baseline); G4 used 4 epochs (its own tuned value), E1 3.
POINT = dict(n=64, t=64, lr=1e-3, epochs=3, minibatches=8, ent=0.01, lam=0.625, gamma=0.97, clip=0.2, vf=1.0,
             max_grad_norm=0.5, value_norm=0.99, adv_norm="batch", warmup=0.05, remat=False, ent_final=-1.0)
TRUCK = dict(mem_token=True, prev_act=True, head_skip=True, width=128)        # E1: 0.284M
BASE = dict(ln=True, skip=True)                                               # G4: gru256+ln+skip, 1.01M


def truck(arm=None, **point):
    return dict(arm_key="tf_gru", arm={**TRUCK, **(arm or {})}, point={**POINT, **point})


def base(arm=None, **point):
    return dict(arm_key="ppo_gru", arm={**BASE, **(arm or {})}, point={**POINT, "epochs": 4, **point})


CONFIGS = {
    "E1": truck(),
    "G4": base(),
    # --- K: more PPO updates per env step (same 1M steps; TASK-024 found larger batches worse) ---------------------------------------
    "K1": truck(t=32),                 # 64 envs x 32 steps: 488 updates instead of 244
    "K2": truck(n=32),                 # 32 envs x 64 steps: 488 updates, 4 envs per minibatch
    # --- TP: transition-prediction auxiliary (candidate token of the executed action predicts S_{t+1} - S_t and r_t) ----------------
    "TP1": truck(arm=dict(tp_coef=1.0)),
    "TP03": truck(arm=dict(tp_coef=0.3)),
    # --- EA: entropy coefficient annealed 0.01 -> 0 over training ----------------------------------------------------------------------
    "EA": truck(ent_final=0.0),
    # --- EV: next-reward-event contrastive auxiliary (intra-trajectory, Moon et al. 2023 style, no achievement memory) ----------------
    "EV03": truck(arm=dict(ev_coef=0.3)),
    "EV1": truck(arm=dict(ev_coef=1.0)),
    "TP1h": truck(arm=dict(tp_coef=1.0, tp_src="head")),   # control: TP through an MLP head instead of the candidate tokens
    # --- baseline with the same method changes (P9) ------------------------------------------------------------------------------------
    "G4TP1": base(arm=dict(tp_coef=1.0)),
    "G4TP03": base(arm=dict(tp_coef=0.3)),
    "G4EA": base(ent_final=0.0),
    "G4EV03": base(arm=dict(ev_coef=0.3)),
    "G4EV1": base(arm=dict(ev_coef=1.0)),
    "G4K1": base(t=32),
    "G4K2": base(n=32),
}


def command(name, seed, out, role, steps=1_000_000, offset=0, protocol="EP-A-tuning", budget="TASK-025 screening"):
    c, p = CONFIGS[name], CONFIGS[name]["point"]
    cmd = ["python", "scripts/run_epa_mini.py", "--arm", c["arm_key"], "--arm-kwargs", json.dumps(c["arm"]), "--seeds", str(seed), "--role", role,
           "--steps", str(steps), "--num-envs", str(p["n"]), "--num-steps", str(p["t"]), "--lr", repr(p["lr"]), "--epochs", str(p["epochs"]),
           "--minibatches", str(p["minibatches"]), "--ent", str(p["ent"]), "--lam", str(p["lam"]), "--gamma", str(p["gamma"]),
           "--clip", str(p["clip"]), "--vf", str(p["vf"]), "--max-grad-norm", str(p["max_grad_norm"]), "--value-norm", str(p["value_norm"]),
           "--adv-norm", p["adv_norm"], "--out", out, "--protocol", protocol, "--tuning-budget", budget]
    if p["warmup"]:
        cmd += ["--warmup", str(p["warmup"])]
    if p["remat"]:
        cmd += ["--remat"]
    if p["ent_final"] >= 0:
        cmd += ["--ent-final", str(p["ent_final"])]
    if offset:
        cmd += ["--eval-seed-offset", str(offset), "--seed-convention", "mlflow_v11"]
    return cmd


def jobs_for(stage, names, seeds, role="tune", offset=0, protocol="EP-A-tuning", budget="TASK-025 screening"):
    jobs = []
    for s in seeds:   # seed-major: every config advances together
        for n in names:
            out = f"{ROOT}/{stage}/{n}__s{s}.json"
            sec = 260 if CONFIGS[n]["arm_key"] == "ppo_gru" else 520
            jobs.append(dict(name=f"{n}__s{s}", out=out, seed=s, mem=4400, sec=sec, env={}, extra=[],
                             cmd=command(n, s, out, role, offset=offset, protocol=protocol, budget=budget)))
    return jobs


def parse_seeds(txt):
    out = []
    for part in txt.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def load_stage(stage):
    rows = {}
    for f in sorted(glob.glob(f"{ROOT}/{stage}/*.json")):
        d = json.load(open(f))
        n = os.path.basename(f).split("__s")[0]
        dev = d.get("provenance", {}).get("device", "?")
        n += "@kag" if "T4" in dev or "P100" in dev else ""   # platform tag: configs are compared on the same platform only
        r = d["per_seed"][0]
        rows.setdefault(n, []).append(dict(seed=d["seeds"][0], reward=r["reward_pct"], score=r["score_pct"], params=d["params_total"],
                                           diag=r.get("train_diag", {}), curve=r["train_curve_return"], length=r["eval_mean_length"],
                                           secs=r["train_seconds"], rates=r["achievement_rates_pct"]))
    return rows


def report(stage):
    rows = load_stage(stage)
    print(f"{'config':10s} {'n':>2s} {'params':>8s} {'reward':>13s} {'sd':>5s} {'score':>12s} {'sd':>5s} {'r+s':>6s} {'len':>5s} {'train_s':>7s} | per-seed reward")
    for n, v in rows.items():
        rw, sc = np.array([x["reward"] for x in v]), np.array([x["score"] for x in v])
        se = lambda a: a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
        sd = lambda a: a.std(ddof=1) if len(a) > 1 else float("nan")
        print(f"{n:10s} {len(v):2d} {v[0]['params']:8d} {rw.mean():6.2f}+-{se(rw):5.2f} {sd(rw):5.2f} {sc.mean():5.2f}+-{se(sc):4.2f} {sd(sc):5.2f} "
              f"{rw.mean() + sc.mean():6.2f} {np.mean([x['length'] for x in v]):5.0f} {np.mean([x['secs'] for x in v]):7.0f} | "
              f"{dict((x['seed'], round(x['reward'], 1)) for x in v)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "run", "report", "final"])
    ap.add_argument("--stage", default="K")
    ap.add_argument("--configs", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--deadline", default="")
    ap.add_argument("--final-dir", default="final")
    a = ap.parse_args()
    if a.cmd == "list":
        for k, v in CONFIGS.items():
            print(k, v["arm_key"], v["arm"], {kk: vv for kk, vv in v["point"].items() if POINT.get(kk) != vv})
        return
    if a.cmd == "report":
        return report(a.stage)
    if not a.deadline or not a.seeds:
        raise SystemExit("--deadline and --seeds are required")
    prov = rs.host_provenance()
    if a.cmd == "final":
        if prov["GIT_DIRTY"] == "1":
            raise SystemExit("uncommitted changes under src/ scripts/ tests/ docker/: commit before the final stage")
        seeds = parse_seeds(a.seeds)
        if max(seeds) >= 1000:
            raise SystemExit("final seeds must be < 1000")
        names = a.config.split(",")
        jobs = jobs_for(a.final_dir, names, seeds, role="final", offset=TEST_SEED_OFFSET, protocol="EP-A",
                        budget=f"TASK-025: configs fixed from screening on tuning seeds 3000-3999 ({a.config})")
        root = f"{ROOT}/{a.final_dir}"
    else:
        seeds = parse_seeds(a.seeds)
        if min(seeds) < 3000:
            raise SystemExit("TASK-025 tuning seeds are 3000-3999 (disjoint from every earlier seed)")
        jobs = jobs_for(a.stage, a.configs.split(","), seeds)
        root = f"{ROOT}/{a.stage}"
    todo = [j for j in jobs if not rs.done(j)]
    rs.log(f"{a.cmd} {a.stage}: {len(jobs)} jobs ({len(todo)} to do), {a.workers} workers, commit {prov['GIT_COMMIT'][:8]} dirty={prov['GIT_DIRTY']}")
    os.makedirs(root, exist_ok=True)
    rs.Scheduler(jobs, a.workers, dt.datetime.fromisoformat(a.deadline), root, prov).run()


if __name__ == "__main__":
    main()
