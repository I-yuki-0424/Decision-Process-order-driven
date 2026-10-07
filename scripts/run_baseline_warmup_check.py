"""TASK-024 equal-effort tuning (P9) of the pure-RL baseline, given the same knobs and seeds that were used on Truck.
  python scripts/run_baseline_warmup_check.py tune  --configs B1,B2 --seeds 2000-2002 --deadline 2026-10-07T12:00   # tuning seeds only
  python scripts/run_baseline_warmup_check.py final --configs B2 --deadline ...                                       # EP-A seeds 62-71 / test 444-453
  python scripts/run_baseline_warmup_check.py report
Results: output/phase1/truck_scale_t024/{baseline,final}/<config>__s<seed>.json. Configs are fixed in docs/experiments/2026-10-05_truck_scale_study/README.md."""
import argparse, datetime as dt, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_phase1_recipe_sweep as rs

REF = dict(rs.ANCHORS["ref"], warmup=0.05)
E1PT = dict(n=64, t=64, lr=1e-3, epochs=3, minibatches=8, ent=0.01, lam=0.625, gamma=0.97, clip=0.2, vf=1.0, max_grad_norm=0.5,
            value_norm=0.99, adv_norm="batch", warmup=0.05)   # the Truck E1 recipe point
G512, G256LS, G512LS = {"width": 512}, {"ln": True, "skip": True}, {"width": 512, "ln": True, "skip": True}
CONFIGS = {
    "B1": (G512, REF),                               # TASK-023 baseline + warm-up
    "B2": (G512, E1PT),                              # baseline at Truck's E1 recipe point
    "B3": (G256LS, E1PT),
    "B4": (G512LS, E1PT),
    "B5": (G512, dict(E1PT, lr=5e-4)),
    "B6": (G512, dict(REF, lr=1e-3)),
    # round 2 (around the round-1 winner B3 = gru256+ln+skip at the Truck E1 point; same knobs that mattered for Truck)
    "G1": (G256LS, dict(E1PT, lr=1.5e-3)),
    "G2": (G256LS, dict(E1PT, gamma=0.95)),
    "G3": (G256LS, dict(E1PT, ent=0.003)),
    "G4": (G256LS, dict(E1PT, epochs=4)),
}


def build(names, seeds, role):
    jobs = []
    for s in seeds:
        for n in names:
            kw, p = CONFIGS[n]
            d = "baseline" if role == "tune" else "final"
            out = f"output/phase1/truck_scale_t024/{d}/{n}__s{s}.json"
            cmd = rs.cli("ppo_gru", kw, p, s, role, out, 1_000_000, "EP-A-tuning" if role == "tune" else "EP-A",
                         "TASK-024 equal-effort baseline tuning: configs B1-B6 x 3 tuning seeds", offset=382 if role == "final" else 0)
            cmd += ["--warmup", str(p["warmup"])]
            jobs.append(dict(name=f"{n}__s{s}", out=out, seed=s, mem=2500, sec=400, env={}, extra=[], cmd=cmd))
    return jobs


def best_config():
    """Max mean(reward_pct + score_pct) over the 3 tuning seeds; configs within 3 points of the best -> the one with fewer parameters."""
    rows = {}
    for f in glob.glob("output/phase1/truck_scale_t024/baseline/*.json"):
        j = json.load(open(f)); r = j["per_seed"][0]
        rows.setdefault(os.path.basename(f).split("__s")[0], []).append((r["reward_pct"] + r["score_pct"], j["params_total"]))
    rows = {n: (np.mean([x[0] for x in v]), v[0][1]) for n, v in rows.items() if len(v) == 3}
    top = max(v[0] for v in rows.values())
    return min((n for n, v in rows.items() if v[0] >= top - 3), key=lambda n: (rows[n][1], -rows[n][0]))   # fewer parameters, then higher mean r+s


def report():
    for d in ("baseline", "final"):
        rows = {}
        for f in sorted(glob.glob(f"output/phase1/truck_scale_t024/{d}/*.json")):
            j = json.load(open(f)); r = j["per_seed"][0]
            rows.setdefault(os.path.basename(f).split("__s")[0], []).append((r["reward_pct"], r["score_pct"], j["params_total"]))
        for n, v in rows.items():
            a = np.array([x[:2] for x in v]); se = a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else [float("nan")] * 2
            print(f"{d:8s} {n:4s} n={len(v)} params={v[0][2]} reward {a[:, 0].mean():.2f}+-{se[0]:.2f} score {a[:, 1].mean():.2f}+-{se[1]:.2f} "
                  f"r+s {a.mean(0).sum():.2f} | {np.round(a[:, 0], 0).astype(int).tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "final", "report"])
    ap.add_argument("--configs", default="B1,B2,B3,B4,B5,B6")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--deadline", default="")
    a = ap.parse_args()
    if a.cmd == "report":
        sys.exit(report())
    lo, _, hi = (a.seeds or ("2000-2002" if a.cmd == "tune" else "62-71")).partition("-")
    if a.configs == "best":
        a.configs = best_config()
        print("baseline config selected on tuning seeds:", a.configs, flush=True)
    prov = rs.host_provenance()
    if a.cmd == "final" and prov["GIT_DIRTY"] == "1":
        raise SystemExit("commit first")
    rs.Scheduler(build(a.configs.split(","), range(int(lo), int(hi or lo) + 1), a.cmd), 1, dt.datetime.fromisoformat(a.deadline),
                 "output/phase1/truck_scale_t024/baseline", prov).run()
