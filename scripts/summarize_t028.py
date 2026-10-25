"""TASK-20261025-028 summary: scaling curve, per-achievement attribution of the gain, mid-run evaluations of S5M, paired replay differences.

  python scripts/summarize_t028.py [output/phase1/scaling_t028/S]
Reading rules: docs/experiments/2026-10-25_data_scaling_reuse/README.md section 1 (fixed before the runs).
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from src.environment.craftax_env_adapter import ACHIEVEMENT_NAMES  # noqa: E402

SCALE = [("S05M", 0.5), ("S1M", 1.0), ("S2M", 2.0), ("S5M", 5.0)]
REUSE = ["RP2", "RP4", "RP4e2"]


def load(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        rows.setdefault(os.path.basename(f).split("__s")[0], {})[r["seeds"][0]] = r["per_seed"][0]
    return rows


def ms(x):
    x = np.asarray(x, float)
    return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan"))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "output/phase1/scaling_t028/S"
    rows = load(d)
    print("== Q1 scaling (reward_pct / score_pct, mean +- SE over seeds; length; train s)")
    for n, m in SCALE + [(r, 1.0) for r in REUSE]:
        if n not in rows:
            print(f"{n:6s} missing")
            continue
        v = list(rows[n].values())
        (rm, rse), (sm, sse) = ms([x["reward_pct"] for x in v]), ms([x["score_pct"] for x in v])
        print(f"{n:6s} {m:4.1f}M n={len(v)} reward {rm:6.2f} +- {rse:4.2f}  score {sm:6.2f} +- {sse:4.2f}  len {np.mean([x['eval_mean_length'] for x in v]):5.0f}"
              f"  train_s {np.mean([x['train_seconds'] for x in v]):6.0f}  seeds {sorted(rows[n])}")
    have = [(n, m) for n, m in SCALE if n in rows]
    if len(have) >= 2:
        xs = np.log2([m for _, m in have])
        ys = [np.mean([x["reward_pct"] for x in rows[n].values()]) for n, _ in have]
        print("slope per doubling between consecutive budgets:",
              {f"{have[i][0]}->{have[i + 1][0]}": round(float((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])), 2) for i in range(len(have) - 1)})
    rate = lambda n: np.mean([x["achievement_rates_pct"] for x in rows[n].values()], 0)
    for a, b in (("S1M", "S5M"), ("S1M", "S2M"), ("S05M", "S1M")):
        if a in rows and b in rows:
            delta = (rate(b) - rate(a)) / 22.0
            order = np.argsort(-delta)
            tot = delta.sum()
            print(f"== Q2 {a} -> {b}: delta reward_pct {tot:+.2f}; per achievement (rate {a} -> {b}, contribution, cumulative share of the gain)")
            cum = 0.0
            for i in order:
                cum += delta[i]
                print(f"   {ACHIEVEMENT_NAMES[i]:20s} {rate(a)[i]:5.1f} -> {rate(b)[i]:5.1f}  {delta[i]:+5.2f}  {cum / tot * 100 if tot else 0:5.0f}%")
    if "S5M" in rows:
        print("== S5M mid-run evaluations (lr not yet annealed; descriptive): env_steps reward score | rates of achievements < 50% at 1M budget")
        low = [i for i in range(22)] if "S1M" not in rows else [i for i in range(22) if rate("S1M")[i] < 50]
        pts = {}
        for x in rows["S5M"].values():
            for e in x.get("intermediate_evals", []) + [dict(env_steps=-1, reward_pct=x["reward_pct"], score_pct=x["score_pct"],
                                                              achievement_rates_pct=x["achievement_rates_pct"])]:
                pts.setdefault(e["env_steps"], []).append(e)
        for s in sorted(pts, key=lambda k: k if k > 0 else 1e12):
            v = pts[s]
            r = np.mean([e["achievement_rates_pct"] for e in v], 0)
            print(f"   {'final' if s < 0 else s:>8} reward {np.mean([e['reward_pct'] for e in v]):6.2f} score {np.mean([e['score_pct'] for e in v]):6.2f} | "
                  + " ".join(f"{ACHIEVEMENT_NAMES[i][:12]}={r[i]:.0f}" for i in low))
    if "S1M" in rows:
        print("== Q4 replay vs S1M (paired by seed): mean diff +- SE; candidate rule: reward diff >= +2.0 and > 2 SE")
        for n in REUSE:
            if n not in rows:
                continue
            common = sorted(set(rows[n]) & set(rows["S1M"]))
            dr = [rows[n][s]["reward_pct"] - rows["S1M"][s]["reward_pct"] for s in common]
            dsc = [rows[n][s]["score_pct"] - rows["S1M"][s]["score_pct"] for s in common]
            (m1, s1), (m2, s2) = ms(dr), ms(dsc)
            print(f"   {n:6s} n={len(common)} reward {m1:+5.2f} +- {s1:4.2f}  score {m2:+5.2f} +- {s2:4.2f}  candidate={bool(m1 >= 2.0 and m1 > 2 * s1)}")


if __name__ == "__main__":
    main()
