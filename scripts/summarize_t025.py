"""TASK-20261008-025 final summary: per-block results, roadmap gate checks (surpass = mean - 2 SE > reference), Truck-vs-baseline differences.

  python scripts/summarize_t025.py            # prints markdown tables; reads only result files under output/phase1/

Blocks: TASK-024 finals (62-71, 72-81; local) and TASK-025 finals A (82-91, local), B (92-101, Kaggle T4), C (102-111, Kaggle T4),
D (112-121, local), E (122-131, Kaggle T4). Reference values are quoted from MASTER_GUIDANCE <references> (Dedieu et al. 2025, Table 1); not edited here.
"""
import glob
import json
import os

import numpy as np

REF = {"G1.1": dict(src="MFRL_4M_reimpl", reward=47.40, score=10.71, params=4.0e6),     # SOURCE: MASTER_GUIDANCE ref MFRL_4M_reimpl
       "G1.2": dict(src="MFRL_best", reward=55.49, score=16.77, params=None)}           # SOURCE: MASTER_GUIDANCE ref MFRL_best
BLOCKS = [("T024-1", "output/phase1/truck_scale_t024/final", range(62, 72)), ("T024-2", "output/phase1/truck_scale_t024/final", range(72, 82)),
          ("A", "output/phase1/push_t025/finalA", range(82, 92)), ("B", "output/phase1/push_t025/finalB", range(92, 102)),
          ("C", "output/phase1/push_t025/finalC", range(102, 112)), ("D", "output/phase1/push_t025/finalD", range(112, 122)),
          ("E", "output/phase1/push_t025/finalE", range(122, 132)),
          ("F", "output/phase1/push_t025/finalF", range(132, 142)),
          ("G", "output/phase1/push_t025/finalG", range(142, 152))]
ARMS = ["E1", "G4", "TP1", "G4TP1", "TP1h"]


def load(directory, arm, seeds):
    rows = []
    for s in seeds:
        f = os.path.join(directory, f"{arm}__s{s}.json")
        if os.path.exists(f):
            d = json.load(open(f))
            r = d["per_seed"][0]
            rows.append(dict(seed=s, reward=r["reward_pct"], score=r["score_pct"], params=d["params_total"], dep=d["params_deployed"],
                             steps=d["env_steps_total"], role=d["role"], protocol=d["protocol"], device=d.get("provenance", {}).get("device", "?"),
                             commit=d["git_commit"][:8], dirty=d.get("provenance", {}).get("git_dirty")))
    return rows


def ms(x):
    x = np.asarray(x, float)
    return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan"))


def main():
    data = {(b, a): load(d, a, s) for b, d, s in BLOCKS for a in ARMS}
    print("## Per block (reward_pct / score_pct, mean +- SE over seeds)\n")
    print("| block | arm | n | device | params_total | reward | score | G1.1 (r-2SE > 47.40, s-2SE > 10.71) | G1.2 (r-2SE > 55.49, s-2SE > 16.77) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b, _, _ in BLOCKS:
        for a in ARMS:
            v = data[(b, a)]
            if not v:
                continue
            rm, rs = ms([x["reward"] for x in v])
            sm, ss = ms([x["score"] for x in v])
            g = {k: (rm - 2 * rs > r["reward"]) and (sm - 2 * ss > r["score"]) and (r["params"] is None or v[0]["params"] <= r["params"])
                 for k, r in REF.items()}
            full = len(v) == 10
            print(f"| {b} | {a} | {len(v)} | {v[0]['device'].replace('NVIDIA GeForce ', '')} | {v[0]['params']:,} | {rm:.2f} +- {rs:.2f} | {sm:.2f} +- {ss:.2f} | "
                  f"{('PASS' if g['G1.1'] else 'fail') if full else 'n<10'} ({rm - 2 * rs:.2f} / {sm - 2 * ss:.2f}) | "
                  f"{('PASS' if g['G1.2'] else 'fail') if full else 'n<10'} |")
    print("\n## Differences (independent seeds, same seed numbers within a block): mean diff +- SE (z)\n")
    print("| pair | block | reward diff | score diff |")
    print("|---|---|---|---|")
    for x, y in [("E1", "G4"), ("TP1", "G4TP1"), ("G4TP1", "G4"), ("TP1", "E1"), ("TP1h", "E1"), ("TP1h", "G4TP1")]:
        pooled = {"reward": [], "score": []}
        for b, _, _ in BLOCKS:
            vx, vy = data[(b, x)], data[(b, y)]
            if len(vx) < 2 or len(vy) < 2:
                continue
            out = []
            for k in ("reward", "score"):
                mx, sx = ms([r[k] for r in vx])
                my, sy = ms([r[k] for r in vy])
                d, se = mx - my, np.hypot(sx, sy)
                pooled[k].append((d, se))
                out.append(f"{d:+.2f} +- {se:.2f} (z {d / se:.1f})")
            print(f"| {x} - {y} | {b} | {out[0]} | {out[1]} |")
        if len(pooled["reward"]) > 1:   # inverse-variance pooled over blocks
            out = []
            for k in ("reward", "score"):
                w = np.array([1 / se ** 2 for _, se in pooled[k]])
                d = float((w * np.array([d for d, _ in pooled[k]])).sum() / w.sum())
                se = float(1 / np.sqrt(w.sum()))
                out.append(f"{d:+.2f} +- {se:.2f} (z {d / se:.1f})")
            print(f"| {x} - {y} | pooled ({len(pooled['reward'])} blocks) | {out[0]} | {out[1]} |")
    print("\n## Per-seed values\n")
    for b, _, _ in BLOCKS:
        for a in ARMS:
            v = data[(b, a)]
            if v:
                print(f"- {b} {a}: " + ", ".join(f"{x['seed']}: {x['reward']:.1f}/{x['score']:.1f}" for x in v)
                      + f"  (commit {sorted({x['commit'] for x in v})}, dirty {sorted({str(x['dirty']) for x in v})}, steps {v[0]['steps']:,}, "
                        f"params_deployed {v[0]['dep']:,})")


if __name__ == "__main__":
    main()
