"""Aggregate the structure-as-input studies (arc-proposals TASK-20260929-016).

  python scripts/aggregate_arc_structure_input.py --planning output/arc_proposals/structure_input/planning \
      --prediction output/arc_proposals/structure_input/prediction --out output/arc_proposals/structure_input/summary.md
Planning: MPC return in the true env per (arm, N) over seeds (higher = better; true-model ceiling reported).
Prediction: median-over-seeds mean nRMSE per (task, arm, variant), ID / OOD (lower = better).
Also prints the falsification check: does the physics feature beat every control feature (poly4/hifreq/zero/harmonic)?
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def planning(paths):
    rows = []
    for path in paths:
        f = os.path.join(path, "structure_planning_results.json")
        if os.path.exists(f):
            rows += json.load(open(f))
    ceil = [r["ret"] for r in rows if r["arm"] == "true_model_mpc"] or [float("nan")]
    g = defaultdict(list)
    for r in rows:
        if r["arm"] != "true_model_mpc":
            g[(r["arm"], r["n_traj"])].append(r["ret"])
    sizes = sorted({k[1] for k in g})
    arms = list(dict.fromkeys(k[0] for k in g))
    out = [f"### Pendulum swing-up MPC return (true-model ceiling {ceil[0]:.1f}); mean ± sd (median) over seeds\n",
           "| arm | " + " | ".join(f"N={n}" for n in sizes) + " |", "|---|" + "---|" * len(sizes)]
    for a in arms:
        cells = []
        for n in sizes:
            v = np.array(g.get((a, n), []))
            cells.append(f"{v.mean():7.1f} ± {v.std(ddof=1) if len(v) > 1 else 0:5.1f} ({np.median(v):7.1f}) n={len(v)}" if len(v) else "-")
        out.append(f"| {a} | " + " | ".join(cells) + " |")
    ctrl = ["feat:poly4", "feat:hifreq", "feat:zero"]
    out.append("\nFalsification check (per N: physics-carrying features vs control features, mean return):")
    for n in sizes:
        m = {a: np.mean(g[(a, n)]) for a in arms if (a, n) in g}
        if "feat:smallangle" in m and all(c in m for c in ctrl):
            best_ctrl = max(m[c] for c in ctrl)
            out.append(f"- N={n}: feat:smallangle {m['feat:smallangle']:.1f} vs best control {best_ctrl:.1f} "
                       f"({'physics feature better by %.1f' % (m['feat:smallangle'] - best_ctrl)}); hn {m.get('hn', float('nan')):.1f}")
    return "\n".join(out)


def prediction(path):
    rows = [r for r in json.load(open(os.path.join(path, "structure_prediction_results.json"))) if "error" not in r]
    out = []
    for task in dict.fromkeys(r["task"] for r in rows):
        g = defaultdict(lambda: ([], []))
        for r in rows:
            if r["task"] == task:
                g[(r["arm"], r["variant"])][0].append(r["id"]["rmse_mean"])
                g[(r["arm"], r["variant"])][1].append(r["ood"]["rmse_mean"])
        out.append(f"\n### {task}: median mean-nRMSE over seeds (ID / OOD), lower = better\n")
        out.append("| arm | variant | ID | OOD | n |")
        out.append("|---|---|---|---|---|")
        for (arm, var), (i, o) in g.items():
            out.append(f"| {arm} | {var} | {np.median(i):.3f} | {np.median(o):.3f} | {len(i)} |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planning", nargs="+")
    ap.add_argument("--prediction")
    ap.add_argument("--out")
    a = ap.parse_args()
    parts = []
    if a.planning:
        parts.append(planning(a.planning))
    if a.prediction and os.path.exists(os.path.join(a.prediction, "structure_prediction_results.json")):
        parts.append(prediction(a.prediction))
    text = "\n\n".join(parts)
    print(text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text + "\n")
