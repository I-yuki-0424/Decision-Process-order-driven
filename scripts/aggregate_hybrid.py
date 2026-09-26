"""Aggregate hybrid world-model benchmark results: median over seeds, mean-over-rollout nRMSE, ID and OOD."""
import glob
import json
import os
import sys

import numpy as np

E = sys.argv[1]
TASKS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["kepler_pert", "kepler_wrongmu", "fall_drag", "pendulum"]
R = []
for f in sorted(glob.glob(f"{E}/**/hybrid_results.json", recursive=True)):
    R += json.load(open(f))
bad = [r for r in R if "error" in r]
R = [r for r in R if "error" not in r]
ARMS = []
for r in R:
    if r["arm"] not in ARMS:
        ARMS.append(r["arm"])
sizes = sorted({r["n_traj"] for r in R if r["arm"] != "persistence"})


def cell(task, arm, split, n):
    v = [r[split]["rmse_mean"] for r in R if r["task"] == task and r["arm"] == arm and split in r
         and (arm == "persistence" or r["n_traj"] == n)]
    if not v:
        return "n/a"
    m = np.median(v)
    return f"{m:.3f}" if m < 100 else f"{m:.0e}"


L = [f"# Hybrid world-model results (median of seeds; mean nRMSE over the 200-250-step rollout; lower is better; {len(bad)} failed runs)", ""]
for task in TASKS:
    L += [f"## {task}", "", "| arm | params | " + " | ".join(f"ID N={n} | OOD N={n}" for n in sizes) + " |", "|---|---|" + "---|" * (2 * len(sizes))]
    for a in ARMS:
        par = next((r["params"] for r in R if r["task"] == task and r["arm"] == a), 0)
        L.append(f"| {a} | {par} | " + " | ".join(f"{cell(task, a, 'id', n)} | {cell(task, a, 'ood', n)}" for n in sizes) + " |")
    L.append("")
open(f"{E}/hybrid_tables.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
