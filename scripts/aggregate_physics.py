"""Aggregate physics-benchmark JSONs into tables + plots (median over seeds)."""
import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = sys.argv[1]
R = []
for f in glob.glob(f"{E}/*/physics_results.json"):
    R += json.load(open(f))
R = [r for r in R if "error" not in r]
errs = 0
ARMS = ["persistence", "mlp", "hn_gen", "hn_sep", "port_sep", "idea6_full", "oracle_port"]
TASKS = ["kepler", "fall_drag", "sine"]


def med(task, arm, split, key, n=None, noise=None):
    v = [r[split][key] for r in R if r["task"] == task and r["arm"] == arm and split in r
         and (n is None or r["n_traj"] == n) and (noise is None or r["noise"] == noise)]
    return (float(np.median(v)), len(v)) if v else (np.nan, 0)


lines = ["# Results tables (median over 3 seeds; normalised RMSE of the state at the END of the long open-loop rollout; "
         "lower is better; 1.0 ~ predicting the data's spread)", ""]
for task in TASKS:
    for split in ("id", "ood"):
        lines += [f"## {task} - {split.upper()} initial conditions", "",
                  "| arm | params | N=16 clean | N=16 noisy | N=128 clean | N=128 noisy | N=1024 clean | N=1024 noisy |", "|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            if arm == "persistence":
                v, _ = med(task, arm, split, "rmse_end")
                lines.append(f"| {arm} | 0 | " + " | ".join([f"{v:.2f}"] * 6) + " |")
                continue
            par = next((r["params"] for r in R if r["task"] == task and r["arm"] == arm), 0)
            cells = []
            for n in (16, 128, 1024):
                for nz in (0.0, 0.02):
                    v, k = med(task, arm, split, "rmse_end", n, nz)
                    cells.append(f"{v:.3f}" if v < 100 else f"{v:.0e}")
            lines.append(f"| {arm} | {par} | " + " | ".join(cells) + " |")
        lines.append("")
lines += ["## Short-horizon (h=10 = training horizon) error, ID, noisy, N=128", "", "| task | " + " | ".join(ARMS[1:]) + " |", "|---|" + "---|" * 6]
for task in TASKS:
    lines.append(f"| {task} | " + " | ".join(f"{med(task, a, 'id', 'rmse_h10', 128, 0.02)[0]:.3f}" for a in ARMS[1:]) + " |")
lines += ["", "## Drift of TRUE invariants at end of rollout (ID, clean, N=128; median |dI|/|I0|; Kepler: energy, angular momentum)", "",
          "| task | " + " | ".join(ARMS[1:]) + " |", "|---|" + "---|" * 6]
for task in TASKS:
    cells = []
    for a in ARMS[1:]:
        v = [r["id"]["inv_drift_end"] for r in R if r["task"] == task and r["arm"] == a and r["n_traj"] == 128 and r["noise"] == 0.0]
        cells.append(", ".join(f"{x:.2g}" for x in np.median(v, 0)) if v else "n/a")
    lines.append(f"| {task} | " + " | ".join(cells) + " |")
lines += ["", "## Mean nRMSE over the whole rollout (26 sampled horizons; robust to the end-point coincidence noted in the write-up), N=128, clean", "",
          "| task / split | " + " | ".join(ARMS) + " |", "|---|" + "---|" * 7]
for task in TASKS:
    for split in ("id", "ood"):
        cells = []
        for a in ARMS:
            cs = [r[split]["rmse_curve"] for r in R if r["task"] == task and r["arm"] == a and split in r
                  and (a == "persistence" or (r["n_traj"] == 128 and r["noise"] == 0.0))]
            cells.append(f"{np.median(np.mean(cs, 1)):.3f}" if cs else "n/a")
        lines.append(f"| {task} {split} | " + " | ".join(cells) + " |")
open(f"{E}/results_tables.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")

os.makedirs(f"{E}/plots", exist_ok=True)
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for j, task in enumerate(TASKS):
    for i, split in enumerate(("id", "ood")):
        for arm in ARMS[1:]:
            xs = [16, 128, 1024]
            ys = [med(task, arm, split, "rmse_end", n, 0.0)[0] for n in xs]
            ax[i, j].plot(xs, ys, marker="o", label=arm, ls="--" if arm == "oracle_port" else "-")
        ax[i, j].set_xscale("log"); ax[i, j].set_yscale("log"); ax[i, j].set_title(f"{task} {split.upper()} (clean)")
        ax[i, j].set_xlabel("training trajectories"); ax[i, j].set_ylabel("end-of-rollout nRMSE")
ax[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{E}/plots/data_efficiency.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for j, task in enumerate(TASKS):
    for arm in ARMS[1:]:
        cs = [r["id"]["rmse_curve"] for r in R if r["task"] == task and r["arm"] == arm and r["n_traj"] == 128 and r["noise"] == 0.0]
        if cs:
            ax[j].plot(np.median(cs, 0), label=arm)
    ax[j].set_yscale("log"); ax[j].set_title(f"{task}: error vs horizon (ID, clean, N=128)"); ax[j].set_xlabel("rollout step (sampled)")
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{E}/plots/error_vs_horizon.png", dpi=110)
print("\n".join(lines))
