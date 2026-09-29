"""Plot the world-model architecture series on the Kepler two-body problem (E3 of
docs/experiments/SUMMARY_2026-09-25_to_26_world_model_series.md), derived from the same
idea6/Hamiltonian-structure hypothesis as the six Craftax candidate architectures.

Real data only: output/experiments/2026-09-25_physics/kepler/physics_results.json (N=128,
clean, median over 3 seeds; persistence has no training data so 1 record). Metric matches the
summary's correction #4 (mean nRMSE OVER THE ROLLOUT, not end-of-rollout, since end-of-rollout
is periodicity-degenerate) and scripts/aggregate_physics.py's own aggregation
(median_over_seeds(mean_over_rollout(rmse_curve))).
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "output/plots"
os.makedirs(OUT_DIR, exist_ok=True)

d = json.load(open("output/experiments/2026-09-25_physics/kepler/physics_results.json", encoding="utf-8"))
arms = ["persistence", "mlp", "hn_gen", "hn_sep", "port_sep", "idea6_full", "oracle_port"]
arm_labels = {
    "persistence": "persistence\n(x_t+k=x_t)",
    "mlp": "mlp\n(x_dot=MLP(x))",
    "hn_gen": "hn_gen\n(J grad MLP(x))",
    "hn_sep": "hn_sep\n(known kinetic\n+ learned V)",
    "port_sep": "port_sep\n((J-R) grad H)",
    "idea6_full": "idea6_full\n(WorldModel class\nas implemented)",
    "oracle_port": "oracle_port\n(TRUE potential,\nupper bound)",
}
colors = {
    "persistence": "#95a5a6", "mlp": "#7f8c8d", "hn_gen": "#3498db", "hn_sep": "#27ae60",
    "port_sep": "#e67e22", "idea6_full": "#c0392b", "oracle_port": "#2c3e50",
}

def agg(arm):
    recs = [r for r in d if r["arm"] == arm and r["noise"] == 0.0 and (r["n_traj"] == 128 or arm == "persistence")]
    id_recs = [r for r in recs if "id" in r]
    ood_recs = [r for r in recs if "ood" in r]
    id_med = float(np.median([np.mean(r["id"]["rmse_curve"]) for r in id_recs]))
    ood_med = float(np.median([np.mean(r["ood"]["rmse_curve"]) for r in ood_recs]))
    drift_med = float(np.median([r["id"]["inv_drift_end"][0] for r in id_recs]))
    id_curves = np.array([r["id"]["rmse_curve"] for r in id_recs])
    ood_curves = np.array([r["ood"]["rmse_curve"] for r in ood_recs])
    return id_med, ood_med, drift_med, np.median(id_curves, axis=0), np.median(ood_curves, axis=0)

stats = {a: agg(a) for a in arms}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel 1: ID vs OOD bar chart
ax = axes[0, 0]
x = np.arange(len(arms))
width = 0.35
id_vals = [stats[a][0] for a in arms]
ood_vals = [stats[a][1] for a in arms]
ax.bar(x - width / 2, id_vals, width, label="in-distribution", color=[colors[a] for a in arms])
ax.bar(x + width / 2, ood_vals, width, label="out-of-distribution", color=[colors[a] for a in arms], alpha=0.5)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([a for a in arms], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("mean normalised RMSE over rollout (log scale, lower=better)")
ax.set_title("Kepler two-body: architecture series, N=128 trajectories, clean\n(median over 3 seeds; oracle_port = upper bound, not a real method)")
ax.legend(fontsize=9)
ax.grid(axis="y", which="major", alpha=0.3)

# Panel 2: energy-drift (physical-consistency check)
ax = axes[0, 1]
drift_vals = [stats[a][2] for a in arms]
ax.bar(x, drift_vals, color=[colors[a] for a in arms])
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([a for a in arms], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("|energy drift| / |energy_0| at rollout end (log scale)")
ax.set_title("Energy conservation over the rollout\n(structure-respecting arms should drift far less)")
ax.grid(axis="y", which="major", alpha=0.3)

# Panel 3: ID error-vs-horizon curves
ax = axes[1, 0]
for a in arms:
    if a in ("persistence", "idea6_full", "oracle_port"):
        continue  # keep this panel to the 4 comparable structured/unstructured learners
    ax.plot(stats[a][3], color=colors[a], label=a, linewidth=2)
ax.plot(stats["persistence"][3], color=colors["persistence"], label="persistence", linewidth=1.5, linestyle=":")
ax.set_xlabel("rollout step")
ax.set_ylabel("normalised RMSE (ID)")
ax.set_title("Error growth over the rollout (ID)\n-- structure (hn_sep/port_sep) saturates, generic mlp keeps drifting")
ax.legend(fontsize=8)

# Panel 4: OOD error-vs-horizon curves
ax = axes[1, 1]
for a in arms:
    if a in ("persistence", "idea6_full", "oracle_port"):
        continue
    ax.plot(stats[a][4], color=colors[a], label=a, linewidth=2)
ax.plot(stats["persistence"][4], color=colors["persistence"], label="persistence", linewidth=1.5, linestyle=":")
ax.set_xlabel("rollout step")
ax.set_ylabel("normalised RMSE (OOD)")
ax.set_title("Error growth over the rollout (OOD)\n-- generic learners (mlp, hn_gen) diverge; hn_sep stays bounded")
ax.legend(fontsize=8)

fig.suptitle("World-model architecture series on the Kepler two-body problem\n(Hamiltonian-structure hypothesis test, derived from the idea6 world-model candidate architecture)",
             fontsize=13, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, "fair_two_body_kepler_architecture_series.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote", os.path.join(OUT_DIR, "fair_two_body_kepler_architecture_series.png"))
for a in arms:
    print(f"{a:12s} ID={stats[a][0]:.3f} OOD={stats[a][1]:.3f} energy_drift={stats[a][2]:.4g}")
