"""Final consolidated tables for the hybrid world-model study (pooled seeds, planning, partial obs, lambda, sweep pointer)."""
import glob
import json
import sys

import numpy as np

H = sys.argv[1]
TASKS = ["kepler_pert", "kepler_wrongmu", "fall_drag", "pendulum"]
out = []


def load(pattern):
    R = []
    for f in sorted(glob.glob(pattern, recursive=True)):
        R += json.load(open(f))
    return [r for r in R if "error" not in r]


# ---- pooled prediction table, N=64, noise 2%, dedupe by (task,arm,seed)
P = []
for t in TASKS:
    P += load(f"{H}/main/{t}/hybrid_results.json")
for d in ("feat_pred", "skip_pred", "rep_seeds", "rep_skip0"):
    P += load(f"{H}/{d}/hybrid_results.json")
seen, U = set(), []
for r in P:
    if r["arm"] == "persistence" or r["n_traj"] != 64 or r["noise"] != 0.02:
        continue
    k = (r["task"], r["arm"], r["seed"])
    if k not in seen:
        seen.add(k)
        U.append(r)
arms = ["mlp", "hn", "phys_hn", "phys_feat", "skip_feat", "skip_feat0", "hyb_sum_mlp", "rssm", "hyb_sum_rssm", "hyb_tf_rssm"]
out += ["## A. State prediction, N=64 trajectories, 2% noise, pooled seeds (mean nRMSE over the rollout; median [min, max])", "",
        "| task | arm | seeds | ID | OOD |", "|---|---|---|---|---|"]
for t in TASKS:
    for a in arms:
        v = [r for r in U if r["task"] == t and r["arm"] == a]
        if not v:
            continue
        i = [r["id"]["rmse_mean"] for r in v]
        o = [r["ood"]["rmse_mean"] for r in v]
        out.append(f"| {t} | {a} | {len(v)} | {np.median(i):.3f} [{min(i):.3f}, {max(i):.3f}] | {np.median(o):.3f} [{min(o):.3f}, {max(o):.3f}] |")

# ---- planning
PL = []
for d in ("plan", "plan_feat", "plan_skip", "plan_skip0"):
    PL += load(f"{H}/{d}/planning_results.json")
refs = [r for r in PL if r["arm"].endswith("mpc")]
out += ["", "## B. Planning: pendulum swing-up, CEM-MPC with the learned model (episode return; higher is better; 24 episodes x 3 seeds)", "",
        "References: " + "; ".join(f"{r['arm']} {r['ret']:.0f} (success {r['success']:.2f})" for r in {r['arm']: r for r in refs}.values()) + ". "
        "Success is 100% for every learned model, so return (speed of swing-up) is the informative metric.", "",
        "| model | N=8 | N=32 | N=128 | preset given |", "|---|---|---|---|---|"]
desc = {"mlp": "none", "hn": "none (known kinetic term only)", "phys_hn": "small-angle q^2/2 as hard term (WRONG at large angle)",
        "hyb": "as phys_hn + MLP residual", "phys_feat": "q^2/2 as input feature only (WRONG)", "skip_feat": "feature + learned-scale skip (wrong preset)",
        "skip_feat0": "feature + skip starting closed (wrong preset)", "oracle_phys": "TRUE -cos(q) hard term (upper bound)",
        "oracle_feat": "TRUE -cos(q) as feature (upper bound)", "oracle_skip": "TRUE feature + skip (upper bound)"}
for a in ["mlp", "hn", "phys_hn", "hyb", "phys_feat", "skip_feat", "skip_feat0", "oracle_phys", "oracle_feat", "oracle_skip"]:
    cells = []
    for n in (8, 32, 128):
        v = [r["ret"] for r in PL if r["arm"] == a and r["n_traj"] == n]
        cells.append(f"{np.median(v):.0f}" if v else "n/a")
    out.append(f"| {a} | " + " | ".join(cells) + f" | {desc[a]} |")

# ---- partial observation
PO = load(f"{H}/partial/*/partial_results.json")
out += ["", "## C. Partial observation (only noisy position observed; mean nRMSE, ID, median of 3 seeds)", "",
        "| task | obs noise | dynamics | oracle state | finite-difference p | learned filter |", "|---|---|---|---|---|---|"]
for t in ("pendulum", "fall_drag", "kepler_pert"):
    for nz in (0.02, 0.05):
        for a in ("phys_hn", "hn", "mlp", "hyb_sum_mlp"):
            c = []
            for e in ("oracle", "fd", "learned"):
                v = [r["id"]["rmse_mean"] for r in PO if r["task"] == t and r["arm"] == a and r["estimator"] == e and r["obs_noise"] == nz]
                c.append(f"{np.median(v):.3f}" if v else "n/a")
            out.append(f"| {t} | {nz} | {a} | " + " | ".join(c) + " |")

# ---- residual penalty
out += ["", "## D. Residual penalty (hyb_sum_*; N=64, 2% noise, ID/OOD mean nRMSE, median of 3 seeds)", "", "| task | arm | lam=0 | lam=0.1 | lam=1 |", "|---|---|---|---|---|"]
main = {}
for t in ("fall_drag", "pendulum"):
    for r in load(f"{H}/main/{t}/hybrid_results.json"):
        if r["n_traj"] == 64:
            main.setdefault((t, r["arm"], 0.0), []).append(r)
for lam, d in ((0.1, "lam/lam0.1"), (1.0, "lam/lam1")):
    for r in load(f"{H}/{d}/hybrid_results.json"):
        if r["arm"] != "persistence":
            main.setdefault((r["task"], r["arm"], lam), []).append(r)
for t in ("fall_drag", "pendulum"):
    for a in ("hyb_sum_mlp", "hyb_sum_rssm"):
        c = []
        for lam in (0.0, 0.1, 1.0):
            v = main.get((t, a, lam), [])
            c.append(f"{np.median([r['id']['rmse_mean'] for r in v]):.3f} / {np.median([r['ood']['rmse_mean'] for r in v]):.3f}" if v else "n/a")
        out.append(f"| {t} | {a} | " + " | ".join(c) + " |")

open(f"{H}/final_tables.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
