"""Aggregate pfirst_results.json: mean (and per-seed values) of planning return / nRMSE by (ic, n_active, arm)."""
import json, sys
from collections import defaultdict
import numpy as np

res = json.load(open(sys.argv[1]))
ref = [r for r in res if r["arm"] == "true_model_mpc"]
print("true-model MPC return:", round(ref[0]["ret"], 1) if ref else "n/a")
g = defaultdict(list)
for r in res:
    if r["arm"] != "true_model_mpc": g[(r["ic"], r["n_active"], r["arm"])].append(r)
print(f"{'ic':7s}{'Na':>4s} {'arm':11s} n  return(mean, median)   [per seed]                nRMSE act / pas (wide-IC, 40-step)")
for k in sorted(g):
    v = g[k]; rt = np.array([x["ret"] for x in v])
    print(f"{k[0]:7s}{k[1]:4d} {k[2]:11s} {len(v)}  {rt.mean():8.1f} {np.median(rt):8.1f}   {np.round(rt).astype(int).tolist()!s:26s} "
          f"{np.mean([x['nrmse_active_wideIC'] for x in v]):.3f} / {np.mean([x['nrmse_passive_wideIC'] for x in v]):.3f}")
