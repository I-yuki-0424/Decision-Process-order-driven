"""Derivative of the structure-as-input proposal (arc-proposals TASK-20260929-016): choose the ROUTE for the known-but-incomplete preset
(no preset `hn` | feature-only `phys_feat` | closed-gate skip `skip_feat0` | open-gate skip `skip_feat` | hard term `phys_hn`) by held-out IN-distribution
rollout error only (what a modeller can measure), then report the OUT-of-distribution error of the chosen route.
Compared against: every fixed route, and the hindsight-best route by OOD (an unattainable bound). No new training: uses the seed-matched results of
scripts/run_structure_input_prediction.py (preset variant only). Selection metric = ID mean-nRMSE on held-out ICs; evaluation = OOD mean-nRMSE.
  python scripts/analyze_structure_route_selection.py output/arc_proposals/structure_input/prediction output/arc_proposals/structure_input/prediction2
"""
import json, os, sys
from collections import defaultdict
import numpy as np

ROUTES = [("hn", "none"), ("phys_feat", "preset"), ("skip_feat0", "preset"), ("skip_feat", "preset"), ("phys_hn", "preset")]
rows = []
for d in sys.argv[1:]:
    f = os.path.join(d, "structure_prediction_results.json")
    if os.path.exists(f):
        rows += [r for r in json.load(open(f)) if "error" not in r]
by = defaultdict(dict)
for r in rows:
    key = (r["arm"], r["variant"])
    if key in ROUTES:
        by[(r["task"], r["seed"])][key] = (r["id"]["rmse_mean"], r["ood"]["rmse_mean"])
tasks = sorted({k[0] for k in by})
lines = ["| task | " + " | ".join(f"{a}" for a, _ in ROUTES) + " | **selected by ID** | hindsight-best (OOD) | n seeds | picks |", "|---|" + "---|" * (len(ROUTES) + 4)]
tot = defaultdict(list)
for t in tasks:
    seeds = [s for (tt, s), v in by.items() if tt == t and all(k in v for k in ROUTES)]
    if not seeds:
        continue
    fixed = {k: [] for k in ROUTES}; sel, best, picks = [], [], []
    for s in seeds:
        v = by[(t, s)]
        for k in ROUTES: fixed[k].append(v[k][1])
        c = min(ROUTES, key=lambda k: v[k][0]); sel.append(v[c][1]); picks.append(c[0])
        best.append(min(v[k][1] for k in ROUTES))
    med = lambda x: float(np.median(x))
    lines.append(f"| {t} | " + " | ".join(f"{med(fixed[k]):.3f}" for k in ROUTES) + f" | **{med(sel):.3f}** | {med(best):.3f} | {len(seeds)} | "
                 + ", ".join(f"{p}:{picks.count(p)}" for p in dict.fromkeys(picks)) + " |")
    for k in ROUTES: tot[k[0]].append(med(fixed[k]))
    tot["selected"].append(med(sel)); tot["best"].append(med(best))
print("Median OOD nRMSE over seeds (lower = better); columns = fixed routes; selection uses ID error only.\n")
print("\n".join(lines))
print("\nGeometric-mean OOD nRMSE across tasks: " + ", ".join(f"{k}={np.exp(np.mean(np.log(v))):.3f}" for k, v in tot.items()))
