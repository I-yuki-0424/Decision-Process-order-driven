"""Parameter-count vs performance: hidden-width sweep (median over seeds, N=64, noise 0.02, mean nRMSE)."""
import glob
import json
import sys

import numpy as np

E = sys.argv[1]  # dir containing hid*/hybrid_results.json plus optional extra json paths in argv[2:]
extra = sys.argv[2:]
R = []
for f in glob.glob(f"{E}/hid*/hybrid_results.json") + extra:
    for r in json.load(open(f)):
        if "error" not in r and r["arm"] != "persistence" and r["n_traj"] == 64:
            r.setdefault("hid", 64)
            R.append(r)
tasks = sorted({r["task"] for r in R})
arms = sorted({r["arm"] for r in R})
hids = sorted({r["hid"] for r in R})
L = ["# Parameter scaling (N=64, noise 2%, median of seeds; cell = params | ID mean nRMSE | OOD mean nRMSE)", ""]
for t in tasks:
    L += [f"## {t}", "", "| arm | " + " | ".join(f"hid={h}" for h in hids) + " |", "|---|" + "---|" * len(hids)]
    for a in arms:
        cells = []
        for h in hids:
            rr = [r for r in R if r["task"] == t and r["arm"] == a and r["hid"] == h]
            if not rr:
                cells.append("n/a")
                continue
            cells.append(f"{int(np.median([r['params'] for r in rr]))} \\| {np.median([r['id']['rmse_mean'] for r in rr]):.3f} \\| "
                         f"{np.median([r['ood']['rmse_mean'] for r in rr]):.3f}")
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    L.append("")
open(f"{E}/sweep_tables.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
