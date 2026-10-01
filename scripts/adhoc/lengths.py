import glob, json, os
import numpy as np
for f in sorted(glob.glob("output/phase1/final_1000k_all/*__*.json")) + sorted(glob.glob("output/phase1/final_100k/*__*.json")):
    r = json.load(open(f))
    L = [s["eval_mean_length"] for s in r["per_seed"]]
    c = sum(s["eval_censored"] for s in r["per_seed"])
    print(os.path.basename(os.path.dirname(f))[:16], r["arm"][:24], round(np.mean(L)), round(min(L)), round(max(L)), "censored", c)
