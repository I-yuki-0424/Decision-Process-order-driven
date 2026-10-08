import json, glob, sys, numpy as np
root = "output/phase1/truck_scale_t024/final"
lo, hi = (int(x) for x in sys.argv[1].split("-"))
for cfg in sys.argv[2:]:
    fs = [f for f in sorted(glob.glob(f"{root}/{cfg}__s*.json")) if lo <= int(f.rsplit("__s", 1)[1].split(".")[0]) <= hi]
    r, s = [], []
    for f in fs:
        d = json.load(open(f)); x = d["per_seed"][0]; r.append(x["reward_pct"]); s.append(x["score_pct"])
    r, s = np.array(r), np.array(s); n = len(r)
    se = lambda a: a.std(ddof=1) / np.sqrt(n)
    print(f"{cfg} n={n} params={d['params_total']} steps={d['env_steps_total']} reward {r.mean():.2f}+-{se(r):.2f} (sd {r.std(ddof=1):.2f}) score {s.mean():.2f}+-{se(s):.2f}")
    print("  reward", np.round(r, 1).tolist()); print("  score ", np.round(s, 1).tolist())
