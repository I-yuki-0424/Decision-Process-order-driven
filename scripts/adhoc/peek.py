import glob, json, os, sys
for pat in sys.argv[1:]:
    for f in sorted(glob.glob(pat)):
        r = json.load(open(f))
        print(os.path.basename(os.path.dirname(f))[:22], os.path.basename(f)[:38], len(r["seeds"]),
              [round(s["reward_pct"], 1) for s in r["per_seed"]][:10], "score", [round(s["score_pct"], 2) for s in r["per_seed"]][:10])
