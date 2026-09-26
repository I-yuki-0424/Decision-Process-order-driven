"""Aggregate Stage-B result.json files (any number of dirs) by arm: final sampled return, last-3-eval mean, crafter score."""
import glob, json, re, sys
import numpy as np

rows = {}
for d in sys.argv[1:]:
    for f in glob.glob(f"{d}/**/*.result.json", recursive=True):
        r = json.load(open(f)); m = re.match(r".*__(\w+)__s(\d+)$", r["tag"])
        arm, seed = m.group(1), int(m.group(2))
        eh = r["eval_history"]
        rows.setdefault(arm, {})[seed] = dict(
            final=r["final_sampled"]["mean_return"], last3=float(np.mean([e["mean_return"] for e in eh[-3:]])),
            crafter=r["final_sampled"]["crafter_score"], random=r["random_policy"]["mean_return"], src=f)
print(f"{'arm':12s} n  final_return(mean±sd)  last3_eval(mean±sd)  crafter   per-seed last3")
for arm, s in sorted(rows.items()):
    fin = np.array([v["final"] for v in s.values()]); l3 = np.array([v["last3"] for v in s.values()]); cr = np.array([v["crafter"] for v in s.values()])
    print(f"{arm:12s} {len(s)}  {fin.mean():.3f}±{fin.std():.3f}        {l3.mean():.3f}±{l3.std():.3f}       {cr.mean():.2f}   "
          + " ".join(f"s{k}={v['last3']:.2f}" for k, v in sorted(s.items())))
rnd = [v["random"] for s in rows.values() for v in s.values()]
print(f"uniform-random reference return (same evaluator): {np.mean(rnd):.3f}")
