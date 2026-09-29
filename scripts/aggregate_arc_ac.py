"""Aggregate actor-critic litmus / Stage-B results (arc-proposals TASK-20260929-016).
  python scripts/aggregate_arc_ac.py output_remote/arc_ac_litmus_v1 [output_remote/... ] [--out summary.md]
Per (lr, arm): final sampled-policy mean return and crafter score over seeds (256 eval episodes each), plus per-seed values.
Reference lines (quoted from arc.tex sec.2.5, REINFORCE, 12,800 episodes, transformer_branch d=256): oracle_act 3.32+-0.06, base 3.25+-0.14.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

REF = "REINFORCE reference (arc.tex 2.5, 12,800 eps): oracle_act 3.32 ± 0.06 vs base 3.25 ± 0.14 (last-3-eval return)"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("roots", nargs="+"); ap.add_argument("--out")
    a = ap.parse_args(); g = defaultdict(list); greedy = []
    for root in a.roots:
        for f in glob.glob(os.path.join(root, "**", "*.result.json"), recursive=True):
            d = json.load(open(f)); c = d["config"]; arm = d["tag"].split("__")[1]
            g[(c["lr"], arm)].append((c["seed"], d["final_sampled"]["mean_return"], d["final_sampled"]["crafter_score"],
                                      d["final_greedy"]["mean_return"], d["random_policy"]["mean_return"], c["episodes"]))
        for f in glob.glob(os.path.join(root, "**", "greedy1_reference.json"), recursive=True):
            greedy.append(json.load(open(f)))
    out = [f"{REF}\n"]
    if greedy: out.append(f"Trivial reference policy 'argmax of the leaked 1-step reward': return {greedy[0]['mean_return']:.2f}, crafter {greedy[0]['crafter_score']:.2f}\n")
    out += ["| lr | arm | n | return (sampled) mean ± sd | per seed | crafter mean | greedy return | random |", "|---|---|---|---|---|---|---|---|"]
    for (lr, arm), v in sorted(g.items()):
        v.sort(); r = np.array([x[1] for x in v]); cr = np.array([x[2] for x in v])
        out.append(f"| {lr:g} | {arm} | {len(v)} | {r.mean():.2f} ± {r.std(ddof=1) if len(r) > 1 else 0:.2f} | "
                   f"{', '.join(f'{x:.2f}' for x in r)} | {cr.mean():.2f} | {np.mean([x[3] for x in v]):.2f} | {np.mean([x[4] for x in v]):.2f} |")
    for lr in sorted({k[0] for k in g}):
        if (lr, "base") in g and (lr, "oracle_act") in g:
            b = np.array([x[1] for x in g[(lr, "base")]]); o = np.array([x[1] for x in g[(lr, "oracle_act")]])
            out.append(f"\nlr={lr:g}: oracle_act - base = {o.mean() - b.mean():+.2f}; seed ranges "
                       f"{'NON-overlapping' if o.min() > b.max() else 'overlapping'} (oracle_act min {o.min():.2f} vs base max {b.max():.2f})")
    text = "\n".join(out); print(text)
    if a.out: open(a.out, "w", encoding="utf-8").write(text + "\n")

if __name__ == "__main__": main()
