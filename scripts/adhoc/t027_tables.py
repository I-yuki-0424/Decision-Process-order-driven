"""TASK-027: markdown tables of the screening stages (reads only output/phase1/push_t026/<stage>/*.json)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run_t026_sweep as sw  # noqa: E402


def table(stage, ref=None):
    rows = sw.load_stage(stage)
    out = ["| config | n | params_total | reward | score | r+s | per-seed reward |", "|---|---|---|---|---|---|---|"]
    for n, v in rows.items():
        rw, sc = np.array([x["reward"] for x in v]), np.array([x["score"] for x in v])
        se = lambda a: a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
        out.append(f"| {n.replace('@kag', '')} | {len(v)} | {v[0]['params']:,} | {rw.mean():.2f} +- {se(rw):.2f} | {sc.mean():.2f} +- {se(sc):.2f} | {rw.mean() + sc.mean():.1f} | "
                   + " ".join(f"{x['reward']:.1f}" for x in sorted(v, key=lambda x: x['seed'])) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    for st, title in (("R1", "R1 (seeds 4000-4003): lr shape, width, entropy, lambda, health coefficient, Truck"), ("R2", "R2 (5000-5003): latent prediction"),
                      ("R3", "R3 (5000-5003): health coefficient 0 stacked"), ("R4", "R4 (5000-5003): coefficient sensitivity, rs0 + entropy/width, Truck"),
                      ("R5", "R5 (5000-5003): LP combinations"), ("R5c", "R5c (6000-6003): confirmation of LP"), ("R6", "R6 (5000-5003): full stacks"),
                      ("R7", "R7 (5000-5003): width round")):
        if os.path.isdir(f"{sw.ROOT}/{st}"):
            print(f"\n**{title}**\n\n{table(st)}")
