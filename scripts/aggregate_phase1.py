"""Aggregate final Phase-1 result files into a markdown table with per-seed values and pairwise differences.

  python scripts/aggregate_phase1.py output/phase1/final_100k [--ref ppo_mlp] [--md out.md]
Pairwise column: difference in mean reward_pct / score_pct vs the reference arm, with the Welch standard error of the difference and
the 'surpass' test (own mean - 2 SE > reference mean, roadmap definition, applied to the INTERNAL reference; not a paper comparison).
"""
import argparse
import glob
import json
import os

import numpy as np


def load(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(f) in ("selection.json",):
            continue
        r = json.load(open(f))
        rows[os.path.basename(f).split("__")[0]] = r
    return rows


def se(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref", default="ppo_mlp")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    rows = load(a.dir)
    ref = rows.get(a.ref)
    out = [f"| arm | params_total | params_deployed | env_steps | seeds | reward_pct (mean ± SE) | score_pct (mean ± SE) | Δreward vs {a.ref} | Δscore vs {a.ref} | surpass {a.ref}? |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        rw = [s["reward_pct"] for s in r["per_seed"]]
        sc = [s["score_pct"] for s in r["per_seed"]]
        d = ""
        if ref is not None and name != a.ref:
            rr = [s["reward_pct"] for s in ref["per_seed"]]
            rs = [s["score_pct"] for s in ref["per_seed"]]
            dr, ds = np.mean(rw) - np.mean(rr), np.mean(sc) - np.mean(rs)
            ser, ses = np.hypot(se(rw), se(rr)), np.hypot(se(sc), se(rs))
            sur = (np.mean(rw) - 2 * se(rw) > np.mean(rr)) and (np.mean(sc) - 2 * se(sc) > np.mean(rs))
            d = f"{dr:+.2f} ± {ser:.2f} | {ds:+.2f} ± {ses:.2f} | {'yes' if sur else 'no'}"
        else:
            d = "– | – | –"
        out.append(f"| {r['arm']} | {r['params_total']:,} | {r['params_deployed']:,} | {r['env_steps_total']:,} | {len(rw)} | "
                   f"{np.mean(rw):.2f} ± {se(rw):.2f} | {np.mean(sc):.2f} ± {se(sc):.2f} | {d} |")
    out.append("")
    out.append("Per-seed reward_pct / score_pct:")
    for name, r in rows.items():
        out.append(f"- {r['arm']} (seeds {r['seeds']}): reward " + ", ".join(f"{s['reward_pct']:.1f}" for s in r["per_seed"])
                   + " | score " + ", ".join(f"{s['score_pct']:.2f}" for s in r["per_seed"]))
    text = "\n".join(out)
    print(text)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
