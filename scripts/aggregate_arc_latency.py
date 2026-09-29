"""Aggregate ChunkPPO latency sweeps (arc-proposals TASK-20260929-016): final eval_return (protocol v2, first-episode stochastic) per
(env, delta, arm) over seeds, plus late-training train_return, and the decision-rule checks of DESIGN.md sec.4.

  python scripts/aggregate_arc_latency.py output/arc_proposals/latency/main_1M [more roots ...] --out output/arc_proposals/latency/summary_main_1M.md
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load(root, at_ticks=None):
    rows = []
    for f in glob.glob(os.path.join(root, "**", "summary.json"), recursive=True):
        d = json.load(open(f))
        c = d["cfg"]
        recs = d["records"]
        ev = [r for r in recs if "eval_return" in r]
        if not ev:
            continue
        last = ev[-1]
        settled = [r["train_return"] for r in recs if r.get("train_settled")][-5:]
        if at_ticks:  # learning-curve read-out: mean settled train_return within +-8% of `at_ticks` env ticks
            w = [r["train_return"] for r in recs if r.get("train_settled") and r["train_return"] == r["train_return"]
                 and abs(r["env_ticks"] - at_ticks) <= 0.08 * at_ticks]
            if not w:
                continue
            last = dict(last, eval_return=float(np.mean(w)))
            settled = [float(np.mean(w))]
        rows.append(dict(env=c["env"], delta=c["delta"], arm=c["arm"], seed=c["seed"], ticks=c["total_env_ticks"],
                         eval=last["eval_return"], eval_greedy=last.get("eval_return_greedy", np.nan),
                         cens=last.get("eval_censored_frac", np.nan), train=float(np.mean(settled)) if settled else np.nan,
                         fskill=last.get("forecast_skill", np.nan), file=f))
    return rows


def fmt(v):
    v = np.asarray(v, float)
    return f"{v.mean():8.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:5.2f} (n={len(v)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--at-ticks", type=int, default=None, help="replace eval by settled train_return around this many ticks")
    ap.add_argument("--metric", default="eval", choices=["eval", "train", "eval_greedy"])
    a = ap.parse_args()
    rows = [r for root in a.roots for r in load(root, a.at_ticks)]
    out = []
    for env in sorted({r["env"] for r in rows}):
        for ticks in sorted({r["ticks"] for r in rows if r["env"] == env}):
            g = defaultdict(list)
            for r in rows:
                if r["env"] == env and r["ticks"] == ticks:
                    g[(r["delta"], r["arm"])].append(r[a.metric])
            arms = ["ignore", "augment", "wm", "wm_passive", "oracle"]
            out.append(f"\n### {env}, {ticks:,} ticks, metric={a.metric} (mean ± sd over seeds)\n")
            out.append("| delta | " + " | ".join(arms) + " |")
            out.append("|---|" + "---|" * len(arms))
            for d in sorted({k[0] for k in g}):
                cells = [fmt(g[(d, arm)]) if (d, arm) in g else "-" if not (d == 0 and arm != "augment") else "(=augment)" for arm in arms]
                out.append(f"| {d} | " + " | ".join(cells) + " |")
            # decision rules
            out.append("")
            for d in sorted({k[0] for k in g}):
                if d == 0:
                    continue
                aug, wm, orc, ign = (np.array(g.get((d, x), [np.nan])) for x in ("augment", "wm", "oracle", "ignore"))
                if len(aug) > 1 and len(wm) > 1:
                    sep = "non-overlapping" if wm.min() > aug.max() else ("reversed" if wm.max() < aug.min() else "overlapping")
                    out.append(f"- delta={d}: wm-augment = {wm.mean() - aug.mean():+.2f} ({sep} seed ranges); "
                               f"oracle-augment = {orc.mean() - aug.mean():+.2f}; augment-ignore = {aug.mean() - ign.mean():+.2f}")
    text = "\n".join(out)
    print(text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
