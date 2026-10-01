"""Bar plots (mean ± SE over seeds, individual seeds as dots) of reward_pct and score_pct per arm.

  python scripts/plot_phase1.py out.png output/phase1/final_100k output/phase1/final_1000k output/phase1/final_1000k_derived
Reference lines: the untrained init policy (output/phase1/reference) and, for 1M, the G1.0 thresholds (0.9 x MFRL_4M_reimpl, Dedieu 2025 Table 1).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out, dirs = sys.argv[1], sys.argv[2:]
sets = []
for d in dirs:
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json"))) if not f.endswith("selection.json")]
    sets.append((os.path.basename(d), rows))
fig, axes = plt.subplots(len(sets), 2, figsize=(13, 3.6 * len(sets)), squeeze=False)
ref = json.load(open("output/phase1/reference/untrained_init_mlp.json"))
for i, (name, rows) in enumerate(sets):
    rows = sorted(rows, key=lambda r: r["reward_pct_mean"])
    for j, (key, label, g10) in enumerate((("reward_pct", "reward_pct", 42.7), ("score_pct", "score_pct", 9.6))):
        ax = axes[i][j]
        vals = [[s[key] for s in r["per_seed"]] for r in rows]
        means = [np.mean(v) for v in vals]
        ses = [np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for v in vals]
        ax.barh(range(len(rows)), means, xerr=ses, color="#4c78a8", alpha=0.85)
        for k, v in enumerate(vals):
            ax.plot(v, [k] * len(v), "k.", ms=3, alpha=0.6)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["arm"] for r in rows], fontsize=8)
        ax.axvline(ref[key + "_mean"], color="gray", ls=":", label="untrained init")
        if "1000k" in name:
            ax.axvline(g10, color="crimson", ls="--", label="G1.0 threshold")
        ax.set_title(f"{name}: {label} (mean ± SE, dots = seeds)", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig(out, dpi=130)
print("wrote", out)
