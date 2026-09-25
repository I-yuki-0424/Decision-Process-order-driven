"""Summary plot for the Idea-6 study (Stage A error curves + Stage B training returns)."""
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = "output/experiments/2026-09-25_idea6"
os.makedirs(E + "/plots", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
r = json.load(open(E + "/stageA/worldmodel_result.json"))
h = np.arange(1, 9)
for k in ("persistence", "mlp", "hn"):
    ax[0].plot(h, r[k], marker="o", label=k)
ax[0].set_xlabel("steps ahead (noop)")
ax[0].set_ylabel("held-out MSE (std. units)")
ax[0].legend()
ax[0].set_title("Stage A: passive vitals prediction")
for f in sorted(glob.glob(E + "/stageB_*/*.train_log.jsonl")):
    tag = os.path.basename(f).split(".")[0]
    y = np.array([json.loads(line)["train_return"] for line in open(f)])
    ax[1].plot(np.convolve(y, np.ones(20) / 20, "valid"), label=tag.split("__", 1)[1], lw=1)
ax[1].axhline(1.46, color="k", ls="--", label="random")
ax[1].set_title("Stage B: train return (20-update avg)")
ax[1].set_xlabel("update (16 episodes each)")
ax[1].legend(fontsize=6)
plt.tight_layout()
plt.savefig(E + "/plots/idea6_summary.png", dpi=120)
