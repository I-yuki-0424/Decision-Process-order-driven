"""Plot the two 'fair, matched-budget, corrected-evaluation-protocol' architecture comparisons
produced this session (no synthetic data, all numbers read from real committed result files):

Fig 1: candidate architectures (transformer_branch, variant_5_1..5_4, mdp_branch) at the SAME
       matched budget (lr=0.001, 250 updates, 2000-episode cap, T=200, shared random-policy
       baseline) from output/experiments/2026-09-25_convergence/local, with crafter_score
       corrected for the auto-reset achievement leak (TASK-20260928-015 item 3 fix) via
       output/experiments/reeval_v2_summary.json (eval-only re-eval of the real saved checkpoint,
       no retraining).

Fig 2: ChunkPPO actor choice (MLP vs Transformer) at the SAME matched budget (num_envs=1024,
       15 updates, seeds 0/1) from output/latency_chunking/kaggle_v2/, under metric_protocol v2
       (unbiased first-episode stochastic eval_return, TASK-20260928-015 items 8/9 fix).
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "output/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Fig 1: candidate architectures, matched budget, corrected crafter_score
# ---------------------------------------------------------------------------
reeval = json.load(open("output/experiments/reeval_v2_summary.json", encoding="utf-8"))
conv = [r for r in reeval if r["group"] == "2026-09-25_convergence/local"]
conv.sort(key=lambda r: r["candidate"])

tags = [r["tag"] for r in conv]
candidates = [r["candidate"] for r in conv]
returns = [r["final_sampled"]["mean_return"] for r in conv]
crafter = [r["final_sampled"]["crafter_score"] for r in conv]
rand_return = conv[0]["random_policy"]["mean_return"]
rand_crafter = conv[0]["random_policy"]["crafter_score"]

cand_colors = {
    "mdp_branch": "#8e44ad",
    "transformer_branch": "#2980b9",
    "variant_5_1": "#c0392b",
    "variant_5_2": "#27ae60",
    "variant_5_3": "#d35400",
    "variant_5_4": "#7f8c8d",
}
colors = [cand_colors[c] for c in candidates]
labels = [t.replace("__", "\n") for t in tags]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
x = np.arange(len(tags))
bars = ax.bar(x, returns, color=colors)
ax.axhline(rand_return, color="black", linestyle="--", linewidth=1.5, label=f"random policy ({rand_return:.2f})")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("mean_return (trained, sampled policy)")
ax.set_title("Return -- unaffected by the auto-reset leak\n(matched budget: lr=1e-3, 2000-episode cap, T=200)")
ax.legend(fontsize=8)
ax.axhline(0, color="#cccccc", linewidth=0.8)

ax = axes[1]
bars = ax.bar(x, crafter, color=colors)
ax.axhline(rand_crafter, color="black", linestyle="--", linewidth=1.5, label=f"random policy ({rand_crafter:.2f})")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("Crafter score (corrected, masked_achievements)")
ax.set_title("Crafter score -- CORRECTED for the auto-reset leak\n(eval-only re-eval of the real saved checkpoint, TASK-015 item 3)")
ax.legend(fontsize=8)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cand_colors.values()]
fig.legend(handles, cand_colors.keys(), loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04), fontsize=9, title="architecture")
fig.suptitle("Candidate architectures on Craftax-Classic -- matched-budget, corrected-evaluation comparison", y=1.1, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fair_candidate_architecture_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2: ChunkPPO actor (MLP vs Transformer), matched Kaggle budget, protocol v2
# ---------------------------------------------------------------------------
runs = {
    ("MLP", "s0"): "output/latency_chunking/kaggle_v2/craftax_calibration/s0/summary.json",
    ("MLP", "s1"): "output/latency_chunking/kaggle_v2/craftax_calibration/s1/summary.json",
    ("Transformer", "s0"): "output/latency_chunking/kaggle_v2/craftax_calibration_transformer/s0/summary.json",
    ("Transformer", "s1"): "output/latency_chunking/kaggle_v2/craftax_calibration_transformer/s1/summary.json",
}
data = {k: json.load(open(v, encoding="utf-8")) for k, v in runs.items()}

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
actor_colors = {"MLP": "#2980b9", "Transformer": "#c0392b"}

ax = axes[0]
for (actor, seed), d in data.items():
    recs = d["records"]
    updates = [r["update"] for r in recs]
    tr = [r["train_return_all"] for r in recs]
    ax.plot(updates, tr, color=actor_colors[actor], linestyle="-" if seed == "s0" else "--",
             marker="o", markersize=3, label=f"{actor} {seed}")
ax.set_xlabel("PPO update")
ax.set_ylabel("train_return_all (mean return, all completed episodes)")
ax.set_title("Training-return curve\n(num_envs=1024, matched budget)")
ax.legend(fontsize=8)

ax = axes[1]
for (actor, seed), d in data.items():
    recs = [r for r in d["records"] if "eval_return" in r]
    updates = [r["update"] for r in recs]
    ev = [r["eval_return"] for r in recs]
    ax.plot(updates, ev, color=actor_colors[actor], linestyle="-" if seed == "s0" else "--",
             marker="o", markersize=5, label=f"{actor} {seed}")
ax.set_xlabel("PPO update")
ax.set_ylabel("eval_return (stochastic, first-episode, uncensored)")
ax.set_title("Eval-return curve -- CORRECTED metric_protocol v2\n(unbiased vs. old greedy/windowed estimator)")
ax.legend(fontsize=8)

ax = axes[2]
actors = ["MLP", "Transformer"]
x = np.arange(len(actors))
width = 0.35
final_stoch = {a: [data[(a, s)]["records"][-1]["eval_return"] for s in ("s0", "s1")] for a in actors}
final_greedy = {a: [data[(a, s)]["records"][-1]["eval_return_greedy"] for s in ("s0", "s1")] for a in actors}
stoch_mean = [np.mean(final_stoch[a]) for a in actors]
stoch_err = [np.std(final_stoch[a]) for a in actors]
greedy_mean = [np.mean(final_greedy[a]) for a in actors]
greedy_err = [np.std(final_greedy[a]) for a in actors]
ax.bar(x - width / 2, stoch_mean, width, yerr=stoch_err, capsize=4, label="stochastic (headline)",
        color=[actor_colors[a] for a in actors])
ax.bar(x + width / 2, greedy_mean, width, yerr=greedy_err, capsize=4, label="greedy", alpha=0.5,
        color=[actor_colors[a] for a in actors])
for i, a in enumerate(actors):
    for s_i, s in enumerate(("s0", "s1")):
        ax.scatter(x[i] - width / 2, final_stoch[a][s_i], color="black", zorder=5, s=15)
        ax.scatter(x[i] + width / 2, final_greedy[a][s_i], color="black", zorder=5, s=15)
ax.set_xticks(x)
ax.set_xticklabels(actors)
ax.set_ylabel("final eval_return (update 15)")
ax.set_title("Final matched-budget comparison\n(mean +/- std over seeds 0,1; dots = individual seeds)")
ax.legend(fontsize=8)

fig.suptitle("ChunkPPO actor architecture: MLP vs Transformer -- matched-budget (num_envs=1024, 15 updates), corrected eval protocol v2", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT_DIR, "fair_actor_architecture_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("Wrote:")
print(" ", os.path.join(OUT_DIR, "fair_candidate_architecture_comparison.png"))
print(" ", os.path.join(OUT_DIR, "fair_actor_architecture_comparison.png"))
