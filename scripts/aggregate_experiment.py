"""Aggregate candidate convergence results (local + kaggle) into split markdown reports + plots."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = sys.argv[1]
R = {}
for src in ("local", "kaggle"):
    for f in sorted(glob.glob(f"{E}/{src}/*.result.json")):
        r = json.load(open(f, encoding="utf-8")); r["source"] = src; R[r["tag"]] = r

def curve(r, key, w=10):
    rows = [json.loads(l) for l in open(f"{E}/{r['source']}/{r['tag']}.train_log.jsonl", encoding="utf-8")]
    y = np.array([x[key] for x in rows]); k = np.ones(w) / w
    return np.arange(1, len(y) + 1) * r["config"]["batch"], np.convolve(y, k, mode="valid"), rows

os.makedirs(f"{E}/plots", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
rnd = next(iter(R.values()))["random_policy"]["mean_return"]
for tag, r in R.items():
    x, y, _ = curve(r, "train_return"); ax[0].plot(x[9:], y, label=tag)
    x, y, _ = curve(r, "train_unlocked"); ax[1].plot(x[9:], y, label=tag)
ax[0].axhline(rnd, color="k", ls="--", label="uniform-random policy (eval)")
ax[0].set_title("Train episode return (10-update moving avg)"); ax[1].set_title("Train achievements unlocked / episode")
for a in ax: a.set_xlabel("episodes")
ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig(f"{E}/plots/learning_curves.png", dpi=120); plt.close()

tags = list(R)
fig, ax = plt.subplots(figsize=(11, 4.5)); w = 0.27; xs = np.arange(len(tags))
for i, (k, lab) in enumerate([("untrained_sampled", "untrained"), ("final_sampled", "trained (sampled)"), ("random_policy", "random")]):
    ax.bar(xs + (i - 1) * w, [R[t][k]["mean_return"] for t in tags], w, label=lab)
ax.set_xticks(xs); ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=7); ax.set_ylabel("mean eval return (64 eps, T=200)")
ax.legend(); plt.tight_layout(); plt.savefig(f"{E}/plots/final_eval_return.png", dpi=120); plt.close()

L = ["# Results table", "", "Eval: 64 episodes, T=200 steps, Craftax-Classic. Return = env reward sum. Crafter score from achievement rates.", "",
     "| run | src | params | episodes | wall s | first-10 train ret | last-10 train ret | eval ret (untrained→trained) | random ret | crafter (trained/greedy/random) | last entropy | skipped updates |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for t, r in R.items():
    _, _, rows = curve(r, "train_return")
    f10 = np.mean([x["train_return"] for x in rows[:10]]); l10 = np.mean([x["train_return"] for x in rows[-10:]])
    L.append(f"| {t} | {r['source']} | {r['n_params']} | {r['config']['episodes']} | {r['wall_s']:.0f} | {f10:.2f} | {l10:.2f} | "
             f"{r['untrained_sampled']['mean_return']:.2f}→{r['final_sampled']['mean_return']:.2f} | {r['random_policy']['mean_return']:.2f} | "
             f"{r['final_sampled']['crafter_score']:.2f}/{r['final_greedy']['crafter_score']:.2f}/{r['random_policy']['crafter_score']:.2f} | "
             f"{rows[-1]['entropy']:.2f} | {rows[-1].get('skipped_updates', 'n/a')} |")
open(f"{E}/results_table.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
