"""Figures + number dump for docs/Latex/Result.tex (arc-proposals TASK-20260929-016).

Reads the raw result JSONs produced by the arc-proposals runs (branch claude/arc-design-proposals-training-3a3ac9, output/arc_proposals/)
and writes PNGs into docs/Latex/figures/ (prefix res_). Every plotted number is computed here from those files.

  python scripts/make_result_plots.py --data-root <checkout that has output/arc_proposals> --fig-dir docs/Latex/figures
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARM_COL = {"ignore": "#8c8c8c", "augment": "#1f77b4", "wm": "#d62728", "wm_passive": "#ff7f0e", "oracle": "#2ca02c"}


def jl(path):
    return json.load(open(path, encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ structure-as-input
def fig_structure(root, fig_dir):
    rows = []
    for d in ("planning", "planning2"):
        f = f"{root}/structure_input/{d}/structure_planning_results.json"
        if os.path.exists(f):
            rows += jl(f)
    ceil = [r["ret"] for r in rows if r["arm"] == "true_model_mpc"][0]
    g = defaultdict(list)
    for r in rows:
        if r["arm"] != "true_model_mpc":
            g[(r["arm"], r["n_traj"])].append(r["ret"])
    groups = [
        ("mlp", "MLP, no structure", "#7f7f7f"), ("hn", "no feature (hn)", "#7f7f7f"), ("feat:zero", "zero feature", "#7f7f7f"),
        ("feat:smallangle", "small-angle q²/2 (proposal)", "#d62728"), ("feat:scaled", "1.5 q² (wrong const.)", "#d62728"),
        ("feat:true", "true -cos q (oracle)", "#8c1c1c"),
        ("feat:abs", "|q| (generic)", "#2ca02c"), ("feat:cubic", "q³/3 (generic)", "#2ca02c"), ("feat:poly4", "q⁴/4 (generic)", "#2ca02c"),
        ("feat:q2q4", "[q²/2, q⁴/4] (generic)", "#2ca02c"),
        ("feat:hifreq", "sin(5q+.7) (bounded)", "#9467bd"), ("feat:cos2", "-cos 2q (bounded)", "#9467bd"),
        ("skip0:smallangle", "skip gate init 0 + preset", "#ff7f0e"), ("skip1:smallangle", "skip gate init 1 + preset", "#ff7f0e"),
        ("hard:smallangle", "hard term, preset", "#ff7f0e"), ("hard:poly4", "hard term, q⁴/4", "#ff7f0e")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 6.2), sharey=True)
    for ax, n in zip(axs, [8, 32, 128]):
        for i, (arm, lab, col) in enumerate(groups):
            v = np.array(g.get((arm, n), []))
            if len(v) == 0:
                continue
            ax.barh(i, v.mean() + 600, left=-600, color=col, alpha=.85)
            ax.errorbar(v.mean(), i, xerr=v.std(ddof=1) if len(v) > 1 else 0, color="k", capsize=2, lw=1)
            ax.text(-596, i, f" n={len(v)}", va="center", fontsize=7, color="w")
        ax.axvline(ceil, color="k", ls="--", lw=1)
        ax.set_xlim(-600, -180)
        ax.set_title(f"N = {n} training trajectories")
        ax.set_xlabel("MPC return in the true environment (higher is better)")
        ax.grid(axis="x", alpha=.3)
    axs[0].set_yticks(range(len(groups)))
    axs[0].set_yticklabels([g_[1] for g_ in groups], fontsize=8)
    axs[0].invert_yaxis()
    fig.suptitle(f"Pendulum swing-up: feature handed to the learned potential (bars = mean, whiskers = sd; dashed = true-model MPC ceiling {ceil:.1f})")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/res_structure_planning.png", dpi=140)
    plt.close(fig)

    rows = []
    for d in ("prediction", "prediction2"):
        f = f"{root}/structure_input/{d}/structure_prediction_results.json"
        if os.path.exists(f):
            rows += [r for r in jl(f) if "error" not in r]
    tasks = ["kepler_pert", "kepler_wrongmu", "fall_drag", "pendulum"]
    cols = [("hn", "none", "no preset (hn)", "#7f7f7f"), ("mlp", "none", "MLP", "#bcbd22"), ("phys_feat", "preset", "feature: preset", "#d62728"),
            ("phys_feat", "harmonic", "feature: generic (quadratic; q^4 on pendulum)", "#2ca02c"), ("phys_feat", "zero", "feature: zero", "#aaaaaa"),
            ("skip_feat", "preset", "gate init 1 + preset", "#ff7f0e"), ("skip_feat0", "preset", "gate init 0 + preset", "#e6a15c"),
            ("phys_hn", "preset", "hard term: preset", "#9467bd")]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.8))
    for ax, t in zip(axs, tasks):
        for i, (arm, var, lab, col) in enumerate(cols):
            v = [r["ood"]["rmse_mean"] for r in rows if r["task"] == t and r["arm"] == arm and r["variant"] == var]
            if not v and var == "harmonic":
                v = [r["ood"]["rmse_mean"] for r in rows if r["task"] == t and r["arm"] == arm and r["variant"] == "poly4"]
            v = [x for x in v if np.isfinite(x)]
            if not v:
                continue
            ax.bar(i, np.median(v), color=col, label=lab if t == tasks[0] else None)
            ax.plot([i, i], [min(v), max(v)], color="k", lw=1)
        ax.set_yscale("log")
        ax.set_title(t, fontsize=9)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=.3)
    axs[0].set_ylabel("OOD rollout nRMSE (median, min-max; log)")
    fig.legend(loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, 1))
    fig.savefig(f"{fig_dir}/res_structure_prediction.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ actor-critic
def load_ac(root):
    runs = {}
    for f in glob.glob(f"{root}/ac_*/**/*.result.json", recursive=True):
        d = jl(f)
        c = d["config"]
        tag = d["tag"]
        arm = tag.split("__")[1] + ("_big" if "_big" in tag else "")
        runs[(arm, c["lr"], c["episodes"], c["seed"])] = d
    return runs


def fig_ac(root, fig_dir):
    runs = load_ac(root)
    g1 = jl(f"{root}/ac_litmus_v1/lr_job0/greedy1_reference.json")["mean_return"]
    rnd = float(np.mean([d["random_policy"]["mean_return"] for d in runs.values()]))
    lab = {"base": "base", "oracle_act": "oracle_act\n(leaked)", "oracle": "oracle\n(true noop\nfuture)", "wm_full": "wm_full", "wm_untrain": "wm_untrain\n(random)",
           "wm_act": "wm_act\n(learned)", "wm_act_big": "wm_act\nbig", "wm_act_untrain": "wm_act_untrain\n(random)"}
    col = {"base": "#7f7f7f", "oracle_act": "#d62728", "oracle": "#2ca02c", "wm_full": "#1f77b4", "wm_untrain": "#aec7e8", "wm_act": "#ff7f0e",
           "wm_act_big": "#e6550d", "wm_act_untrain": "#fdd0a2"}
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, eps, title in ((axs[0], 12800, "12.8k episodes (200 updates)"), (axs[1], 49920, "49.9k episodes (780 updates)")):
        order = [a for a in lab if any(k[0] == a and k[1] == 1e-3 and k[2] == eps for k in runs)]
        for i, a in enumerate(order):
            v = [d["final_sampled"]["mean_return"] for k, d in runs.items() if k[0] == a and k[1] == 1e-3 and k[2] == eps]
            ax.bar(i, np.mean(v), color=col[a], alpha=.8)
            ax.scatter(np.full(len(v), i) + np.linspace(-.18, .18, len(v)), v, color="k", s=10, zorder=3)
            ax.text(i, 0.15, f"n={len(v)}", ha="center", fontsize=7)
        ax.axhline(g1, color="b", ls=":", lw=1)
        ax.text(len(order) - .5, g1 + .05, "1-step-greedy on leaked reward", fontsize=7, ha="right", color="b")
        ax.axhline(rnd, color="gray", ls="--", lw=1)
        ax.text(len(order) - .5, rnd + .05, "random policy", fontsize=7, ha="right", color="gray")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([lab[a] for a in order], fontsize=7)
        ax.set_title(f"Actor-critic, lr 1e-3, {title}")
        ax.set_ylabel("final mean return (sampled policy)")
        ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/res_ac_final_returns.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for a in ["base", "oracle_act", "wm_act", "wm_act_big", "wm_act_untrain", "wm_untrain", "wm_full", "oracle"]:
        hs = [d["eval_history"] for k, d in runs.items() if k[0] == a and k[1] == 1e-3 and k[2] == 49920]
        if not hs:
            continue
        L = min(len(h) for h in hs)
        x = [h["episodes"] for h in hs[0][:L]]
        y = np.array([[h["mean_return"] for h in hh[:L]] for hh in hs])
        ax.plot(x, y.mean(0), color=col[a], label=f"{a} (n={len(hs)})", lw=2 if a in ("base", "oracle_act") else 1.2)
    ax.set_xlabel("training episodes")
    ax.set_ylabel("eval mean return (64 episodes, sampled)")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    ax.set_title("Actor-critic learning curves, lr 1e-3 (mean over seeds)")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/res_ac_curves.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    combos = [(3e-4, 12800), (1e-3, 12800), (3e-3, 12800), (1e-3, 49920)]
    w = .35
    for j, (a, c) in enumerate((("base", "#7f7f7f"), ("oracle_act", "#d62728"))):
        for i, (lr, eps) in enumerate(combos):
            v = [d["final_sampled"]["mean_return"] for k, d in runs.items() if k[0] == a and k[1] == lr and k[2] == eps]
            if not v:
                continue
            ax.bar(i + (j - .5) * w, np.mean(v), w, color=c, label=a if i == 0 else None)
            ax.scatter(np.full(len(v), i + (j - .5) * w), v, color="k", s=8, zorder=3)
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels([f"lr {lr:g}\n{eps//1000}k eps" for lr, eps in combos])
    ax.set_ylabel("final mean return")
    ax.axhline(g1, color="b", ls=":", lw=1)
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    ax.set_title("oracle_act litmus vs learning rate / budget")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/res_ac_lr.png", dpi=140)
    plt.close(fig)
    return runs


# ------------------------------------------------------------------------------------------------ latency
def load_lat(paths):
    rows, seen = [], set()
    for p in paths:
        for f in glob.glob(f"{p}/**/summary.json", recursive=True):
            d = jl(f)
            c = d["cfg"]
            ev = [r for r in d["records"] if "eval_return" in r]
            if not ev:
                continue
            k = (c["env"], c["delta"], c["arm"], c["seed"], c["total_env_ticks"], round(ev[-1]["eval_return"], 9))
            if k in seen:
                continue
            seen.add(k)
            rows.append(dict(env=c["env"], delta=c["delta"], arm=c["arm"], seed=c["seed"], ticks=c["total_env_ticks"], eval=ev[-1]["eval_return"]))
    return rows


def lat_panel(ax, rows, env, ticks, title):
    arms = ["ignore", "augment", "wm", "wm_passive", "oracle"]
    ds = sorted({r["delta"] for r in rows if r["env"] == env and r["ticks"] == ticks})
    for k, a in enumerate(arms):
        xs, ms, ss = [], [], []
        for d in ds:
            v = [r["eval"] for r in rows if r["env"] == env and r["ticks"] == ticks and r["delta"] == d and (r["arm"] == a or (d == 0 and r["arm"] == "augment"))]
            if v:
                xs.append(d + (k - 2) * .07)
                ms.append(np.mean(v))
                ss.append(np.std(v, ddof=1) if len(v) > 1 else 0)
        if xs:
            ax.errorbar(xs, ms, yerr=ss, color=ARM_COL[a], marker="o", ms=4, capsize=2, lw=1.4, label=a)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("latency δ (ticks)")
    ax.grid(alpha=.3)
    ax.set_xticks(ds)


def fig_latency(root, fig_dir):
    L = f"{root}/latency"
    A = load_lat([f"{L}/tunedA_2M", f"{L}/kaggle_tunedA_2M", f"{L}/tunedA_2M_extra"])
    B = load_lat([f"{L}/tunedB_2M", f"{L}/tunedB_2M_d2"])
    T10 = load_lat([f"{L}/kaggle_10M", f"{L}/kaggle_10M_d24"])
    D = load_lat([f"{L}/main_1M"])
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
    lat_panel(axs[0], D, "intercept", 1_000_000, "default config, 1M ticks (6 seeds)")
    lat_panel(axs[1], A, "intercept", 2_000_000, "config A (lr 1e-3), 2M ticks (8-12 seeds)")
    lat_panel(axs[2], B, "intercept", 2_000_000, "config B (lr 3e-4), 2M ticks (6 seeds)")
    lat_panel(axs[3], T10, "intercept", 10_000_000, "config A, 10M ticks (3 seeds)")
    axs[0].set_ylabel("intercept: eval return")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/res_latency_intercept.png", dpi=140)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    lat_panel(axs[0], D, "pendulum_goal", 1_000_000, "default config, 1M ticks (1 seed)")
    lat_panel(axs[1], A, "pendulum_goal", 2_000_000, "config A, 2M ticks (8 seeds)")
    lat_panel(axs[2], T10, "pendulum_goal", 10_000_000, "config A, 10M ticks (3 seeds)")
    axs[0].set_ylabel("pendulum_goal: eval return")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/res_latency_pendulum.png", dpi=140)
    plt.close(fig)
    for name, rows in (("A", A), ("B", B), ("10M", T10), ("default1M", D)):
        print(f"== latency {name}")
        for env in ("intercept", "pendulum_goal"):
            for tk in sorted({r["ticks"] for r in rows if r["env"] == env}):
                for d in sorted({r["delta"] for r in rows if r["env"] == env and r["ticks"] == tk}):
                    cells = []
                    for a in ["ignore", "augment", "wm", "wm_passive", "oracle"]:
                        v = [r["eval"] for r in rows if r["env"] == env and r["ticks"] == tk and r["delta"] == d and r["arm"] == a]
                        cells.append(f"{a}:{np.mean(v):+.2f}±{np.std(v, ddof=1) if len(v) > 1 else 0:.2f}(n={len(v)})" if v else f"{a}:-")
                    print(f"{env} {tk} d={d} " + " ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="checkout containing output/arc_proposals")
    ap.add_argument("--fig-dir", default="docs/Latex/figures")
    a = ap.parse_args()
    root = os.path.join(a.data_root, "output", "arc_proposals")
    os.makedirs(a.fig_dir, exist_ok=True)
    fig_structure(root, a.fig_dir)
    runs = fig_ac(root, a.fig_dir)
    fig_latency(root, a.fig_dir)
    print("== actor-critic (lr, episodes, arm): n, mean, sd")
    agg = defaultdict(list)
    for (arm, lr, eps, seed), d in runs.items():
        agg[(lr, eps, arm)].append(d["final_sampled"]["mean_return"])
    for k, v in sorted(agg.items()):
        print(k, len(v), round(float(np.mean(v)), 2), round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else "")
