"""
Plotting and Visualization Utilities for Craftax-Classic RL Benchmarks (Run-Seq: #004).

Generates high-resolution comparative graphics tagged with Run Sequence IDs:
1. Crafter Score S_crafter & Achievement Count Comparisons
2. Detailed 22 Achievement Unlock Percentage Breakdowns
3. Off-Policy & Hierarchical Transformer Summaries
"""

import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

# ACHIEVEMENT_NAMES lives in src/environment/craftax_env_adapter.py; imported
# lazily inside plot_candidate_achievement_breakdown to avoid a hard
# dependency on `craftax` for callers who only use the other plot functions.


def plot_craftax_benchmark_results(
    results: List[dict],
    output_dir: str = "output/plots",
    run_seq: str = "Run-Seq: #005",
):
    """Generate high-resolution visual plots for Craftax-Classic RL Benchmark (1,000 Episodes)."""
    os.makedirs(output_dir, exist_ok=True)

    names = [r["model_name"] for r in results]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Chart 1: Crafter Score & Average Unlocked Achievements
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    crafter_scores = [r["crafter_score"] for r in results]
    bars1 = axes[0].bar(names, crafter_scores, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_title(f"Crafter Score S_crafter (%) [{run_seq}]", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Crafter Score (%)", fontsize=10)
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

    avg_unlocked = [r["avg_unlocked_count"] for r in results]
    bars2 = axes[1].bar(names, avg_unlocked, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_title(f"Avg Achievements Unlocked (out of 22) [{run_seq}]", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Unlocked Count", fontsize=10)
    axes[1].set_ylim(0, 24)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f} / 22", ha='center', va='bottom', fontweight='bold')

    for ax in axes:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)

    plt.tight_layout()
    plot_path1 = os.path.join(output_dir, "craftax_crafter_score_summary_seq005.png")
    plt.savefig(plot_path1, dpi=300)
    plt.close()


def plot_offpolicy_benchmark_results(
    results: List[dict],
    loss_history: List[float],
    output_dir: str = "output/plots",
    run_seq: str = "Run-Seq: #003",
):
    """Generate visual graphics for Off-Policy Learning & Abstraction Embedding E_abs benchmark."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(range(1, len(loss_history) + 1), loss_history, color="#d62728", linewidth=2.0, label="Off-Policy TD Loss L_TD(θ)")
    axes[0].set_title(f"Off-Policy Q-Learning Loss Convergence [{run_seq}]", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Training Steps", fontsize=10)
    axes[0].set_ylabel("TD Loss", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    names = [r["model_name"] for r in results]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    progress_rates = [r["avg_progress_rate"] * 100 for r in results]

    bars = axes[1].bar(names, progress_rates, color=colors, alpha=0.85, edgecolor="black")
    axes[1].axhline(80, color="red", linestyle="--", label="Goal Target (80%)")
    axes[1].set_title(f"Goal Progress Rate (%) across Off-Policy & E_abs Variants [{run_seq}]", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Progress Rate (%)", fontsize=10)
    axes[1].set_ylim(0, 110)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "offpolicy_loss_convergence_seq003.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()


def plot_predicted_transition_distribution(
    transitions_by_model: Dict[str, List[np.ndarray]],
    output_dir: str = "output/plots",
    run_seq: str = "Candidate-Suite",
):
    """Distribution of the real per-step predicted transition effect
    (DecisionVectorD.predicted_next_state) each candidate produced during a
    real evaluation run (src/pipeline/candidate_benchmark.py's
    evaluate_candidate_agent). Craftax has no ground-truth W_res (the
    synthetic DecisionProcessEnv's resource_effects matrix -- see
    src/environment/gymnax_decision_env.py around line 120 -- has no Craftax
    equivalent), so this plots what each candidate actually predicted the
    resource-effect vector to be, one histogram per candidate, flattened
    across resource dimensions and steps.
    """
    os.makedirs(output_dir, exist_ok=True)

    names = [n for n, t in transitions_by_model.items() if len(t) > 0]
    if not names:
        return

    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), squeeze=False)
    axes = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    for ax, name, color in zip(axes, names, colors):
        values = np.concatenate([np.ravel(np.asarray(v)) for v in transitions_by_model[name]])
        ax.hist(values, bins=40, color=color, alpha=0.85, edgecolor="black")
        ax.set_title(f"{name}\nPredicted Transition W", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted resource-effect value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"Predicted Next-State Transition Distribution (W) [{run_seq}]", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "candidate_predicted_transition_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()


def plot_candidate_achievement_breakdown(
    results: List[dict],
    output_dir: str = "output/plots",
    run_seq: str = "Candidate-Suite",
):
    """Per-achievement unlock-rate breakdown (all 22 Craftax achievements),
    one grouped horizontal bar chart per model, from real evaluation results
    (results[i]["achievement_unlock_rates"], computed in
    src/pipeline/craftax_benchmark.py / src/pipeline/candidate_benchmark.py's
    evaluate_*_agent). Complements plot_craftax_benchmark_results, which only
    plots the aggregated Crafter score and average unlocked count.
    """
    from src.environment.craftax_env_adapter import ACHIEVEMENT_NAMES

    os.makedirs(output_dir, exist_ok=True)
    if not results:
        return

    names = [r["model_name"] for r in results]
    num_achievements = len(ACHIEVEMENT_NAMES)
    y_pos = np.arange(num_achievements)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    fig, ax = plt.subplots(figsize=(10, max(6, num_achievements * 0.35)))
    bar_height = 0.8 / max(1, len(results))

    for i, (r, color) in enumerate(zip(results, colors)):
        rates = r["achievement_unlock_rates"]
        offsets = y_pos + i * bar_height - 0.4 + bar_height / 2
        ax.barh(offsets, rates, height=bar_height, color=color, alpha=0.85, edgecolor="black", label=r["model_name"])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ACHIEVEMENT_NAMES, fontsize=8)
    ax.set_xlabel("Unlock Rate (%)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_title(f"Per-Achievement Unlock Rate Breakdown (22 Achievements) [{run_seq}]", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "candidate_per_achievement_breakdown.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()


def plot_hierarchical_benchmark_results(
    results: List[dict],
    output_dir: str = "output/plots",
    run_seq: str = "Run-Seq: #002",
):
    """Generate high-resolution visual plots for 5th-Idea Hierarchical Benchmark (|A|=2000)."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = [r["model_name"] for r in results]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    progress_rates = [r["avg_progress_rate"] * 100 for r in results]
    bars1 = axes[0].bar(names, progress_rates, color=colors, alpha=0.85, edgecolor="black")
    axes[0].axhline(80, color="red", linestyle="--", label="Goal Target (80%)")
    axes[0].set_title(f"Goal Progress Rate (%) under |A| = 2000 [{run_seq}]", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Progress Rate (%)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 110)
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    latencies = [r["execution_ms_per_step"] for r in results]
    bars2 = axes[1].bar(names, latencies, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_title(f"Inference Latency (ms/step) under |A| = 2000 [{run_seq}]", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Execution Time (ms/step)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.2f} ms", ha='center', va='bottom', fontweight='bold')

    for ax in axes:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "hierarchical_vs_flat_scaling_seq002.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
