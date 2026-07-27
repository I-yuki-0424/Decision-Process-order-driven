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


def plot_craftax_benchmark_results(
    results: List[dict],
    output_dir: str = "output/plots",
    run_seq: str = "Run-Seq: #004",
):
    """Generate high-resolution visual plots for Craftax-Classic RL Benchmark."""
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
    plot_path1 = os.path.join(output_dir, "craftax_crafter_score_summary_seq004.png")
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
