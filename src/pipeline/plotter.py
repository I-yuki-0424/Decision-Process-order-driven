"""
Plotting and Visualization Utilities for 5th-Idea Hierarchical Transformer Benchmarks (|A|=2000).

Generates high-resolution comparative graphics tagged with Run Sequence IDs (Run-Seq: #002):
1. 5th-Idea Hierarchical (Toggle ON) vs Flat (Toggle OFF) vs MDP Baseline (|A|=2000)
2. Inference Latency & Expansion Efficiency Bar Charts
"""

import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np


def plot_full_benchmark_results(
    results: List[dict],
    output_dir: str = "output/plots",
    run_seq: str = "Run-Seq: #001",
):
    """Generate comprehensive comparison plot suite from benchmark results list."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = [r["model_name"] for r in results]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    # Goal Success Rate (%)
    success_rates = [r["success_rate"] * 100 for r in results]
    bars1 = axes[0, 0].bar(names, success_rates, color=colors, alpha=0.85, edgecolor="black")
    axes[0, 0].axhline(80, color="red", linestyle="--", label="1st-Idea Goal (80%)")
    axes[0, 0].set_title(f"Goal Success Rate (%) [{run_seq}]", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Success Rate (%)")
    axes[0, 0].set_ylim(0, 110)
    axes[0, 0].legend()
    for bar in bars1:
        yval = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    # Average Decision Steps
    avg_steps = [r["avg_steps"] for r in results]
    bars2 = axes[0, 1].bar(names, avg_steps, color=colors, alpha=0.85, edgecolor="black")
    axes[0, 1].set_title(f"Average Process Steps (~100 steps) [{run_seq}]", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Steps")
    for bar in bars2:
        yval = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}", ha='center', va='bottom', fontweight='bold')

    # Goal Progress Rate (%)
    progress_rates = [r["avg_progress_rate"] * 100 for r in results]
    bars3 = axes[1, 0].bar(names, progress_rates, color=colors, alpha=0.85, edgecolor="black")
    axes[1, 0].set_title(f"Average Final Progress Rate (%) [{run_seq}]", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Progress Rate (%)")
    axes[1, 0].set_ylim(0, 110)
    for bar in bars3:
        yval = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    # Exposure Bias Resilience (%)
    resilience = [r["exposure_bias_resilience"] * 100 for r in results]
    bars4 = axes[1, 1].bar(names, resilience, color=colors, alpha=0.85, edgecolor="black")
    axes[1, 1].set_title(f"Exposure Bias Resilience (%) [{run_seq}]", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Resilience Score (%)")
    axes[1, 1].set_ylim(0, 110)
    for bar in bars4:
        yval = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    for ax in axes.flat:
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "benchmark_metrics_summary_seq001.png")
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

    # 1. Final Progress Rate (%) Comparison (|A|=2000)
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

    # 2. Per-step Inference Latency (ms/step) Comparison (|A|=2000)
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
