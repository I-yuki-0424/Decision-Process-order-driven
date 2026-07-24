"""
Plotting and Visualization Utilities for 4th-Idea Decision Transformer.

Generates plots for:
1. Training Loss & Metric Curves
2. Progress Rate Trajectories (Greedy vs. Beam Search)
3. Multi-dimensional Cost Trajectories
4. Noise Sensitivity & Recovery Rate Analysis
"""

import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np

from src.pipeline.evaluator import EvaluationResult


def create_plot_directory(output_dir: str = "output/plots") -> str:
    """Ensure plot destination directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_training_curves(
    total_losses: List[float],
    policy_losses: List[float],
    validity_losses: List[float],
    output_path: str = "output/plots/training_curves.png",
):
    """Plot and save training loss and metric progression."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    steps = np.arange(1, len(total_losses) + 1)
    ax1.plot(steps, total_losses, label="Total Loss", color="#1f77b4", linewidth=2)
    ax1.plot(steps, policy_losses, label="Policy CE Loss", color="#ff7f0e", linestyle="--")
    ax1.set_title("Training Loss Trajectory", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, validity_losses, label="Noise Validity BCE Loss", color="#2ca02c", linewidth=2)
    ax2.set_title("Noise Injection Recovery Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_evaluation_comparison(
    greedy_res: EvaluationResult,
    beam_res: EvaluationResult,
    output_path: str = "output/plots/evaluation_comparison.png",
):
    """Plot comparative progress rate and cost trajectories between Greedy and Beam Search."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Progress Rate Trajectory Comparison
    g_traj = np.array(greedy_res.progress_trajectories[0])
    b_traj = np.array(beam_res.progress_trajectories[0])

    ax1.plot(g_traj, label=f"3rd-Idea Greedy (Success: {greedy_res.success_rate:.0%})", color="#d62728", linestyle="--", linewidth=2)
    ax1.plot(b_traj, label=f"4th-Idea Beam Search (Success: {beam_res.success_rate:.0%})", color="#1f77b4", linewidth=2.5)
    ax1.axhline(0.80, color="gray", linestyle=":", label="1st-Idea Target (80%)")
    ax1.set_title("Goal Progress Rate over Steps", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Decision Step (N > 100)")
    ax1.set_ylabel("Progress Rate")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Multi-dimensional Cost Trajectory Comparison
    g_costs = np.cumsum(np.array(greedy_res.cost_trajectories[0]), axis=0)
    b_costs = np.cumsum(np.array(beam_res.cost_trajectories[0]), axis=0)

    for c in range(min(g_costs.shape[1], 4)):
        ax2.plot(g_costs[:, c], linestyle="--", alpha=0.6, label=f"Cost Channel {c+1} (Greedy)")
        ax2.plot(b_costs[:, c], linewidth=2, label=f"Cost Channel {c+1} (Beam)")

    ax2.set_title("Cumulative Multi-Dimensional Cost Trajectory", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Decision Step")
    ax2.set_ylabel("Accumulated Cost")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
