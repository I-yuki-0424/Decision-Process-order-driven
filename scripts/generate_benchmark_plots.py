"""
Generate Empirical Benchmark Plots from output/kaggle_benchmark_results.json.

Creates visualizations for:
1. Throughput Speed (SPS) vs. Model Parameter Scale & Depth
2. Mean Episode Reward & Crafter Score Across Architecture Scales
3. Loss Optimization Convergence (Initial vs. Final Loss) across Models
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def generate_plots():
    json_path = "output/kaggle_benchmark_results.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist!")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    results = data["benchmark_results"]
    os.makedirs("output/plots", exist_ok=True)

    names = [r["model_name"] for r in results]
    params = [r["trainable_parameters"] for r in results]
    sps = [r["sps_throughput"] for r in results]
    rewards = [r["mean_episode_reward"] for r in results]
    crafter_scores = [r["crafter_score"] for r in results]
    init_losses = [r["initial_loss"] for r in results]
    final_losses = [r["final_loss"] for r in results]

    # Colors
    colors = ['#2c3e50', '#34495e', '#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']

    # -------------------------------------------------------------
    # PLOT 1: SPS Throughput vs. Model Parameters & Scale
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(names, sps, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    
    for bar, val in zip(bars, sps):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{val:.1f} SPS", ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.title("1. Inference & Rollout Throughput (SPS) Across Model Scale", fontsize=13, fontweight='bold')
    plt.xlabel("Model Configuration", fontsize=11, fontweight='bold')
    plt.ylabel("Step-Per-Second (SPS)", fontsize=11, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontweight='bold')
    plt.ylim(0, max(sps) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    p1_path = "output/plots/sps_throughput_vs_parameters.png"
    plt.savefig(p1_path, dpi=300)
    plt.close()
    print(f"Saved plot: {p1_path}")

    # -------------------------------------------------------------
    # PLOT 2: Mean Reward & Crafter Score Scaling
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    x_indices = np.arange(len(names))
    width = 0.35

    rects1 = ax1.bar(x_indices - width/2, rewards, width, label='Mean Episode Reward', color='#2980b9', edgecolor='black')
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x_indices + width/2, crafter_scores, width, label='Crafter Score (S_crafter)', color='#27ae60', edgecolor='black')

    ax1.set_xlabel('Model Configuration', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Mean Episode Reward', color='#2980b9', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Crafter Score (S_crafter)', color='#27ae60', fontweight='bold', fontsize=11)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(names, rotation=25, ha='right', fontweight='bold')

    for r1, r2 in zip(rects1, rects2):
        h1 = r1.get_height()
        h2 = r2.get_height()
        if h1 >= 0:
            ax1.text(r1.get_x() + r1.get_width()/2.0, h1 + 0.05, f"{h1:.2f}", ha='center', va='bottom', fontsize=8, color='#2980b9', fontweight='bold')
        else:
            ax1.text(r1.get_x() + r1.get_width()/2.0, h1 - 0.12, f"{h1:.2f}", ha='center', va='top', fontsize=8, color='#2980b9', fontweight='bold')
            
        if h2 > 0:
            ax2.text(r2.get_x() + r2.get_width()/2.0, h2 + 0.01, f"{h2:.3f}", ha='center', va='bottom', fontsize=8, color='#27ae60', fontweight='bold')

    plt.title("2. Environment Performance & Crafter Score Across Model Scale", fontsize=13, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()

    p2_path = "output/plots/reward_and_crafter_score_scaling.png"
    plt.savefig(p2_path, dpi=300)
    plt.close()
    print(f"Saved plot: {p2_path}")

    # -------------------------------------------------------------
    # PLOT 3: Loss Optimization Convergence (Initial vs. Final Loss)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5.5))
    
    x = np.arange(len(names))
    width = 0.35

    plt.bar(x - width/2, init_losses, width, label='Initial Loss (Step 0)', color='#e74c3c', alpha=0.85, edgecolor='black')
    plt.bar(x + width/2, final_losses, width, label='Final Loss (Step 10)', color='#2ecc71', alpha=0.85, edgecolor='black')

    for idx_i, val in enumerate(init_losses):
        plt.text(x[idx_i] - width/2, val + (max(init_losses)*0.02), f"{val:.1f}", ha='center', va='bottom', fontsize=8, color='#c0392b', fontweight='bold')
    for idx_f, val in enumerate(final_losses):
        plt.text(x[idx_f] + width/2, val + (max(init_losses)*0.02), f"{val:.1f}", ha='center', va='bottom', fontsize=8, color='#27ae60', fontweight='bold')

    plt.title("3. Optax Gradient Loss Reduction (Initial vs. Final Loss)", fontsize=13, fontweight='bold')
    plt.xlabel("Model Configuration", fontsize=11, fontweight='bold')
    plt.ylabel("Loss Magnitude", fontsize=11, fontweight='bold')
    plt.xticks(x, names, rotation=25, ha='right', fontweight='bold')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    p3_path = "output/plots/loss_convergence_comparison.png"
    plt.savefig(p3_path, dpi=300)
    plt.close()
    print(f"Saved plot: {p3_path}")


if __name__ == "__main__":
    generate_plots()
