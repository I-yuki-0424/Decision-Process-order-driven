"""
Script to render plots for Massive Scaling (10B -> 10T), Pure Macro Variant, and Extended Context Window.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("output/plots", exist_ok=True)

# -------------------------------------------------------------
# Plot 7: Massive Step-Scale Scaling Laws (10B to 10T Steps)
# -------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(9, 5))

scales = [10, 100, 1000, 10000]  # in Billions
cluster_hours = [1.10, 10.97, 109.70, 1097.03]
reward_ceilings = [0.0394, 0.0439, 0.0485, 0.0530]

color1 = '#2980b9'
ax1.set_xlabel('Total Training Step Scale (Billions of Steps [Log Scale])', fontsize=11, fontweight='bold')
ax1.set_ylabel('64-GPU Cluster Compute Time (Hours)', color=color1, fontsize=11, fontweight='bold')
line1 = ax1.plot(scales, cluster_hours, marker='o', color=color1, linewidth=2.5, label='Cluster Compute Time (Hours)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, which='both', linestyle='--', alpha=0.4)

ax2 = ax1.twinx()
color2 = '#8e44ad'
ax2.set_ylabel('Projected Mean Reward Ceiling', color=color2, fontsize=11, fontweight='bold')
line2 = ax2.plot(scales, reward_ceilings, marker='s', color=color2, linewidth=2.5, linestyle='--', label='Reward Scaling Ceiling')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0.03, 0.06)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.title('Massive Step-Scale Scaling Laws (10B to 10T Steps across 64-GPU Cluster)', fontsize=13, fontweight='bold')
plt.tight_layout()
p7_path = "output/plots/Run-Seq-007_massive_step_scale_laws_10b_to_10t.png"
plt.savefig(p7_path, dpi=300)
plt.close()
print(f"Saved: {p7_path}")

# -------------------------------------------------------------
# Plot 8: Pure Macro vs Hierarchical Architecture Speedup (151.26x)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
variants = ['Pure Macro Variant\n(90k Direct Transformer Steps)', 'Hierarchical Architecture\n(100 Macro + 90k Micro Scan)']
sps_vals = [261.57, 39564.05]
colors = ['#e74c3c', '#27ae60']

bars = ax.bar(variants, sps_vals, color=colors, width=0.45)
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, y + 1000, f'{y:,.1f} SPS', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_yscale('log')
ax.set_ylim(100, 100000)
ax.set_ylabel('Execution Speed (SPS) [Log Scale]', fontsize=11, fontweight='bold')
ax.set_title('Architectural Throughput: Pure Macro (O(L^2)) vs Hierarchical (151.26x Speedup)', fontsize=12, fontweight='bold')
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
p8_path = "output/plots/Run-Seq-008_pure_macro_vs_hierarchical_speedup.png"
plt.savefig(p8_path, dpi=300)
plt.close()
print(f"Saved: {p8_path}")

# -------------------------------------------------------------
# Plot 9: Extended Context Window Scaling (128 vs 256 History)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
configs = ['Standard Context\n(L_hist = 128)', 'Extended Context\n(L_hist = 256 [Doubled])']
tokens = [2130, 4258]

bars = ax.bar(configs, tokens, color=['#34495e', '#16a085'], width=0.45)
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, y + 100, f'{y:,} Tokens', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Total Context Sequence Capacity (Tokens)', fontsize=11, fontweight='bold')
ax.set_title('Extended Context Window Capacity Scaling (2.0x History Retention)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 5000)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
p9_path = "output/plots/Run-Seq-009_extended_context_window_scaling.png"
plt.savefig(p9_path, dpi=300)
plt.close()
print(f"Saved: {p9_path}")

print("All massive scaling plots generated successfully in output/plots/")
