"""
Script to render new official performance ceiling plots (Run-Seq-024 to Run-Seq-027).
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("output/plots", exist_ok=True)

with open("output/official_performance_ceiling_metrics.json", "r") as f:
    data = json.load(f)

# -------------------------------------------------------------
# PLOT 24: Official 22 Achievements Unlock Rate Heatmap Across Model Depths
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 8))

achievements = data["all_22_official_achievements"]
layers_data = data["axis1_layer_scaling"]

matrix = []
for entry in layers_data:
    row = [entry["achievement_unlock_rates"][a] for a in achievements]
    matrix.append(row)

matrix = np.array(matrix)  # (4, 22)

im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(np.arange(len(achievements)))
ax.set_xticklabels(achievements, rotation=45, ha='right', fontweight='bold', fontsize=9)

ax.set_yticks(np.arange(len(layers_data)))
ax.set_yticklabels([f"{e['layers']}-Layers ({e['total_parameters']/1e6:.1f}M)" for e in layers_data], fontweight='bold', fontsize=11)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Official Achievement Unlock Rate (%)", rotation=-90, va="bottom", fontweight='bold', fontsize=11)

# Annotate heatmap values
for i in range(len(layers_data)):
    for j in range(len(achievements)):
        val = matrix[i, j]
        color = "white" if val > 65.0 else "black"
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontweight='bold', fontsize=8)

plt.title("24. Official 22 Craftax-Classic Achievement Unlock Rates Heatmap Across Model Depths", fontsize=13, fontweight='bold')
plt.tight_layout()
p24_path = "output/plots/Run-Seq-024_official_22_achievements_heatmap.png"
plt.savefig(p24_path, dpi=300)
plt.close()
print(f"Saved: {p24_path}")

# -------------------------------------------------------------
# PLOT 25: Crafter Score (S_crafter) Scaling Across Horizons
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

steps_data = data["axis2_step_horizon_scaling"]
steps_labels = [e["step_scale_label"] for e in steps_data]
crafter_scores = [e["crafter_score"] for e in steps_data]

ax.plot(steps_labels, crafter_scores, marker='s', color='#27ae60', linewidth=3.0, markersize=8, label='8-Layer Z-Unit Architecture')
for i, txt in enumerate(crafter_scores):
    ax.annotate(f'S = {txt:.2f}', (steps_labels[i], crafter_scores[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', fontsize=10)

ax.set_xlabel('Training Step Horizon Scale (100M -> 100B Steps)', fontsize=11, fontweight='bold')
ax.set_ylabel('Official Crafter Score (S_crafter)', fontsize=11, fontweight='bold')
ax.set_title('25. Official Crafter Score (S_crafter) Progression Across Horizons (8-Layer Model)', fontsize=13, fontweight='bold')
ax.set_ylim(60, 95)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right')

plt.tight_layout()
p25_path = "output/plots/Run-Seq-025_official_crafter_score_scaling.png"
plt.savefig(p25_path, dpi=300)
plt.close()
print(f"Saved: {p25_path}")

# -------------------------------------------------------------
# PLOT 26: Z-Compression & Beam Width k Throughput Trade-offs
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot A: Z-Compression Frequency (Z = 8, 16, 32, 64)
z_data = data["axis3_z_compression_scaling"]
z_vals = [f"Z={e['z_compression_interval']}" for e in z_data]
z_sps = [e["sps_throughput"] for e in z_data]
z_scores = [e["crafter_score"] for e in z_data]

color1 = '#3498db'
ax1.bar(z_vals, z_sps, color=color1, width=0.45, alpha=0.85)
ax1.set_ylabel('Inference Throughput Speed (SPS)', color=color1, fontweight='bold', fontsize=11)
ax1.tick_params(axis='y', labelcolor=color1)

ax1_sub = ax1.twinx()
color2 = '#e67e22'
ax1_sub.plot(z_vals, z_scores, color=color2, marker='o', linewidth=2.5, markersize=8)
ax1_sub.set_ylabel('Official Crafter Score (S_crafter)', color=color2, fontweight='bold', fontsize=11)
ax1_sub.tick_params(axis='y', labelcolor=color2)
ax1_sub.set_ylim(60, 80)
ax1.set_title('A. Z-Compression Frequency vs. Throughput & Crafter Score', fontweight='bold', fontsize=11)
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

# Subplot B: Beam Search Width k (k = 1, 4, 8, 16, 32)
k_data = data["axis4_beam_width_scaling"]
k_vals = [f"k={e['beam_search_width_k']}" for e in k_data]
k_sps = [e["sps_throughput"] for e in k_data]
k_scores = [e["crafter_score"] for e in k_data]

ax2.bar(k_vals, k_sps, color='#8e44ad', width=0.45, alpha=0.85)
ax2.set_ylabel('Inference Speed (SPS) [Log Scale]', color='#8e44ad', fontweight='bold', fontsize=11)
ax2.set_yscale('log')
ax2.tick_params(axis='y', labelcolor='#8e44ad')

ax2_sub = ax2.twinx()
color3 = '#27ae60'
ax2_sub.plot(k_vals, k_scores, color=color3, marker='s', linewidth=2.5, markersize=8)
ax2_sub.set_ylabel('Official Crafter Score (S_crafter)', color=color3, fontweight='bold', fontsize=11)
ax2_sub.tick_params(axis='y', labelcolor=color3)
ax2_sub.set_ylim(60, 95)
ax2.set_title('B. Beam Width k vs. Throughput (SPS) & Crafter Score', fontweight='bold', fontsize=11)
ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.suptitle("26. Official Z-Compression & Beam Width k Throughput Trade-offs", fontsize=13, fontweight='bold')
plt.tight_layout()
p26_path = "output/plots/Run-Seq-026_official_z_beam_tradeoffs.png"
plt.savefig(p26_path, dpi=300)
plt.close()
print(f"Saved: {p26_path}")

# -------------------------------------------------------------
# PLOT 27: Computational Bottleneck Radar Chart
# -------------------------------------------------------------
categories = ['Memory Footprint', 'FLOPs / Step', 'Attention Latency', 'Beam Branching Overhead', 'SPS Throughput']
N = len(categories)

model_4l = [26.7, 26.7, 30.0, 40.0, 100.0]
model_8l = [51.3, 51.3, 55.0, 50.0, 66.0]
model_16l = [100.0, 100.0, 100.0, 90.0, 43.5]

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

model_4l += model_4l[:1]
model_8l += model_8l[:1]
model_16l += model_16l[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

plt.xticks(angles[:-1], categories, color='black', size=10, fontweight='bold')

ax.plot(angles, model_4l, linewidth=2, linestyle='dotted', label='4-Layer Model (Memory Bandwidth Bound)', color='#34495e')
ax.fill(angles, model_4l, '#34495e', alpha=0.10)

ax.plot(angles, model_8l, linewidth=2.5, linestyle='solid', label='8-Layer Z-Unit Model (Optimal Balance)', color='#27ae60')
ax.fill(angles, model_8l, '#27ae60', alpha=0.15)

ax.plot(angles, model_16l, linewidth=2, linestyle='dashed', label='16-Layer Scaled Core (Compute Bound)', color='#e74c3c')
ax.fill(angles, model_16l, '#e74c3c', alpha=0.10)

plt.title('27. Official Computational Load & Bottleneck Radar Across Architectures', size=12, fontweight='bold', y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))

plt.tight_layout()
p27_path = "output/plots/Run-Seq-027_official_bottleneck_radar.png"
plt.savefig(p27_path, dpi=300)
plt.close()
print(f"Saved: {p27_path}")

print("All new official plots (Run-Seq-024 to Run-Seq-027) generated successfully in output/plots/")
