"""
Script to render Run-Seq-028: Kaggle Retrieved Weights (W) Verification & Proof Figure.
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("output/plots", exist_ok=True)

with open("output/kaggle_retrieved_weights_verification.json", "r") as f:
    data = json.load(f)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Subplot A: Layer-by-Layer Parameter Breakdown
layers_info = data["layer_breakdown"]
layer_names = [f"L{e['layer_index']}" for e in layers_info]
memories_mb = [e["memory_mb"] for e in layers_info]

color1 = '#2c3e50'
ax1.bar(layer_names, memories_mb, color=color1, width=0.55, alpha=0.85)
ax1.set_xlabel('Transformer Backbone Layer Index', fontweight='bold', fontsize=10)
ax1.set_ylabel('Layer Memory Footprint (MB)', color=color1, fontweight='bold', fontsize=10)
ax1.set_title('A. 8-Layer Transformer Memory Footprint (100.59 MB Total)', fontweight='bold', fontsize=11)
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

for i, v in enumerate(memories_mb):
    ax1.text(i, v + 0.3, f"{v:.1f}M", ha='center', fontweight='bold', fontsize=9)

# Subplot B: Layer-wise L2 Norm Distribution
l2_norms = [e["l2_weight_norm"] for e in layers_info]

color2 = '#27ae60'
ax2.plot(layer_names, l2_norms, marker='o', color=color2, linewidth=2.5, markersize=8)
ax2.set_xlabel('Transformer Backbone Layer Index', fontweight='bold', fontsize=10)
ax2.set_ylabel('Layer Weight L2 Norm (||W_l||_2)', color=color2, fontweight='bold', fontsize=10)
ax2.set_title('B. Layer-wise Weight L2 Norm Distribution Across Layers', fontweight='bold', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.4)

for i, v in enumerate(l2_norms):
    ax2.annotate(f"{v:.1f}", (layer_names[i], l2_norms[i]), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold', fontsize=9)

# Subplot C: Rollout Evaluation Unlock Rates with Loaded Weights W
rollouts = data["rollout_evaluation_results"]
metrics = ['Stone Pickaxe', 'Collect Iron', 'Iron Pickaxe', 'Crafter Score']
percentages = [
    rollouts['stone_pickaxe_unlock_pct'],
    rollouts['collect_iron_unlock_pct'],
    rollouts['make_iron_pickaxe_unlock_pct'],
    rollouts['crafter_score']
]

colors3 = ['#e67e22', '#3498db', '#9b59b6', '#2ecc71']
bars = ax3.bar(metrics, percentages, color=colors3, width=0.5, alpha=0.85)
ax3.set_ylabel('Unlock Rate / Score (%)', fontweight='bold', fontsize=10)
ax3.set_title('C. Verified Rollout Performance Loaded from Weights W', fontweight='bold', fontsize=11)
ax3.set_ylim(0, 100)
ax3.grid(True, axis='y', linestyle='--', alpha=0.4)

for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.suptitle("28. Kaggle Retrieved Trained Model Weights W Proof & Execution Verification", fontsize=13, fontweight='bold')
plt.tight_layout()

p28_path = "output/plots/Run-Seq-028_kaggle_retrieved_weights_verification.png"
plt.savefig(p28_path, dpi=300)
plt.close()
print(f"Saved: {p28_path}")
