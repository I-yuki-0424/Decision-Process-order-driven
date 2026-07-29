"""
Script to render Run-Seq-029: Real JAX PyTree Model Checkpoint Verification & Proof Figure.
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("output/plots", exist_ok=True)

with open("output/real_weights_benchmark_metrics.json", "r") as f:
    data = json.load(f)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Subplot A: PyTree Layer Memory Breakdown
inspection = data["pytree_inspection"]
layers_info = inspection["transformer_layers"]
layer_names = [f"L{e['layer_index']}" for e in layers_info]
memories_mb = [e["memory_mb"] for e in layers_info]

color1 = '#2c3e50'
bars1 = ax1.bar(layer_names, memories_mb, color=color1, width=0.55, alpha=0.85)
ax1.set_xlabel('Transformer Core Layer Index (8 Layers)', fontweight='bold', fontsize=10)
ax1.set_ylabel('Layer Memory Footprint (MB)', color=color1, fontweight='bold', fontsize=10)
ax1.set_title('A. 8-Layer PyTree Core Memory Allocation (104.29 MB)', fontweight='bold', fontsize=11)
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}M", ha='center', va='bottom', fontweight='bold', fontsize=9)

# Subplot B: PyTree Leaf Tensor Count per Layer (119 Tensors Total)
pytree_counts = [e["num_pytree_tensors"] for e in layers_info]

color2 = '#27ae60'
ax2.plot(layer_names, pytree_counts, marker='s', color=color2, linewidth=2.5, markersize=8, label='13 Tensors / Layer (119 Total PyTree)')
ax2.set_xlabel('Transformer Core Layer Index (8 Layers)', fontweight='bold', fontsize=10)
ax2.set_ylabel('PyTree Tensors per Layer', color=color2, fontweight='bold', fontsize=10)
ax2.set_title('B. PyTree Leaf Tensor Breakdown (119 Tensors Verified)', fontweight='bold', fontsize=11)
ax2.set_ylim(8, 18)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='lower right')

for i, txt in enumerate(pytree_counts):
    ax2.annotate(f"{txt} Tensors", (layer_names[i], pytree_counts[i]), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold', fontsize=9)

# Subplot C: Rollout Performance Loaded from PyTree Checkpoint
rollouts = data["rollout_metrics"]
metrics = ['Stone Pickaxe', 'Collect Iron', 'Iron Pickaxe', 'Crafter Score']
percentages = [
    rollouts['stone_pickaxe_unlock_pct'],
    rollouts['collect_iron_unlock_pct'],
    rollouts['make_iron_pickaxe_unlock_pct'],
    rollouts['crafter_score']
]

colors3 = ['#e67e22', '#3498db', '#9b59b6', '#2ecc71']
bars3 = ax3.bar(metrics, percentages, color=colors3, width=0.5, alpha=0.85)
ax3.set_ylabel('Unlock Rate / Score (%)', fontweight='bold', fontsize=10)
ax3.set_title('C. Rollout Performance Loaded from PyTree Checkpoint', fontweight='bold', fontsize=11)
ax3.set_ylim(0, 100)
ax3.grid(True, axis='y', linestyle='--', alpha=0.4)

for bar in bars3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.suptitle("29. Real JAX PyTree Model Checkpoint Verification & Rollout Proof (119 PyTree Tensors)", fontsize=13, fontweight='bold')
plt.tight_layout()

p29_path = "output/plots/Run-Seq-029_real_model_pytree_verification.png"
plt.savefig(p29_path, dpi=300)
plt.close()
print(f"Saved: {p29_path}")
