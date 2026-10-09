"""TASK-025: mean per-achievement unlock rate (%) and eval episode length per arm over all TASK-025 final blocks (A-H)."""
import glob
import json
import os

import numpy as np
from craftax.craftax_classic.constants import Achievement

NAMES = [a.name.lower() for a in sorted(Achievement, key=lambda a: a.value)]
arms = {}
for f in glob.glob("output/phase1/push_t025/final[A-H]/*.json"):
    a = os.path.basename(f).split("__s")[0]
    arms.setdefault(a, []).append(json.load(open(f))["per_seed"][0])
print("| achievement | " + " | ".join(f"{a} (n={len(arms.get(a, []))})" for a in ["E1", "G4", "TP1", "G4TP1", "TP1h"]) + " |")
print("|---|" + "---|" * 5)
rates = {a: np.mean([x["achievement_rates_pct"] for x in v], 0) for a, v in arms.items()}
for i, n in enumerate(NAMES):
    print(f"| {n} | " + " | ".join(f"{rates[a][i]:.1f}" if a in rates else "-" for a in ["E1", "G4", "TP1", "G4TP1", "TP1h"]) + " |")
print("| eval episode length | " + " | ".join(f"{np.mean([x['eval_mean_length'] for x in arms[a]]):.0f}" if a in arms else "-"
                                           for a in ["E1", "G4", "TP1", "G4TP1", "TP1h"]) + " |")
