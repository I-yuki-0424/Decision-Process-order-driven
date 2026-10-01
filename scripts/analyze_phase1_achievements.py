"""Per-achievement unlock rate (% of first episodes, mean over seeds) per arm from final result files (host python, no craftax needed).

  python scripts/analyze_phase1_achievements.py output/phase1/final_100k
"""
import glob
import json
import sys

import numpy as np

NAMES = ["collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink", "make_wood_pickaxe", "make_wood_sword",
         "place_plant", "defeat_zombie", "collect_stone", "place_stone", "eat_plant", "defeat_skeleton", "make_stone_pickaxe",
         "make_stone_sword", "wake_up", "place_furnace", "collect_coal", "collect_iron", "collect_diamond", "make_iron_pickaxe",
         "make_iron_sword"]   # same order as src/environment/craftax_env_adapter.ACHIEVEMENT_NAMES

rows = {}
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    r = json.load(open(f))
    if "per_seed" in r:
        rows[r["arm"]] = np.mean([s["achievement_rates_pct"] for s in r["per_seed"]], 0)
print("| achievement | " + " | ".join(rows) + " |")
print("|---|" + "---|" * len(rows))
for i, n in enumerate(NAMES):
    if max(v[i] for v in rows.values()) > 0.5:
        print(f"| {n} | " + " | ".join(f"{v[i]:.1f}" for v in rows.values()) + " |")
