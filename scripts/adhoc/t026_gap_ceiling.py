"""TASK-026: reward_pct gap to G1.2 and the per-achievement ceiling of the TASK-025 arms (reads nothing from disk; rates copied from RESULTS.md section 5).

  python scripts/adhoc/t026_gap_ceiling.py
"""
import numpy as np

# SOURCE: docs/experiments/2026-10-08_phase1_push/RESULTS.md section 5 (unlock rate %, 22 achievements in Craftax-Classic order)
RATES = {
    "E1": [99.6, 98.0, 36.3, 97.5, 61.7, 88.7, 72.5, 95.4, 47.3, 76.8, 71.1, 0.0, 4.2, 8.2, 12.3, 68.9, 73.8, 26.0, 0.8, 0, 0, 0],
    "G4": [99.8, 98.8, 44.3, 96.8, 73.5, 84.0, 79.8, 96.1, 9.9, 74.2, 63.5, 0.0, 1.2, 9.0, 9.1, 77.3, 64.2, 14.0, 0.4, 0, 0, 0],
    "TP1": [99.8, 98.4, 35.0, 97.3, 60.6, 85.3, 66.7, 95.9, 50.3, 71.9, 66.8, 0.0, 2.9, 7.8, 4.5, 68.9, 67.1, 20.4, 0.7, 0, 0, 0],
    "GRU-TP": [99.7, 98.5, 75.0, 96.2, 70.5, 87.7, 81.4, 95.6, 15.7, 79.1, 69.0, 0.0, 2.3, 17.0, 16.3, 65.8, 69.8, 24.5, 1.1, 0, 0, 0],
    "TP1h": [99.6, 98.4, 47.7, 99.0, 63.9, 91.5, 78.7, 98.3, 59.3, 81.1, 74.6, 0.0, 5.5, 17.3, 15.3, 49.8, 77.1, 30.4, 1.6, 0, 0, 0],
}
G12_REWARD = 55.49   # SOURCE: MASTER_GUIDANCE roadmap, MFRL_best (Dedieu et al. 2025 Table 1)

for k, v in RATES.items():
    print(f"{k:7s} reward_pct {np.mean(v):6.2f}  achievements/episode {sum(v) / 100:5.2f}")
best = np.max(np.array(list(RATES.values())), axis=0)
print(f"union of per-achievement best: reward_pct {best.mean():.2f} ({best.sum() / 100:.2f}/episode); G1.2 needs {G12_REWARD} ({G12_REWARD * 22 / 100:.2f}/episode)")
for name, sd in (("GRU-TP block sd 1.14", 1.14), ("TP1h pooled sd 3.9", 3.9)):
    print(f"{name}: mean needed for mean-2SE > {G12_REWARD} over 10 seeds = {G12_REWARD + 2 * sd / np.sqrt(10):.2f}")
