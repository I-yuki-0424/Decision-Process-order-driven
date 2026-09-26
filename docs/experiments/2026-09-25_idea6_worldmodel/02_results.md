# Results

Plot: `output/experiments/2026-09-25_idea6/plots/idea6_summary.png`.

## Stage A (held-out standardised MSE)

| steps ahead | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| persistence | 0.040 | 0.082 | 0.162 | 0.319 |
| mlp | 0.037 | 0.066 | 0.101 | 0.144 |
| hn | 0.037 | 0.062 | 0.098 | 0.132 |

Skill vs persistence at 8 steps: hn 0.585, mlp 0.548. 70% of test states have at least one vital changing within 8 noop steps.

## Stage B (final sampled-policy eval return, 64 episodes; uniform-random = 1.46; seed0 / seed1)

| arm | return s0 / s1 | mean | crafter score (mean) |
|---|---|---|---|
| base (no WM features) | 2.78 / 3.10 | 2.94 | 3.62 |
| wm_hn (trained) | 2.94 / 2.93 | 2.94 | 3.34 |
| wm_mlp (trained) | 3.17 / 3.29 | 3.23 | 3.67 |
| wm_hn_untrained (control) | 2.95 / 3.32 | 3.14 | 3.65 |

Seed-to-seed spread within an arm is 0.0-0.4, as large as any between-arm gap.
