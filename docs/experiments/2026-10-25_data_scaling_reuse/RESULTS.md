# TASK-20261025-028 results: data scaling (0.5M-5M) and data reuse at 1M

Runs: Kaggle kernels `dpod-t028-a` / `dpod-t028-b` (2 x T4 each), pushed 2026-10-25 08:14 UTC from commit `c330e88` (git_dirty false), both COMPLETE by 10:01 UTC
(about 1 h 47 min wall). 28/28 result files returned. Seeds 7000-7003, test seed = train seed, 256 fresh envs, sampled policy, final parameters.
Files: `results/*.json` (plain copies of `output/phase1/scaling_t028/S/`). Summary: `python scripts/summarize_t028.py output/phase1/scaling_t028/S`.
Reading rules: README.md section 1, fixed before the runs. **Nothing here is gate evidence.** 2M/5M are `diag-data-scaling` (beyond the EP-A budget),
1M runs are `EP-A-tuning`, 0.5M is `EP-A-mini`.

## Q1: scaling (GRU-TP, `Base_O02_D04_S00`, recipe of TASK-025 G4TP1; lr annealed over each run's own budget)

| run | env steps | reward_pct | score_pct | eval length | train s / seed (T4) |
|---|---|---|---|---|---|
| S05M | 0.5M | 35.42 ± 1.14 | 10.36 ± 0.43 | 179 | 232 |
| S1M | 1M | 48.46 ± 0.75 | 18.22 ± 1.19 | 238 | 375 |
| S2M | 2M | 60.41 ± 1.53 | 28.11 ± 1.28 | 264 | 660 |
| S5M | 5M | 67.46 ± 0.77 | 34.34 ± 0.71 | 288 | 1541 |

Gain per doubling of env steps: 0.5M->1M +13.0, 1M->2M +12.0, 2M->5M +5.3 (per doubling; +7.1 over 1.32 doublings).
Context only (other platform, not mixed): TASK-026 CPU 2M = 57.7 (2 seeds).

## Q2: what drives the 1M -> 5M gain (+19.0 reward_pct)

Contribution = delta rate / 22. Achievements making up >= 70 % of the gain (pre-registered definition), largest first:

| achievement | rate 1M -> 5M (%) | contribution | cumulative |
|---|---|---|---|
| make_stone_pickaxe | 12.3 -> 85.4 | +3.32 | 17 % |
| make_stone_sword | 21.8 -> 86.6 | +2.95 | 33 % |
| collect_coal | 23.9 -> 82.8 | +2.68 | 47 % |
| defeat_zombie | 13.4 -> 67.3 | +2.45 | 60 % |
| defeat_skeleton | 1.7 -> 42.9 | +1.87 | 70 % |

Next are collect_iron 1.0 -> 28.4 (+1.25), then place_stone and place_furnace (70.5 -> 97, +1.2 each). Negative: wake_up 63.6 -> 45.4 (-0.83), place_plant and collect_sapling
(about -0.25 each). Still 0 % at 5M: eat_plant, make_iron_pickaxe, make_iron_sword, collect_diamond.
1M -> 2M (+12.0): defeat_zombie, make_stone_pickaxe, make_stone_sword and collect_coal give 67 %. 0.5M -> 1M (+13.0): early-tier skills (eat_cow, place_stone,
collect_stone, place_furnace) give 61 %.

S5M mid-run evaluations (lr not yet annealed; descriptive only):

| env steps | reward | score | stone_pickaxe | stone_sword | coal | zombie | skeleton | iron |
|---|---|---|---|---|---|---|---|---|
| 0.5M | 33.6 | 10.0 | 1 | 0 | 5 | 9 | 1 | 0 |
| 1M | 42.8 | 14.9 | 4 | 7 | 15 | 21 | 1 | 0 |
| 2M | 53.0 | 22.0 | 37 | 36 | 48 | 19 | 4 | 1 |
| 3M | 59.9 | 27.8 | 62 | 60 | 58 | 43 | 14 | 5 |
| 4M | 63.6 | 31.3 | 76 | 75 | 70 | 51 | 28 | 14 |
| 5M (final) | 67.5 | 34.3 | 85 | 87 | 83 | 67 | 43 | 28 |

At equal step counts, the budget-matched runs are 5-7 points above the 5M run's mid-run points (1M: 48.5 vs 42.8; 2M: 60.4 vs 53.0). Annealing lr to zero within the
budget is worth about that much. The tier order is fixed: the early tier, then the stone tier and coal, then combat, then iron. The iron tools and diamond never appear.

## Q3: is the problem only the step count?

Under the pre-registered rule, **yes, up to 2M, with a caveat beyond that.** Reward rises about linearly in log2(steps) from 0.5M to 2M (+13.0, +12.0 per
doubling, no change of slope). The 1M->2M and 1M->5M gains come from achievements that already occur at 1M (12-24 %), not from achievements at 0 % at 1M. The
binding constraint at 1M is how fast the agent turns behaviours it has already seen into reliable ones (sample efficiency), not reaching unseen states.
Caveats:
(a) The slope halves after 2M (+5.3 per doubling).
(b) The four deepest achievements (iron tools, diamond, eat_plant) stay at 0 % even at 5M. That second wall is an exploration/credit-horizon problem that more steps of the
same kind do not solve within 5M.
(c) Interpolating linearly in log2, S1M -> S2M crosses the MFRL_best reference 55.49 at about 1.5M steps. This is descriptive only: it is diag, a different budget, 4 tuning seeds,
and it does not count toward G1.2.

## Q4: data reuse at 1M (paired with S1M on the same seeds)

| arm | K | epochs | reward_pct | paired diff vs S1M | score diff | train s | candidate (>= +2.0 and > 2 SE) |
|---|---|---|---|---|---|---|---|
| RP2 | 2 | 4 | 44.14 ± 0.65 | -4.31 ± 0.61 | -2.60 ± 0.56 | 612 | no |
| RP4 | 4 | 4 | 15.64 ± 9.04 | -32.81 ± 8.83 | -13.97 ± 2.08 | 1064 | no |
| RP4e2 | 4 | 2 | 43.08 ± 1.09 | -5.38 ± 1.19 | -4.17 ± 1.26 | 606 | no |

**No reuse arm is a candidate; all three are significantly worse than plain PPO.** Training is numerically unstable (`train_diag`). In the 10 logged diagnostic points per run,
S1M's approx_kl stays at or below 0.074. RP2 reaches 0.25-0.55, with a 194 spike on seed 7002. Every RP4 and RP4e2 seed has spikes of 1e3 to 1e32, and grad_norm (before clipping) reaches 1e9-1e12.
RP4 seeds 7000 and 7002 became NaN (seed 7002 at about 20 % of training, seed 7000 at about 60 %) and score 0. Their rows are kept as recorded; they are failed runs of this arm,
not evidence about replay in general. Entropy also falls faster than in S1M (mid-run, seed 7000: RP4 about 0.6, RP2 about 0.95, S1M about 1.3).
What this shows: **this** replay learner (stale GRU start states with no burn-in, ratios to the proximal policy, truncated-IS GAE, K x more SGD steps per sample at
unchanged lr and clip) does not improve 1M-step efficiency and is unstable for K = 4. The run does not separate the possible causes (stale recurrent state, the K x larger
number of optimizer steps, the approx_kl estimator on near-zero probabilities), so it does not rule out reuse as such. It does rule out this simple form.
MLflow note: `Base_O02_D08_S01` groups RP4 and RP4e2 (epochs 4 / 2) into one summary row (n = 8). Use the table above.

## Bottom line

At 1M, GRU-TP is limited by sample efficiency on behaviours it has already discovered (the stone tier, coal and combat), and reward grows steadily with more data up to 2M.
Beyond 2M the gain slows, and the iron/diamond tier stays unreached even at 5M. Reusing data with this replay learner made things worse and, for K = 4, unstable.
