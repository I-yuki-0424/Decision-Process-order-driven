# Phase 1 results report (state of 2026-10-19)

Scope: every Phase-1 result recorded in this repository (TASK-017, -020, -023, -024, -025; all EP-A, Craftax-Classic, symbolic 1345-d observation, 999,424 env steps,
sampled policy, first episode of 256 fresh envs per seed, final checkpoint), compared against each other and against the roadmap gates. Numbers are quoted from
`STATE.yaml` and the `RESULTS.md` / `README.md` files named in section 10; nothing here is re-measured except the arithmetic in section 6 (reproducible with the
snippet there) and the tuning attempt of section 8. Paper references are quoted from MASTER_GUIDANCE (Dedieu et al. 2025, Table 1: MFRL_4M_reimpl 47.40 / 10.71 at 4.0M
parameters, MFRL_best 55.49 / 16.77 at 55.6M; image observation, which differs from our modality, caveat recorded in TASK-020). Metrics are `reward_pct / score_pct`.

## 1. Bottom line

1. **Phase 1 is not done.** The phase closes only when every gate passes (roadmap rule `phase_advance`); G1.2, the closing gate, fails for every arm in every one of the
   30 arm-blocks of TASK-024/-025. G1.0 passes, G1.1 passed once on a pre-registered block at its second attempt, G1.S is not a run.
2. **The remaining gap is on one metric.** Score already clears G1.2 (GRU-TP pooled 18.51 +- 0.21; needs 16.77). Reward does not: it must reach about 56.2 for a
   low-variance arm (mean - 2 SE > 55.49) against 48.46 pooled for the best low-variance arm (GRU-TP) and 49.36 for the highest-mean arm (Truck + TP head). That is +7.7 reward
   (about 1.6 more achievements per episode: 10.65 now, 12.21 needed at the threshold; more at the needed mean of 56.2).
3. **What moved the numbers, in order of size:** the shared PPO recipe (+9 to +13 on the baseline, G1.0), the transition-prediction auxiliary on a recurrent policy (+3.1 on the
   GRU), the Truck encoder at a stable recipe (+1.8 over the re-tuned baseline with 3.6x fewer parameters). Everything else tried (world-model features, pure Transformer,
   ChunkPPO at k=1, EV auxiliary, scale, rollout and epoch variants) moved nothing measurable or hurt.
4. **Even a perfect mix of the 1M-step policies would not reach G1.2** (section 6): taking the best unlock rate of any arm for every achievement gives 52.6 reward.
5. **The gap is sample efficiency, not model capacity (TASK-026 diagnostic, not EP-A).** The same GRU-TP config trained for 2M / 3M / 4M steps (tuning seeds, 2 seeds each) reaches
   57.7 / 66.3 / 65.5 reward and 25.6 / 33.4 / 31.9 score, against 48.8 / 17.7 at 1M (4 seeds). It crosses G1.2's thresholds between 1x and 2x the EP-A budget and plateaus near 66 by 3M.
   A 1M-step method therefore has to learn about 1.7-2x more per environment step; model size, width (512: no gain) and ten single-knob variants did not provide that (section 8).
   This corrects the first draft of this report and the pre-registration, which read the gap as a missing capability.

## 2. Gate status

| gate | requirement (EP-A) | status | evidence |
|---|---|---|---|
| G1.0 | baseline >= 42.7 / 9.6 | **PASS** | G4 (gru256+ln+skip, 1.01M) 45.42 +- 0.30 / 15.48 +- 0.27, 20 seeds (TASK-024); both blocks pass alone. With the old recipe it failed (ppo_gru 32.45 / 8.16, TASK-020; gru512 41.56 / 13.59, TASK-023) |
| G1.1 | mean - 2 SE > 47.40 and > 10.71, params_total <= 4.0M | **passed on a pre-registered block at the 2nd attempt, by GRU-TP only; operator ruling pending** | GRU-TP block I: 48.72 +- 0.36 (-2SE 48.01) / 18.82 +- 0.58 (17.67), 1.25M params. Its first gate block A failed by 0.02 reward (47.38). Blocks B, C, F, G, H also meet the relation. Truck + TP head fails its block I (48.78 +- 0.96, -2SE 46.85); plain Truck E1 fails in all 8 blocks (seed spread) |
| G1.2 | mean - 2 SE > 55.49 and > 16.77 | **FAIL, every arm, every block** | best block mean 50.72 (Truck + TP head, block F); best low-variance arm 48.72 |
| G1.S | EP-B screening | not run (EP-B protocol not used in the Phase-1 push) | - |

The roadmap's `surpass` relation is the strict one (mean - 2 SE over 10 seeds). An earlier statement "block 1 alone passed G1.1" (E1 48.54 +- 1.50) was withdrawn:
under the relation it is 45.53 < 47.40 (TASK-025 critical review).

## 3. The arms

| arm (registry id) | what it is | params_total / deployed |
|---|---|---|
| ppo_mlp | PureJaxRL-style PPO, separate tanh MLPs | 2.44M |
| ppo_gru, **G4** (Base_O02_D02_S00 = gru256+ln+skip) | dense embed -> GRU -> actor / critic; G4 adds LayerNorm before every dense layer and heads reading [h, embedding] (the reference-baseline architecture) | 0.88M (ppo_gru); 1.01M / 0.88M (G4) |
| cnn_gru | CNN over the 7x9 map + GRU | 1.59M |
| tf, tf_wm, tf_wm_random | pure Transformer over tile, scalar, history and 17 candidate-action tokens (logit of action a read from its token: the 4th idea); `_wm`: plus action-conditioned one-step world-model features; `_random`: same WM frozen at init (control) | 0.20M |
| ChunkPPO k=1 | the `chunk_ppo.py` path at chunk length 1: reduces to PPO with a small Transformer actor | - |
| **Truck** (Idea4_O01_D01_S00, `tf_gru`) | tile-token Transformer encoder (d64, 2 layers) per step + GRU memory; candidate-action tokens give the logits | 0.45M |
| **Truck-improved E1** (`tfgru64x2w128+mem+pa+hs`) | Truck + memory token (previous GRU state enters the Transformer) + previous action + [LN(h), CLS] heads, GRU width 128 | 0.284M / 0.283M |
| **TP** (transition prediction) | auxiliary loss: from the step features and the executed action predict S_{t+1} - S_t (1345-d) and the reward; trained only on executed transitions (P4), no extra env steps (P10), head trained and counted in params_total but never used to act | - |
| **GRU-TP** (Base_O02_D04_S00, `G4TP1`) | G4 + TP through an MLP head on [h, embedding, one-hot(action)], weight 1 | 1.254M / 0.880M |
| TP1 (`E1+tp1`, candidate-token readout) | E1 + TP carried by the executed action's candidate token (the 4th idea's S_{t+1} = S_t + W_res[A_t], learned, one linear map) | 0.371M / 0.283M |
| **TP1h** (Truck + TP head) | E1 + TP through a separate MLP head (post-hoc arm, chosen after block B) | 0.484M / 0.283M |
| EV | next-reward-event contrastive auxiliary (Moon et al. 2023 style) | harmful, stopped |

## 4. How the numbers moved (chronological; each row is one protocol-consistent 10-seed result unless stated)

| stage | arm | reward_pct | score_pct | note |
|---|---|---|---|---|
| TASK-017 (default recipe, 1M, seeds 0-9) | ppo_gru | 33.57 +- 0.83 | 9.12 +- 0.48 | 0.88M |
| | ppo_mlp / cnn_gru | 30.14 +- 0.46 / 32.95 +- 0.57 | 7.08 / 8.35 | |
| | tf / tf_wm / tf_wm_random | 29.24 / 28.58 / 27.16 | 6.67 / 6.71 / 5.86 | trained WM = frozen random WM |
| | chunkppo_k1 | 23.90 +- 0.34 | 4.58 | |
| | tf_gru (Truck) | 39.19 +- 1.27 | 11.45 +- 0.45 | **unreliable, do not cite** (uncommitted code, designed on eval seeds; audit TASK-022) |
| TASK-020 (tuned, old recipe, seeds 42-51) | ppo_gru / cnn_gru / Truck | 32.45 / 34.64 / 34.19 | 8.16 / 9.30 / 9.19 | G1.0 FAIL; Truck - ppo_gru +1.7 not distinguishable |
| TASK-023 (reference recipe, seeds 52-61) | gru512 (2.80M) | 41.56 +- 0.28 | 13.59 +- 0.17 | G1.0 FAIL narrowly (needs 42.7) |
| | Truck-improved (0.467M) | 39.29 +- 2.17 | 11.08 +- 1.25 | seed range 28-51 |
| TASK-024 (stable recipe: warm-up 5 %, entropy 0.01, lr 1e-3, gamma 0.97; 20 seeds) | G4 (1.01M) | 45.42 +- 0.30 | 15.48 +- 0.27 | **G1.0 PASS** |
| | E1 (0.284M) | 47.12 +- 0.99 | 17.27 +- 0.84 | G1.1 not established |
| TASK-025 (pooled over the blocks listed in RESULTS.md) | E1, 80 seeds | 47.21 +- 0.41 | 17.22 +- 0.35 | |
| | G4, 80 seeds | 45.31 +- 0.17 | 15.50 +- 0.15 | |
| | TP1, 20 seeds | 45.47 +- 0.97 | 15.59 +- 0.73 | TP via candidate tokens hurts |
| | **GRU-TP (G4TP1), 70 seeds** | **48.46 +- 0.15** | **18.51 +- 0.21** | lowest seed spread among the TP arms |
| | **Truck + TP head (TP1h), 50 seeds** | **49.36 +- 0.55** | **18.64 +- 0.52** | highest mean, sd 3.9 |

The recipe, not the architecture, produced the largest jump (baseline 32.45 -> 45.42 across TASK-020 -> TASK-024). Tuning at 100k steps does not transfer to 1M
(MLP = GRU at 100k, GRU +3.4 at 1M), so every selection above was made at the full 1M budget.

## 5. Head-to-head comparisons (differences of means with SE; independent seeds, same seed numbers within a block)

| comparison | result | reading |
|---|---|---|
| E1 (0.284M) - G4 (1.01M), 8 blocks, 80 seeds each | +1.77 +- 0.44 reward (z 4.0), +1.54 +- 0.37 score (z 4.2); ahead in all 8 blocks | Truck-improved beats the re-tuned baseline with 3.6x fewer parameters; modest (~+4 % relative). The earlier TASK-020/-023 "no better" was partly a recipe artefact |
| GRU-TP - G4, 4 blocks | +3.06 +- 0.35 reward (z 8.7), +2.86 +- 0.35 score (z 8.2) | TP is the one lever that clearly worked, on the recurrent baseline |
| TP1 (candidate tokens carry TP) - E1, 2 blocks | -2.03 +- 1.11 reward (z -1.8) | TP through the 4th-idea readout hurts the Transformer (-3.49 +- 0.86 vs GRU-TP on the same blocks) |
| TP1h (separate head) - E1, 2 blocks | +2.35 +- 1.26 (z 1.9), post-hoc arm | helps when the head, not the candidate tokens, carries TP |
| TP1h - GRU-TP, 5 blocks | +0.40 +- 0.52 reward, -0.48 +- 0.53 score | tie, with 2.6x fewer parameters (0.484M vs 1.254M; deployed 0.283M vs 0.880M) |
| E1 vs E1 without candidate tokens (block 2) | 0.00 +- 1.9 reward, +0.9 +- 1.6 score | the 4th-idea candidate-token readout is not shown to contribute |
| E1 vs E1 without mem / prev-action / head-skip (block 2) | +1.4 +- 1.6 reward, +3.0 +- 1.3 score (z 2.4) | those three additions carry a measurable score gain |
| Truck scale 0.14M / 0.28M / 0.47M / 0.93M / 1.66M (tuning seeds) | 40.8 / 48.2 / 48.9 / 47.4 / 46.5, seed sd 2.4 / 1.3 / 3.4 / 6.0 / 7.7 | plateau 0.28M-0.93M; larger = more seed spread, no gain |
| Pure Transformer, WM features, ChunkPPO k=1 vs pure RL (TASK-017, default recipe) | no better than ppo_gru; trained WM = frozen random WM at 100k and 1M | these ideas do not help in this setting |
| EV auxiliary | -9 to -11 reward on the first seed, stopped | harmful |
| K1/K2 (2x updates), entropy annealing, gamma 0.99, width 256, 3 layers on top of TP | none exceeded the pre-registered +3 (reward+score) | recipe plateau |

## 6. Per-achievement profile and the ceiling argument

Unlock rates (%), mean over the TASK-025 final seeds of each arm (RESULTS.md section 5; E1 n=60, G4 n=60, TP1 n=20, GRU-TP n=60, TP1h n=40):

| achievement | E1 | G4 | TP1 | GRU-TP | TP1h | best of arms |
|---|---|---|---|---|---|---|
| collect_wood / place_table | 99.6 / 98.0 | 99.8 / 98.8 | 99.8 / 98.4 | 99.7 / 98.5 | 99.6 / 98.4 | 99.8 / 98.8 |
| eat_cow | 36.3 | 44.3 | 35.0 | **75.0** | 47.7 | 75.0 |
| collect_sapling / place_plant | 97.5 / 95.4 | 96.8 / 96.1 | 97.3 / 95.9 | 96.2 / 95.6 | 99.0 / 98.3 | 99.0 / 98.3 |
| collect_drink | 61.7 | 73.5 | 60.6 | 70.5 | 63.9 | 73.5 |
| make_wood_pickaxe / sword | 88.7 / 72.5 | 84.0 / 79.8 | 85.3 / 66.7 | 87.7 / 81.4 | 91.5 / 78.7 | 91.5 / 81.4 |
| defeat_zombie | 47.3 | 9.9 | 50.3 | 15.7 | **59.3** | 59.3 |
| collect_stone / place_stone | 76.8 / 71.1 | 74.2 / 63.5 | 71.9 / 66.8 | 79.1 / 69.0 | 81.1 / 74.6 | 81.1 / 74.6 |
| make_stone_pickaxe / sword | 8.2 / 12.3 | 9.0 / 9.1 | 7.8 / 4.5 | 17.0 / 16.3 | 17.3 / 15.3 | 17.3 / 16.3 |
| wake_up | 68.9 | 77.3 | 68.9 | 65.8 | 49.8 | 77.3 |
| place_furnace / collect_coal | 73.8 / 26.0 | 64.2 / 14.0 | 67.1 / 20.4 | 69.8 / 24.5 | 77.1 / 30.4 | 77.1 / 30.4 |
| defeat_skeleton / collect_iron | 4.2 / 0.8 | 1.2 / 0.4 | 2.9 / 0.7 | 2.3 / 1.1 | 5.5 / 1.6 | 5.5 / 1.6 |
| eat_plant, diamond, iron pickaxe, iron sword | 0 | 0 | 0 | 0 | 0 | 0 |
| mean = reward_pct (achievements / 22) | 47.23 | 45.27 | 45.47 | 48.42 | 49.50 | **52.63** |
| eval episode length | 195 | 226 | 197 | 231 | 208 | |

Reproduction: `python scripts/adhoc/t026_gap_ceiling.py` (rates copied from RESULTS.md section 5; prints the per-arm means, the union of the best per-achievement rates = 52.63 = 11.58 achievements
per episode, G1.2's 55.49 = 12.21 per episode, and the mean needed at a given seed spread: 56.21 for GRU-TP's block sd 1.14, 57.96 for TP1h's pooled sd 3.9). The per-arm means differ slightly
from the pooled block means of section 4 because RESULTS.md section 5 covers only the TASK-025 seeds of each arm and rounds the rates.

Reading:
* GRU-TP and Truck-TP1h gain in different places. TP raises eat_cow (45 -> 75 %) and stone tools (9 -> 17 %) on the GRU; Truck defeats zombies far more often (47-59 % vs 10-16 %)
  and collects more coal. The two profiles are complementary but even their union (52.6) is below 55.49; the gap is in the deep chain, where every arm is near zero:
  iron <= 1.6 %, skeleton <= 5.5 %, stone tools 8-17 %, and eat_plant / diamond / iron tools 0.
* The evaluation episodes are short (195-231 steps). Zombies cause 70-82 % of deaths for every arm (about 55 % at night; death diagnostic on one tuning seed per arm,
  illustrative). Reaching iron and diamond needs both survival and long-horizon credit assignment (gamma 0.97 is about a 33-step horizon; gamma 0.99 on Truck gave no gain).
* This profile describes **1M-step policies**. With more data the deep chain does get unlocked (section 8: at 3M-4M steps stone pickaxe/sword 77-84 %, coal 74-78 %, iron 14-16 %,
  zombie 91 %, skeleton 23-31 %), so the 52.6 ceiling is a statement about the EP-A budget, not about what this model family can represent. eat_plant, diamond and iron tools stay at 0 % even at 4M.
* Hence the +7.7 reward gap is not a "last few points" problem of the 1M-step recipe, but it is also not a capacity wall: it is the amount learned per environment step.

## 7. Reliability and caveats (all recorded, none hidden)

* **Noise.** Runs are not bit-reproducible; per-seed spread is sd ~1.1 (GRU arms) to 3-5 (Truck arms), which is why 2-seed tuning means are optimistic by several points
  (winner's curse: TP1 48.69 on 4 tuning seeds -> 43.71 / 47.22 on fresh blocks). Claims here come from 10-seed blocks or pooled blocks only.
* **Post-hoc arms.** GRU-TP was registered as the baseline arm and became a gate candidate after block B (so block B is not gate evidence for it; block A, pre-registered, failed by 0.02;
  block I passed). TP1h was chosen after block B to test a hypothesis. Every block is reported, none dropped.
* **Three platforms.** Local RTX 3060 Ti (jax 0.10.2), Kaggle T4 (jax 0.11.1), and (TASK-026 only) a 4-core CPU (jax 0.12.0). Within a block every arm ran on the same platform; between-platform shifts
  measured so far are inside the noise.
* **Modality caveat.** The paper baselines use the 63x63x3 image; our runs use the 1345-d symbolic observation (operator decision 2026-10-01: image rejected for compute). Our G1.0/G1.1 numbers
  calibrate the pipeline for this modality only and may be easier than the image setting.
* **Equal effort (P9) only roughly met.** Truck had more recipe screening (~160 jobs) than the baseline (~30 plus TASK-023's search); GRU-TP's tuning is if anything less than Truck's.
* **Integrity.** Audit TASK-022 found no fabricated or leaked number; 16 early tuning files came from uncommitted code (provenance only); the TASK-017 Truck result is unreliable. The
  evaluator is post auto-reset fix. No simulator access at decision time in any gate arm.
* **Not tested.** Image observation, Dyna-style use of the learned TP model, anything above 4M parameters for the Truck family, TASK-021 (Kaggle TPU tuning, never executed).

## 8. Tuning attempt and diagnostics (TASK-026, this session)

Pre-registration with five amendments, each written before the runs it covers: `README.md` in this directory. Per-seed result files: `results/` in this directory (plain-text copies of
`output/phase1/g12_t026/`, which is LFS-tracked and could not be pushed from the cloud session: no Hugging Face credentials). Platform: 4-core CPU, jax 0.12.0, Python 3.13, commit recorded in each file;
role `tune`, tuning seeds 4000-4999 (never used before, disjoint from every final seed), 999,424 env steps, symbolic 1345-d, sampled policy, 256 evaluation envs per seed. No config was selected on, and no gate was read from, these seeds.

**8.1 One-factor screen around GRU-TP (R0) at the EP-A budget** (mean over seeds 4000-4001; R0 over 4000-4003 = 48.83 / 17.67 +- 0.45 / 0.67; qualification line = R0 + 2.0 reward on the same seeds = 50.62)

| id | change vs R0 | params_total | reward | score | qualifies |
|---|---|---|---|---|---|
| R0 | none (GRU-TP) | 1.25M | 48.62 (2 seeds), 48.83 (4 seeds) | 18.12 / 17.67 | reference |
| R1 | lr 2e-3 | 1.25M | 48.70 | 19.32 | no (+0.1) |
| R2 | lr 5e-4 | 1.25M | 44.82 | 14.65 | no (-3.8) |
| R3 | entropy 0.02 | 1.25M | 46.88 | 18.21 | no (-1.7) |
| R4 | gamma 0.99, lambda 0.8 | 1.25M | 45.84 | 17.19 | no (-2.8) |
| R5 | width 512 | 3.64M | 48.43 (2 seeds); 48.93 +- 0.50 / 19.47 +- 0.44 (4 seeds) vs R0 48.83 / 17.67 | 19.10 | no (reward +0.10 +- ~0.5 on 4 seeds; score +1.8) |
| R6 | 32 envs x 128 steps | 1.25M | 48.14 | 18.75 | no (-0.5) |
| R7 | 8 PPO epochs | 1.25M | 41.18 | 13.38 | no (-7.4: more reuse of the batch hurts) |
| R8 | 32 envs x 64 steps (2x updates) | 1.25M | 45.60 | 16.86 | no (-3.0) |
| R9 | 16 envs x 64 steps (4x updates) | 1.25M | 43.85 | 14.35 | no (-4.8) |
| R10 | 32 envs x 32 steps (4x updates) | 1.25M | 45.36 | 15.26 | no (-3.3) |

More PPO updates per environment step (R7, R8, R9, R10) is worse in every variant tried: the sample-efficiency gap is not closed by update count.
Best tuning-seed mean among all configs: **48.93 (R5, width 512, 4 seeds; R1 48.70 on 2 seeds)** against the stop line of 54.0, so by the pre-registered rule **no final (EP-A) block was run and no gate is claimed.** Width 512 improves score by +1.8 on 4 seeds
but score is not the failing metric. Tuning on 2-4 seeds can only detect effects of about 2 reward or more (R0 sd 0.9); a +7.7 effect would have been obvious and did not occur.

**8.2 Data-scaling diagnostic (labelled `diag-*`, not EP-A, not a gate; independent runs, lr schedule scaled to the run length; R0; seeds 4000-4001 beyond 1M)**

| env steps | seeds | reward | score | eval length | stone pickaxe / sword | coal | iron | zombie | skeleton | wake_up |
|---|---|---|---|---|---|---|---|---|---|---|
| 1M (EP-A budget) | 4 | 48.83 +- 0.45 | 17.67 +- 0.67 | 235 | 17 / 5 % | 25 % | 1 % | 13 % | 3 % | 72 % |
| 2M | 2 | 57.66 +- 0.90 | 25.59 +- 1.84 | 280 | 63 / 56 % | 65 % | 6 % | 9 % | 6 % | 63 % |
| 3M | 2 | 66.34 +- 0.80 | 33.41 +- 0.30 | 278 | 77 / 76 % | 74 % | 16 % | 91 % | 23 % | 69 % |
| 4M | 2 | 65.53 +- 0.15 | 31.86 +- 0.24 | 283 | 81 / 84 % | 78 % | 14 % | 91 % | 31 % | 15 % |
| G1.2 (MFRL_best, Dedieu 2025 Table 1, 55.6M params, image obs, 1M steps) | | 55.49 | 16.77 | | | | | | | |

Reading: at 2x the budget the policy already clears G1.2's thresholds with n = 2 (reward - 2 SE 55.87); the training return was still rising at 4M; the plateau near 66 (3M-4M) is close to the
transformer-world-model + Dyna result of Dedieu et al. (MBRL_fast 67.42 / 27.91 at 58.5M params, 1M steps; MASTER_GUIDANCE). eat_plant, diamond and iron tools stay at 0 %. Caveats: 2 seeds (the 1M row uses 4 seeds, 4000-4003; the others 4000-4001), single config, CPU platform; the shape of the curve (not the exact numbers) is the finding.

## 9. Recommendation

* Do not treat Phase 1 as closed. Phase 2 stays locked; MASTER_GUIDANCE `current_phase` stays 1. No Phase-2 candidate was designed or built (the request was conditional on a closed Phase 1).
* The operator should decide whether GRU-TP's second-attempt pass closes G1.1 (the roadmap defines no retry rule; nothing was selected or dropped, all 7 blocks are reported).
* **Knob tuning is exhausted for G1.2.** Ten single-knob variants at 1M steps (TASK-026) and ~200 earlier jobs found nothing that moves reward by more than ~1 beyond TP.
* **The next lever must raise learning per environment step by about 1.7-2x** (the 2M-step point crosses G1.2). Candidate directions, none implemented (strictness: unconfirmed components; each needs an operator decision,
  and the first two need a GPU to be practical):
  (a) Dyna-style use of the learned transition-prediction head: imagined rollouts/updates from a one-step model trained on executed transitions only (P4); env steps unchanged (P10), compute and the model counted in
      params_total. Supporting evidence: TP alone gave +3.1 reward (the one auxiliary signal that helped), and Dedieu et al. reach 67.4 reward at 1M steps with a transformer world model + Dyna (58.5M params, MASTER_GUIDANCE MBRL_fast).
  (b) Off-policy / replay-based learner with high update-to-data ratio. Evidence against naive reuse inside PPO: 8 epochs per batch lost 7 reward (R7); K1/K2 (2x updates) gave nothing on Truck.
  (c) More auxiliary learning signals of the TP kind (what helped), screened on tuning seeds first. EV (contrastive next-reward event) hurt, so the choice matters.
  (d) Image observation + CNN as in the reference (rejected earlier on compute grounds).
* Compute: the cloud session has no GPU; a 1M-step GRU run takes ~25-28 min on 2 CPU cores (two run in parallel), a 4M-step run ~92 min. A 10-seed GRU gate block is ~2.5-3 h with 2 workers, a Truck block roughly double.
  This session's CPU path (`scripts/run_t026_cpu_sweep.py`) is usable for GRU-class screens and diagnostics.

## 10. Sources

* `docs/core/STATE.yaml` tasks TASK-20260930-017, TASK-20261001-020, TASK-20261004-022, TASK-20261004-023, TASK-20261005-024, TASK-20261008-025.
* `docs/experiments/2026-10-08_phase1_push/{README,RESULTS}.md` (per-block tables, all per-seed values, achievement table), `2026-10-05_truck_scale_study/RESULTS.md`,
  `2026-10-04_truck_recipe_plan/RESULTS_2026-10-05.md`, `2026-09-30_phase1_model_families/README.md`, `2026-10-04_phase1_score_audit/`.
* `docs/core/MASTER_GUIDANCE.xml` roadmap (gates, EP-A, references with source ids MFRL_4M_reimpl, MFRL_best, MBRL_fast; caveat C1-C3).
* Knowledge-DB: `repo-notes/decision-process-order-driven/research-results-summary-phase1-2026-10.md` (state of 2026-10-10; updated by TASK-026).
