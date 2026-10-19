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
4. **Even a perfect mix of the arms would not reach G1.2** (section 6): taking the best unlock rate of any arm for every achievement gives 52.6 reward.

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
* Hence G1.2 is not a "last few points" problem of the existing family: reward counts achievements, and +7.7 points means unlocking one to two achievements more per episode that
  no arm unlocks at all today.

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

## 8. Tuning attempt (TASK-026, this session)

(see the end of this file; filled in after the runs finished)

## 9. Recommendation

* Do not treat Phase 1 as closed. Phase 2 stays locked; MASTER_GUIDANCE `current_phase` stays 1.
* The operator should decide whether GRU-TP's second-attempt pass closes G1.1 (the roadmap does not define a retry rule; nothing was selected or dropped, all 7 blocks are reported).
* For G1.2 the evidence points at a new capability, not tuning: survival (zombie deaths at ~200 steps) and deep-chain credit assignment (stone tools -> iron). Candidate directions, none
  implemented (strictness: unconfirmed components): (a) Dyna-style planning/imagination with the TP head (a learned one-step model; env steps unchanged under P10, compute and params
  counted; Dedieu et al. report 67.4 reward for a Transformer world model + Dyna at 58.5M params, so the ceiling exists but at far higher cost); (b) a larger replay / more updates per
  step via an off-policy learner; (c) image modality + CNN to match the reference. Each needs an operator decision and a GPU.
* Compute: the cloud session has no GPU; a 1M-step GRU run takes ~28 min on 4 CPU cores, Truck arms ~2x. A 10-seed gate block of a GRU arm is ~5 h, a Truck block ~10 h on this machine.

## 10. Sources

* `docs/core/STATE.yaml` tasks TASK-20260930-017, TASK-20261001-020, TASK-20261004-022, TASK-20261004-023, TASK-20261005-024, TASK-20261008-025.
* `docs/experiments/2026-10-08_phase1_push/{README,RESULTS}.md` (per-block tables, all per-seed values, achievement table), `2026-10-05_truck_scale_study/RESULTS.md`,
  `2026-10-04_truck_recipe_plan/RESULTS_2026-10-05.md`, `2026-09-30_phase1_model_families/README.md`, `2026-10-04_phase1_score_audit/`.
* `docs/core/MASTER_GUIDANCE.xml` roadmap (gates, EP-A, references with source ids MFRL_4M_reimpl, MFRL_best, MBRL_fast; caveat C1-C3).
* Knowledge-DB: `repo-notes/decision-process-order-driven/research-results-summary-phase1-2026-10.md` (state of 2026-10-10; updated by TASK-026).
