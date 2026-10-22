# TASK-20261022-027 results: sample-efficiency ideas for G1.2 (executed 2026-10-22 00:15 .. ~10:30 JST)

Pre-registration and amendments 0-7(+): `README.md` in this directory. Protocol EP-A for the final blocks (999,424 env steps, first episode of 256 fresh envs per seed, sampled policy,
final parameters, clean commit, Kaggle T4), EP-A-tuning for screening. References quoted from MASTER_GUIDANCE (Dedieu et al. 2025 Table 1, MFRL_best 55.49 / 16.77 for G1.2).
Raw result files: `output/phase1/push_t026/` (LFS; this machine has no git-lfs, so plain copies are committed under `results/` here).

## 1. Bottom line
* **G1.2 is not reached and Phase 1 stays open.** The best arm of any kind (reward-coefficient 0 + entropy 0.005 + width 512, P5-contested) has block-J reward 52.21 +- 0.61 (-2SE 50.98) against 55.49 needed.
* **No P5-clean lever moved reward.** lr schedule shape (cosine, knee), lr 2e-3, entropy, lambda, gamma, width, and the new latent-prediction auxiliary (LP) were neutral or harmful on fresh seeds.
  On the pre-registered block J, LP vs the same-seed control: reward +0.28 +- 0.60, score +0.86 +- 0.38 (control GRU-TP 49.33 +- 0.35 / 19.01 +- 0.46; LP 49.61 +- 0.55 / 19.88 +- 0.41). Its tuning-seed gains (+2.6, +1.1) were selection noise.
* **The one lever that moved the numbers is the operator's idea 2, and it is reward shaping.** Zeroing the +-0.1 health terms of the TRAINING reward (`rew_small=0`) gave +3.4 reward / +3.6 score on two independent
  sets of tuning seeds (4000-4003, 5000-5003; sd over seeds 0.6-1.9) and a monotone dose-response (coefficient 1 / 0.5 / 0.25 / 0 -> 47.4 / 48.5 / 49.4 / 50.8 reward). On the fresh final block the stacked arm gained +2.9 +- 0.8 reward over
  the same-seed control (z 3.5), the smaller stack (rs0 + LP) +1.0 +- 1.1: less than the tuning seeds suggested. It violates P5 / EP-A ("reward unchanged"), so its result files carry the protocol
  label `EP-A-P5-contested-reward-coefficient`, MLflow tags them gate_eligible=false with their own arm labels (`+rs0`), and **it cannot count toward any gate unless the operator amends P5.**
* **Why it is not a bigger win:** the data-scaling diagnostic of TASK-026 (2M steps -> 57.7 reward) says G1.2 needs about 1.7-2x data efficiency; the best contested arm closes about +3 of the ~+9 reward that 2x data gives in that diagnostic (no efficiency factor is claimed from this).

## 2. The three operator ideas, as tested
1. **LR schedule that leaves the steep phase quickly.** The harness already anneals linearly to 0 with 5 % warm-up (not new). Tested shapes (GRU-TP, tuning seeds 4000-4003, baseline r+s 64.5): cosine 2e-3 63.4 (cosine at 1e-3 was defined but not run),
   knee (2e-3, x0.3 at 25 % then linear to 0) 67.0, lr 2e-3 66.2; with rs0 on seeds 5000-5003 (baseline rs0 71.9): knee 70.5, lr 2e-3 70.8; with LP: knee 64.9. None reaches the pre-registered +3.0 line reliably; the effects are within noise.
2. **Reward coefficient learned from part of the 1M steps.** Critical review (README): this is reward shaping (P5), a per-achievement weighting needs achievement identity (external knowledge) and was NOT tried; plain scaling is a no-op under value normalisation. Only a coefficient on the
   |r| < 0.5 terms (the health terms, Craftax-Classic reward = achievements + 0.1 x delta health) was run. Because the effect is monotone and ends at coefficient 0, a "learned" coefficient would have to converge to 0, i.e. "ignore health reward"; I did not build the learning
   procedure (no principled criterion that does not already encode that the metric counts achievements). Operator decision needed: amend P5 (then re-run as an EP-A gate block with the coefficient declared, equal effort for the baseline) or keep the rule.
3. **Parameter adjustment.** One-at-a-time neighbours of GRU-TP and of the rs0 stacks: lambda 0.9 (-12 reward), gamma 0.99, entropy 0.02 harmful; entropy 0.005 and width 512 mildly positive only together with rs0 (block J: 52.2); width 768 / 1024 and a wider TP head: see section 4. TASK-026 found the same null result for the plain arm.

## 3. Screening tables (Kaggle T4, 4 seeds each; reward / score; r+s = sum)

**R1 (seeds 4000-4003): lr shape, width, entropy, lambda, health coefficient, Truck**

| config | n | params_total | reward | score | r+s | per-seed reward |
|---|---|---|---|---|---|---|
| G4TP1 | 4 | 1,254,358 | 47.11 +- 0.79 | 17.39 +- 1.03 | 64.5 | 48.7 48.2 46.2 45.4 |
| G_cos2 | 4 | 1,254,358 | 47.28 +- 0.46 | 16.15 +- 0.38 | 63.4 | 47.2 46.5 48.6 46.8 |
| G_ent005 | 4 | 1,254,358 | 48.84 +- 0.61 | 18.81 +- 1.28 | 67.7 | 49.7 48.3 47.4 50.0 |
| G_kn2a | 4 | 1,254,358 | 47.99 +- 0.94 | 18.99 +- 0.91 | 67.0 | 46.2 48.2 50.6 47.0 |
| G_lam9 | 4 | 1,254,358 | 39.32 +- 0.70 | 12.80 +- 0.50 | 52.1 | 38.0 40.9 38.3 40.1 |
| G_lr2 | 4 | 1,254,358 | 48.49 +- 1.07 | 17.75 +- 0.67 | 66.2 | 48.4 48.9 45.8 50.9 |
| G_rs0 | 4 | 1,254,358 | 50.89 +- 0.32 | 21.15 +- 0.91 | 72.0 | 51.7 50.8 50.1 50.9 |
| G_rs2 | 4 | 1,254,358 | 47.98 +- 0.20 | 17.03 +- 0.53 | 65.0 | 47.5 48.1 47.9 48.4 |
| G_w512 | 4 | 3,640,790 | 50.12 +- 0.66 | 19.55 +- 0.99 | 69.7 | 51.8 50.3 49.8 48.6 |
| TP1h | 4 | 484,117 | 45.94 +- 0.49 | 16.52 +- 0.68 | 62.5 | 45.7 47.3 45.7 45.0 |
| T_kn2a | 4 | 484,117 | 45.84 +- 2.38 | 16.48 +- 2.16 | 62.3 | 43.5 50.1 49.5 40.3 |
| T_lr2 | 4 | 484,117 | 46.99 +- 0.79 | 18.57 +- 0.35 | 65.6 | 48.5 48.2 45.7 45.5 |

**R2 (5000-5003): latent prediction**

| config | n | params_total | reward | score | r+s | per-seed reward |
|---|---|---|---|---|---|---|
| G4TP1 | 4 | 1,254,358 | 47.43 +- 1.10 | 17.39 +- 1.05 | 64.8 | 47.4 44.4 49.4 48.5 |
| G_lp03 | 4 | 1,521,622 | 48.39 +- 0.23 | 17.93 +- 0.93 | 66.3 | 48.8 47.7 48.4 48.7 |
| G_lp1 | 4 | 1,521,622 | 50.03 +- 1.48 | 20.56 +- 1.59 | 70.6 | 47.9 47.7 50.3 54.1 |
| G_lp1k6 | 4 | 1,521,622 | 48.63 +- 0.89 | 19.65 +- 0.62 | 68.3 | 46.5 50.5 49.7 47.9 |
| G_lp3 | 4 | 1,521,622 | 47.83 +- 0.29 | 19.44 +- 0.56 | 67.3 | 48.3 48.3 47.3 47.4 |

**R3 (5000-5003): health coefficient 0 stacked**

| config | n | params_total | reward | score | r+s | per-seed reward |
|---|---|---|---|---|---|---|
| G_rs0 | 4 | 1,254,358 | 50.83 +- 0.94 | 21.03 +- 1.03 | 71.9 | 51.5 51.7 48.0 52.1 |
| G_rs0_ent02 | 4 | 1,254,358 | 45.97 +- 1.64 | 17.78 +- 1.35 | 63.7 | 44.8 43.7 50.8 44.5 |
| G_rs0_g99 | 4 | 1,254,358 | 48.21 +- 0.70 | 18.27 +- 0.85 | 66.5 | 49.8 46.6 48.9 47.5 |
| G_rs0_kn2a | 4 | 1,254,358 | 50.33 +- 0.51 | 20.13 +- 0.91 | 70.5 | 50.0 49.2 50.5 51.6 |
| G_rs0_lam9 | 4 | 1,254,358 | 40.30 +- 0.70 | 13.68 +- 0.45 | 54.0 | 39.3 39.3 40.3 42.3 |
| G_rs0_lr2 | 4 | 1,254,358 | 50.42 +- 0.61 | 20.42 +- 0.37 | 70.8 | 50.1 51.3 51.5 48.8 |

**R4 (5000-5003): coefficient sensitivity, rs0 + entropy/width, Truck**

| config | n | params_total | reward | score | r+s | per-seed reward |
|---|---|---|---|---|---|---|
| G_rs025 | 4 | 1,254,358 | 49.44 +- 0.96 | 19.66 +- 1.02 | 69.1 | 52.2 49.0 47.9 48.6 |
| G_rs05 | 4 | 1,254,358 | 48.50 +- 0.58 | 18.82 +- 0.89 | 67.3 | 48.0 49.7 49.2 47.1 |
| G_rs0_ent005 | 4 | 1,254,358 | 49.72 +- 0.78 | 19.07 +- 1.42 | 68.8 | 47.4 50.8 50.1 50.5 |
| G_rs0_ent005_w512 | 4 | 3,640,790 | 52.93 +- 1.18 | 21.72 +- 1.59 | 74.6 | 49.6 55.2 53.6 53.3 |
| G_rs0_w512 | 4 | 3,640,790 | 51.40 +- 1.06 | 21.61 +- 0.98 | 73.0 | 53.3 52.7 50.9 48.6 |
| T_rs0 | 4 | 484,117 | 48.17 +- 2.10 | 18.97 +- 2.53 | 67.1 | 44.6 52.6 50.9 44.5 |

**R5 (5000-5003): LP combinations**

| config | n | params_total | reward | score | r+s | per-seed reward |
|---|---|---|---|---|---|---|
| G_lp1_ent005 | 4 | 1,521,622 | 49.65 +- 0.39 | 18.68 +- 0.34 | 68.3 | 50.8 49.0 49.3 49.5 |
| G_lp1_kn2a | 4 | 1,521,622 | 47.21 +- 0.52 | 17.68 +- 1.16 | 64.9 | 46.4 47.4 46.4 48.6 |
| G_lp1_w512 | 4 | 4,170,710 | 47.91 +- 0.62 | 17.54 +- 1.27 | 65.5 | 47.9 46.9 47.2 49.6 |
| G_rs0_lp1 | 4 | 1,521,622 | 52.13 +- 0.51 | 22.63 +- 0.55 | 74.8 | 52.9 50.8 52.9 51.9 |

**R5c (6000-6003): confirmation of LP**

| config | n | params_total | reward | score | r+s | per-seed reward |
|---|---|---|---|---|---|---|
| G4TP1 | 4 | 1,254,358 | 48.54 +- 1.01 | 18.72 +- 1.13 | 67.3 | 50.1 50.5 47.1 46.5 |
| G_lp1 | 4 | 1,521,622 | 49.61 +- 1.01 | 19.01 +- 1.07 | 68.6 | 48.3 47.7 50.4 52.1 |

**R6 (5000-5003): full stacks**

| config | n | params_total | reward | score | r+s | per-seed reward |
|---|---|---|---|---|---|---|
| G_lp1_w512_ent005 | 4 | 4,170,710 | 48.74 +- 0.61 | 16.33 +- 0.76 | 65.1 | 47.3 49.2 50.1 48.3 |
| G_rs0_lp1_ent005 | 4 | 1,521,622 | 52.32 +- 0.54 | 21.18 +- 0.81 | 73.5 | 53.5 52.1 50.9 52.8 |
| G_rs0_lp1_w512 | 4 | 4,170,710 | 51.60 +- 1.03 | 21.11 +- 0.57 | 72.7 | 51.5 48.7 53.4 52.8 |
| G_rs0_lp1_w512_ent005 | 4 | 4,170,710 | 52.34 +- 1.06 | 21.22 +- 0.81 | 73.6 | 52.1 53.3 54.5 49.5 |


## 4. Width round R7 (kernel j, seeds 5000-5003; reference rs0 + entropy 0.005 + width 512 = 52.93 / 21.72)
| config | n | params_total | reward | score | r+s | per-seed reward |
|---|---|---|---|---|---|---|
| G_rs0_ent005_w1024 | 4 | 12,345,814 | 51.12 +- 0.34 | 20.28 +- 0.44 | 71.4 | 51.1 51.7 51.6 50.2 |
| G_rs0_ent005_w1024_tph512 | 4 | 13,656,022 | 51.90 +- 1.11 | 20.76 +- 1.13 | 72.7 | 54.4 52.4 49.0 51.9 |
| G_rs0_ent005_w512_tph512 | 4 | 4,557,782 | 52.68 +- 0.56 | 22.06 +- 1.39 | 74.7 | 51.7 51.8 54.0 53.3 |
| G_rs0_ent005_w768 | 4 | 7,337,942 | 54.35 +- 0.45 | 23.75 +- 0.11 | 78.1 | 54.8 53.0 54.6 54.9 |

Reading: width 768 reached 54.35 +- 0.45 / 23.75 +- 0.11 (r+s 78.1), the highest tuning-seed number of the task, but the width response is non-monotone (512: 52.93, 768: 54.35, 1024: 51.12; a 512-wide TP head 52.68), it is the best of four configs
screened on the same four seeds, and TASK-026 / R1 saw width 512 shrink on fresh seeds. It is a P5-contested arm (rs0). **A lead, not a result**: no 10-seed final block fitted before the 11:30 deadline (about 70 min on one kernel slot after 10:25),
so nothing about width 768 is gate evidence. The next step for an operator who admits the reward coefficient would be a pre-registered 10-seed block of G_rs0_ent005_w768 (7.3M params) against a same-seed control; G1.2 would still need about +2 reward on the tuning-seed estimate even if it held (55.49 needed at mean - 2 SE, i.e. mean about 56.4).

## 5. Final blocks (EP-A, clean commit, Kaggle T4; `python scripts/summarize_t026.py`)
| arm | n | protocol | params_total / deployed | reward | score | G1.1 | G1.2 | commit (dirty) |
|---|---|---|---|---|---|---|---|---|
| G4TP1 | 10 | EP-A | 1,254,358 / 879,763 | 49.33 +- 0.35 (-2SE 48.62) | 19.01 +- 0.46 (-2SE 18.09) | PASS | fail | 0a085597 (False) |
| G_lp1 | 10 | EP-A | 1,521,622 / 879,763 | 49.61 +- 0.55 (-2SE 48.52) | 19.88 +- 0.41 (-2SE 19.06) | PASS | fail | 0a085597 (False) |
| G_rs0_lp1 | 10 | EP-A-P5-contested-reward-coefficient | 1,521,622 / 879,763 | 50.35 +- 0.84 (-2SE 48.66) | 20.28 +- 0.65 (-2SE 18.99) | met (P5-contested: not eligible) | fail (P5-contested: not eligible) | 311b30d1 (False) |
| G_rs0_ent005_w512 | 10 | EP-A-P5-contested-reward-coefficient | 3,640,790 / 2,805,395 | 52.21 +- 0.61 (-2SE 50.98) | 20.32 +- 0.78 (-2SE 18.76) | met (P5-contested: not eligible) | fail (P5-contested: not eligible) | 311b30d1 (False) |

Paired difference to the same-seed control G4TP1 (mean +- SE of the 10 per-seed differences; seed s uses test seed s+382 for every arm):

* G_lp1 - G4TP1 (n=10): reward +0.28 +- 0.60 (z 0.5), score +0.86 +- 0.38 (z 2.3)
* G_rs0_lp1 - G4TP1 (n=10): reward +1.02 +- 1.09 (z 0.9), score +1.27 +- 0.96 (z 1.3)
* G_rs0_ent005_w512 - G4TP1 (n=10): reward +2.88 +- 0.81 (z 3.5), score +1.31 +- 1.07 (z 1.2)

Block K (train 182-191 / test 564-573), G_rs0_ent005_w512 (n=10): reward 53.10 +- 0.47 (-2SE 52.16), score 21.86 +- 0.52 (-2SE 20.82); G1.2 fail
Pooled blocks J+K (n=20, descriptive): reward 52.66 +- 0.39, score 21.09 +- 0.49

Per-seed reward:
* G4TP1: 172:47.5, 173:49.3, 174:50.0, 175:47.4, 176:50.2, 177:50.9, 178:49.8, 179:49.8, 180:49.5, 181:48.9
* G_lp1: 172:49.2, 173:51.9, 174:49.3, 175:48.8, 176:49.5, 177:52.2, 178:47.7, 179:47.4, 180:48.5, 181:51.5
* G_rs0_lp1: 172:51.2, 173:52.3, 174:50.8, 175:53.9, 176:43.9, 177:49.5, 178:49.9, 179:50.7, 180:49.2, 181:52.1
* G_rs0_ent005_w512: 172:53.7, 173:53.6, 174:53.1, 175:52.8, 176:49.1, 177:53.1, 178:48.4, 179:52.6, 180:51.6, 181:54.0


## 6. Gate status
G1.0 / G1.1 as in TASK-025 (G1.1 passed on a pre-registered block at its second attempt by GRU-TP; operator ruling pending). G1.2: **FAIL for every arm**, P5-clean and P5-contested. No claim of Phase-1 closure.

## 7. What is open / recommended next
* Operator decisions: (a) is any reward-coefficient arm admissible (P5 amendment)? (b) does GRU-TP's second-attempt G1.1 pass close G1.1?
* The remaining distance to G1.2 (reward +4 to +6 beyond the best contested arm, +7 beyond the clean arm) is a 1.7x data-efficiency gap. Known routes that were not built tonight: Dyna-style imagination with a learned model (Dedieu et al. reach 67.4 at 1M steps),
  an off-policy / high update-to-data learner (more epochs and smaller batches hurt the PPO learner), image modality. Each needs a design decision.
* Caveats: tuning seeds 4000-4003 overlap TASK-026's; the final-run tuning budget (~35 configs x 4 seeds) is larger than the control's (none: control = TASK-025 config), which favours the tuned arms (P9 note); LP/rs0 screening gains shrank on fresh seeds (winner's curse).
