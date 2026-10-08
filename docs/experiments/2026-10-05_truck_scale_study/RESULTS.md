# TASK-20261005-024 results: Truck seed variance, parameter scale, architecture attribution (executed 2026-10-05 .. 2026-10-08)

Local RTX 3060 Ti in Docker (jax 0.10.2, flax 0.12.8, craftax 1.6.1; `output/phase1/recipe_t023/environment.txt`). Pre-registration and eleven amendments (each written before the jobs it covers): `README.md` in this directory.
Files: `output/phase1/truck_scale_t024/{AB,C,D,E,FS,X,R,baseline,final}/`; reports: `python scripts/run_truck_scale_sweep.py report --stage <S>`, `python scripts/run_baseline_warmup_check.py report`,
`python scripts/adhoc/agg_final.py <seed-lo>-<seed-hi> <config...>`. MLflow (`mlflow_local/`, rebuilt in this checkout): 343 runs, 0 UNREGISTERED.

## 1. Summary
* Truck's large seed variance was **mostly an optimisation-recipe problem, not a parameter-count problem**: 5 % lr warm-up + entropy 0.01 + lr 1e-3 + gamma 0.97 raised Truck-improved (0.284M) from 39.29 +- 2.17 reward / 11.08 score
  (TASK-023 final) to **47.12 +- 0.99 / 17.27 +- 0.84** (20 EP-A seeds). Its seed sd fell from 6.9 to 4.4 but is still ~3.3x the re-tuned baseline's.
* **Parameter scale** (hidden dimension): a plateau from 0.28M to 0.93M on final seeds, a drop at 0.14M (tuning seeds), no gain and growing seed spread above 0.5M. The small model keeps its edge; scaling up is not a lever.
* **Truck vs baseline** (baseline re-tuned with the same knobs, G4 gru256+ln+skip 1.01M: 45.42 +- 0.30 / 15.48 +- 0.27 over 20 seeds): Truck-improved is ahead by +1.70 +- 1.03 reward (z 1.65) and +1.78 +- 0.88 score (z 2.03):
  probably slightly better, not clearly so, with 3.6x fewer parameters.
* **Attribution**: with the stable recipe, removing the candidate-action tokens changes nothing measurable (reward -0.00 +- 1.9, score -0.9 +- 1.6 on 10 seeds); removing memory token + previous action + head skip together costs about 3 score points
  (X5 vs E1, z 2.4) and ~1.4 reward (not significant). The 4th-idea readout (candidate tokens) is not shown to contribute; the model works as a Transformer feature encoder + GRU + linear heads.
* **Gates**: G1.0 PASS; G1.1 **not established** (reward 48.54 in the first 10-seed block, 45.70 in the replication block, pooled 47.12 < 47.40; score condition holds); G1.2 FAIL (reward 47.12 < 55.49). Phase 1 stays open.

## 2. Answers to the operator's questions
* **Cause of the variance.** TASK-023's recipe point used lr 1.5e-3, entropy 0.003, no warm-up, gamma 0.99; early PPO updates clipped 28-41 % of ratios, and training curves separated in learning speed within the first 30 % of the steps.
  On tuning seeds 2000-2002 the old Truck gave 35.97 reward (sd 3.6). Stage by stage (tuning seeds, 3-4 each): warm-up 5 % + entropy 0.01 -> 42.0-44.4 (sd 2.8-3.0); + lr 1e-3 -> 46.5 (sd 2.0); + gamma 0.97 -> 48.2 (sd 1.3; E1).
  Worse: lr 7e-4 / 3e-4, 2 epochs, larger batches (128 / 256 envs: fewer updates), clip 0.1, vf 0.5, 128-step rollouts, value-norm 0.95. No gain: entropy 0.02 / 0.005, warm-up 15 %, gamma 0.95, lambda 0.55, 4 epochs, 16 minibatches (R2 48.7, equal to E1).
  Batch size was therefore not the cause (the operator's suspicion): larger batches did not reduce variance or improve the mean.
* **Bigger or smaller hidden dimension.** At the old recipe, d >= 128 was clearly worse (25-31 reward; d256 collapsed to 12 as entropy fell too fast): an lr/size interaction. At the stable recipe (tuning seeds, 4 each): 0.14M 40.8; 0.28M 48.2;
  0.47M 48.9; 0.93M 47.4; 1.66M 46.5 reward, with seed sd 2.4, 1.3, 3.4, 6.0, 7.7. Finals (10 seeds, block 1): 0.28M 48.54, 0.47M 46.17, 0.93M 47.15. Hidden dimension is not a lever beyond ~0.3M; layers / sequence length were not varied.

## 3. Final results (EP-A; 256 fresh envs per seed; ~1M env steps; clean tree; train seeds / test seeds never used before)
Block 1 = train 62-71 / test 444-453; block 2 = train 72-81 / test 454-463.

| model | params_total | seeds | reward_pct | score_pct |
|---|---|---|---|---|
| **Truck-improved E1** (d64, GRU w128, +mem+pa+hs) | 0.284M | block 1 (10) | 48.54 +- 1.50 (sd 4.8) | 17.78 +- 1.25 |
| | | block 2 (10) | 45.70 +- 1.18 (sd 3.7) | 16.76 +- 1.15 |
| | | **pooled (20)** | **47.12 +- 0.99** (sd 4.4) | **17.27 +- 0.84** |
| baseline of record G4 (gru256+ln+skip, 4 epochs, same recipe point) | 1.013M | block 1 | 45.68 +- 0.37 | 15.42 +- 0.33 |
| | | block 2 | 45.15 +- 0.47 | 15.54 +- 0.46 |
| | | **pooled (20)** | **45.42 +- 0.30** (sd 1.3) | **15.48 +- 0.27** |
| Truck S2 (GRU w256) | 0.467M | block 1 | 46.17 +- 0.95 | 16.87 +- 0.90 |
| Truck S3 (d128, w256) | 0.928M | block 1 | 47.15 +- 1.19 | 17.46 +- 1.12 |
| Truck X1: E1 without candidate tokens | 0.282M | block 2 | 45.70 +- 1.45 | 15.88 +- 1.16 |
| Truck X5: E1 without mem / pa / hs (original structure) | 0.273M | block 2 | 44.26 +- 1.06 | 13.76 +- 0.51 |
| (TASK-023, old recipe) Truck-improved / baseline gru512 | 0.467M / 2.80M | 52-61 | 39.29 +- 2.17 / 41.56 +- 0.28 | 11.08 +- 1.25 / 13.59 +- 0.17 |

Per-seed values are in the result files and in README. Differences (E1 - G4): block 1 +2.86 +- 1.55 / +2.35 +- 1.30; block 2 +0.55 +- 1.27 / +1.21 +- 1.24; pooled +1.70 +- 1.03 / +1.78 +- 0.88 (reward / score).
Block 2, same seeds: E1 - X1 = 0.00 +- 1.87 reward, +0.88 +- 1.63 score; E1 - X5 = +1.44 +- 1.58 reward, +3.00 +- 1.26 score; X5 - G4 = -0.89 +- 1.16 reward, -1.78 +- 0.69 score.

## 4. Gates (numbers from this file and MASTER_GUIDANCE; protocol EP-A)
* **G1.0 PASS**: baseline G4 reward 45.42 >= 42.7 and score 15.48 >= 9.6 (both blocks pass individually). It required giving the baseline the same recipe knobs (section 6); with TASK-023's recipe it failed (41.56).
* **G1.1 NOT ESTABLISHED**: needs reward > 47.40 and score > 10.71 with params_total <= 4.0M. Block 1 alone passed (48.54 / 17.78), the pre-registered replication did not (45.70), pooled reward is 47.12 +- 0.99 (below the threshold by 0.3 SE).
  The score condition (> 10.71) holds clearly in both blocks; the parameter condition holds (0.284M). The reward condition is undecided within noise; do not cite G1.1 as passed.
* **G1.2 FAIL**: reward 47.12 < 55.49 (score 17.27 > 16.77 holds only marginally; both are required).
* Phase 1 stays open. Reference numbers were not edited.

## 5. Recipe round R (tuning seeds 2000-2003; E1 on the same seeds 48.15 / 17.19)
R1 clip 0.1: 43.4 / 15.2; R2 16 minibatches: 48.7 / 17.6; R3 vf 0.5: 45.5 / 16.0; R4 entropy 0.005: 48.7 / 16.7; R5 128-step rollouts: 43.7 / 14.5; R6 value-norm 0.95: 45.6 / 16.0. None exceeds E1 by the pre-registered 4 r+s points, so no R final was run:
the recipe has plateaued at ~48 reward on the tuning seeds within noise.

## 6. Fairness and caveats (all recorded, none hidden)
* Equal effort (P9) is only roughly met: Truck had ~160 screening jobs (stages AB, C, D, E, FS, X, R; seeds 2000-2005) after TASK-023; the baseline had TASK-023's search plus 10 configs x 3 seeds at Truck's recipe point (B1-B6, G1-G4).
  The baseline gained +4 reward from Truck's recipe point (gru512: 41.56 -> G4 45.68 on block 1), so the earlier "Truck is not better than the baseline" statement (TASK-020 / 023) was partly a recipe artefact for both arms. A larger baseline effort could still narrow the remaining gap.
* Tuning seeds 2000-2005 were reused across many comparisons: tuning numbers are optimistic (S2 48.89 tuning vs 46.17 final; E1 48.15 vs 47.12 pooled). The replication block was reserved and untouched until its jobs.
* The first 10-seed block (48.54) was a favourable draw for E1; this is why the G1.1 claim made earlier in this task's first report was withdrawn. S2 / S3 were only run on block 1, X1 / X5 only on block 2, so cross-block comparisons mix seed sets.
* Stage AB ran only 45 of 96 planned jobs (its deadline passed during an idle session; 2 concurrent jobs ran 2.5x slower each, later stages ran serially). Scale claims at the old recipe rest on 2-4 seeds.
* The automatic baseline selector broke a tie by dictionary order (B3) and was stopped before any result existed; the rule was fixed to "fewer parameters, then higher mean" -> G4 (README amendment 8). S3 block 1 needed a resume (start-deadline cut off seeds 70-71).
* C2 seeds 2000-2002 were copied from stage C into stage D (identical runs; `audit_phase1_results.py` reports 3 'copy' groups; MLflow counts them twice). TASK-023's S0 / S1 (replication, death diagnostic) remain not run.
* The machine clock is TST (JST - 1 h); the operator's deadlines were converted accordingly after an earlier overrun (memory note `deadline-timezone`).
* Not done: finals for 0.14M and for R2 / R4 (not qualifying by the pre-registered rules); layer-count variation; a death-cause diagnostic of the Truck agents.
