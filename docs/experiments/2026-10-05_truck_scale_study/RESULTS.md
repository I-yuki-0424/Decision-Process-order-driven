# TASK-20261005-024 results: Truck seed variance and parameter scale (executed 2026-10-05 .. 2026-10-08)

Local RTX 3060 Ti in Docker (jax 0.10.2, flax 0.12.8, craftax 1.6.1; `output/phase1/recipe_t023/environment.txt`). Pre-registration and amendments (each written before the jobs it covers): `README.md` in this directory.
Files: `output/phase1/truck_scale_t024/{AB,C,D,E,FS,baseline,final}/`; reports: `python scripts/run_truck_scale_sweep.py report --stage <S>`, `python scripts/run_baseline_warmup_check.py report`.

## 1. Answers to the operator's questions
* **Why was Truck's seed variance large?** Mostly the optimisation recipe, not model size. TASK-023's recipe point used lr 1.5e-3, entropy 0.003, no warm-up, gamma 0.99; its first PPO updates clipped 28-41 % of ratios, and
  training curves separated in learning speed within the first 30 % of the steps. On tuning seeds 2000-2002 the old Truck gave 35.97 reward (sd 3.6); adding **5 % lr warm-up + entropy 0.01** gave 42.0-44.4 (sd 2.8-3.0); then **lr 1e-3** (46.5) and
  **gamma 0.97** (48.2 reward, sd 1.3 over 4 seeds). lr 7e-4 / 3e-4, 2 epochs and bigger batches (128/256 envs, fewer updates) were worse. Entropy 0.02, warm-up 15 %, gamma 0.95, lambda 0.55, 4 epochs: no gain.
* **Does a bigger or smaller hidden dimension help?** No, once the recipe is stable. At the old recipe, d >= 128 was clearly worse (25-31 reward; d256 collapsed to 12 as the entropy fell too fast): an lr/size interaction, not a capacity effect.
  At the stable recipe (tuning seeds, 4 each): 0.14M 40.8 reward; 0.28M 48.2; 0.47M 48.9; 0.93M 47.4; 1.66M 46.5, i.e. a plateau from 0.28M, a drop below it, and **seed sd growing with size** (1.3, 3.4, 6.0, 7.7).
  Final seeds confirm the plateau (below). The small model therefore keeps its advantage: nothing is gained by scaling the hidden dimension (layers and sequence length untouched; batch size was tested: larger batches did not help).

## 2. Final results (EP-A; train seeds 62-71 / test 444-453, never used before; clean tree; 256 fresh envs per seed; ~1M steps)
| model | params_total | reward_pct | score_pct | per-seed reward |
|---|---|---|---|---|
| Truck-improved **E1** (d64, GRU w128, +mem+pa+hs) | 0.284M | **48.54 +- 1.50** (sd 4.8) | **17.78 +- 1.25** | 53.0 46.3 50.4 53.2 41.2 46.1 47.4 48.9 56.1 42.6 |
| Truck-improved S2 (GRU w256) | 0.467M | 46.17 +- 0.95 (sd 3.0) | 16.87 +- 0.90 | 46.4 45.6 46.4 44.9 50.4 44.3 49.1 47.4 39.4 47.9 |
| Truck-improved S3 (d128, w256) | 0.928M | 47.15 +- 1.19 (sd 3.8) | 17.46 +- 1.12 | 52.8 42.8 49.5 41.9 51.8 49.3 43.8 44.7 46.9 48.0 |
| baseline of record G4 (gru256+ln+skip, 4 epochs, same recipe point) | 1.013M | 45.68 +- 0.37 (sd 1.2) | 15.42 +- 0.33 | 46.1 44.5 45.2 46.8 44.0 44.1 46.8 45.6 46.7 47.0 |
| (TASK-023 Truck-improved, old recipe) | 0.467M | 39.29 +- 2.17 | 11.08 +- 1.25 | |
| (TASK-023 baseline gru512, old recipe) | 2.80M | 41.56 +- 0.28 | 13.59 +- 0.17 | |

Differences to the baseline (Welch, 10 vs 10): E1 +2.86 +- 1.55 reward (z 1.8), +2.35 +- 1.30 score (z 1.8); S2 +0.48 +- 1.02 / +1.45 +- 0.96; S3 +1.47 +- 1.25 / +2.04 +- 1.17.
All three Truck sizes are above the baseline on both metrics (pooled 30 Truck runs: 47.29 reward, 17.37 score vs 45.68, 15.42), none individually beyond ~2 SE; the three sizes are indistinguishable from each other.
Truck's seed spread (sd 3.0-4.8) is still 2.5-4x the baseline's (1.2): the best Truck seeds (53-56) are above every baseline seed, the worst (39-43) are below.

## 3. Gates (numbers from this file and MASTER_GUIDANCE; protocol EP-A)
* **G1.0 PASS**: baseline G4 reward 45.68 >= 42.7, score 15.42 >= 9.6 (after giving the baseline the same recipe knobs: 10 configs x 3 tuning seeds, section 4).
* **G1.1 PASS as written, thin margin**: E1 reward 48.54 > 47.40 and score 17.78 > 10.71, params_total 0.284M <= 4.0M. The reward margin (+1.14) is 0.76 SE of E1's mean; the S2/S3 sizes (46.17, 47.15) are below 47.40.
  Read it as "point estimate above the published reference", not as a significant win; the reference has its own SE (0.58).
* **G1.2 FAIL**: reward 48.54 < 55.49 (score 17.78 > 16.77, but both are required). Phase 1 stays open.

## 4. Fairness / caveats (all recorded, none hidden)
* Equal effort (P9) is only roughly met: Truck had ~150 screening jobs (stages AB, C, D, E, FS; seeds 2000-2005) after TASK-023; the baseline had TASK-023's search plus 10 configs x 3 seeds at Truck's recipe point (B1-B6, G1-G4).
  The baseline itself gained 4 reward from Truck's recipe point (gru512 TASK-023 final 41.56 -> G4 45.68), so the earlier "Truck is not better than the baseline" statement (TASK-020/023) was partly a recipe artefact for both arms.
* Tuning seeds 2000-2005 were reused across many comparisons (selection bias; tuning numbers are optimistic: S2 48.89 on tuning seeds vs 46.17 on finals; E1 48.15 vs 48.54). The finals use fresh seeds and one config per row.
* Stage AB ran only 45 of 96 planned jobs (its deadline passed during an idle session; 2 concurrent jobs ran 2.5x slower each, later stages ran serially). Scale claims at the old recipe rest on 2-4 seeds.
* The automatic baseline selector broke a tie by dictionary order (B3) and was stopped before any result existed; the rule was fixed to "fewer parameters, then higher mean" -> G4 (README amendment 8).
* S3's final needed a resume (its first run's start-deadline cut off seeds 70-71); same commit family and config, clean tree. S1 (0.14M) has no final (tuning seeds only: 40.8 reward).
* C2 seeds 2000-2002 were copied from stage C into stage D (identical runs; `audit_phase1_results.py` reports 3 'copy' groups; MLflow counts them twice). TASK-023's S0/S1 (replication, death diagnostic) remain not run.
* Architecture attribution is NOT yet established: at the stable recipe nobody has re-tested whether memory token / previous action / head skip / candidate tokens matter (TASK-023 tested them only at the old, noisy recipe).
