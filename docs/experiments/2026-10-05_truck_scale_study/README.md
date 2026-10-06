# TASK-20261005-024: Truck seed variance and parameter scale (pre-registration, written before any study run)

Operator request (2026-10-05, 20 h): the biggest open question is Truck's seed-to-seed variance; find out whether learning rate / optimiser
or parameter count (hidden dimension) is the cause, try fewer and more parameters, remember the goal is a better *Truck* (not baseline),
batch size is a possible but unlikely cause.

## Facts from TASK-023 that motivate the design (final seeds 52-61, Truck-improved = tfgru64x2w256+mem+pa+hs, 0.467M)
* reward_pct per seed 31.0 28.2 38.5 36.9 45.0 42.6 42.1 43.4 34.3 50.8 (sd 6.9); gru512 baseline 40.1-43.0 (sd 0.9).
* The spread is in *learning speed*, not in evaluation noise: Truck's training-return curves separate already at 20-30 % of the steps
  (e.g. seed 61 reaches return 10.3, seed 53 only 5.4 at 1M) and all curves are still rising at 1M, whereas the baseline's ten curves coincide.
  The best Truck seeds (45-51) beat the baseline's mean; the median seed does not. Reducing the variance is therefore worth more than a higher mean.
* Winning recipe point r01 uses lr 1.5e-3 with 8 sequences (8 envs x 64 steps = 512 steps) per minibatch; the first PPO updates showed clip fractions of 0.28-0.41
  and approximate KL 0.015-0.02 (CPU/GPU smoke, 3 updates). That is the main reason lr and batch are on the list.

## Code added for this study (defaults off or invisible; all 22 tests of tests/test_epa_harness.py pass)
* `PPOConfig.warmup` (linear lr ramp-up over a fraction of optimiser steps), `PPOConfig.remat` (jax.checkpoint of the per-step forward in the
  recurrent loss: same maths, gradients agree to 1e-4, far less VRAM; needed for d >= 192 and for 128/256 envs), CLI `--warmup --remat`.
* Per-update training diagnostics appended to the stats (approx KL, clip fraction, pre-clip gradient norm), stored binned in each result file as
  `train_diag` (entropy, value loss, KL, clip fraction, grad norm; 10 bins). The old stats layout is unchanged.
* `scripts/run_truck_scale_sweep.py` (driver + `report`), results in `output/phase1/truck_scale_t024/`.

## Design
Base **T0** = TASK-023 Truck-improved at its recipe point r01: d 64, 2 layers, 4 heads, GRU width 256, +mem +pa +hs; 64 envs x 64 steps,
3 epochs x 8 minibatches, lr 1.4954e-3 annealed, gamma 0.99, lambda 0.625, entropy 0.003, vf 1.0, grad-norm 0.5, value-norm 0.99, batch-standardised GAE.
Screening seeds: **2000-2005** (6 per config, role `tune`, disjoint from every seed used so far: 42-61, 424-443, 1000-1002; evaluation on 256 fresh envs).
Metric: mean and sd of reward_pct and score_pct over the 6 seeds; selection value = mean(reward_pct + score_pct). The sd is reported next to it
(6 seeds give an sd with ~30 % relative error: a difference of mean r+s below ~5 points is not a decision).

* **Stage A (stability factors, one change at a time vs T0):** `T0` (re-run on the new seeds), `lr7e-4`, `lr3e-4`, `warm5` (5 % warm-up), `ent01` (entropy 0.01),
  `env128` (128 envs: 16 sequences per minibatch, remat), `env256` (32 sequences per minibatch, remat). Interpretation rule: a factor is "a cause of the variance"
  if its sd of reward_pct is below ~60 % of T0's AND the mean is not worse by more than 3 points; if the mean falls but sd falls too the trade-off is reported.
* **Stage B (hidden dimension, T0 recipe):** `d32w128` 0.14M, `d32w256` 0.31M, `d64w128` 0.28M, T0 0.47M, `d64w512` 1.13M, `d96w256` 0.67M, `d128w256` 0.93M,
  `d128w512` 1.66M, `d192w384` 1.93M (6 heads), `d256w512` 3.30M (8 heads; at the 4.0M limit of G1.1). Sequence/token layout unchanged (the user expects the
  scale lever to be the hidden dimensions: model width d, GRU width).
* **Stage C (combination):** pre-registered rule: the A-factor settings that pass the interpretation rule (up to 2, combined if both pass) x the 2 best Stage-B sizes
  by mean r+s (plus the best Stage-B size at T0 settings); 6 seeds (2000-2005 again for paired comparability, or 2006-2011 if more are needed).
* **Stage F (final):** EP-A, seeds 62-71 / test 444-453 (never used), clean tree, 10 seeds, for (i) the best Stage-C/B/A config by the selection value and (ii) the best
  config with params_total <= 0.5M if (i) is larger. The baseline gru512 (seeds 52-61: 41.56 +- 0.28 / 13.59 +- 0.17) is the reference; it is NOT re-tuned in this study.

## Rules
* All 1M env steps counted; tuning seeds only for screening; finals only once per config; P1-P12 apply (no reward shaping, no extra information).
* The Stage-A/B tests answer *which factor drives the variance*; they do not close a gate. Gates only via Stage F under EP-A and G1.0's baseline calibration,
  which stays failed (41.56 < 42.7) until a baseline run says otherwise: any Truck-vs-gate statement carries that caveat.
* Everything that fails or is excluded (OOM etc.) is recorded; nothing is selected on the final seeds.

## Amendment 2026-10-06 13:00 (after stage AB; written before any stage-C job)
Stage AB ran from 2026-10-05 23:16 until its deadline 2026-10-06 08:00 (the wall clock advanced while the session was idle). Jobs ran ~2.5x slower than the TASK-023
jobs (2 concurrent, ~21 min each, CPU-bound containers) so only 45 of the 96 planned jobs finished: T0 3 seeds (2000-2002), warm5/ent01/lr7e-4/lr3e-4/env128/env256/d64w128/
d64w512 3 seeds, d32w128/d32w256 4, d96w256/d128w256/d128w512/d192w384/d256w512 2. Seeds 2003-2005 are missing for most configs. With ~4.8 h left of the 20 h the plan is cut:
* Stage C (3 seeds 2000-2002, paired with T0's): `C1` warm-up 5 % + entropy 0.01 (0.467M); `C2` = C1 with GRU width 128 (0.284M); `C3` = C1 with d 128, lr 7e-4
  (the "more parameters, lr scaled down" test; 0.93M); `C4` = C1 with d 32, width 128 (0.143M, the "fewer parameters" test).
* Stage F: EP-A finals (seeds 62-71 / test 444-453) for ONE config, the one with the best mean r+s among {warm5, ent01, d64w128, C1-C4} on seeds 2000-2002 (ties / differences
  < 3 points: the one with fewer parameters). Started no later than 15:30 so it can end before the 20 h limit (about 17:45). No further screening after that choice.
* Stage-B claims are limited to what 2-4 seeds can show; the table (per-seed values) is reported as is.

## Stage C result and amendment 2 (2026-10-06 20:00; written before stage D)
Jobs now run one at a time (2 concurrent jobs were 2.5x slower each: 9 min per job alone). Time frame: operator granted a fresh 20 h at 2026-10-06 18:12.
Stage C (seeds 2000-2002; T0 on the same seeds: 35.97 / 9.58, per-seed 35 33 40):

| config | params | reward_pct | sd | score_pct | per-seed reward |
|---|---|---|---|---|---|
| C1 warm 5 % + ent 0.01 (d64 w256) | 0.467M | 42.00 +- 1.75 | 3.0 | 13.91 | 42 45 39 |
| C2 C1 + GRU width 128 | 0.284M | 44.43 +- 1.64 | 2.8 | 14.76 | 41 46 46 |
| C3 C1 + d 128, lr 7e-4 | 0.928M | 43.47 +- 1.90 | 3.3 | 13.92 | 45 46 40 |
| C4 C1 + d 32, width 128 | 0.143M | 40.21 +- 0.34 | 0.6 | 13.08 | 41 40 40 |

Reading: the warm-up + higher entropy recipe removes most of the instability and lifts every size; parameter count between 0.14M and 0.93M barely matters once the recipe is right.
Stage D (seeds 2000-2003, 4 seeds, C2 re-run on 2003 for pairing), one change at a time from C2: `D1` warm-up 15 %, `D2` entropy 0.02, `D3` lr 1e-3, `D4` gamma 0.97,
`D5` 2 epochs. Selection value mean(reward_pct + score_pct); the base of the finals = best of {C2, D1-D5} (differences < 3 points: fewer parameters / simpler). Afterwards: the
same two knobs (warm-up, entropy) are applied to the gru512 baseline on 3 tuning seeds for equal effort (P9) before any comparison.

## Stage D result and amendment 3 (2026-10-06 23:00; written before stage E)
Stage D (seeds 2000-2003, 4 seeds, one change from C2 = d64 w128 warm 5 % ent 0.01 lr 1.4954e-3 gamma 0.99 3 epochs):

| config | reward_pct (sd) | score_pct | r+s | per-seed reward |
|---|---|---|---|---|
| C2 | 43.53 +- 1.47 (2.9) | 14.20 | 57.7 | 41 46 46 41 |
| D1 warm-up 15 % | 43.67 +- 1.50 (3.0) | 14.48 | 58.2 | 44 48 42 41 |
| D2 entropy 0.02 | 42.92 +- 2.68 (5.4) | 13.46 | 56.4 | 46 46 45 35 |
| **D3 lr 1e-3** | **46.52 +- 0.99 (2.0)** | 14.83 | **61.4** | 47 48 44 47 |
| D4 gamma 0.97 | 44.78 +- 2.23 (4.5) | 15.58 | 60.4 | 45 47 49 39 |
| D5 2 epochs | 37.82 +- 0.54 (1.1) | 10.94 | 48.8 | 38 37 39 37 |

Stage E (seeds 2000-2003; base D3): `E1` D3 + gamma 0.97, `E2` D3 + 4 epochs, `E3` D3 + gamma 0.97 + 4 epochs, `E4` D3 with lr 7e-4. Then the parameter-scale sweep is repeated at the best
of {D3, E1-E4} (stage S: d32w128, d64w256, d128w256, d128w512 at that recipe, 4 seeds), since stage AB's scale results were obtained at a recipe that is unstable for large models.

## Stage E result and amendment 4 (2026-10-07 01:35; written before stage F/S)
Stage E (seeds 2000-2003; base D3 = d64 w128, warm 5 %, ent 0.01, lr 1e-3, gamma 0.99, lambda 0.625, 3 epochs):

| config | reward_pct (sd) | score_pct (sd) | r+s | per-seed reward |
|---|---|---|---|---|
| **E1 gamma 0.97** | **48.15 +- 0.62 (1.3)** | **17.19 (3.0)** | **65.3** | 47 48 47 50 |
| E2 4 epochs | 44.88 +- 1.16 (2.3) | 14.56 (1.6) | 59.4 | 42 47 46 44 |
| E3 gamma 0.97 + 4 epochs | 46.76 +- 1.67 (3.3) | 16.13 (2.1) | 62.9 | 46 48 50 43 |
| E4 lr 7e-4 | 39.96 +- 2.45 (4.9) | 12.37 (1.6) | 52.3 | 40 41 45 33 |

E1 is the new base. These are tuning-seed numbers (4 seeds each, many comparisons on the same seeds: optimistic); only stage F-final on fresh seeds counts.
Stage F/S (seeds 2000-2003): `F1` E1 with gamma 0.95, `F2` E1 with lambda 0.55; scale at the E1 recipe: `S1` d32 w128 (0.14M), `S2` d64 w256 (0.47M), `S3` d128 w256 (0.93M), `S4` d128 w512 (1.66M);
E1 itself (d64 w128, 0.28M) is the 5th size. Then: the baseline gru512 + warm-up check (3 tuning seeds, equal effort) and EP-A finals (seeds 62-71 / test 444-453) for the best Truck config
by mean r+s over {E1, F1, F2, S1-S4}; if a different size is within 3 points of the best, the smaller is preferred. A second final is run for the best config at another size only if time allows.
