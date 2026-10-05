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
