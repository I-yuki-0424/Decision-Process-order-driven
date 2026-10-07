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

## Stage F/S result and amendment 5: finals plan (2026-10-07 05:15; written before any final job)
Stage F/S (seeds 2000-2003; E1 = d64 w128, 0.284M, lr 1e-3, gamma 0.97, lambda 0.625, warm-up 5 %, entropy 0.01, 3 epochs x 8 minibatches, 64 envs x 64 steps):

| config | params | reward_pct (sd) | score_pct (sd) | r+s | per-seed reward |
|---|---|---|---|---|---|
| E1 | 0.284M | 48.15 (1.3) | 17.19 (3.0) | 65.3 | 47 48 47 50 |
| F1 gamma 0.95 | 0.284M | 46.61 (4.1) | 15.84 | 62.5 | 44 44 53 46 |
| F2 lambda 0.55 | 0.284M | 47.23 (5.6) | 17.28 | 64.5 | 45 49 54 41 |
| S1 d32 w128 | 0.143M | 40.82 (2.4) | 12.66 | 53.5 | 38 41 43 41 |
| S2 d64 w256 | 0.467M | 48.89 (3.4) | 18.13 | 67.0 | 45 48 48 54 |
| S3 d128 w256 | 0.928M | 47.39 (6.0) | 17.42 | 64.8 | 54 40 50 45 |
| S4 d128 w512 | 1.655M | 46.48 (7.7) | 17.60 | 64.1 | 55 42 38 50 |

Reading: mean performance plateaus between 0.28M and 1.7M parameters and drops at 0.14M; the seed-to-seed sd grows with size (1.3 -> 3.4 -> 6.0 -> 7.7 reward points from 0.28M to 1.66M),
so size does not buy accuracy but does cost stability once the recipe is fixed. Selection per the pre-registered rule: E1 (0.28M) is within 3 r+s points of the best (S2, 0.47M) -> E1 is the Truck of record.
Finals (EP-A, seeds 62-71 / test 444-453, clean tree, 10 seeds each), in this order, each only started if it can end before 12:40 on 2026-10-07: (1) E1; (2) the equal-effort baseline check
(gru512 at the TASK-023 point + 5 % warm-up, 3 tuning seeds; a baseline final on the same seeds 62-71 follows only if warm-up helps it by >= 2 r+s points, else the TASK-023 baseline finals
(41.56 / 13.59, seeds 52-61) stay the reference); (3) S2 0.47M; (4) S3 0.93M; (5) S1 0.14M -- these three give the final-seed scale curve the operator asked for.

## E1 final result and amendment 6: equal-effort baseline (2026-10-07 06:50; written before any baseline job of this study)
E1 final (EP-A, seeds 62-71 / test 444-453, clean tree, commit 764b4d17): reward_pct **48.54 +- 1.50** (sd 4.8), score_pct **17.78 +- 1.25**, params_total 0.2836M; per-seed reward
53.0 46.3 50.4 53.2 41.2 46.1 47.4 48.9 56.1 42.6. Point estimates exceed the G1.1 thresholds (47.40 / 10.71, params <= 4.0M) and the G1.2 *score* (16.77), not the G1.2 reward (55.49).
Why this is not yet a gate statement: (i) G1.0 (baseline >= 42.7 reward) failed with the TASK-023 baseline (41.56) so Phase-1 comparisons are formally invalid; (ii) P9 equal effort: Truck got far more
tuning (~110 tuning jobs since TASK-023, 4 knob families) than the baseline (TASK-023: 4+ jobs/point, no warm-up/lr/gamma search at the Truck point). Therefore, BEFORE the remaining scale finals, the baseline gets the
same knobs: `B1` TASK-023 baseline (gru512, ref point) + 5 % warm-up; `B2` gru512 at Truck's E1 recipe point; `B3` gru256+ln+skip at E1 point; `B4` gru512+ln+skip (3.3M) at E1 point; `B5` B2 with lr 5e-4;
`B6` B1 with lr 1e-3; 3 tuning seeds each (2000-2002); best by mean r+s -> baseline final on seeds 62-71 (same seeds as Truck). Order afterwards: S2 final, S3 final, S1 final while time remains
(hard stop 2026-10-07 12:40 for starting jobs).

## Baseline round 1 and amendment 7 (2026-10-07 08:15; written before round 2)
Round 1 (seeds 2000-2002; mean reward / score): B1 gru512 ref+warm-up 40.95 / 13.20; B2 gru512 at Truck's E1 point 42.50 / 13.57; **B3 gru256+ln+skip (1.01M) at the E1 point 44.68 / 14.52**
(r+s 59.2); B4 gru512+ln+skip (3.33M) 43.89 / 15.02 (58.9); B5 (B2, lr 5e-4) 41.48 / 12.48; B6 (B1, lr 1e-3) 39.73 / 12.32. Truck E1 on the same first three seeds: 47, 48, 47 reward (r+s ~65).
The baseline profits from the Truck recipe point too (+4 reward), so the Truck-vs-baseline claim must use the re-tuned baseline. Round 2 around B3: `G1` lr 1.5e-3, `G2` gamma 0.95,
`G3` entropy 0.003, `G4` 4 epochs (3 seeds each). Baseline final = argmax mean(r+s) over B1-B6, G1-G4 on tuning seeds (within 3 points -> fewer parameters), 10 seeds 62-71 / test 444-453.
Order: round 2 -> baseline final -> S2 final -> S3 final (jobs start only if they can end before 12:40; an unfinished final is reported as incomplete, never as a result).

## Baseline round 2 and amendment 8 (2026-10-07 09:05)
Round 2 (seeds 2000-2002, B3 base = gru256+ln+skip at the Truck E1 point; mean reward / score / r+s): G1 lr 1.5e-3 43.84 / 15.10 / 58.95; G2 gamma 0.95 43.06 / 14.58 / 57.63; G3 entropy 0.003 43.18 / 12.52 / 55.71;
**G4 4 epochs 45.99 / 15.00 / 61.00**; (B3 44.68 / 14.52 / 59.19). Four configs (B3, G1, G4; B4 3.3M excluded by size) are within 3 points of the best; all three small ones have the same parameter count.
The automatic selector broke that tie by dictionary order and started a B3 final; that run was stopped after its first job (no result file existed, nothing was read) because the tie-break rule was underspecified
and arbitrary. Rule fixed to "fewer parameters, then higher mean r+s" -> **G4** is the baseline of record, final on seeds 62-71 / test 444-453. The selection function was corrected (this commit).

## Finals done; amendment 9: architecture attribution at the stable recipe (2026-10-07 18:30 machine time = 19:30 JST; written before any stage-X job)
Final results are in `RESULTS.md` (E1 48.54 / 17.78, S2 46.17 / 16.87, S3 47.15 / 17.46, baseline G4 45.68 / 15.42; G1.0 pass, G1.1 pass on point estimates, G1.2 fail).
Completion deadline given by the operator: 2026-10-08 11:30 JST = 10:30 machine time; no job may start after 07:00 machine time on 10-08, the written report must be in the repo by 09:30.
Open question: with the stable recipe, which of Truck's structural parts matter? Stage X (seeds 2000-2003, same as E1's tuning seeds 47 / 48 / 47 / 50 reward) removes one part from E1 at a time:
`X1` no candidate-action tokens (logits from the head only), `X2` no memory token, `X3` no previous-action input, `X4` heads read h only, `X5` all three options off (the original Truck structure, d64 w128).
Decision rules (fixed now): a part "matters" if removing it lowers mean reward+score by >= 6 points (about 2 SE of a 4-seed mean) AND the sign is the same for reward and score; a part whose removal changes the mean by < 3 points is
"not shown to matter". Only the finals decide anything on the gates; a new final (seeds 72-81 / test 454-463 are reserved) is run only for a configuration that is not clearly worse than E1 AND has fewer parameters/parts, to test
whether a simpler Truck is as good, or for a variant that beats E1 by >= 6 r+s on the 4 seeds.

## Stage X result and amendment 10: replication block and attribution finals (2026-10-07 21:10 machine time = 22:10 JST; written before any of these jobs)
Stage X (seeds 2000-2003; E1 on the same seeds: 48.15 reward / 17.19 score, r+s 65.3): X1 no candidate tokens 46.05 / 16.62 (r+s 62.7); X2 no memory token 46.21 / 15.01 (61.2); X3 no previous action 45.76 / 14.90 (60.7);
X4 no head skip 46.27 / 15.32 (61.6); X5 original structure (no mem/pa/hs) 46.09 / 15.78 (61.9). Every removal costs 2.7-4.6 r+s points, none reaches the pre-registered 6-point rule -> no single part is shown to matter
(4 seeds cannot see effects below ~5 points); in particular the candidate-action tokens (the 4th-idea readout) are not shown to contribute.
Next, in this order (each a 10-seed EP-A final, clean tree, seeds 72-81 / test 454-463, the block reserved in amendment 9):
(1) E1 and (2) baseline G4 on the new block (a replication of the thin G1.1 margin and an equal-seed comparison; pooled with seeds 62-71 -> 20 seeds each), then (3) X1 and (4) X5 (simplest Truck variants; 'does the structure matter at 10 seeds').
Start of any job after 2026-10-08 07:00 machine time (08:00 JST) is forbidden; the report is finalised by 09:30 machine time (10:30 JST).
