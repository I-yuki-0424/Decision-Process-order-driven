# TASK-20261125-041 results: Dyna-style PPO toward G1.2

Pre-registration and amendments 1-4: README.md (same directory). All runs: PC RTX 3060 Ti (WSL Ubuntu-22.04, jax 0.12.1, craftax 1.7.0),
Craftax-Classic symbolic 1345-d, default episode length and reward, 999,424 env steps, 256 fresh eval envs per seed, sampled policy,
final params, tuning seeds 7100-7103 (never used before). Per-seed files: results/<stage>/. No gate id: tuning and diagnostics only.

## Verdict

* **G1.2 is not reached and no final EP-A block was run.** The qualification line (tuning mean >= 56.5) was missed by every candidate;
  all three learned-model variants are 5-7 reward below the GRU-TP control.
* **The cause found in TASK-026/028 is now confirmed by intervention.** Giving the same GRU-TP learner policy-fresh data at the same
  1M real steps lifts it from 48.8 to 56.9 (1:1 imagined:real) and 60.5 (2:1). The 2:1 level is at the G1.2 bar with headroom, but
  only with the true simulator (oracle control: simulator branching, P3, never a gate).
* **What blocks the admissible version is the reward model.** With true next observations but learned rewards the score falls
  below the control (41.6 vs 48.3); with learned observations but true rewards it is above it (49.8). Predicted reward on true
  achievement steps stays at 0.30-0.38 of the real value in every variant.

## All arms (reward_pct; paired difference against the control C0 on the same seeds)

| stage | arm | protocol | seeds: reward_pct | reward mean +- SE | score mean +- SE | paired vs C0 | params_total | commit |
|---|---|---|---|---|---|---|---|---|
| D | L1 | diag-dyna-model | 7190: 38.60 | 38.60 +- nan | 13.07 +- nan | - | 2,490,963 | 95d2708 |
| F | F_model_model | oracle-diag-factorial | 7100: 42.92 / 7101: 43.55 | 43.24 +- 0.32 | 12.60 +- 0.05 | -5.10 +- 0.72 | 2,498,930 | 5aab350b autotune0 |
| F | F_model_sim | oracle-diag-factorial | 7100: 49.68 / 7101: 49.98 | 49.83 +- 0.15 | 18.24 +- 0.24 | +1.49 +- 0.55 | 2,498,930 | 5aab350b autotune0 |
| F | F_sim_model | oracle-diag-factorial | 7100: 40.87 / 7101: 42.28 | 41.57 +- 0.70 | 12.77 +- 1.01 | -6.76 +- 1.10 | 2,498,930 | 5aab350b autotune0 |
| F | F_sim_sim | oracle-diag-factorial | 7100: 54.92 / 7101: 56.14 | 55.53 +- 0.61 | 22.86 +- 0.93 | +7.19 +- 1.01 | 2,498,930 | 5aab350b autotune0 |
| S1 | C0 | EP-A-tuning | 7100: 48.74 / 7101: 47.94 / 7102: 50.91 / 7103: 47.44 | 48.76 +- 0.76 | 19.65 +- 0.84 | - | 1,254,358 | b29e33a1 |
| S1 | L1 | EP-A-tuning | 7100: 32.97 / 7101: 38.85 / 7102: 42.44 / 7103: 36.97 | 37.81 +- 1.97 | 10.50 +- 1.20 | -10.95 +- 1.66 | 2,490,963 | b29e33a1 |
| S1 | O1 | oracle-upper-bound | 7100: 55.79 / 7101: 55.54 / 7102: 56.62 / 7103: 59.66 | 56.90 +- 0.95 | 23.99 +- 1.40 | +8.15 +- 1.41 | 1,254,358 | b29e33a1 |
| S2 | L2 | EP-A-tuning | 7100: 43.20 / 7101: 43.45 / 7102: 44.19 / 7103: 43.98 | 43.71 +- 0.23 | 13.62 +- 0.45 | -5.05 +- 0.70 | 2,498,930 | 7798db82 autotune0 |
| S2 | O2 | oracle-upper-bound | 7100: 63.76 / 7101: 57.32 / 7102: 59.22 / 7103: 61.83 | 60.53 +- 1.42 | 28.35 +- 1.11 | +11.77 +- 1.71 | 1,254,358 | 7798db82 |
| S3 | L3 | EP-A-tuning | 7100: 42.58 / 7101: 47.48 / 7102: 40.20 / 7103: 44.46 | 43.68 +- 1.54 | 13.90 +- 0.75 | -5.08 +- 2.21 | 2,498,930 | 7798db82 autotune0 |
| S4 | L4 | EP-A-tuning | 7100: 43.45 / 7101: 40.45 / 7102: 40.36 / 7103: 42.93 | 41.80 +- 0.81 | 13.17 +- 0.57 | -6.96 +- 1.35 | 2,515,378 | ad0744c8 autotune0 |

Arms: C0 GRU-TP (Base_O02_D04_S00, TASK-025 recipe). L1 learned Dyna, model v1. O1 oracle Dyna 1:1. L2 model v2 (cell-summed map loss,
two-hot reward), 16 model steps/update. O2 oracle 2:1. L3 = L2 with horizon 4 x 1024 rows. L4 = L2 with the model conditioned on the
agent's GRU state. F_<obs>_<rew>: horizon-1 factorial, next observation + termination and reward each from the simulator (sim) or
model v2 (model). D: held-out model diagnostic of L1. Oracle and F arms are upper-bound / diagnostic controls (P3), never EP-A.

## Horizon-1 factorial (seeds 7100-7101; same-seed C0 = 48.34)

| next observation / reward | simulator | learned model v2 |
|---|---|---|
| simulator | 55.53 +- 0.61 | 41.57 +- 0.70 |
| learned model v2 | 49.83 +- 0.15 | 43.24 +- 0.32 |

Reward source: -14.0 (sim obs) / -6.6 (model obs). Observation source: -5.7 (sim reward) / +1.7 (model reward, within noise).

## Model quality (held-out: the next real rollout before the model trains on it; end of training)

| | L1 (v1) | L2 (v2) | L3 | L4 (cond) |
|---|---|---|---|---|
| block accuracy, cells that change | 0.03 | 0.87 | 0.87 | 0.87 |
| inventory + intrinsics exactly right | 0.68 | 0.65 | 0.63 | 0.50 |
| reward MSE (predict-0 baseline) | 0.045 (0.044) | 0.029 (0.043) | 0.027 (0.038) | 0.028 (0.039) |
| predicted reward on true achievement steps (true ~1) | 0.25 | 0.35 | 0.34 | 0.30 |
| predicted / real termination rate | 0.0052 / 0.0054 | 0.005 / 0.005 | | |

(L1 values from diagnostic D, seed 7190; L2-L4 from seed 7100 / 7101 / 7100 of their own runs.)

## Reading (evidence level stated)

* Tested: the Dyna update (one-step imagined PPO gradients, shared lr schedule) is not the problem; the oracle with the identical
  update gains +8.2 (1:1) and +11.8 (2:1). The extra optimiser steps are the same in O1 and L1, so they do not explain L1's loss.
* Tested: the v1 map loss averaged over cells made the model copy the view (changed-cell accuracy 0.03); v2 fixes that (0.87).
* Tested: the reward model is the dominant remaining error (factorial). One plausible account, not established: Craftax pays each
  achievement once per episode and the observation does not show which are done, so reward is not a function of (obs, action);
  0.30-0.38 is what an average over first and repeated events would look like. Conditioning on the agent's GRU state (L4) did not
  help, which is consistent with the GRU state not encoding achievement history, or with the model not using it; not separated.
  A competing account: the reward head is under-trained on rare events (about 1 % of steps) and the two-hot target spreads them.
* Not tested: a recurrent / sequence world model trained to predict reward from its own history (Dedieu et al. 2025 use a
  transformer world model over history, MBRL_fast 67.42 / 27.91, MASTER_GUIDANCE).

## Caveats

* 4 tuning seeds per arm (F: 2). L2-L4 and F ran with `XLA_FLAGS=--xla_gpu_autotune_level=0` (compile-time OOM on the shared GPU), C0
  with default autotuning; this changes kernel choice, not the algorithm.
* PC packages differ from earlier Kaggle runs (craftax 1.7.0 vs 1.6.1); C0 was re-run here, and its mean (48.76) matches the earlier
  pooled GRU-TP value (48.46 +- 0.15).
* Kaggle was not usable (kaggle.json returns 401, no broker service, no token on the PC), so everything ran serially on one GPU.
* Failed attempts (OOM) are recorded in results/failures and MLflow (end_reason oom) and are not part of any aggregate.
* params_total of the learned arms includes the model (~1.24M); oracle arms have no model. The Dyna gain, where it exists, is
  extra compute, not parameters.

## Next step (recommended, not started)

A model whose reward head has memory of the episode: e.g. a small recurrent or attention model over the last k (obs, action) pairs
of real trajectories, trained on executed transitions only, with the H=1 factorial cell F_sim_model as the acceptance test (it must
reach the control or better before any full Dyna run). Second: improve inventory/intrinsics prediction (0.65 exact). With both,
re-run the 2:1 ratio (oracle 60.5) as the candidate; the final EP-A block (seeds 192-201 / test 574-583) only if its tuning mean >= 56.5.
