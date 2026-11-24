# TASK-20261125-041: Dyna-style PPO toward G1.2 (pre-registration, written before any result)

Operator request (2026-11-25): re-establish the cause of the Phase-1 stall, list candidate fixes and the current state, verify on the
PC and Kaggle (the VPS lacks resources), and if possible bring Phase 1 to a state where it can close.

## Where Phase 1 stands (quoted from STATE.yaml TASK-023..028; references from MASTER_GUIDANCE)

| gate | requirement (EP-A, mean - 2 SE over 10 seeds) | best P5-clean evidence | status |
|---|---|---|---|
| G1.0 | baseline >= 42.7 / 9.6 | G4 gru256+ln+skip 45.42 +- 0.30 / 15.48 (20 seeds) | PASS |
| G1.1 | > 47.40 / 10.71 (MFRL_4M_reimpl), params_total <= 4.0M | GRU-TP 1.25M, block I 48.72 +- 0.36 / 18.82 (2nd pre-registered block; block A missed by 0.02); pooled 70 seeds 48.46 +- 0.15 | passed on the 2nd attempt, **operator ruling pending** |
| G1.2 | > 55.49 / 16.77 (MFRL_best, 55.6M params, image obs) | GRU-TP+LP block J 49.61 +- 0.55; rs0 stack (P5-contested) block K 53.10 +- 0.47 | FAIL, gap about +7 reward |

"Matching at a quarter of the size": Truck E1 (0.284M) beat the re-tuned GRU baseline G4 (1.01M) in all 8 blocks, +1.77 +- 0.44 reward
(z 4.0). That is a real result, but against our own baseline; Truck alone (pooled 47.12) does not pass G1.1, and nothing reaches G1.2.

## Cause (as established, with its evidence level)

* Tested: the gap is **sample efficiency at 1M steps**, not capacity. Same GRU-TP: 0.5M 35.4, 1M 48.5, 2M 57.7-60.4, 5M 67.5 (TASK-026/028).
  Width x2 (3.6M params) and ten single knobs changed nothing beyond noise (TASK-026/027).
* Tested: the PPO+GRU learner cannot reuse real data more than ~4 passes per sample (8 epochs -7.6; replay K=2/4 worse, KL spikes; TASK-026/028).
* Tested (diagnostic, P5-contested): most of the rs0 gain is defeat_zombie (objective change, not efficiency).
* Not tested: any method that adds *policy-fresh* data without new env steps. Dyna is the standard one, and the reference work at the
  same budget (Dedieu et al. 2025, MBRL_fast 67.42 / 27.91, MASTER_GUIDANCE, caveats none) gets its gain from a world model + Dyna.
* The BOTSYSTEM side track (TASK-029..040) did not touch Phase 1: no Phase-1 run was made after TASK-028 (2026-10-25).

## Critical review of Dyna here

* Leakage: the model is trained only on executed transitions (P4), imagined rollouts start from states the agent visited with the GRU
  state it had there; no simulator call at decision time. Evaluation unchanged (P3 clean).
* Budget: imagined transitions are not env steps (P10) for the learned model. The oracle arm (simulator branched from visited states)
  IS extra simulator use: labelled upper-bound control, never EP-A, never a gate.
* Claim: a Dyna gain comes from **extra compute**, not from parameters or architecture. G1.2 has no compute or parameter condition, so
  it is admissible for the gate, but it must never be reported as "accuracy without scaling" for free; params_total includes the model.
* Upper bound: with a perfect model at a 1:1 imagined:real ratio Dyna is at best "2M fresh steps" (57.7-60.4), and the imagined updates
  use one-step (non-recurrent) gradients, so even the oracle may fall short. The oracle arm measures this before any claim.

## Design (code: `src/pipeline/epa_dyna.py`, flag `--dyna learned|oracle` of `scripts/run_epa_mini.py`, commit b29e33a)

Base = GRU-TP (Base_O02_D04_S00, TASK-025 G4TP1 recipe, unchanged). Per update after the real PPO step: model trained 8 Adam steps
(batch 1024) on the last 8 real rollouts; 256 rows x 16 imagined steps (= 4096 = 1:1 with the real batch) from random visited
states; PPO 2 epochs x 8 minibatches on the imagined batch; imagination starts after 10 % of the updates; one lr schedule over all
optimiser steps. Model: 3 conv layers (64 ch) over the 7x9 map + MLP (256), predicts the next symbolic observation in the env's
format, reward and termination; 1.24M params, params_total 2.49M.

## Stages and decision rules (fixed now)

* **S1** (PC RTX 3060 Ti, tuning seeds 7100-7103, never used before): C0 = GRU-TP control, L1 = learned Dyna, O1 = oracle Dyna.
  * O1 - C0 < +2.0: the Dyna update itself (one-step imagined gradients, 1:1) is too weak; fix the update before any model work.
  * L1 - C0 >= +2.0 (same seeds) qualifies L1 for S2; < +2.0 with O1 qualified: the model is the bottleneck.
* **S2** (only if S1 qualifies something): at most 3 variants of imagination ratio / horizon / model size on the same seeds; selection
  = max mean reward_pct over 7100-7103.
* **Final EP-A block** only if the selected arm's tuning mean >= 56.5: seeds 192-201 / test 574-583, GRU-TP control on the same seeds,
  clean commit, all seeds listed. The gate claim is G1.2 only if mean - 2 SE > 55.49 and > 16.77.
* Equal effort (P9): the control is the already-tuned GRU-TP; Dyna gets at most S1 + S2 (<= 4 configs x 4 seeds), less than GRU-TP had.

## Amendment 1 (after S1, before S2): results, diagnosis, S2 variants

S1 (PC, tuning seeds 7100-7103, commit b29e33a; files in results/S1): C0 48.76 +- 0.76 / 19.65; L1 37.81 +- 1.97 / 10.50
(paired -10.95 +- 1.66); O1 (oracle, upper bound) 56.90 +- 0.95 / 23.99 (paired +8.15 +- 1.41). Rule: O1 qualifies, L1 does not ->
the model is the bottleneck. The Dyna update itself works; the oracle gain is not an artefact of extra optimiser steps (same steps).

Diagnostic D (seed 7190, role diag, commit 95d2708, held-out = the next real rollout before the model trains on it), end of training:
block accuracy 0.94 overall but **0.03 on cells that change** (falls from 0.34 during training: the model learns to copy the view);
**reward MSE 0.045 vs 0.044 for predicting 0** (predicted reward on achievement steps ~0.2 instead of 1); done calibrated (0.0052 vs
0.0054); inventory+intrinsics exact 0.68. Cause found in the loss: block and mob terms were AVERAGED over the 63 cells while the vector
terms were summed. Model v2 (`wm_version=2`): cell terms summed, reward as a two-hot categorical over 31 bins in [-1, 2] (sampled in
imagination). v1 is unchanged (S1 reproducible).

S2 (same seeds 7100-7103, at most 3 candidate variants, selection = max mean reward_pct):
* L2 = L1 with `wm_version=2, wm_steps=16`.
* Further variants (L3, L4) are chosen from L2's dyna_diag and recorded here before they run.
* Oracle diagnostics (never candidates, never gates): O2 = oracle at 2:1 imagination (`im_envs=512`) to measure the ceiling headroom,
  because O1 at 1:1 (56.90) is only ~1.4 above the reference mean.
