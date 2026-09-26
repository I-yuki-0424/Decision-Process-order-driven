# Passive ("do-nothing") world model v2 -- decoupled pre-training (TASK-20260927-008)

Operator hypothesis: if the model can predict what happens when nothing is done, choosing actions gets easy; train this model first,
independently of the policy. Data: `output/experiments/2026-09-27_passive_wm_v2/` (summaries `stageB_summary.txt`, `pf_summary.txt`).
Code: `src/model/candidates/passive_wm_full.py`, `src/environment/craftax_future_adapter.py`, `scripts/run_passive_wm_v2.py`,
`scripts/run_passive_mobmap.py`, `scripts/run_passive_policy_v2.py`, `scripts/run_passive_first_planning.py`,
`scripts/run_pwm_job.py`, `scripts/kaggle_push_pwm.py`, `scripts/aggregate_pwm.py`, `scripts/aggregate_pfirst.py`.
Compute: local RTX 3060 Ti (up to 3 containers, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, no OOM) + Kaggle kernels `dpod-pwm-oracle`, `dpod-pwm-wm`
(separate notebooks; a 2-session GPU cap was hit, so a third parallel kernel was not possible).

## Objections to the premise (recorded before running)
1. The TASK-004 passive model predicted 4 vitals, whose noop future is a near-function of the current vitals the policy already sees. Its null
   result says nothing about passive models in general. -> Stage A2 re-tests with the full observation.
2. A world model is fit by prediction. "RL" can only mean the data-collection policy (curiosity/coverage). Not implemented as RL; the coverage
   variable was tested directly (experiment 3: which states the passive data covers).
3. Chess analogy: a noop future is not the opponent's move; it omits the agent's own effect. Passive prediction alone yields no plan;
   a planner still needs the action effect. Tested in experiment 3 as passive-pretrained + learned action effect.
4. Noop branches are simulator-branched labels (offline only). On real robots they exist only where the behaviour policy idled; otherwise the
   passive model has to be an action-conditional model queried at u=0.

## 1. What is predictable under noop in Craftax-Classic (Stage A, 384 random-policy episodes x 250 steps, held-out episodes)
Skill vs persistence = 1 - MSE/MSE(no change), horizons 1/2/4/8 (2 replicates: local + Kaggle, same numbers within 0.03).

| group | h=1 | h=2 | h=4 | h=8 |
|---|---|---|---|---|
| vitals (MLP on summary) | 0.02 | 0.10 | 0.30 | 0.55 |
| light/sleep | 0.89 | 0.66 | 0.62 | 0.78 |
| mob counts | -0.21 | -0.22 | -0.15 | -0.06 |
| tile counts, inventory | exactly unchanged under noop (nothing to predict) | | | |

The full 1345-d input is not better than the 39-d summary (overfits). Mob motion, spatial (Stage A2, 7x9x4 mob map, frames with a mob in view):
skill vs persistence 0.07/0.12/0.16/0.16; only 8.5% (h=1) / 1.3% (h=8) of newly-occupied mob cells are recalled (precision 36% / 14%).
=> The predictable passive future is slow and mostly clock-like (vitals, daylight); the "opponent" (mobs) is close to unpredictable from a frame.

## 2. Does it help the action policy on Craftax? (Stage B; transformer_branch d=256, 277K params, 4,800 episodes/run, REINFORCE)
Mean over seeds of the last three evaluations (32 episodes each); uniform random = 1.27 (same evaluator).

| arm | seeds | last-3 eval return | final 64-ep return |
|---|---|---|---|
| base (no extra features) | 4 | 3.02 +- 0.08 | 3.03 |
| oracle: TRUE simulated noop future (upper bound) | 4 | 2.94 +- 0.14 | 3.09 |
| wm_full (frozen trained WM, local) | 4 | 3.05 +- 0.12 | 3.02 |
| wm_untrain (random-init control, local) | 4 | 3.01 +- 0.10 | 3.12 |
| kwm_full / kwm_untrain (Kaggle replicate, own WM) | 4 / 4 | 2.98 +- 0.21 / 3.00 +- 0.20 | 3.21 / 3.19 |
| oracle_act: POSITIVE CONTROL, true 1-step reward of all 17 actions (pure leakage) | 2 | 2.90 +- 0.15 | 2.98 |

All arms are indistinguishable (seed sd 0.1-0.2). Even perfect knowledge of the noop future does not help, and **the positive control
(telling the policy each action's reward) does not help either**. Therefore this benchmark/learner combination cannot detect a
world-model benefit at all (a 300-update REINFORCE learner saturates near 3.0 regardless of extra information). The conclusion is
"no evidence", not "evidence of no effect" for anything richer than this learner.
Caveats: the oracle noise realisation is one sample; 2 seeds for the positive control; one learner/size/budget.

## 3. Decoupled passive pre-training where the test has power (pendulum swing-up, CEM-MPC; true-model ceiling -196)
Same task as `scripts/run_wm_planning.py`. Passive data = 256 trajectories with u=0; active data = 8 or 32 random-torque trajectories; 3 seeds.
Planning return (higher is better; median per cell for Na=8 / Na=32; details in `pf_summary.txt`):

| passive-data ICs | mlp (active only) | mlp pooled | factored f+gu, joint | factored, passive-pretrained f frozen | hn (active only) | hn, passive-pretrained V,R frozen, only G trained |
|---|---|---|---|---|---|---|
| narrow (same region as active) | -598 / -589 | -597 / -585 | -531 / -570 | -453 / -570 | -272 / -264 | -262 / -261 |
| wide (whole circle) | -620 / -605 | -203 / -201 | -198 / -198 | -203 / -199 | -266 / -264 | **-195 / -195** |

- The gain comes from **passive data covering states the active data never visits** (upright region), not from the decoupled training schedule:
  pooling passive+active in one plain MLP (-203) matches the factored pre-trained model (-203) and the jointly trained factored model (-198).
- With narrow passive data (no new coverage) there is no gain for MLP/factored models: 256 passive trajectories are worthless if they do not reach the region the plan needs.
- The port-Hamiltonian prior (known kinetic energy) is the only thing that helps with narrow data (-262 vs -590); pre-training its potential on wide passive data
  reaches the ceiling with only G (2 numbers) learned from 8 active trajectories.
- Structure caveat: this plant IS control-affine with constant G, so the passive/active split is the correct inductive bias; no transfer claim to
  Craftax (discrete actions, non-affine interactions). "Wide" passive data requires releasing the system from any state, unavailable for most real robots.

## Verdict on the operator hypothesis
- "Predicting the do-nothing future makes action choice easy" holds only in a narrow form: (a) control-affine/near-physical system, (b) passive data covers
  the states the plan will visit, (c) the passive model has a correct structure prior. Where these hold (pendulum), a pre-trained passive Hamiltonian plus a tiny
  action-effect head reaches the planning ceiling from 8 action-labelled trajectories.
- On Craftax the passive future carries little decision-relevant information (vitals/daylight clocks; mobs unpredictable) and this learner cannot even exploit leaked answers,
  so the idea is unproven there. A stronger learner first: the positive control must move before world-model arms are informative.
- "Reinforcement-learn the world model" was not pursued as RL (a WM is trained by prediction). The controllable lever is data coverage; a curiosity/coverage-driven
  collector is an untested follow-up that needs a task where the positive control is detectable.

## Blockers / limits
- Stage B: 4 seeds (2 for oracle_act; its seed 2 was stopped to free the GPU), one architecture and budget.
- Stage A trained on random-policy states only; the shift to policy-visited states was not measured.
- The Kaggle WM kernel retrains its own WM (independent replicate), so `kwm_*` are replicates, not the same weights.
