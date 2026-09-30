# arc.tex design proposals: training results (2026-09-29/30, TASK-20260929-016)

Operator request: use local RTX 3060 Ti + Kaggle T4 to train the three proposals catalogued in `docs/Latex/arc.tex` and measure how useful each is,
including derivative models and parameter adjustments based on medium-scale results. Section mapping used (PDF page numbers): "§4 structure-as-input" =
`phys_feat` / `skip_feat0` (world-model series, p.18-21); "§2.5/2.4/2.8 actor-critic + oracle_act litmus" = Idea-6 policy-feature study, the lr sweep and the
TASK-011 handoff (p.10-14); "§3 ChunkPPO delta>0" = latency/goal-chunking (p.15-17).

All numbers below come from real env steps / real forward passes (raw result JSONs under `output/arc_proposals/`, run logs next to them). No number was typed in by hand
except the two REINFORCE reference values quoted from arc.tex §2.5.

## Verdicts (one line each)

| proposal | verdict | key evidence |
|---|---|---|
| 1. Structure-as-input world model | **Partly holds, but not for the stated reason.** Feeding a formula as an *input feature* is safe and, on the pendulum, decisive (planning return -204 vs -271 without it; ceiling -196). But the gain comes from an *unbounded, monotone-growing feature* (|q|, q^3, q^4 do as well); the physics content is not needed. Bounded/oscillatory or zero-information features give nothing. Hard-coding a wrong formula is much worse (-450..-540). | §1 below |
| 2. Actor-critic + oracle_act litmus | **Litmus passes; the leaked signal is not deployable.** With PPO+GAE+critic (lr 1e-3) `oracle_act` beats `base` by +1.0 (12.8k eps) and +1.8 (50k eps), non-overlapping seeds; REINFORCE showed +0.07. A noop-future world model adds nothing beyond random features (even the true noop future is worth only ~+0.2 at 12.8k eps and nothing at 49.9k). A *learned* action-reward model (no simulator at policy time) recovers ~25-30% of the leaked gain at 12.8k eps and ~7-14% (n.s.) at 50k eps. | §2 below |
| 3. ChunkPPO delta>0 | **Not supported.** No reproducible benefit of the forecast arms (`wm`, `wm_passive`) over `augment`/`ignore`: the single significant gap (intercept delta=2, +0.66, p=0.003, 12 seeds, config A) vanishes under a second learner config (augment 0.67 vs wm 0.60), a different delta shows a gap there where `ignore` equals `wm`, everything closes at 10M ticks, and on `pendulum_goal` forecast arms are worse (-7..-16, n.s.). | §3 below |

## 1. Structure-as-input world model (arc.tex §4)

Code: `scripts/run_structure_input_planning.py`, `scripts/run_structure_input_prediction.py`, `scripts/analyze_structure_route_selection.py`,
aggregation `scripts/aggregate_arc_structure_input.py`. Results: `output/arc_proposals/structure_input/{planning,planning2,prediction,prediction2}/`,
summaries `summary_planning.md`, `summary_prediction.md`, `route_selection.md`.

**Critical review recorded before running.** The earlier result (arc.tex §4.3: wrong small-angle preset as *feature* -> near-ceiling swing-up; as *hard term* -> -480..-530) cannot
distinguish "the feature carries useful physics" from "any input feature that extrapolates helps a tanh-MLP potential". Every arm below uses the same network, data, optimiser and steps;
only the scalar feature handed to the learned potential changes. `feat:true` (the true potential) is an oracle upper bound by construction and is not evidence for the method.

**Pendulum swing-up MPC, true-model ceiling -196.1, 6-12 seeds (mean ± sd), N = training trajectories** (`summary_planning.md`):

| arm (feature given to the learned potential) | N=8 | N=32 | N=128 |
|---|---|---|---|
| none (`hn`) | -270.5 ± 10 | -251.4 ± 27 | -252.0 ± 18 |
| zero feature (same architecture, no information) | -271.3 ± 12 | -251.5 ± 29 | -252.0 ± 19 |
| **small-angle preset q²/2 (the proposal)** | **-204.3 ± 1.0** | -206.0 ± 3.4 | -209.4 ± 2.8 |
| true potential -cos q (oracle) | -200.8 ± 1.6 | -200.5 ± 0.8 | -199.4 ± 0.8 |
| scaled 1.5 q² (right form, wrong constant) | -205.0 ± 4.4 | -203.7 ± 1.9 | -207.1 ± 2.5 |
| **|q| (generic, no physics)** | -206.2 ± 4.8 | -201.5 ± 2.3 | -203.7 ± 5.8 |
| **q³/3 (generic, no physics)** | -209.0 ± 5.6 | -202.4 ± 1.4 | -198.7 ± 1.1 |
| q⁴/4 (generic; bimodal at N=8) | -223 ± 43 (median -200) | -201.6 ± 5.9 | -197.8 ± 2.2 |
| [q²/2, q⁴/4] (generic pair) | -241 ± 58 | -237 ± 45 | -203 ± 12 |
| sin(5q+0.7), -cos 2q (bounded, oscillatory) | -245 ± 43, -312 ± 87 | -240 ± 38, -266 ± 36 | -225 ± 21, -229 ± 31 |
| skip gate init 0 + preset | -227.5 | -249 | -292 |
| skip gate init 1 + preset / hard term (preset) | -507 / -536 | -490 / -525 | -475 / -490 |
| MLP (no structure) | -555 | -566 | -530 |

Reading: (a) the feature route beats no-feature by ~65 return units at N=8 and never hurts when the feature is smooth and unbounded; (b) that same gain is reproduced by generic
|q| and q³ features, so **the claim "known physics as input" reduces to "an extrapolation-friendly input basis"** on this testbed; the physics preset is merely the most *reliable* member
at N=8 (sd 1 vs 43 for q⁴); (c) bounded features and a hard/skip wrong formula are neutral-to-harmful, as the earlier finding said.

**Rollout prediction on the E4 tasks, median mean-nRMSE over 5 seeds, feature-swap controls** (`summary_prediction.md`, N=64, 2% noise). Selected rows (ID / OOD):

| task | hn (no preset) | phys_feat preset | phys_feat generic quadratic | phys_feat zero | phys_hn (hard) preset |
|---|---|---|---|---|---|
| kepler_pert | 0.140 / 0.132 | 0.149 / 0.107 | 0.139 / 0.111 | 0.140 / 0.132 | 0.187 / 0.142 |
| pendulum | 0.013 / 2.52 | 0.013 / **0.63** | (poly4) 0.015 / 2.11 | 0.013 / 2.52 | 0.037 / 3.95 |
| fall_drag | 0.467 / 2.47 | 0.672 / 2.56 | 1.28 / 2.90 | 0.467 / 2.47 | **0.149 / 0.298** |

Reading: on Kepler the feature route is within noise of `hn` (which already carries the Hamiltonian prior) and a generic quadratic does as well as the Newton preset; on `fall_drag` the *hard* term
wins because the preset (gravity) is exactly right for the part it covers. So the "feature, not hard term" rule holds only when the formula is wrong; when the formula is right, hard is better
(no single route wins: skip_feat OOD 0.32/0.09 on fall_drag/Kepler but 4.08 on pendulum).

**Derivative: route selection by held-out ID error** (`route_selection.md`, geometric-mean OOD nRMSE across the 4 tasks): fixed routes hn 0.573, phys_feat 0.407, skip_feat0 0.351,
skip_feat 0.340, phys_hn 0.467; **ID-selected 0.298**, hindsight-best 0.187. Selection beats every fixed route on average but fails exactly where it matters: on the pendulum every route has ID
error ≈0.013 and the OOD lock-in of a wrong preset is invisible in ID data, so it picks `hn` in 3/5 seeds (OOD 2.39 vs 0.40 for the best route). The "automatic trust rule" suggested in
`04_recommendations.md` therefore cannot work with in-distribution validation alone; it would need validation data from the extrapolation region.

**Caveats.** One task family (pendulum + 3 prediction tasks), small networks (~5K params), 2,000-3,000 Adam steps; the planning testbed's gain is an extrapolation-to-upright effect specific to the
pendulum's coverage (|θ0|<1.5). Nothing here transfers to Craftax (no physical prior there, and the earlier Craftax study found no benefit).

## 2. Actor-critic learner and the oracle_act litmus (arc.tex §2.5, §2.8, STATE TASK-20260927-011)

Code: `src/pipeline/candidate_actor_critic.py` (PPO clip + GAE(0.95) + learned MLP critic on the 39-d observation summary + t/T, identical in every arm), `scripts/run_actor_critic_litmus.py`,
`scripts/run_ac_stageb.py`, `scripts/run_action_reward_model.py`, `src/model/candidates/action_reward_model.py`, `src/environment/craftax_actreward_adapter.py`,
aggregation `scripts/aggregate_arc_ac.py`. Actor = `transformer_branch`, d=256, 277K params, T=250, 64 episodes/update, 4 epochs x 4 minibatches. Budgets: 200 updates = 12,800 episodes
(same as the REINFORCE budget diagnostic), and 780 updates = 49,920 episodes (< the 50,000 cap). Results: `output/arc_proposals/ac_*`.

**Critical review recorded before running.** `oracle_act` is a positive control by design (answer leakage: true 1-step reward of all 17 actions). Measured before training: a trivial policy
"argmax of the leaked 1-step reward" scores mean return **3.17** (crafter 2.76), i.e. the level REINFORCE learners reach with *no* leak (3.0-3.3), so the earlier "oracle_act ≈ base" could mean
"the signal adds nothing beyond what was already learned", not "the learner is underpowered". The litmus is only informative if `oracle_act` clearly exceeds both `base` and that greedy reference.

**Step 1: litmus (final sampled-policy mean return, 256 eval episodes per run; mean ± sd over seeds):**

| learner / budget | base | oracle_act | gap | separation |
|---|---|---|---|---|
| REINFORCE, 12.8k eps (arc.tex §2.5, 4 seeds) | 3.25 ± 0.14 | 3.32 ± 0.06 | +0.07 | overlapping |
| AC lr 3e-4, 12.8k eps (n=2) | 3.40 ± 0.03 | 3.68 ± 0.28 | +0.28 | non-overlapping, small |
| **AC lr 1e-3, 12.8k eps (n=10 / n=6)** | **3.47 ± 0.08** | **4.49 ± 0.26** | **+1.02** | non-overlapping (min 4.10 vs max 3.59) |
| AC lr 3e-3, 12.8k eps (n=4) | 3.90 ± 0.13 | 5.41 ± 0.23 | +1.51 | non-overlapping |
| **AC lr 1e-3, 49.9k eps (n=4)** | 4.36 ± 0.16 | **6.13 ± 0.17** (crafter 8.2 vs 4.8) | +1.78 | non-overlapping (min 5.96 vs max 4.50) |

`oracle_act` also exceeds the 1-step-greedy reference (3.17) by 1.3-3.1, so the learner is exploiting the signal beyond greedy use. **Verdict: under actor-critic the litmus passes decisively
(needs lr >= 1e-3; at 3e-4 the gap is marginal).** The learner also improves `base` itself: 3.47 at 12.8k and 4.36 at 49.9k episodes (REINFORCE: ~3.25 at 12.8k). Note the well-known greedy-vs-sampled
gap persists (base greedy 0.97 vs sampled 3.47).

**Step 2: does the passive (noop) world model help, now that the learner is decisive? (lr 1e-3; 12.8k episodes: n=8 seeds each, base n=10; 49.9k episodes: wm_untrain n=4, wm_full/oracle n=2, base n=4):**

| arm | 12.8k eps | vs base | 49.9k eps | vs base |
|---|---|---|---|---|
| base | 3.47 ± 0.08 | - | 4.36 ± 0.16 | - |
| wm_untrain (random-init WM features, no learned content: control) | 3.62 ± 0.19 | +0.15 | **4.73 ± 0.30** (4.93, 4.99, 4.34, 4.68) | **+0.37** (Welch p=0.08) |
| wm_full (trained passive WM) | 3.69 ± 0.16 | +0.22 | 4.38 (4.45, 4.31) | +0.02 |
| oracle (TRUE simulated noop future; upper bound) | 3.67 ± 0.19 | +0.20 | 4.18 (4.16, 4.20) | -0.18 |
| oracle_act (leaked 1-step action rewards; positive control) | 4.49 ± 0.26 (n=6) | +1.02 | 6.13 ± 0.17 (n=4) | +1.78 |

At 12.8k episodes `wm_full - wm_untrain` = +0.07 and `oracle - wm_untrain` = +0.06, both within seed noise; at 49.9k episodes the trained WM (4.38) and even the true noop future (4.18) are no better than `base`, while the
random-init WM is the best of the three (4.73 ± 0.30, n=4; the first two seeds alone read 4.96). **Even perfect knowledge of the noop future is worth at most ~+0.2 (12.8k) and nothing at 49.9k, and the trained WM is indistinguishable from or worse than a random network reading
the same observation**, so the noop-anticipation proposal is not supported; the lift of the random-feature control is consistent with extra observation information (a random projection of the full 1345-d observation vs the 39-d summary
the actor otherwise sees) reaching the policy, not with anticipation (side observation, +0.37 with p=0.08 at 49.9k: suggestive only; worth a dedicated test if the 39-d summary bottleneck matters). This is a much cleaner negative than the REINFORCE-era "underpowered".

**Derivative: a deployable version of the oracle_act signal.** `oracle_act` uses the simulator at policy time, so its gain is not attainable as is. `run_action_reward_model.py` trains an MLP on the
full observation to predict the 17 one-step rewards from *offline* simulator-branched labels (256 random-policy episodes) and serves those predictions in the same feature slots at policy time
(no simulator at policy time). Held-out episodes: R² of the action-dependent part 0.49; among the 18% of states where actions differ, argmax(prediction) hits a best action in 86% of states
(uniform 12%, "always the globally best action" 64%). 

A larger model (1,024 random-policy episodes, 117k train states, 20k steps; `output/arc_proposals/actreward_big`) improves the held-out numbers (R² of the action-dependent part 0.62, best-action hit rate 87%).

**Actor-critic with the learned action-reward features** (lr 1e-3; `wm_act_untrain` = same architecture, random init, the control for "extra input dims"):

| arm | 12.8k eps | 49.9k eps |
|---|---|---|
| base | 3.47 ± 0.08 (n=10) | 4.36 ± 0.16 (n=4) |
| wm_act_untrain (control) | 3.52 ± 0.08 (n=8) | 4.30 ± 0.09 (n=4) |
| **wm_act** (small model) | **3.75 ± 0.18 (n=8)**; +0.28 vs base (Welch p=0.003), +0.23 vs control (p=0.008) | 4.49 ± 0.14 (n=4); +0.13 vs base (p=0.25) |
| wm_act (big model) | 3.78 ± 0.10 (n=4) | 4.61 ± 0.03 (n=2); +0.25 vs base (p=0.04, n=2) |
| oracle_act (leaked; upper bound) | 4.49 ± 0.26 (n=6) | 6.13 ± 0.17 (n=4) |

At 12.8k episodes the *deployable* learned signal is statistically real but recovers only ~25-30% of the leaked-signal gain (+0.28 of +1.02; the larger model did not improve the policy at this budget despite better held-out R²), and at 50k episodes
it keeps only a small, not clearly significant edge (+0.13 small model, +0.25 big model with n=2, i.e. ~7-14% of the +1.78 gap) — mostly a sample-efficiency effect that the learner closes by itself. Most of the `oracle_act` gap (+1.8 at 50k episodes) is therefore not obtainable from a model that only reads the observation; a plausible (untested) reason is that the leaked value is the simulator's exact answer, including the
sparse, adjacency-dependent rewards that the learned model recovers only partially (R² 0.5-0.6).

## 3. ChunkPPO with delta > 0 (arc.tex §3, DESIGN.md §4 steps 2-3)

Code: existing `src/pipeline/chunk_ppo.py` (unchanged); new driver `scripts/run_latency_sweep2.py` (exposes every config field), aggregation `scripts/aggregate_arc_latency.py`. Metric protocol v2:
`eval_return` = first-episode stochastic return of 64 fresh envs. Results: `output/arc_proposals/latency/`.

**Parameter adjustment from the medium-scale result.** The pre-registered 1M-tick default config (64 envs x 32 cycles x k=8 = 16,384 ticks/update) yields only ~61 PPO updates. On `intercept`
the learner improves (delta=0 return 0.58) but `pendulum_goal` does not learn at all (delta=0 eval -178 = untrained level, all arms/deltas ≈ -180: the sweep cannot discriminate arms; 1 seed complete,
sweep stopped). Pilots at delta=0, 2M ticks: (A) 16 envs x 16 cycles, lr 1e-3 -> intercept 0.70, pendulum_goal -63; (B) 32 envs x 16 cycles, lr 3e-4, ent 0.003 -> 0.72, -85. Config **A** was adopted for the
sweeps ("tunedA": 2,048 ticks/update, 976 updates over 2M ticks); everything else (critic, arms, k=8, passive-pretraining charged to wm arms) is unchanged.

**Default-config 1M ticks, intercept, 6 seeds** (`summary_main_1M_default_config.md`): delta 0/1/2/4/8 augment 0.58/0.48/0.32/0.02/-0.71; wm-augment = 0.00/-0.11/+0.08/+0.20 (all overlapping);
oracle-augment = -0.06/+0.03/+0.17/+0.39. **No forecast-arm claim can be made at this under-trained budget.**

**tunedA, 2M ticks, 8 seeds per cell** (`summary_tunedA_2M_8seeds.md`; eval_return, mean ± sd; permutation p vs augment):

| env / delta | ignore | augment | wm | wm_passive | oracle |
|---|---|---|---|---|---|
| intercept 0 | | 0.69 ± 0.15 | | | |
| intercept 1 | 0.44 | 0.25 ± 0.64 | 0.60 (p=0.16) | 0.23 | 0.60 |
| **intercept 2** | 0.43 | -0.01 ± 0.57 | **0.65 ± 0.08 (p=0.003)** | 0.52 (p=0.02) | 0.65 (p=0.004) |
| intercept 4 | -0.15 | -0.02 ± 0.58 | -0.10 (p=0.80) | -0.12 | 0.21 |
| intercept 8 | -0.68 | -0.59 ± 0.50 | -0.45 | -0.42 | 0.03 (p=0.005) |
| pendulum_goal 1 / 2 / 4 / 8 (augment) | | -90.6 / -92.0 / -97.7 / -99.2 | -107 / -103 / -104 / -111 | -106 / -108 / -110 / -118 | -92 / -90.5 / -94 / -101 |

Reading (config A only; see the robustness checks below): (i) on `intercept` at delta=2 (pooled 12 seeds: wm 0.64 ± 0.08 vs augment -0.02 ± 0.56, 12/12 vs 4/12 seeds above 0.4) the forecast arms match the perfect-foresight `oracle` (0.65) while `augment` is unreliable (sd 0.57: some seeds learn, some fail) — the sample-efficiency
form of the claim (DESIGN.md issue 4), and `wm_passive` (the original no-op proposal) captures most of it (0.52) because the exogenous target dynamics dominate there, as issue 6 predicted;
(ii) the effect is **not** present at delta=1, 4 or 8, and `oracle` beats causal arms at delta=8 only because it sees unpredictable future noise (not a bar a forecaster can reach);
(iii) on `pendulum_goal` forecast arms are worse than `augment` (-7..-16, n.s.) and nothing separates `augment` from `ignore`, `oracle`, i.e. the task is not latency-limited in this range;
(iv) `augment` never beats `ignore` significantly (delta>=1), so the committed-prefix input itself carries little usable information for this learner.
**Robustness check: same sweep under a second learner config (config B: 32 envs x 16 cycles, lr 3e-4, ent 0.003; intercept, 2M ticks, 6 seeds, `summary_tunedB_2M_intercept.md`).**

| delta | ignore | augment | wm | wm_passive | oracle |
|---|---|---|---|---|---|
| 1 | 0.73 | 0.74 ± 0.06 | 0.72 | 0.74 | 0.75 |
| 2 | 0.65 | 0.67 ± 0.05 | 0.60 ± 0.09 | 0.67 | 0.72 |
| 4 | 0.42 ± 0.13 | 0.17 ± 0.25 | **0.46 ± 0.09** (p=0.02 vs augment) | 0.41 | 0.67 ± 0.03 |
| 8 | -0.18 | -0.11 ± 0.07 | -0.06 | -0.28 | 0.20 ± 0.30 |

The delta=2 advantage seen under config A **vanishes** here (`augment` 0.67 ± 0.05, n=6; it was 0.03 ± 0.57 at lr 1e-3), while a different delta (4) shows a gap, and there `ignore` (0.42) is as good as `wm`
(0.46). So in both configs the "forecast advantage" is `augment` failing to optimise at one particular delta, not `wm` beating a no-forecast baseline (`wm` never beats `ignore` significantly under
config B; under config A `wm` 0.64 vs `ignore` 0.30 ± 0.58 at delta=2 is the only such cell).

**10M-tick budget check (Kaggle, config A, intercept, 3 seeds; delta=2 and 4, plus the delta=8 run):** (`output/arc_proposals/latency/kaggle_10M*`)

| delta (10M ticks) | ignore | augment | wm | wm_passive | oracle |
|---|---|---|---|---|---|
| 2 | 0.73 ± 0.12 | 0.72 ± 0.13 | 0.73 ± 0.03 | 0.70 ± 0.04 | 0.79 ± 0.04 |
| 4 | 0.56 ± 0.11 | 0.59 ± 0.10 | 0.58 ± 0.06 | 0.59 ± 0.06 | 0.76 ± 0.05 |
| 8 (intercept) | 0.15 ± 0.22 | -0.05 ± 0.06 | -0.07 ± 0.20 | -0.11 ± 0.34 | 0.66 ± 0.05 |
| 8 (pendulum_goal) | -99.7 ± 16.6 | -96.1 ± 15.4 | -89.3 ± 15.6 | -93.1 ± 16.6 | -95.7 ± 18.3 |

With 5x the budget the gaps that existed at 2M **close completely** (delta=2: all causal arms 0.70-0.73; delta=4: 0.56-0.59), exactly as DESIGN.md issue 4 predicted (a forecast is a function of
(o_c, u_c): it can only add sample efficiency). `oracle` stays higher at delta=4/8 only because it sees unpredictable future noise, so it is not a bar a forecaster can reach.

**Verdict for proposal 3.** No reproducible benefit of `wm` / `wm_passive` over `augment`/`ignore`: apparent gaps are config- and delta-specific optimisation failures of `augment`, disappear under a
second learner config and at 10M ticks, and on `pendulum_goal` the forecast arms are slightly worse. The one structural observation that is robust is that the raw committed-prefix input (`augment`) is no better
than `ignore` (not significantly better in any of the 8 config x delta cells at 2M ticks; significantly worse only at config B delta=4, p=0.049), i.e. this learner extracts little from `u_c`; a forecast merely repackages the same information. What the sweep does establish is the
cost of latency itself: for the best causal arm the achievable intercept return falls from ≈0.7-0.8 (delta=0) to ≈0.7 (delta 1-2), ≈0.5-0.6 (delta=4) and ≈0-0.15 (delta=8).

## 4. Deviations, mistakes and caveats (kept for transparency)

* Git LFS pointers: `output/experiments/**/wm_mlp_full.pkl` etc. were 130-byte LFS pointers in the worktree; the first Kaggle push embedded them (Stage-B wm arms would have crashed).
  Fixed by `git lfs pull` locally and, for Kaggle, by retraining Stage A inside the kernel (`scripts/run_ac_stageb.py`, default args = the original 2026-09-27 settings), so the Kaggle Stage-B WM is a *re-trained*
  instance of the same recipe, not the archived checkpoint.
* Kaggle allows only 2 concurrent GPU sessions; jobs were queued with a retry loop (`scripts/kaggle_run_arc.py`). The original poller (`kaggle_run_candidates_smoke.poll_with_live_logs`) re-downloads all
  kernel outputs every minute; with checkpoint-heavy kernels that pulled 3 GB and threatened the disk, so a status-only watcher (`scripts/kaggle_watch_fetch.py`) fetching result files only replaced it.
  Checkpoints of completed local runs were deleted (`scripts/adhoc/clean_ckpts.sh`); only summaries/results are kept.
* Local GPU was shared by up to 8 containers; wall times are not comparable across runs. All comparisons use identical configs and seeds per arm.
* The `pendulum_goal` default-config sweep was stopped after seed 0 (non-learning, uninformative); its partial files remain in `output/arc_proposals/latency/main_1M/pendulum_goal/`.
* Seed counts differ by cell and are stated per table; single-learner-config results (actor-critic lr 1e-3) rest on n=6-10 seeds for the litmus but n=2 for the 50k-episode budget.

## 5. Reproduce

```
python scripts/run_structure_input_planning.py --out output/arc_proposals/structure_input/planning --seeds 0 1 2 3 4 5
python scripts/run_structure_input_prediction.py --out output/arc_proposals/structure_input/prediction
python scripts/run_actor_critic_litmus.py --out OUT --arms base oracle_act --seeds 0 1 2 3 --lr 1e-3 --greedy1-ref
python scripts/run_latency_sweep2.py --env intercept --deltas 0,1,2,4,8 --seeds 0,1,2,3 --total-ticks 2000000 --set num_envs=16 --set cycles_per_update=16 --set lr=0.001 --out OUT
python scripts/kaggle_run_arc.py --slug SLUG --run-id RUN --scripts ... --job "scripts/... args"   # Kaggle T4x2, one job per GPU
```
