# Latency-aware goal chunking: design, logical review, pre-registered tests (TASK-20260928-012)

Status: **code only, nothing trained.** This implements the redesign agreed on 2026-09-28. The target is a model that generates
action chunks toward a goal while the world keeps moving during its own computation. It replaces the
"no-op world model as a policy feature" line (TASK-008/009/010/011).

## 1. What the code implements

| Component | File | Notes |
|---|---|---|
| Latency decision process | `src/environment/latency_envs.py` | cycle = snapshot o_c; committed prefix u_c runs for delta ticks while the agent "thinks"; the new chunk P_c then runs; P_c's tail becomes u_{c+1}. `oracle_forecast` = exact s(t_c+delta) (same per-tick keys). |
| Cores | same | `pendulum_goal` (gravity = passive drift; torque-limited so goals need a pumping sequence), `intercept` (exogenous randomly turning target), `craftax` (Craftax-Classic symbolic, `step_env`, noop = reference). |
| Factored world model | `src/model/factored_world_model.py` | s' = s + dt·rate_sd·[f(x) + h(x,a) − h(x,a_ref)]; f is the reference-continuation dynamics, pretrainable on "do nothing" data; multi-step loss; explicit dt; zero-init = persistence. |
| Chunk generator | `src/model/chunk_policy.py` | Transformer with a prefix-LM mask over [goal, state, history, committed prefix, bos, plan]. The context is bidirectional; the plan is causal. It samples the chunk autoregressively as a joint distribution. Continuous-time sinusoidal token times. Temporal-distance head (context-only). MLP actor for k=1 calibration. |
| Learner | `src/pipeline/chunk_ppo.py` | Chunk-level PPO (token ratios, chunk advantage, GAE over cycles with gamma^k), asymmetric critic, hindsight distance auxiliary, WM pretraining/co-training, eval, checkpoint/resume (every 100K ticks), 24h time limit, `dry_run`. |
| Runner | `scripts/run_latency_experiment.py` | Sweep over delta × arms × seeds; `--calibrate`; `--dry-run`; `--calls-per-tick` charges the k decoding passes as latency. |
| Tests | `tests/test_latency_chunking.py` | Env semantics, oracle exactness, arm isolation (no leakage), WM identities, causality, logp consistency, GAE and hindsight vs manual computation, update paths traced with `jax.eval_shape` (no parameter updates). |

Arms. Only the **actor's** state input differs between them.

| arm | actor state | information vs `augment` |
|---|---|---|
| ignore | o_c (pretends there is no latency) | less (u_c unknown except through history correlation) |
| augment | o_c + u_c | complete (Markov) |
| wm | forecast ŝ(t_c+δ) from (o_c, u_c), f+e co-trained | same information, compressed; loses stochastic information (MSE mean) |
| wm_passive | forecast with f only; WM trained only on reference-action data | less: ignores its own committed actions; this is the original no-op proposal |
| oracle | exact s(t_c+δ) incl. future noise | more than any causal agent (perfect foresight); upper bound only |

## 2. Logical issues found while implementing

Each issue is listed with its resolution in the code. Issues 1–4 concern the agreed design itself.

1. **Planning past the commit horizon has no learning signal.** Tokens after position k never execute, so PPO's advantage does not depend on them. The receding-horizon lookahead in Real-Time Chunking works because those policies are trained by imitation. → Plan length H = k (enforced by construction). Planning further ahead would need a model-based or imitation signal.
2. **Hindsight relabelling is incompatible with on-policy PPO.** A relabelled goal g' makes the stored actions samples of π(·|s,g), not π(·|s,g'). → Relabelling feeds only the auxiliary temporal-distance head, never the policy gradient. Its target is the behaviour-policy time to a future achieved goal (a sampled future, not the first hit and not the optimal distance). It is a diagnostic of temporal-order representation, not a planner. A true goal-conditioned learner (HER, HIQL, contrastive RL) needs an off-policy method.
3. **The f + e factorisation has no structural content for discrete actions.** Any F(s,a) = F(s,a_ref) + [F(s,a) − F(s,a_ref)]. The only content is the choice of a_ref and the option to pretrain f on reference-action data. (A control-affine g(s)·u would be a real prior for continuous actions; no continuous environment is implemented.)
4. **A forecast adds no information.** ŝ(t_c+δ) is a deterministic function of (o_c, u_c), so in the infinite-data limit `wm` ≤ `augment`. Under exogenous noise it is strictly worse, because the MSE forecast is a mean. The only possible advantage is **sample efficiency**, which is what the delayed-MDP literature reports: model-based compensation vs a state augmentation whose input grows with δ. → The claim must be tested at a fixed modest budget with learning curves. Theory predicts the gap closes at large budgets. See §4.
5. **Passive pretraining uses extra environment data.** → Its ticks are charged to the wm arms' budget (`equal_env_ticks=True`), so their PPO gets fewer updates.
6. **Passive coverage depends on the environment, which gives a sharp prediction.** Passive data come from the environment's own reset distribution; releasing a real system from arbitrary states is not assumed. For `pendulum_goal`, zero torque from near-hanging states covers only small oscillations, so little benefit is expected. For `intercept`, the target's motion does not depend on the agent, so passive data covers the exogenous part completely. **The no-op model should help when exogenous dynamics dominate the forecast error, and not otherwise** (consistent with the TASK-008 pendulum result: coverage is what matters).
7. **`ignore` still has some information about u.** The history contains P_{c−1}[0:k−δ], which is correlated with its tail u_c. `ignore` is the naive baseline, not a zero-information baseline.
8. **Fair learner power.** The critic is used only in training, so it legitimately sees (o_c, u_c, goal, tick/max_ticks) in every arm (the asymmetric actor-critic pattern). Learner power is equal across arms, and time-limit truncation (treated as terminal, as in PureJaxRL) is Markov for the critic. The calibration preset turns the time feature off to match the published baseline.
9. **Reward attribution.** R_c includes the prefix ticks, which P_c cannot influence. This adds variance but no bias, because the critic sees u_c.
10. **"Generate the whole chunk at once" is not free.** Autoregressive decoding costs k sequential passes (no KV cache). With `--calls-per-tick`, a larger k raises δ. Decoupling k from latency would need a parallel joint decoder (discrete diffusion or flow); none is implemented.
11. **Timing bug (fixed during implementation).** Forecast arms put history tokens relative to the forecast time. They are relative to the snapshot, δ·dt earlier. → `ActorInput.obs_age` added, with a test.
12. **Scope limits.** The world model is Markov in the observation, which is invalid for Craftax's partial view, so Craftax is used only for calibration. δ is static per run (real systems have jitter; the time embeddings are ready for a padded variable-δ prefix). The existing macro/micro head is not reused: it factors the action index with a micro distribution that does not depend on the cluster, so it is not temporal abstraction. Latent-subgoal hierarchy (HIQL/Director) is **deferred**: it needs the off-policy goal-conditioned value learning that issue 2 excludes, and should wait for a calibrated flat learner.

## 3. Budget and metrics

- Budget = environment ticks (rollout + passive). The world model's gradient steps are compute, not data. They are not charged to the tick budget; wall time is reported.
- Primary metric: mean return of episodes finished during training rollouts (stochastic policy; the PureJaxRL reporting convention). Secondary: greedy evaluation every `eval_every` updates. `forecast_skill` (1 − err/err_persistence of the actor's state input at t_c+δ, measured on the policy's own states) is logged. It is 0 for ignore/augment and 1 for oracle by construction, which gives a self-check.
- All numbers come from real environment ticks and model forward passes (data-integrity rules).

## 4. Pre-registered steps and decision rules

1. **Learner calibration (must pass first).** `--calibrate` runs plain PPO (k=1, δ=0, MLP) on Craftax-Classic at 1M ticks, ≥2 seeds. First check the hyperparameters against `craftax_baselines/ppo.py`. The result must land in the published PPO range at that budget. If it is clearly below, stop: that is a learner bug, not a finding. 1b: the same with `actor=transformer` (k=1), to confirm the chunk actor is not weaker than the MLP.
2. **Main sweep** on `intercept`, then `pendulum_goal`: δ ∈ {0,1,2,4,8}, k=8, 1M ticks, ≥4 seeds, all five arms.
   - Expected: ignore degrades fastest with δ; oracle is the ceiling; δ=0 is a shared anchor.
   - **WM claim supported** if `wm` > `augment` at δ ≥ 4 with non-overlapping seed ranges, and the gap grows with δ.
   - **Refuted for this budget** if `augment` ≈ `oracle`, meaning the learner compensates from (o_c, u_c) directly and forecasting has nothing to add.
   - Then one 10M-tick point at δ=8: theory predicts the `wm`–`augment` gap shrinks.
3. **No-op model in its intended setting.** In `intercept`, `wm_passive` vs `wm` measures what is lost by ignoring the agent's own committed actions. Prediction: small at δ·v_agent ≪ catch radius, and growing with δ. In `pendulum_goal`, `wm_passive` is expected to be poor (the agent's torque effect is not negligible there).

Long runs need operator authorisation (ADR-002 covers Craftax Phase II; the intercept and pendulum sweeps are cheap but not yet authorised).
