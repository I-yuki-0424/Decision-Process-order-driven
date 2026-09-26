# Summary of the 2026-09-25/26 experiment series: candidate architectures, Idea 6 world model, physics benchmark, hybrid world model

Consolidates five experiments. Detailed per-experiment write-ups are linked in section 9. All numbers come from real training runs and
real environment/simulator steps; raw data is under `output/experiments/`. Evidence level is stated for each claim; several claims rest on
few seeds and small models and are marked as hypotheses.

---

## 1. Overview of what was tested and why

| # | Experiment | Question | Domain | Date |
|---|---|---|---|---|
| E1 | Candidate convergence | Do the DPOD candidate architectures learn Craftax within ~2,000 episodes? | Craftax-Classic (RL) | 09-25 |
| E2 | Idea 6 on Craftax | Does a passive-dynamics world model (Hamiltonian NN) help an action Transformer? | Craftax-Classic | 09-25 |
| E3 | Closed-form physics benchmark | Is the Hamiltonian prior effective where physics really is conservative? | Kepler, falling body with drag, sine | 09-26 |
| E4 | Hybrid world model | Does "known physics + HN + Dreamer-like latent, fused" beat simpler models? | Kepler (perturbed), falling body, pendulum | 09-26 |
| E5 | Planning test | Does world-model quality translate into decision quality? | Pendulum swing-up with MPC | 09-26 |

Compute: local WSL2 Docker with an RTX 3060 Ti (several containers in parallel) plus Kaggle GPU for part of E1 and E2. Runs are small by design
(episode cap per proposal below 50,000; actual: 2,000 to 4,800 episodes in RL, 2,000 Adam steps in supervised runs).

---

## 2. Model formulas

Notation: `x=(q,p)` phase-space state, `J=[[0,I],[-I,0]]`, `R=L L^T` (positive semi-definite dissipation, `L` lower-triangular),
`H` scalar Hamiltonian, `dt` step size. All learned vector fields are integrated with RK4 unless noted.

### 2.1 Action-selecting candidates (E1, E2)
Input `N={A,S,H,T}` (actions, system state, history, target) encoded by channel-independent projections, then attention over tokens. Output
`DecisionVector d` with `action_logits`, `estimated_costs`, `progress_rate_pred`, `predicted_next_state`, `validity`. Candidates: `variant_5_1..5_4`,
`mdp_branch`, `transformer_branch` (MDP-search variants route action selection through a discretisation `argmax`).

Policy training (both E1 and E2), per batch of episodes:

```
G_t   = sum_k gamma^k r_{t+k}              (gamma = 0.99, return-to-go)
b_t   = batch mean of G_t at timestep t     (baseline)
A_t   = (G_t - b_t) / sigma(A)              (normalised advantage)
L     = -mean( logpi(a_t | s_t) * A_t ) - 0.01 * H[pi] + 0.1 * mean( (estimated_costs - 1)^2 )
```
The `argmax_legacy` control replaces `A_t` by the raw `G_t` weight. Optimiser AdamW lr 1e-3 with gradient clipping 1.0.

### 2.2 Idea 6 world model (existing `WorldModel` class, unchanged)
```
m_t   = gate(x_t, wind_t)                              (hard one-hot, straight-through)
H_m   = |p|^2 / 2m + sum_i V_i(q) [preset potentials] + MLP_theta(q)
x_dot = (J - R_m) grad H_m(x) + G u + F_m(x, eta, wind) + r_theta(x, wind, m)
x_{t+1} = semi_implicit_Euler(x_t, x_dot, dt)
```
`F_m` are preset non-conservative forces (drag, buoyancy, ground contact), `r_theta` is a zero-initialised residual MLP. Presets are hand-specified.

### 2.3 Craftax passive world model (E2)
`q` = standardised vitals (health, food, drink, energy) from the real observation; `p = Enc(f)` learned momentum; `wind = W(f)` learned exogenous code
from a 39-dimensional observation summary `f`. Predicts `q_{t+1..t+8}` under the noop action (open-loop rollout), trained on real noop transitions.
Variants: `hn` (the `WorldModel` above with all presets masked off), `mlp` (`dx = MLP(q,p,wind)`), and persistence. The policy then receives 8 extra
state features `[q_{t+8}-q_t, q_{t+1}-q_t]` (clipped to [-5,5]).

### 2.4 Physics-benchmark arms (E3)
- `mlp`: `x_dot = MLP(x)`.
- `hn_gen`: `x_dot = J grad MLP(x)` (fully generic Hamiltonian).
- `hn_sep`: `H = |p|^2/2 + MLP(q)`, `x_dot = J grad H` (known kinetic term, learned potential).
- `port_sep`: `x_dot = (J - R) grad H` with learned `R`.
- `idea6_full`: the `WorldModel` class as implemented (K=1, presets off, own semi-implicit Euler with 8 substeps).
- `oracle_port`: `port_sep` plus the TRUE potential given as a preset (upper bound by construction).
- `persistence`: `x_{t+k} = x_t`.

### 2.5 Hybrid world-model arms (E4, E5)
Known-physics term `H_p(x) = |p|^2/2 + V_p(q)` where `V_p` is a preset that is deliberately incomplete or wrong. Learned term `H_theta(q) = MLP_theta(q)`.

```
phys_hn     :  x_dot = (J - R) grad [ H_p + H_theta ]                          (preset as HARD term)
phys_hn_mu  :  same with H_p using exp(mu) * V_p                               (learned constant)
phys_feat   :  H = |p|^2/2 + MLP_theta([q, V_p(q)])                            (preset as INPUT FEATURE)
skip_feat   :  H = |p|^2/2 + MLP_theta([q, V_p(q)]) + a * V_p(q)   (a init 1)   (feature + learned-scale skip)
skip_feat0  :  same, a init 0                                                   (skip starts closed)
hyb_sum_*   :  x_dot = (J - R) grad [H_p + H_theta] + r(x; h', z)              (residual added, MLP or Dreamer-lite)
hyb_tf_rssm :  x_dot = Transformer( tokens[ f_phys, f_hn, r ], x )             (learned fusion)
hyb_gate_mlp:  x_dot = f_port + sigmoid(g(x)) * r(x)                            (gate init closed)
hyb_force_mlp: residual added to dp/dt only
```
Dreamer-lite (reduced RSSM; **not DreamerV3**): `h' = GRU(h, z)`, prior `p(z'|h')`, posterior `q(z'|h', obs)`, categorical `z` with 4 groups x 8 classes,
straight-through gradients, 1% unimix, KL balancing (0.5 dynamics / 0.1 representation) with 1-nat free bits, teacher-forced posterior pass plus open-loop
imagination pass; decoder gives a residual field `r(x; h', z)`. No symlog heads, no actor-critic, no image encoder.

Oracle arms (`oracle_*`) receive the TRUE potential and are upper bounds, not evidence for a method.

---

## 3. Experimental conditions

| | E1 | E2 | E3 | E4 | E5 |
|---|---|---|---|---|---|
| Data | on-policy Craftax rollouts | random-policy noop branches (256 eps x 200 steps) + policy RL | exact-integration trajectories | as E3 + perturbed tasks | offline random-torque trajectories (N = 8/32/128) |
| Training | REINFORCE, 250 updates x 8 eps = 2,000 eps/run, T=200, d_model 32 (~6K params) | WM: 6,000 Adam steps; policy 300 x 16 = 4,800 eps, d_model 512 (~0.88M params), T=250 | 3,000 Adam steps, 10-step window loss, N=16/128/1024 | 2,000 steps, N=64/512 (sweeps: N=64) | 2,000 steps |
| Seeds | 1 | 2 per arm | 3 | 3 (5 pooled for key arms) | 3 |
| Noise | n/a | n/a | 0 and 2% | 2% (some 0%) | 2% |
| Metric | eval return, Crafter score | held-out MSE; eval return | normalised RMSE vs horizon, invariant drift | mean nRMSE over 200-250 step rollout, ID and OOD | planning return (24 episodes), success rate |
| Baselines | uniform-random, untrained | persistence, MLP, untrained-WM control | persistence, MLP | persistence, MLP, oracle | true-model MPC, null-model MPC |

Physics tasks (all `m=1`, exact equations known only to the harness): Kepler two-body `H=|p|^2/2 - 1/|q|`; falling body `H=p^2/2+g q` with drag `-c|v|v`
(`g=1, c=0.3`); noisy harmonic oscillator `H=(p^2+q^2)/2`; perturbed Kepler (`+0.25/r^5` repulsive term, linear drag 0.02, preset = Newton, `mu=1`);
Kepler with a wrong preset constant (`mu=0.7`); damped pendulum (`-sin q`, preset = harmonic `q^2/2`, OOD = large amplitude); torque-limited swing-up
(`|u|<=0.7`, goal upright, CEM MPC with horizon 60, 3x256 samples, replanning each step).

---

## 4. Results

### 4.1 E1: candidate convergence on Craftax (old adapter; 2,000 episodes; uniform-random reference return 1.46)
| candidate | eval return untrained -> trained |
|---|---|
| transformer_branch | 0.54 -> 1.11 (1.24 with `argmax_legacy`) |
| variant_5_2 | 0.46 -> 0.82 (1.41 legacy) |
| variant_5_3 | -0.32 -> 0.11 |
| mdp_branch | 0.07 -> -0.10 |
| variant_5_1 | -0.32 -> -0.50 |
| variant_5_4 | -0.52 -> -0.54 |

No trained candidate beat the random policy. MDP-search candidates emitted raw values up to ~1e7 with -1e9 masks, making the softmax one-hot and the entropy NaN
(normalising the logits fixed `mdp_branch`'s NaNs but it still did not learn; `variant_5_1`/`5_4` NaN persisted from an undiagnosed source).
**Later found to be invalid as a test of architecture:** the old adapter hid the observation map/inventory and re-drew action features every episode (see 4.2).

### 4.2 E2: Idea 6 on Craftax
Stage A (world model of the noop dynamics, held-out standardised MSE at horizon 8): persistence 0.319, MLP 0.144, HN 0.132 (58% / 55% skill vs persistence).
Stage B (with a new observation-exposing adapter; transformer_branch 0.88M params; random = 1.46):

| arm | return seed0 / seed1 | mean |
|---|---|---|
| base (no WM features) | 2.78 / 3.10 | 2.94 |
| + trained HN WM | 2.94 / 2.93 | 2.94 |
| + trained MLP WM | 3.17 / 3.29 | 3.23 |
| + untrained HN WM (control) | 2.95 / 3.32 | 3.14 |

The jump from ~1.4 to ~3.0 is due to the adapter fix, not to the world model. World-model features gave no measurable benefit; the untrained control matches the trained one.

### 4.3 E3: closed-form physics (mean nRMSE over the rollout, N=128, clean; lower is better)
| task | split | persistence | mlp | hn_gen | hn_sep | port_sep | idea6_full | oracle |
|---|---|---|---|---|---|---|---|---|
| Kepler | ID | 1.375 | 0.660 | 0.101 | **0.075** | 0.241 | 1.271 | 0.001 |
| Kepler | OOD | 1.660 | 1.079 | 0.459 | **0.259** | 0.291 | 12.038 | 0.000 |
| Falling + drag | ID | 2.726 | 0.753 | 1.616 | 0.802 | 0.384 | **0.296** | 0.142 |
| Falling + drag | OOD | 2.725 | **0.108** | 3.178 | 2.620 | 2.280 | 0.742 | 0.304 |
| Sine | ID | 1.319 | 0.055 | 0.041 | **0.020** | 0.023 | 0.035 | 0.000 |
| Sine | OOD | 3.059 | 3.086 | 3.132 | 2.932 | 2.896 | **0.789** | 0.000 |

Kepler energy drift over the rollout: MLP 44%, `hn_sep` 0.6%. With 16 training trajectories `hn_sep` (end error 0.23) already beat the MLP trained on 1,024 (1.7).

### 4.4 E4: hybrid world model (N=64, 2% noise, median over 5 seeds unless noted; mean nRMSE, ID / OOD)
| task | mlp | hn | phys_hn (hard preset) | phys_feat | skip_feat | skip_feat0 | hyb_sum_mlp |
|---|---|---|---|---|---|---|---|
| kepler_pert | .362/.641 | **.130**/.136 | .217/.161 | .172/.174 | .135/**.101** | .176/.179 | .404/.567 |
| kepler_wrongmu (mu=0.7) | .362/.641 | .129/.136 | .350/.263 | .192/.227 | **.124/.116** | .197/.231 | .363/.543 |
| fall_drag | .883/**.147** | .475/2.483 | .142/.295 | .670/2.553 | **.141**/.315 | .507/2.169 | .147/.341 |
| pendulum | .027/1.268 | .015/2.770 | .068/3.937 | **.014**/.589 | .014/4.013 | .014/**.361** | .025/3.552 |

Dreamer-lite components (3 seeds): stand-alone `rssm` Kepler 2.6 to 2.7 (worse than persistence 1.26), `hyb_sum_rssm` Kepler 1.3 to 3.2, `hyb_tf_rssm` Kepler 0.31 to 0.51;
fall_drag: `rssm` 0.344, `hyb_sum_rssm` 0.242, `hyb_tf_rssm` 0.680. Parameters: 28K to 37K versus 4.4K for `hn`.

Other E4 results:
- Parameter count (hidden 16 to 256, fall_drag ID): `phys_hn` 0.134 -> 0.122 (325 -> 66.6K params), `mlp` 0.717 -> 0.967, `hyb_sum_mlp` 0.194 -> 0.105, `rssm` 0.239 -> 0.442.
- Residual penalty `lam` 0 / 0.1 / 1 (fall_drag ID, `hyb_sum_mlp`): 0.118 / 0.673 / 1.299.
- Learnable constant on the preset (`phys_hn_mu`): kepler_wrongmu ID 0.350 -> 0.191.
- Kepler noise-free, 2,000 -> 6,000 steps: `hn` 0.029 -> 0.013, `phys_hn` 0.129 -> 0.051, `hyb_sum_mlp` 0.280 -> 0.130.
- Partial observation (only noisy position observed; Kepler, obs noise 2%): momentum from finite difference 1.48 to 2.99, from a learned filter 0.38 to 0.56, oracle state 0.13 to 0.41; ranking of dynamics models unchanged.

### 4.5 E5: planning (pendulum swing-up; true-model MPC return -196, null-model -799; success 100% for every learned model)
| model | N=8 | N=32 | N=128 | preset |
|---|---|---|---|---|
| mlp | -600 | -589 | -576 | none |
| hn | -271 | -265 | -253 | none (known kinetic only) |
| phys_hn | -531 | -517 | -479 | wrong small-angle, hard term |
| hyb (phys_hn + MLP residual) | -461 | -484 | -524 | as phys_hn |
| **phys_feat** | **-204** | **-204** | **-207** | wrong small-angle, as feature |
| skip_feat | -491 | -466 | -454 | wrong preset, skip open |
| skip_feat0 | -234 | -227 | -234 | wrong preset, skip closed |
| oracle_phys / feat / skip | -197 / -200 / -197 | -196 / -201 / -196 | -196 / -200 / -196 | TRUE potential |

---

## 5. Corrections made during the series (results that were invalid and were replaced)
1. E1's conclusion about candidate architectures was measured on an adapter that hid the observation and randomised action features; it should be re-run (only `transformer_branch` has been re-tested on the fixed adapter).
2. First hybrid Kepler task had an attractive perturbation; 30-34% of orbits plunged or were ejected, making every arm worse than persistence. Fixed (repulsive term), rerun, invalid runs kept in `main_invalid_kepler_v1/`.
3. First planning task could not be solved even with the true model (random shooting, short horizon). Replaced by CEM with horizon 60.
4. In E3, the end-of-rollout error is degenerate for the sine task (test horizon is almost 4 periods); the mean-over-rollout metric is used instead.

---

## 6. Hypotheses supported by the data (with evidence level)
| # | Hypothesis | Evidence | Level |
|---|---|---|---|
| H1 | A Hamiltonian prior with a known kinetic term is substantially better than a generic model when the dynamics are conservative and coordinates canonical | E3 Kepler/sine, E4 Kepler; energy drift 0.6% vs 44% | strong within tested tasks (3-5 seeds) |
| H2 | The Hamiltonian family gives no advantage (sometimes harm) for dissipative, state-dependent friction | E3/E4 fall_drag (`hn` 0.475, generic MLP wins OOD 0.147) | moderate |
| H3 | Extrapolation beyond the training support requires the correct functional form; learned MLP/GRU/Transformer parts do not extrapolate | E3 sine OOD, E4 pendulum/fall OOD, E5 | consistent across tasks; not proven |
| H4 | A wrong preset used as a hard term is worse than no preset; used as an input feature it is far less harmful and can be very helpful | E4 kepler_wrongmu, E5 (-204 vs -480..-530) | moderate; one planning task |
| H5 | A direct skip on the preset helps when its form is right and hurts when wrong, so the skip is a trust decision | skip_feat vs skip_feat0 across Kepler/fall/pendulum | moderate |
| H6 | Structure matters more than data or parameters for these tasks | E4 sweep (325 vs 66K params equal), E5 (N=8 ~ N=128) | moderate (toy tasks) |
| H7 | A free residual added to a structured model can absorb the dynamics and hurt long rollouts; penalising it hurts when the physics is incomplete | E4 Kepler hyb_sum (0.40 vs 0.13), lam sweep | moderate |
| H8 | A reduced Dreamer-like recurrent latent is unstable on long deterministic-physics rollouts and gives no advantage at 6-90x the parameters | E4 rssm/hyb_sum_rssm | moderate for the reduced version; untested for real DreamerV3 |
| H9 | Under partial observation, state estimation dominates error more than the choice of dynamics model | E4 partial-observation test | moderate; filter used simulator labels |
| H10 | Rollout error alone under-reports differences in decision quality; success rate saturates while return differs 3x | E5 | consistent |
| H11 | A world model without a physical prior does not help policy learning in Craftax | E2 (untrained control equals trained) | moderate (2 seeds) |

---

## 7. Conclusions
1. **Idea 6 as an architecture is effective only in a narrow, theory-predicted regime:** conservative or lightly damped, smooth, canonical-coordinate physics. It was not effective in Craftax, in dissipative nonlinear friction, or as implemented in the `WorldModel` class (the class underperformed its own components; the cause is not isolated, the integrator/gate/residual are the suspects).
2. **The proposed unified model (known physics + HN + DreamerV3-like latent, fused) did not beat a plain structured Hamiltonian model** in these tests. The reduced Dreamer-like component was the least reliable part; Transformer fusion was mixed and not better than a sum.
3. **The proposed division of labour is inverted in the data:** known equations/HN carry structure and extrapolation; learned parts only recover what the equations omit inside the data support.
4. **Most promising design direction found:** `(J - R) grad H` core with the known kinetic term; known physics supplied as input features to the learned potential, plus an optional learned-scale skip when the preset's form is trusted; an unregularised force-only residual only for known unmodelled dissipation.
5. **Evaluation practice:** judge world models by planning return in the true environment, use oracle arms only as labelled upper bounds, compare to persistence and to parameter-matched MLPs, and always check the task for degenerate cases (two of this series' first designs were invalid).
6. **The cost-parity claim (same results as modern LLM-based architectures at lower cost) is untested:** no LLM baseline was run; Craftax scores here are from ~2,000-4,800 episodes of REINFORCE and are not comparable to published results.

---

## 8. Limitations
- Few seeds (1 to 5), mostly ~5K-parameter models, 2,000 optimisation steps, hyper-parameters not tuned per arm; several gaps are within seed ranges.
- Toy physics tasks and one control task; no transfer claim to non-physical domains.
- Reduced RSSM only; DreamerV3's representation learning on high-dimensional input was not tested.
- The learned partial-observation filter used true-momentum labels from the simulator.
- Craftax candidate comparison (E1) was on the flawed adapter and is only partly re-tested.

## 9. Pointers
- E1: `docs/experiments/2026-09-25_candidate_convergence/` · data `output/experiments/2026-09-25_convergence/`
- E2: `docs/experiments/2026-09-25_idea6_worldmodel/` · data `output/experiments/2026-09-25_idea6/`
- E3: `docs/experiments/2026-09-26_physics_benchmark/` · data `output/experiments/2026-09-25_physics/`
- E4/E5: `docs/experiments/2026-09-26_hybrid_worldmodel/` (includes `00_log.md`) · data `output/experiments/2026-09-26_hybrid/`
- Tasks in `docs/core/STATE.yaml`: TASK-20260925-003/-004, TASK-20260926-005/-006.
