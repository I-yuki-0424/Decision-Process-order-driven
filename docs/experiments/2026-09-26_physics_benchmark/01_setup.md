# Setup

Code: `scripts/run_physics_benchmark.py`, `scripts/aggregate_physics.py`. Ran on the local RTX 3060 Ti in Docker (three
containers in parallel, one per task; Kaggle not used because these are tiny supervised jobs). 324 training runs plus references.

## Tasks (exact equations; data from RK4 with 50 sub-steps per sample step; models never see the equations)
| task | equations | n | properties | train horizon / test horizon |
|---|---|---|---|---|
| kepler | planar two-body reduced, H = p²/2 − 1/\|q\| | 2 | conservative, nonlinear, energy + angular momentum | 10 / 250 steps (dt 0.05, about 2 periods) |
| fall_drag | H = p²/2 + g q plus drag −c\|v\|v (g=1, c=0.3) | 1 | **dissipative**, nonlinear | 10 / 200 steps |
| sine | H = (p²+q²)/2 | 1 | conservative, linear; noisy observations | 10 / 250 steps (dt 0.1) |

Test initial conditions are held out; ID uses the training range, OOD shifts it (Kepler: larger radii; fall: 3x higher;
sine: 2-3x amplitude). Observation noise 0 or 2% of each state's std, applied to training data and to the test input x0
(the target truth is clean).

## Arms (each learns a continuous vector field; hidden width 64, 2 layers; about 4.4-5.2K params, matched within 20%)
- `mlp`: dx = MLP(x), generic neural ODE (RK4).
- `hn_gen`: J ∇H with H an MLP of (q, p) (generic Hamiltonian NN, RK4).
- `hn_sep`: existing `HamiltonianOps.hamiltonian` + J, kinetic p²/2m known, V(q)=MLP learned (RK4).
- `port_sep`: same plus learned PSD dissipation R, (J−R)∇H (existing `port_field`, RK4).
- `idea6_full`: existing `WorldModel.step` unchanged, K=1, presets off, its own semi-implicit Euler with 8 substeps.
- `oracle_port`: port_sep plus the TRUE potential as a preset. **Uses task knowledge on purpose**: sanity check that the
  harness is correct and an upper bound for "known formulas" (Approach C); not a fair comparison.
- `persistence`: x stays at x0.

Training: 3,000 Adam steps (cosine decay), batch 128 windows, multi-step loss over a 10-step open-loop rollout, N = 16 / 128 /
1024 trajectories, 3 seeds; tables report the median over seeds. Metric: normalised RMSE (per-state std) vs rollout horizon, and
drift of the TRUE invariants along the predicted rollout.

## Known metric limitation
The end-of-rollout error is misleading for sine: the test horizon is 25 time units, close to 4 periods, so persistence
looks good at the last step by coincidence. Use the mean-over-horizon table for sine (persistence 1.32 there). With noisy
input, Kepler's end error is noise-limited: even `oracle_port` reaches about 1.0-1.4 because initial-state noise is amplified
over 2 orbits.
