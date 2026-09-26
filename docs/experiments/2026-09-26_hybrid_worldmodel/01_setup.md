# Setup

Code: `scripts/run_hybrid_worldmodel.py` (prediction benchmark), `scripts/run_partial_obs.py`, `scripts/run_wm_planning.py`,
aggregation `scripts/aggregate_hybrid_final.py` / `aggregate_hybrid.py` / `aggregate_sweep.py`. Everything ran in the local Docker
image on the RTX 3060 Ti (several containers in parallel; two jobs at once occasionally hit GPU OOM and were relaunched).

## Architectures (all continuous-time vector fields, same RK4 integrator)
- `mlp`: dx = MLP(x).
- `hn`: (J-R) grad H, H = |p|^2/2 (known kinetic) + MLP(q); R learned PSD (existing `HamiltonianOps`).
- `phys`: preset only (parameter-free).  `phys_hn`: preset potential as a **hard term** + learned MLP(q) + R.
- `phys_hn_mu`: as phys_hn with a learned scale on the preset (known form, unknown constant).
- `phys_feat`: preset potential enters as an **input feature** of the learned potential, H = |p|^2/2 + MLP([q, Vp(q)]).
- `skip_feat`: phys_feat + a direct skip a*Vp(q), a learned, initialised 1.  `skip_feat0`: same, a initialised 0 (closed).
- `rssm` (Dreamer-lite): GRU + 4x8 categorical latent, straight-through, 1% unimix, DreamerV3 KL balancing 0.5/0.1 with free bits,
  teacher-forced posterior pass + open-loop imagination pass. **Not DreamerV3** (no symlog heads, no actor-critic, no image encoder).
- `hyb_sum_mlp` / `hyb_sum_rssm` / `hyb_force_mlp` / `hyb_gate_mlp`: phys_hn + residual (MLP / Dreamer-lite / force-only / gated).
- `hyb_tf_rssm`: Transformer fusion (3 tokens: physics field, learned-Hamiltonian field, latent residual; 1 attention layer) instead of a sum.
- Oracle arms (`oracle_*`) receive the TRUE potential on purpose as upper bounds. They are not evidence for a method.

## Tasks (true equations known to the harness only)
| task | truth | preset given to the model |
|---|---|---|
| kepler_pert | Newton + repulsive +0.25/r^5 perturbation + linear drag 0.02 | Newton, mu=1 (incomplete) |
| kepler_wrongmu | same truth | Newton with WRONG mu=0.7 |
| fall_drag | gravity + quadratic drag 0.3\|v\|v | gravity only |
| pendulum | -sin(q) with damping 0.05 | harmonic q^2/2 (wrong at large angles); OOD = large amplitude |
| pendulum swing-up (planning) | torque-limited (\|u\|<=0.7), start hanging, goal upright | as pendulum |

Training: 2,000 Adam steps, 10-step open-loop multi-step loss, N=64 or 512 trajectories (planning: 8/32/128), 2% observation
noise, up to 5 seeds. Metric: mean normalised RMSE over a 200-250-step open-loop rollout on held-out initial conditions
(ID) and out-of-distribution ones (OOD). Planning: CEM MPC (3 x 256 samples, horizon 60, replan every step) using only the
learned model; the environment is stepped with the true dynamics; 24 episodes per model/seed.
Partial observation: only noisy position observed, momentum estimated by finite difference, a learned filter, or oracle.

## Mistakes made and corrected (kept in the repo for transparency)
- First Kepler task used an attractive perturbation; 30-34% of test orbits plunged or were ejected, so every arm (even the
  parameter-free preset) was worse than persistence. Found by direct simulation, fixed to a repulsive one, rerun. The invalid results are in
  `output/experiments/2026-09-26_hybrid/main_invalid_kepler_v1/` and are not used anywhere.
- First planning task could not be solved even with the true model (random shooting, short horizon, success 0); replaced by CEM
  with horizon 60. The uninformative first version is not reported.
