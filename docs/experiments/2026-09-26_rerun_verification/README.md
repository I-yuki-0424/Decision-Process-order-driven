# Re-run verification after pipeline fixes (2026-09-26)

Data: `output/experiments/2026-09-26_rerun/` (`local/` Craftax, `kaggle_mdp/` Craftax MDP candidates, `physics_i6*/` idea6 ablation,
`dreamer/` Dreamer-lite fidelity). Setup: Craftax via `CraftaxObsAdapter` (`--adapter obs`), REINFORCE, 250 updates x 8 episodes (2,000 episodes),
T=200, d_model=32, lr in {3e-4, 1e-3, 3e-3}, **1 seed per config**, sampled-policy eval return (32-64 episodes). Not re-run (already valid):
`transformer_branch` fixed-adapter study (TASK-004), Dreamer-lite baseline (TASK-006 `main`), physics baselines (TASK-005).

## 1. NaN entropy (variant_5_1 / 5_4): root cause and fix
`solve_bellman_topk` is not a contraction: the "discount" G = gamma*(id ratio) reaches 9e3 and P*G multiplies Q each iteration, so Q overflows float32
(max|Q| 5e5 at init, `inf` after ~14 iterations) -> `inf-inf=NaN`. Real Craftax obs, random init: variant_5_1 51/164 non-finite logits; variant_5_4 0/164 at init
but NaN entropy in 6-22 of 250 training updates. Fix (opt-in, default unchanged): `stable=True` clips P to [0,1], normalises P per (s,a), clips G to <= gamma.
Registry: `*_stable`. Result: 0 non-finite in 164 passes (max|Q| 600) and 0 NaN-entropy updates in all stable training runs (raw: variant_5_1 250/250).
Diagnostic: `scripts/adhoc/diagnose_mdp_nan.py`.

## 2. Craftax candidates on the fixed adapter (return; random policy = 1.47)
| candidate | lr 3e-4 | lr 1e-3 | lr 3e-3 | untrained |
|---|---|---|---|---|
| transformer_branch | 2.64 | 2.68 | 2.80 | 1.07 |
| variant_5_2 | 2.62 | 3.15 | 2.93 | 1.63 |
| variant_5_3 | 3.08 | 3.25 | 2.98 | 2.07 |
| variant_5_1 (raw / stable) | 0.67 / 1.52 | 0.15 / 1.61 | 0.86 / 2.12 | 1.96 / 2.17 |
| variant_5_4 (raw / stable) | 2.17 / 1.98 | 0.60 / 0.50 | 0.54 / 0.50 | 2.14 / 1.84 |
| mdp_branch (raw / stable) | 0.71 / 0.45 | 0.66 / 0.64 | 0.62 / 0.63 | 0.45 / 0.57 |

* The old "nothing beats random" conclusion was an adapter artefact for the attention/dense candidates (variant_5_2, 5_3, transformer_branch: 2.6-3.25 vs 1.47).
* It **still holds for the MDP-search candidates** (variant_5_1/5_4, mdp_branch) even after the NaN fix: no trained config is meaningfully above its own untrained
  init, and most are far below random. Consistent with the STATE hazard (argmax discretisation blocks policy gradient), so NaN was not the reason.
* Differences among transformer_branch / 5_2 / 5_3 (2.6-3.25) are within what one seed and 32-64 eval episodes can resolve; no ranking is claimed.
  Per-arm tuning was one 3-point lr sweep, seed 0 only; other hyper-parameters (entropy coef, d_model) untuned.

## 3. idea6_full ablation (Kepler, 128 trajectories, noise 0, 3 seeds; ID end-of-rollout nRMSE)
hn_sep 0.12-0.18 | idea6_full ~1.7 (OOD 30-43) | no gate (K=1: provably identical) 1.7 | RK4 only 1.7 | no residual 1.1-1.3 | no residual+RK4 0.7-0.9 |
scaled HNN init 0.4-0.5 | no residual+RK4+init 0.45-0.58 | **+ no learned dissipation R: 0.10-0.12 (OOD 0.61-0.74, same as hn_sep)**.
Causes on conservative systems, in order: learned dissipation R (energy drift), HNN init scale 0.1 / semi-implicit Euler (long horizon), unconstrained residual
(10-step error 0.07 vs 0.004). The gate is irrelevant at K=1. On dissipative `fall_drag`, removing R hurts (5.8 vs 1.4), so R must stay there: `WorldModel` is a
design trade-off, not a single bug; the source `WorldModel` class was NOT modified (ablations live in `scripts/run_physics_benchmark.py` `i6_*` arms).

## 4. Dreamer-lite fidelity (kepler_pert / pendulum, N=512, noise 0.02, 3 seeds; ID end nRMSE, median)
Original Dreamer-lite (28K params): kepler_pert 8.46, pendulum 0.28 (hn 0.25 / 0.03). Larger (163K): 2.50 / 0.38; + LayerNorm/SiLU/symlog: 6.07 / 1.84 (worse);
+ 16x16 latent (307K): 1.83 / 0.59; + 12k steps: 1.41 / 0.68. So part of the earlier weakness was size/simplification (6x better on kepler_pert), but it remains
~6x worse than the 4K-parameter Hamiltonian model, and improvement is non-monotone. Still NOT full DreamerV3 (no symlog heads/twohot, no actor-critic, parameter-free LN).

## Caveats
Single seed for Craftax, 2,000 episodes only; MDP-candidate seeds 1-2 were started on Kaggle then abandoned (not used). Local Craftax variant_5_2/5_3 first crashed with GPU OOM
(two containers sharing the GPU) and were re-run. Kaggle launches from Git Bash need `MSYS_NO_PATHCONV=1`; `scripts/kaggle_fetch_abs.py` recovers mangled output paths.
