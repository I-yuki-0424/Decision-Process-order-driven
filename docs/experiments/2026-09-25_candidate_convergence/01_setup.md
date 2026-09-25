# Setup
- Task: Craftax-Classic via `CraftaxEnvAdapter`; episode cap T=200 steps; d_model=32.
- Budget per run: 250 updates x 8 parallel episodes = 2,000 episodes (limit was <50,000). Fixed seed 2026, single seed (no variance estimate).
- Training: REINFORCE, discounted return-to-go (gamma 0.99), per-timestep batch-mean baseline, normalised advantages, entropy bonus 0.01, AdamW lr 1e-3, grad clip 1.0.
- Eval: 64 episodes with the same jitted rollout for trained / untrained (same init) / uniform-random reference.
- Compute in parallel:
  - Local: WSL2 Docker (`docker/Dockerfile` image `dpod-local`), RTX 3060 Ti 8 GB: transformer_branch, variant_5_2, variant_5_3 (+ A/B runs, + normlogits runs).
  - Kaggle (kernel `bfloat16/dpod-candidates-convergence`, credentials from repo `kaggle.json`): variant_5_1, variant_5_4, mdp_branch. Kernel ran 22.7 min on GPU.
- Code: `src/pipeline/candidate_experiment.py`, `scripts/run_candidate_experiment.py`, `scripts/kaggle_run_experiment.py`, `scripts/aggregate_experiment.py`.
- Not run: variants beyond the six registered candidates; WorldModel (deliberately unwired, see STATE.yaml).
