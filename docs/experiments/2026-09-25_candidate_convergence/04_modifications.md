# Modifications and their documentation (all outside existing model files; `src/model/**` untouched)
1. New jitted driver `candidate_experiment.py` (existing `candidate_benchmark.py` untouched): whole episodes in `lax.scan`, vmapped over 8 envs; per-episode loss via `lax.map` + `jax.checkpoint` (first version OOM'd on 8 GB).
2. History buffer truncated to a static 16-row window (originals grow +1 row/step, un-jittable, O(T) cost per step). Changes stateful candidates' effective memory.
3. REINFORCE with sampling, baseline, entropy bonus (mode `reinforce`) vs legacy loss shape (`argmax_legacy`, still samples during rollout, uses raw return weights) — A/B.
4. `optax.apply_if_finite` guard (never triggered).
5. `reinforce_normlogits`: standardises MDP-search logits (masked -1e9 -> min finite - 10) — A/B for finding 3.
6. Baselines: uniform-random and untrained-init evaluated with the identical evaluator.
7. `docker` image built from unchanged `docker/Dockerfile`; runs use `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
