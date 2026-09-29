"""Minimal smoke test for grid_search_benchmark.make_jit_eval_episode / eval_episode
under real jax.jit, to confirm the TracerBoolConversionError / int()-on-tracer bug is
actually fixed (not just theorized). No synthetic data: real CraftaxEnvAdapter + real
model forward passes, just a tiny num_steps and a single (n, k, z, is_causal) config.
"""
import sys, os
sys.path.insert(0, ".")

import jax
import numpy as np

from src.environment.craftax_env_adapter import CraftaxEnvAdapter, NUM_ACHIEVEMENTS
from src.model.transformer_decision_core import init_model_parameters
from src.pipeline.grid_search_benchmark import make_jit_eval_episode, PHASE2_MILESTONES
from src.environment.craftax_env_adapter import calculate_crafter_score

print("JAX backend:", jax.default_backend())

adapter = CraftaxEnvAdapter()
rng = jax.random.PRNGKey(0)
k_init, k_eval = jax.random.split(rng)

params = init_model_parameters(
    k_init,
    num_layers=2,
    d_model=32,
    num_heads=2,
    num_actions=adapter.num_actions,
    action_feat_dim=adapter.action_feat_dim,
    num_costs=adapter.num_costs,
    num_resources=adapter.num_resources,
    target_dim=8,
)

eval_fn = make_jit_eval_episode(adapter, k_val=3, z_val=32, is_causal=True, num_steps=5)
jit_eval = jax.jit(eval_fn)

final_achievements, mean_progress, context_util, cum_cost = jit_eval(params, k_eval)
jax.effects_barrier()

final_achievements_np = np.asarray(final_achievements)
print("final_achievements shape:", final_achievements_np.shape, "dtype:", final_achievements_np.dtype)
print("final_achievements:", final_achievements_np)
print("mean_progress:", float(mean_progress))
print("context_util:", float(context_util))
print("cum_cost:", np.asarray(cum_cost))

crafter_score = calculate_crafter_score([float(final_achievements_np[i] * 100.0) for i in range(NUM_ACHIEVEMENTS)])
diamond_unlocked = bool(final_achievements_np[PHASE2_MILESTONES["collect_diamond"]] > 0)
print("crafter_score:", crafter_score)
print("diamond_unlocked:", diamond_unlocked)
print("SMOKE TEST PASSED")
