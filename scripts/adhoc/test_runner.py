import sys
import os
import traceback

sys.path.insert(0, os.path.abspath("."))

from src.environment.gymnax_decision_env import DecisionProcessEnv, EnvParams
from src.model.baseline_model import init_baseline_parameters, forward_baseline_transformer
from src.model.transformer_decision_core import init_model_parameters, forward_decision_transformer
from src.model.beam_search import beam_search_init, beam_search_step
from src.pipeline.benchmark import evaluate_model_variant, train_baseline
import jax
import jax.numpy as jnp

print("Step 1: Init PRNG & Env")
k = jax.random.PRNGKey(2026)
k_init, k_train, k_eval = jax.random.split(k, 3)
env = DecisionProcessEnv(params=EnvParams(max_steps=20, num_actions=16))

print("Step 2: Init parameters")
full_params = init_model_parameters(k_init, num_layers=2, d_model=256, num_heads=4)
baseline_params = init_baseline_parameters(k_init, num_layers=2, d_model=256, num_heads=4)

print("Step 3: Train baseline")
trained_baseline = train_baseline(env, baseline_params, k_train, num_steps=5)
print("Baseline trained.")

print("Step 4: Evaluate 4th-Idea (Beam Search)...")
m1 = evaluate_model_variant(
    model_name="4th-Idea (Full Proposed)",
    params=full_params,
    env=env,
    rng_key=k_eval,
    is_baseline=False,
    use_beam_search=True,
    beam_width=3,
    num_episodes=2,
)
print("4th-Idea evaluated:", m1.model_name, m1.success_rate)

print("Step 5: Evaluate 3rd-Idea Baseline...")
m2 = evaluate_model_variant(
    model_name="3rd-Idea (Greedy Baseline)",
    params=trained_baseline,
    env=env,
    rng_key=k_eval,
    is_baseline=True,
    use_beam_search=False,
    num_episodes=2,
)
print("3rd-Idea evaluated:", m2.model_name, m2.success_rate)

print("Step 6: Evaluate Ablation...")
m3 = evaluate_model_variant(
    model_name="Ablation (Noise Inj Only)",
    params=full_params,
    env=env,
    rng_key=k_eval,
    is_baseline=False,
    use_beam_search=False,
    num_episodes=2,
)
print("Ablation evaluated:", m3.model_name, m3.success_rate)
print("ALL BENCHMARK SUITE STEPS SUCCESSFUL!")
