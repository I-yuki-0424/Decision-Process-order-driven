import sys
import os
import traceback
sys.path.insert(0, os.path.abspath("."))

import jax
from src.environment.gymnax_decision_env import DecisionProcessEnv
from src.model.transformer_decision_core import init_model_parameters
from src.pipeline.benchmark import evaluate_model_variant

print("Testing evaluate_model_variant for 4th-Idea...")
k = jax.random.PRNGKey(42)
env = DecisionProcessEnv()
params = init_model_parameters(k)
try:
    m = evaluate_model_variant(
        model_name="4th-Idea Test",
        params=params,
        env=env,
        rng_key=k,
        is_baseline=False,
        use_beam_search=True,
        beam_width=3,
        num_episodes=1,
    )
    print("SUCCESS! Result:", m.model_name, m.success_rate)
except Exception as e:
    traceback.print_exc()
