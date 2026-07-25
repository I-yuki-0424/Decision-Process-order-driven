import sys
import os
import traceback

sys.path.insert(0, os.path.abspath("."))

import jax
import jax.numpy as jnp
from src.environment.gymnax_decision_env import DecisionProcessEnv, EnvParams
from src.model.transformer_decision_core import init_model_parameters, forward_decision_transformer
from src.model.beam_search import beam_search_init, beam_search_step

print("=== Testing Fixed Beam Search & Environment Progression ===")
k = jax.random.PRNGKey(42)
env = DecisionProcessEnv(params=EnvParams(max_steps=50, num_actions=16))
obs, env_state, actions_data = env.reset(k)
params = init_model_parameters(k)

beam_state = beam_search_init(obs.state, obs, beam_width=3)

progress_list = []
for step in range(30):
    step_key = jax.random.fold_in(k, step)
    beam_state = beam_search_step(params, beam_state, actions_data, obs.target, beam_width=3)
    
    # Extract action index of top beam at current step
    best_action_idx = int(beam_state.beams.history.action_indices[0, beam_state.step_count - 1])
    
    obs, env_state, reward, done, info = env.step(step_key, env_state, best_action_idx, actions_data)
    progress_list.append(float(obs.state.progress_rate))

print(f"Initial Progress Rate : {progress_list[0]:.4f}")
print(f"Midway Progress Rate  : {progress_list[15]:.4f}")
print(f"Final Progress Rate   : {progress_list[-1]:.4f}")

if progress_list[-1] > 0.0:
    print("SUCCESS: Progress rate is NON-ZERO and increasing cleanly!")
else:
    print("WARNING: Progress rate is still 0.0!")
