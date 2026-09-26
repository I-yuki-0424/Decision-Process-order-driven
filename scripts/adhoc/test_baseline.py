import sys
sys.path.insert(0, ".")
import jax
from src.environment.gymnax_decision_env import DecisionProcessEnv
from src.model.baseline_model import init_baseline_parameters
from src.pipeline.benchmark import train_baseline

k = jax.random.PRNGKey(0)
env = DecisionProcessEnv()
params = init_baseline_parameters(k)
print("Testing train_baseline...")
p_trained = train_baseline(env, params, k, num_steps=5)
print("SUCCESS!")
