import jax
import jax.numpy as jnp
from src.environment.craftax_env_adapter import CraftaxEnvAdapter

env = CraftaxEnvAdapter()

@jax.jit
def scan_fn(carry, _):
    rng, env_state, act_data = carry
    
    rng, step_key = jax.random.split(rng)
    # Just take action 0
    action = jnp.array(0, dtype=jnp.int32)
    next_o, next_e_st, reward, done, info = env.step(step_key, env_state, action, act_data)
    
    rng, reset_key = jax.random.split(rng)
    reset_o, reset_e_st, reset_a_data = env.reset(reset_key)
    
    next_e_st = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), reset_e_st, next_e_st)
    act_data = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), reset_a_data, act_data)
    
    return (rng, next_e_st, act_data), reward

rng = jax.random.PRNGKey(0)
rng, reset_key = jax.random.split(rng)
o, e_st, a_data = env.reset(reset_key)

carry = (rng, e_st, a_data)
carry, rewards = jax.lax.scan(scan_fn, carry, None, length=10)
print("Success!")
