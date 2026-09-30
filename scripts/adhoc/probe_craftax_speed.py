"""Probe: Craftax-Classic symbolic obs layout, and raw env step throughput (random policy) on the current JAX backend."""
import sys
import time

sys.path.insert(0, ".")
import jax
import jax.numpy as jnp

from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv

env = CraftaxClassicSymbolicEnv()
p = env.default_params
print("backend", jax.default_backend(), "max_timesteps", getattr(p, "max_timesteps", None))
obs, st = env.reset(jax.random.PRNGKey(0), p)
print("obs", obs.shape, "num_actions", env.action_space(p).n)
print("state fields", list(st.__dataclass_fields__))
print("achievements", st.achievements.shape, st.achievements.dtype)

for n_envs in (256, 1024, 4096):
    def step_fn(carry, _):
        s, k = carry
        k, ka, ks = jax.random.split(k, 3)
        a = jax.random.randint(ka, (n_envs,), 0, 17)
        o, s2, r, d, i = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(ks, n_envs), s, a, p)
        return (s2, k), r

    @jax.jit
    def run(key):
        ks = jax.random.split(key, n_envs)
        o, s = jax.vmap(env.reset, in_axes=(0, None))(ks, p)
        (s, _), r = jax.lax.scan(step_fn, (s, key), None, length=128)
        return r.sum()

    run(jax.random.PRNGKey(1)).block_until_ready()
    t = time.time()
    for i in range(3):
        run(jax.random.PRNGKey(i + 2)).block_until_ready()
    dt = (time.time() - t) / 3
    print(f"n_envs={n_envs}: {n_envs * 128 / dt:,.0f} env steps/s")
