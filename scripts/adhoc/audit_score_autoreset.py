import sys; sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from src.environment.craftax_env_adapter import calculate_crafter_score, NUM_ACHIEVEMENTS
env = CraftaxClassicSymbolicEnv(); p = env.default_params
T, N = 600, 96
def roll(key):
    k0, k1 = jax.random.split(key)
    _, s = env.reset(k0, p)
    def body(c, t):
        s, ach_all, ach_first, alive, ach_last, ended = c
        k = jax.random.fold_in(k1, t); ka, ke = jax.random.split(k)
        a = jax.random.randint(ka, (), 0, 17)
        _, s2, r, d, _ = env.step(ke, s, a, p)
        a2 = s2.achievements.astype(jnp.float32)
        ach_all = jnp.maximum(ach_all, a2)                                  # candidate_experiment style
        ach_first = jnp.maximum(ach_first, a2 * alive)                      # correct: first life only
        alive2 = alive * (1 - d.astype(jnp.float32))
        return (s2, ach_all, ach_first, alive2, a2, jnp.maximum(ended, d.astype(jnp.float32))), (d, alive)
    z = jnp.zeros(NUM_ACHIEVEMENTS)
    (sf, ach_all, ach_first, alive, ach_last, ended), (ds, al) = jax.lax.scan(body, (s, z, z, jnp.array(1.0), z, jnp.array(0.0)), jnp.arange(T))
    life = al.sum()
    return ach_all, ach_first, ach_last, life, ds.sum(), ended
out = jax.jit(jax.vmap(roll))(jax.random.split(jax.random.PRNGKey(0), N))
ach_all, ach_first, ach_last, life, ndone, ended = [np.asarray(x) for x in out]
rates = lambda a: [float(a[:, i].mean() * 100) for i in range(NUM_ACHIEVEMENTS)]
print("mean first-life length", life.mean(), " frac died within T", ended.mean(), " mean #dones in T", ndone.mean())
print("score correct(first life)      ", calculate_crafter_score(rates(ach_first)), " unlocked/ep", ach_first.sum(1).mean())
print("score candidate_experiment-style", calculate_crafter_score(rates(ach_all)), " unlocked/ep", ach_all.sum(1).mean())
# craftax_benchmark.evaluate style: achievements read from the FINAL env_state at loop exit (post auto-reset if it died)
died = ended > 0
print("craftax_benchmark-style, eps that died: unlocked read from final state (should be ~0):", ach_last[died].sum(1).mean() if died.any() else None,
      "| true first-life unlocked for those eps:", ach_first[died].sum(1).mean() if died.any() else None)
# binary union across episodes (phase2_runner / grid_search style)
u = (ach_first.max(0) > 0).astype(float) * 100
print("phase2-style union-over-episodes score (rates in {0,100}):", calculate_crafter_score(list(u)), " k=", int((u > 0).sum()))
u2 = (ach_all.max(0) > 0).astype(float) * 100
print("  ... with post-reset:", calculate_crafter_score(list(u2)), "k=", int((u2 > 0).sum()))
for k in (1, 3, 5, 8):
    print("  fixed function of k: k=%d -> %.2f" % (k, 101 ** (k / 22) - 1))
