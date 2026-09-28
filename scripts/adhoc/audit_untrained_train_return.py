import sys; sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np
from src.pipeline.chunk_ppo import ChunkPPO, craftax_calibration_config, episode_summary
cfg = craftax_calibration_config(num_envs=128)
alg = ChunkPPO(cfg)
key = jax.random.PRNGKey(0)
st = alg.init(jax.random.PRNGKey(1))          # UNTRAINED policy, never updated
ws = alg.reset_envs(jax.random.PRNGKey(2))
print("== UNTRAINED policy, same 'train_return' protocol as ChunkPPO.run (persistent envs, 64-cycle windows)")
first_ret = np.full(cfg.num_envs, np.nan); seen = np.zeros(cfg.num_envs, bool)
for u in range(15):
    key, kc = jax.random.split(key)
    ws, _, traj, _ = alg.collect(st["pp"], st["wm"], st["stats"], ws, kc, n_cycles=cfg.cycles_per_update)
    ep = episode_summary(traj)
    done = np.asarray(traj.out.done); fr = np.asarray(traj.out.finished_return)
    for c in range(done.shape[0]):
        m = done[c] & ~seen; first_ret[m] = fr[c][m]; seen |= done[c]
    print(f"update {u+1:2d} train_return={ep['mean_return']:.3f} episodes={ep['episodes']}  (envs with a finished 1st episode so far: {seen.mean():.2f})")
print("mean return of FIRST episode of every env (unbiased for this policy, censored:", (~seen).mean(), "):", np.nanmean(first_ret))
for greedy in (False, True):
    for w in (1, 4):
        pass
    for w in (1, 4):
        c2 = alg.cfg._replace(eval_windows=w); a2 = ChunkPPO(c2)
        ev = a2.evaluate(st, jax.random.PRNGKey(5), greedy=greedy)
        print(f"evaluate(untrained, greedy={greedy}, eval_windows={w}): mean_return={ev['mean_return']:.3f} episodes={ev['episodes']}")
