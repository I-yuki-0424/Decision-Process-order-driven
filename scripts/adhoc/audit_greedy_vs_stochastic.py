import sys; sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np
from src.model.checkpoint import load_pytree_checkpoint
from src.pipeline.chunk_ppo import ChunkPPO, ChunkPPOConfig

def first_episode_returns(alg, st, key, greedy, max_windows=40):
    """Unbiased per-policy return: FIRST episode of every env, run until (almost) all envs have finished."""
    cfg = alg.cfg
    k1, k2 = jax.random.split(key)
    ws = alg.reset_envs(k1)
    seen = np.zeros(cfg.num_envs, bool); ret = np.full(cfg.num_envs, np.nan); length = np.zeros(cfg.num_envs)
    t = 0
    for kk in jax.random.split(k2, max_windows):
        ws, _, traj, _ = alg.collect(st["pp"], st["wm"], st["stats"], ws, kk, n_cycles=cfg.eval_cycles, greedy=greedy)
        done = np.asarray(traj.out.done); fr = np.asarray(traj.out.finished_return)
        for c in range(done.shape[0]):
            m = done[c] & ~seen; ret[m] = fr[c][m]; length[m] = t + c + 1; seen |= done[c]
        t += done.shape[0]
        if seen.mean() > 0.999: break
    return np.nanmean(ret), np.nanmedian(length[seen]), seen.mean(), t

for name in ("mlp_s0_step15", "mlp_s1_step15"):
    payload = load_pytree_checkpoint(f"output/latency_chunking/craftax_calibration/{name[4:6]}/checkpoints/step_15.pkl")
    cfg = ChunkPPOConfig(**payload["config"]["cfg"])._replace(num_envs=256, eval_windows=4)
    alg = ChunkPPO(cfg); st = payload["params"]["st"]
    print(f"== {name}  (cfg num_envs overridden to 256 for CPU)", flush=True)
    for greedy in (True, False):
        r, ml, frac, T = first_episode_returns(alg, st, jax.random.PRNGKey(7), greedy)
        ev = alg.evaluate(st, jax.random.PRNGKey(7), greedy=greedy)
        print(f"  greedy={greedy!s:5}  UNBIASED first-episode return={r:.3f} (median len {ml:.0f}, finished {frac:.3f}, ticks {T})"
              f" | code's evaluate() (fresh start, 4x64 ticks) = {ev['mean_return']:.3f} over {ev['episodes']} eps", flush=True)
