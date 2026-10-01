"""ChunkPPO (src/pipeline/chunk_ppo.py, unmodified) evaluated with the Phase-1 achievement metrics.

ChunkPPO's own evaluate() reports env return. Here a subclass overrides `evaluate` (called by ChunkPPO.run at the final update) to
(a) capture the final training state and (b) score the FIRST episode of >= 256 fresh envs with masked_achievements(), exactly like
src/pipeline/epa_harness.py. Training itself is the untouched ChunkPPO code path (asymmetric MLP critic, chunk-level PPO).

With k = 1 and delta = 0 there is no chunking and no latency: the actor is a Transformer over [state, history] tokens and the
learner is PPO (see STATE TASK-20260930-017, objection 1). k > 1 executes the whole chunk open-loop (Phase 3 territory).
Limitation for k > 1: achievements unlocked inside the final cycle of an episode are not visible (the wrapper resets in-cycle).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from src.environment.craftax_env_adapter import NUM_ACHIEVEMENTS, masked_achievements
from src.model.epa_policies import n_params
from src.pipeline.chunk_ppo import ChunkPPO, ChunkPPOConfig, craftax_calibration_config
from src.pipeline.epa_harness import EPISODE_LIMIT, summarize_achievements


class EpaChunkPPO(ChunkPPO):
    def __init__(self, cfg: ChunkPPOConfig, eval_envs: int = 256):
        super().__init__(cfg)
        self.eval_envs, self.final_state, self.final_eval = eval_envs, None, None
        self._eval_cycles_fn = jax.jit(self._eval_cycles, static_argnames=("n_cycles",))

    def _eval_cycles(self, pp, wm, stats, ws, ach, alive, length, key, n_cycles):
        step = jax.vmap(partial(self._env_cycle, greedy=False), in_axes=(None, None, None, 0, 0))

        def body(c, kk):
            ws, ach, alive, length = c
            ws2, tr = step(pp, wm, stats, ws, jax.random.split(kk, self.eval_envs))
            d = tr.out.done
            ach = jnp.maximum(ach, masked_achievements(ws2.core.achievements, d[:, None], alive[:, None]))
            length = length + alive.astype(jnp.int32) * self.cfg.k
            return (ws2, ach, alive & ~d, length), None

        c, _ = jax.lax.scan(body, (ws, ach, alive, length), jax.random.split(key, n_cycles))
        return c

    def evaluate(self, st, key, greedy: bool = False):
        if greedy:   # the final-update hook also asks for the greedy return; not part of this protocol
            return dict(mean_return=float("nan"), episodes=0, terminal_rate=float("nan"), censored_frac=float("nan"),
                        mean_return_incl_censored=float("nan"))
        self.final_state = st
        k0, k1 = jax.random.split(jax.random.fold_in(key, 0xE7A1))
        n = self.eval_envs
        ws = jax.vmap(self.env.reset)(jax.random.split(k0, n))
        ach = jnp.zeros((n, NUM_ACHIEVEMENTS), jnp.float32)
        alive, length = jnp.ones((n,), jnp.bool_), jnp.zeros((n,), jnp.int32)
        chunk_cycles = max(250 // self.cfg.k, 1)
        steps = 0
        while steps < EPISODE_LIMIT + 1 and bool(alive.any()):
            k1, sub = jax.random.split(k1)
            ws, ach, alive, length = self._eval_cycles_fn(st["pp"], st["wm"], st["stats"], ws, ach, alive, length, sub,
                                                           n_cycles=chunk_cycles)
            steps += chunk_cycles * self.cfg.k
        out = summarize_achievements(np.asarray(ach))
        out.update(eval_episodes=int(n), eval_censored=int(np.asarray(alive).sum()),
                   eval_mean_length=float(np.asarray(length).mean()))
        self.final_eval = out
        return dict(mean_return=out["reward_pct"], episodes=n, terminal_rate=float("nan"), censored_frac=0.0,
                    mean_return_incl_censored=out["reward_pct"])

    def param_counts(self):
        pp = self.final_state["pp"]
        return n_params(pp) + n_params(self.final_state["wm"]), n_params(pp["actor"])


def chunkppo_config(seed, k, hist, total_steps, num_envs, cycles, lr, ent=0.01, epochs=4, minibatches=4, **kw):
    """Craftax-Classic ChunkPPO config with the PPO settings shared by every Phase-1 arm; the actor is the
    autoregressive chunk Transformer (d=64, 2 layers). Critic: the ChunkPPO asymmetric MLP critic (3 x 512)."""
    return craftax_calibration_config(actor="transformer", k=k, delta=0, hist_len=hist, num_envs=num_envs,
                                      cycles_per_update=cycles, lr=lr, ent_coef=ent, update_epochs=epochs,
                                      num_minibatches=minibatches, total_env_ticks=total_steps, seed=seed,
                                      eval_every=10 ** 9, dist_coef=0.0, arm="ignore", **kw)
