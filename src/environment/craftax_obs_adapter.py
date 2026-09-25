"""Craftax adapter that actually exposes the observation to the model (Idea-6 study, adapter "v2").

Why this exists: CraftaxEnvAdapter builds InputContextN from only 4 vitals + 4 achievement flags and discards
the symbolic observation (7x9 local map, inventory, mobs). ActionsData.features/costs are also re-drawn
randomly every episode, so action identity is not stable. A policy built on that input cannot learn Craftax
regardless of architecture (see docs/experiments/2026-09-25_idea6_worldmodel/03_findings.md).

Changes vs CraftaxEnvAdapter (documented, minimal):
  * SystemState.resource_levels = [vitals(4), inventory(12), light, sleeping, block-type counts in view(17),
    mob counts in view(4)] (+ optional `feature_fn(F)` outputs, e.g. world-model anticipation). Fixed summary
    statistics of the real observation, no hand-coded task knowledge.
  * ActionsData.features are fixed per-action vectors (one-hot + the existing resource_effects row), costs const.
"""
import jax
import jax.numpy as jnp

from src.environment.craftax_env_adapter import CRAFTAX_RESOURCE_EFFECTS, CraftaxEnvAdapter
from src.model.types import ActionsData

N_BLOCK, N_MOB, VIEW = 17, 4, 63
MAP_END = VIEW * (N_BLOCK + N_MOB)  # 1323
BASE_DIM = 4 + 12 + 2 + N_BLOCK + N_MOB  # 39


def obs_to_features(obs):
    """(1345,) symbolic obs -> (BASE_DIM,) fixed summary. Layout: map(1323) inv(12) intrinsics(4) dir(4) light sleep."""
    m = obs[:MAP_END].reshape(VIEW, N_BLOCK + N_MOB)
    blocks = m[:, :N_BLOCK].sum(0) / VIEW
    mobs = m[:, N_BLOCK:].sum(0)
    inv = obs[MAP_END:MAP_END + 12]
    intr = obs[MAP_END + 12:MAP_END + 16]
    ls = obs[MAP_END + 20:MAP_END + 22]
    return jnp.concatenate([intr, inv, ls, blocks, mobs]).astype(jnp.float32)


class CraftaxObsAdapter(CraftaxEnvAdapter):
    def __init__(self, max_episode_steps: int = 1000, feature_fn=None, extra_dim: int = 0):
        super().__init__(max_episode_steps)
        self.feature_fn = feature_fn
        self.extra_dim = extra_dim
        self.num_resources = BASE_DIM + extra_dim

    def _build_actions_data(self, rng_key):
        n = self.num_actions
        eff = jnp.array(CRAFTAX_RESOURCE_EFFECTS, dtype=jnp.float32)
        feats = jnp.concatenate([jnp.eye(n), eff, jnp.zeros((n, self.action_feat_dim - n - eff.shape[1]))], axis=1)
        return ActionsData(
            features=feats, costs=jnp.full((n, self.num_costs), 0.5),
            preconditions=jnp.zeros((n, 2), dtype=jnp.int32), valid_mask=jnp.ones((n,), dtype=jnp.bool_),
            abstraction_scales=jnp.ones((n,)), resource_effects=eff,
        )

    def _build_input_context(self, obs_raw, env_state, actions_data, **kw):
        ctx = super()._build_input_context(obs_raw, env_state, actions_data, **kw)
        f = obs_to_features(obs_raw)
        extra = self.feature_fn(f) if self.feature_fn is not None else jnp.zeros((self.extra_dim,))
        res = jnp.concatenate([f, extra.astype(jnp.float32)])
        return ctx._replace(state=ctx.state._replace(resource_levels=res))
