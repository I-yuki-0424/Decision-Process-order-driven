"""CraftaxObsAdapter + 78 'anticipation' features about the noop future (horizons 2 and 8, standardised summary deltas).

feature_source:
  zeros  : constant zeros (base arm)
  oracle : delta of the summary after 2 / 8 simulator noop steps branched from the CURRENT env state, standardised with the
           same per-horizon delta std as the world model. ORACLE / UPPER BOUND by construction (uses the simulator at policy
           time; one sampled noise realisation, so it is even stronger than the conditional mean a WM can give).
  oracle_act : POSITIVE CONTROL, not a world model: true 1-step reward of every action (answer leakage by design; shows
           whether this policy learner can exploit decision-relevant lookahead at all)
  wm     : the same features predicted by a passive world model from the current raw observation (no simulator access).
"""
import jax
import jax.numpy as jnp

from src.environment.craftax_obs_adapter import CraftaxObsAdapter, obs_to_features
from src.model.candidates import passive_wm_full as pw

EXTRA = 2 * pw.F


class CraftaxFutureAdapter(CraftaxObsAdapter):
    def __init__(self, max_episode_steps, feature_source, wm_params=None, delta_sd=None):
        super().__init__(max_episode_steps, None, EXTRA)
        assert feature_source in ("zeros", "oracle", "oracle_act", "wm")
        self.source = feature_source
        self.wm_params = wm_params
        self.delta_sd = None if delta_sd is None else jnp.asarray(delta_sd)  # (n_h, F)

    def _extra(self, obs_raw, env_state):
        if self.source == "zeros":
            return jnp.zeros((EXTRA,))
        if self.source == "wm":
            return pw.policy_features(pw.predict_std_delta(self.wm_params, obs_raw))
        env, P = self.raw_env, self.raw_env.default_params
        if self.source == "oracle_act":  # POSITIVE CONTROL (pure answer leakage): true immediate reward of each of the 17 actions
            key = jax.random.fold_in(jax.random.PRNGKey(11), env_state.timestep)
            rew = jax.vmap(lambda a: env.step(key, env_state, a, P)[2])(jnp.arange(self.num_actions))
            return jnp.concatenate([rew.astype(jnp.float32), jnp.zeros((EXTRA - self.num_actions,))])
        f0 = obs_to_features(obs_raw)
        key = jax.random.fold_in(jax.random.PRNGKey(7), env_state.timestep)

        def body(c, i):
            s = c
            o, s2, _, _, _ = env.step(jax.random.fold_in(key, i), s, 0, P)
            return s2, obs_to_features(o)
        _, fs = jax.lax.scan(body, env_state, jnp.arange(pw.HORIZONS[-1]))
        d = jnp.stack([fs[h - 1] - f0 for h in pw.HORIZONS]) / self.delta_sd
        return pw.policy_features(d)

    def _build_input_context(self, obs_raw, env_state, actions_data, **kw):
        ctx = super(CraftaxObsAdapter, self)._build_input_context(obs_raw, env_state, actions_data, **kw)
        f = obs_to_features(obs_raw)
        res = jnp.concatenate([f, self._extra(obs_raw, env_state).astype(jnp.float32)])
        return ctx._replace(state=ctx.state._replace(resource_levels=res))
