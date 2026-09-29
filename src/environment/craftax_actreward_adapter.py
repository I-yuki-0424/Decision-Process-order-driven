"""CraftaxObsAdapter whose 78 'anticipation' slots carry a LEARNED 1-step action-reward prediction (first 17 slots), see
src/model/candidates/action_reward_model.py. Same slot layout as CraftaxFutureAdapter('oracle_act'), so policy and learner are unchanged;
only the source of the 17 numbers differs (learned from the observation vs. true simulator branching)."""
import jax.numpy as jnp

from src.environment.craftax_future_adapter import EXTRA, CraftaxFutureAdapter
from src.model.candidates import action_reward_model as arm


class CraftaxActRewardAdapter(CraftaxFutureAdapter):
    def __init__(self, max_episode_steps, ar_params):
        super().__init__(max_episode_steps, "zeros")
        self.source = "wm_act"
        self.ar_params = ar_params

    def _extra(self, obs_raw, env_state):
        pred = arm.predict(self.ar_params, obs_raw)
        return jnp.concatenate([pred.astype(jnp.float32), jnp.zeros((EXTRA - arm.N_ACT,))])
