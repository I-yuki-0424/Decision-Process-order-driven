"""
Craftax-Classic Environment Adapter wrapping CraftaxClassicSymbolicEnv into InputContextN.

Adapts Craftax-Classic (JAX-native accelerated Crafter) for 5th-Idea Decision Transformer:
1. Maps 17 discrete Craftax actions to ActionsData
2. Maps Craftax state & 22 achievement flags to SystemState
3. Computes exact Crafter Score: S_crafter = exp(1/N * sum(ln(1 + s_i))) - 1
"""

from typing import Dict, List, NamedTuple, Tuple, Any
import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from src.model.types import (
    ActionHistory,
    ActionsData,
    InputContextN,
    SystemState,
    TransitionTarget,
)


ACHIEVEMENT_NAMES = [
    "collect_wood", "place_table", "eat_plant", "collect_stone",
    "place_furnace", "collect_coal", "collect_iron", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
    "defeat_zombie", "defeat_skeleton", "eat_cow", "collect_drink",
    "make_torch", "place_stone", "wake_up", "survive_day",
]

NUM_ACHIEVEMENTS = len(ACHIEVEMENT_NAMES)  # 22


def calculate_crafter_score(achievement_unlock_rates: List[float]) -> float:
    """Calculate Crafter Score S_crafter = exp(1/N * sum(ln(1 + s_i))) - 1.

    Args:
        achievement_unlock_rates: List of 22 percentage rates s_i in [0.0, 100.0]

    Returns:
        Crafter score scalar in [0.0, 100.0]
    """
    log_sum = np.sum([np.log(1.0 + max(0.0, float(s))) for s in achievement_unlock_rates])
    score = np.exp(log_sum / float(NUM_ACHIEVEMENTS)) - 1.0
    return float(score)


class CraftaxEnvAdapter:
    """Adapter wrapping CraftaxClassicSymbolicEnv into InputContextN."""

    def __init__(self):
        self.raw_env = CraftaxClassicSymbolicEnv()
        self.num_actions = 17
        self.num_achievements = NUM_ACHIEVEMENTS
        self.action_feat_dim = 32
        self.num_costs = 4
        self.num_resources = 8

    def reset(self, rng_key: jax.random.PRNGKey) -> Tuple[InputContextN, Any, ActionsData]:
        """Reset Craftax-Classic environment."""
        obs_raw, env_state = self.raw_env.reset(rng_key, self.raw_env.default_params)
        actions_data = self._build_actions_data(rng_key)
        input_n = self._build_input_context(obs_raw, env_state, actions_data, step_count=0)
        return input_n, env_state, actions_data

    def step(
        self,
        rng_key: jax.random.PRNGKey,
        env_state: Any,
        action_idx: int,
        actions_data: ActionsData,
        step_count: int = 0,
    ) -> Tuple[InputContextN, Any, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Step Craftax-Classic environment."""
        obs_raw, next_env_state, reward, done, info = self.raw_env.step(
            rng_key, env_state, action_idx, self.raw_env.default_params
        )
        input_n = self._build_input_context(obs_raw, next_env_state, actions_data, step_count=step_count + 1)
        return input_n, next_env_state, reward, done, info

    def _build_actions_data(self, rng_key: jax.random.PRNGKey) -> ActionsData:
        k1, k2 = jax.random.split(rng_key)
        features = jax.random.normal(k1, (self.num_actions, self.action_feat_dim))
        costs = jax.random.uniform(k2, (self.num_actions, self.num_costs), minval=0.5, maxval=2.0)
        
        # Abstraction scale: 17 fine actions
        abstraction_scales = jnp.ones((self.num_actions,), dtype=jnp.float32)

        return ActionsData(
            features=features,
            costs=costs,
            preconditions=jnp.zeros((self.num_actions, 2), dtype=jnp.int32),
            valid_mask=jnp.ones((self.num_actions,), dtype=jnp.bool_),
            abstraction_scales=abstraction_scales,
        )

    def _build_input_context(
        self,
        obs_raw: jnp.ndarray,
        env_state: Any,
        actions_data: ActionsData,
        step_count: int = 0,
    ) -> InputContextN:
        # Extract player state resources: health, food, drink, energy + inventory
        player_state = jnp.zeros((self.num_resources,))
        if hasattr(env_state, 'player_health'):
            player_state = jnp.array([
                float(env_state.player_health),
                float(env_state.player_food),
                float(env_state.player_drink),
                float(env_state.player_energy),
                0.0, 0.0, 0.0, 0.0
            ])

        # Achievement unlock progress (count unlocked / 22)
        achievements_unlocked = jnp.zeros((NUM_ACHIEVEMENTS,))
        if hasattr(env_state, 'achievements'):
            achievements_unlocked = env_state.achievements.astype(jnp.float32)
        
        progress_rate = jnp.mean(achievements_unlocked)

        sys_state = SystemState(
            resource_levels=player_state,
            available_costs=jnp.array([100.0, 50.0, 30.0, 20.0]),
            status_flags=achievements_unlocked[:4],
            progress_rate=progress_rate,
        )

        history = ActionHistory(
            action_indices=jnp.zeros((128,), dtype=jnp.int32),
            rewards=jnp.zeros((128,), dtype=jnp.float32),
            cost_changes=jnp.zeros((128, self.num_costs), dtype=jnp.float32),
            noise_mask=jnp.zeros((128,), dtype=jnp.bool_),
            seq_len=jnp.array(step_count, dtype=jnp.int32),
        )

        target = TransitionTarget(
            target_state=jnp.array([10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
            conditions=jnp.zeros((4,)),
            deadline_step=jnp.array(300),
        )

        return InputContextN(
            actions=actions_data,
            state=sys_state,
            history=history,
            target=target,
        )
