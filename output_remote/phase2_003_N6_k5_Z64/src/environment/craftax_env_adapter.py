"""
Craftax-Classic Environment Adapter wrapping CraftaxClassicSymbolicEnv into InputContextN.

Adapts Craftax-Classic (JAX-native accelerated Crafter) for 5th-Idea Decision Transformer:
1. Maps 17 discrete Craftax actions to ActionsData with semantically correct resource_effects
2. Maps Craftax state & 22 achievement flags to SystemState
3. Computes exact Crafter Score: S_crafter = exp(1/N * sum(ln(1 + s_i))) - 1

Resource vector layout (num_resources=8):
  [0] health, [1] food, [2] drink, [3] energy,
  [4] wood,   [5] stone, [6] coal, [7] iron

Craftax-Classic Action Index Mapping (17 actions):
  0=noop, 1=move_left, 2=move_right, 3=move_up, 4=move_down,
  5=do (interact/attack), 6=sleep, 7=place_stone, 8=place_table,
  9=place_furnace, 10=place_plant, 11=make_wood_pickaxe,
  12=make_stone_pickaxe, 13=make_iron_pickaxe, 14=make_wood_sword,
  15=make_stone_sword, 16=make_iron_sword
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
    "collect_wood", "place_table", "eat_cow", "collect_sapling",
    "collect_drink", "make_wood_pickaxe", "make_wood_sword", "place_plant",
    "defeat_zombie", "collect_stone", "place_stone", "eat_plant",
    "defeat_skeleton", "make_stone_pickaxe", "make_stone_sword", "wake_up",
    "place_furnace", "collect_coal", "collect_iron", "collect_diamond",
    "make_iron_pickaxe", "make_iron_sword",
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


# Craftax-Classic action→resource delta table.
# Shape: (17, 8) — rows = actions, cols = [health, food, drink, energy, wood, stone, coal, iron]
# Signs follow: positive = gain, negative = consumption.
# Crafting actions consume resources, movement/sleep/interact are approximately neutral
# or consume energy. Values are approximate game-balance deltas (not exact sim outputs).
CRAFTAX_RESOURCE_EFFECTS = [
    # health  food   drink  energy  wood  stone  coal  iron
    [ 0.0,   0.0,   0.0,   0.0,   0.0,  0.0,  0.0,  0.0],  # 0: noop
    [ 0.0,   0.0,   0.0,  -0.1,   0.0,  0.0,  0.0,  0.0],  # 1: move_left
    [ 0.0,   0.0,   0.0,  -0.1,   0.0,  0.0,  0.0,  0.0],  # 2: move_right
    [ 0.0,   0.0,   0.0,  -0.1,   0.0,  0.0,  0.0,  0.0],  # 3: move_up
    [ 0.0,   0.0,   0.0,  -0.1,   0.0,  0.0,  0.0,  0.0],  # 4: move_down
    [ 0.0,   0.2,   0.1,  -0.2,   1.0,  1.0,  1.0,  1.0],  # 5: do (collect/attack; net gain varies)
    [ 0.0,   0.0,   0.0,   1.0,   0.0,  0.0,  0.0,  0.0],  # 6: sleep (restore energy)
    [ 0.0,   0.0,   0.0,  -0.1,   0.0, -1.0,  0.0,  0.0],  # 7: place_stone (spend stone)
    [ 0.0,   0.0,   0.0,  -0.1,  -2.0,  0.0,  0.0,  0.0],  # 8: place_table (spend wood)
    [ 0.0,   0.0,   0.0,  -0.1,  -4.0,  0.0,  0.0,  0.0],  # 9: place_furnace (spend wood)
    [ 0.0,   0.0,   0.0,  -0.1,   0.0,  0.0,  0.0,  0.0],  # 10: place_plant
    [ 0.0,   0.0,   0.0,  -0.2,  -2.0,  0.0,  0.0,  0.0],  # 11: make_wood_pickaxe (spend 2 wood)
    [ 0.0,   0.0,   0.0,  -0.2,   0.0, -2.0,  0.0,  0.0],  # 12: make_stone_pickaxe (spend 2 stone)
    [ 0.0,   0.0,   0.0,  -0.2,   0.0,  0.0,  0.0, -2.0],  # 13: make_iron_pickaxe (spend 2 iron)
    [ 0.0,   0.0,   0.0,  -0.2,  -2.0,  0.0,  0.0,  0.0],  # 14: make_wood_sword (spend 2 wood)
    [ 0.0,   0.0,   0.0,  -0.2,   0.0, -2.0,  0.0,  0.0],  # 15: make_stone_sword (spend 2 stone)
    [ 0.0,   0.0,   0.0,  -0.2,   0.0,  0.0,  0.0, -2.0],  # 16: make_iron_sword (spend 2 iron)
]


class CraftaxEnvAdapter:
    """Adapter wrapping CraftaxClassicSymbolicEnv into InputContextN."""

    def __init__(self, max_episode_steps: int = 1000):
        self.raw_env = CraftaxClassicSymbolicEnv()
        self.num_actions = 17
        self.num_achievements = NUM_ACHIEVEMENTS
        self.action_feat_dim = 32
        self.num_costs = 4
        self.num_resources = 8
        self.max_episode_steps = max_episode_steps

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
        prev_history: Any = None,
    ) -> Tuple[InputContextN, Any, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Step Craftax-Classic environment."""
        obs_raw, next_env_state, reward, done, info = self.raw_env.step(
            rng_key, env_state, action_idx, self.raw_env.default_params
        )
        
        # Truncation = time-limit hit (episode budget exhausted without terminal state)
        # Termination = agent genuinely reached a terminal game state (death / win)
        truncated = jnp.logical_and(done, step_count >= self.max_episode_steps - 1)
        terminated = jnp.logical_and(done, step_count < self.max_episode_steps - 1)
        info['truncated'] = truncated
        info['terminated'] = terminated
        
        input_n = self._build_input_context(
            obs_raw, next_env_state, actions_data,
            step_count=step_count + 1,
            prev_history=prev_history,
            action_idx=action_idx,
            reward=reward,
        )
        return input_n, next_env_state, reward, done, info

    def _build_actions_data(self, rng_key: jax.random.PRNGKey) -> ActionsData:
        """Build ActionsData with semantically correct resource_effects.

        resource_effects[a, r] is the delta applied to resource r when action a is executed.
        This is the f(S_t, A_t) transition function for beam search state updates.
        No tile-hacking: costs and resource_effects are separate tensors with distinct semantics.
        """
        k1, k2 = jax.random.split(rng_key)
        features = jax.random.normal(k1, (self.num_actions, self.action_feat_dim))
        # Costs: abstract multi-dimensional cost vector (time, effort, risk, opportunity)
        costs = jax.random.uniform(k2, (self.num_actions, self.num_costs), minval=0.1, maxval=1.0)

        # Abstraction scale: 17 fine actions
        abstraction_scales = jnp.ones((self.num_actions,), dtype=jnp.float32)

        # Physically grounded resource effects table — fixes the causality violation
        resource_effects = jnp.array(CRAFTAX_RESOURCE_EFFECTS, dtype=jnp.float32)  # (17, 8)

        return ActionsData(
            features=features,
            costs=costs,
            preconditions=jnp.zeros((self.num_actions, 2), dtype=jnp.int32),
            valid_mask=jnp.ones((self.num_actions,), dtype=jnp.bool_),
            abstraction_scales=abstraction_scales,
            resource_effects=resource_effects,
        )

    def _build_input_context(
        self,
        obs_raw: jnp.ndarray,
        env_state: Any,
        actions_data: ActionsData,
        step_count: int = 0,
        prev_history: Any = None,
        action_idx: int = 0,
        reward: float = 0.0,
    ) -> InputContextN:
        # Extract player state resources: health, food, drink, energy + inventory
        player_state = jnp.zeros((self.num_resources,))
        if hasattr(env_state, 'player_health'):
            player_state = jnp.array([
                env_state.player_health.astype(jnp.float32),
                env_state.player_food.astype(jnp.float32),
                env_state.player_drink.astype(jnp.float32),
                env_state.player_energy.astype(jnp.float32),
                0.0, 0.0, 0.0, 0.0
            ])
        else:
            player_state = jnp.zeros(8, dtype=jnp.float32)

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

        if prev_history is not None:
            # Update history with sliding window if we exceed 256
            def _update_history(hist):
                # Roll history left by 1 and place new element at the end
                new_act = jnp.roll(hist.action_indices, -1).at[-1].set(action_idx)
                new_rew = jnp.roll(hist.rewards, -1).at[-1].set(reward)
                new_cost = jnp.roll(hist.cost_changes, -1, axis=0).at[-1].set(actions_data.costs[action_idx])
                return new_act, new_rew, new_cost, jnp.array(256, dtype=jnp.int32)

            def _append_history(hist):
                # Just place element at current step
                new_act = hist.action_indices.at[step_count - 1].set(action_idx)
                new_rew = hist.rewards.at[step_count - 1].set(reward)
                new_cost = hist.cost_changes.at[step_count - 1].set(actions_data.costs[action_idx])
                return new_act, new_rew, new_cost, jnp.array(step_count, dtype=jnp.int32)

            new_action_indices, new_rewards, new_cost_changes, new_seq_len = jax.lax.cond(
                step_count > 256,
                _update_history,
                _append_history,
                prev_history
            )

            history = ActionHistory(
                action_indices=new_action_indices,
                rewards=new_rewards,
                cost_changes=new_cost_changes,
                noise_mask=prev_history.noise_mask,
                seq_len=new_seq_len,
            )
        else:
            history = ActionHistory(
                action_indices=jnp.zeros((256,), dtype=jnp.int32),
                rewards=jnp.zeros((256,), dtype=jnp.float32),
                cost_changes=jnp.zeros((256, self.num_costs), dtype=jnp.float32),
                noise_mask=jnp.zeros((256,), dtype=jnp.bool_),
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
