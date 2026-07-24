"""
Gymnax-Compatible Decision Process Environment.

Implements a Markov Decision Process (MDP) process generation environment
compatible with JAX jit and vmap primitives.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from src.model.types import (
    ActionHistory,
    ActionsData,
    InputContextN,
    SystemState,
    TransitionTarget,
)


class EnvParams(NamedTuple):
    """Parameters for DecisionProcessEnv."""
    max_steps: int = 150
    num_actions: int = 16
    action_feat_dim: int = 32
    num_costs: int = 4
    num_resources: int = 8
    target_dim: int = 8
    history_len: int = 128
    goal_tolerance: float = 0.05


class EnvState(NamedTuple):
    """Internal environment state."""
    resource_levels: jnp.ndarray   # (num_resources,)
    available_costs: jnp.ndarray   # (num_costs,)
    target_state: jnp.ndarray      # (num_resources,)
    initial_distance: jnp.ndarray  # ()
    action_history: jnp.ndarray    # (history_len,)
    rewards_history: jnp.ndarray   # (history_len,)
    cost_history: jnp.ndarray      # (history_len, num_costs)
    step_count: jnp.ndarray        # ()
    done: jnp.ndarray              # ()


class DecisionProcessEnv:
    """Gymnax-style Decision Process Environment for vectorized MDP simulation."""

    def __init__(self, params: EnvParams = EnvParams()):
        self.params = params

    def _get_obs(self, env_state: EnvState, actions_data: ActionsData) -> InputContextN:
        """Construct InputContextN from EnvState."""
        current_dist = jnp.linalg.norm(env_state.resource_levels - env_state.target_state)
        progress_rate = jnp.clip(1.0 - (current_dist / (env_state.initial_distance + 1e-6)), 0.0, 1.0)

        sys_state = SystemState(
            resource_levels=env_state.resource_levels,
            available_costs=env_state.available_costs,
            status_flags=jnp.zeros((4,)),
            progress_rate=progress_rate,
        )

        history = ActionHistory(
            action_indices=env_state.action_history,
            rewards=env_state.rewards_history,
            cost_changes=env_state.cost_history,
            noise_mask=jnp.zeros((self.params.history_len,), dtype=jnp.bool_),
            seq_len=env_state.step_count,
        )

        target = TransitionTarget(
            target_state=env_state.target_state,
            conditions=jnp.zeros((4,)),
            deadline_step=jnp.array(self.params.max_steps),
        )

        return InputContextN(
            actions=actions_data,
            state=sys_state,
            history=history,
            target=target,
        )

    def reset(self, rng_key: jax.random.PRNGKey) -> Tuple[InputContextN, EnvState, ActionsData]:
        """Reset environment to initial state."""
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        resource_levels = jax.random.uniform(k1, (self.params.num_resources,), minval=10.0, maxval=100.0)
        target_state = resource_levels + jax.random.uniform(k2, (self.params.num_resources,), minval=20.0, maxval=80.0)
        initial_dist = jnp.linalg.norm(resource_levels - target_state)

        available_costs = jnp.array([1000.0, 500.0, 300.0, 200.0])

        env_state = EnvState(
            resource_levels=resource_levels,
            available_costs=available_costs,
            target_state=target_state,
            initial_distance=initial_dist,
            action_history=jnp.zeros((self.params.history_len,), dtype=jnp.int32),
            rewards_history=jnp.zeros((self.params.history_len,), dtype=jnp.float32),
            cost_history=jnp.zeros((self.params.history_len, self.params.num_costs), dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False, dtype=jnp.bool_),
        )

        # Generate random action features & costs
        act_features = jax.random.normal(k3, (self.params.num_actions, self.params.action_feat_dim))
        act_costs = jax.random.uniform(k4, (self.params.num_actions, self.params.num_costs), minval=1.0, maxval=20.0)
        actions_data = ActionsData(
            features=act_features,
            costs=act_costs,
            preconditions=jnp.zeros((self.params.num_actions, 2), dtype=jnp.int32),
            valid_mask=jnp.ones((self.params.num_actions,), dtype=jnp.bool_),
        )

        obs = self._get_obs(env_state, actions_data)
        return obs, env_state, actions_data

    def step(
        self,
        rng_key: jax.random.PRNGKey,
        env_state: EnvState,
        action_idx: int,
        actions_data: ActionsData,
    ) -> Tuple[InputContextN, EnvState, jnp.ndarray, jnp.ndarray, dict]:
        """Execute step given action index."""
        # 1. Resource transition based on selected action
        action_cost = actions_data.costs[action_idx]
        delta_resource = action_cost[:self.params.num_resources] if self.params.num_costs >= self.params.num_resources else jnp.pad(action_cost, (0, self.params.num_resources - self.params.num_costs))
        
        new_resource = env_state.resource_levels + delta_resource
        new_dist = jnp.linalg.norm(new_resource - env_state.target_state)

        # Reward: distance reduction towards target - cost penalty
        prev_dist = jnp.linalg.norm(env_state.resource_levels - env_state.target_state)
        dist_reward = (prev_dist - new_dist) / (env_state.initial_distance + 1e-6)
        reward = dist_reward - 0.01 * jnp.sum(action_cost)

        # 2. Update step count and histories
        idx = jnp.minimum(env_state.step_count, self.params.history_len - 1)
        new_action_hist = env_state.action_history.at[idx].set(action_idx)
        new_rewards_hist = env_state.rewards_history.at[idx].set(reward)
        new_cost_hist = env_state.cost_history.at[idx].set(action_cost)

        new_step_count = env_state.step_count + 1

        # Check termination: goal reached or max steps exceeded
        goal_reached = new_dist <= (self.params.goal_tolerance * env_state.initial_distance)
        max_steps_exceeded = new_step_count >= self.params.max_steps
        done = jnp.logical_or(goal_reached, max_steps_exceeded)

        new_env_state = EnvState(
            resource_levels=new_resource,
            available_costs=env_state.available_costs - action_cost,
            target_state=env_state.target_state,
            initial_distance=env_state.initial_distance,
            action_history=new_action_hist,
            rewards_history=new_rewards_hist,
            cost_history=new_cost_hist,
            step_count=new_step_count,
            done=done,
        )

        obs = self._get_obs(new_env_state, actions_data)
        info = {
            "goal_reached": goal_reached,
            "progress_rate": obs.state.progress_rate,
            "distance": new_dist,
        }

        return obs, new_env_state, reward, done, info
