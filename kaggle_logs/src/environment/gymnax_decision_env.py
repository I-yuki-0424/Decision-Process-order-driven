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
    max_steps: int = 100
    num_actions: int = 2000
    num_macro_clusters: int = 50
    num_fine_actions: int = 40
    action_feat_dim: int = 32
    num_costs: int = 4
    num_resources: int = 8
    target_dim: int = 8
    history_len: int = 128
    goal_tolerance: float = 0.20  # Progress rate >= 80% achieves goal target
    simplify_stationary: bool = True  # Enforce stationary deterministic dynamics for high SPS


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
        k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)

        resource_levels = jax.random.uniform(k1, (self.params.num_resources,), minval=10.0, maxval=30.0)
        delta_target = jax.random.uniform(k2, (self.params.num_resources,), minval=20.0, maxval=40.0)
        target_state = resource_levels + delta_target
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

        # Generate action features & delta resources designed to cover ALL resource dimensions
        act_features = jax.random.normal(k3, (self.params.num_actions, self.params.action_feat_dim))
        
        # Base cost variation per action
        base_step_cost = delta_target[:self.params.num_costs] / (self.params.max_steps * 0.6)
        act_costs = jnp.repeat(base_step_cost[None, :], self.params.num_actions, axis=0)
        cost_variation = jax.random.uniform(k4, (self.params.num_actions, self.params.num_costs), minval=0.5, maxval=1.5)
        act_costs = act_costs * cost_variation

        # Generate explicit semantic resource effect matrix W_res (num_actions, num_resources)
        base_step_resource = delta_target / (self.params.max_steps * 0.6)
        resource_effects = jnp.repeat(base_step_resource[None, :], self.params.num_actions, axis=0)
        res_var = jax.random.uniform(k5, (self.params.num_actions, self.params.num_resources), minval=0.5, maxval=1.5)
        resource_effects = resource_effects * res_var

        # Abstraction scales: 40 for macro clusters, 1 for fine actions
        macro_scales = jnp.repeat(jnp.array([40.0]), self.params.num_macro_clusters)
        fine_scales = jnp.repeat(jnp.array([1.0]), self.params.num_fine_actions)
        abstraction_scales = jnp.tile(macro_scales[:self.params.num_actions // self.params.num_fine_actions], self.params.num_fine_actions)
        if abstraction_scales.shape[0] < self.params.num_actions:
            abstraction_scales = jnp.pad(abstraction_scales, (0, self.params.num_actions - abstraction_scales.shape[0]), constant_values=1.0)

        actions_data = ActionsData(
            features=act_features,
            costs=act_costs,
            preconditions=jnp.zeros((self.params.num_actions, 2), dtype=jnp.int32),
            valid_mask=jnp.ones((self.params.num_actions,), dtype=jnp.bool_),
            abstraction_scales=abstraction_scales,
            resource_effects=resource_effects,
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
        # 1. Exact Resource transition based on selected action using resource_effects
        action_cost = actions_data.costs[action_idx]
        if actions_data.resource_effects is not None:
            delta_resource = actions_data.resource_effects[action_idx]
        else:
            delta_resource = jnp.tile(action_cost, (self.params.num_resources // self.params.num_costs + 1,))[:self.params.num_resources]
        
        # 1b. Re-introduce Non-Stationary Dynamics (stochastic enemy spawns & meter decay) when False
        if not self.params.simplify_stationary:
            k_enemy, k_decay = jax.random.split(rng_key)
            enemy_attack_prob = jax.random.uniform(k_enemy) < 0.15  # 15% chance of enemy attack
            damage_impact = jnp.where(enemy_attack_prob, 2.5, 0.0)
            passive_decay = jax.random.uniform(k_decay, (self.params.num_resources,), minval=0.1, maxval=0.5)
            
            # Reduce health (index 0) and food (index 1) under disturbance
            disturbance = passive_decay.at[0].add(damage_impact)
            delta_resource = delta_resource - disturbance

        new_resource = env_state.resource_levels + delta_resource
        new_dist = jnp.linalg.norm(new_resource - env_state.target_state)

        # Reward: progress rate increase towards target
        prev_dist = jnp.linalg.norm(env_state.resource_levels - env_state.target_state)
        dist_reward = (prev_dist - new_dist) / (env_state.initial_distance + 1e-6)
        reward = dist_reward - 0.001 * jnp.sum(action_cost)

        # 2. Update step count and histories
        idx = jnp.minimum(env_state.step_count, self.params.history_len - 1)
        new_action_hist = env_state.action_history.at[idx].set(action_idx)
        new_rewards_hist = env_state.rewards_history.at[idx].set(reward)
        new_cost_hist = env_state.cost_history.at[idx].set(action_cost)

        new_step_count = env_state.step_count + 1

        # Check termination: progress rate >= 80% or max steps exceeded
        progress_rate = jnp.clip(1.0 - (new_dist / (env_state.initial_distance + 1e-6)), 0.0, 1.0)
        goal_reached = progress_rate >= 0.80
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
            "progress_rate": progress_rate,
            "distance": new_dist,
        }

        return obs, new_env_state, reward, done, info
