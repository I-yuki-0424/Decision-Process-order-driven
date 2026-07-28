"""
Hierarchical Macro/Micro Execution Pipeline for 5th-Idea Decision Transformer.

Implements Directive #3:
- Macro Steps (Transformer Core): ~100 high-level decision steps. Responsible for selecting macro state transitions.
- Micro Steps (MDP Primitive Engine): ~90,000 primitive step executions (~900 steps per macro decision) carrying out low-level deterministic trajectories.
- JAX-Accelerated using jax.lax.scan and jax.jit for maximum GPU/TPU Step-Per-Second (SPS) throughput.
"""

from typing import NamedTuple, Tuple, Dict, Any
import jax
import jax.numpy as jnp

from src.environment.gymnax_decision_env import DecisionProcessEnv, EnvParams, EnvState
from src.model.hierarchical_transformer import (
    HierarchicalModelParameters,
    forward_hierarchical_transformer,
)
from src.model.types import (
    ActionsData,
    InputContextN,
    SystemState,
    ActionHistory,
    TransitionTarget,
)


class HierarchicalConfig(NamedTuple):
    """Configuration for Hierarchical Macro/Micro Pipeline."""
    macro_steps: int = 100          # 100 high-level Macro Transformer steps
    micro_steps_per_macro: int = 900 # 900 primitive steps per Macro step (90,000 micro steps total)
    total_micro_steps: int = 90000
    target_verification_steps: int = 1000000  # 1M verification target across parallel runs


class HierarchicalStepResult(NamedTuple):
    """Result of a single Macro step containing Micro execution rollout summary."""
    macro_step_idx: jnp.ndarray
    selected_action: jnp.ndarray
    macro_reward: jnp.ndarray
    micro_steps_executed: jnp.ndarray
    final_progress_rate: jnp.ndarray
    done: jnp.ndarray


def run_micro_trajectory(
    env: DecisionProcessEnv,
    rng_key: jax.random.PRNGKey,
    env_state: EnvState,
    action_idx: jnp.ndarray,
    actions_data: ActionsData,
    num_micro_steps: int = 900,
) -> Tuple[EnvState, InputContextN, jnp.ndarray, jnp.ndarray]:
    """Execute low-level primitive MDP trajectory using jax.lax.scan.

    Carries out num_micro_steps primitive environment transitions for the chosen macro target.
    """
    def _micro_step_fn(carry, key):
        curr_state, total_reward, is_done = carry
        next_obs, next_state, r, done, _ = env.step(key, curr_state, action_idx, actions_data)
        new_done = jnp.logical_or(is_done, done)
        new_reward = total_reward + jnp.where(is_done, 0.0, r)
        return (next_state, new_reward, new_done), (next_obs, r, done)

    keys = jax.random.split(rng_key, num_micro_steps)
    (final_env_state, total_reward, done), (obs_seq, reward_seq, done_seq) = jax.lax.scan(
        _micro_step_fn,
        (env_state, jnp.array(0.0, dtype=jnp.float32), env_state.done),
        keys,
    )

    final_obs = env._get_obs(final_env_state, actions_data)
    return final_env_state, final_obs, total_reward, done


class HierarchicalExecutionEngine:
    """JAX-accelerated Hierarchical Macro/Micro Execution Engine."""

    def __init__(self, config: HierarchicalConfig = HierarchicalConfig()):
        self.config = config
        self.env = DecisionProcessEnv(
            params=EnvParams(
                max_steps=config.total_micro_steps,
                simplify_stationary=True,
            )
        )

    def run_macro_episode(
        self,
        params: HierarchicalModelParameters,
        rng_key: jax.random.PRNGKey,
        use_abstraction_embed: bool = True,
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray]:
        """Run complete hierarchical episode (100 Macro Steps, 90,000 Micro Steps).

        Returns:
            final_env_state, total_macro_reward, cumulative_micro_steps.
        """
        k_reset, k_episode = jax.random.split(rng_key)
        obs, env_state, actions_data = self.env.reset(k_reset)

        def _macro_step_fn(carry, step_idx):
            curr_obs, curr_env_state, k_curr, cum_micro_steps, is_done = carry
            k_model, k_micro, k_next = jax.random.split(k_curr, 3)

            # 1. Macro Step (Transformer decision)
            decision_d, _ = forward_hierarchical_transformer(
                params,
                curr_obs,
                use_hierarchical=True,
                use_abstraction_embed=use_abstraction_embed,
                is_training=False,
            )
            macro_action = jnp.argmax(decision_d.action_logits)

            # 2. Micro Step Trajectory (~900 primitive step executions via jax.lax.scan)
            next_env_state, next_obs, macro_reward, done = run_micro_trajectory(
                self.env,
                k_micro,
                curr_env_state,
                macro_action,
                actions_data,
                num_micro_steps=self.config.micro_steps_per_macro,
            )

            new_done = jnp.logical_or(is_done, done)
            new_cum_micro_steps = cum_micro_steps + self.config.micro_steps_per_macro

            result = HierarchicalStepResult(
                macro_step_idx=step_idx,
                selected_action=macro_action,
                macro_reward=macro_reward,
                micro_steps_executed=self.config.micro_steps_per_macro,
                final_progress_rate=next_obs.state.progress_rate,
                done=new_done,
            )

            return (next_obs, next_env_state, k_next, new_cum_micro_steps, new_done), result

        macro_indices = jnp.arange(self.config.macro_steps)
        init_carry = (obs, env_state, k_episode, jnp.array(0, dtype=jnp.int32), env_state.done)

        (final_obs, final_env_state, _, total_micro_steps, _), step_results = jax.lax.scan(
            _macro_step_fn,
            init_carry,
            macro_indices,
        )

        total_reward = jnp.sum(step_results.macro_reward)
        return final_env_state, total_reward, total_micro_steps
