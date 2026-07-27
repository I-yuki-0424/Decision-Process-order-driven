"""
Off-Policy Learning Engine for 5th-Idea Hierarchical Decision Transformer.

Implements:
1. Offline Experience Replay Buffer D_off = {(s_t, a_t, r_t, s_{t+1}, done)}
2. Off-Policy Temporal Difference Q-Learning Loss with Target Network parameter updates
3. Iterative loss convergence & sample efficiency tracking
"""

from typing import Dict, List, NamedTuple, Tuple, Any
import jax
import jax.numpy as jnp
import optax

from src.environment.gymnax_decision_env import DecisionProcessEnv, EnvParams
from src.model.hierarchical_transformer import (
    HierarchicalModelParameters,
    forward_hierarchical_transformer,
)
from src.model.types import InputContextN, ActionsData, SystemState, ActionHistory, TransitionTarget


class Transition(NamedTuple):
    """Offline transition tuple."""
    obs: InputContextN
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: InputContextN
    done: jnp.ndarray


def collect_offline_experience(
    env: DecisionProcessEnv,
    rng_key: jax.random.PRNGKey,
    num_episodes: int = 10,
    steps_per_ep: int = 50,
) -> List[Transition]:
    """Collect offline experience replay dataset D_off using sub-optimal behavior policy."""
    dataset = []
    keys = jax.random.split(rng_key, num_episodes)
    num_res = env.params.num_resources
    num_costs = env.params.num_costs

    for ep in range(num_episodes):
        obs, env_state, actions_data = env.reset(keys[ep])
        ep_keys = jax.random.split(keys[ep], steps_per_ep)

        for step in range(steps_per_ep):
            step_key = ep_keys[step]
            k_act, k_env = jax.random.split(step_key)

            # Sub-optimal behavior policy: 50% random action, 50% greedy step
            p_rand = jax.random.uniform(k_act)
            if float(p_rand) < 0.50:
                act = int(jax.random.randint(k_act, (), 0, env.params.num_actions))
            else:
                delta_r = jax.vmap(lambda c: jnp.tile(c, (num_res // num_costs + 1,))[:num_res])(actions_data.costs)
                next_r = obs.state.resource_levels[None, :] + delta_r
                dists = jnp.linalg.norm(next_r - obs.target.target_state[None, :], axis=-1)
                act = int(jnp.argmin(dists))

            next_obs, env_state, reward, done, _ = env.step(k_env, env_state, act, actions_data)

            transition = Transition(
                obs=obs,
                action=jnp.array(act, dtype=jnp.int32),
                reward=reward,
                next_obs=next_obs,
                done=done,
            )
            dataset.append(transition)
            obs = next_obs
            if done:
                break

    return dataset


def train_off_policy_hierarchical_model(
    env: DecisionProcessEnv,
    params: HierarchicalModelParameters,
    dataset: List[Transition],
    rng_key: jax.random.PRNGKey,
    use_abstraction_embed: bool = True,
    num_train_steps: int = 100,
    gamma: float = 0.95,
) -> Tuple[HierarchicalModelParameters, List[float]]:
    """Train Hierarchical Decision Model using Off-Policy Q-learning loss over offline dataset."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3),
    )
    opt_state = optimizer.init(params)
    curr_params = params
    curr_opt_state = opt_state
    target_params = params

    loss_history = []
    num_samples = len(dataset)
    keys = jax.random.split(rng_key, num_train_steps)

    def off_policy_loss_fn(p, target_p, trans: Transition):
        # 1. Forward pass on current state
        decision_curr, _ = forward_hierarchical_transformer(
            p,
            trans.obs,
            use_hierarchical=True,
            use_abstraction_embed=use_abstraction_embed,
            is_training=True,
        )
        q_curr = decision_curr.q_values[trans.action]

        # 2. Forward pass on next state using target network
        decision_next, _ = forward_hierarchical_transformer(
            target_p,
            trans.next_obs,
            use_hierarchical=True,
            use_abstraction_embed=use_abstraction_embed,
            is_training=False,
        )
        max_q_next = jnp.max(decision_next.q_values)
        target_val = trans.reward + (1.0 - trans.done.astype(jnp.float32)) * gamma * max_q_next

        # 3. TD error + policy cross-entropy loss
        td_loss = jnp.square(q_curr - target_val)
        policy_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=decision_curr.action_logits[None, :],
            labels=trans.action[None],
        )[0]

        total_loss = td_loss + 0.5 * policy_loss
        return total_loss

    grad_fn = jax.value_and_grad(off_policy_loss_fn, argnums=0)

    for step in range(num_train_steps):
        # Sample random transition from offline dataset
        idx = int(jax.random.randint(keys[step], (), 0, num_samples))
        sample_trans = dataset[idx]

        loss_val, grads = grad_fn(curr_params, target_params, sample_trans)
        updates, curr_opt_state = optimizer.update(grads, curr_opt_state, curr_params)
        curr_params = optax.apply_updates(curr_params, updates)
        loss_history.append(float(loss_val))

        # Polyak target network update every 10 steps
        if (step + 1) % 10 == 0:
            target_params = jax.tree_util.tree_map(
                lambda p_curr, p_targ: 0.90 * p_targ + 0.10 * p_curr,
                curr_params,
                target_params,
            )

    return curr_params, loss_history
