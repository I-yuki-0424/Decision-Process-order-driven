"""
JAX & Optax Training Pipeline for 4th-Idea Decision Transformer.

Implements vectorized, JIT-compiled training step with Noise Injection regularization.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import optax

from src.model.transformer_decision_core import (
    ModelParameters,
    forward_decision_transformer,
)
from src.model.types import InputContextN


class TrainingMetrics(NamedTuple):
    """Training metrics collected per step."""
    total_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    cost_loss: jnp.ndarray
    progress_loss: jnp.ndarray
    validity_loss: jnp.ndarray


def compute_loss(
    params: ModelParameters,
    input_n: InputContextN,
    target_action: jnp.ndarray,
    target_cost: jnp.ndarray,
    target_progress: jnp.ndarray,
    rng_key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, TrainingMetrics]:
    """Compute multi-task loss with noise injection regularization."""
    # Forward pass with noise injection
    decision_d, _ = forward_decision_transformer(
        params,
        input_n,
        rng_key=rng_key,
        is_training=True,
    )

    # 1. Action Policy Cross-Entropy Loss
    policy_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=decision_d.action_logits[None, :],
        labels=target_action[None],
    )[0]

    # 2. Cost Estimation MSE Loss
    cost_loss = jnp.mean((decision_d.estimated_costs - target_cost) ** 2)

    # 3. Progress Rate MSE Loss
    progress_loss = (decision_d.progress_rate_pred - target_progress) ** 2

    # 4. Noise Validity Binary Cross-Entropy Loss
    target_validity = 1.0 - jnp.mean(input_n.history.noise_mask.astype(jnp.float32))
    validity_loss = - (target_validity * jnp.log(decision_d.validity_score + 1e-6) + 
                       (1.0 - target_validity) * jnp.log(1.0 - decision_d.validity_score + 1e-6))

    total_loss = policy_loss + 0.5 * cost_loss + 1.0 * progress_loss + 0.2 * validity_loss

    metrics = TrainingMetrics(
        total_loss=total_loss,
        policy_loss=policy_loss,
        cost_loss=cost_loss,
        progress_loss=progress_loss,
        validity_loss=validity_loss,
    )

    return total_loss, metrics


def train_step(
    params: ModelParameters,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    input_n: InputContextN,
    target_action: jnp.ndarray,
    target_cost: jnp.ndarray,
    target_progress: jnp.ndarray,
    rng_key: jax.random.PRNGKey,
) -> Tuple[ModelParameters, optax.OptState, TrainingMetrics]:
    """Execute a single JIT-compiled training step."""
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (_, metrics), grads = grad_fn(params, input_n, target_action, target_cost, target_progress, rng_key)

    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, metrics
