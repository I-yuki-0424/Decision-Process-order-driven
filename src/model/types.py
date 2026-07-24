"""
Types and PyTree Data Structures for 4th-Idea JAX Decision Process Architecture.

This module defines structured types for:
- Input Context N = {Actions A, State S, History H, Transition Target T}
- Decision Vector d = A(costs) + A(conditions) + S(Can use Costs) + H(reward) + H(Cost-change) + T(conditions(H))
- KV Cache state for Transformer autoregressive step generation
- Beam Search state for holding (S_t, A_t) candidate trajectories
"""

from typing import NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp


class ActionsData(NamedTuple):
    """Actions / Choices representation (A).

    Attributes:
        features: Shape (num_actions, num_action_features) - static/dynamic parameters of choices
        costs: Shape (num_actions, num_costs) - multi-dimensional cost associated with each action
        preconditions: Shape (num_actions, max_preconditions) - indices of required preceding actions
        valid_mask: Shape (num_actions,) - boolean mask indicating valid choices
    """
    features: jnp.ndarray
    costs: jnp.ndarray
    preconditions: jnp.ndarray
    valid_mask: jnp.ndarray


class SystemState(NamedTuple):
    """System State representation (S).

    Attributes:
        resource_levels: Shape (num_resources,) - current budget/resource capacities
        available_costs: Shape (num_costs,) - maximum available cost budget
        status_flags: Shape (num_status_flags,) - status flags of system
        progress_rate: Shape () - float scalar indicating global progress rate [0, 1]
    """
    resource_levels: jnp.ndarray
    available_costs: jnp.ndarray
    status_flags: jnp.ndarray
    progress_rate: jnp.ndarray


class ActionHistory(NamedTuple):
    """Action History representation (H).

    Attributes:
        action_indices: Shape (seq_len,) - sequence of selected action indices
        rewards: Shape (seq_len,) - sequence of observed scalar rewards
        cost_changes: Shape (seq_len, num_costs) - sequence of multi-dimensional cost deltas
        noise_mask: Shape (seq_len,) - boolean mask indicating injected noise (sub-optimal actions)
        seq_len: Shape () - actual filled sequence length
    """
    action_indices: jnp.ndarray
    rewards: jnp.ndarray
    cost_changes: jnp.ndarray
    noise_mask: jnp.ndarray
    seq_len: jnp.ndarray


class TransitionTarget(NamedTuple):
    """Transition Target representation (T).

    Attributes:
        target_state: Shape (num_resources,) - goal state configuration
        conditions: Shape (num_conditions,) - goal constraints on history and final state
        deadline_step: Shape () - maximum allowed step count for process completion
    """
    target_state: jnp.ndarray
    conditions: jnp.ndarray
    deadline_step: jnp.ndarray


class InputContextN(NamedTuple):
    """Complete Input Context N = {A, S, H, T} for 4th-Idea Decision Model.

    Adheres to 4th-Idea Channel Independence:
    Physical properties and scales (Time, Cost, State, History) are structurally
    segregated into distinct PyTree fields / channels.
    """
    actions: ActionsData
    state: SystemState
    history: ActionHistory
    target: TransitionTarget


class DecisionVectorD(NamedTuple):
    """Multi-dimensional Decision Vector d.

    d = A(Costs) + A(conditions) + S(Can use Costs) + H(reward) + H(Cost-change) + T(conditions(H))

    Attributes:
        action_logits: Shape (num_actions,) - logit probabilities over candidate actions
        estimated_costs: Shape (num_costs,) - predicted cost impact of selected decision
        predicted_next_state: Shape (num_resources,) - predicted state S_{t+1}
        progress_rate_pred: Shape () - predicted global goal progress rate [0.0, 1.0]
        validity_score: Shape () - predicted logical validity score (rejecting noise)
    """
    action_logits: jnp.ndarray
    estimated_costs: jnp.ndarray
    predicted_next_state: jnp.ndarray
    progress_rate_pred: jnp.ndarray
    validity_score: jnp.ndarray


class KVCacheLayer(NamedTuple):
    """Key-Value Cache for a single Transformer Attention Layer.

    Attributes:
        cached_keys: Shape (batch_size, num_heads, max_seq_len, head_dim)
        cached_values: Shape (batch_size, num_heads, max_seq_len, head_dim)
        current_len: Shape (batch_size,) - active cached length
    """
    cached_keys: jnp.ndarray
    cached_values: jnp.ndarray
    current_len: jnp.ndarray


class KVCacheState(NamedTuple):
    """Complete Key-Value Cache across all Transformer layers."""
    layers: Tuple[KVCacheLayer, ...]


class BeamCandidate(NamedTuple):
    """Single Beam Candidate holding paired (S_t, A_t) in accordance with Bellman causality.

    Attributes:
        state: SystemState - state S_t
        history: ActionHistory - trajectory history H_t
        cum_cost: Shape (num_costs,) - accumulated multi-dimensional cost
        progress_rate: Shape () - progress rate towards target T
        score: Shape () - total beam evaluation score (log prob + progress - cost penalty)
        kv_cache: KVCacheState - cached keys and values up to current step
    """
    state: SystemState
    history: ActionHistory
    cum_cost: jnp.ndarray
    progress_rate: jnp.ndarray
    score: jnp.ndarray
    kv_cache: Optional[KVCacheState] = None


class BeamSearchState(NamedTuple):
    """Beam Search State holding K concurrent beam candidates.

    Attributes:
        beams: PyTree of K BeamCandidates (batched over beam dimension K)
        active_mask: Shape (K,) - boolean mask indicating non-terminated beams
        step_count: Shape () - current decision step count
    """
    beams: BeamCandidate
    active_mask: jnp.ndarray
    step_count: jnp.ndarray
