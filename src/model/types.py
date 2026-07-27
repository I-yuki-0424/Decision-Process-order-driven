"""
Types and PyTree Data Structures for 4th & 5th-Idea JAX Decision Process Architecture.

This module defines structured types for:
- Input Context N = {Actions A, State S, History H, Transition Target T}
- Decision Vector d = A(costs) + A(conditions) + S(Can use Costs) + H(reward) + H(Cost-change) + T(conditions(H))
- Hierarchical Decision Vector d (5th-Idea: Macro Cluster Head + Micro Fine Head)
- KV Cache state for Transformer autoregressive step generation
- Beam Search state for holding (S_t, A_t) candidate trajectories
- Working Memory state for compressed history representation
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
        abstraction_scales: Shape (num_actions,) - coverage scale N_enpass (e.g. fine actions encompassed)
    """
    features: jnp.ndarray
    costs: jnp.ndarray
    preconditions: jnp.ndarray
    valid_mask: jnp.ndarray
    abstraction_scales: Optional[jnp.ndarray] = None


class SystemState(NamedTuple):
    """System State representation (S)."""
    resource_levels: jnp.ndarray
    available_costs: jnp.ndarray
    status_flags: jnp.ndarray
    progress_rate: jnp.ndarray


class ActionHistory(NamedTuple):
    """Action History representation (H)."""
    action_indices: jnp.ndarray
    rewards: jnp.ndarray
    cost_changes: jnp.ndarray
    noise_mask: jnp.ndarray
    seq_len: jnp.ndarray


class TransitionTarget(NamedTuple):
    """Transition Target representation (T)."""
    target_state: jnp.ndarray
    conditions: jnp.ndarray
    deadline_step: jnp.ndarray


class InputContextN(NamedTuple):
    """Complete Input Context N = {A, S, H, T} for Decision Model."""
    actions: ActionsData
    state: SystemState
    history: ActionHistory
    target: TransitionTarget


class DecisionVectorD(NamedTuple):
    """Multi-dimensional Decision Vector d."""
    action_logits: jnp.ndarray
    estimated_costs: jnp.ndarray
    predicted_next_state: jnp.ndarray
    progress_rate_pred: jnp.ndarray
    validity_score: jnp.ndarray


class HierarchicalDecisionVectorD(NamedTuple):
    """5th-Idea Hierarchical Decision Vector d.

    Decomposes action selection into Macro Cluster Head (M clusters) + Micro Fine Action Head (K actions/cluster).
    """
    macro_logits: jnp.ndarray
    micro_logits: jnp.ndarray
    action_logits: jnp.ndarray
    estimated_costs: jnp.ndarray
    predicted_next_state: jnp.ndarray
    progress_rate_pred: jnp.ndarray
    validity_score: jnp.ndarray
    selected_macro_cluster: jnp.ndarray
    q_values: Optional[jnp.ndarray] = None


class WorkingMemoryState(NamedTuple):
    """5th-Idea Compressed Working Memory State for long sequence processing.

    Attributes:
        compressed_memory: Shape (memory_slots, d_model) - compressed latent history tokens
        last_compressed_step: Shape () - last step index where memory compression occurred
    """
    compressed_memory: jnp.ndarray
    last_compressed_step: jnp.ndarray


class KVCacheLayer(NamedTuple):
    """Key-Value Cache for a single Transformer Attention Layer."""
    cached_keys: jnp.ndarray
    cached_values: jnp.ndarray
    current_len: jnp.ndarray


class KVCacheState(NamedTuple):
    """Complete Key-Value Cache across all Transformer layers."""
    layers: Tuple[KVCacheLayer, ...]


class BeamCandidate(NamedTuple):
    """Single Beam Candidate holding paired (S_t, A_t) in accordance with Bellman causality."""
    state: SystemState
    history: ActionHistory
    cum_cost: jnp.ndarray
    progress_rate: jnp.ndarray
    score: jnp.ndarray
    kv_cache: Optional[KVCacheState] = None


class BeamSearchState(NamedTuple):
    """Beam Search State holding K concurrent beam candidates."""
    beams: BeamCandidate
    active_mask: jnp.ndarray
    step_count: jnp.ndarray
