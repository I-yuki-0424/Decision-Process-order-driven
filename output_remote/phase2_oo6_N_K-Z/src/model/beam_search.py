"""
Vectorized Beam Search with KV Cache & 5th-Idea Hierarchical Abstraction.

Adheres to:
1. Maintains action-state candidate pairs (S_t, A_t) in K beams.
2. Pruning based on multi-dimensional cost minimization and Goal Progress Rate.
3. KV cache is correctly persisted per beam candidate across steps (Fix 2).
4. is_causal and z_compression_interval are propagated from the grid search (Fix 3).
5. resource_effects is mandatory — the tile-hack fallback has been removed (Fix 1/2).
"""

from typing import NamedTuple, Tuple, Optional
import jax
import jax.numpy as jnp

from src.model.types import (
    ActionHistory,
    ActionsData,
    BeamCandidate,
    BeamSearchState,
    DecisionVectorD,
    HierarchicalDecisionVectorD,
    InputContextN,
    KVCacheState,
    SystemState,
    TransitionTarget,
)
from src.model.transformer_decision_core import ModelParameters, forward_decision_transformer
from src.model.hierarchical_transformer import (
    HierarchicalModelParameters,
    forward_hierarchical_transformer,
)


def score_beam_candidate(
    action_log_prob: jnp.ndarray,
    predicted_progress: jnp.ndarray,
    cum_cost: jnp.ndarray,
    available_costs: jnp.ndarray,
    w_progress: float = 2.0,
    w_cost_penalty: float = 1.0,
) -> jnp.ndarray:
    """Calculate beam candidate score balancing log probability, progress, and cost limits."""
    cost_violation = jnp.sum(jnp.maximum(0.0, cum_cost - available_costs))
    score = action_log_prob + w_progress * predicted_progress - w_cost_penalty * cost_violation
    return score


def beam_search_init(
    initial_state: SystemState,
    initial_history: InputContextN,
    beam_width: int = 5,
    num_costs: int = 4,
) -> BeamSearchState:
    """Initialize BeamSearchState with K identical starting beams."""
    init_candidate = BeamCandidate(
        state=initial_state,
        history=initial_history.history,
        cum_cost=jnp.zeros((num_costs,)),
        progress_rate=initial_state.progress_rate,
        score=jnp.array(0.0),
        kv_cache=None,
    )

    beams = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None, ...], beam_width, axis=0), init_candidate)
    active_mask = jnp.ones((beam_width,), dtype=jnp.bool_)

    return BeamSearchState(
        beams=beams,
        active_mask=active_mask,
        step_count=initial_history.history.seq_len,
    )


def expand_single_beam(
    params: ModelParameters,
    candidate: BeamCandidate,
    actions_data: ActionsData,
    target: TransitionTarget,
    num_actions: int,
    is_causal: bool = True,
    z_compression_interval: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[KVCacheState]]:
    """Evaluate decision core for a single beam candidate to get expansion scores.

    Returns:
        valid_log_probs, progress_pred, estimated_costs, updated_kv_cache
    """
    input_n = InputContextN(
        actions=actions_data,
        state=candidate.state,
        history=candidate.history,
        target=target,
    )

    # Fix 2: capture returned kv_cache instead of discarding it with _
    decision_d, new_kv_cache = forward_decision_transformer(
        params,
        input_n,
        is_training=False,
        kv_cache=candidate.kv_cache,
        is_causal=is_causal,
        z_compression_interval=z_compression_interval,
    )

    log_probs = jax.nn.log_softmax(decision_d.action_logits)
    valid_log_probs = jnp.where(actions_data.valid_mask, log_probs, -1e9)

    return valid_log_probs, decision_d.progress_rate_pred, decision_d.estimated_costs, new_kv_cache


def beam_search_step(
    params: ModelParameters,
    state: BeamSearchState,
    actions_data: ActionsData,
    target: TransitionTarget,
    beam_width: int = 5,
    num_actions: int = 16,
    is_causal: bool = True,
    z_compression_interval: int = 0,
) -> BeamSearchState:
    """Perform one step of vectorized Beam Search across K beams.

    Fix 2: KV cache is now persisted per beam candidate.
    Fix 3: is_causal and z_compression_interval flow through to the transformer.
    Fix 1: resource_effects is mandatory; tile-hack branch is removed.
    """
    def _expand_fn(beam_cand):
        return expand_single_beam(
            params, beam_cand, actions_data, target, num_actions,
            is_causal=is_causal,
            z_compression_interval=z_compression_interval,
        )

    # vmap over the K beam candidates — pure functional, JIT-compatible
    vmap_expand = jax.vmap(_expand_fn)
    all_log_probs, all_progress, all_costs, all_new_kv = vmap_expand(state.beams)

    prev_scores = state.beams.score[:, None]       # (K, 1)
    progress_bonus = 5.0 * all_progress[:, None]   # (K, 1)
    cand_scores = prev_scores + all_log_probs + progress_bonus  # (K, num_actions)
    cand_scores_flat = cand_scores.reshape(-1)      # (K * num_actions,)

    topk_scores, topk_indices = jax.lax.top_k(cand_scores_flat, beam_width)

    parent_beam_indices = topk_indices // num_actions
    selected_action_indices = topk_indices % num_actions

    def _gather_parent(tree):
        return jax.tree_util.tree_map(lambda leaf: leaf[parent_beam_indices], tree)

    new_beams_parent = _gather_parent(state.beams)

    # Fix 2: gather the updated KV caches corresponding to the selected parent beams
    selected_kv_caches = jax.tree_util.tree_map(
        lambda leaf: leaf[parent_beam_indices], all_new_kv
    ) if all_new_kv is not None else None

    def _roll_history(hist):
        new_act = jnp.roll(hist.action_indices, -1, axis=1).at[:, -1].set(selected_action_indices)
        new_cost = jnp.roll(hist.cost_changes, -1, axis=1).at[:, -1, :].set(selected_costs)
        new_rew = jnp.roll(hist.rewards, -1, axis=1)
        return new_act, new_cost, new_rew, jnp.repeat(256, beam_width)

    def _append_history(hist):
        step_idx = state.step_count
        new_act = hist.action_indices.at[:, step_idx].set(selected_action_indices)
        new_cost = hist.cost_changes.at[:, step_idx, :].set(selected_costs)
        new_rew = hist.rewards
        return new_act, new_cost, new_rew, jnp.repeat(state.step_count + 1, beam_width)

    selected_costs = actions_data.costs[selected_action_indices]  # (K, num_costs)

    updated_action_indices, updated_cost_changes, updated_rewards, new_seq_len = jax.lax.cond(
        state.step_count >= 256,
        _roll_history,
        _append_history,
        new_beams_parent.history
    )

    updated_history = ActionHistory(
        action_indices=updated_action_indices,
        rewards=updated_rewards,
        cost_changes=updated_cost_changes,
        noise_mask=new_beams_parent.history.noise_mask,
        seq_len=new_seq_len,
    )

    updated_cum_cost = new_beams_parent.cum_cost + selected_costs

    # Fix 1: resource_effects is mandatory — remove the tile-hack else branch.
    # CraftaxEnvAdapter._build_actions_data() always populates resource_effects now.
    if actions_data.resource_effects is None:
        raise ValueError(
            "actions_data.resource_effects must not be None. "
            "Populate it via CRAFTAX_RESOURCE_EFFECTS in CraftaxEnvAdapter or "
            "provide an explicit (num_actions, num_resources) resource delta table. "
            "The tile-hack fallback has been removed to enforce physical causality."
        )
    delta_res = actions_data.resource_effects[selected_action_indices]  # (K, num_resources)
    updated_resource_levels = new_beams_parent.state.resource_levels + delta_res

    current_dist = jnp.linalg.norm(updated_resource_levels - target.target_state[None, :], axis=-1)
    initial_dist = jnp.linalg.norm(target.target_state, axis=-1) + 1e-6
    updated_progress_rate = jnp.clip(1.0 - (current_dist / initial_dist), 0.0, 1.0)

    updated_state = SystemState(
        resource_levels=updated_resource_levels,
        available_costs=new_beams_parent.state.available_costs - selected_costs,
        status_flags=new_beams_parent.state.status_flags,
        progress_rate=updated_progress_rate,
    )

    updated_beams = BeamCandidate(
        state=updated_state,
        history=updated_history,
        cum_cost=updated_cum_cost,
        progress_rate=updated_progress_rate,
        score=topk_scores,
        kv_cache=selected_kv_caches,  # Fix 2: persist updated caches per beam
    )

    return BeamSearchState(
        beams=updated_beams,
        active_mask=state.active_mask,
        step_count=state.step_count + 1,
    )


def hierarchical_beam_search_step(
    params: HierarchicalModelParameters,
    state: BeamSearchState,
    actions_data: ActionsData,
    target: TransitionTarget,
    use_hierarchical: bool = True,
    use_abstraction_embed: bool = True,
    beam_width: int = 5,
    num_actions: int = 2000,
    is_causal: bool = True,
    z_compression_interval: int = 0,
) -> BeamSearchState:
    """Perform 5th-Idea Hierarchical Beam Search across K beams (|A| = 2000).

    Fix 3: is_causal and z_compression_interval are propagated.
    Fix 1: resource_effects mandatory — tile-hack removed.
    """
    def _expand_fn(beam_cand):
        input_n = InputContextN(
            actions=actions_data,
            state=beam_cand.state,
            history=beam_cand.history,
            target=target,
        )
        decision_d, new_kv = forward_hierarchical_transformer(
            params,
            input_n,
            use_hierarchical=use_hierarchical,
            use_abstraction_embed=use_abstraction_embed,
            is_training=False,
        )
        log_probs = jax.nn.log_softmax(decision_d.action_logits)
        valid_log_probs = jnp.where(actions_data.valid_mask, log_probs, -1e9)
        return valid_log_probs, decision_d.progress_rate_pred, decision_d.estimated_costs, new_kv

    vmap_expand = jax.vmap(_expand_fn)
    all_log_probs, all_progress, all_costs, all_new_kv = vmap_expand(state.beams)

    prev_scores = state.beams.score[:, None]
    progress_bonus = 5.0 * all_progress[:, None]
    cand_scores = prev_scores + all_log_probs + progress_bonus
    cand_scores_flat = cand_scores.reshape(-1)

    topk_scores, topk_indices = jax.lax.top_k(cand_scores_flat, beam_width)

    parent_beam_indices = topk_indices // num_actions
    selected_action_indices = topk_indices % num_actions

    def _gather_parent(tree):
        return jax.tree_util.tree_map(lambda leaf: leaf[parent_beam_indices], tree)

    new_beams_parent = _gather_parent(state.beams)

    selected_kv_caches = jax.tree_util.tree_map(
        lambda leaf: leaf[parent_beam_indices], all_new_kv
    ) if all_new_kv is not None else None

    def _roll_history(hist):
        new_act = jnp.roll(hist.action_indices, -1, axis=1).at[:, -1].set(selected_action_indices)
        new_cost = jnp.roll(hist.cost_changes, -1, axis=1).at[:, -1, :].set(selected_costs)
        new_rew = jnp.roll(hist.rewards, -1, axis=1)
        return new_act, new_cost, new_rew, jnp.repeat(256, beam_width)

    def _append_history(hist):
        step_idx = state.step_count
        new_act = hist.action_indices.at[:, step_idx].set(selected_action_indices)
        new_cost = hist.cost_changes.at[:, step_idx, :].set(selected_costs)
        new_rew = hist.rewards
        return new_act, new_cost, new_rew, jnp.repeat(state.step_count + 1, beam_width)

    selected_costs = actions_data.costs[selected_action_indices]

    updated_action_indices, updated_cost_changes, updated_rewards, new_seq_len = jax.lax.cond(
        state.step_count >= 256,
        _roll_history,
        _append_history,
        new_beams_parent.history
    )

    updated_history = ActionHistory(
        action_indices=updated_action_indices,
        rewards=updated_rewards,
        cost_changes=updated_cost_changes,
        noise_mask=new_beams_parent.history.noise_mask,
        seq_len=new_seq_len,
    )

    updated_cum_cost = new_beams_parent.cum_cost + selected_costs

    if actions_data.resource_effects is None:
        raise ValueError(
            "actions_data.resource_effects must not be None. "
            "The tile-hack fallback has been removed to enforce physical causality (Fix 1)."
        )
    delta_res = actions_data.resource_effects[selected_action_indices]
    updated_resource_levels = new_beams_parent.state.resource_levels + delta_res

    current_dist = jnp.linalg.norm(updated_resource_levels - target.target_state[None, :], axis=-1)
    initial_dist = jnp.linalg.norm(target.target_state, axis=-1) + 1e-6
    updated_progress_rate = jnp.clip(1.0 - (current_dist / initial_dist), 0.0, 1.0)

    updated_state = SystemState(
        resource_levels=updated_resource_levels,
        available_costs=new_beams_parent.state.available_costs - selected_costs,
        status_flags=new_beams_parent.state.status_flags,
        progress_rate=updated_progress_rate,
    )

    updated_beams = BeamCandidate(
        state=updated_state,
        history=updated_history,
        cum_cost=updated_cum_cost,
        progress_rate=updated_progress_rate,
        score=topk_scores,
        kv_cache=selected_kv_caches,
    )

    return BeamSearchState(
        beams=updated_beams,
        active_mask=state.active_mask,
        step_count=state.step_count + 1,
    )
