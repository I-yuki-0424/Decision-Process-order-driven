"""
Vectorized Beam Search with KV Cache & Bellman Causality Preservation.

Adheres to 4th-Idea Section 4:
1. Maintains action-state candidate pairs (S_t, A_t) in K beams.
2. Pruning based on multi-dimensional cost minimization and Goal Progress Rate.
3. KV Cache reuse across decision steps.
4. Vectorized top-K selection using JAX vmap.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from src.model.types import (
    ActionHistory,
    ActionsData,
    BeamCandidate,
    BeamSearchState,
    DecisionVectorD,
    InputContextN,
    SystemState,
    TransitionTarget,
)
from src.model.transformer_decision_core import ModelParameters, forward_decision_transformer


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
        step_count=jnp.array(0, dtype=jnp.int32),
    )


def expand_single_beam(
    params: ModelParameters,
    candidate: BeamCandidate,
    actions_data: ActionsData,
    target: TransitionTarget,
    num_actions: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate decision core for a single beam candidate to get expansion scores."""
    input_n = InputContextN(
        actions=actions_data,
        state=candidate.state,
        history=candidate.history,
        target=target,
    )
    
    decision_d, _ = forward_decision_transformer(
        params,
        input_n,
        is_training=False,
        kv_cache=candidate.kv_cache,
    )
    
    log_probs = jax.nn.log_softmax(decision_d.action_logits)
    valid_log_probs = jnp.where(actions_data.valid_mask, log_probs, -1e9)
    
    return valid_log_probs, decision_d.progress_rate_pred, decision_d.estimated_costs


def beam_search_step(
    params: ModelParameters,
    state: BeamSearchState,
    actions_data: ActionsData,
    target: TransitionTarget,
    beam_width: int = 5,
    num_actions: int = 16,
) -> BeamSearchState:
    """Perform one step of vectorized Beam Search across K beams.

    Expands K beams x A actions -> K*A candidates, updates (S_t, A_t) pairs, then prunes top K by score.
    """
    # 1. Vectorized expansion across all K active beams using vmap
    def _expand_fn(beam_cand):
        return expand_single_beam(params, beam_cand, actions_data, target, num_actions)

    vmap_expand = jax.vmap(_expand_fn)
    all_log_probs, all_progress, all_costs = vmap_expand(state.beams)
    # Shapes:
    # all_log_probs: (K, num_actions)
    # all_progress: (K,)
    # all_costs: (K, num_costs)

    # 2. Compute total candidate scores for all K * num_actions branches
    prev_scores = state.beams.score[:, None]  # (K, 1)
    progress_bonus = 5.0 * all_progress[:, None]  # (K, 1)
    cand_scores = prev_scores + all_log_probs + progress_bonus  # (K, num_actions)
    cand_scores_flat = cand_scores.reshape(-1)  # (K * num_actions,)

    # 3. Select Top-K beam indices
    topk_scores, topk_indices = jax.lax.top_k(cand_scores_flat, beam_width)

    parent_beam_indices = topk_indices // num_actions
    selected_action_indices = topk_indices % num_actions

    # 4. Gather selected parent candidates
    def _gather_parent(tree):
        return jax.tree_util.tree_map(lambda leaf: leaf[parent_beam_indices], tree)

    new_beams_parent = _gather_parent(state.beams)

    # 5. Update history and state for selected Top-K beams (Bellman Causality Pair S_t, A_t)
    step_idx = jnp.minimum(state.step_count, new_beams_parent.history.action_indices.shape[1] - 1)
    
    # Update action indices history at current step
    updated_action_indices = new_beams_parent.history.action_indices.at[:, step_idx].set(selected_action_indices)
    
    # Update cost changes history at current step
    selected_costs = actions_data.costs[selected_action_indices]  # (K, num_costs)
    updated_cost_changes = new_beams_parent.history.cost_changes.at[:, step_idx, :].set(selected_costs)

    updated_history = ActionHistory(
        action_indices=updated_action_indices,
        rewards=new_beams_parent.history.rewards,
        cost_changes=updated_cost_changes,
        noise_mask=new_beams_parent.history.noise_mask,
        seq_len=jnp.repeat(state.step_count + 1, beam_width),
    )

    # Update cumulative cost
    updated_cum_cost = new_beams_parent.cum_cost + selected_costs

    # Update state resource levels based on selected action (Bellman causality)
    num_res = new_beams_parent.state.resource_levels.shape[1]
    delta_res = selected_costs[:, :num_res] if selected_costs.shape[1] >= num_res else jnp.pad(selected_costs, ((0, 0), (0, num_res - selected_costs.shape[1])))
    updated_resource_levels = new_beams_parent.state.resource_levels + delta_res

    # Calculate updated progress rate towards target
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
        kv_cache=new_beams_parent.kv_cache,
    )

    return BeamSearchState(
        beams=updated_beams,
        active_mask=state.active_mask,
        step_count=state.step_count + 1,
    )
