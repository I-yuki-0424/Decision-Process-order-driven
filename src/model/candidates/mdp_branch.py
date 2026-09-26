"""MDPBranch (docs/DPOD.ipynb cell 19): the "ideal structure" pipeline with
stage 4 = Bellman/MDP top-k action search over candidates that survive the
physical filter (shared_stages.physical_key_filter /
shared_stages.candidate_survival_mask).

Critical-review fix applied (see docs/core/STATE.yaml, task
TASK-20260925-001): the notebook scalarizes each candidate via
`TensorOps.embeddings_to_scalar_id` (mean-pool) before MDPSolver -- the same
defect noted for Variant5_1Bace/Variant5_4. This port uses
`shared_stages.discretize_embeddings_to_ids` instead, and additionally masks
the final action_logits by `candidate_survives` directly, so the physical
filter is enforced end to end regardless of what the (still-placeholder)
discretized MDP search produces.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from src.model.types import DecisionVectorD, InputContextN
from src.model.channel_encoder import (
    EncoderParameters,
    encode_channel_independent,
    init_channel_encoder_params,
)
from src.model.candidates.shared_stages import (
    DecisionHeadParameters,
    SharedStageParameters,
    apply_decision_heads,
    build_mdp_transition_array,
    candidate_survival_mask,
    compress_step_to_history,
    compute_physical_feasibility,
    discretize_embeddings_to_ids,
    init_decision_head_params,
    init_shared_stage_params,
    mdp_topk_to_action_logits,
    physical_key_filter,
    predict_next_goal,
    self_attention_now_goal_actions,
    solve_bellman_topk,
    split_encoded_tokens,
)


class MDPBranchParameters(NamedTuple):
    encoder_params: EncoderParameters
    shared: SharedStageParameters
    heads: DecisionHeadParameters


def init_mdp_branch_parameters(
    rng_key,
    d_model: int = 512,
    num_actions: int = 16,
    action_feat_dim: int = 32,
    num_costs: int = 4,
    num_resources: int = 8,
    target_dim: int = 8,
) -> MDPBranchParameters:
    keys = jax.random.split(rng_key, 3)
    encoder_params = init_channel_encoder_params(
        keys[0], d_model=d_model, num_actions=num_actions, action_feat_dim=action_feat_dim,
        num_costs=num_costs, num_resources=num_resources, target_dim=target_dim,
    )
    shared = init_shared_stage_params(keys[1], d_model=d_model, num_actions=num_actions)
    heads = init_decision_head_params(keys[2], d_model=d_model, num_costs=num_costs, num_resources=num_resources)
    return MDPBranchParameters(encoder_params=encoder_params, shared=shared, heads=heads)


def forward_mdp_branch(
    params: MDPBranchParameters,
    input_n: InputContextN,
    history_buffer: jnp.ndarray,
    num_heads: int = 8,
    k_mdp: int = 3,
    p_coe: float = 0.1,
    r_coe: float = 0.5,
    gamma_coe: float = 0.9, stable_mdp: bool = False,
) -> Tuple[DecisionVectorD, jnp.ndarray]:
    """history_buffer: (n_hist, d_model) running compressed-history buffer
    (stage 6 output of a previous step, or an initial zero buffer).
    Returns (decision_d, updated_history_buffer)."""
    tokens = encode_channel_independent(params.encoder_params, input_n)
    num_actions = input_n.actions.features.shape[0]
    state_tok, target_tok, actions_tok, _ = split_encoded_tokens(tokens, num_actions)

    physically_possible = compute_physical_feasibility(input_n.state, input_n.actions)  # (2, num_actions)

    stage1 = self_attention_now_goal_actions(state_tok, target_tok, actions_tok, num_heads)
    query_repr = stage1[:2]
    actions_repr = stage1[2:]

    stage2 = physical_key_filter(
        query_repr, actions_repr, params.shared.w_k, params.shared.b_k, num_heads, physically_possible,
    )
    next_goal = predict_next_goal(stage2, params.shared.w_goal, params.shared.b_goal)

    candidate_survives = candidate_survival_mask(physically_possible)
    filtered_candidates = jnp.where(candidate_survives[:, None], actions_repr, 0.0)

    state_rep = jnp.repeat(state_tok, num_actions, axis=0)
    next_goal_rep = jnp.repeat(next_goal, num_actions, axis=0)
    state_ids = discretize_embeddings_to_ids(state_rep, params.shared.w_discretize, params.shared.b_discretize)
    next_ids = discretize_embeddings_to_ids(next_goal_rep, params.shared.w_discretize, params.shared.b_discretize)
    action_ids = discretize_embeddings_to_ids(filtered_candidates, params.shared.w_discretize, params.shared.b_discretize)

    mdp_array = build_mdp_transition_array(state_ids, next_ids, action_ids, p_coe, r_coe, gamma_coe, stable=stable_mdp)
    top_actions, top_values = solve_bellman_topk(mdp_array, k_mdp, n=num_actions, stable=stable_mdp)
    action_logits = mdp_topk_to_action_logits(top_actions, top_values, num_actions)
    action_logits = jnp.where(candidate_survives, action_logits, -1e9)

    updated_history = compress_step_to_history(filtered_candidates, params.shared.w_z, params.shared.b_z, history_buffer)

    pooled = jnp.mean(jnp.concatenate([query_repr, actions_repr], axis=0), axis=0)
    costs, next_state, progress, validity = apply_decision_heads(params.heads, pooled)

    decision_d = DecisionVectorD(
        action_logits=action_logits,
        estimated_costs=costs,
        predicted_next_state=next_state,
        progress_rate_pred=progress,
        validity_score=validity,
    )
    return decision_d, updated_history
