"""TransformerBranch (docs/DPOD.ipynb cell 20): the "ideal structure" pipeline
with stage 4 = attention-score ranking over the physically-filtered
candidates, offered as the alternative to MDPBranch's Bellman/MDP search.

Critical-review note (see docs/core/STATE.yaml, task TASK-20260925-001): this
is the strongest of the ported candidates -- unlike MDPBranch (and
Variant5_1Bace/Variant5_4), it needs no discretization placeholder at all.
Candidate exclusion is enforced via `candidate_survives` applied directly to
real attention/dot-product scores (excluded candidates get -inf and can never
be top-k'd), and the returned action_logits index directly into the actual
`actions` candidate array -- no lossy scalar-id detour.
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
    candidate_survival_mask,
    compress_step_to_history,
    compute_physical_feasibility,
    init_decision_head_params,
    init_shared_stage_params,
    physical_key_filter,
    predict_next_goal,
    self_attention_now_goal_actions,
    split_encoded_tokens,
)


class TransformerBranchParameters(NamedTuple):
    encoder_params: EncoderParameters
    shared: SharedStageParameters  # w_discretize/b_discretize unused by this branch
    heads: DecisionHeadParameters


def init_transformer_branch_parameters(
    rng_key: jax.random.PRNGKey,
    d_model: int = 512,
    num_actions: int = 16,
    action_feat_dim: int = 32,
    num_costs: int = 4,
    num_resources: int = 8,
    target_dim: int = 8,
) -> TransformerBranchParameters:
    keys = jax.random.split(rng_key, 3)
    encoder_params = init_channel_encoder_params(
        keys[0], d_model=d_model, num_actions=num_actions, action_feat_dim=action_feat_dim,
        num_costs=num_costs, num_resources=num_resources, target_dim=target_dim,
    )
    shared = init_shared_stage_params(keys[1], d_model=d_model, num_actions=num_actions)
    heads = init_decision_head_params(keys[2], d_model=d_model, num_costs=num_costs, num_resources=num_resources)
    return TransformerBranchParameters(encoder_params=encoder_params, shared=shared, heads=heads)


def forward_transformer_branch(
    params: TransformerBranchParameters,
    input_n: InputContextN,
    history_buffer: jnp.ndarray,
    num_heads: int = 8,
) -> Tuple[DecisionVectorD, jnp.ndarray]:
    """history_buffer: (n_hist, d_model) running compressed-history buffer.
    Returns (decision_d, updated_history_buffer)."""
    tokens = encode_channel_independent(params.encoder_params, input_n)
    num_actions = input_n.actions.features.shape[0]
    state_tok, target_tok, actions_tok, _ = split_encoded_tokens(tokens, num_actions)
    d_model = tokens.shape[-1]

    physically_possible = compute_physical_feasibility(input_n.state, input_n.actions)  # (2, num_actions)

    stage1 = self_attention_now_goal_actions(state_tok, target_tok, actions_tok, num_heads)
    query_repr = stage1[:2]
    actions_repr = stage1[2:]

    stage2 = physical_key_filter(
        query_repr, actions_repr, params.shared.w_k, params.shared.b_k, num_heads, physically_possible,
    )
    next_goal = predict_next_goal(stage2, params.shared.w_goal, params.shared.b_goal)

    candidate_survives = candidate_survival_mask(physically_possible)

    # Stage 4: raw (unmasked-by-softmax) query-key dot-product scores over the
    # candidates, scaled -- this is what makes top-k ranking usable, since
    # dot_product_attention itself returns only a weighted output, not
    # per-candidate weights. candidate_survives is applied directly to these
    # scores, guaranteeing a physically disallowed candidate can never be
    # selected regardless of its embedding content.
    scores = jnp.matmul(next_goal, actions_repr.T)[0] / jnp.sqrt(d_model)
    action_logits = jnp.where(candidate_survives, scores, -jnp.inf)

    filtered_candidates = jnp.where(candidate_survives[:, None], actions_repr, 0.0)
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
