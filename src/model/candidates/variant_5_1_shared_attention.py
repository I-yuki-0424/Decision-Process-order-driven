"""Variant5_1Bace (docs/DPOD.ipynb cell 11): concatenate actions/target/state/
history into one sequence, run `num_l` layers of shared-weight self-attention
with a residual update, then hand the result to the Bellman/MDP solver.

Critical-review fix applied (see docs/core/STATE.yaml, task
TASK-20260925-001): the notebook builds its MDPSolver input array via
`jnp.column_stack` of raw d-dimensional embedding blocks, which `MDPSolver`
documents as needing 6 single scalar columns (S, S_1, A, P, R, G) -- a real
shape/semantics bug. This port replaces that column_stack with
`shared_stages.discretize_embeddings_to_ids` (the same placeholder
discretization fix used by MDPBranch) feeding
`shared_stages.build_mdp_transition_array`, so the MDP solver receives the
scalar ids it actually expects instead of collapsed or mis-shaped embeddings.
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
    discretize_embeddings_to_ids,
    init_decision_head_params,
    init_shared_stage_params,
    mdp_topk_to_action_logits,
    scaled_dot_product_attention,
    solve_bellman_topk,
    split_encoded_tokens,
)


class Variant51LayerParameters(NamedTuple):
    """K/V projections, weight-tied across the `num_l` scan steps -- matches
    the notebook, which passes a single `k_s`/`v_s` pair reused by every
    layer of `jax.lax.scan`, rather than per-layer weights."""
    w_k: jnp.ndarray  # (d_model, d_model)
    w_v: jnp.ndarray  # (d_model, d_model)


class Variant51Parameters(NamedTuple):
    encoder_params: EncoderParameters
    layer: Variant51LayerParameters
    shared: SharedStageParameters
    heads: DecisionHeadParameters


def _glorot(key: jax.random.PRNGKey, in_dim: int, out_dim: int) -> jnp.ndarray:
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    return jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)


def init_variant_5_1_parameters(
    rng_key: jax.random.PRNGKey,
    d_model: int = 512,
    num_actions: int = 16,
    action_feat_dim: int = 32,
    num_costs: int = 4,
    num_resources: int = 8,
    target_dim: int = 8,
) -> Variant51Parameters:
    keys = jax.random.split(rng_key, 4)
    encoder_params = init_channel_encoder_params(
        keys[0], d_model=d_model, num_actions=num_actions, action_feat_dim=action_feat_dim,
        num_costs=num_costs, num_resources=num_resources, target_dim=target_dim,
    )
    l_keys = jax.random.split(keys[1], 2)
    layer = Variant51LayerParameters(
        w_k=_glorot(l_keys[0], d_model, d_model), w_v=_glorot(l_keys[1], d_model, d_model),
    )
    shared = init_shared_stage_params(keys[2], d_model=d_model, num_actions=num_actions)
    heads = init_decision_head_params(keys[3], d_model=d_model, num_costs=num_costs, num_resources=num_resources)
    return Variant51Parameters(encoder_params=encoder_params, layer=layer, shared=shared, heads=heads)


def forward_variant_5_1(
    params: Variant51Parameters,
    input_n: InputContextN,
    num_l: int = 2,
    num_heads: int = 8,
    k_mdp: int = 3,
    p_coe: float = 0.1,
    r_coe: float = 0.5,
    gamma_coe: float = 0.9,
) -> DecisionVectorD:
    """num_l, num_heads, k_mdp must be concrete Python ints when jitted
    (jax.lax.scan length and jax.lax.top_k both require static values)."""
    tokens = encode_channel_independent(params.encoder_params, input_n)
    num_actions = input_n.actions.features.shape[0]

    k_s = jnp.matmul(tokens, params.layer.w_k)
    v_s = jnp.matmul(tokens, params.layer.w_v)

    def layer_fn(carry, _):
        attn = scaled_dot_product_attention(carry, k_s, v_s, num_heads)
        return carry + attn, None

    y, _ = jax.lax.scan(layer_fn, tokens, xs=None, length=num_l)

    state_tok, target_tok, actions_tok, _ = split_encoded_tokens(y, num_actions)

    state_ids = discretize_embeddings_to_ids(
        jnp.repeat(state_tok, num_actions, axis=0), params.shared.w_discretize, params.shared.b_discretize,
    )
    goal_ids = discretize_embeddings_to_ids(
        jnp.repeat(target_tok, num_actions, axis=0), params.shared.w_discretize, params.shared.b_discretize,
    )
    action_ids = discretize_embeddings_to_ids(actions_tok, params.shared.w_discretize, params.shared.b_discretize)

    mdp_array = build_mdp_transition_array(state_ids, goal_ids, action_ids, p_coe, r_coe, gamma_coe)
    top_actions, top_values = solve_bellman_topk(mdp_array, k_mdp, n=num_actions)
    action_logits = mdp_topk_to_action_logits(top_actions, top_values, num_actions)

    pooled = jnp.mean(y, axis=0)
    costs, next_state, progress, validity = apply_decision_heads(params.heads, pooled)

    return DecisionVectorD(
        action_logits=action_logits,
        estimated_costs=costs,
        predicted_next_state=next_state,
        progress_rate_pred=progress,
        validity_score=validity,
    )
