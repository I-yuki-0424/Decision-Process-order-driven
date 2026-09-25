"""Variant5_4 (docs/DPOD.ipynb cell 14): single-shot variant -- one attention
pass from `state` over (actions, target, history), a physical-constraint
projection W_k, a dense compression predicting S', then the Bellman/MDP
search. No `num_l` loop, unlike the other three variants.

Critical-review fix applied (see docs/core/STATE.yaml, task
TASK-20260925-001): the notebook mean-pools each (num_actions, d) embedding
block to one scalar per row before building the MDP transition array (the
same `TensorOps.embeddings_to_scalar_id` defect flagged for Variant5_1Bace and
MDPBranch). This port uses `shared_stages.discretize_embeddings_to_ids`
instead.
"""

from typing import NamedTuple
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


def _sinusoidal_positional_encoding(seq_len: int, d_model: int) -> jnp.ndarray:
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))
    pe = jnp.zeros((seq_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe


class Variant54Parameters(NamedTuple):
    encoder_params: EncoderParameters
    shared: SharedStageParameters  # reuses w_k/b_k as the physical-constraint projection
    w_dense: jnp.ndarray  # (d_model, d_model) compresses to the predicted S'
    b_dense: jnp.ndarray  # (d_model,)
    heads: DecisionHeadParameters


def _glorot(key: jax.random.PRNGKey, in_dim: int, out_dim: int) -> jnp.ndarray:
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    return jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)


def init_variant_5_4_parameters(
    rng_key: jax.random.PRNGKey,
    d_model: int = 512,
    num_actions: int = 16,
    action_feat_dim: int = 32,
    num_costs: int = 4,
    num_resources: int = 8,
    target_dim: int = 8,
) -> Variant54Parameters:
    keys = jax.random.split(rng_key, 4)
    encoder_params = init_channel_encoder_params(
        keys[0], d_model=d_model, num_actions=num_actions, action_feat_dim=action_feat_dim,
        num_costs=num_costs, num_resources=num_resources, target_dim=target_dim,
    )
    shared = init_shared_stage_params(keys[1], d_model=d_model, num_actions=num_actions)
    heads = init_decision_head_params(keys[2], d_model=d_model, num_costs=num_costs, num_resources=num_resources)
    return Variant54Parameters(
        encoder_params=encoder_params, shared=shared,
        w_dense=_glorot(keys[3], d_model, d_model), b_dense=jnp.zeros((d_model,)),
        heads=heads,
    )


def forward_variant_5_4(
    params: Variant54Parameters,
    input_n: InputContextN,
    num_heads: int = 8,
    k_mdp: int = 3,
    p_coe: float = 0.1,
    r_coe: float = 0.5,
    gamma_coe: float = 0.9,
) -> DecisionVectorD:
    tokens = encode_channel_independent(params.encoder_params, input_n)
    num_actions = input_n.actions.features.shape[0]
    state_tok, target_tok, actions_tok, hist_tok = split_encoded_tokens(tokens, num_actions)
    d_model = tokens.shape[-1]

    hist_pe = hist_tok + _sinusoidal_positional_encoding(hist_tok.shape[0], d_model)
    kv = jnp.concatenate([actions_tok, target_tok, hist_pe], axis=0)
    y_attn = scaled_dot_product_attention(state_tok, kv, kv, num_heads)

    # Physical-constraint projection W_K (an ordinary learned projection; the
    # "constraint" is applied downstream via masking, not by restricting W_K's
    # values -- same convention as MDPBranch/TransformerBranch's physical_key_filter).
    constrained = jnp.matmul(y_attn, params.shared.w_k) + params.shared.b_k
    s_prime = jnp.matmul(constrained, params.w_dense) + params.b_dense  # (1, d_model)

    state_ids = discretize_embeddings_to_ids(
        jnp.repeat(state_tok, num_actions, axis=0), params.shared.w_discretize, params.shared.b_discretize,
    )
    sprime_ids = discretize_embeddings_to_ids(
        jnp.repeat(s_prime, num_actions, axis=0), params.shared.w_discretize, params.shared.b_discretize,
    )
    action_ids = discretize_embeddings_to_ids(actions_tok, params.shared.w_discretize, params.shared.b_discretize)

    mdp_array = build_mdp_transition_array(state_ids, sprime_ids, action_ids, p_coe, r_coe, gamma_coe)
    top_actions, top_values = solve_bellman_topk(mdp_array, k_mdp, n=num_actions)
    action_logits = mdp_topk_to_action_logits(top_actions, top_values, num_actions)

    pooled = jnp.mean(jnp.concatenate([state_tok, target_tok, actions_tok, hist_tok], axis=0), axis=0)
    costs, _, progress, validity = apply_decision_heads(params.heads, pooled)
    # predicted_next_state comes from this variant's own s_prime stage (its defining
    # output), projected through the shared next_state head instead of from pooled context.
    next_state = jnp.matmul(s_prime[0], params.heads.w_next_state) + params.heads.b_next_state

    return DecisionVectorD(
        action_logits=action_logits,
        estimated_costs=costs,
        predicted_next_state=next_state,
        progress_rate_pred=progress,
        validity_score=validity,
    )
