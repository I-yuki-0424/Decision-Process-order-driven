"""Variant5_3 (docs/DPOD.ipynb cell 13): three-channel split -- actions (A),
target+state (T+S), history (H) -- each self-attend with sinusoidal position
encoding, accumulated via a residual carry over `num_l` layers.

Per-candidate action scores are read from the accumulated A-channel segment
after the residual loop (see variant_5_2_channel_split.py's module docstring
for why this port adds an explicit scoring head: the notebook hands its fused
representation to an unspecified downstream search)."""

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
    apply_decision_heads,
    init_decision_head_params,
    scaled_dot_product_attention,
    split_encoded_tokens,
)


def _sinusoidal_positional_encoding(seq_len: int, d_model: int) -> jnp.ndarray:
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))
    pe = jnp.zeros((seq_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe


class Variant53Parameters(NamedTuple):
    encoder_params: EncoderParameters
    w_action: jnp.ndarray  # (d_model, 1) per-candidate action scoring head
    b_action: jnp.ndarray  # (1,)
    heads: DecisionHeadParameters


def _glorot(key: jax.random.PRNGKey, in_dim: int, out_dim: int) -> jnp.ndarray:
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    return jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)


def init_variant_5_3_parameters(
    rng_key: jax.random.PRNGKey,
    d_model: int = 512,
    num_actions: int = 16,
    action_feat_dim: int = 32,
    num_costs: int = 4,
    num_resources: int = 8,
    target_dim: int = 8,
) -> Variant53Parameters:
    keys = jax.random.split(rng_key, 3)
    encoder_params = init_channel_encoder_params(
        keys[0], d_model=d_model, num_actions=num_actions, action_feat_dim=action_feat_dim,
        num_costs=num_costs, num_resources=num_resources, target_dim=target_dim,
    )
    heads = init_decision_head_params(keys[1], d_model=d_model, num_costs=num_costs, num_resources=num_resources)
    return Variant53Parameters(
        encoder_params=encoder_params,
        w_action=_glorot(keys[2], d_model, 1), b_action=jnp.zeros((1,)),
        heads=heads,
    )


def forward_variant_5_3(
    params: Variant53Parameters,
    input_n: InputContextN,
    num_l: int = 2,
    num_heads: int = 8,
) -> DecisionVectorD:
    """num_l, num_heads must be concrete Python ints when jitted (jax.lax.scan
    length requires a static value)."""
    tokens = encode_channel_independent(params.encoder_params, input_n)
    num_actions = input_n.actions.features.shape[0]
    state_tok, target_tok, actions_tok, hist_tok = split_encoded_tokens(tokens, num_actions)
    d_model = tokens.shape[-1]

    q = actions_tok + _sinusoidal_positional_encoding(actions_tok.shape[0], d_model)
    k_bace = jnp.concatenate([target_tok, state_tok], axis=0)
    k = k_bace + _sinusoidal_positional_encoding(k_bace.shape[0], d_model)
    v = hist_tok + _sinusoidal_positional_encoding(hist_tok.shape[0], d_model)

    q_split = q.shape[0]
    k_split = q_split + k.shape[0]
    qkv_init = jnp.concatenate([q, k, v], axis=0)

    def attention_fn(carry, _):
        qkv = qkv_init + carry
        cq, ck, cv = jnp.split(qkv, [q_split, k_split], axis=0)
        y_q = scaled_dot_product_attention(cq, cq, cq, num_heads)
        y_k = scaled_dot_product_attention(ck, ck, ck, num_heads)
        y_v = scaled_dot_product_attention(cv, cv, cv, num_heads)
        return jnp.concatenate([y_q, y_k, y_v], axis=0), None

    carry_init = jnp.zeros_like(qkv_init)
    final_carry, _ = jax.lax.scan(attention_fn, carry_init, xs=None, length=num_l)
    y = qkv_init + final_carry

    actions_repr = y[:q_split]
    pooled = jnp.mean(y, axis=0)

    action_logits = (jnp.matmul(actions_repr, params.w_action) + params.b_action).squeeze(-1)
    costs, next_state, progress, validity = apply_decision_heads(params.heads, pooled)

    return DecisionVectorD(
        action_logits=action_logits,
        estimated_costs=costs,
        predicted_next_state=next_state,
        progress_rate_pred=progress,
        validity_score=validity,
    )
