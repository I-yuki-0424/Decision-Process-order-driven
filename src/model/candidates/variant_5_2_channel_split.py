"""Variant5_2 (docs/DPOD.ipynb cell 12): separate attention for
actions/target/state (the "AGS" channel) vs. self-attention for history,
integrated by a dense fusion layer, layered `num_l` times.

The notebook hands its fused (actions, target, state) representation to a
downstream MDP/Bellman search that isn't specified per-variant (that's what
MDPBranch/TransformerBranch cover separately). To make this a standalone,
trainable `forward_*` over InputContextN, action selection here is a direct
dense scoring head over the fused pooled representation -- the channel-split
fusion mechanism (this variant's actual idea) is preserved unchanged; only the
"and then search for an action" tail is filled in, the same way it would need
to be for any of these variants to run end to end.
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
    apply_decision_heads,
    init_decision_head_params,
    scaled_dot_product_attention,
    split_encoded_tokens,
)


class Variant52LayerParameters(NamedTuple):
    """K/V projection for the AGS attention branch, weight-tied across `num_l`
    scan steps (matches the notebook's single k_s/v_s pair reused every
    layer), plus the dense fusion layer combining the AGS and history
    attention outputs."""
    w_k: jnp.ndarray      # (d_model, d_model)
    w_v: jnp.ndarray      # (d_model, d_model)
    w_dense: jnp.ndarray  # (d_model, d_model)
    b_dense: jnp.ndarray  # (d_model,)


class Variant52Parameters(NamedTuple):
    encoder_params: EncoderParameters
    layer: Variant52LayerParameters
    w_action: jnp.ndarray  # (d_model, 1) per-candidate action scoring head
    b_action: jnp.ndarray  # (1,)
    heads: DecisionHeadParameters


def _glorot(key: jax.random.PRNGKey, in_dim: int, out_dim: int) -> jnp.ndarray:
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    return jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)


def init_variant_5_2_parameters(
    rng_key: jax.random.PRNGKey,
    d_model: int = 512,
    num_actions: int = 16,
    action_feat_dim: int = 32,
    num_costs: int = 4,
    num_resources: int = 8,
    target_dim: int = 8,
) -> Variant52Parameters:
    keys = jax.random.split(rng_key, 4)
    encoder_params = init_channel_encoder_params(
        keys[0], d_model=d_model, num_actions=num_actions, action_feat_dim=action_feat_dim,
        num_costs=num_costs, num_resources=num_resources, target_dim=target_dim,
    )
    l_keys = jax.random.split(keys[1], 4)
    layer = Variant52LayerParameters(
        w_k=_glorot(l_keys[0], d_model, d_model), w_v=_glorot(l_keys[1], d_model, d_model),
        w_dense=_glorot(l_keys[2], d_model, d_model), b_dense=jnp.zeros((d_model,)),
    )
    a_keys = jax.random.split(keys[2], 1)
    heads = init_decision_head_params(keys[3], d_model=d_model, num_costs=num_costs, num_resources=num_resources)
    return Variant52Parameters(
        encoder_params=encoder_params, layer=layer,
        w_action=_glorot(a_keys[0], d_model, 1), b_action=jnp.zeros((1,)),
        heads=heads,
    )


def forward_variant_5_2(
    params: Variant52Parameters,
    input_n: InputContextN,
    num_l: int = 2,
    num_heads: int = 8,
) -> DecisionVectorD:
    """num_l, num_heads must be concrete Python ints when jitted (jax.lax.scan
    length requires a static value)."""
    tokens = encode_channel_independent(params.encoder_params, input_n)
    num_actions = input_n.actions.features.shape[0]
    state_tok, target_tok, actions_tok, hist_tok = split_encoded_tokens(tokens, num_actions)

    q_ags_init = jnp.concatenate([actions_tok, target_tok, state_tok], axis=0)
    n_ags = q_ags_init.shape[0]
    k_s = jnp.matmul(tokens, params.layer.w_k)
    v_s = jnp.matmul(tokens, params.layer.w_v)

    def layer_fn(carry, _):
        q_ags, h = carry
        y_ags = scaled_dot_product_attention(q_ags, k_s, v_s, num_heads)
        y_hist = scaled_dot_product_attention(h, h, h, num_heads)
        combined = jnp.concatenate([y_ags, y_hist], axis=0)
        out = jnp.matmul(combined, params.layer.w_dense) + params.layer.b_dense
        return (out[:n_ags], out[n_ags:]), None

    (final_q, final_h), _ = jax.lax.scan(layer_fn, (q_ags_init, hist_tok), xs=None, length=num_l)

    actions_repr = final_q[:num_actions]
    pooled = jnp.mean(jnp.concatenate([final_q, final_h], axis=0), axis=0)

    action_logits = (jnp.matmul(actions_repr, params.w_action) + params.b_action).squeeze(-1)
    costs, next_state, progress, validity = apply_decision_heads(params.heads, pooled)

    return DecisionVectorD(
        action_logits=action_logits,
        estimated_costs=costs,
        predicted_next_state=next_state,
        progress_rate_pred=progress,
        validity_score=validity,
    )
