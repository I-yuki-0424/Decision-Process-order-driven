"""
Baseline Decision Transformer (3rd-Idea Architecture).

Serves as the baseline comparison model for 4th-Idea.
Implements:
1. Standard concatenated feature encoding (No Channel Independence).
2. Clean sequence training (No Noise Injection, Teacher Forcing).
3. Greedy single-pass autoregressive inference (No Beam Search, No KV Cache state pairs).
"""

from typing import NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp

from src.model.types import DecisionVectorD, InputContextN
from src.model.transformer_decision_core import LayerParameters, layer_norm, glorot_uniform


class BaselineModelParameters(NamedTuple):
    """Parameters for 3rd-Idea Baseline Greedy Transformer."""
    w_concat_in: jnp.ndarray   # Dense projection from concatenated features to d_model
    layers: Tuple[LayerParameters, ...]
    w_action: jnp.ndarray      # Head for action logits
    b_action: jnp.ndarray
    w_cost: jnp.ndarray        # Head for cost predictions
    b_cost: jnp.ndarray
    w_progress: jnp.ndarray    # Head for progress rate prediction
    b_progress: jnp.ndarray
    d_model: int
    num_heads: int
    head_dim: int


def init_baseline_parameters(
    rng_key: jax.random.PRNGKey,
    num_layers: int = 4,
    d_model: int = 512,
    num_heads: int = 8,
    d_ff: int = 2048,
    num_actions: int = 16,
    action_feat_dim: int = 32,
    num_costs: int = 4,
    num_resources: int = 8,
    target_dim: int = 8,
    max_history_len: int = 128,
) -> BaselineModelParameters:
    """Initialize parameters for 3rd-Idea Baseline model."""
    keys = jax.random.split(rng_key, num_layers + 3)
    head_dim = d_model // num_heads

    # Direct concatenation vector size:
    # State (num_resources) + Target (target_dim) + Action features (num_actions * action_feat_dim) + History summary
    concat_dim = num_resources + target_dim + num_actions * action_feat_dim + max_history_len * (1 + num_costs)

    w_concat_in = glorot_uniform(keys[0], concat_dim, d_model)

    layers = []
    for i in range(num_layers):
        l_keys = jax.random.split(keys[i + 1], 6)
        layer = LayerParameters(
            w_q=glorot_uniform(l_keys[0], d_model, d_model),
            w_k=glorot_uniform(l_keys[1], d_model, d_model),
            w_v=glorot_uniform(l_keys[2], d_model, d_model),
            w_o=glorot_uniform(l_keys[3], d_model, d_model),
            w_ff1=glorot_uniform(l_keys[4], d_model, d_ff),
            b_ff1=jnp.zeros((d_ff,)),
            w_ff2=glorot_uniform(l_keys[5], d_ff, d_model),
            b_ff2=jnp.zeros((d_model,)),
            gamma1=jnp.ones((d_model,)),
            beta1=jnp.zeros((d_model,)),
            gamma2=jnp.ones((d_model,)),
            beta2=jnp.zeros((d_model,)),
        )
        layers.append(layer)

    h_keys = jax.random.split(keys[-1], 3)
    w_action = glorot_uniform(h_keys[0], d_model, num_actions)
    b_action = jnp.zeros((num_actions,))
    w_cost = glorot_uniform(h_keys[1], d_model, num_costs)
    b_cost = jnp.zeros((num_costs,))
    w_progress = glorot_uniform(h_keys[2], d_model, 1)
    b_progress = jnp.zeros((1,))

    return BaselineModelParameters(
        w_concat_in=w_concat_in,
        layers=tuple(layers),
        w_action=w_action,
        b_action=b_action,
        w_cost=w_cost,
        b_cost=b_cost,
        w_progress=w_progress,
        b_progress=b_progress,
        d_model=d_model,
        num_heads=num_heads,
        head_dim=head_dim,
    )


def encode_baseline_features(input_n: InputContextN) -> jnp.ndarray:
    """Concatenate features into a single flat vector (No Channel Independence)."""
    state_flat = input_n.state.resource_levels
    target_flat = input_n.target.target_state
    act_flat = input_n.actions.features.reshape(-1)
    
    hist_act = input_n.history.action_indices[:, None].astype(jnp.float32)
    hist_cost = input_n.history.cost_changes
    hist_flat = jnp.concatenate([hist_act, hist_cost], axis=-1).reshape(-1)

    concat = jnp.concatenate([state_flat, target_flat, act_flat, hist_flat], axis=0)
    return concat


def forward_baseline_transformer(
    params: BaselineModelParameters,
    input_n: InputContextN,
) -> DecisionVectorD:
    """Forward pass through 3rd-Idea Baseline Model (Single-pass Greedy)."""
    # 1. Direct concatenation encoding
    concat_vec = encode_baseline_features(input_n)
    
    # Project to d_model sequence of length 1 token
    token = jnp.matmul(concat_vec[None, :], params.w_concat_in)  # (1, d_model)

    # 2. Transformer layers (without KV cache)
    x = token
    for layer in params.layers:
        norm_x = layer_norm(x, layer.gamma1, layer.beta1)
        q = jnp.matmul(norm_x, layer.w_q).reshape(1, params.num_heads, params.head_dim)
        k = jnp.matmul(norm_x, layer.w_k).reshape(1, params.num_heads, params.head_dim)
        v = jnp.matmul(norm_x, layer.w_v).reshape(1, params.num_heads, params.head_dim)

        q = jnp.transpose(q, (1, 0, 2))
        k = jnp.transpose(k, (1, 0, 2))
        v = jnp.transpose(v, (1, 0, 2))

        attn = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) / jnp.sqrt(params.head_dim)
        attn_probs = jax.nn.softmax(attn, axis=-1)
        attn_out = jnp.matmul(attn_probs, v)
        attn_out = jnp.transpose(attn_out, (1, 0, 2)).reshape(1, params.d_model)
        attn_out = jnp.matmul(attn_out, layer.w_o)

        x = x + attn_out

        norm_x2 = layer_norm(x, layer.gamma2, layer.beta2)
        ff = jax.nn.gelu(jnp.matmul(norm_x2, layer.w_ff1) + layer.b_ff1)
        ff = jnp.matmul(ff, layer.w_ff2) + layer.b_ff2
        x = x + ff

    ctx = x[0]

    # 3. Heads
    action_logits = jnp.matmul(ctx, params.w_action) + params.b_action
    estimated_costs = jnp.matmul(ctx, params.w_cost) + params.b_cost
    progress_pred = jax.nn.sigmoid(jnp.matmul(ctx, params.w_progress) + params.b_progress)[0]

    return DecisionVectorD(
        action_logits=action_logits,
        estimated_costs=estimated_costs,
        predicted_next_state=input_n.state.resource_levels,
        progress_rate_pred=progress_pred,
        validity_score=jnp.array(1.0),  # Baseline assumes clean validity
    )
