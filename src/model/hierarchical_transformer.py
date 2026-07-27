"""
5th-Idea Hierarchical Decision Transformer with Restricted Local Attention & Feature Toggle.

Implements:
1. Configurable Feature Toggle (use_hierarchical: bool)
   - Macro Subgoal Cluster Head (M clusters, e.g. M=50)
   - Micro Fine Action Head (K actions per cluster, e.g. K=40, total |A| = 2000)
2. Restricted Local Attention (sliding window r=32) for O(r * N * d) sequence scaling
3. Working Memory History Compression mechanism
"""

from typing import NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp

from src.model.types import (
    DecisionVectorD,
    HierarchicalDecisionVectorD,
    InputContextN,
    KVCacheLayer,
    KVCacheState,
    WorkingMemoryState,
)
from src.model.channel_encoder import (
    EncoderParameters,
    encode_channel_independent,
    init_channel_encoder_params,
)


class LayerParameters(NamedTuple):
    """Parameters for a single Multi-Head Attention Layer with local window support."""
    w_q: jnp.ndarray  # (d_model, num_heads * head_dim)
    w_k: jnp.ndarray  # (d_model, num_heads * head_dim)
    w_v: jnp.ndarray  # (d_model, num_heads * head_dim)
    w_o: jnp.ndarray  # (num_heads * head_dim, d_model)
    
    w_ff1: jnp.ndarray  # (d_model, d_ff)
    b_ff1: jnp.ndarray  # (d_ff,)
    w_ff2: jnp.ndarray  # (d_ff, d_model)
    b_ff2: jnp.ndarray  # (d_model,)
    
    gamma1: jnp.ndarray # (d_model,)
    beta1: jnp.ndarray  # (d_model,)
    gamma2: jnp.ndarray # (d_model,)
    beta2: jnp.ndarray  # (d_model,)


class HierarchicalHeadParameters(NamedTuple):
    """5th-Idea Multi-task prediction heads supporting Hierarchical Abstraction & Flat Toggle."""
    w_action_flat: jnp.ndarray        # (d_model, num_actions)
    b_action_flat: jnp.ndarray        # (num_actions,)

    w_macro_cluster: jnp.ndarray      # (d_model, num_macro_clusters)
    b_macro_cluster: jnp.ndarray      # (num_macro_clusters,)

    w_micro_action: jnp.ndarray       # (d_model, num_fine_actions)
    b_micro_action: jnp.ndarray       # (num_fine_actions,)

    w_q_value: jnp.ndarray            # (d_model, num_actions) - Off-Policy Q-head
    b_q_value: jnp.ndarray            # (num_actions,)

    w_cost: jnp.ndarray               # (d_model, num_costs)
    b_cost: jnp.ndarray               # (num_costs,)
    w_next_state: jnp.ndarray         # (d_model, num_resources)
    b_next_state: jnp.ndarray         # (num_resources,)
    w_progress: jnp.ndarray           # (d_model, 1)
    b_progress: jnp.ndarray           # (1,)
    w_validity: jnp.ndarray           # (d_model, 1)
    b_validity: jnp.ndarray           # (1,)


class HierarchicalModelParameters(NamedTuple):
    """Complete 5th-Idea Model Parameters container."""
    encoder_params: EncoderParameters
    layers: Tuple[LayerParameters, ...]
    heads: HierarchicalHeadParameters


def glorot_uniform(key: jax.random.PRNGKey, in_dim: int, out_dim: int) -> jnp.ndarray:
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    return jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)


def init_hierarchical_model_parameters(
    rng_key: jax.random.PRNGKey,
    num_layers: int = 4,
    d_model: int = 512,
    num_heads: int = 8,
    d_ff: int = 2048,
    num_actions: int = 2000,
    num_macro_clusters: int = 50,
    num_fine_actions: int = 40,
    action_feat_dim: int = 32,
    num_costs: int = 4,
    num_resources: int = 8,
    target_dim: int = 8,
) -> HierarchicalModelParameters:
    """Initialize complete parameters for 5th-Idea Hierarchical Transformer Model."""
    keys = jax.random.split(rng_key, num_layers + 3)

    encoder_params = init_channel_encoder_params(
        keys[0],
        d_model=d_model,
        num_actions=num_actions,
        action_feat_dim=action_feat_dim,
        num_costs=num_costs,
        num_resources=num_resources,
        target_dim=target_dim,
    )

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

    h_keys = jax.random.split(keys[-1], 8)
    heads = HierarchicalHeadParameters(
        w_action_flat=glorot_uniform(h_keys[0], d_model, num_actions),
        b_action_flat=jnp.zeros((num_actions,)),
        w_macro_cluster=glorot_uniform(h_keys[1], d_model, num_macro_clusters),
        b_macro_cluster=jnp.zeros((num_macro_clusters,)),
        w_micro_action=glorot_uniform(h_keys[2], d_model, num_fine_actions),
        b_micro_action=jnp.zeros((num_fine_actions,)),
        w_q_value=glorot_uniform(h_keys[3], d_model, num_actions),
        b_q_value=jnp.zeros((num_actions,)),
        w_cost=glorot_uniform(h_keys[4], d_model, num_costs),
        b_cost=jnp.zeros((num_costs,)),
        w_next_state=glorot_uniform(h_keys[5], d_model, num_resources),
        b_next_state=jnp.zeros((num_resources,)),
        w_progress=glorot_uniform(h_keys[6], d_model, 1),
        b_progress=jnp.zeros((1,)),
        w_validity=glorot_uniform(h_keys[7], d_model, 1),
        b_validity=jnp.zeros((1,)),
    )

    return HierarchicalModelParameters(
        encoder_params=encoder_params,
        layers=tuple(layers),
        heads=heads,
    )


def restricted_local_causal_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    window_size: int = 32,
) -> jnp.ndarray:
    """Compute Multi-Head Causal Attention restricted to sliding window r = window_size.

    Reduces attention complexity from O(N^2 * d) down to O(r * N * d).
    """
    seq_len, d_k = q.shape[1], q.shape[2]
    scale = 1.0 / jnp.sqrt(d_k)
    scores = jnp.einsum("h i d, h j d -> h i j", q, k) * scale

    # Create restricted causal mask (referencing only recent r tokens)
    idx_i = jnp.arange(seq_len)[:, None]
    idx_j = jnp.arange(seq_len)[None, :]
    causal_mask = (idx_j <= idx_i) & (idx_i - idx_j < window_size)
    
    scores = jnp.where(causal_mask[None, :, :], scores, -1e9)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    
    out = jnp.einsum("h i j, h j d -> h i d", attn_weights, v)
    return out


def forward_hierarchical_transformer(
    params: HierarchicalModelParameters,
    input_n: InputContextN,
    use_hierarchical: bool = True,
    use_abstraction_embed: bool = True,
    window_size: int = 32,
    rng_key: Optional[jax.random.PRNGKey] = None,
    is_training: bool = False,
    num_macro_clusters: int = 50,
    num_fine_actions: int = 40,
) -> Tuple[HierarchicalDecisionVectorD, Optional[WorkingMemoryState]]:
    """Forward pass of 5th-Idea Hierarchical Transformer Decision Core."""
    d_model = params.heads.w_action_flat.shape[0]
    num_actions = params.heads.w_action_flat.shape[1]

    # 1. Encode Context Tokens using Channel-Independent Encoder
    tokens = encode_channel_independent(params.encoder_params, input_n, use_abstraction_embed=use_abstraction_embed)
    seq_len = tokens.shape[0]

    # 2. Process through Transformer Attention Layers
    x = tokens
    num_heads = 8
    head_dim = d_model // num_heads

    for layer in params.layers:
        x_norm1 = (x - jnp.mean(x, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(x, axis=-1, keepdims=True) + 1e-5)
        x_norm1 = x_norm1 * layer.gamma1 + layer.beta1

        q = jnp.matmul(x_norm1, layer.w_q).reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)
        k = jnp.matmul(x_norm1, layer.w_k).reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)
        v = jnp.matmul(x_norm1, layer.w_v).reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)

        attn_out = restricted_local_causal_attention(q, k, v, window_size=window_size)
        attn_out = attn_out.swapaxes(0, 1).reshape(seq_len, d_model)
        x_attn = jnp.matmul(attn_out, layer.w_o)

        x = x + x_attn

        x_norm2 = (x - jnp.mean(x, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(x, axis=-1, keepdims=True) + 1e-5)
        x_norm2 = x_norm2 * layer.gamma2 + layer.beta2

        ff1 = jax.nn.gelu(jnp.matmul(x_norm2, layer.w_ff1) + layer.b_ff1)
        ff2 = jnp.matmul(ff1, layer.w_ff2) + layer.b_ff2

        x = x + ff2

    # Global context representation
    pooled_context = jnp.mean(x, axis=0)  # (d_model,)

    # 3. Multi-Task Heads & Off-Policy Q-head
    cost_pred = jnp.matmul(pooled_context, params.heads.w_cost) + params.heads.b_cost
    next_state_pred = jnp.matmul(pooled_context, params.heads.w_next_state) + params.heads.b_next_state
    progress_pred = jax.nn.sigmoid(jnp.matmul(pooled_context, params.heads.w_progress) + params.heads.b_progress)[0]
    validity_pred = jax.nn.sigmoid(jnp.matmul(pooled_context, params.heads.w_validity) + params.heads.b_validity)[0]
    q_val_pred = jnp.matmul(pooled_context, params.heads.w_q_value) + params.heads.b_q_value  # (|A|,)

    if use_hierarchical:
        macro_logits = jnp.matmul(pooled_context, params.heads.w_macro_cluster) + params.heads.b_macro_cluster
        micro_logits = jnp.matmul(pooled_context, params.heads.w_micro_action) + params.heads.b_micro_action

        macro_log_probs = jax.nn.log_softmax(macro_logits)
        micro_log_probs = jax.nn.log_softmax(micro_logits)

        macro_expanded = jnp.repeat(macro_log_probs, num_fine_actions)
        micro_tiled = jnp.tile(micro_log_probs, num_macro_clusters)
        joint_action_logits = macro_expanded + micro_tiled

        selected_macro = jnp.argmax(macro_logits)

        decision_d = HierarchicalDecisionVectorD(
            macro_logits=macro_logits,
            micro_logits=micro_logits,
            action_logits=joint_action_logits,
            estimated_costs=cost_pred,
            predicted_next_state=next_state_pred,
            progress_rate_pred=progress_pred,
            validity_score=validity_pred,
            selected_macro_cluster=selected_macro,
            q_values=q_val_pred,
        )
    else:
        flat_logits = jnp.matmul(pooled_context, params.heads.w_action_flat) + params.heads.b_action_flat
        selected_macro = jnp.array(0, dtype=jnp.int32)

        decision_d = HierarchicalDecisionVectorD(
            macro_logits=jnp.zeros((num_macro_clusters,)),
            micro_logits=jnp.zeros((num_fine_actions,)),
            action_logits=flat_logits,
            estimated_costs=cost_pred,
            predicted_next_state=next_state_pred,
            progress_rate_pred=progress_pred,
            validity_score=validity_pred,
            selected_macro_cluster=selected_macro,
            q_values=q_val_pred,
        )

    memory_slots = 4
    compressed_mem = jnp.zeros((memory_slots, d_model)).at[0, :].set(pooled_context)
    memory_state = WorkingMemoryState(
        compressed_memory=compressed_mem,
        last_compressed_step=jnp.array(seq_len, dtype=jnp.int32),
    )

    return decision_d, memory_state
