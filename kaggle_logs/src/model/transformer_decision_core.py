"""
Generic Transformer-based Decision Core Architecture for 4th-Idea.

Implements:
- Multi-Head Self-Attention (8-16 heads, d_k ~ 64, d_model = 512)
- Causal Masking & KV Cache support
- Noise Injection mechanism for robust recovery during training (Section 3)
- Multi-Task Decision Heads generating DecisionVectorD:
  d = A(Costs) + A(conditions) + S(Can use Costs) + H(reward) + H(Cost-change) + T(conditions(H))
"""

from typing import NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp

from src.model.types import (
    DecisionVectorD,
    InputContextN,
    KVCacheLayer,
    KVCacheState,
    ActionHistory,
)
from src.model.channel_encoder import (
    EncoderParameters,
    encode_channel_independent,
    init_channel_encoder_params,
)


class LayerParameters(NamedTuple):
    """Parameters for a single Multi-Head Attention Transformer Layer."""
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


class HeadParameters(NamedTuple):
    """Multi-task prediction heads for Decision Vector d."""
    w_action: jnp.ndarray       # (d_model, num_actions)
    b_action: jnp.ndarray       # (num_actions,)
    w_cost: jnp.ndarray         # (d_model, num_costs)
    b_cost: jnp.ndarray         # (num_costs,)
    w_next_state: jnp.ndarray   # (d_model, num_resources)
    b_next_state: jnp.ndarray   # (num_resources,)
    w_progress: jnp.ndarray     # (d_model, 1)
    b_progress: jnp.ndarray     # (1,)
    w_validity: jnp.ndarray     # (d_model, 1)
    b_validity: jnp.ndarray     # (1,)
    w_token_validity: jnp.ndarray # (d_model, 1)
    b_token_validity: jnp.ndarray # (1,)


class ModelParameters(NamedTuple):
    """Complete 4th-Idea Transformer Decision Model parameters (PyTree compatible)."""
    encoder_params: EncoderParameters
    layers: Tuple[LayerParameters, ...]
    heads: HeadParameters


def glorot_uniform(key: jax.random.PRNGKey, in_dim: int, out_dim: int) -> jnp.ndarray:
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    return jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)


def build_decision_attention_mask(num_actions: int, num_hist: int, is_causal: bool = True) -> jnp.ndarray:
    """Construct block attention mask for [State, Target, Actions..., History...].

    - State (idx 0) & Target (idx 1): attend to State & Target
    - Actions (idx 2..2+num_actions-1): set-level bidirectional attention across all actions, plus Context & History
    - History (idx 2+num_actions..end): causal lower-triangular attention among history steps, plus Context
    """
    seq_len = 2 + num_actions + num_hist
    mask = jnp.zeros((seq_len, seq_len), dtype=jnp.float32)

    # 1. State & Target attend to State & Target
    mask = mask.at[:2, :2].set(1.0)

    # 2. Action candidate tokens: attend to State/Target (:2), all Actions (2:2+num_actions), and History (2+num_actions:)
    mask = mask.at[2:2 + num_actions, :].set(1.0)

    # 3. History sequence tokens: attend to State/Target (:2), and history
    mask = mask.at[2 + num_actions:, :2].set(1.0)
    
    if is_causal:
        hist_mask = jnp.tril(jnp.ones((num_hist, num_hist), dtype=jnp.float32))
    else:
        hist_mask = jnp.ones((num_hist, num_hist), dtype=jnp.float32)
        
    mask = mask.at[2 + num_actions:, 2 + num_actions:].set(hist_mask)

    return mask


def init_model_parameters(
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
) -> ModelParameters:
    """Initialize complete parameters for 4th-Idea Transformer model."""
    keys = jax.random.split(rng_key, num_layers + 2)
    head_dim = d_model // num_heads

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

    h_keys = jax.random.split(keys[-1], 6)
    heads = HeadParameters(
        w_action=glorot_uniform(h_keys[0], d_model, num_actions),
        b_action=jnp.zeros((num_actions,)),
        w_cost=glorot_uniform(h_keys[1], d_model, num_costs),
        b_cost=jnp.zeros((num_costs,)),
        w_next_state=glorot_uniform(h_keys[2], d_model, num_resources),
        b_next_state=jnp.zeros((num_resources,)),
        w_progress=glorot_uniform(h_keys[3], d_model, 1),
        b_progress=jnp.zeros((1,)),
        w_validity=glorot_uniform(h_keys[4], d_model, 1),
        b_validity=jnp.zeros((1,)),
        w_token_validity=glorot_uniform(h_keys[5], d_model, 1),
        b_token_validity=jnp.zeros((1,)),
    )

    return ModelParameters(
        encoder_params=encoder_params,
        layers=tuple(layers),
        heads=heads,
    )


def inject_noise_to_history(
    key: jax.random.PRNGKey,
    history: ActionHistory,
    num_actions: int,
    noise_prob: float = 0.15,
) -> ActionHistory:
    """Inject sub-optimal or random invalid actions into history H during training.

    Section 3 of 4th-Idea:
    Trains the attention mechanism to recognize noise tokens, ignore invalid context,
    and recover the optimal path, alleviating Exposure Bias.
    """
    key1, key2 = jax.random.split(key)
    seq_len = history.action_indices.shape[0]
    
    # Bernoulli mask for noise injection sites
    noise_mask = jax.random.bernoulli(key1, p=noise_prob, shape=(seq_len,))
    
    # Generate random noisy action indices
    noisy_actions = jax.random.randint(key2, shape=(seq_len,), minval=0, maxval=num_actions)
    
    # Select noisy action index where noise_mask is True
    augmented_indices = jnp.where(noise_mask, noisy_actions, history.action_indices)
    
    return ActionHistory(
        action_indices=augmented_indices,
        rewards=history.rewards,
        cost_changes=history.cost_changes,
        noise_mask=jnp.logical_or(history.noise_mask, noise_mask),
        seq_len=history.seq_len,
    )


def layer_norm(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def forward_layer(
    params: LayerParameters,
    x: jnp.ndarray,
    num_heads: int,
    head_dim: int,
    kv_layer_cache: Optional[KVCacheLayer] = None,
    attn_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Optional[KVCacheLayer]]:
    """Forward pass through a single Multi-Head Attention layer with optional KV Cache and Attention Mask."""
    seq_len, d_model = x.shape

    # Pre-LN
    norm_x = layer_norm(x, params.gamma1, params.beta1)

    # Q, K, V Projections
    q = jnp.matmul(norm_x, params.w_q).reshape(seq_len, num_heads, head_dim)
    k = jnp.matmul(norm_x, params.w_k).reshape(seq_len, num_heads, head_dim)
    v = jnp.matmul(norm_x, params.w_v).reshape(seq_len, num_heads, head_dim)

    # Transpose to (num_heads, seq_len, head_dim)
    q = jnp.transpose(q, (1, 0, 2))
    k = jnp.transpose(k, (1, 0, 2))
    v = jnp.transpose(v, (1, 0, 2))

    if kv_layer_cache is not None:
        # Append to KV cache if provided
        k = jnp.concatenate([kv_layer_cache.cached_keys, k], axis=-2)
        v = jnp.concatenate([kv_layer_cache.cached_values, v], axis=-2)
        new_kv_layer_cache = KVCacheLayer(cached_keys=k, cached_values=v, current_len=jnp.array(k.shape[1]))
    else:
        new_kv_layer_cache = None

    kv_seq_len = k.shape[1]
    scale = 1.0 / jnp.sqrt(head_dim)
    attn_weights = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) * scale  # (num_heads, seq_len, kv_seq_len)

    # Block Attention Mask or Causal Fallback Mask
    if attn_mask is not None:
        attn_weights = jnp.where(attn_mask[None, :, :] == 1.0, attn_weights, -1e9)
    else:
        mask = jnp.tril(jnp.ones((seq_len, kv_seq_len)))
        attn_weights = jnp.where(mask[None, :, :] == 1, attn_weights, -1e9)
    attn_probs = jax.nn.softmax(attn_weights, axis=-1)

    # Attention Output
    attn_out = jnp.matmul(attn_probs, v)  # (num_heads, seq_len, head_dim)
    attn_out = jnp.transpose(attn_out, (1, 0, 2)).reshape(seq_len, d_model)
    attn_out = jnp.matmul(attn_out, params.w_o)

    # Residual 1
    x = x + attn_out

    # Pre-LN FeedForward
    norm_x2 = layer_norm(x, params.gamma2, params.beta2)
    ff = jax.nn.gelu(jnp.matmul(norm_x2, params.w_ff1) + params.b_ff1)
    ff = jnp.matmul(ff, params.w_ff2) + params.b_ff2

    # Residual 2
    x = x + ff

    return x, new_kv_layer_cache


def forward_decision_transformer(
    params: ModelParameters,
    input_n: InputContextN,
    rng_key: Optional[jax.random.PRNGKey] = None,
    is_training: bool = False,
    kv_cache: Optional[KVCacheState] = None,
    is_causal: bool = True,
    z_compression_interval: int = 0,
) -> Tuple[DecisionVectorD, Optional[KVCacheState]]:
    """Forward pass through complete DecisionTransformerCore.

    Returns:
        DecisionVectorD and updated KVCacheState.
    """
    # 1. Apply Noise Injection during training
    if is_training and rng_key is not None:
        num_actions = params.heads.w_action.shape[1]
        history = inject_noise_to_history(rng_key, input_n.history, num_actions=num_actions)
        input_n = input_n._replace(history=history)

    # 2. Channel-Independent Encoding
    tokens = encode_channel_independent(params.encoder_params, input_n)

    num_actions = input_n.actions.features.shape[0]
    num_hist = input_n.history.action_indices.shape[0]

    # Z-Compression: Pool history tokens in groups of Z if z_compression_interval > 0
    if z_compression_interval > 0:
        hist_tokens = tokens[2 + num_actions:]
        
        # Calculate how many groups we have
        # To make it compatible with JAX static shapes, we can do a padded reshape and mean.
        # But simpler: apply a static convolution/pooling equivalent, or since this is a demonstration,
        # we can just use the provided WorkingMemoryState mechanism or just keep it dynamic.
        # For statically shaped JAX, we reshape history into (num_hist // Z, Z, d_model) and take mean.
        # This requires num_hist to be divisible by Z.
        # Let's just simulate it by doing a simple mean pooling if divisible
        
        # Ensure num_hist is cleanly divisible, else we pad or just compress the largest chunk
        # Assuming num_hist = 256, Z=32, 64, 128 (all divide 256 cleanly)
        compressed_hist = hist_tokens.reshape(num_hist // z_compression_interval, z_compression_interval, tokens.shape[-1]).mean(axis=1)
        # Update tokens with compressed history
        tokens = jnp.concatenate([tokens[:2 + num_actions], compressed_hist], axis=0)
        num_hist = num_hist // z_compression_interval
        

    # 3. Transformer Layer Stack with Block Attention Mask
    d_model = params.heads.w_action.shape[0]
    num_heads = 8
    head_dim = d_model // num_heads

    attn_mask = build_decision_attention_mask(num_actions, num_hist, is_causal=is_causal)

    new_kv_layers = []
    for i, layer_params in enumerate(params.layers):
        layer_kv = kv_cache.layers[i] if kv_cache is not None else None
        tokens, new_kv = forward_layer(
            layer_params,
            tokens,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_layer_cache=layer_kv,
            attn_mask=attn_mask,
        )
        if new_kv is not None:
            new_kv_layers.append(new_kv)

    # 4. Pooling / Aggregation
    context_vector = tokens[-1]  # (d_model,)

    # 5. Multi-task Decision Heads
    heads = params.heads
    action_logits = jnp.matmul(context_vector, heads.w_action) + heads.b_action
    estimated_costs = jnp.matmul(context_vector, heads.w_cost) + heads.b_cost
    predicted_next_state = jnp.matmul(context_vector, heads.w_next_state) + heads.b_next_state
    progress_rate_pred = jax.nn.sigmoid(jnp.matmul(context_vector, heads.w_progress) + heads.b_progress)[0]
    validity_score = jax.nn.sigmoid(jnp.matmul(context_vector, heads.w_validity) + heads.b_validity)[0]

    # Token-level validity logits for history tokens
    hist_tokens_repr = tokens[2 + num_actions:]  # (num_hist, d_model)
    token_validity_logits = (jnp.matmul(hist_tokens_repr, heads.w_token_validity) + heads.b_token_validity).squeeze(-1)

    decision_d = DecisionVectorD(
        action_logits=action_logits,
        estimated_costs=estimated_costs,
        predicted_next_state=predicted_next_state,
        progress_rate_pred=progress_rate_pred,
        validity_score=validity_score,
        token_validity_logits=token_validity_logits,
    )

    new_cache_state = KVCacheState(layers=tuple(new_kv_layers)) if new_kv_layers else None

    return decision_d, new_cache_state
