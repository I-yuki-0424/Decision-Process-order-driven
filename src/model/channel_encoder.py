"""
Channel-Independent Encoder for 4th-Idea Decision Process Architecture.

Adheres to 4th-Idea Section 5.3:
Channel Independence separates variables of distinct physical nature and scales
(Time, Cost, State, History) into independent channels prior to Multi-Head Attention,
preventing oversmoothing and preserving feature-specific dynamics.
"""

from typing import Dict, NamedTuple, Tuple
import jax
import jax.numpy as jnp
from src.model.types import InputContextN


class EncoderParameters(NamedTuple):
    """Parameters for Channel-Independent Encoder."""
    w_actions: jnp.ndarray       # Shape (action_feat_dim, d_model)
    w_costs: jnp.ndarray         # Shape (num_costs, d_model)
    w_state: jnp.ndarray         # Shape (num_resources, d_model)
    w_history: jnp.ndarray       # Shape (hist_feat_dim, d_model)
    w_target: jnp.ndarray        # Shape (target_dim, d_model)
    w_abstraction: jnp.ndarray   # Shape (1, d_model) - D-dimensional abstraction embedding
    channel_pos_embed: jnp.ndarray  # Shape (num_channels, d_model)


def init_channel_encoder_params(
    rng_key: jax.random.PRNGKey,
    d_model: int = 512,
    num_actions: int = 16,
    action_feat_dim: int = 32,
    num_costs: int = 4,
    num_resources: int = 8,
    max_history_len: int = 128,
    target_dim: int = 8,
) -> EncoderParameters:
    """Initialize weights for the Channel-Independent Encoder."""
    keys = jax.random.split(rng_key, 7)
    
    def glorot(key, in_dim, out_dim):
        limit = jnp.sqrt(6.0 / (in_dim + out_dim))
        return jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)

    w_actions = glorot(keys[0], action_feat_dim, d_model)
    w_costs = glorot(keys[1], num_costs, d_model)
    w_state = glorot(keys[2], num_resources, d_model)
    w_history = glorot(keys[3], 1 + num_costs, d_model)  # action index embed + cost deltas
    w_target = glorot(keys[4], target_dim, d_model)
    w_abstraction = glorot(keys[5], 1, d_model)
    
    # 4 channels: Action, State, History, Target
    channel_pos_embed = glorot(keys[6], 4, d_model)

    return EncoderParameters(
        w_actions=w_actions,
        w_costs=w_costs,
        w_state=w_state,
        w_history=w_history,
        w_target=w_target,
        w_abstraction=w_abstraction,
        channel_pos_embed=channel_pos_embed,
    )


def encode_channel_independent(
    params: EncoderParameters,
    input_n: InputContextN,
    use_abstraction_embed: bool = True,
) -> jnp.ndarray:
    """Encode InputContextN into channel-separated token sequences for Attention."""
    # 1. Action Channel Tokens
    act_tokens = jnp.matmul(input_n.actions.features, params.w_actions)
    
    # Inject D-dimensional Abstraction Vector E_abs if enabled
    if use_abstraction_embed and input_n.actions.abstraction_scales is not None:
        scales = input_n.actions.abstraction_scales[:, None].astype(jnp.float32)
        e_abs = jnp.matmul(jnp.log1p(scales), params.w_abstraction)  # (num_actions, d_model)
        act_tokens = act_tokens + e_abs

    act_tokens = act_tokens + params.channel_pos_embed[0]

    # 2. State Channel Token
    # state.resource_levels: (num_resources,) -> (1, d_model)
    state_feat = input_n.state.resource_levels
    state_token = jnp.matmul(state_feat[None, :], params.w_state)
    state_token = state_token + params.channel_pos_embed[1]

    # 3. History Channel Tokens
    # history.action_indices: (seq_len,), history.cost_changes: (seq_len, num_costs)
    hist_act = input_n.history.action_indices[:, None].astype(jnp.float32)
    hist_cost = input_n.history.cost_changes
    hist_feat = jnp.concatenate([hist_act, hist_cost], axis=-1)  # (seq_len, 1 + num_costs)
    hist_tokens = jnp.matmul(hist_feat, params.w_history)
    hist_tokens = hist_tokens + params.channel_pos_embed[2]

    # 4. Target Channel Token
    # target.target_state: (target_dim,) -> (1, d_model)
    target_token = jnp.matmul(input_n.target.target_state[None, :], params.w_target)
    target_token = target_token + params.channel_pos_embed[3]

    # Concatenate tokens across sequence dimension: [State, Target, Actions..., History...]
    all_tokens = jnp.concatenate([state_token, target_token, act_tokens, hist_tokens], axis=0)
    return all_tokens
