"""Stages shared by the "ideal structure" pipeline candidates (docs/DPOD.ipynb,
cells 18-20: `IdealStructurePipeline`, `MDPBranch`, `TransformerBranch`) and by
`Variant5_1Bace` / `Variant5_4` (cells 11 & 14), which reuse the Bellman/MDP
solver (`MDPSolver` in the notebook).

Ported onto the InputContextN / channel-encoder contract: every tensor here is
a channel-encoded token from src/model/channel_encoder.py, not a raw embedding.

Critical-review note (see docs/core/STATE.yaml, task TASK-20260925-001):
`TensorOps.embeddings_to_scalar_id` in the notebook mean-pools a (N, d)
embedding block down to one scalar per row. The notebook's own docstring says
this makes the MDP branch's top_actions come out identical across every row
and batch element -- nothing is actually being distinguished/learned through
that path. `discretize_embeddings_to_ids` below is the documented replacement:
a small learned linear classifier head trained jointly through the action loss.
This is an explicit placeholder for an open design question, not a validated
final discretization scheme -- it only guarantees the shapes MDPSolver expects
line up with real per-candidate distinctions instead of a collapsed constant.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from src.model.types import ActionsData, SystemState


class SharedStageParameters(NamedTuple):
    """Parameters for physical_key_filter / predict_next_goal /
    compress_step_to_history (MDPBranch & TransformerBranch, verbatim-shared in
    the notebook) plus the discretization head (Variant5_1Bace & Variant5_4)."""
    w_k: jnp.ndarray            # (d_model, d_model) physical-key projection
    b_k: jnp.ndarray            # (d_model,)
    w_goal: jnp.ndarray         # (d_model, d_model) next-goal prediction
    b_goal: jnp.ndarray         # (d_model,)
    w_z: jnp.ndarray            # (d_model, d_model) per-step history compression
    b_z: jnp.ndarray            # (d_model,)
    w_discretize: jnp.ndarray   # (d_model, num_actions) discretization classifier head
    b_discretize: jnp.ndarray   # (num_actions,)


def _glorot(key: jax.random.PRNGKey, in_dim: int, out_dim: int) -> jnp.ndarray:
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    return jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)


def init_shared_stage_params(
    rng_key: jax.random.PRNGKey, d_model: int = 512, num_actions: int = 16,
) -> SharedStageParameters:
    keys = jax.random.split(rng_key, 4)
    return SharedStageParameters(
        w_k=_glorot(keys[0], d_model, d_model), b_k=jnp.zeros((d_model,)),
        w_goal=_glorot(keys[1], d_model, d_model), b_goal=jnp.zeros((d_model,)),
        w_z=_glorot(keys[2], d_model, d_model), b_z=jnp.zeros((d_model,)),
        w_discretize=_glorot(keys[3], d_model, num_actions), b_discretize=jnp.zeros((num_actions,)),
    )


def scaled_dot_product_attention(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, num_heads: int, bias: jnp.ndarray = None,
) -> jnp.ndarray:
    """Manual multi-head attention, matching the style of
    src/model/transformer_decision_core.py's forward_layer (kept dependency-free
    of flax.linen rather than porting the notebook's `nn.dot_product_attention`
    calls verbatim)."""
    seq_q, d_model = q.shape
    seq_k = k.shape[0]
    head_dim = d_model // num_heads
    qh = q.reshape(seq_q, num_heads, head_dim).transpose(1, 0, 2)
    kh = k.reshape(seq_k, num_heads, head_dim).transpose(1, 0, 2)
    vh = v.reshape(seq_k, num_heads, head_dim).transpose(1, 0, 2)
    scale = 1.0 / jnp.sqrt(head_dim)
    scores = jnp.matmul(qh, jnp.transpose(kh, (0, 2, 1))) * scale
    if bias is not None:
        scores = scores + bias[None, :, :]
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(weights, vh)
    return out.transpose(1, 0, 2).reshape(seq_q, d_model)


def self_attention_now_goal_actions(
    state_tok: jnp.ndarray, target_tok: jnp.ndarray, actions_tok: jnp.ndarray, num_heads: int,
) -> jnp.ndarray:
    """Stage 1 (IdealStructurePipeline.self_attention_now_goal_actions, cell 18):
    'now' (state), 'goal' (target), and 'actions' jointly cross-attend, so every
    element can see every other element."""
    q = jnp.concatenate([state_tok, target_tok, actions_tok], axis=0)
    return scaled_dot_product_attention(q, q, q, num_heads)


def physical_key_filter(
    query_state: jnp.ndarray, candidates: jnp.ndarray, w_k: jnp.ndarray, b_k: jnp.ndarray,
    num_heads: int, physically_possible: jnp.ndarray,
) -> jnp.ndarray:
    """Stage 2 (cell 18): project `candidates` through a learned key weight W_K,
    then attend to them from `query_state`. `physically_possible` (bool, shape
    (q_length, kv_length)) becomes an additive -1e9 bias, driving softmax weight
    to zero at any (query, key) pair marked False."""
    key = jnp.matmul(candidates, w_k) + b_k
    bias = jnp.where(physically_possible, 0.0, -1e9)
    return scaled_dot_product_attention(query_state, key, candidates, num_heads, bias=bias)


def candidate_survival_mask(physically_possible: jnp.ndarray) -> jnp.ndarray:
    """A candidate survives if it is physically possible from at least one
    query row. Fixed candidate count, disallowed rows zeroed / scored -inf
    downstream instead of merely down-weighted (cell 18)."""
    return jnp.any(physically_possible, axis=0)


def predict_next_goal(filtered_state: jnp.ndarray, w_goal: jnp.ndarray, b_goal: jnp.ndarray) -> jnp.ndarray:
    """Stage 3 (cell 18): pool the filtered now/goal representation and pass
    through one dense layer, compressing to a single (1, d) predicted next goal S'."""
    pooled = jnp.mean(filtered_state, axis=0, keepdims=True)
    return jnp.matmul(pooled, w_goal) + b_goal


def compress_step_to_history(
    candidates: jnp.ndarray, w_z: jnp.ndarray, b_z: jnp.ndarray, history_buffer: jnp.ndarray,
) -> jnp.ndarray:
    """Stage 6 (cell 18): compress this step's (already physically-filtered)
    candidates into a single (1, d) vector and append it to the running
    history buffer H."""
    pooled = jnp.mean(candidates, axis=0, keepdims=True)
    step_repr = jnp.matmul(pooled, w_z) + b_z
    return jnp.concatenate([history_buffer, step_repr], axis=0)


def compute_physical_feasibility(state: SystemState, actions: ActionsData) -> jnp.ndarray:
    """Real, derived `physically_possible` grid for `physical_key_filter` /
    `candidate_survival_mask`: an action is physically feasible when it is
    marked valid AND every cost dimension it consumes fits within the state's
    currently available budget. This is computed from real ActionsData /
    SystemState fields (never synthetic), shape (2, num_actions) -- one query
    row each for 'now' (state) and 'goal' (target), both using the same
    feasibility-from-current-budget test."""
    num_costs = actions.costs.shape[-1]
    budget = state.available_costs[:num_costs]
    cost_ok = jnp.all(actions.costs <= budget[None, :], axis=-1)
    feasible = jnp.logical_and(actions.valid_mask, cost_ok)  # (num_actions,)
    return jnp.broadcast_to(feasible[None, :], (2, feasible.shape[0]))


def discretize_embeddings_to_ids(x: jnp.ndarray, w_discretize: jnp.ndarray, b_discretize: jnp.ndarray) -> jnp.ndarray:
    """Documented replacement for `TensorOps.embeddings_to_scalar_id` (see
    module docstring): a learned linear classifier head producing a discrete
    id per row, instead of a mean-pool that collapses every row to the same
    scalar."""
    logits = jnp.matmul(x, w_discretize) + b_discretize
    return jnp.argmax(logits, axis=-1)


def build_mdp_transition_array(
    state_ids: jnp.ndarray, next_ids: jnp.ndarray, action_ids: jnp.ndarray,
    p_coe: float, r_coe: float, gamma_coe: float, stable: bool = False,
) -> jnp.ndarray:
    """MDPSolver._build_transition_array_single (cell 8), unchanged, operating
    on already-discretized integer-valued ids (see discretize_embeddings_to_ids)
    instead of the notebook's raw column_stack of embedding blocks."""
    state = state_ids.astype(jnp.float32)
    next_state = next_ids.astype(jnp.float32)
    actions = action_ids.astype(jnp.float32)
    # eps is deliberately not tiny (1e-3, not 1e-8): state/next_state are small
    # integer discretization ids (0..num_actions-1), so a 1e-8 floor let
    # `gradient` reach ~1e11 whenever next_state discretized to 0, which
    # overflowed float32 to +/-inf during solve_bellman_topk's value iteration
    # and produced `inf - inf = NaN` Q-values -- observed as a 100%-NaN
    # predicted_next_state on real Kaggle GPU verification (task
    # TASK-20260925-002-candidates-smoke-verify, mdp_branch). Clipping the
    # gradient bounds P/R/G to a finite, still-large-relative-to-normal range
    # without changing behavior when next_state is not near zero.
    eps = jnp.asarray(1e-3, dtype=jnp.float32)
    sign = jnp.where(next_state >= 0, 1.0, -1.0)
    safe_next_state = jnp.where(jnp.abs(next_state) < eps, sign * eps, next_state)
    gradient = jnp.clip(state / safe_next_state * 100.0, -1e4, 1e4)
    P = p_coe * gradient
    R = r_coe * gradient
    G = gamma_coe * gradient
    if stable:
        # Root-cause fix for NaN (see docs/experiments/2026-09-26_rerun_verification): P*G multiplies Q every
        # value-iteration step; with G ~ gamma*gradient (up to 9e3) the recursion is not a contraction and Q
        # overflows float32 -> inf -> NaN. Make it a genuine discounted MDP: P in [0,1] (row-normalised in
        # solve_bellman_topk), discount G in [0, gamma_coe].
        P = jnp.clip(P, 0.0, 1.0)
        G = jnp.clip(G, 0.0, gamma_coe)
    return jnp.column_stack([state, next_state, actions, P, R, G])


def solve_bellman_topk(array_to_action: jnp.ndarray, k: int, n: int, stable: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """MDPSolver._solve_single (cell 8), unchanged Bellman-optimality value
    iteration:
        Q*(s,a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma max_a' Q*(s',a')]
    `n` is the static number of distinct discretized ids (num_actions here),
    sizing the Q-table and bounding the iteration count. `k` must be a
    concrete Python int (jax.lax.top_k requires it)."""
    S = jnp.clip(array_to_action[:, 0].astype(jnp.int32), 0, n - 1)
    S_1 = jnp.clip(array_to_action[:, 1].astype(jnp.int32), 0, n - 1)
    A = jnp.clip(array_to_action[:, 2].astype(jnp.int32), 0, n - 1)
    P = array_to_action[:, 3]
    R = array_to_action[:, 4]
    G = array_to_action[:, 5]

    state_action_id = S * n + A
    if stable:  # normalise P per (s,a) so the backup is a convex combination (contraction with G<=gamma<1)
        mass = jax.ops.segment_sum(P, state_action_id, num_segments=n * n)
        P = P / jnp.maximum(mass[state_action_id], 1.0)
    Q0 = jnp.zeros((n, n), dtype=array_to_action.dtype)

    def cond_fn(carry):
        it, _, delta = carry
        return jnp.logical_and(it < n, delta > 1e-6)

    def body_fn(carry):
        it, Q, _ = carry
        topk_next = jax.lax.top_k(Q[S_1], k)[0]
        soft_max_next = jnp.mean(topk_next, axis=-1)
        transition_value = P * (R + G * soft_max_next)
        Q_flat = jax.ops.segment_sum(transition_value, state_action_id, num_segments=n * n)
        Q_new = Q_flat.reshape(n, n)
        delta = jnp.max(jnp.abs(Q_new - Q))
        return (it + 1, Q_new, delta)

    init_carry = (0, Q0, jnp.asarray(jnp.inf, dtype=array_to_action.dtype))
    _, Q, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    top_values, top_actions = jax.lax.top_k(Q, k)
    return top_actions, top_values


def mdp_topk_to_action_logits(top_actions: jnp.ndarray, top_values: jnp.ndarray, num_actions: int) -> jnp.ndarray:
    """Scatter the MDP branch's top-k (action_id, Q-value) pairs into a full
    per-candidate logits vector (-1e9 elsewhere), so an MDP-search candidate
    plugs into the same softmax action-selection / cross-entropy training path
    as the attention-score candidates. `top_actions` are ids in [0, num_actions)
    from discretize_embeddings_to_ids, so they double as action indices."""
    logits = jnp.full((num_actions,), -1e9, dtype=top_values.dtype)
    flat_actions = jnp.reshape(top_actions, (-1,))
    flat_values = jnp.reshape(top_values, (-1,))
    return logits.at[flat_actions].max(flat_values)


class DecisionHeadParameters(NamedTuple):
    """The non-action-selection part of DecisionVectorD, shared across every
    candidate module so action selection (MDP search / attention-score
    ranking / dense classification) stays the only place they differ."""
    w_cost: jnp.ndarray
    b_cost: jnp.ndarray
    w_next_state: jnp.ndarray
    b_next_state: jnp.ndarray
    w_progress: jnp.ndarray
    b_progress: jnp.ndarray
    w_validity: jnp.ndarray
    b_validity: jnp.ndarray


def init_decision_head_params(
    rng_key: jax.random.PRNGKey, d_model: int, num_costs: int, num_resources: int,
) -> DecisionHeadParameters:
    keys = jax.random.split(rng_key, 4)
    return DecisionHeadParameters(
        w_cost=_glorot(keys[0], d_model, num_costs), b_cost=jnp.zeros((num_costs,)),
        w_next_state=_glorot(keys[1], d_model, num_resources), b_next_state=jnp.zeros((num_resources,)),
        w_progress=_glorot(keys[2], d_model, 1), b_progress=jnp.zeros((1,)),
        w_validity=_glorot(keys[3], d_model, 1), b_validity=jnp.zeros((1,)),
    )


def apply_decision_heads(
    heads: DecisionHeadParameters, pooled_context: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """pooled_context: (d_model,). Returns (estimated_costs, predicted_next_state,
    progress_rate_pred, validity_score) for DecisionVectorD."""
    costs = jnp.matmul(pooled_context, heads.w_cost) + heads.b_cost
    next_state = jnp.matmul(pooled_context, heads.w_next_state) + heads.b_next_state
    progress = jax.nn.sigmoid(jnp.matmul(pooled_context, heads.w_progress) + heads.b_progress)[0]
    validity = jax.nn.sigmoid(jnp.matmul(pooled_context, heads.w_validity) + heads.b_validity)[0]
    return costs, next_state, progress, validity


def split_encoded_tokens(tokens: jnp.ndarray, num_actions: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split encode_channel_independent's [State, Target, Actions..., History...]
    concatenation back into (state_tok, target_tok, actions_tok, hist_tok)."""
    state_tok = tokens[0:1]
    target_tok = tokens[1:2]
    actions_tok = tokens[2:2 + num_actions]
    hist_tok = tokens[2 + num_actions:]
    return state_tok, target_tok, actions_tok, hist_tok
