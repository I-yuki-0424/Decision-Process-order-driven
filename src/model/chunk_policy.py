"""Autoregressive action-chunk generator with continuous-time token embeddings, plus the PPO critic.

Token layout (one sequence per decision):
    [goal] [state] [hist_1 .. hist_M] [prefix_1 .. prefix_P] | [bos] [plan_0 .. plan_{k-2}]
    '------------ context: bidirectional -----------------'   '--- causal; attends to all context ---'
The output at plan position j predicts plan_j, so the chunk is a JOINT distribution  p(a_0..a_{k-1}) = prod_j
p(a_j | context, a_<j). Independent per-step heads would mix modes across steps (the non-autoregressive
multimodality problem); an autoregressive factorisation cannot.
Every token carries a sinusoidal embedding of its time in seconds relative to the moment plan_0 executes:
state at -state_age, history at -obs_age - (M - i) dt, committed prefix at -obs_age + i dt, plan_j at +j dt.
(For forecast arms state_age = 0 but obs_age = delta*dt: the history is older than the forecast state.)
With a fixed delta these offsets are constants; they matter once latency varies (not modelled yet).
Context tokens never attend to plan tokens, so the context encoding (used by the temporal-distance head) is identical
with or without the plan.

Also here: the MLP actor used only for the plain-PPO calibration run (k = 1), and the critic MLP.
"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

GOAL, STATE, HIST, PREFIX, BOS, PLAN = range(6)


class ActorInput(NamedTuple):
    state: jnp.ndarray       # (D,) observation (ignore/augment) or forecast of the state at plan start (wm/oracle)
    goal: jnp.ndarray        # (G,)
    hist_obs: jnp.ndarray    # (M, D)
    hist_act: jnp.ndarray    # (M,) int
    hist_valid: jnp.ndarray  # (M,) bool
    prefix: jnp.ndarray      # (P,) int committed actions (augment arm only; P = 0 otherwise)
    state_age: jnp.ndarray   # () seconds from the state token's time to plan start (0 for a forecast at plan start)
    obs_age: jnp.ndarray     # () seconds from the snapshot o_c (history/prefix reference) to plan start


class ChunkPolicyConfig(NamedTuple):
    obs_dim: int
    goal_dim: int
    num_actions: int
    k: int
    dt: float
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_time: int = 8          # sinusoid pairs; timescales geometric from 1 to 1000 ticks
    mlp_ratio: int = 4


def _dense(key, i, o, scale=1.0):
    return {"w": jax.random.normal(key, (i, o)) * scale / jnp.sqrt(max(i, 1)), "b": jnp.zeros((o,))}


def _apply(p, x):
    return x @ p["w"] + p["b"]


def _ln(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * g + b


def init_chunk_policy_parameters(key, cfg: ChunkPolicyConfig):
    d, r = cfg.d_model, cfg.mlp_ratio
    ks = iter(jax.random.split(key, 8 + 4 * cfg.n_layers))
    blocks = []
    for _ in range(cfg.n_layers):
        blocks.append({
            "ln1_g": jnp.ones((d,)), "ln1_b": jnp.zeros((d,)),
            "qkv": _dense(next(ks), d, 3 * d), "o": _dense(next(ks), d, d, 0.5),
            "ln2_g": jnp.ones((d,)), "ln2_b": jnp.zeros((d,)),
            "fc1": _dense(next(ks), d, r * d), "fc2": _dense(next(ks), r * d, d, 0.5),
        })
    return {
        "obs_proj": _dense(next(ks), cfg.obs_dim, d),
        "goal_proj": _dense(next(ks), cfg.goal_dim, d),
        "act_emb": jax.random.normal(next(ks), (cfg.num_actions, d)) * 0.02,
        "bos": jax.random.normal(next(ks), (d,)) * 0.02,
        "type_emb": jax.random.normal(next(ks), (6, d)) * 0.02,
        "time_proj": _dense(next(ks), 2 * cfg.n_time, d, 0.1),
        "blocks": blocks,
        "lnf_g": jnp.ones((d,)), "lnf_b": jnp.zeros((d,)),
        "head": _dense(next(ks), d, cfg.num_actions, 0.01),   # near-uniform initial policy
        "dist": [_dense(next(ks), d, d), _dense(jax.random.fold_in(key, 7), d, 1, 0.1)],
    }


def _time_features(cfg, t_seconds):
    ticks = t_seconds / cfg.dt
    scales = 1000.0 ** (jnp.arange(cfg.n_time) / max(cfg.n_time - 1, 1))
    z = ticks[:, None] / scales[None, :]
    return jnp.concatenate([jnp.sin(z), jnp.cos(z)], axis=-1)


def _context(params, cfg, ai: ActorInput):
    M, P = ai.hist_obs.shape[0], ai.prefix.shape[0]
    age, oage = ai.state_age, ai.obs_age
    tok = jnp.concatenate([
        _apply(params["goal_proj"], ai.goal)[None],
        _apply(params["obs_proj"], ai.state)[None],
        _apply(params["obs_proj"], ai.hist_obs) + params["act_emb"][ai.hist_act],
        params["act_emb"][ai.prefix],
    ], axis=0)
    types = jnp.array([GOAL, STATE] + [HIST] * M + [PREFIX] * P)
    times = jnp.concatenate([jnp.zeros((1,)), -age[None],
                             -oage - (M - jnp.arange(M)) * cfg.dt, -oage + jnp.arange(P) * cfg.dt])
    valid = jnp.concatenate([jnp.ones((2,), jnp.bool_), ai.hist_valid, jnp.ones((P,), jnp.bool_)])
    return tok, types, times, valid


def _plan(params, cfg, plan_actions):
    k = cfg.k
    tok = jnp.concatenate([params["bos"][None], params["act_emb"][plan_actions[: k - 1]]], axis=0)
    types = jnp.array([BOS] + [PLAN] * (k - 1))
    times = jnp.arange(k) * cfg.dt   # time of the action predicted at this position
    return tok, types, times


def _encode(params, cfg, tok, types, times, mask):
    x = tok + params["type_emb"][types] + _apply(params["time_proj"], _time_features(cfg, times))
    L, d = x.shape
    H = cfg.n_heads
    for b in params["blocks"]:
        h = _ln(x, b["ln1_g"], b["ln1_b"])
        q, kk, v = jnp.split(_apply(b["qkv"], h), 3, axis=-1)
        split = lambda z: z.reshape(L, H, d // H).transpose(1, 0, 2)
        q, kk, v = split(q), split(kk), split(v)
        att = jnp.einsum("hqd,hkd->hqk", q, kk) / jnp.sqrt(d // H)
        att = jax.nn.softmax(jnp.where(mask[None], att, -1e9), axis=-1)
        out = jnp.einsum("hqk,hkd->hqd", att, v).transpose(1, 0, 2).reshape(L, d)
        x = x + _apply(b["o"], out)
        h = _ln(x, b["ln2_g"], b["ln2_b"])
        x = x + _apply(b["fc2"], jax.nn.gelu(_apply(b["fc1"], h)))
    return _ln(x, params["lnf_g"], params["lnf_b"])


def build_prefix_lm_mask(ctx_valid, k: int):
    """Context rows see valid context keys only; plan rows see valid context keys + plan keys up to themselves."""
    C = ctx_valid.shape[0]
    L = C + k
    q, kk = jnp.arange(L)[:, None], jnp.arange(L)[None, :]
    ctx_key = kk < C
    allowed = jnp.where(q < C, ctx_key, ctx_key | ((kk >= C) & (kk <= q)))
    key_valid = jnp.concatenate([ctx_valid, jnp.ones((k,), jnp.bool_)])
    return allowed & key_valid[None, :]


def forward_chunk_policy(params, cfg: ChunkPolicyConfig, ai: ActorInput, plan_actions):
    """Teacher-forced logits (k, A) for the chunk `plan_actions` (k,)."""
    ct, cty, cti, cv = _context(params, cfg, ai)
    pt, pty, pti = _plan(params, cfg, plan_actions)
    h = _encode(params, cfg, jnp.concatenate([ct, pt]), jnp.concatenate([cty, pty]),
                jnp.concatenate([cti, pti]), build_prefix_lm_mask(cv, cfg.k))
    return _apply(params["head"], h[ct.shape[0]:])


def forward_temporal_distance(params, cfg: ChunkPolicyConfig, ai: ActorInput):
    """Predicted log(1 + ticks) from the actor's state to reaching ai.goal (context-only pass)."""
    ct, cty, cti, cv = _context(params, cfg, ai)
    C = ct.shape[0]
    mask = jnp.broadcast_to(cv[None, :], (C, C))
    h = _encode(params, cfg, ct, cty, cti, mask)[1]   # state token
    return _apply(params["dist"][1], jax.nn.gelu(_apply(params["dist"][0], h)))[0]


def sample_chunk(logits_fn: Callable, params, ai: ActorInput, key, k: int, greedy: bool = False):
    """Autoregressive sampling (k sequential passes). Returns actions (k,) int32 and per-token log-probs (k,)."""
    def body(buf, x):
        i, kk = x
        logp = jax.nn.log_softmax(logits_fn(params, ai, buf)[i])
        a = jnp.argmax(logp) if greedy else jax.random.categorical(kk, logp)
        return buf.at[i].set(a), (a, logp[a])
    _, (acts, logps) = jax.lax.scan(body, jnp.zeros((k,), jnp.int32), (jnp.arange(k), jax.random.split(key, k)))
    return acts.astype(jnp.int32), logps


def chunk_log_probs(logits_fn: Callable, params, ai: ActorInput, actions):
    """Teacher-forced per-token log-probs (k,) and entropies (k,)."""
    logp_all = jax.nn.log_softmax(logits_fn(params, ai, actions))
    logp = jnp.take_along_axis(logp_all, actions[:, None], axis=-1)[:, 0]
    ent = -(jnp.exp(logp_all) * logp_all).sum(-1)
    return logp, ent


# ----------------------------------------------------------------------------------------------------------------------
# MLP actor (calibration only) and critic -- orthogonal init as in PureJaxRL
# ----------------------------------------------------------------------------------------------------------------------

def _ortho_mlp(key, sizes, last_scale):
    keys = jax.random.split(key, len(sizes) - 1)
    out = []
    for i, (k, a, b) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        s = last_scale if i == len(sizes) - 2 else jnp.sqrt(2.0)
        out.append({"w": jax.nn.initializers.orthogonal(s)(k, (a, b)), "b": jnp.zeros((b,))})
    return out


def _tanh_mlp(layers, x):
    for p in layers[:-1]:
        x = jnp.tanh(_apply(p, x))
    return _apply(layers[-1], x)


def mlp_actor_input_dim(obs_dim, goal_dim, prefix_len, num_actions):
    return obs_dim + goal_dim + prefix_len * num_actions


def init_mlp_actor_parameters(key, in_dim: int, num_actions: int, width: int = 512, n_layers: int = 2):
    return _ortho_mlp(key, [in_dim] + [width] * n_layers + [num_actions], 0.01)


def forward_mlp_actor(params, ai: ActorInput, plan_actions, num_actions: int):
    """k = 1 only. Uses state, goal and one-hot prefix; ignores history."""
    x = jnp.concatenate([ai.state, ai.goal, jax.nn.one_hot(ai.prefix, num_actions).reshape(-1)])
    return _tanh_mlp(params, x)[None]


def init_critic_parameters(key, in_dim: int, width: int = 256, n_layers: int = 2):
    return _ortho_mlp(key, [in_dim] + [width] * n_layers + [1], 1.0)


def forward_critic(params, x):
    return _tanh_mlp(params, x)[0]
