"""Factored action-conditional world model used to forecast the state at the moment a plan starts executing.

    s_{i+1} = s_i + dt * rate_sd * [ f(x_i) + e(x_i, a_i) ],   e(x, a) = h(x, onehot(a)) - h(x, onehot(a_ref)),
    x = (s - mu) / sd

f is the dynamics under the reference continuation: it is identified because e(x, a_ref) == 0 exactly, so f can be
pretrained on reference-action ("do nothing") data and e is learned from active data.
Design notes (see docs/experiments/2026-09-28_latency_chunking/DESIGN.md):
  * For DISCRETE actions this factorisation is a reparameterisation, not a restriction: any F(s, a) can be written as
    F(s, a_ref) + [F(s, a) - F(s, a_ref)]. Its only content is the choice of a_ref and the option to pretrain f.
    (For continuous actions a control-affine g(s)(u - u_ref) would be a real inductive bias; no continuous env here.)
  * dt is an explicit input (Euler step of a learned rate), so irregular tick lengths are representable.
  * Deterministic and trained with MSE -> it predicts the conditional mean; under exogenous noise the forecast is not a
    sample, and it carries no uncertainty. Markov in the observation (no history input): adequate for pendulum_goal and
    intercept, NOT for Craftax's partial view.
  * Output layers are zero-initialised, so an untrained model is exactly the persistence forecast s_{t+delta} = s_t.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class WorldModelStats(NamedTuple):
    mu: jnp.ndarray       # (D,) observation mean
    sd: jnp.ndarray       # (D,) observation std (floored)
    rate_sd: jnp.ndarray  # (D,) std of (s' - s) / dt (0 for constant dims -> those are predicted exactly as persistent)


def identity_stats(obs_dim: int) -> WorldModelStats:
    return WorldModelStats(jnp.zeros((obs_dim,)), jnp.ones((obs_dim,)), jnp.ones((obs_dim,)))


def compute_world_model_stats(obs, next_obs, valid, dt: float) -> WorldModelStats:
    """obs/next_obs (..., D), valid (...,) bool. Statistics over valid transitions only."""
    D = obs.shape[-1]
    o, n, w = obs.reshape(-1, D), next_obs.reshape(-1, D), valid.reshape(-1).astype(jnp.float32)
    cnt = jnp.maximum(w.sum(), 1.0)
    mean = lambda x: (x * w[:, None]).sum(0) / cnt
    mu = mean(o)
    sd = jnp.sqrt(mean((o - mu) ** 2))
    r = (n - o) / dt
    rate_sd = jnp.sqrt(mean((r - mean(r)) ** 2))
    return WorldModelStats(mu, jnp.maximum(sd, 1e-3), rate_sd)


def _mlp_init(key, sizes, last_scale):
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for i, (k, a, b) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        scale = last_scale if i == len(sizes) - 2 else 1.0
        layers.append({"w": jax.random.normal(k, (a, b)) * scale / jnp.sqrt(max(a, 1)), "b": jnp.zeros((b,))})
    return layers


def _mlp(layers, x):
    for layer in layers[:-1]:
        x = jax.nn.gelu(x @ layer["w"] + layer["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def init_world_model_parameters(key, obs_dim: int, num_actions: int, hidden: int = 256):
    kf, kh = jax.random.split(key)
    return {"f": _mlp_init(kf, [obs_dim, hidden, hidden, obs_dim], 0.0),
            "h": _mlp_init(kh, [obs_dim + num_actions, hidden, hidden, obs_dim], 0.0)}


def forward_world_model_step(params, stats: WorldModelStats, s, a, ref_action: int, dt: float,
                             use_effect: bool = True, train_f: bool = True):
    """One tick. use_effect=False drops e (pure reference-continuation forecast, the 'wm_passive' arm)."""
    f_params = params["f"] if train_f else jax.lax.stop_gradient(params["f"])
    x = (s - stats.mu) / stats.sd
    rate = _mlp(f_params, x)
    if use_effect:
        A = params["h"][0]["w"].shape[0] - s.shape[-1]
        h = lambda act: _mlp(params["h"], jnp.concatenate([x, jax.nn.one_hot(act, A)]))
        rate = rate + h(a) - h(ref_action)
    return s + dt * rate * stats.rate_sd


def forward_world_model_rollout(params, stats, s0, actions, ref_action: int, dt: float,
                                use_effect: bool = True, train_f: bool = True):
    """s0 (D,), actions (L,) -> predicted states s_1..s_L, shape (L, D)."""
    def body(s, a):
        s2 = forward_world_model_step(params, stats, s, a, ref_action, dt, use_effect, train_f)
        return s2, s2
    return jax.lax.scan(body, s0, actions)[1]


def forecast_state(params, stats, s0, prefix, ref_action: int, dt: float, use_effect: bool = True):
    """Forecast of the observation after the committed prefix has executed (len(prefix) ticks)."""
    if prefix.shape[0] == 0:
        return s0
    return forward_world_model_rollout(params, stats, s0, prefix, ref_action, dt, use_effect)[-1]


def _error_scale(stats, dt):
    return stats.rate_sd * dt + 1e-3


def world_model_loss(params, stats, obs_w, act_w, valid, ref_action: int, dt: float,
                     use_effect: bool = True, train_f: bool = True):
    """Multi-step (open-loop, teacher actions) loss. obs_w (B, L+1, D), act_w (B, L), valid (B,) bool."""
    roll = lambda s0, a: forward_world_model_rollout(params, stats, s0, a, ref_action, dt, use_effect, train_f)
    pred = jax.vmap(roll)(obs_w[:, 0], act_w)
    err = ((pred - obs_w[:, 1:]) / _error_scale(stats, dt)) ** 2
    per = err.mean(axis=(1, 2))
    w = valid.astype(jnp.float32)
    return (per * w).sum() / jnp.maximum(w.sum(), 1.0)


def world_model_skill(params, stats, obs_w, act_w, valid, ref_action: int, dt: float, use_effect: bool = True):
    """Per-horizon skill vs persistence: 1 - MSE(model) / MSE(s_t held constant), shape (L,)."""
    roll = lambda s0, a: forward_world_model_rollout(params, stats, s0, a, ref_action, dt, use_effect)
    pred = jax.vmap(roll)(obs_w[:, 0], act_w)
    tgt = obs_w[:, 1:]
    sc = _error_scale(stats, dt)
    w = valid.astype(jnp.float32)[:, None]
    mse = lambda p: ((((p - tgt) / sc) ** 2).mean(-1) * w).sum(0) / jnp.maximum(w.sum(), 1.0)
    return 1.0 - mse(pred) / jnp.maximum(mse(obs_w[:, :1]), 1e-12)


def extract_windows(key, obs_before, next_obs, act, alive, end, length: int, n_windows: int):
    """Sample training windows from per-tick streams (N, T, ...). A window of `length` ticks is valid iff every tick was
    executed and no episode ended before its last tick. Returns obs_w (n, L+1, D), act_w (n, L), valid (n,)."""
    N, T = act.shape
    if T < length:
        raise ValueError(f"stream of {T} ticks is shorter than the window length {length}")
    ke, ks = jax.random.split(key)
    env_i = jax.random.randint(ke, (n_windows,), 0, N)
    start = jax.random.randint(ks, (n_windows,), 0, T - length + 1)

    def one(n, s):
        sl = lambda x: jax.lax.dynamic_slice_in_dim(x[n], s, length, axis=0)
        ob = jnp.concatenate([jax.lax.dynamic_slice_in_dim(obs_before[n], s, 1, axis=0), sl(next_obs)], axis=0)
        ok = sl(alive).all() & ~sl(end)[:-1].any()
        return ob, sl(act), ok

    return jax.vmap(one)(env_i, start)
