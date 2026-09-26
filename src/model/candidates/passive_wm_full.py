"""Passive ("do nothing") world model v2: full observation in, multi-horizon future summary change out.

Why v2: the TASK-004 model predicted only the 4 vitals under noop. That target is a near-deterministic function of the
current vitals (already policy input), so it carried no new information. Here the target is the noop change of the whole
39-dim observation summary (vitals, inventory, light, tile counts, mob counts) at several horizons, from the FULL 1345-d
symbolic observation (mob/map positions included). Targets come from simulator noop branches, used only as offline
training data; nothing here touches the simulator at policy time.
"""
import jax
import jax.numpy as jnp

HORIZONS = (1, 2, 4, 8)
F = 39  # obs_to_features dim


def init_params(key, in_dim: int, hidden: int = 512, out_f: int = F, n_h: int = len(HORIZONS)):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(
        w1=jax.random.normal(k1, (in_dim, hidden)) / jnp.sqrt(in_dim), b1=jnp.zeros((hidden,)),
        w2=jax.random.normal(k2, (hidden, hidden)) / jnp.sqrt(hidden), b2=jnp.zeros((hidden,)),
        w3=jax.random.normal(k3, (hidden, out_f * n_h)) * 0.01, b3=jnp.zeros((out_f * n_h,)),
        mu=jnp.zeros((in_dim,)), sd=jnp.ones((in_dim,)),          # input standardisation
        dsd=jnp.ones((n_h, out_f)),                                # per-horizon delta std (target scale)
    )


def predict_std_delta(params, x):
    """x (in_dim,) -> predicted delta of summary features, in units of delta-std, shape (n_h, F)."""
    h = (x - params["mu"]) / params["sd"]
    h = jax.nn.gelu(h @ params["w1"] + params["b1"])
    h = jax.nn.gelu(h @ params["w2"] + params["b2"])
    return (h @ params["w3"] + params["b3"]).reshape(len(HORIZONS), -1)


def policy_features(std_delta):
    """(n_h, F) standardised deltas -> bounded policy features for horizons 2 and 8 (78 dims)."""
    return jnp.clip(jnp.concatenate([std_delta[1], std_delta[3]]), -5.0, 5.0)
