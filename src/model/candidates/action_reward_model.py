"""Learned one-step action-reward model (deployable stand-in for the `oracle_act` leak; arc-proposals TASK-20260929-016).

oracle_act hands the policy the TRUE 1-step reward of every action (simulator branching at policy time = answer leakage). This model is
trained OFFLINE on simulator-branched labels (like passive_wm_full: labels are used only as training data) and at policy time maps the
raw 1345-d symbolic observation to 17 predicted rewards -- no simulator access at policy time, so it is deployable.
"""
import jax
import jax.numpy as jnp

N_ACT = 17


def init_params(key, in_dim: int, hidden: int = 512, n_act: int = N_ACT):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(
        w1=jax.random.normal(k1, (in_dim, hidden)) / jnp.sqrt(in_dim), b1=jnp.zeros((hidden,)),
        w2=jax.random.normal(k2, (hidden, hidden)) / jnp.sqrt(hidden), b2=jnp.zeros((hidden,)),
        w3=jax.random.normal(k3, (hidden, n_act)) * 0.01, b3=jnp.zeros((n_act,)),
        mu=jnp.zeros((in_dim,)), sd=jnp.ones((in_dim,)),
    )


def predict(params, x):
    """x (in_dim,) raw observation -> (n_act,) predicted 1-step reward per action (reward units)."""
    h = (x - params["mu"]) / params["sd"]
    h = jax.nn.gelu(h @ params["w1"] + params["b1"])
    h = jax.nn.gelu(h @ params["w2"] + params["b2"])
    return h @ params["w3"] + params["b3"]
