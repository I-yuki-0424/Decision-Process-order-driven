"""Idea-6 world model specialised to Craftax "do-nothing" (noop) dynamics of the vitals.

Spec (operator, 2026-09-25): 1) train the world model to predict what happens if NOTHING is done; 2) then train
the action-selecting Transformer with the (frozen) world model. Here:
  q = standardised vitals (health, food, drink, energy) from the real observation
  p = learned momentum  Enc(features)  (the "canonical coordinates" gap from DPOD.ipynb open issue 1)
  wind = learned exogenous code W(features)   (map/inventory/mobs summary, held constant over the rollout)
  'hn'  : WorldModel.step / rollout from world_model.py UNCHANGED (K-mode port-Hamiltonian, gate, residual);
          FormulaPresets are all masked OFF -- gravity/drag/buoyancy/ground contact have no counterpart in a
          Craftax vitals model and copying Craftax's decay rules into presets would be oracle leakage.
  'mlp' : same encoder, generic dx = MLP([q,p,wind]), Euler integration (A/B control for the Hamiltonian prior).
  persistence: q_{t+k} = q_t (reference, no parameters).
Targets are real noop transitions of the simulator, used ONLY as offline training data; no simulator access at
policy time.
"""
import jax
import jax.numpy as jnp

from src.model.candidates.world_model import FormulaPresets, WorldModel

N = 4  # vitals
HORIZON = 8
ETA = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # m=1, everything else 0 (no presets)
MASK = jnp.zeros((2, FormulaPresets.N_POT + FormulaPresets.N_FORCE), dtype=bool)


def _lin(key, i, o, s=None):
    return dict(w=jax.random.normal(key, (i, o)) * (s or (1.0 / jnp.sqrt(i))), b=jnp.zeros((o,)))


def _mlp(p, x):
    return jnp.tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def _mlp_init(key, i, h, o, s2=0.1):
    k1, k2 = jax.random.split(key)
    return dict(w1=jax.random.normal(k1, (i, h)) / jnp.sqrt(i), b1=jnp.zeros((h,)),
                w2=jax.random.normal(k2, (h, o)) * s2, b2=jnp.zeros((o,)))


def init_params(key, kind: str, feat_dim: int, hidden: int = 64, modes: int = 2):
    k = jax.random.split(key, 4)
    enc = dict(p=_mlp_init(k[0], feat_dim, hidden, N, 0.3), wind=_mlp_init(k[1], feat_dim, hidden, N, 0.3),
               mu=jnp.zeros((feat_dim,)), sd=jnp.ones((feat_dim,)))
    if kind == "hn":
        core = WorldModel.init_parameters(k[2], N, 1, N, modes, hidden)
    elif kind == "mlp":
        core = _mlp_init(k[2], 3 * N, 2 * hidden, 2 * N, 0.01)
    else:
        raise ValueError(kind)
    return dict(enc=enc, core=core)


def _encode(params, feats):
    f = (feats - params["enc"]["mu"]) / params["enc"]["sd"]
    q = f[:N]  # first N features are the vitals (obs_to_features layout), standardised
    p = _mlp(params["enc"]["p"], f)
    wind = _mlp(params["enc"]["wind"], f)
    return jnp.concatenate([q, p]), wind


def predict(params, kind: str, feats, horizon: int = HORIZON):
    """feats (F,) -> predicted standardised vitals q_{t+1..t+horizon}, shape (horizon, N). Open-loop."""
    x0, wind = _encode(params, feats)
    if kind == "hn":
        traj, _ = WorldModel.rollout(params["core"], x0, jnp.zeros((horizon, 1)),
                                     jnp.tile(wind, (horizon, 1)), ETA, MASK, 1.0)
        return traj[:, :N]

    def body(x, _):
        dx = _mlp(params["core"], jnp.concatenate([x, wind]))
        xn = x + dx
        return xn, xn[:N]
    _, q = jax.lax.scan(body, x0, None, length=horizon)
    return q


def anticipation_features(params, kind: str, feats):
    """Policy-side features: predicted passive change of vitals at +1 and +HORIZON steps (standardised units)."""
    q = predict(params, kind, feats)
    q0 = (feats[:N] - params["enc"]["mu"][:N]) / params["enc"]["sd"][:N]
    return jnp.clip(jnp.concatenate([q[-1] - q0, q[0] - q0]), -5.0, 5.0)  # bounded: untrained/poor WMs must not blow up the policy input
