"""Structure-as-input world model, decision-quality test with FEATURE-SWAP CONTROLS (arc.tex sec. 4.3; arc-proposals TASK-20260929-016).

Prior claim (2026-09-26, sec. 4.3): handing a known-but-wrong formula (small-angle potential q^2/2) to the learned potential as an INPUT
FEATURE (`phys_feat`) gets pendulum swing-up MPC within 4-11 return of the true-model ceiling from 8-128 trajectories, whereas the same
formula as a hard term is 275-325 worse. Critical review: the result cannot tell 'the feature carries useful physics' from 'ANY input
feature that extrapolates polynomially helps a tanh-MLP potential'. Falsifiable controls, all with the SAME network/data/optimiser/budget:
  feat:smallangle  the original preset q^2/2 (right at small angle, wrong at large)
  feat:true        -cos q, the TRUE potential (upper bound; uses task knowledge on purpose, NOT evidence for a method)
  feat:scaled      3*q^2/2 (right form, wrong constant)
  feat:poly4       q^4/4 (extrapolating polynomial with WRONG shape)
  feat:hifreq      sin(5q+0.7) (bounded, non-extrapolating, unrelated)
  feat:zero        0 (identical architecture, no information: capacity/extra-input control)
  feat:abs/cubic/q2q4/cos2  generic physics-free bases: |q|, q^3/3, the even polynomial pair [q^2/2, q^4/4], and -cos(2q)
  skip0:/skip1:<f> feature + learned-scale direct skip a*f (a starts 0 / 1);  hard:<f>  feature as a fixed additive term
  hn, mlp          references without any preset
Decision rule: the STRUCTURE claim is supported only if feat:smallangle (and feat:scaled) clearly beat feat:poly4/hifreq/zero and hn;
if poly4 ~ smallangle the gain is 'extrapolating basis', not physics knowledge.
Task = scripts/run_wm_planning.py (torque-limited pendulum swing-up, CEM MPC with the learned model, true env); its code is imported unchanged
and its `init`/`model_field` dispatch is extended for the new arms. All returns are real MPC episodes in the true environment.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import run_wm_planning as W  # noqa: E402
from src.model.candidates.world_model import HamiltonianOps  # noqa: E402

FEATS = {
    "smallangle": lambda q: 0.5 * q ** 2,
    "true": lambda q: -jnp.cos(q),
    "scaled": lambda q: 1.5 * q ** 2,
    "poly4": lambda q: 0.25 * q ** 4,
    "hifreq": lambda q: jnp.sin(5.0 * q + 0.7),
    "zero": lambda q: 0.0 * q,
    # generic, physics-free bases (added after the first controls: the 'wrong' small-angle preset is itself just a quadratic)
    "abs": lambda q: jnp.abs(q),
    "cubic": lambda q: q ** 3 / 3.0,
    "q2q4": lambda q: jnp.stack([0.5 * q ** 2, 0.25 * q ** 4]),
    "cos2": lambda q: -jnp.cos(2.0 * q),
}
_orig_init, _orig_field = W.init, W.model_field


def init2(arm, key):
    if ":" not in arm:
        return _orig_init(arm, key)
    kind, _ = arm.split(":")
    k = jax.random.split(key, 3)
    nf = int(jnp.atleast_1d(FEATS[arm.split(":")[1]](jnp.zeros(()))).shape[0])
    P = dict(v=HamiltonianOps.init_parameters(k[0], 1 if kind == "hard" else 1 + nf, W.HID),
             L=jax.random.normal(k[1], (2, 2)) * 0.01, G=jnp.zeros(2))
    P["a"] = jnp.ones(()) if kind == "skip1" else jnp.zeros(())
    return P


def field2(arm, P, x, u):
    if ":" not in arm:
        return _orig_field(arm, P, x, u)
    kind, name = arm.split(":")
    f = FEATS[name]

    def H(z):
        q, p = z[:1], z[1:]
        kin = 0.5 * jnp.sum(p ** 2)
        if kind == "hard":
            return kin + HamiltonianOps.mlp_scalar(P["v"], q) + jnp.sum(f(q[0]))
        h = kin + HamiltonianOps.mlp_scalar(P["v"], jnp.concatenate([q, jnp.atleast_1d(f(q[0]))]))
        return h + P["a"] * jnp.sum(f(q[0])) if kind in ("skip0", "skip1") else h

    R = HamiltonianOps.dissipation_matrix(P["L"])
    return (W.J - R) @ jax.grad(H)(x) + P["G"] * u


W.init, W.model_field = init2, field2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=["mlp", "hn", "feat:smallangle", "feat:true", "feat:scaled", "feat:poly4",
                                                   "feat:hifreq", "feat:zero", "skip0:smallangle", "skip0:poly4",
                                                   "skip1:smallangle", "hard:smallangle", "hard:poly4"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[8, 32, 128])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--no-refs", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res, t0 = [], time.time()
    print("backend", jax.default_backend(), flush=True)
    if not a.no_refs:
        ret, best = W.mpc_eval(lambda x, u: W.true_field(x, u), a.episodes, 0)
        res.append(dict(arm="true_model_mpc", ret=float(ret.mean()), success=float((best < 0.3).mean()), seed=0, n_traj=0))
        print("reference true-model MPC", res[-1], flush=True)
    for seed in a.seeds:
        for n in a.sizes:
            xs, us = W.collect(np.random.default_rng(seed + 1000 * n), n)
            for arm in a.arms:
                P, tl = W.train(arm, xs, us, a.noise, seed, a.steps)
                ret, best = W.mpc_eval(lambda x, u, P=P, arm=arm: W.model_field(arm, P, x, u), a.episodes, 100 + seed)
                r = dict(arm=arm, n_traj=n, seed=seed, train_loss=tl, ret=float(ret.mean()), success=float((best < 0.3).mean()),
                         min_dist=float(best.mean()), skip_gate=float(P["a"]) if "a" in P and ":" in arm else None,
                         params=int(sum(x.size for x in jax.tree_util.tree_leaves(P))))
                res.append(r)
                print(f"{arm:18s} N={n:4d} s={seed} ret={r['ret']:8.1f} succ={r['success']:.2f} mind={r['min_dist']:.2f} "
                      f"loss={tl:.4f} t={time.time()-t0:.0f}s", flush=True)
                json.dump(res, open(f"{a.out}/structure_planning_results.json", "w"))
    json.dump(res, open(f"{a.out}/structure_planning_results.json", "w"))


if __name__ == "__main__":
    main()
