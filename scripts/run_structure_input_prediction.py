"""Structure-as-input world model, rollout-prediction test with FEATURE-SWAP CONTROLS (arc.tex sec. 4.2; TASK-20260929-016).

Same tasks/models/optimiser as scripts/run_hybrid_worldmodel.py (imported unchanged). Only the scalar 'preset potential' Vp(q) handed to the
learned potential changes. Arms that consume Vp: phys_feat (feature only), skip_feat0 (feature + gated skip, gate starts closed), phys_hn (hard
additive term). Variants of Vp per task:
  preset    the task's own known-but-incomplete formula (Newton mu=1 / gravity / small-angle)         <- the claim under test
  true      the TRUE potential of the conservative part (upper bound; uses task knowledge on purpose)
  harmonic  0.5|q|^2 (generic quadratic, unrelated to Kepler / fall)     poly4  |q|^4/4     hifreq  sin(5|q|+0.7)     zero  0
Decision rule: 'structure as input' is supported only if `preset`/`true` beat `harmonic/poly4/hifreq/zero` under the same arm, OOD included.
References without any preset: hn, mlp. Metric: mean normalised rollout RMSE (ID and OOD held-out ICs), median over seeds.
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

import run_hybrid_worldmodel as H  # noqa: E402
import run_physics_benchmark as pb  # noqa: E402


def make_vp(task_name, variant, mu_p):
    if variant == "harmonic":
        return lambda q: 0.5 * jnp.sum(q ** 2)
    if variant == "poly4":
        return lambda q: 0.25 * jnp.sum(q ** 2) ** 2
    if variant == "hifreq":
        return lambda q: jnp.sin(5.0 * jnp.sqrt(jnp.sum(q ** 2) + 1e-6) + 0.7)
    if variant == "zero":
        return lambda q: 0.0 * jnp.sum(q)
    if variant == "true":
        if task_name.startswith("kepler"):
            return lambda q: -1.0 / jnp.sqrt(jnp.sum(q ** 2) + 1e-6) + (0.25 / 3.0) / (jnp.sum(q ** 2) + 1e-6) ** 1.5
        if task_name == "pendulum":
            return lambda q: 1.0 - jnp.cos(q[0])
        raise ValueError("no closed-form conservative potential for fall_drag (drag is non-conservative); skip 'true'")
    raise ValueError(variant)


class SwapTask(H.HTask):
    def __init__(self, name, variant):
        super().__init__(name)
        self.variant = variant
        self._vp = None if variant == "preset" else make_vp(name, variant, self.mu_p)

    def Vp(self, q):
        return H.HTask.Vp(self, q) if self._vp is None else self._vp(q)


VARIANTS = {"kepler_pert": ["preset", "true", "harmonic", "poly4", "hifreq", "zero"],
            "kepler_wrongmu": ["preset", "true", "harmonic", "hifreq", "zero"],
            "fall_drag": ["preset", "harmonic", "poly4", "hifreq", "zero"],
            "pendulum": ["preset", "true", "poly4", "hifreq", "zero"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(VARIANTS))
    ap.add_argument("--arms", nargs="+", default=["phys_feat", "skip_feat0", "phys_hn"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[64])
    ap.add_argument("--noises", nargs="+", type=float, default=[0.02])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("backend", jax.default_backend(), flush=True)
    res, t0 = [], time.time()
    for tn in a.tasks:
        rng = np.random.default_rng(999)
        base = H.HTask(tn)
        test = {}
        for split, ood in (("id", False), ("ood", True)):
            x0 = base.ic(rng, 64, ood)
            test[split] = (x0, base.simulate(x0, base.test_h)[:, 1:])
        jobs = [("hn", "none"), ("mlp", "none")] + [(arm, v) for v in VARIANTS[tn] for arm in a.arms]
        for arm, variant in jobs:
            task = SwapTask(tn, "preset" if variant == "none" else variant)
            for n_traj in a.sizes:
                for noise in a.noises:
                    for seed in a.seeds:
                        try:
                            r = H.run_one(task, arm, n_traj, noise, seed, a.steps, test)
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            r = dict(error=repr(e))
                        r.update(task=tn, arm=arm, variant=variant, n_traj=n_traj, noise=noise, seed=seed, steps=a.steps)
                        res.append(r)
                        if "error" not in r:
                            print(f"{tn:14s} {arm:11s} {variant:9s} N={n_traj} s={seed} ID={r['id']['rmse_mean']:.3f} "
                                  f"OOD={r['ood']['rmse_mean']:.3f} t={time.time()-t0:.0f}s", flush=True)
                        json.dump(res, open(f"{a.out}/structure_prediction_results.json", "w"))
    json.dump(res, open(f"{a.out}/structure_prediction_results.json", "w"))


if __name__ == "__main__":
    main()
