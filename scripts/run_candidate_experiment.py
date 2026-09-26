"""CLI: run fast candidate convergence experiments (local GPU container or Kaggle).

  python scripts/run_candidate_experiment.py --out output/experiments/local \
      --candidates transformer_branch variant_5_2 --updates 200 --batch 8
"""
import argparse
import sys

sys.path.insert(0, ".")

import jax  # noqa: E402

from src.pipeline.candidate_experiment import run_experiment  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--candidates", nargs="+", required=True)
    p.add_argument("--modes", nargs="+", default=["reinforce"])
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lrs", type=float, nargs="+", default=None, help="lr sweep (overrides --lr)")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="overrides --seed; one run per seed")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--adapter", choices=["legacy", "obs"], default="legacy",
                   help="obs = CraftaxObsAdapter (fixed: exposes the observation, stable action features)")
    p.add_argument("--eval-eps", type=int, default=32)
    p.add_argument("--tag-suffix", default="")
    a = p.parse_args()
    print("JAX backend:", jax.default_backend(), jax.devices(), flush=True)
    from src.environment.craftax_obs_adapter import CraftaxObsAdapter
    for name, mode, seed, lr in [(n, m, s, l) for n in a.candidates for m in a.modes for s in (a.seeds or [a.seed]) for l in (a.lrs or [a.lr])]:
        if True:
            try:
                adapter = CraftaxObsAdapter(a.T, None, 8) if a.adapter == "obs" else None
                tag = f"{name}__{mode}__{a.adapter}{a.tag_suffix}__lr{lr:g}__s{seed}"
                run_experiment(name, a.out, updates=a.updates, batch=a.batch, T=a.T, d_model=a.d_model,
                               lr=lr, ent_coef=a.ent_coef, mode=mode, eval_every=a.eval_every, eval_eps=a.eval_eps,
                               seed=seed, adapter=adapter, tag=tag)
            except Exception as e:  # keep going so one failing candidate doesn't lose the others
                import traceback, json, os
                traceback.print_exc()
                os.makedirs(a.out, exist_ok=True)
                with open(os.path.join(a.out, f"{name}__{mode}.FAILED.json"), "w") as f:
                    json.dump({"candidate": name, "mode": mode, "error": repr(e)}, f)


if __name__ == "__main__":
    main()
