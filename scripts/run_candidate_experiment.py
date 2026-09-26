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
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    print("JAX backend:", jax.default_backend(), jax.devices(), flush=True)
    for name in a.candidates:
        for mode in a.modes:
            try:
                run_experiment(name, a.out, updates=a.updates, batch=a.batch, T=a.T, d_model=a.d_model,
                               lr=a.lr, mode=mode, eval_every=a.eval_every, seed=a.seed)
            except Exception as e:  # keep going so one failing candidate doesn't lose the others
                import traceback, json, os
                traceback.print_exc()
                os.makedirs(a.out, exist_ok=True)
                with open(os.path.join(a.out, f"{name}__{mode}.FAILED.json"), "w") as f:
                    json.dump({"candidate": name, "mode": mode, "error": repr(e)}, f)


if __name__ == "__main__":
    main()
