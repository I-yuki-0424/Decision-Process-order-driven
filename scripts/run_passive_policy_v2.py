"""Stage B (v2): does the noop-future help the action-selecting Transformer? Identical policy/training in all arms; only the
78 'anticipation' features differ.

  base       : zeros
  oracle     : TRUE simulated noop future (upper bound; uses the simulator at policy time -- NOT a deployable model)
  oracle_act : POSITIVE CONTROL: true one-step reward of each action (pure leakage, tests learner power)
  wm_full    : frozen trained passive WM (Stage A mlp_full) -- no simulator at policy time
  wm_untrain : same WM architecture, random init (control for 'extra input dims with no learned content')
"""
import argparse, os, pickle, sys
sys.path.insert(0, ".")
import jax, jax.numpy as jnp
from src.environment.craftax_future_adapter import CraftaxFutureAdapter
from src.model.candidates import passive_wm_full as pw
from src.pipeline.candidate_experiment import run_experiment


def make_adapter(arm, wm_dir, T):
    dsd = pickle.load(open(f"{wm_dir}/delta_stats.pkl", "rb"))["dsd"]
    if arm == "base":
        return CraftaxFutureAdapter(T, "zeros")
    if arm == "oracle_act":
        return CraftaxFutureAdapter(T, "oracle_act", delta_sd=dsd)
    if arm == "oracle":
        return CraftaxFutureAdapter(T, "oracle", delta_sd=dsd)
    tr = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(f"{wm_dir}/wm_mlp_full.pkl", "rb")))
    if arm == "wm_untrain":
        p = pw.init_params(jax.random.PRNGKey(123), tr["w1"].shape[0], tr["w1"].shape[1])
        p["mu"], p["sd"], p["dsd"] = tr["mu"], tr["sd"], tr["dsd"]  # borrow input stats only
        return CraftaxFutureAdapter(T, "wm", wm_params=p)
    assert arm == "wm_full"
    return CraftaxFutureAdapter(T, "wm", wm_params=tr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--wm-dir", required=True)
    ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--candidate", default="transformer_branch")
    ap.add_argument("--updates", type=int, default=300); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--T", type=int, default=250); ap.add_argument("--d-model", type=int, default=256)
    a = ap.parse_args()
    print("backend", jax.default_backend(), flush=True)
    for seed in a.seeds:
        for arm in a.arms:
            run_experiment(a.candidate, a.out, updates=a.updates, batch=a.batch, T=a.T, d_model=a.d_model,
                           mode="reinforce", eval_every=50, eval_eps=32, seed=seed, adapter=make_adapter(arm, a.wm_dir, a.T),
                           tag=f"{a.candidate}__{arm}__s{seed}", ckpt_every_updates=100)
