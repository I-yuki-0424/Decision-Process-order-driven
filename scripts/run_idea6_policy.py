"""Stage B of the Idea-6 study: does a frozen, trained noop-world-model help the action-selecting Transformer?

Arms (identical model/adapter/training; only the 8 'anticipation' state features differ):
  base           : anticipation features = zeros
  wm_hn          : trained port-Hamiltonian world model (Stage A)
  wm_mlp         : trained MLP world model (Stage A)
  wm_hn_untrained: same HN architecture, random init (control: extra features with no learned content)
"""
import argparse, os, pickle, sys
sys.path.insert(0, ".")
import jax, jax.numpy as jnp
from src.environment.craftax_obs_adapter import CraftaxObsAdapter
from src.model.candidates import passive_world_model as pwm
from src.pipeline.candidate_experiment import run_experiment


def make_adapter(arm, wm_dir, T, feat_dim=39):
    if arm == "base":
        return CraftaxObsAdapter(T, None, 8)
    kind = "mlp" if arm == "wm_mlp" else "hn"
    if arm == "wm_hn_untrained":
        params = pwm.init_params(jax.random.PRNGKey(123), "hn", feat_dim)
        tr = pickle.load(open(f"{wm_dir}/wm_hn.pkl", "rb"))  # borrow only the input standardisation stats
        params["enc"]["mu"], params["enc"]["sd"] = jnp.asarray(tr["enc"]["mu"]), jnp.asarray(tr["enc"]["sd"])
    else:
        params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(f"{wm_dir}/wm_{kind}.pkl", "rb")))
    return CraftaxObsAdapter(T, lambda f: pwm.anticipation_features(params, kind, f), 8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--wm-dir", required=True)
    ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--candidate", default="transformer_branch")
    ap.add_argument("--updates", type=int, default=250); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--T", type=int, default=250); ap.add_argument("--d-model", type=int, default=256)
    a = ap.parse_args()
    print("backend", jax.default_backend(), flush=True)
    for seed in a.seeds:
        for arm in a.arms:
            ad = make_adapter(arm, a.wm_dir, a.T)
            run_experiment(a.candidate, a.out, updates=a.updates, batch=a.batch, T=a.T, d_model=a.d_model,
                           mode="reinforce", eval_every=50, eval_eps=32, seed=seed, adapter=ad,
                           tag=f"{a.candidate}__{arm}__s{seed}", ckpt_every_updates=100)
