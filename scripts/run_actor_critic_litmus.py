"""Actor-critic litmus test (arc.tex sec. 2.8 / STATE TASK-20260927-011 step 1): does `oracle_act` (leaked true 1-step reward
of every action) now clearly beat `base` once the REINFORCE learner is replaced by PPO + learned critic?

  python scripts/run_actor_critic_litmus.py --out output/arc_proposals/ac_litmus --arms base oracle_act --seeds 0 1 2 3
  python scripts/run_actor_critic_litmus.py --out ... --greedy1-ref   # score the 'argmax leaked 1-step reward' policy

Arms: base (zeros), oracle_act (positive control, answer leakage by design), oracle (true noop future, upper bound),
wm_full / wm_untrain (need --wm-dir with the Stage-A checkpoints), wm_act / wm_act_untrain (learned 1-step action-reward model, --ar-dir).
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, ".")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.environment.craftax_future_adapter import CraftaxFutureAdapter  # noqa: E402
from src.model.candidates import passive_wm_full as pw  # noqa: E402
from src.pipeline.candidate_actor_critic import build_greedy1_rollout, run_ac_experiment  # noqa: E402
from src.pipeline.candidate_experiment import evaluate  # noqa: E402


def make_adapter(arm, wm_dir, T, ar_dir="output/arc_proposals/actreward"):
    if arm in ("wm_act", "wm_act_untrain"):  # LEARNED (deployable) 1-step action-reward model in the oracle_act slot layout
        from src.environment.craftax_actreward_adapter import CraftaxActRewardAdapter
        from src.model.candidates import action_reward_model as ar
        tr = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(f"{ar_dir}/ar_full.pkl", "rb")))
        if arm == "wm_act_untrain":  # control: same architecture/input stats, random init (no learned content)
            p = ar.init_params(jax.random.PRNGKey(123), tr["w1"].shape[0], tr["w1"].shape[1])
            p["mu"], p["sd"] = tr["mu"], tr["sd"]
            tr = p
        return CraftaxActRewardAdapter(T, tr)
    if arm == "base":
        return CraftaxFutureAdapter(T, "zeros")
    if arm == "oracle_act":
        return CraftaxFutureAdapter(T, "oracle_act")
    dsd = pickle.load(open(f"{wm_dir}/delta_stats.pkl", "rb"))["dsd"]
    if arm == "oracle":
        return CraftaxFutureAdapter(T, "oracle", delta_sd=dsd)
    tr = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(f"{wm_dir}/wm_mlp_full.pkl", "rb")))
    if arm == "wm_untrain":
        p = pw.init_params(jax.random.PRNGKey(123), tr["w1"].shape[0], tr["w1"].shape[1])
        p["mu"], p["sd"], p["dsd"] = tr["mu"], tr["sd"], tr["dsd"]
        return CraftaxFutureAdapter(T, "wm", wm_params=p)
    assert arm == "wm_full"
    return CraftaxFutureAdapter(T, "wm", wm_params=tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=["base", "oracle_act"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--candidate", default="transformer_branch")
    ap.add_argument("--ar-dir", default="output/arc_proposals/actreward")
    ap.add_argument("--wm-dir", default="output/experiments/2026-09-27_passive_wm_v2/stageA_random")
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=250)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--greedy1-ref", action="store_true")
    a = ap.parse_args()
    print("backend", jax.default_backend(), flush=True)
    os.makedirs(a.out, exist_ok=True)
    if a.greedy1_ref:
        ad = CraftaxFutureAdapter(a.T, "oracle_act")
        rb = build_greedy1_rollout(ad, a.T)
        ev = evaluate(rb, None, jax.random.PRNGKey(0), 512, 64, greedy=False)
        ev["policy"] = "argmax leaked 1-step reward (ties random)"
        json.dump(ev, open(os.path.join(a.out, "greedy1_reference.json"), "w"), indent=2)
        print("greedy1 reference", {k: v for k, v in ev.items() if k != "achievement_rates"}, flush=True)
    for seed in a.seeds:
        for arm in a.arms:
            run_ac_experiment(a.candidate, a.out, make_adapter(arm, a.wm_dir, a.T, a.ar_dir),
                              tag=f"{a.candidate}__{arm}__ac{a.tag_suffix}__s{seed}", updates=a.updates, batch=a.batch,
                              T=a.T, d_model=a.d_model, lr=a.lr, ent_coef=a.ent_coef, gae_lambda=a.gae_lambda,
                              epochs=a.epochs, minibatches=a.minibatches, eval_every=a.eval_every, seed=seed)


if __name__ == "__main__":
    main()
