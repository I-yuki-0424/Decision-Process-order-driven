"""Eval-only re-evaluation of already-trained candidate checkpoints under the post-fix
masked_achievements code path (TASK-20260928-015 item 3 consequence, follow-up task
"evaluation-re-executions"). Does NOT retrain anything: loads a saved parameter
checkpoint (.pkl, written by src.model.checkpoint.AsyncCheckpointManager during the
original training run), rebuilds the exact same rollout function
(src.pipeline.candidate_experiment.build_fns) and adapter the original run used, and
re-runs ONLY the same final evaluate() calls the training script ran at the end
(final_sampled / final_greedy / untrained_sampled / random_policy), with the SAME
PRNGKey derivation (from the run's own seed) so the eval episodes are the same ones
the original run scored -- only masked_achievements is now applied on the achievement
side, which is the only thing item 3's fix changed. mean_return / mean_len are
recomputed too, purely as a sanity check: they are NOT expected to move, since the
auto-reset leak only affects the achievement-derived fields (crafter_score,
mean_unlocked, achievement_rates).

Each entry below is an explicit (not auto-detected) description of one original run,
built by reading:
  - the run's own tag.result.json (candidate name, mode, config: T/d_model/batch/seed/lr/ent_coef)
  - the run's own checkpoint dir (output/experiments/.../checkpoints/<tag>/step_<N>.pkl);
    the highest-N file is used, which src.pipeline.candidate_experiment.run_experiment
    always force-saves at step==updates (i.e. it is the SAME trained params the
    original final_sampled/final_greedy/etc. were computed from)
  - the adapter the ORIGINAL launcher script used for that experiment directory,
    confirmed by reading scripts/run_candidate_experiment.py (2026-09-26_rerun/local,
    tag contains "__obs__" -> CraftaxObsAdapter(T, None, 8)) and
    scripts/run_idea6_policy.py (2026-09-25_idea6/{smokeB,stageB_local}, arm parsed
    from the tag -> CraftaxObsAdapter + optionally a frozen Stage-A world-model
    feature fn loaded from the matching stageA/stageA_smoke wm_*.pkl). The
    2026-09-25_convergence run predates CraftaxObsAdapter's introduction (see
    docs/core/STATE.yaml TASK-20260925-004's blocker "TASK-003 candidate conclusions
    were measured on the observation-blind adapter") and used run_experiment's own
    default (adapter=None -> plain CraftaxEnvAdapter).

Run inside the dpod-local Docker container (needs jax/craftax); example:
  docker run --rm --gpus all -v <repo>:/workspace -w /workspace dpod-local:latest \
      python scripts/adhoc/reeval_candidate_checkpoints.py
"""
import glob
import json
import os
import pickle
import sys

sys.path.insert(0, ".")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.environment.craftax_env_adapter import CraftaxEnvAdapter  # noqa: E402
from src.environment.craftax_obs_adapter import CraftaxObsAdapter  # noqa: E402
from src.model.candidates import passive_world_model as pwm  # noqa: E402
from src.pipeline.candidate_experiment import build_fns, evaluate  # noqa: E402


def latest_checkpoint(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, "step_*.pkl"))
    if not files:
        return None
    return max(files, key=lambda p: int(os.path.basename(p)[len("step_"):-len(".pkl")]))


def load_params(ckpt_path):
    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x) if hasattr(x, "__array__") else x, payload["params"]), payload["step"]


# ---------------------------------------------------------------- adapter builders (mirror the original launchers)
def adapter_legacy(T):
    return CraftaxEnvAdapter(max_episode_steps=T)


def adapter_obs(T):
    return CraftaxObsAdapter(T, None, 8)


def adapter_idea6(arm, wm_dir, T):
    """Mirrors scripts/run_idea6_policy.py make_adapter() exactly, restricted to the
    two arms that have local checkpoints (base, wm_hn)."""
    if arm == "base":
        return CraftaxObsAdapter(T, None, 8)
    if arm != "wm_hn":
        raise ValueError(f"no local checkpoint for idea6 arm {arm!r}; not handled here")
    params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(f"{wm_dir}/wm_hn.pkl", "rb")))
    return CraftaxObsAdapter(T, lambda f: pwm.anticipation_features(params, "hn", f), 8)


# ---------------------------------------------------------------- explicit run list
def build_runs():
    runs = []

    # --- 2026-09-25_convergence/local: 8 tags, plain CraftaxEnvAdapter (pre-dates CraftaxObsAdapter) ---
    conv_dir = "output/experiments/2026-09-25_convergence/local"
    for tag in [
        "mdp_branch__reinforce_normlogits",
        "transformer_branch__argmax_legacy",
        "transformer_branch__reinforce",
        "variant_5_1__reinforce_normlogits",
        "variant_5_2__argmax_legacy",
        "variant_5_2__reinforce",
        "variant_5_3__reinforce",
        "variant_5_4__reinforce_normlogits",
    ]:
        runs.append(dict(
            group="2026-09-25_convergence/local",
            tag=tag,
            result_json=f"{conv_dir}/{tag}.result.json",
            ckpt_dir=f"{conv_dir}/checkpoints/{tag}",
            adapter_fn=lambda T: adapter_legacy(T),
        ))

    # --- 2026-09-25_idea6: smokeB (stageA_smoke WM) + stageB_local (stageA WM); base + wm_hn arms only ---
    for group, T_dir, wm_dir in [
        ("2026-09-25_idea6/smokeB", "output/experiments/2026-09-25_idea6/smokeB",
         "output/experiments/2026-09-25_idea6/stageA_smoke"),
        ("2026-09-25_idea6/stageB_local_s0", "output/experiments/2026-09-25_idea6/stageB_local",
         "output/experiments/2026-09-25_idea6/stageA"),
        ("2026-09-25_idea6/stageB_local_s1", "output/experiments/2026-09-25_idea6/stageB_local",
         "output/experiments/2026-09-25_idea6/stageA"),
    ]:
        seeds = [0] if "smokeB" in group else ([0] if group.endswith("_s0") else [1])
        for seed in seeds:
            for arm in ("base", "wm_hn"):
                tag = f"transformer_branch__{arm}__s{seed}"
                runs.append(dict(
                    group=group,
                    tag=tag,
                    result_json=f"{T_dir}/{tag}.result.json",
                    ckpt_dir=f"{T_dir}/checkpoints/{tag}",
                    adapter_fn=(lambda T, arm=arm, wm_dir=wm_dir: adapter_idea6(arm, wm_dir, T)),
                ))

    # --- 2026-09-26_rerun/local: 9 tags, CraftaxObsAdapter(T, None, 8) ---
    rerun_dir = "output/experiments/2026-09-26_rerun/local"
    for tag in [
        "transformer_branch__reinforce__obs__lr0.0003__s0",
        "transformer_branch__reinforce__obs__lr0.001__s0",
        "transformer_branch__reinforce__obs__lr0.003__s0",
        "variant_5_2__reinforce__obs__lr0.0003__s0",
        "variant_5_2__reinforce__obs__lr0.001__s0",
        "variant_5_2__reinforce__obs__lr0.003__s0",
        "variant_5_3__reinforce__obs__lr0.0003__s0",
        "variant_5_3__reinforce__obs__lr0.001__s0",
        "variant_5_3__reinforce__obs__lr0.003__s0",
    ]:
        runs.append(dict(
            group="2026-09-26_rerun/local",
            tag=tag,
            result_json=f"{rerun_dir}/{tag}.result.json",
            ckpt_dir=f"{rerun_dir}/checkpoints/{tag}",
            adapter_fn=lambda T: adapter_obs(T),
        ))

    return runs


def reeval_one(run):
    with open(run["result_json"], "r", encoding="utf-8") as f:
        old = json.load(f)
    ckpt_path = latest_checkpoint(run["ckpt_dir"])
    if ckpt_path is None:
        return dict(**run, status="NO_CHECKPOINT")

    cfg = old["config"]
    name, mode = old["candidate"], old["mode"]
    T, d_model, batch, seed = cfg["T"], cfg["d_model"], cfg["batch"], cfg["seed"]
    lr, ent_coef = cfg.get("lr", 1e-3), cfg.get("ent_coef", 0.01)

    adapter = run["adapter_fn"](T)
    spec, rollout_batch, _update, _opt = build_fns(name, adapter, d_model, T, lr, ent_coef, mode)

    trained_params, ckpt_step = load_params(ckpt_path)
    key = jax.random.PRNGKey(seed)
    k_init, _k_train, k_eval = jax.random.split(key, 3)

    new_final_sampled = evaluate(rollout_batch, trained_params, k_eval, 64, batch, greedy=False)
    new_final_greedy = evaluate(rollout_batch, trained_params, k_eval, 64, batch, greedy=True)
    init_params = spec.init_fn(k_init, d_model=d_model, num_actions=adapter.num_actions,
                                action_feat_dim=adapter.action_feat_dim, num_costs=adapter.num_costs,
                                num_resources=adapter.num_resources, target_dim=8)
    new_untrained = evaluate(rollout_batch, init_params, k_eval, 64, batch, greedy=False)
    new_random = evaluate(rollout_batch, trained_params, k_eval, 64, batch, greedy=False, random_policy=True)

    result = dict(
        tag=run["tag"], group=run["group"], candidate=name, mode=mode,
        source_result_json=run["result_json"], source_checkpoint=ckpt_path, checkpoint_step=ckpt_step,
        config=cfg,
        final_sampled=new_final_sampled, final_greedy=new_final_greedy,
        untrained_sampled=new_untrained, random_policy=new_random,
        note=("Eval-only re-evaluation of the ORIGINAL trained checkpoint under the post-fix "
              "masked_achievements() code path (no retraining). mean_return/mean_len are "
              "reproduced as a sanity check and are not expected to change; crafter_score/"
              "mean_unlocked/achievement_rates are the corrected fields."),
    )

    old_fs, old_fg = old["final_sampled"], old["final_greedy"]
    sanity_ok = (
        abs(old_fs["mean_return"] - new_final_sampled["mean_return"]) < 1e-3
        and abs(old_fs["mean_len"] - new_final_sampled["mean_len"]) < 1e-3
    )
    return dict(
        **result, status="OK", sanity_return_len_unchanged=sanity_ok,
        delta=dict(
            final_sampled_crafter_score=dict(old=old_fs["crafter_score"], new=new_final_sampled["crafter_score"]),
            final_sampled_mean_unlocked=dict(old=old_fs["mean_unlocked"], new=new_final_sampled["mean_unlocked"]),
            final_greedy_crafter_score=dict(old=old_fg["crafter_score"], new=new_final_greedy["crafter_score"]),
            final_greedy_mean_unlocked=dict(old=old_fg["mean_unlocked"], new=new_final_greedy["mean_unlocked"]),
        ),
    )


def main():
    print("backend", jax.default_backend(), flush=True)
    runs = build_runs()
    summary = []
    for run in runs:
        print(f"[reeval] {run['group']}/{run['tag']} ...", flush=True)
        try:
            out = reeval_one(run)
        except Exception as e:  # keep going; one failing tag shouldn't lose the rest
            import traceback
            traceback.print_exc()
            out = dict(group=run["group"], tag=run["tag"], status="ERROR", error=repr(e))
        summary.append(out)
        if out["status"] == "OK":
            d = out["delta"]
            print(f"  OK ckpt={os.path.basename(out['source_checkpoint'])} sanity_ok={out['sanity_return_len_unchanged']} "
                  f"crafter(sampled) {d['final_sampled_crafter_score']['old']:.3f} -> {d['final_sampled_crafter_score']['new']:.3f} "
                  f"unlocked(sampled) {d['final_sampled_mean_unlocked']['old']:.3f} -> {d['final_sampled_mean_unlocked']['new']:.3f}",
                  flush=True)
            out_dir = os.path.dirname(out["source_result_json"])
            v2_path = os.path.join(out_dir, f"{run['tag']}.result.v2.json")
            with open(v2_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        else:
            print(f"  {out['status']}: {out.get('error', '')}", flush=True)

    summary_path = "output/experiments/reeval_v2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {summary_path}")
    n_ok = sum(1 for s in summary if s["status"] == "OK")
    n_bad_sanity = sum(1 for s in summary if s["status"] == "OK" and not s["sanity_return_len_unchanged"])
    print(f"{n_ok}/{len(summary)} re-evaluated OK; {n_bad_sanity} failed the return/len sanity check")


if __name__ == "__main__":
    main()
