"""Train + evaluate one Phase-1 arm on Craftax-Classic (symbolic 1345-d, uncapped episodes) for a list of seeds.

  python scripts/run_epa_mini.py --arm ppo_mlp --seeds 0,1,2 --steps 100000 --num-envs 16 --num-steps 32 --lr 1e-3 --out output/phase1/x.json

Arms: ppo_mlp | ppo_gru | tf | tf_wm | tf_wm_random.  `--role tune` uses tuning seeds only (>= 1000) and is what config selection may read;
`--role final` uses evaluation seeds 0..9 and must be run with a config that was fixed BEFORE looking at any final-seed number.
`--role replication` re-runs an earlier config on its earlier seeds to test reproducibility (never selected, never a gate result);
`--role diag` trains on a tuning seed for diagnostics only (e.g. with --save-params; never selected).
Protocol label EP-A-mini = EP-A rules with a training budget below 1M steps (never gating, never comparable to papers).
Every result file records run provenance (git commit, dirty flag, evaluator commit, code hash, package versions); final runs
refuse to start from a tree with uncommitted changes unless --allow-dirty (TASK-20261004-022).
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time

sys.path.insert(0, ".")
import jax  # noqa: E402
import numpy as np  # noqa: E402

from src.model.epa_policies import ARM_BUILDERS  # noqa: E402
from src.pipeline.epa_harness import PPOConfig, Trainer, curve_summary, make_record, run_provenance, write_json  # noqa: E402


def git_commit():
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def save_params(directory, a, arm_name, cfg, seed, params, prov):
    """Final parameters of one seed with everything needed to rebuild the policy (scripts/diagnose_phase1_deaths.py loads it)."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{os.path.basename(a.out)[:-5]}__s{seed}.pkl")
    with open(path, "wb") as f:
        pickle.dump(dict(arm_key=a.arm, arm_kwargs=json.loads(a.arm_kwargs), arm=arm_name, config=cfg._asdict(), seed=seed,
                         test_seed=seed + a.eval_seed_offset, role=a.role, provenance=prov,
                         params=jax.tree_util.tree_map(np.asarray, params)), f)
    print(f"saved params -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(ARM_BUILDERS))
    ap.add_argument("--arm-kwargs", default="{}", help="JSON kwargs for the arm builder, e.g. '{\"hist\": 4}'")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--role", default="tune", choices=["tune", "final", "replication", "diag"])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--num-steps", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--value-norm", type=float, default=0.0, help="EMA decay of the value-target mean/std (0 = off)")
    ap.add_argument("--adv-norm", default="minibatch", choices=["minibatch", "batch"])
    ap.add_argument("--save-params", default="", help="directory: also write the final params of every seed (pickle of numpy arrays)")
    ap.add_argument("--allow-dirty", action="store_true", help="let a final run start from a tree with uncommitted changes (recorded)")
    ap.add_argument("--eval-envs", type=int, default=256)
    ap.add_argument("--eval-seed-offset", type=int, default=0, help="evaluation uses seed+offset (MLflow convention: train 42+k -> test 424+k = offset 382)")
    ap.add_argument("--seed-convention", default="legacy", help="recorded in the result file (legacy | mlflow_v11)")
    ap.add_argument("--protocol", default="EP-A-mini")
    ap.add_argument("--tuning-budget", default="unspecified")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-every", type=int, default=0)
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s != ""]
    if a.role in ("tune", "diag") and min(seeds) < 1000:
        raise SystemExit("tuning/diagnostic runs must use tuning seeds >= 1000 (disjoint from evaluation seeds)")
    if a.role == "final" and max(seeds) >= 1000:
        raise SystemExit("final runs use evaluation seeds < 1000")
    if a.role == "replication" and "replication" not in a.protocol:
        raise SystemExit("replication runs need a protocol label containing 'replication' (they never count as results)")
    prov = run_provenance(extra_files=[os.path.relpath(__file__)])
    if a.role == "final" and prov["git_dirty"] is not False and not a.allow_dirty:
        raise SystemExit(f"final run from a tree with uncommitted changes (git_dirty={prov['git_dirty']}): commit first "
                         "so git_commit identifies the code, or pass --allow-dirty (recorded in the result file)")
    arm = ARM_BUILDERS[a.arm](**json.loads(a.arm_kwargs))
    cfg = PPOConfig(total_steps=a.steps, num_envs=a.num_envs, num_steps=a.num_steps, epochs=a.epochs,
                    minibatches=a.minibatches, lr=a.lr, ent=a.ent, lam=a.lam, gamma=a.gamma, clip=a.clip, vf=a.vf,
                    max_grad_norm=a.max_grad_norm, value_norm=a.value_norm, adv_norm=a.adv_norm)
    tr = Trainer(arm, cfg)
    print(f"arm={arm.name} backend={jax.default_backend()} updates={tr.n_updates} env_steps={tr.env_steps_total} cfg={cfg}")
    print(f"provenance: commit={prov['git_commit'][:8]} dirty={prov['git_dirty']} code={prov['code_sha256'][:12]} "
          f"packages={prov['packages']} xla_flags='{prov['xla_flags']}'", flush=True)
    per_seed = []
    for s in seeds:
        t0 = time.time()
        params, curve, train_s = tr.train(s, log_every=a.log_every)
        ev = tr.evaluate(params, s + a.eval_seed_offset, a.eval_envs)
        if not per_seed:
            p_total, p_dep = arm.param_counts(params)
        row = dict(seed=s, test_seed=s + a.eval_seed_offset, reward_pct=ev["reward_pct"], score_pct=ev["score_pct"], eval_episodes=ev["eval_episodes"],
                   eval_censored=ev["eval_censored"], eval_mean_length=ev["eval_mean_length"],
                   achievement_rates_pct=ev["achievement_rates_pct"], train_curve_return=curve_summary(curve),
                   train_seconds=train_s, wall_seconds=time.time() - t0)
        per_seed.append(row)
        print(f"seed {s}: reward_pct={row['reward_pct']:.2f} score_pct={row['score_pct']:.2f} "
              f"len={row['eval_mean_length']:.0f} censored={row['eval_censored']} train={train_s:.0f}s wall={row['wall_seconds']:.0f}s",
              flush=True)
        if a.save_params:
            save_params(a.save_params, a, arm.name, cfg, s, params, prov)
        record = make_record(arm.name, a.arm, json.loads(a.arm_kwargs), cfg._asdict(), per_seed, a.role, a.protocol,
                             tr.env_steps_total, p_total, p_dep, a.tuning_budget, git_commit(), provenance=prov)
        record["seed_convention"] = a.seed_convention
        if a.role == "final" and a.allow_dirty:
            record["allow_dirty"] = True
        write_json(a.out, record)
    print(f"DONE reward_pct={record['reward_pct_mean']:.2f}+-{record['reward_pct_se']:.2f} score_pct={record['score_pct_mean']:.2f}+-{record['score_pct_se']:.2f} "
          f"params_total={p_total} deployed={p_dep}")


if __name__ == "__main__":
    main()
