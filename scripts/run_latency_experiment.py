"""Return-vs-latency experiment for the latency / goal-chunking learner (TASK-20260928-012/013, design in
docs/experiments/2026-09-28_latency_chunking/DESIGN.md).

Usage (run from repo root):
  python scripts/run_latency_experiment.py --dry-run --env intercept          # wiring check: 1 rollout, no update
  python scripts/run_latency_experiment.py --env intercept --deltas 0,1,2,4,8 --k 8 --seeds 0,1,2,3
  python scripts/run_latency_experiment.py --calibrate --seeds 0,1            # step 1: plain PPO, MLP actor
  python scripts/run_latency_experiment.py --calibrate --calibrate-actor transformer --seeds 0,1 --num-envs 128
                                                                               # step 1b: chunk actor vs MLP
Step 1 is --calibrate (the learner must reproduce published PPO numbers before any arm comparison is read).
Long runs need operator authorisation (ADR-002 covers Craftax Phase-II runs of 1M-10M steps).
At delta = 0 every arm sees the same state, so only 'augment' is run there (the shared anchor of the curves).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")
from src.environment.latency_envs import charged_latency  # noqa: E402
from src.pipeline.chunk_ppo import ARMS, ChunkPPO, ChunkPPOConfig, craftax_calibration_config  # noqa: E402


def build_configs(a):
    ints = lambda s: [int(x) for x in s.split(",") if x != ""]
    runs = []
    for seed in ints(a.seeds):
        common = dict(seed=seed, total_env_ticks=a.total_ticks, time_limit_s=a.time_limit_h * 3600.0)
        if a.calibrate:
            overrides = dict(common)
            if a.num_envs is not None:   # None = keep craftax_calibration_config's own num_envs (1024, verified)
                overrides["num_envs"] = a.num_envs
            cfg = craftax_calibration_config(actor=a.calibrate_actor, **overrides)
            name = "craftax_calibration" if a.calibrate_actor == "mlp" else f"craftax_calibration_{a.calibrate_actor}"
            runs.append((f"{name}/s{seed}", cfg))
            continue
        for base_delta in ints(a.deltas):
            delta = charged_latency(base_delta, a.k, a.calls_per_tick or None)
            if delta > a.k:
                print(f"skip delta={delta}: exceeds commit length k={a.k}")
                continue
            for arm in (["augment"] if delta == 0 else a.arms.split(",")):
                cfg = ChunkPPOConfig(env=a.env, arm=arm, delta=delta, k=a.k, num_envs=a.num_envs or 64, **common)
                if arm in ("wm", "wm_passive") and a.passive_envs > 0:
                    cfg = cfg._replace(passive_envs=a.passive_envs, passive_ticks_per_env=a.passive_ticks,
                                       passive_pretrain_steps=a.passive_steps)
                runs.append((f"{a.env}/d{delta}_{arm}/s{seed}", cfg))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="intercept", choices=["intercept", "pendulum_goal", "craftax"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--deltas", default="0,1,2,4,8")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--total-ticks", type=int, default=1_000_000)
    ap.add_argument("--num-envs", type=int, default=None,
                    help="default 64 for the delta-sweep; --calibrate keeps craftax_calibration_config's own 1024 "
                         "unless this is set explicitly (e.g. to fit a smaller GPU)")
    ap.add_argument("--passive-envs", type=int, default=64, help="reference-action pretraining (wm arms); 0 = off")
    ap.add_argument("--passive-ticks", type=int, default=200)
    ap.add_argument("--passive-steps", type=int, default=2000)
    ap.add_argument("--calls-per-tick", type=float, default=0.0,
                    help="charge the k sequential decoding passes as extra latency (0 = fixed delta sweep)")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calibrate-actor", default="mlp", choices=["mlp", "transformer"],
                    help="DESIGN.md sec.4 step 1 uses mlp; step 1b re-runs with transformer (k=1) to confirm the "
                         "chunk actor is not weaker than the MLP")
    ap.add_argument("--dry-run", action="store_true", help="one rollout + loss/grad evaluation per config, no update")
    ap.add_argument("--time-limit-h", type=float, default=24.0)
    ap.add_argument("--out", default="output/latency_chunking")
    a = ap.parse_args()

    runs = build_configs(a)
    print(f"{len(runs)} configs")
    index = []
    for name, cfg in runs:
        runner = ChunkPPO(cfg)
        if a.dry_run:
            info = runner.dry_run()
            print(name, json.dumps(info))
            continue
        out_dir = os.path.join(a.out, name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"== {name}: {runner.n_updates} updates x {runner.ticks_per_update} ticks "
              f"(+{runner.passive_ticks} passive)")
        runner.run(out_dir=out_dir, log_fn=lambda s, n=name: print(f"[{n}] {s}", flush=True))
        index.append(dict(name=name, dir=out_dir, cfg=cfg._asdict()))
        with open(os.path.join(a.out, "index.json"), "w") as f:
            json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
