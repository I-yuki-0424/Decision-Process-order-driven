"""Generic ChunkPPO sweep driver with arbitrary ChunkPPOConfig overrides (arc-proposals TASK-20260929-016).

scripts/run_latency_experiment.py is left untouched; this driver exposes every config field through --set so derivative runs
(different batch shape, lr, entropy, k, budget, passive-data size ...) need no code change.

  python scripts/run_latency_sweep2.py --env intercept --deltas 0,4,8 --arms augment,wm --seeds 0,1 --total-ticks 4000000 \
      --set num_envs=16 --set cycles_per_update=16 --set lr=1e-3 --out output/arc_proposals/latency/tuned
Rules as in run_latency_experiment: delta=0 runs only 'augment' (all arms see the same state there); delta must be <= k;
wm/wm_passive get the passive reference-action pretraining given by --passive-* (its env ticks are charged to their budget).
"""
import argparse
import ast
import json
import os
import sys

sys.path.insert(0, ".")
from src.pipeline.chunk_ppo import ChunkPPO, ChunkPPOConfig  # noqa: E402


def parse_sets(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="intercept", choices=["intercept", "pendulum_goal"])
    ap.add_argument("--arms", default="ignore,augment,wm,wm_passive,oracle")
    ap.add_argument("--deltas", default="0,1,2,4,8")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--total-ticks", type=int, default=1_000_000)
    ap.add_argument("--passive-envs", type=int, default=64)
    ap.add_argument("--passive-ticks", type=int, default=200)
    ap.add_argument("--passive-steps", type=int, default=2000)
    ap.add_argument("--set", action="append", default=[], help="ChunkPPOConfig field override, e.g. --set lr=1e-3")
    ap.add_argument("--time-limit-h", type=float, default=24.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ints = lambda s: [int(x) for x in s.split(",") if x != ""]
    ov = parse_sets(a.set)
    runs = []
    for seed in ints(a.seeds):
        for delta in ints(a.deltas):
            if delta > a.k:
                continue
            for arm in (["augment"] if delta == 0 else a.arms.split(",")):
                cfg = ChunkPPOConfig(env=a.env, arm=arm, delta=delta, k=a.k, seed=seed, total_env_ticks=a.total_ticks,
                                     time_limit_s=a.time_limit_h * 3600.0)
                cfg = cfg._replace(**ov)
                if arm in ("wm", "wm_passive") and a.passive_envs > 0:
                    cfg = cfg._replace(passive_envs=a.passive_envs, passive_ticks_per_env=a.passive_ticks,
                                       passive_pretrain_steps=a.passive_steps)
                runs.append((f"{a.env}/d{delta}_{arm}/s{seed}", cfg))
    print(f"{len(runs)} configs, overrides={ov}", flush=True)
    index = []
    for name, cfg in runs:
        runner = ChunkPPO(cfg)
        out_dir = os.path.join(a.out, name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"== {name}: {runner.n_updates} updates x {runner.ticks_per_update} ticks (+{runner.passive_ticks} passive)", flush=True)
        runner.run(out_dir=out_dir, log_fn=lambda s, n=name: print(f"[{n}] {s}", flush=True))
        index.append(dict(name=name, dir=out_dir, cfg=cfg._asdict()))
        json.dump(index, open(os.path.join(a.out, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
