"""ChunkPPO arm of the Phase-1 comparison (same evaluator/record format as scripts/run_epa_mini.py).

  python scripts/run_epa_chunkppo.py --k 1 --hist 4 --seeds 1000,1001 --steps 100000 --num-envs 16 --cycles 32 --lr 1e-3 --out out.json
k = 1 is the Phase-1 (step-by-step) setting; k > 1 commits k actions open-loop (Phase 3, informational only).
"""
import argparse
import subprocess
import sys
import os
import time

sys.path.insert(0, ".")
import jax  # noqa: E402

from src.pipeline.epa_chunkppo import EpaChunkPPO, chunkppo_config  # noqa: E402
from src.pipeline.epa_harness import make_record, write_json  # noqa: E402


def git_commit():
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--hist", type=int, default=4)
    ap.add_argument("--seeds", default="1000")
    ap.add_argument("--role", default="tune", choices=["tune", "final"])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--cycles", type=int, default=32, help="chunk cycles per env per update (steps per update = envs*cycles*k)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-envs", type=int, default=256)
    ap.add_argument("--protocol", default="EP-A-mini")
    ap.add_argument("--tuning-budget", default="unspecified")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s != ""]
    if a.role == "tune" and min(seeds) < 1000:
        raise SystemExit("tuning runs must use tuning seeds >= 1000")
    if a.role == "final" and max(seeds) >= 1000:
        raise SystemExit("final runs use evaluation seeds < 1000")
    per_seed, name = [], f"chunkppo_k{a.k}_h{a.hist}"
    for s in seeds:
        t0 = time.time()
        cfg = chunkppo_config(s, a.k, a.hist, a.steps, a.num_envs, a.cycles, a.lr, epochs=a.epochs,
                              minibatches=a.minibatches)
        runner = EpaChunkPPO(cfg, a.eval_envs)
        if not per_seed:
            print(f"{name} backend={jax.default_backend()} updates={runner.n_updates} "
                  f"env_steps={runner.n_updates * runner.ticks_per_update}", flush=True)
        summary = runner.run(out_dir=None, log_fn=lambda *_: None)
        ev = runner.final_eval
        p_total, p_dep = runner.param_counts()
        env_steps = runner.n_updates * runner.ticks_per_update
        recs = summary["records"]
        row = dict(seed=s, reward_pct=ev["reward_pct"], score_pct=ev["score_pct"], eval_episodes=ev["eval_episodes"],
                   eval_censored=ev["eval_censored"], eval_mean_length=ev["eval_mean_length"],
                   achievement_rates_pct=ev["achievement_rates_pct"],
                   train_curve_return=[r["train_return_all"] for r in recs[:: max(len(recs) // 10, 1)]],
                   train_seconds=recs[-1]["wall_s"] if recs else 0.0, wall_seconds=time.time() - t0)
        per_seed.append(row)
        print(f"seed {s}: reward_pct={row['reward_pct']:.2f} score_pct={row['score_pct']:.2f} "
              f"len={row['eval_mean_length']:.0f} censored={row['eval_censored']} wall={row['wall_seconds']:.0f}s", flush=True)
        record = make_record(name, "chunkppo", dict(k=a.k, hist=a.hist), cfg._asdict(), per_seed, a.role, a.protocol,
                             env_steps, p_total, p_dep, a.tuning_budget, git_commit())
        write_json(a.out, record)
    print(f"DONE reward_pct={record['reward_pct_mean']:.2f}+-{record['reward_pct_se']:.2f} "
          f"score_pct={record['score_pct_mean']:.2f}+-{record['score_pct_se']:.2f} params_total={p_total} deployed={p_dep}")


if __name__ == "__main__":
    main()
