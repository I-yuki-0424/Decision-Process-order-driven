"""Pick the best config per arm from TUNING outputs only (tuning seeds >= 1000). Selection metric: mean reward_pct over tuning seeds.

  python scripts/select_phase1_config.py output/phase1/tune_100k [extra_dir ...]
Several directories (a grid plus its extensions) are pooled per arm. Writes <first dir>/selection.json and prints the tuning table. Refuses to read any file whose seeds include an evaluation seed (< 1000) (P8).
"""
import glob
import json
import os
import re
import sys


def main():
    dirs = sys.argv[1:]
    d = dirs[0]
    per_arm = {}
    for f in sorted(f for dd in dirs for f in glob.glob(os.path.join(dd, "*.json"))):
        if os.path.basename(f) == "selection.json":
            continue
        r = json.load(open(f))
        if min(r["seeds"]) < 1000 or r["role"] != "tune":
            raise SystemExit(f"{f}: contains evaluation seeds; selection must use tuning seeds only")
        arm, cfg = os.path.basename(f)[:-5].split("__")
        n, t, lr = re.match(r"n(\d+)_t(\d+)_lr(.+)", cfg).groups()
        per_arm.setdefault(arm, []).append(dict(config=[int(n), int(t), float(lr)], file=f, reward=r["reward_pct_mean"],
                                                 reward_se=r["reward_pct_se"], score=r["score_pct_mean"], n_seeds=len(r["seeds"])))
    selection = {}
    for arm, rows in per_arm.items():
        rows.sort(key=lambda x: -x["reward"])
        print(f"\n{arm}")
        for x in rows:
            print(f"  n={x['config'][0]:3d} T={x['config'][1]:3d} lr={x['config'][2]:g}: reward_pct={x['reward']:.2f}+-{x['reward_se']:.2f} "
                  f"score_pct={x['score']:.2f} (seeds={x['n_seeds']})")
        selection[arm] = dict(config=rows[0]["config"], tuning_reward_pct=rows[0]["reward"], tuning_score_pct=rows[0]["score"],
                              n_configs=len(rows))
    with open(os.path.join(d, "selection.json"), "w") as f:
        json.dump(selection, f, indent=1)
    print("\nselected:", {k: v["config"] for k, v in selection.items()})


if __name__ == "__main__":
    main()
