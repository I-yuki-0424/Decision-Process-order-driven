"""TASK-20261022-027 block-J summary: roadmap gate checks (surpass = mean - 2 SE > reference) for the final arms, paired difference to the same-seed control.

  python scripts/summarize_t026.py            # prints markdown; reads only result files under output/phase1/push_t026/finalJ

Block J: train seeds 172-181 / test 554-563, Kaggle T4, clean tree. References are quoted from MASTER_GUIDANCE <references> (Dedieu et al. 2025, Table 1); not edited here.
"""
import glob
import json
import os

import numpy as np

REF = {"G1.1": dict(src="MFRL_4M_reimpl", reward=47.40, score=10.71, params=4.0e6),     # SOURCE: MASTER_GUIDANCE ref MFRL_4M_reimpl
       "G1.2": dict(src="MFRL_best", reward=55.49, score=16.77, params=None)}           # SOURCE: MASTER_GUIDANCE ref MFRL_best
DIR = "output/phase1/push_t026/finalJ"
SEEDS = range(172, 182)
ARMS = ["G4TP1", "G_lp1", "G_rs0_lp1", "G_rs0_ent005_w512"]


def load(arm):
    rows = []
    for s in SEEDS:
        f = os.path.join(DIR, f"{arm}__s{s}.json")
        if os.path.exists(f):
            d = json.load(open(f))
            r = d["per_seed"][0]
            rows.append(dict(seed=s, reward=r["reward_pct"], score=r["score_pct"], params=d["params_total"], dep=d["params_deployed"], steps=d["env_steps_total"],
                             protocol=d["protocol"], commit=d["git_commit"][:8], dirty=d.get("provenance", {}).get("git_dirty"), test=r["test_seed"]))
    return rows


def ms(x):
    x = np.asarray(x, float)
    return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan"))


def main():
    data = {a: load(a) for a in ARMS}
    print("| arm | n | protocol | params_total / deployed | reward | score | G1.1 | G1.2 | commit (dirty) |\n|---|---|---|---|---|---|---|---|---|")
    for a, v in data.items():
        if not v:
            continue
        rm, rs = ms([x["reward"] for x in v])
        sm, ss = ms([x["score"] for x in v])
        g = {k: (rm - 2 * rs > r["reward"]) and (sm - 2 * ss > r["score"]) and (r["params"] is None or v[0]["params"] <= r["params"]) for k, r in REF.items()}
        part = "" if len(v) == 10 else f" (INCOMPLETE {len(v)}/10: no gate)"
        if "P5-contested" in v[0]["protocol"]:   # numbers meet the relation, but the arm changes the training reward: not gate evidence
            g = {k: ("met" if x else "fail") + " (P5-contested: not eligible)" for k, x in g.items()}
        print(f"| {a} | {len(v)} | {v[0]['protocol']} | {v[0]['params']:,} / {v[0]['dep']:,} | {rm:.2f} +- {rs:.2f} (-2SE {rm - 2 * rs:.2f}) | {sm:.2f} +- {ss:.2f} (-2SE {sm - 2 * ss:.2f}) | "
              f"{g['G1.1'] if isinstance(g['G1.1'], str) else ('PASS' if g['G1.1'] and not part else 'fail')}{part} | {g['G1.2'] if isinstance(g['G1.2'], str) else ('PASS' if g['G1.2'] and not part else 'fail')} | {v[0]['commit']} ({v[0]['dirty']}) |")
    ctl = {x["seed"]: x for x in data["G4TP1"]}
    print("\nPaired difference to the same-seed control G4TP1 (mean +- SE of the 10 per-seed differences; seed s uses test seed s+382 for every arm):\n")
    for a in ARMS[1:]:
        d = [(x["reward"] - ctl[x["seed"]]["reward"], x["score"] - ctl[x["seed"]]["score"]) for x in data[a] if x["seed"] in ctl]
        if len(d) > 1:
            r, s = ms([t[0] for t in d]), ms([t[1] for t in d])
            print(f"* {a} - G4TP1 (n={len(d)}): reward {r[0]:+.2f} +- {r[1]:.2f} (z {r[0] / r[1]:.1f}), score {s[0]:+.2f} +- {s[1]:.2f} (z {s[0] / s[1]:.1f})")
    # block K (train 182-191 / test 564-573): replication of the strongest P5-contested arm only (amendment 7); no control in that block
    rows = []
    for sd in range(182, 192):
        f = os.path.join(DIR.replace("finalJ", "finalK"), f"G_rs0_ent005_w512__s{sd}.json")
        if os.path.exists(f):
            r = json.load(open(f))["per_seed"][0]
            rows.append((r["reward_pct"], r["score_pct"]))
    if rows:
        a = np.array(rows)
        rm, rs = ms(a[:, 0])
        sm, ss = ms(a[:, 1])
        j = np.array([(x["reward"], x["score"]) for x in data["G_rs0_ent005_w512"]])
        pooled = np.vstack([j, a])
        pr, ps = ms(pooled[:, 0]), ms(pooled[:, 1])
        print(f"\nBlock K, G_rs0_ent005_w512 (n={len(a)}): reward {rm:.2f} +- {rs:.2f} (-2SE {rm - 2 * rs:.2f}), score {sm:.2f} +- {ss:.2f} (-2SE {sm - 2 * ss:.2f}); "
              f"G1.2 {'PASS' if len(a) == 10 and rm - 2 * rs > 55.49 and sm - 2 * ss > 16.77 else 'fail'}")
        print(f"Pooled blocks J+K (n={len(pooled)}, descriptive): reward {pr[0]:.2f} +- {pr[1]:.2f}, score {ps[0]:.2f} +- {ps[1]:.2f}")
    print("\nPer-seed reward:")
    for a, v in data.items():
        print(f"* {a}: " + ", ".join(f"{x['seed']}:{x['reward']:.1f}" for x in v))


if __name__ == "__main__":
    main()
