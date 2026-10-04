"""Integrity / provenance / reproducibility audit of Phase-1 result files (TASK-20261004-022). Standard library + git only.

  python scripts/audit_phase1_results.py                        # audit output/phase1, write the report, exit 1 on integrity errors
  python scripts/audit_phase1_results.py --strict               # also exit 1 on provenance gaps
  python scripts/audit_phase1_results.py --report docs/experiments/2026-10-04_phase1_score_audit/audit_report.json

ERROR (integrity; any one makes the number untrustworthy):
  metric      per-seed reward_pct / score_pct are not reproduced from the stored per-achievement rates
              (reward_pct = sum(rates)/22, score_pct = exp(mean ln(1+rate)) - 1, roadmap definitions)
  granularity a rate, the achievement total or the length total is not a whole number of episodes (fabricated or edited values
              would almost never land on multiples of 100/eval_episodes)
  aggregate   stored mean / SE (ddof=1) are not those of the listed per-seed values
  budget      EP-A* file with more than 1,000,000 env steps, or auxiliary steps not counted
  seeds       tuning run on an evaluation seed (< 1000), or final run on a tuning seed (>= 1000)
  eval        EP-A final with fewer than 256 evaluation episodes
  params      params_deployed > params_total or non-positive counts
  duplicate   identical per-seed outcome for different (arm, config, seed) -- copies of the same run are listed as info
PROVENANCE (the reporting rule: the recorded commit must identify the code):
  code_absent the arm builder / option used by the file does not exist at the recorded git_commit (uncommitted code ran)
  dirty       new-format file whose provenance says git_dirty is true/unknown for a final run
  no_eval_commit / no_versions   field missing (files written before TASK-022 cannot have them)
REPRO: the same (arm, config, seed) measured in different runs (different commit or session) -> both values listed.
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
from collections import defaultdict

BUDGET = 1_000_000
CFG_DEFAULTS = {"value_norm": 0.0, "adv_norm": "minibatch"}   # PPOConfig fields added by TASK-023 (absent = old default)
_src_cache = {}


def git_show(commit, path):
    key = (commit, path)
    if key not in _src_cache:
        try:
            _src_cache[key] = subprocess.check_output(["git", "show", f"{commit}:{path}"], stderr=subprocess.DEVNULL).decode()
        except Exception:
            _src_cache[key] = None
    return _src_cache[key]


def commit_exists(commit):
    return subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], stderr=subprocess.DEVNULL).returncode == 0


def crafter_score(rates):
    return math.exp(sum(math.log(1.0 + max(0.0, r)) for r in rates) / len(rates)) - 1.0


def mean_se(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))


def whole(x, tol):
    return abs(x - round(x)) <= tol


def norm_cfg(cfg):
    """Full config as a hashable key (the run seed some families store in it is dropped; new fields get their old default)."""
    c = {k: v for k, v in cfg.items() if k != "seed"}
    if "lam" in c:   # epa_harness PPOConfig
        for k, v in CFG_DEFAULTS.items():
            c.setdefault(k, v)
    return tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in c.items()))


def code_present(rec):
    """Did the code path named by the file exist at its recorded commit? Returns (ok, reason)."""
    commit = rec.get("git_commit", "unknown")
    if not commit or commit == "unknown" or not commit_exists(commit):
        return False, f"git_commit {commit!r} is not a commit of this repository"
    key = rec.get("arm_key", "")
    if key == "chunkppo":
        ok = git_show(commit, "src/pipeline/epa_chunkppo.py") is not None
        return ok, "" if ok else "src/pipeline/epa_chunkppo.py absent at the recorded commit"
    pol = git_show(commit, "src/model/epa_policies.py")
    if pol is None:
        return False, "src/model/epa_policies.py absent at the recorded commit"
    if f'"{key}":' not in pol:
        return False, f"arm builder {key!r} absent from ARM_BUILDERS at the recorded commit"
    for k in rec.get("arm_kwargs", {}):
        if f"{k}=" not in pol:
            return False, f"arm option {k!r} absent at the recorded commit"
    harness = git_show(commit, "src/pipeline/epa_harness.py") or ""
    for k in ("value_norm", "adv_norm"):
        if k in rec.get("config", {}) and rec["config"][k] != CFG_DEFAULTS[k] and f"{k}:" not in harness:
            return False, f"PPOConfig field {k!r} absent at the recorded commit"
    return True, ""


def audit_file(path, rec, findings, outcomes):
    def add(level, code, msg, seed=None):
        findings.append(dict(file=path, level=level, code=code, seed=seed, msg=msg))

    protocol, role = rec.get("protocol", ""), rec.get("role", "")
    for r in rec["per_seed"]:
        s, rates, n = r.get("seed"), r.get("achievement_rates_pct"), r.get("eval_episodes")
        if rates is None or n is None:
            add("ERROR", "metric", "per-seed row without achievement_rates_pct / eval_episodes", s)
            continue
        if len(rates) != 22:
            add("ERROR", "metric", f"{len(rates)} achievement rates instead of 22", s)
            continue
        rw, sc = sum(rates) / 22.0, crafter_score(rates)
        if abs(rw - r["reward_pct"]) > 1e-3 or abs(sc - r["score_pct"]) > 1e-3:
            add("ERROR", "metric", f"stored reward/score {r['reward_pct']:.5f}/{r['score_pct']:.5f} vs recomputed {rw:.5f}/{sc:.5f}", s)
        if not all(whole(x * n / 100.0, 2e-3) for x in rates):
            add("ERROR", "granularity", f"rates are not multiples of 100/{n}", s)
        if not whole(r["reward_pct"] * 22.0 * n / 100.0, 2e-2):
            add("ERROR", "granularity", "reward_pct does not correspond to a whole number of unlocks", s)
        if "eval_mean_length" in r and not whole(r["eval_mean_length"] * n, 0.5):
            add("ERROR", "granularity", "eval_mean_length * episodes is not an integer step count", s)
        if role == "final" and protocol == "EP-A" and n < 256:
            add("ERROR", "eval", f"{n} evaluation episodes < 256", s)
        if r.get("eval_censored", 0):
            add("INFO", "censored", f"{r['eval_censored']} censored evaluation episodes", s)
        sig = (round(r["reward_pct"], 6), round(r["score_pct"], 6), tuple(round(x, 6) for x in rates))
        outcomes[sig].append(dict(file=path, arm=rec.get("arm"), cfg=norm_cfg(rec.get("config", {})), seed=s,
                                  commit=rec.get("git_commit"), test_seed=r.get("test_seed")))
    seeds = [r["seed"] for r in rec["per_seed"]]
    if len(rec["per_seed"]) and "reward_pct_mean" in rec:
        for m in ("reward_pct", "score_pct"):
            mu, se = mean_se([r[m] for r in rec["per_seed"]])
            st_mu, st_se = rec.get(f"{m}_mean"), rec.get(f"{m}_se")
            bad_se = not (math.isnan(se) and (st_se is None or (isinstance(st_se, float) and math.isnan(st_se)))) and \
                (st_se is None or abs(se - st_se) > 1e-6)
            if st_mu is None or abs(mu - st_mu) > 1e-6 or bad_se:
                add("ERROR", "aggregate", f"{m}: stored {st_mu}/{st_se} vs recomputed {mu}/{se}")
    if protocol.startswith("EP-A"):
        steps = rec.get("env_steps_total", 0) + rec.get("env_steps_auxiliary", 0)
        if steps > BUDGET:
            add("ERROR", "budget", f"{steps} env steps > {BUDGET}")
        if "env_steps_auxiliary" not in rec:
            add("ERROR", "budget", "env_steps_auxiliary not reported")
    if role in ("tune", "diag") and min(seeds) < 1000:
        add("ERROR", "seeds", f"{role} run on evaluation seeds {seeds}")
    if role == "final" and max(seeds) >= 1000:
        add("ERROR", "seeds", f"final run on tuning seeds {seeds}")
    pt, pd = rec.get("params_total", 0), rec.get("params_deployed", 0)
    if not (0 < pd <= pt):
        add("ERROR", "params", f"params_total={pt} params_deployed={pd}")
    ok, why = code_present(rec)
    if not ok:
        add("PROVENANCE", "code_absent", why)
    prov = rec.get("provenance")
    if prov is None:
        add("PROVENANCE", "no_versions", "no evaluator_commit / dirty flag / code hash / package versions (pre-TASK-022 format)")
    else:
        if role == "final" and prov.get("git_dirty") is not False and not rec.get("allow_dirty"):
            add("PROVENANCE", "dirty", f"final run with git_dirty={prov.get('git_dirty')}")
        if not rec.get("evaluator_commit") or rec.get("evaluator_commit") == "unknown":
            add("PROVENANCE", "no_eval_commit", "evaluator_commit missing")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=["output/phase1"])
    ap.add_argument("--report", default="docs/experiments/2026-10-04_phase1_score_audit/audit_report.json")
    ap.add_argument("--strict", action="store_true", help="exit 1 on PROVENANCE findings as well")
    a = ap.parse_args()
    files = sorted({f.replace("\\", "/") for r in a.roots for f in glob.glob(os.path.join(r, "**", "*.json"), recursive=True)})
    findings, outcomes, audited, skipped = [], defaultdict(list), [], []
    for f in files:
        try:
            rec = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            skipped.append(dict(file=f, why=f"unreadable: {e}"))
            continue
        if not isinstance(rec, dict) or not isinstance(rec.get("per_seed"), list) or not rec["per_seed"]:
            skipped.append(dict(file=f, why="not a result file (no per_seed rows)"))
            continue
        audited.append(f)
        audit_file(f, rec, findings, outcomes)
    copies = []
    for sig, runs in outcomes.items():   # identical outcomes: copies of one run are fine, anything else is not
        keys = {(r["arm"], r["cfg"], r["seed"], r["commit"]) for r in runs}
        if len(runs) > 1 and len(keys) == 1:
            copies.append([r["file"] for r in runs])
        elif len(keys) > 1:
            findings.append(dict(file=runs[0]["file"], level="ERROR", code="duplicate", seed=runs[0]["seed"],
                                 msg=f"identical outcome for different runs: {[(r['file'], r['seed']) for r in runs]}"))
    by_run = defaultdict(list)   # same (arm, config, seed, test seed) measured more than once in different sessions
    for runs in outcomes.values():
        for r in runs:
            by_run[(r["arm"], r["cfg"], r["seed"], r["test_seed"] if r["test_seed"] is not None else r["seed"])].append(r)
    repro = []
    for (arm, cfg, seed, tseed), runs in sorted(by_run.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][2]))):
        commits = sorted({r["commit"] for r in runs})
        if len(commits) > 1:
            vals = []
            for c in commits:
                f = next(r["file"] for r in runs if r["commit"] == c)
                row = next(x for x in json.load(open(f))["per_seed"] if x["seed"] == seed)
                vals.append(dict(commit=c[:8], file=f, reward_pct=round(row["reward_pct"], 2), score_pct=round(row["score_pct"], 2)))
            repro.append(dict(arm=arm, seed=seed, config=dict(cfg), runs=vals,
                              reward_spread=round(max(v["reward_pct"] for v in vals) - min(v["reward_pct"] for v in vals), 2)))
    counts = defaultdict(int)
    for x in findings:
        counts[f"{x['level']}:{x['code']}"] += 1
    report = dict(kind="phase1_result_audit", task="TASK-20261004-022", roots=a.roots, files_audited=len(audited),
                  files_skipped=len(skipped), finding_counts=dict(sorted(counts.items())), findings=findings,
                  copies=copies, reproducibility_pairs=repro, skipped=skipped)
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"audited {len(audited)} result files ({len(skipped)} other json skipped); findings: {dict(sorted(counts.items()))}")
    for x in findings:
        if x["level"] in ("ERROR", "PROVENANCE") and x["code"] not in ("no_versions",):
            print(f"  {x['level']:10s} {x['code']:12s} {x['file']} seed={x['seed']}: {x['msg']}")
    print(f"copies of the same run: {len(copies)} groups; reproducibility pairs (same arm/config/seed, different commit): {len(repro)}")
    for p in repro:
        print(f"  {p['arm']} seed {p['seed']}: " + ", ".join(f"{v['commit']} {v['reward_pct']}/{v['score_pct']}" for v in p["runs"])
              + f"  (reward spread {p['reward_spread']})")
    print(f"report -> {a.report}")
    errors = counts and any(k.startswith("ERROR") for k in counts)
    prov = any(k.startswith("PROVENANCE") and k != "PROVENANCE:no_versions" for k in counts)
    sys.exit(1 if errors or (a.strict and prov) else 0)


if __name__ == "__main__":
    main()
