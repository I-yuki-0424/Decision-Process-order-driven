"""Backfill / append result JSON files into a LOCAL MLflow store (SQLite file, no server, no cloud).

  python scripts/mlflow_ingest.py [PATH ...] [--root output] [--store mlflow_local] [--dry-run]

PATH may be result files or directories (default: --root). Re-running is idempotent: every run carries a
`source_key` tag (path relative to --root, plus `#seed<N>` for per-seed children) and is skipped if already ingested.
Only metrics that are present in the source file are logged; nothing is computed or estimated here.

Supported schemas (detected by content, not path):
  phase1   {arm_key, per_seed, ...}        -> experiment `phase1-epa`; 1 aggregate run per file + 1 child run per seed
  result   {candidate, final_sampled, ...} -> experiment `candidate-results`; 1 run per file (eval_history as step metrics)
  latency  {cfg, records, ...}             -> experiment `latency-chunking`; 1 run per file, records downsampled
Everything else (benchmarks, replays, selection.json, ...) is counted as skipped and listed in the summary.

Files that are still Git-LFS pointers (not smudged) are skipped and counted; run `git lfs pull` or point --root at a checkout
that has the real files. Raw files are NOT copied into MLflow (only a `source_path` tag), so the store stays small.
"""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

LFS_MAGIC = b"version https://git-lfs"
MAX_STEP_POINTS = 100  # per metric per run; long per-update curves are strided down to this
# latency records carry ~25 per-update columns; only these curves are kept (the full records stay in the source file)
LATENCY_CURVES = ("train_return", "eval_return", "eval_return_greedy", "entropy", "vl", "forecast_skill")
MAX_PARAM_LEN = 500
BATCH = 900  # < MLflow's 1000-metric log_batch limit

EXP_PHASE1 = "phase1-epa"
EXP_RESULT = "candidate-results"
EXP_LATENCY = "latency-chunking"
ORACLE_WORDS = ("oracle",)


def default_store():
    return os.environ.get("MLFLOW_LOCAL_DIR", "mlflow_local")


def store_uri(store):
    return "sqlite:///" + str(Path(store).resolve() / "mlflow.db").replace("\\", "/")


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            s = json.dumps(v) if isinstance(v, (list, tuple)) else str(v)
            out[key] = s[:MAX_PARAM_LEN]
    return out


def stride(points, n=MAX_STEP_POINTS):
    if len(points) <= n:
        return points
    idx = sorted({round(i * (len(points) - 1) / (n - 1)) for i in range(n)})
    return [points[i] for i in idx]


class RunSpec:
    """Plain container: one MLflow run to create."""

    def __init__(self, key, name, start_ms, params, tags, metrics, series, children=()):
        self.key, self.name, self.start_ms = key, name, start_ms
        self.params, self.tags = params, tags
        self.metrics = metrics  # {name: value}
        self.series = series  # {name: [(step, value), ...]}
        self.children = list(children)


def is_oracle(*names):
    return any(w in str(n).lower() for n in names for w in ORACLE_WORDS)


def detect(d):
    if not isinstance(d, dict):
        return None
    if "arm_key" in d and "per_seed" in d:
        return "phase1"
    if "candidate" in d and ("final_sampled" in d or "error" in d):
        return "result"
    if "cfg" in d and "records" in d:
        return "latency"
    return None


def _ach_metrics(rates, names, prefix):
    m = {}
    for i, v in enumerate(rates or []):
        if finite(v):
            label = names[i] if names and i < len(names) else f"{i:02d}"
            m[f"{prefix}ach_rate_pct/{label}"] = float(v)
    return m


def build_phase1(d, key, mtime_ms):
    tags = {"family": "phase1", "architecture": d["arm_key"], "arm": str(d["arm"]), "protocol": str(d.get("protocol")),
            "role": str(d.get("role")), "source_dir": key.rsplit("/", 1)[0], "git_commit": str(d.get("git_commit")),
            "evaluator": str(d.get("evaluator")), "observation": str(d.get("observation")),
            "oracle_control": str(is_oracle(d["arm_key"], d["arm"])).lower()}
    params = flat(d.get("config", {}), "cfg.")
    params.update({"arm_kwargs": json.dumps(d.get("arm_kwargs", {})), "params_total": str(d.get("params_total")),
                   "params_deployed": str(d.get("params_deployed")), "seeds": json.dumps(d.get("seeds")),
                   "tuning_budget": str(d.get("tuning_budget"))[:MAX_PARAM_LEN], "eval_policy": str(d.get("policy"))})
    metrics = {k: float(d[k]) for k in ("reward_pct_mean", "reward_pct_se", "score_pct_mean", "score_pct_se",
                                        "env_steps_total", "env_steps_auxiliary", "params_total", "params_deployed")
               if finite(d.get(k))}
    metrics["n_seeds"] = float(len(d["per_seed"]))
    kids = []
    for s in d["per_seed"]:
        ctags = dict(tags, seed=str(s["seed"]), level="seed", parent_key=key)
        cm = {k: float(s[k]) for k in ("reward_pct", "score_pct", "eval_episodes", "eval_censored", "eval_mean_length",
                                       "train_seconds", "wall_seconds") if finite(s.get(k))}
        cm.update(_ach_metrics(s.get("achievement_rates_pct"), None, ""))
        curve = [(i, float(v)) for i, v in enumerate(s.get("train_curve_return", [])) if finite(v)]
        kids.append(RunSpec(f"{key}#seed{s['seed']}", f"{d['arm']}/seed{s['seed']}", mtime_ms, {}, ctags, cm,
                            {"train_curve_return": curve}))
    tags["level"] = "aggregate"
    return EXP_PHASE1, RunSpec(key, f"{d['arm']} [{key.split('/')[-2]}]", mtime_ms, params, tags, metrics, {}, kids)


def build_result(d, key, mtime_ms):
    cand = d["candidate"]
    mode = d.get("mode") or d.get("learner") or ""
    # variant = run tag minus candidate prefix and seed suffix, so seeds of one configuration group together
    variant = re.sub(r"__s\d+$", "", str(d.get("tag", ""))).removeprefix(f"{cand}__") or str(mode)
    tags = {"family": "candidate-result", "architecture": cand, "variant": variant, "mode": str(mode), "learner": str(d.get("learner")),
            "tag": str(d.get("tag")), "source_dir": key.rsplit("/", 1)[0], "backend": str(d.get("backend")),
            "status": "failed" if "error" in d else str(d.get("status", "ok")),
            "oracle_control": str(is_oracle(cand, variant, d.get("tag", ""))).lower(), "level": "run"}
    seed = d.get("config", {}).get("seed")
    if seed is not None:
        tags["seed"] = str(seed)
    if "error" in d:
        tags["error"] = str(d["error"])[:5000]
    params = flat(d.get("config", {}), "cfg.")
    n_par = d.get("n_actor_params", d.get("n_params"))
    if n_par is not None:
        params["n_params"] = str(n_par)
    metrics = {}
    names = d.get("achievement_names")
    for blk in ("final_sampled", "final_greedy", "random_policy", "untrained_sampled"):
        sub = d.get(blk)
        if not isinstance(sub, dict):
            continue
        for k, v in sub.items():
            if finite(v):
                metrics[f"{blk}/{k}"] = float(v)
        metrics.update(_ach_metrics(sub.get("achievement_rates"), names, f"{blk}/"))
    if finite(d.get("wall_s")):
        metrics["wall_s"] = float(d["wall_s"])
    series = {}
    for e in d.get("eval_history", []):
        step = int(e.get("update", e.get("episodes", 0)))
        for k in ("crafter_score", "mean_return", "mean_unlocked", "mean_len"):
            if finite(e.get(k)):
                series.setdefault(f"eval/{k}", []).append((step, float(e[k])))
    return EXP_RESULT, RunSpec(key, f"{cand}/{d.get('tag', os.path.basename(key))}", mtime_ms, params, tags, metrics, series)


def build_latency(d, key, mtime_ms):
    cfg = d["cfg"]
    arm = str(cfg.get("arm"))
    tags = {"family": "latency-chunking", "architecture": f"latency_{arm}", "arm": arm, "actor": str(cfg.get("actor")),
            "env": str(cfg.get("env")), "source_dir": key.rsplit("/", 1)[0], "backend": str(d.get("backend")),
            "oracle_control": str(is_oracle(arm)).lower(), "level": "run"}
    params = flat(cfg, "cfg.")
    metrics = {k: float(d[k]) for k in ("n_updates", "passive_ticks", "ticks_per_update", "n_settled_updates",
                                        "train_return_settled_mean") if finite(d.get(k))}
    cols = {}
    for r in d["records"]:
        step = r.get("env_ticks", r.get("update"))
        for k, v in r.items():
            if k not in LATENCY_CURVES or not finite(v):
                continue
            cols.setdefault(k, []).append((int(step), float(v)))
    series = {k: stride(v) for k, v in cols.items()}
    for k in ("eval_return", "eval_return_greedy"):  # last logged eval as a headline scalar
        if cols.get(k):
            metrics[f"final/{k}"] = cols[k][-1][1]
    return EXP_LATENCY, RunSpec(key, f"{arm}/{cfg.get('env')}/{os.path.basename(os.path.dirname(key))}", mtime_ms, params, tags,
                                metrics, series)


BUILDERS = {"phase1": build_phase1, "result": build_result, "latency": build_latency}


def is_lfs_pointer(path):
    with open(path, "rb") as f:
        return f.read(len(LFS_MAGIC)) == LFS_MAGIC


def iter_json(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            yield from sorted(p.rglob("*.json"))
        elif p.suffix == ".json":
            yield p


def existing_finished_keys(client, exp_ids):
    from mlflow.entities import ViewType
    done, stale = set(), []
    token = None
    while True:
        page = client.search_runs(exp_ids, max_results=1000, run_view_type=ViewType.ACTIVE_ONLY, page_token=token)
        for r in page:
            k = r.data.tags.get("source_key")
            if not k:
                continue
            (done.add(k) if r.info.status == "FINISHED" else stale.append(r.info.run_id))
        token = page.token
        if not token:
            return done, stale


def write_run(client, exp_id, spec, parent_id=None, finish=True):
    from mlflow.entities import Metric, Param, RunTag
    tags = dict(spec.tags, source_key=spec.key)
    if parent_id:
        tags["mlflow.parentRunId"] = parent_id
    run = client.create_run(exp_id, start_time=spec.start_ms, tags=tags, run_name=spec.name)
    rid = run.info.run_id
    ts = spec.start_ms
    ents = [Metric(k, v, ts, 0) for k, v in spec.metrics.items()]
    for k, pts in spec.series.items():
        ents += [Metric(k, v, ts, s) for s, v in pts]
    params = [Param(k, v) for k, v in spec.params.items()]
    for i in range(0, max(len(params), 1), 100):
        client.log_batch(rid, params=params[i:i + 100])
    for i in range(0, len(ents), BATCH):
        client.log_batch(rid, metrics=ents[i:i + BATCH])
    if finish:
        client.set_terminated(rid, "FINISHED", end_time=ts)
    return rid, len(ents)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="result files / directories (default: --root)")
    ap.add_argument("--root", default="output", help="keys are paths relative to this dir; also the default scan dir")
    ap.add_argument("--store", default=default_store(), help="local store dir holding mlflow.db (env MLFLOW_LOCAL_DIR)")
    ap.add_argument("--dry-run", action="store_true", help="parse and count only; touch nothing")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    files = list(iter_json(a.paths or [root]))
    counts, skipped_dirs, specs = Counter(), Counter(), []
    for f in files:
        f = f.resolve()
        try:
            key = f.relative_to(root).as_posix()
        except ValueError:
            key = f.name
        if is_lfs_pointer(f):
            counts["lfs_pointer"] += 1
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            counts["unreadable"] += 1
            continue
        fam = detect(d)
        if fam is None:
            counts["unsupported"] += 1
            skipped_dirs[key.rsplit("/", 1)[0] if "/" in key else "."] += 1
            continue
        specs.append((fam, key, d, int(f.stat().st_mtime * 1000)))
        counts[fam] += 1
    print("scanned:", dict(counts))
    if counts["lfs_pointer"]:
        print(f"  WARNING: {counts['lfs_pointer']} files are un-smudged Git-LFS pointers (skipped). "
              "Use `git lfs pull` or --root <checkout with real files>.")
    if skipped_dirs:
        print("  unsupported schema, by dir:", dict(skipped_dirs.most_common(8)))
    if a.dry_run:
        return 0

    import mlflow
    from mlflow import MlflowClient
    Path(a.store).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(store_uri(a.store))
    client = MlflowClient()
    exp_ids = {}
    for name in (EXP_PHASE1, EXP_RESULT, EXP_LATENCY):
        e = client.get_experiment_by_name(name)
        exp_ids[name] = e.experiment_id if e else client.create_experiment(name)
    done, stale = existing_finished_keys(client, list(exp_ids.values()))
    for rid in stale:  # leftovers of an interrupted ingest
        client.delete_run(rid)

    t0, n_new, n_skip, n_metrics = time.time(), 0, 0, 0
    for fam, key, d, mt in specs:
        exp, spec = BUILDERS[fam](d, key, mt)
        if spec.key in done:
            n_skip += 1
            continue
        # parent is finished last, so a crash mid-way leaves a RUNNING parent that the next run cleans up
        pid, m = write_run(client, exp_ids[exp], spec, finish=not spec.children)
        n_new, n_metrics = n_new + 1, n_metrics + m
        for c in spec.children:
            _, m = write_run(client, exp_ids[exp], c, parent_id=pid)
            n_new, n_metrics = n_new + 1, n_metrics + m
        if spec.children:
            client.set_terminated(pid, "FINISHED", end_time=spec.start_ms)
    print(f"ingested {n_new} runs ({n_metrics} metric rows), skipped {n_skip} already present, "
          f"{time.time() - t0:.1f}s -> {store_uri(a.store)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
