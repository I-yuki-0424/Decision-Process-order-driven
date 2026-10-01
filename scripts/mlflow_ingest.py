"""Backfill / append result files into the LOCAL MLflow store (SQLite file, no server, no cloud).

  python scripts/mlflow_ingest.py [PATH ...] [--root output] [--store mlflow_local] [--dry-run]
  python scripts/mlflow_ingest.py --record-failure --model-id Idea4_O01_D01_S00 --phase 1 --end-reason interrupted --note "..."

Rules: MASTER_GUIDANCE.xml <experiment_tracking>. Summary:
  * Experiment = phase (`phase-1`, ...); anything not tied to a roadmap phase goes to `phase-0-legacy`.
  * One run per training execution (= per seed). Run name = `<model_id>_<YYYYMMDD-HHMMSS>` (UTC start time). Aggregates (mean/SE)
    are computed by mlflow_report.py, never stored.
  * model_id comes from docs/experiments/MODEL_REGISTRY.yaml (matched on the `arm` string); unmatched -> `UNREGISTERED`.
    Only the operator creates proper names; this script copies `proper_name` from the registry and refuses entries that have one
    without `proper_name_by: operator`.
  * Parameters use the official names of docs/experiments/MLFLOW_PARAM_NAMES.yaml; unknown ones are logged as `x.<name>` and counted.
  * Minimal logging: final scalars + at most MAX_STEP_POINTS points of one or two curves. No artifacts; raw files stay the source of
    truth (`source_key` tag).
  * Failures/interruptions are recorded as runs with status FAILED/KILLED and `end_reason`; the report never aggregates them.

After ingesting, the registry (MODEL_REGISTRY.yaml) is synced into the MLflow Models tab (see mlflow_report.sync_models).
Re-running is idempotent (skips runs whose `source_key` already exists). The store is rebuildable: delete mlflow_local/ and ingest
again. Files that are still Git-LFS pointers are skipped and counted.
Detected schemas: phase1 {arm_key, per_seed}, result {candidate, final_sampled|error}, latency {cfg, records}, failure {end_reason, model_id}.
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO / "docs/experiments/MODEL_REGISTRY.yaml"
PARAMS_PATH = REPO / "docs/experiments/MLFLOW_PARAM_NAMES.yaml"
LFS_MAGIC = b"version https://git-lfs"
MAX_STEP_POINTS = 50  # per curve per run
MAX_PARAM_LEN = 500
BATCH = 900  # < MLflow's 1000-metric log_batch limit
LEGACY_EXP = "phase-0-legacy"
UNREGISTERED = "UNREGISTERED"
MISMATCH = []  # score_pct in a file that disagrees with the geometric mean of its own 22 rates
DEFAULT_TRAIN_SEED, DEFAULT_TEST_SEED = 42, 424  # "0424" is stored as the integer 424
END_REASONS = ("completed", "error", "interrupted", "oom", "timeout")
STATUS_OF = {"completed": "FINISHED", "error": "FAILED", "oom": "FAILED", "timeout": "FAILED", "interrupted": "KILLED"}
LATENCY_CURVES = ("train_return", "eval_return")
CANDIDATE_CURVES = ("crafter_score",)
NON_PARAM = {"arm", "env_kwargs"}  # arm is a tag, not a hyperparameter
# Craftax-Classic achievement order = index of `achievement_rates(_pct)` in the result files; must equal ACHIEVEMENT_NAMES in
# src/environment/craftax_env_adapter.py (tests/test_mlflow_tools.py checks that). Metric keys are `ach_rate_pct/<NN>_<name>`.
ACHIEVEMENT_NAMES = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink", "make_wood_pickaxe", "make_wood_sword",
    "place_plant", "defeat_zombie", "collect_stone", "place_stone", "eat_plant", "defeat_skeleton", "make_stone_pickaxe",
    "make_stone_sword", "wake_up", "place_furnace", "collect_coal", "collect_iron", "collect_diamond", "make_iron_pickaxe",
    "make_iron_sword",
]
METRIC_GUIDE = """Headline metrics (same names in every experiment, sort by these):
- score_pct  = Crafter score = geometric mean of the 22 achievement success rates: exp(mean_i ln(1+s_i)) - 1, s_i in percent.
- reward_pct = arithmetic mean of the 22 success rates = (distinct achievements per episode / 22) x 100.
- ach_rate_pct/<NN>_<name> = success rate (%) of achievement NN; NN is the index in the result files.
Phase-1 files carry both from the evaluation; phase-0-legacy files carry crafter_score / mean_unlocked, mapped to the same names here.
Achievement index: """ + ", ".join(f"{i:02d}={n}" for i, n in enumerate(ACHIEVEMENT_NAMES)) + """
Per-seed runs only; mean and SE over seeds: `python scripts/mlflow_report.py top|summary`. Models tab = docs/experiments/MODEL_REGISTRY.yaml."""


def default_store():
    return os.environ.get("MLFLOW_LOCAL_DIR", "mlflow_local")


def store_uri(store):
    return "sqlite:///" + str(Path(store).resolve() / "mlflow.db").replace("\\", "/")


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ---------------------------------------------------------------- registry / parameter names
class Schema:
    def __init__(self):
        reg = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))["models"]
        self.arm_to_model, self.models = {}, reg
        for mid, m in reg.items():
            if not re.fullmatch(r"[A-Za-z0-9]+_O\d{2}_D\d{2}_S\d{2}", mid):
                raise ValueError(f"model id {mid!r} does not match <IdeaName>_O<NN>_D<NN>_S<NN>")
            if m.get("proper_name") and m.get("proper_name_by") != "operator":
                raise ValueError(f"{mid}: proper_name may only be set by the operator (proper_name_by: operator)")
            for arm in m.get("arms", []):
                self.arm_to_model[arm] = mid
        spec = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
        self.alias = {}
        for official, d in spec["params"].items():
            for a in d["aliases"]:
                if self.alias.setdefault(self.norm(a), official) != official:
                    raise ValueError(f"alias {a!r} maps to both {self.alias[self.norm(a)]} and {official}")
        self.family_alias = spec.get("family_aliases", {})
        self.unknown = Counter()

    @staticmethod
    def norm(name):
        return str(name).lower().replace("-", "_")

    def official(self, name, family):
        if name in self.family_alias.get(family, {}):
            return self.family_alias[family][name]
        n = self.norm(name)
        if n in self.alias:
            return self.alias[n]
        self.unknown[name] += 1
        return f"x.{name}"

    def params(self, raw, family):
        out = {}
        for k, v in raw.items():
            if k in NON_PARAM or v is None or v == [] or v == {}:
                continue
            out[self.official(k, family)] = (json.dumps(v) if isinstance(v, (list, tuple, dict)) else str(v))[:MAX_PARAM_LEN]
        return out

    def model_id(self, arm):
        return self.arm_to_model.get(arm, UNREGISTERED)

    def proper_name(self, mid):
        return (self.models.get(mid) or {}).get("proper_name")


# ---------------------------------------------------------------- run specs
class RunSpec:
    def __init__(self, key, experiment, model_id, start_ms, params, tags, metrics, series, status="FINISHED", label=None):
        self.key, self.experiment, self.model_id, self.start_ms = key, experiment, model_id, start_ms
        self.params, self.tags, self.metrics, self.series, self.status = params, tags, metrics, series, status
        self.label = label  # shown in the run name only while the model is UNREGISTERED

    @property
    def name(self):
        stamp = dt.datetime.fromtimestamp(self.start_ms / 1000, dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        base = self.model_id if self.model_id != UNREGISTERED or not self.label else f"{UNREGISTERED}-{self.label}"
        return f"{base}_{stamp}"


def stride(points, n=MAX_STEP_POINTS):
    if len(points) <= n:
        return points
    idx = sorted({round(i * (len(points) - 1) / (n - 1)) for i in range(n)})
    return [points[i] for i in idx]


def is_oracle(*names):
    return any("oracle" in str(n).lower() for n in names)


def detect(d):
    if not isinstance(d, dict):
        return None
    if "arm_key" in d and "per_seed" in d:
        return "phase1"
    if "end_reason" in d and "model_id" in d:
        return "failure"
    if "candidate" in d and ("final_sampled" in d or "error" in d):
        return "result"
    if "cfg" in d and "records" in d:
        return "latency"
    return None


def when(d, mtime_ms):
    ts = d.get("timestamp") or d.get("started_at")
    if isinstance(ts, str):
        try:
            return int(dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000), "recorded"
        except ValueError:
            pass
    return mtime_ms, "file_mtime"  # backfill approximation, flagged by the time_source tag


def crafter_score(rates):
    """exp(mean ln(1+s_i)) - 1 over the 22 success rates in percent (same formula as calculate_crafter_score in the adapter)."""
    return math.exp(sum(math.log1p(max(0.0, float(r))) for r in rates) / len(ACHIEVEMENT_NAMES)) - 1.0


def _ach(rates, names, prefix=""):
    names = names if names and len(names) == len(ACHIEVEMENT_NAMES) else ACHIEVEMENT_NAMES
    return {f"{prefix}ach_rate_pct/{i:02d}_{names[i]}": float(v) for i, v in enumerate(rates or [])
            if finite(v) and i < len(names)}


def base_tags(sch, mid, key, family, t_src):
    tags = {"model_id": mid, "family": family, "source_key": key, "source_dir": key.rsplit("/", 1)[0] if "/" in key else ".",
            "time_source": t_src, "end_reason": "completed", "oracle_control": "false"}
    if sch.proper_name(mid):
        tags["proper_name"] = str(sch.proper_name(mid))
    return tags


def build_phase1(sch, d, key, mtime_ms):
    mid = sch.model_id(d["arm"])
    start, t_src = when(d, mtime_ms)
    params = sch.params({**d.get("config", {}), **d.get("arm_kwargs", {}), "observation": d.get("observation"),
                         "episode_limit": d.get("episode_limit"), "policy": d.get("policy")}, "phase1")
    out = []
    for s in d["per_seed"]:
        tags = base_tags(sch, mid, f"{key}#seed{s['seed']}", "phase1", t_src)
        tags.update({"architecture": d["arm_key"], "arm": str(d["arm"]), "protocol": str(d.get("protocol")),
                     "role": str(d.get("role")), "git_commit": str(d.get("git_commit")), "evaluator": str(d.get("evaluator")),
                     "gate_eligible": str(d.get("protocol") == "EP-A").lower(), "seed_convention": d.get("seed_convention", "legacy"),
                     "oracle_control": str(is_oracle(d["arm_key"], d["arm"])).lower(), "tuning_budget": str(d.get("tuning_budget"))[:500]})
        p = dict(params, **{"seed.train": str(s["seed"])})
        if "test_seed" in s:
            p["seed.test"] = str(s["test_seed"])
        m = {k: float(s[k]) for k in ("reward_pct", "score_pct", "eval_episodes", "eval_censored", "eval_mean_length", "train_seconds")
             if finite(s.get(k))}
        m.update({k: float(d[k]) for k in ("env_steps_total", "env_steps_auxiliary", "params_total", "params_deployed") if finite(d.get(k))})
        m.update(_ach(s.get("achievement_rates_pct"), None))
        rates = s.get("achievement_rates_pct") or []
        if len(rates) == len(ACHIEVEMENT_NAMES) and finite(s.get("score_pct")) and abs(crafter_score(rates) - s["score_pct"]) > 1e-3:
            MISMATCH.append(f"{key}#seed{s['seed']}: score_pct {s['score_pct']:.4f} != geomean of rates {crafter_score(rates):.4f}")
        curve = stride([(i, float(v)) for i, v in enumerate(s.get("train_curve_return", [])) if finite(v)])
        out.append(RunSpec(tags["source_key"], f"phase-{d.get('phase', 1)}", mid, start, p, tags, m, {"train/return": curve},
                           label=d["arm"]))
    return out


def build_result(sch, d, key, mtime_ms):
    cand = d["candidate"]
    mode = d.get("mode") or d.get("learner") or ""
    variant = re.sub(r"__s\d+$", "", str(d.get("tag", ""))).removeprefix(f"{cand}__") or str(mode)
    start, t_src = when(d, mtime_ms)
    tags = base_tags(sch, UNREGISTERED, key, "candidate-result", t_src)
    failed = "error" in d
    tags.update({"architecture": cand, "variant": variant, "mode": str(mode), "tag": str(d.get("tag")), "backend": str(d.get("backend")),
                 "protocol": "EP-B-legacy", "role": "screen", "gate_eligible": "false",
                 "oracle_control": str(is_oracle(cand, variant, d.get("tag", ""))).lower()})
    if failed:
        tags["end_reason"] = "error"
        tags["error_summary"] = str(d["error"])[:500]
    raw = dict(d.get("config", {}))
    p = sch.params(raw, "candidate")
    if raw.get("seed") is not None:
        tags["seed"] = str(raw["seed"])
    m = {}
    names = d.get("achievement_names")
    for blk, keys in (("final_sampled", ("crafter_score", "mean_return", "mean_unlocked", "mean_len")),
                      ("final_greedy", ("crafter_score",)), ("random_policy", ("crafter_score",)), ("untrained_sampled", ("crafter_score",))):
        sub = d.get(blk)
        if isinstance(sub, dict):
            m.update({f"{blk}/{k}": float(sub[k]) for k in keys if finite(sub.get(k))})
    if isinstance(d.get("final_sampled"), dict):
        m.update(_ach(d["final_sampled"].get("achievement_rates"), names, "final_sampled/"))
        rates = d["final_sampled"].get("achievement_rates") or []
        if len(rates) == len(ACHIEVEMENT_NAMES) and all(finite(r) for r in rates):
            m["reward_pct"] = sum(rates) / len(rates)
        if finite(d["final_sampled"].get("crafter_score")):
            m["score_pct"] = float(d["final_sampled"]["crafter_score"])
    for k in ("n_actor_params", "n_params"):
        if finite(d.get(k)):
            m["n_params"] = float(d[k])
    series = {}
    for e in d.get("eval_history", []):
        step = int(e.get("update", e.get("episodes", 0)))
        for k in CANDIDATE_CURVES:
            if finite(e.get(k)):
                series.setdefault(f"eval/{k}", []).append((step, float(e[k])))
    series = {k: stride(v) for k, v in series.items()}
    return [RunSpec(key, LEGACY_EXP, UNREGISTERED, start, p, tags, m, series, "FAILED" if failed else "FINISHED",
                    label=f"{cand}-{variant}")]


def build_latency(sch, d, key, mtime_ms):
    cfg = d["cfg"]
    arm = str(cfg.get("arm"))
    start, t_src = when(d, mtime_ms)
    tags = base_tags(sch, UNREGISTERED, key, "latency-chunking", t_src)
    tags.update({"architecture": f"latency_{arm}", "arm": arm, "backend": str(d.get("backend")), "protocol": "latency-internal",
                 "role": "screen", "gate_eligible": "false", "oracle_control": str(is_oracle(arm)).lower()})
    p = sch.params(cfg, "latency")
    m = {k: float(d[k]) for k in ("n_updates", "passive_ticks", "n_settled_updates", "train_return_settled_mean") if finite(d.get(k))}
    cols = {}
    for r in d["records"]:
        step = r.get("env_ticks", r.get("update"))
        for k in LATENCY_CURVES:
            if finite(r.get(k)):
                cols.setdefault(k, []).append((int(step), float(r[k])))
    if cols.get("eval_return"):
        m["final/eval_return"] = cols["eval_return"][-1][1]
    series = {f"curve/{k}": stride(v) for k, v in cols.items()}
    return [RunSpec(key, LEGACY_EXP, UNREGISTERED, start, p, tags, m, series, label=f"{arm}-{cfg.get('env')}")]


def build_failure(sch, d, key, mtime_ms):
    reason = d["end_reason"]
    if reason not in END_REASONS or reason == "completed":
        raise ValueError(f"{key}: end_reason must be one of {END_REASONS[1:]}")
    mid = d["model_id"]
    if mid != UNREGISTERED and mid not in sch.models:
        raise ValueError(f"{key}: model_id {mid!r} is not in the registry")
    start, t_src = when(d, mtime_ms)
    tags = base_tags(sch, mid, key, "failure", t_src)
    tags.update({"end_reason": reason, "error_summary": str(d.get("note", ""))[:500], "protocol": str(d.get("protocol")),
                 "role": str(d.get("role", "unknown")), "git_commit": str(d.get("git_commit")), "gate_eligible": "false",
                 "seed_convention": "fixed42"})
    p = {"seed.train": str(d.get("train_seed", DEFAULT_TRAIN_SEED)), "seed.test": str(d.get("test_seed", DEFAULT_TEST_SEED))}
    m = {"steps_completed": float(d["steps_completed"])} if finite(d.get("steps_completed")) else {}
    exp = f"phase-{d['phase']}" if d.get("phase") is not None else LEGACY_EXP
    return [RunSpec(key, exp, mid, start, p, tags, m, {}, STATUS_OF[reason])]


BUILDERS = {"phase1": build_phase1, "result": build_result, "latency": build_latency, "failure": build_failure}


# ---------------------------------------------------------------- io
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


def existing_keys(client, exp_ids):
    from mlflow.entities import ViewType
    done, stale, token = set(), [], None
    while True:
        page = client.search_runs(exp_ids, max_results=1000, run_view_type=ViewType.ACTIVE_ONLY, page_token=token)
        for r in page:
            k = r.data.tags.get("source_key")
            if not k:
                continue
            if r.info.status == "RUNNING":  # leftover of an interrupted ingest
                stale.append(r.info.run_id)
            else:
                done.add(k)
        token = page.token
        if not token:
            return done, stale


def write_run(client, exp_id, spec):
    from mlflow.entities import Metric, Param
    run = client.create_run(exp_id, start_time=spec.start_ms, tags=dict(spec.tags), run_name=spec.name)
    rid, ts = run.info.run_id, spec.start_ms
    ents = [Metric(k, v, ts, 0) for k, v in spec.metrics.items()]
    for k, pts in spec.series.items():
        ents += [Metric(k, v, ts, s) for s, v in pts]
    params = [Param(k, v) for k, v in spec.params.items()]
    for i in range(0, len(params), 100):
        client.log_batch(rid, params=params[i:i + 100])
    for i in range(0, len(ents), BATCH):
        client.log_batch(rid, metrics=ents[i:i + BATCH])
    client.set_terminated(rid, spec.status, end_time=ts)
    return len(ents)


def write_failure_file(a, root):
    now = dt.datetime.now(dt.timezone.utc)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip() or "unknown"
    d = {"model_id": a.model_id, "phase": a.phase, "end_reason": a.end_reason, "note": a.note[:500],
         "steps_completed": a.steps_completed, "role": a.role, "protocol": a.protocol, "train_seed": a.train_seed,
         "test_seed": a.test_seed, "git_commit": git, "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    path = Path(root) / "failures" / f"{a.model_id}_{now.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="result files / directories (default: --root)")
    ap.add_argument("--root", default="output", help="keys are paths relative to this dir; also the default scan dir")
    ap.add_argument("--store", default=default_store(), help="local store dir holding mlflow.db (env MLFLOW_LOCAL_DIR)")
    ap.add_argument("--dry-run", action="store_true", help="parse and count only; touch nothing")
    ap.add_argument("--no-sync-models", action="store_true", help="skip rebuilding the MLflow Models tab from the registry")
    f = ap.add_argument_group("failure recording (writes <root>/failures/*.json, then ingests it)")
    f.add_argument("--record-failure", action="store_true")
    f.add_argument("--model-id")
    f.add_argument("--phase", type=int)
    f.add_argument("--end-reason", choices=END_REASONS[1:])
    f.add_argument("--note", default="", help="short cause, <= 500 chars")
    f.add_argument("--steps-completed", type=float)
    f.add_argument("--role", default="unknown", choices=("final", "tune", "screen", "smoke", "unknown"))
    f.add_argument("--protocol", default="unknown")
    f.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    f.add_argument("--test-seed", type=int, default=DEFAULT_TEST_SEED)
    a = ap.parse_args()

    root = Path(a.root).resolve()
    if a.record_failure:
        if not (a.model_id and a.phase is not None and a.end_reason):
            ap.error("--record-failure needs --model-id, --phase and --end-reason")
        if a.model_id != UNREGISTERED and a.model_id not in Schema().models:
            ap.error(f"model_id {a.model_id!r} is not in {REGISTRY_PATH.name}; register it first (or use {UNREGISTERED})")
        a.paths = [write_failure_file(a, root)]
        print("wrote", a.paths[0])

    sch = Schema()
    counts, skipped_dirs, specs = Counter(), Counter(), []
    for fpath in iter_json(a.paths or [root]):
        fpath = fpath.resolve()
        try:
            key = fpath.relative_to(root).as_posix()
        except ValueError:
            key = fpath.name
        if is_lfs_pointer(fpath):
            counts["lfs_pointer"] += 1
            continue
        try:
            d = json.loads(fpath.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            counts["unreadable"] += 1
            continue
        fam = detect(d)
        if fam is None:
            counts["unsupported"] += 1
            skipped_dirs[key.rsplit("/", 1)[0] if "/" in key else "."] += 1
            continue
        specs += BUILDERS[fam](sch, d, key, int(fpath.stat().st_mtime * 1000))
        counts[fam] += 1
    print("scanned:", dict(counts), "| runs:", len(specs))
    if counts["lfs_pointer"]:
        print(f"  WARNING: {counts['lfs_pointer']} files are un-smudged Git-LFS pointers (skipped). "
              "Use `git lfs pull` or --root <checkout with real files>.")
    if skipped_dirs:
        print("  unsupported schema, by dir:", dict(skipped_dirs.most_common(8)))
    reg = Counter(s.model_id for s in specs)
    print(f"  registered runs: {sum(v for k, v in reg.items() if k != UNREGISTERED)}, {UNREGISTERED}: {reg[UNREGISTERED]}")
    if sch.unknown:
        print("  UNKNOWN parameter names (logged as x.<name>; add to MLFLOW_PARAM_NAMES.yaml):", dict(sch.unknown))
    if MISMATCH:
        print(f"  WARNING: {len(MISMATCH)} runs whose score_pct != geometric mean of their rates, e.g. {MISMATCH[0]}")
    if a.dry_run:
        return 0

    import mlflow
    from mlflow import MlflowClient
    Path(a.store).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(store_uri(a.store))
    client = MlflowClient()
    exp_ids = {}
    for name in sorted({s.experiment for s in specs}):
        e = client.get_experiment_by_name(name)
        exp_ids[name] = e.experiment_id if e else client.create_experiment(name)
        client.set_experiment_tag(exp_ids[name], "mlflow.note.content", METRIC_GUIDE)  # shown as the experiment description in the UI
    done, stale = existing_keys(client, [e.experiment_id for e in client.search_experiments()])
    for rid in stale:
        client.delete_run(rid)
    t0, n_new, n_skip, n_metrics = time.time(), 0, 0, 0
    for s in specs:
        if s.key in done:
            n_skip += 1
            continue
        n_metrics += write_run(client, exp_ids[s.experiment], s)
        n_new += 1
    print(f"ingested {n_new} runs ({n_metrics} metric rows), skipped {n_skip} already present, "
          f"{time.time() - t0:.1f}s -> {store_uri(a.store)}")
    if not a.no_sync_models:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from mlflow_report import sync_models  # lazy: mlflow_report imports this module
        print(sync_models(a.store))
    return 0


if __name__ == "__main__":
    sys.exit(main())
