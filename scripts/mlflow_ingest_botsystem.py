"""Index BOTSYSTEM training executions (forge1201-MODED-AIPlayer exchange/<base>/, written by botsystem_skill_ppo.py and the
environment's run directories) into the LOCAL MLflow store. Side track: experiment `side-botsystem`, protocol `side-botsystem`;
never a gate (MASTER_GUIDANCE <experiment_tracking> names experiments after roadmap phases; this track belongs to none, and
filing it under a phase or phase-0-legacy would misstate it, so it gets its own experiment).

  python scripts/mlflow_ingest_botsystem.py <exchange> [--base train3 ...] [--store mlflow_local]

One run per training execution (= one base: a learner with its checkpoint; environment restarts inside it are the same
execution). Re-running replaces the run of that base (a live run's numbers grow). Parameters use the official names of
docs/experiments/MLFLOW_PARAM_NAMES.yaml; the learner's arguments come from its 'model' event (logged since 2026-11-01
evening; older executions ran on the defaults, tagged params_source=defaults). Metrics: final scalars and at most 50 points
per curve; per member m<i>/ with a population. Files stay the source of truth (`source_key`).
"""
import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlflow_ingest import REPO, Schema, default_store, store_uri, stride  # noqa: E402

EXPERIMENT = "side-botsystem"
LEARNER_PARAMS = ("lr", "gamma_s", "lam", "clip", "ent", "vf", "max_grad_norm", "seg", "batch_segs", "epochs", "minibatches",
                  "population", "pbt_interval", "pbt_window", "kind_prior", "stage_window", "stage_windows", "stage_tol",
                  "stage_min_nights", "stage1_max_steps", "seed",
                  # GoalWM learner (botsystem_goal_ppo.py, TASK-20261101-031)
                  "goals", "goal_bonus", "goal_horizon", "p_task", "p_pair", "ext_stage", "her_horizon", "replay", "aux_batch", "aux_coef",
                  "aux_lr", "aux_max_grad_norm", "init")


def lines(p: Path):
    if not p.exists():
        return
    for x in p.open():
        try:
            yield json.loads(x)
        except ValueError:
            pass


def git(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def collect(root: Path, base: str):
    """Everything one execution wrote: learner log, episodes, and the environment's runs <base>-<UTC time>."""
    tl = list(lines(root / base / "train" / "train_log.jsonl"))
    eps = list(lines(root / base / "train" / "episodes.jsonl"))
    import re
    pat = re.compile(re.escape(base) + r"-(s\d+-)?\d{8}T\d{6}$")   # <base>-<time> or a shard's <base>-s<i>-<time>
    runs = sorted(d for d in root.iterdir() if d.is_dir() and pat.match(d.name))
    steps, start, shards = [], None, {}
    for d in runs:
        for e in lines(d / "events.jsonl"):
            if e.get("event") == "run_start":
                start = shards[e.get("shard", -1)] = e
            elif e.get("event") == "step":
                steps.append(e)
    if start:
        start = {**start, "bots": sum(x.get("bots", 0) for x in shards.values()), "shards": len(shards)}
    return tl, eps, runs, steps, start


def build(sch: Schema, root: Path, base: str, aip: Path):
    from botsystem_skill_ppo import parse_args
    tl, eps, runs, steps, start = collect(root, base)
    models = [x for x in tl if x.get("event") == "model"]
    if not models:
        return None
    m = models[-1]
    args = m.get("args") or vars(parse_args(["--root", "x", "--base", base, *(["--population", str(len(m["population"]))] if m.get("population") else [])]))
    params = sch.params({k: args[k] for k in LEARNER_PARAMS if k in args}, "botsystem")
    params.update({"arch.d_model": "64", "arch.n_layers": "2", "arch.n_heads": "4", "arch.hidden_width": "256",
                   "arch.kind_prior": str("kp" in m["name"]), "env.k_candidates": str(m["sizes"]["K"]), "env.id": "botsystem-skills"})
    goal = m["name"].startswith("goalwm")
    rkey = "return_task" if goal else "return_stage"     # GoalWM: episode return under the --ext-stage weights alone
    if start:
        params.update({"env.version": start.get("env_version", ""), "env.bots": str(start.get("bots", "")), "env.shards": str(start["shards"]), "env.game_speed": str(start.get("speed", 1)),
                       "env.episode_limit": str(start.get("max_steps", "")), "env.region": str(start.get("region", ""))})
    ups = [x for x in tl if x.get("event") == "update"]
    pbt = [x for x in tl if x.get("event") == "pbt"]
    stopped = bool(tl) and tl[-1].get("event") == "stop"
    metrics = {"env_steps_total": float(max([x.get("total_steps", 0) for x in ups] or [0])), "params_total": float(m["params_total"]),
               "params_deployed": float(m["params_deployed"]), "updates": float(len(ups)), "episodes": float(len(eps)),
               "pbt_exploits": float(sum(1 for x in pbt if x.get("exploit"))), "env_runs": float(len(runs))}
    if steps:
        ok = np.array([s["ok"] for s in steps], float)
        metrics["cmd_success_rate"] = float(ok.mean())
        metrics["cmd_success_rate_last1000"] = float(ok[-1000:].mean())
        n = collections.Counter(s["cmd"] for s in steps)
        for c, k in n.items():
            metrics[f"cmd_share/{c}"] = k / len(steps)
            metrics[f"cmd_success/{c}"] = float(np.mean([s["ok"] for s in steps if s["cmd"] == c]))
        for r, k in collections.Counter(f'{s["cmd"]}:{s["reason"]}' for s in steps if not s["ok"]).most_common(8):
            metrics[f"fail_share/{r}"] = k / len(steps)
    series = collections.defaultdict(list)
    members = sorted({x.get("member", 0) for x in ups} | {e.get("member", 0) for e in eps}) or [0]
    pre = lambda i: f"m{i}/" if len(members) > 1 else ""
    for i in members:
        mu = [x for x in ups if x.get("member", 0) == i]
        me = [e for e in eps if e.get("member", 0) == i]
        for x in stride(mu):
            series[pre(i) + "entropy"].append((x["update"], x["ent"]))
            series[pre(i) + "reward_mean"].append((x["update"], x["reward_mean"]))
            if goal:
                for k in ("aux_ok", "aux_inv", "aux_reach", "u_gain"):
                    series[k].append((x["update"], x[k]))
                g = x.get("goals", {})
                series["goals_reached_rate"].append((x["update"], g.get("reached", 0) / max(g.get("assigned", 0), 1)))
        if mu:
            metrics[pre(i) + "entropy"] = mu[-1]["ent"]
            if goal:
                metrics.update({k: float(mu[-1][k]) for k in ("aux_ok", "aux_dt", "aux_inv", "aux_reach", "u_ok", "u_dt", "u_gain")})
                g = mu[-1].get("goals", {})
                metrics.update({"goals_assigned": float(g.get("assigned", 0)), "goals_reached": float(g.get("reached", 0)),
                                "task_goals_reached": float(g.get("task_reached", 0)), "goal_archive_dims": float(mu[-1].get("archive", 0))})
        if me:
            ret = np.array([e[rkey] for e in me])
            last = me[-100:]
            if goal:
                metrics[pre(i) + "goals_reached_per_episode_last100"] = float(np.mean([e.get("goals_reached", 0) for e in last]))
            metrics.update({pre(i) + "return_stage_last100": float(np.mean([e[rkey] for e in last])),
                            pre(i) + "nights_last100": float(np.mean([e["nights"] for e in last])),
                            pre(i) + "achievements_last100": float(np.mean([e["achievements"] for e in last])),
                            pre(i) + "invalid_last100": float(np.mean([e["terms"].get("invalid", 0.0) for e in last]))})
            roll = np.convolve(ret, np.ones(min(20, len(ret))) / min(20, len(ret)), mode="valid")
            series[pre(i) + "return_stage_roll20"] = [(j, float(v)) for j, v in stride(list(enumerate(roll)))]
    mid = sch.model_id(m["name"])
    tags = {"model_id": mid, "family": "botsystem", "protocol": "side-botsystem", "role": "train", "arm": m["name"],
            "source_key": f"exchange/{base}", "end_reason": "interrupted" if stopped else "running", "stage": m.get("stage", ""),
            "params_source": "logged" if m.get("args") else "defaults", "git_commit": git(REPO), "git_commit_env": git(aip),
            "population": json.dumps(m.get("population") or [])[:500], "oracle_control": "false"}
    t0 = int(min([x["t"] for x in tl] or [0]) * 1000)
    return {"base": base, "model_id": mid, "start_ms": t0, "params": params, "metrics": metrics, "series": series, "tags": tags,
            "status": "KILLED" if stopped else "RUNNING"}


def log(spec):
    import datetime as dt
    import mlflow
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param, RunTag
    c = MlflowClient()
    exp = c.get_experiment_by_name(EXPERIMENT) or c.get_experiment(c.create_experiment(EXPERIMENT))
    for r in c.search_runs([exp.experiment_id], f"tags.source_key = '{spec['tags']['source_key']}'"):
        c.delete_run(r.info.run_id)
    stamp = dt.datetime.fromtimestamp(spec["start_ms"] / 1000, dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    run = c.create_run(exp.experiment_id, start_time=spec["start_ms"], run_name=f"{spec['model_id']}_{stamp}", tags=spec["tags"])
    now = int(dt.datetime.now().timestamp() * 1000)
    ms = [Metric(k, float(v), now, 0) for k, v in spec["metrics"].items()]
    ms += [Metric(k, float(v), now, int(s)) for k, pts in spec["series"].items() for s, v in pts]
    c.log_batch(run.info.run_id, metrics=ms[:900], params=[Param(k, v) for k, v in spec["params"].items()], tags=[RunTag("mlflow.note.content", NOTE)])
    for i in range(900, len(ms), 900):
        c.log_batch(run.info.run_id, metrics=ms[i:i + 900])
    if spec["status"] != "RUNNING":
        c.set_terminated(run.info.run_id, spec["status"])
    return run.info.run_id


NOTE = """BOTSYSTEM skill-level training (side track, never a gate). Model: SkillTruck, a Truck (Idea4_O01_D01_S00) derivative,
docs: forge1201-MODED-AIPlayer docs/implemented/MODEL-SKILLTRUCK.md. Metrics: return_stage = episode return under the stage's
unmodified reward weights; cmd_* = the environment's commands (share of steps, server-confirmed success); m<i>/ = population member."""


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("exchange"), p.add_argument("--base", nargs="*"), p.add_argument("--store", default=default_store())
    a = p.parse_args()
    import mlflow
    Path(a.store).mkdir(exist_ok=True)
    mlflow.set_tracking_uri(store_uri(a.store))
    root, sch = Path(a.exchange), Schema()
    bases = a.base or sorted(d.name for d in root.iterdir() if (d / "train" / "train_log.jsonl").exists())
    for b in bases:
        spec = build(sch, root, b, root.parent)
        if spec is None:
            print(f"{b}: no model event, skipped")
            continue
        rid = log(spec)
        print(f"{b}: run {rid} ({spec['model_id']}, {spec['status']}), {len(spec['params'])} params, {len(spec['metrics'])} metrics")
    from mlflow_report import sync_models   # the Models tab follows MODEL_REGISTRY.yaml (S08/S09 included)
    print(sync_models(a.store))
    if sch.unknown:
        print("unknown parameter names (add them to MLFLOW_PARAM_NAMES.yaml):", dict(sch.unknown))


if __name__ == "__main__":
    main()
