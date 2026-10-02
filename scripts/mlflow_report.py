"""Aggregate the local MLflow store by model (replaces re-reading the LaTeX notes for 'what did X score?').

  python scripts/mlflow_report.py summary [--md out.md]   # per experiment (phase), per model; mean/SE computed here from per-seed runs
  python scripts/mlflow_report.py top [-n 10] [--by score_pct|reward_pct] [--scope phase1|legacy|all] [--oracle]  # light leaderboard
  python scripts/mlflow_report.py ach [-n 8] [--protocol EP-A]  # named per-achievement success rates (22 rows) of the top groups
  python scripts/mlflow_report.py models                  # rebuild the MLflow Models tab from MODEL_REGISTRY.yaml (ingest does it too)
  python scripts/mlflow_report.py arch NAME               # every run whose model_id / arm / architecture contains NAME
  python scripts/mlflow_report.py size [--warn-gb 1.0]    # DB size / row counts; warns when it is time to consider Cloudflare D1
  python scripts/mlflow_report.py ui                      # prints the command to browse the store

Numbers are copied from the ingested result files; only mean and SE over seeds are computed here (SE = std(ddof=1)/sqrt(n)).
Runs that did not finish (status FAILED/KILLED) are never aggregated; they are listed in a separate section.
phase-1 protocol EP-A is the only gating protocol; other protocols are screening. `oracle_ctl=true` rows are upper-bound controls
(CLAUDE.md P3). phase-0-legacy rows predate the roadmap and are never gate evidence.
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlflow_ingest import (ACHIEVEMENT_NAMES, LEGACY_EXP, METRIC_GUIDE, REGISTRY_PATH, UNREGISTERED, default_store,  # noqa: E402
                           store_uri)

ACH_COLS = [f"ach_rate_pct/{i:02d}_{n}" for i, n in enumerate(ACHIEVEMENT_NAMES)]
GROUP = ["tag.model_id", "tag.arm", "tag.protocol", "tag.role", "param.train.learning_rate", "param.train.rollout_length",
         "param.train.num_envs", "param.train.total_env_steps"]


def load(store):
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(store_uri(store))
    c = MlflowClient()
    rows = []
    for e in c.search_experiments():
        tok = None
        while True:
            page = c.search_runs([e.experiment_id], max_results=1000, page_token=tok)
            for r in page:
                rows.append({**{f"tag.{k}": v for k, v in r.data.tags.items()}, **r.data.metrics,
                             **{f"param.{k}": v for k, v in r.data.params.items()}, "run_id": r.info.run_id,
                             "run_name": r.info.run_name, "status": r.info.status, "experiment": e.name})
            tok = page.token
            if not tok:
                break
    return pd.DataFrame(rows)


def fmt(x, nd=2):
    return "–" if pd.isna(x) else f"{x:.{nd}f}"


def pm(s, nd=2):
    s = s.dropna()
    if s.empty:
        return "–"
    se = s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else float("nan")
    return f"{s.mean():.{nd}f} ± {fmt(se, nd)}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def phase1_groups(df):
    """One (key, seed DataFrame) per configuration (model / arm / protocol / role / main hyperparameters).
    The same result file copied into several dirs counts once; `dirs` lists where it was found."""
    df = df[df["tag.family"] == "phase1"].copy()
    if df.empty:
        return []
    key = ["tag.arm", "tag.protocol", "param.seed.train", "reward_pct", "score_pct", "tag.git_commit", "param.train.learning_rate",
           "param.train.rollout_length"]
    dirs = df.groupby(key, dropna=False)["tag.source_dir"].agg(lambda x: sorted(set(x))).rename("dirs")
    df = df.drop_duplicates(key).merge(dirs, left_on=key, right_index=True)
    return list(df.groupby(GROUP, dropna=False))


def section_phase(df, exp):
    groups = phase1_groups(df)
    if not groups:
        return None
    rows = []
    for k, g in groups:
        all_dirs = sorted({d.split("/")[-1] for ds in g["dirs"] for d in ds})
        rows.append({"model_id": k[0], "arm": k[1], "protocol": k[2].split(" (")[0], "role": k[3], "lr": k[4], "n": len(g),
                     "env_steps": f"{g['env_steps_total'].iloc[0]:,.0f}", "params_total": f"{g['params_total'].iloc[0]:,.0f}",
                     "reward_pct": pm(g["reward_pct"]), "score_pct": pm(g["score_pct"]),
                     "oracle_ctl": g["tag.oracle_control"].iloc[0], "dirs": ",".join(all_dirs[:3]) + ("…" if len(all_dirs) > 3 else ""),
                     "_p": 0 if k[2] == "EP-A" else 1, "_k": g["reward_pct"].mean()})
    t = pd.DataFrame(rows).sort_values(["_p", "protocol", "role", "_k"], ascending=[True, True, True, False]).drop(columns=["_p", "_k"])
    return (f"{exp}  (mean ± SE over n seeds; reward_pct = % of 22 achievements, score_pct = Crafter score; "
            "only protocol EP-A gates a phase, others are screening)"), t


def section_candidates(df):
    df = df[df["tag.family"] == "candidate-result"]
    if df.empty:
        return None
    df = df.drop_duplicates(["tag.tag", "final_sampled/crafter_score"])  # same run copied into several dirs counts once
    rows = []
    for (arch, var, orc), g in df.groupby(["tag.architecture", "tag.variant", "tag.oracle_control"]):
        s = g["final_sampled/crafter_score"]
        rows.append({"architecture": arch, "variant": var, "mode": g["tag.mode"].iloc[0], "runs": len(g), "seeds": g["tag.seed"].nunique(),
                     "oracle_ctl": orc,
                     "crafter_score": pm(s), "best seed": fmt(s.max()), "mean_unlocked": pm(g["final_sampled/mean_unlocked"]),
                     "_k": s.mean()})
    t = pd.DataFrame(rows).sort_values("_k", ascending=False).drop(columns="_k")
    return "candidate-results  (legacy, final_sampled, mean ± SE over runs of one configuration (reruns of a seed count as runs); internal screening)", t


def section_latency(df):
    df = df[df["tag.family"] == "latency-chunking"]
    if df.empty:
        return None
    df = df.drop_duplicates(["param.env.id", "tag.arm", "param.env.latency_delta", "train_return_settled_mean", "final/eval_return"])
    rows = []
    for (env, arm, delta, orc), g in df.groupby(["param.env.id", "tag.arm", "param.env.latency_delta", "tag.oracle_control"],
                                                 dropna=False):
        rows.append({"env": env, "arm": arm, "delta": delta, "oracle_ctl": orc, "n": len(g),
                     "train_return_settled": pm(g["train_return_settled_mean"], 3), "best seed": fmt(g["train_return_settled_mean"].max(), 3),
                     "final eval_return": pm(g["final/eval_return"], 3)})
    t = pd.DataFrame(rows).sort_values(["env", "delta", "train_return_settled"], ascending=[True, True, False])
    return "latency-chunking  (legacy, settled train return; one row per env/arm/delta, mean ± SE over seeds)", t


def section_unfinished(df):
    bad = df[df["status"] != "FINISHED"]
    if bad.empty:
        return "Failed / interrupted runs", pd.DataFrame([{"note": "none recorded"}])
    t = bad.groupby(["experiment", "status", "tag.end_reason", "tag.model_id"]).size().rename("runs").reset_index()
    t.columns = ["experiment", "status", "end_reason", "model_id", "runs"]
    return "Failed / interrupted runs  (excluded from every aggregate above)", t


def cmd_summary(a):
    df = load(a.store)
    out = []
    ok = df[df["status"] == "FINISHED"] if not df.empty else df
    for exp in sorted(df["experiment"].unique()) if not df.empty else []:
        sub = ok[ok["experiment"] == exp]
        secs = [section_phase(sub, exp)] if exp != LEGACY_EXP else [section_candidates(sub), section_latency(sub)]
        for s in secs:
            if s:
                out += [f"## {s[0]}", "", md_table(s[1]), ""]
    if not df.empty:
        s = section_unfinished(df)
        out += [f"## {s[0]}", "", md_table(s[1]), ""]
    text = "\n".join(out)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n", encoding="utf-8")


def leaderboard_rows(df, scope, by, oracle):
    """Ranked configurations: phase-1 groups (a board per protocol) and/or legacy candidate groups."""
    rows = []
    if scope in ("phase1", "all"):
        for k, g in phase1_groups(df):
            rows.append({"board": f"phase-1 {k[2].split(' (')[0]}", "name": k[0] if k[0] != UNREGISTERED else k[1], "arm": k[1],
                         "role": k[3], "n": len(g), "oracle": g["tag.oracle_control"].iloc[0], "g": g})
    if scope in ("legacy", "all"):
        c = df[(df["tag.family"] == "candidate-result") & df["score_pct"].notna()].drop_duplicates(["tag.tag", "score_pct"])
        for (arch, var, orc), g in c.groupby(["tag.architecture", "tag.variant", "tag.oracle_control"]):
            rows.append({"board": "legacy (candidate-results)", "name": f"{arch}/{var}", "arm": g["tag.mode"].iloc[0], "role": "screen",
                         "n": len(g), "oracle": orc, "g": g})
    for r in rows:
        r["_k"] = r["g"][by].mean()
    return [r for r in rows if (oracle or r["oracle"] != "true") and pd.notna(r["_k"])]


def finished(df):
    return df[df["status"] == "FINISHED"] if not df.empty else df


def cmd_top(a):
    rows = leaderboard_rows(finished(load(a.store)), a.scope, a.by, a.oracle)
    if not rows:
        print("no finished runs for this scope")
        return 1
    for board in sorted({r["board"] for r in rows}, key=lambda b: (b != "phase-1 EP-A", b)):
        mine = [r for r in rows if r["board"] == board]
        top = sorted(mine, key=lambda r: -r["_k"])[:a.n]
        t = pd.DataFrame([{"#": i + 1, "name": r["name"], "arm": r["arm"], "role": r["role"], "n": r["n"],
                           "score_pct": pm(r["g"]["score_pct"]), "reward_pct": pm(r["g"]["reward_pct"]), "oracle_ctl": r["oracle"]}
                          for i, r in enumerate(top)])
        print(f"## {board}: top {len(top)} of {len(mine)} by mean {a.by} (mean ± SE over n seeds)\n")
        print(md_table(t) + "\n")
    if not a.oracle:
        print("(oracle upper-bound controls hidden; --oracle shows them)")


def cmd_ach(a):
    rows = [r for r in leaderboard_rows(finished(load(a.store)), "phase1", "score_pct", a.oracle) if r["board"] == f"phase-1 {a.protocol}"]
    top = sorted(rows, key=lambda r: -r["_k"])[:a.n]
    if not top:
        print(f"no phase-1 {a.protocol} groups")
        return 1
    cols = [f"{i + 1}:{r['name']}" for i, r in enumerate(top)]
    body = [{"achievement": c.split("/", 1)[1], **{cn: fmt(r["g"][c].mean()) if c in r["g"] else "–" for cn, r in zip(cols, top)}}
            for c in ACH_COLS]
    body.append({"achievement": "score_pct = geometric mean of the 22: exp(mean ln(1+s)) - 1",
                 **{cn: fmt(r["g"]["score_pct"].mean()) for cn, r in zip(cols, top)}})
    body.append({"achievement": "reward_pct = arithmetic mean of the 22", **{cn: fmt(r["g"]["reward_pct"].mean()) for cn, r in zip(cols, top)}})
    print(f"## success rate (%) per achievement, mean over seeds; top {len(top)} phase-1 {a.protocol} groups by score_pct\n")
    print(md_table(pd.DataFrame(body)))


def sync_models(store):
    """Rebuild the MLflow Models tab from MODEL_REGISTRY.yaml: one Registered Model per model_id, one version per phase-1
    configuration (linked to its first-seed run; no artifact). Version descriptions/tags hold a derived snapshot of mean ± SE and
    are rewritten on every sync. proper_name is copied from the registry (operator-only), never invented here."""
    import re
    import yaml
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException
    ok = finished(load(store))
    groups = {}
    for k, g in phase1_groups(ok):
        groups.setdefault(k[0], []).append((k, g))
    reg = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))["models"]
    c = MlflowClient()
    n_ver = 0
    shown = 0
    for mid, m in reg.items():
        try:
            c.delete_registered_model(mid)
        except MlflowException:
            pass
        if m.get("status") == "archived":   # dead ends stay in the registry (run -> model_id) but are not shown on the Models tab
            continue
        shown += 1
        tags = {"idea": mid.split("_")[0], "status": str(m.get("status")), "arms": ",".join(m.get("arms", [])), "source": "MODEL_REGISTRY.yaml"}
        if m.get("proper_name"):
            tags["proper_name"] = str(m["proper_name"])
        c.create_registered_model(mid, tags=tags, description=str(m.get("description", "")))
        best = {}
        for k, g in sorted(groups.get(mid, []), key=lambda kg: kg[1]["score_pct"].mean()):  # ascending: the best is created last
            proto = k[2].split(" (")[0]
            desc = (f"{k[1]} | {proto} | role={k[3]} | lr={k[4]} rollout={k[5]} envs={k[6]} steps={k[7]} | n={len(g)} seeds | "
                    f"score_pct {pm(g['score_pct'])} | reward_pct {pm(g['reward_pct'])}")
            vt = {"protocol": proto, "role": str(k[3]), "n_seeds": str(len(g)), "score_pct_mean": f"{g['score_pct'].mean():.3f}",
                  "reward_pct_mean": f"{g['reward_pct'].mean():.3f}", "gate_eligible": str(proto == "EP-A").lower(),
                  "oracle_control": str(g["tag.oracle_control"].iloc[0])}
            rid = g.sort_values("param.seed.train")["run_id"].iloc[0]
            best[proto] = c.create_model_version(mid, source=f"runs:/{rid}/model", run_id=rid, tags=vt, description=desc).version
            n_ver += 1
        for proto, ver in best.items():
            c.set_registered_model_alias(mid, "best-" + re.sub(r"[^A-Za-z0-9_-]+", "-", proto).strip("-").lower(), ver)
    return f"models tab: {shown} registered models ({len(reg) - shown} archived hidden), {n_ver} versions (alias best-<protocol> = highest mean score_pct)"


def cmd_models(a):
    import mlflow
    mlflow.set_tracking_uri(store_uri(a.store))
    print(sync_models(a.store))


def cmd_arch(a):
    df = load(a.store)
    cols = ["tag.model_id", "tag.arm", "tag.architecture"]
    hit = df[df[[c for c in cols if c in df]].apply(lambda c: c.str.contains(a.name, case=False, na=False)).any(axis=1)]
    if hit.empty:
        print("no match")
        return 1
    keep = [c for c in ("run_name", "experiment", "status", "tag.end_reason", "tag.protocol", "tag.role", "param.seed.train", "reward_pct",
                        "score_pct", "final_sampled/crafter_score", "train_return_settled_mean", "tag.source_key") if c in hit]
    t = hit[keep].rename(columns=lambda c: c.replace("tag.", "").replace("param.", "")).sort_values("run_name")
    print(f"## {len(hit)} runs matching {a.name!r}")
    print(md_table(t.map(lambda v: fmt(v, 3) if isinstance(v, float) else v)))


def cmd_size(a):
    db = Path(a.store) / "mlflow.db"
    if not db.exists():
        print(f"no store at {db}")
        return 1
    gb = db.stat().st_size / 2**30
    con = sqlite3.connect(db)
    q = lambda t: con.execute(f"select count(*) from {t}").fetchone()[0]  # noqa: E731
    print(f"{db}: {gb:.3f} GiB | runs {q('runs')} | metric rows {q('metrics')} | params {q('params')} | tags {q('tags')}")
    if gb >= a.warn_gb:
        print(f"WARNING: store >= {a.warn_gb} GiB. Report to the operator; move to Cloudflare D1 (limit 10 GB/db) as agreed.")
    else:
        print(f"OK: below the {a.warn_gb} GiB report threshold; local SQLite is sufficient.")
    return 0


def cmd_ui(a):
    print(f"mlflow ui --backend-store-uri {store_uri(a.store)}\n")
    print("Light views in the Runs tab (paste into the search box, sort by the metrics.score_pct column, hide columns via 'Columns'):")
    print('  real runs only : tags.family = "phase1" and tags.oracle_control = "false" and metrics.score_pct > 4')
    print('  gate protocol  : tags.gate_eligible = "true"')
    print('  one model      : tags.model_id = "Idea4_O01_D01_S00"')
    print("  'Group by' tags.model_id gives one aggregated row per model. Per-achievement columns: metrics.ach_rate_pct/NN_name.")
    print("  The experiment description (top of the experiment page) holds the metric definitions and the achievement index:\n")
    print(METRIC_GUIDE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=default_store())
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("summary")
    s.add_argument("--md")
    s.set_defaults(fn=cmd_summary)
    r = sub.add_parser("arch")
    r.add_argument("name")
    r.set_defaults(fn=cmd_arch)
    z = sub.add_parser("size")
    z.add_argument("--warn-gb", type=float, default=1.0)
    z.set_defaults(fn=cmd_size)
    sub.add_parser("ui").set_defaults(fn=cmd_ui)
    t = sub.add_parser("top")
    t.add_argument("-n", type=int, default=10)
    t.add_argument("--by", choices=("score_pct", "reward_pct"), default="score_pct")
    t.add_argument("--scope", choices=("phase1", "legacy", "all"), default="phase1")
    t.add_argument("--oracle", action="store_true", help="include oracle upper-bound controls")
    t.set_defaults(fn=cmd_top)
    h = sub.add_parser("ach")
    h.add_argument("-n", type=int, default=8)
    h.add_argument("--protocol", default="EP-A", help="exact protocol label, e.g. EP-A, EP-A-mini")
    h.add_argument("--oracle", action="store_true")
    h.set_defaults(fn=cmd_ach)
    sub.add_parser("models").set_defaults(fn=cmd_models)
    a = ap.parse_args()
    return a.fn(a) if a.cmd else cmd_summary(argparse.Namespace(store=a.store, md=None))


if __name__ == "__main__":
    sys.exit(main())
