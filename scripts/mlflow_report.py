"""Aggregate the local MLflow store by model (replaces re-reading the LaTeX notes for 'what did X score?').

  python scripts/mlflow_report.py summary [--md out.md]   # per experiment (phase), per model; mean/SE computed here from per-seed runs
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
from mlflow_ingest import LEGACY_EXP, default_store, store_uri  # noqa: E402


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


def section_phase(df, exp):
    df = df[df["tag.family"] == "phase1"].copy()
    if df.empty:
        return None
    key = ["tag.arm", "tag.protocol", "param.seed.train", "reward_pct", "score_pct", "tag.git_commit", "param.train.learning_rate",
           "param.train.rollout_length"]
    dirs = df.groupby(key, dropna=False)["tag.source_dir"].agg(lambda x: sorted(set(x))).rename("dirs")  # same file copied into several dirs
    df = df.drop_duplicates(key).merge(dirs, left_on=key, right_index=True)
    rows = []
    grp = ["tag.model_id", "tag.arm", "tag.protocol", "tag.role", "param.train.learning_rate", "param.train.rollout_length",
           "param.train.num_envs", "param.train.total_env_steps"]
    for k, g in df.groupby(grp, dropna=False):
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
    print(f"mlflow ui --backend-store-uri {store_uri(a.store)}")


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
    a = ap.parse_args()
    return a.fn(a) if a.cmd else cmd_summary(argparse.Namespace(store=a.store, md=None))


if __name__ == "__main__":
    sys.exit(main())
