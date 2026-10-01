"""Aggregate the local MLflow store by architecture (replaces re-reading the LaTeX notes for 'what did X score?').

  python scripts/mlflow_report.py summary [--md out.md]   # per family, per architecture
  python scripts/mlflow_report.py arch NAME               # every run of one architecture (substring match)
  python scripts/mlflow_report.py size [--warn-gb 1.0]    # DB size / row counts; warns when it is time to consider Cloudflare D1
  python scripts/mlflow_report.py ui                      # prints the command to browse the store

Numbers are copied from the ingested result files; nothing is recomputed. Families use different protocols and are NOT
comparable to each other (phase1 = EP-A gating protocol; candidate-results / latency = internal screening). `oracle_control=true`
rows are upper-bound controls and never count toward a gate (CLAUDE.md P3).
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlflow_ingest import EXP_LATENCY, EXP_PHASE1, EXP_RESULT, default_store, store_uri  # noqa: E402


def load(store):
    from mlflow import MlflowClient
    import mlflow
    mlflow.set_tracking_uri(store_uri(store))
    c = MlflowClient()
    out = {}
    for name in (EXP_PHASE1, EXP_RESULT, EXP_LATENCY):
        e = c.get_experiment_by_name(name)
        rows, tok = [], None
        while e is not None:
            page = c.search_runs([e.experiment_id], max_results=1000, page_token=tok)
            for r in page:
                rows.append({**{f"tag.{k}": v for k, v in r.data.tags.items()}, **r.data.metrics,
                             **{f"param.{k}": v for k, v in r.data.params.items()}, "run_id": r.info.run_id,
                             "run_name": r.info.run_name})
            tok = page.token
            if not tok:
                break
        out[name] = pd.DataFrame(rows)
    return out


def fmt(x, nd=2):
    return "–" if pd.isna(x) else f"{x:.{nd}f}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def section_phase1(df):
    df = df[df["tag.level"] == "aggregate"]
    if df.empty:
        return None
    rows = []
    # the same result file is often copied into several dirs (final_1000k_all, *_derived, ...): merge identical rows
    for _, g in df.groupby(["tag.architecture", "tag.arm", "tag.protocol", "reward_pct_mean", "score_pct_mean", "tag.git_commit", "param.cfg.lr"]):
        r = g.iloc[0]
        dirs = sorted(set(g["tag.source_dir"].str.removeprefix("phase1/")))
        rows.append({"dir": dirs[0] + (f" (+{len(dirs) - 1} copies)" if len(dirs) > 1 else ""), "architecture": r["tag.architecture"],
                     "arm": r["tag.arm"], "protocol": r["tag.protocol"].split(" (")[0], "lr": r["param.cfg.lr"], "seeds": int(r["n_seeds"]),
                     "env_steps": f"{r['env_steps_total']:,.0f}", "params_total": f"{r['params_total']:,.0f}",
                     "reward_pct": f"{fmt(r['reward_pct_mean'])} ± {fmt(r['reward_pct_se'])}",
                     "score_pct": f"{fmt(r['score_pct_mean'])} ± {fmt(r['score_pct_se'])}",
                     "_k": r["reward_pct_mean"]})
    t = pd.DataFrame(rows).sort_values(["dir", "_k"], ascending=[True, False]).drop(columns="_k")
    return ("phase1-epa  (mean ± SE over seeds; reward_pct = % of 22 achievements, score_pct = Crafter score; "
            "only protocol EP-A gates a phase, EP-A-mini / -partial are screening)"), t


def section_result(df):
    if df.empty:
        return None
    rows = []
    for (arch, var, orc), g in df.groupby(["tag.architecture", "tag.variant", "tag.oracle_control"]):
        ok = g[g["tag.status"] != "failed"]
        s = ok["final_sampled/crafter_score"] if "final_sampled/crafter_score" in ok else pd.Series(dtype=float)
        rows.append({"architecture": arch, "variant": var, "mode": g["tag.mode"].iloc[0], "seeds": g["tag.seed"].nunique(),
                     "failed": len(g) - len(ok), "oracle_ctl": orc,
                     "crafter_score mean": fmt(s.mean()), "best seed": fmt(s.max()),
                     "unlocked mean": fmt(ok.get("final_sampled/mean_unlocked", pd.Series(dtype=float)).mean()), "_k": s.mean()})
    t = pd.DataFrame(rows).sort_values("_k", ascending=False).drop(columns="_k")
    return "candidate-results  (final_sampled, mean over seeds of one configuration; internal screening; oracle_ctl=true rows are upper-bound controls)", t


def section_latency(df):
    if df.empty:
        return None
    rows = []
    for (env, arm, actor, delta), g in df.groupby(["tag.env", "tag.arm", "tag.actor", "param.cfg.delta"]):
        rows.append({"env": env, "arm": arm, "actor": actor, "delta": delta, "oracle_ctl": g["tag.oracle_control"].iloc[0], "runs": len(g),
                     "train_return_settled mean": fmt(g["train_return_settled_mean"].mean(), 3),
                     "best seed": fmt(g["train_return_settled_mean"].max(), 3),
                     "final eval_return mean": fmt(g.get("final/eval_return", pd.Series(dtype=float)).mean(), 3)})
    t = pd.DataFrame(rows).sort_values(["env", "delta", "train_return_settled mean"], ascending=[True, True, False])
    return "latency-chunking  (settled train return; one row per env/arm/actor/delta, mean over seeds)", t


def cmd_summary(a):
    data = load(a.store)
    out = []
    for name, fn in ((EXP_PHASE1, section_phase1), (EXP_RESULT, section_result), (EXP_LATENCY, section_latency)):
        s = fn(data[name]) if not data[name].empty else None
        if s:
            out += [f"## {s[0]}", "", md_table(s[1]), ""]
    text = "\n".join(out)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n", encoding="utf-8")


def cmd_arch(a):
    data = load(a.store)
    for name, df in data.items():
        if df.empty:
            continue
        hit = df[df["tag.architecture"].str.contains(a.name, case=False, na=False)]
        hit = hit[hit.get("tag.level", "run") != "seed"] if "tag.level" in hit else hit
        if hit.empty:
            continue
        keep = [c for c in ("run_name", "tag.source_dir", "tag.protocol", "reward_pct_mean", "score_pct_mean",
                            "final_sampled/crafter_score", "final_sampled/mean_unlocked", "train_return_settled_mean",
                            "tag.source_key") if c in hit]
        print(f"## {name}: {len(hit)} runs")
        print(md_table(hit[keep].rename(columns=lambda c: c.replace("tag.", "")).map(lambda v: fmt(v, 3) if isinstance(v, float) else v)))
        print()


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
    return (a.fn or cmd_summary)(a) if a.cmd else cmd_summary(argparse.Namespace(store=a.store, md=None))


if __name__ == "__main__":
    sys.exit(main())
