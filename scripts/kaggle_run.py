"""
Kaggle Phase II Automation Orchestrator — with Live Log Streaming.

Single entry point for the full pipeline:
  Specify hyperparams → Build notebook → Push → Poll with live logs →
  Fetch results → Deserialize W → Analyze → Plot

Features (updated v2):
  - Live log streaming: periodically fetches and prints new kernel log lines
    while the kernel runs, so errors are visible immediately.
  - On ERROR status: prints the full error traceback from the log and exits.
  - On COMPLETE: auto-fetches outputs, loads W, analyzes results, plots.

Usage:
    # Single run (full pipeline)
    python scripts/kaggle_run.py --run-id phase2_002 --n-layers 6 --beam-width 5 \\
        --z-step 64 --causal --train-steps 1000000 --eval-steps 500

    # Grid sweep
    python scripts/kaggle_run.py --sweep \\
        --n-layers 4 6 8 --beam-width 5 8 --z-step 32 64 --train-steps 1000000

    # Fetch + analyze + plot for an already-completed kernel
    python scripts/kaggle_run.py --fetch-only --run-id phase2_001

    # Re-plot from already-fetched output_remote/<run-id>/
    python scripts/kaggle_run.py --plot-only --run-id phase2_001
"""

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
KERNEL_ID     = "bfloat16/craftax-classic-1000-episode-rl-benchmark"
KERNEL_SLUG   = "craftax-classic-1000-episode-rl-benchmark"   # filename prefix in outputs
NOTEBOOK_SRC  = "kaggle_kernel/decision_process_benchmark.ipynb"
META_SRC      = "kaggle_kernel/kernel-metadata.json"
OUTPUT_DIR    = Path("output_remote")
POLL_INTERVAL = 30    # seconds between status checks
LOG_INTERVAL  = 60    # seconds between log-sync fetches (roughly)
MAX_WAIT_H    = 9     # Kaggle GPU session max ~9 hours


# ── Subprocess helpers ────────────────────────────────────────────────────────
def run_cmd(cmd: list, check: bool = True, capture: bool = False):
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if check and r.returncode != 0:
            print(f"[ERROR] Command failed: {' '.join(cmd)}\n{r.stderr}")
            sys.exit(1)
        return r.stdout.strip()
    else:
        subprocess.run(cmd, check=check)


def kaggle(*args, check: bool = True) -> str:
    raw = run_cmd(["kaggle"] + list(args), capture=True, check=check)
    # Strip the outdated-version warning that Kaggle CLI 2.0.0 prepends to every output.
    # We only want the actual payload line(s).
    lines = [l for l in raw.splitlines()
             if not l.lower().startswith("warning:")]
    return "\n".join(lines).strip()


# ── Log streaming helpers ─────────────────────────────────────────────────────
def _parse_log_jsonl(log_path: Path) -> list:
    """Parse the Kaggle JSONL log file into a list of {stream, time, data} dicts."""
    if not log_path.exists():
        return []
    raw = log_path.read_text(encoding="utf-8", errors="replace")
    entries = []
    for line in raw.splitlines():
        line = line.strip().lstrip(",").strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return entries


def _fetch_log(log_dir: Path) -> Path:
    """Download the current kernel output (log file) into log_dir. Silent on failure."""
    try:
        run_cmd(["kaggle", "kernels", "output", KERNEL_ID,
                 "-p", str(log_dir), "--force"], check=False, capture=True)
    except Exception:
        pass
    return log_dir / f"{KERNEL_SLUG}.log"


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


def _safe_print(text: str) -> None:
    """Print with cp932-safe fallback: strip ANSI codes, replace unencodable chars."""
    text = _ANSI_RE.sub('', text)   # strip all ANSI escape sequences
    try:
        print(text, flush=True)
    except (UnicodeEncodeError, UnicodeDecodeError):
        enc = sys.stdout.encoding or 'ascii'
        safe = text.encode(enc, errors='replace').decode(enc)
        print(safe, flush=True)


def _print_new_log_entries(entries: list, last_printed: int) -> int:
    """Print log entries from last_printed onward. Returns new last_printed index."""
    new = entries[last_printed:]
    for entry in new:
        stream = entry.get("stream_name", "stdout")
        data   = entry.get("data", "").rstrip("\r\n")
        t      = entry.get("time", 0.0)
        if not data.strip():
            continue
        ts = f"[{t:>8.1f}s]"
        if stream == "stderr":
            lower = data.lower()
            if any(k in lower for k in ("error", "traceback", "exception", "warning:")):
                _safe_print(f"{ts} [ERR] {data}")
        else:
            _safe_print(f"{ts} {data}")
    return last_printed + len(new)


# ── Build phase ───────────────────────────────────────────────────────────────
def build_notebook(cfg: dict) -> None:
    print(f"\n[BUILD] Generating notebook for run_id={cfg['run_id']}...")
    cmd = [
        sys.executable, "build_kaggle_kernel.py",
        "--run-id",     cfg["run_id"],
        "--n-layers",   str(cfg["n_layers"]),
        "--beam-width", str(cfg["beam_width"]),
        "--z-step",     str(cfg["z_step"]),
        "--train-steps",str(cfg["train_steps"]),
        "--eval-steps", str(cfg["eval_steps"]),
        "--seed",       str(cfg["seed"]),
        "--d-model",    str(cfg["d_model"]),
        "--num-heads",  str(cfg["num_heads"]),
        "--lr",         str(cfg["lr"]),
        "--noise-prob", str(cfg["noise_prob"]),
    ]
    if cfg["is_causal"]:
        cmd.append("--causal")
    else:
        cmd.append("--no-causal")
    run_cmd(cmd)
    print(f"[BUILD] Notebook ready: {NOTEBOOK_SRC}")


# ── Push phase ────────────────────────────────────────────────────────────────
def push_kernel() -> None:
    print(f"\n[PUSH] Pushing kernel: {KERNEL_ID}")
    out = kaggle("kernels", "push", "-p", "kaggle_kernel")
    print(f"[PUSH] {out}")


# ── Poll phase with live log streaming ───────────────────────────────────────
def poll_with_live_logs(run_id: str, log_dir: Path,
                        max_wait_s: int = MAX_WAIT_H * 3600) -> bool:
    """Poll kernel status, streaming new log lines every LOG_INTERVAL seconds.

    - Fetches the kernel output (log file) every LOG_INTERVAL seconds.
    - Parses the Kaggle JSONL log format and prints only new lines.
    - On ERROR: prints the full error section and returns False.
    - On COMPLETE: prints remaining logs and returns True.
    """
    print(f"\n[POLL] Monitoring kernel: {KERNEL_ID}")
    print(f"       Log sync every {LOG_INTERVAL}s | Status check every {POLL_INTERVAL}s")
    print(f"       Max wait: {max_wait_s/3600:.1f}h\n")

    log_dir.mkdir(parents=True, exist_ok=True)
    start         = time.time()
    last_log_sync = 0.0
    last_printed  = 0
    poll_count    = 0

    while time.time() - start < max_wait_s:
        now = time.time()

        # ── Periodic log sync ─────────────────────────────────────────────
        if now - last_log_sync >= LOG_INTERVAL:
            log_path = _fetch_log(log_dir)
            entries  = _parse_log_jsonl(log_path)
            last_printed = _print_new_log_entries(entries, last_printed)
            last_log_sync = now

        # ── Status check ──────────────────────────────────────────────────
        raw    = kaggle("kernels", "status", KERNEL_ID, check=False)
        status = raw.lower()
        poll_count += 1

        elapsed = now - start
        if poll_count % 4 == 0:   # print heartbeat every ~2 min
            print(f"[POLL] {datetime.now().strftime('%H:%M:%S')} "
                  f"status={status.strip()!r} elapsed={elapsed/60:.1f}min", flush=True)

        if "error" in status:
            # Final log fetch to get complete error traceback
            time.sleep(5)
            log_path = _fetch_log(log_dir)
            entries  = _parse_log_jsonl(log_path)
            last_printed = _print_new_log_entries(entries, last_printed)
            # Print the full stderr section for debugging
            _safe_print("\n[POLL] ===== KERNEL FAILED - full error log =====")
            for e in entries:
                if e.get("stream_name") == "stderr":
                    _safe_print(e.get("data", "").rstrip())
            _safe_print("[POLL] ================================================\n")
            return False

        if "complete" in status:
            # Final log flush
            time.sleep(3)
            log_path = _fetch_log(log_dir)
            entries  = _parse_log_jsonl(log_path)
            _print_new_log_entries(entries, last_printed)
            elapsed = time.time() - start
            print(f"\n[POLL] COMPLETE in {elapsed/60:.1f} min.", flush=True)
            return True

        time.sleep(POLL_INTERVAL)

    print(f"\n[POLL] WARNING: Timeout after {max_wait_s/3600:.1f}h.", flush=True)
    return False


# ── Fetch phase ───────────────────────────────────────────────────────────────
def fetch_outputs(run_id: str) -> Path:
    """Download all kernel outputs to output_remote/<run_id>/."""
    out_dir = OUTPUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[FETCH] Downloading outputs -> {out_dir}/")
    run_cmd(["kaggle", "kernels", "output", KERNEL_ID,
             "-p", str(out_dir), "--force"], check=False)
    files = [f for f in out_dir.rglob("*") if f.is_file()]
    print(f"[FETCH] Retrieved {len(files)} files:")
    for f in sorted(files):
        size_kb = f.stat().st_size / 1024
        print(f"        {f.relative_to(out_dir)}  ({size_kb:.1f} KB)")
    if not files:
        print("[FETCH] WARNING: 0 files. Kernel may still be running or had no outputs.")
    return out_dir


# ── Weight deserialization ────────────────────────────────────────────────────
def load_weights(out_dir: Path, run_id: str):
    """Load final model weights W from the pickle checkpoint."""
    for pattern in [f"{run_id}_final.pkl", "*.pkl"]:
        matches = list(out_dir.rglob(pattern))
        if matches:
            break
    if not matches:
        print(f"[WEIGHTS] WARNING: No checkpoint .pkl found in {out_dir}")
        return None
    ckpt = matches[0]
    print(f"[WEIGHTS] Loading W from: {ckpt}")
    with open(ckpt, "rb") as f:
        tree = pickle.load(f)
    leaf_count = sum(1 for _ in _iter_leaves(tree))
    print(f"[WEIGHTS] Loaded PyTree. Leaves: {leaf_count:,}")
    return tree


def _iter_leaves(tree):
    if hasattr(tree, "_fields"):
        for field in tree._fields:
            yield from _iter_leaves(getattr(tree, field))
    elif isinstance(tree, (list, tuple)):
        for item in tree:
            yield from _iter_leaves(item)
    elif isinstance(tree, dict):
        for v in tree.values():
            yield from _iter_leaves(v)
    else:
        yield tree


# ── Result analysis ───────────────────────────────────────────────────────────
def analyze_results(out_dir: Path, run_id: str) -> Optional[dict]:
    """Load results JSON and print a summary table."""
    # Search for results JSON
    for pattern in [f"{run_id}_results.json", "*results*.json", "*.json"]:
        matches = [p for p in out_dir.rglob(pattern)
                   if "src" not in str(p) and "config" not in p.name]
        if matches:
            result_path = matches[0]
            break
    else:
        print(f"[ANALYZE] WARNING: No results JSON found in {out_dir}")
        return None

    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    ev  = data.get("evaluation", {})
    tr  = data.get("training",   {})
    cfg = data.get("config",     {})

    w = 60
    print(f"\n{'='*w}")
    print(f"  RESULTS: {run_id}")
    print(f"{'='*w}")
    print(f"  Backend      : {data.get('hardware',{}).get('jax_backend','?').upper()}")
    print(f"  Parameters   : {data.get('hardware',{}).get('param_count',0):,}")
    print(f"  Train Steps  : {tr.get('total_steps',0):,}")
    print(f"  Train SPS    : {tr.get('final_sps',0):,.0f} steps/sec")
    print(f"  Eval SPS     : {ev.get('eval_sps',0):,.0f} steps/sec")
    print(f"  Crafter Score: {ev.get('crafter_score',0):.4f}")
    print(f"  Mean Reward  : {ev.get('mean_reward',0):.4f}")
    print(f"  Progress Rate: {ev.get('mean_progress_rate',0):.4f}")
    print(f"  Context Util : {ev.get('context_util_ratio',0):.3f}")
    print(f"\n  Milestones:")
    for name, ok in ev.get("milestone_results", {}).items():
        print(f"    {'[Y]' if ok else '[N]'} {name}")
    print(f"{'='*w}\n")
    return data


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(out_dir: Path, run_id: str, data: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[PLOT] matplotlib not available; skipping.")
        return

    ev  = data.get("evaluation", {})
    tr  = data.get("training",   {})
    cfg = data.get("config",     {})
    log = tr.get("log", [])

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    PLT = {
        "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d2e",
        "axes.edgecolor": "#3a3d5c",   "axes.labelcolor": "#e0e0ff",
        "xtick.color": "#9090bb",      "ytick.color": "#9090bb",
        "text.color": "#e0e0ff",       "grid.color": "#2a2d4a",
        "grid.linestyle": "--",        "grid.alpha": 0.5,
        "font.family": "monospace",
    }
    plt.rcParams.update(PLT)
    ACCENT, ACCENT2, GREEN, ORANGE = "#7c6af7", "#4fc3f7", "#69f0ae", "#ffb74d"

    def _save(fig, name):
        p = plot_dir / name
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[PLOT] {p}")

    # ── Training curves ───────────────────────────────────────────────────
    if log:
        steps  = [e["step"]          for e in log]
        losses = [e["loss"]          for e in log]
        policy = [e["policy_loss"]   for e in log]
        valid_ = [e["validity_loss"] for e in log]
        sps_v  = [e["sps"]           for e in log]

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(
            f"Training Curves  {run_id}\n"
            f"N={cfg.get('n_layers')}, k={cfg.get('beam_width')}, "
            f"Z={cfg.get('z_step')}, causal={cfg.get('is_causal')}",
            fontsize=12, color="#e0e0ff", y=1.01,
        )
        for ax, y, label, color in [
            (axes[0,0], losses, "Total Loss",    ACCENT),
            (axes[0,1], policy, "Policy Loss",   ACCENT2),
            (axes[1,0], valid_, "Validity Loss", GREEN),
            (axes[1,1], sps_v,  "SPS",           ORANGE),
        ]:
            ax.plot(steps, y, color=color, linewidth=1.8)
            ax.fill_between(steps, y, alpha=0.12, color=color)
            ax.set_title(label, fontsize=9, color="#e0e0ff")
            ax.set_xlabel("Steps", fontsize=8)
            ax.grid(True)
        plt.tight_layout()
        _save(fig, f"{run_id}_training_curves.png")

    # ── Achievement bars ──────────────────────────────────────────────────
    ach = ev.get("achievement_rates", {})
    if ach:
        names  = list(ach.keys())
        values = [ach[n] for n in names]
        ms     = {"collect_wood","collect_stone","collect_coal","collect_iron","collect_diamond"}
        colors = []
        for n, v in zip(names, values):
            if n in ms:
                colors.append(ORANGE if v > 0 else "#5a3d2c")
            else:
                colors.append(GREEN if v > 0 else "#3a3d5c")

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(
            f"Achievement Unlock Rates  {run_id}\nCrafter Score = {ev.get('crafter_score',0):.4f}",
            fontsize=11, color="#e0e0ff",
        )
        bars = ax.barh(names, values, color=colors, edgecolor="#3a3d5c", height=0.7)
        ax.set_xlabel("Unlock Rate (%)", fontsize=9)
        ax.set_xlim(0, 110)
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(v + 1, bar.get_y() + bar.get_height()/2,
                        f"{v:.1f}%", va="center", fontsize=7, color="#e0e0ff")
        ax.grid(True, axis="x")
        plt.tight_layout()
        _save(fig, f"{run_id}_achievements.png")

    # ── Summary card ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis("off")
    fig.patch.set_facecolor("#0f1117")
    rows = [
        ("Run ID",        run_id),
        ("Backend",       data.get("hardware",{}).get("jax_backend","?").upper()),
        ("Parameters",    f"{data.get('hardware',{}).get('param_count',0):,}"),
        ("N / k / Z",     f"{cfg.get('n_layers')} / {cfg.get('beam_width')} / {cfg.get('z_step')}"),
        ("Causal",        str(cfg.get("is_causal"))),
        ("Train Steps",   f"{cfg.get('train_steps',0):,}"),
        ("Train SPS",     f"{tr.get('final_sps',0):,.0f}"),
        ("Crafter Score", f"{ev.get('crafter_score',0):.4f}"),
        ("Coal unlock",   "[Y]" if ev.get("milestone_results",{}).get("collect_coal") else "[N]"),
        ("Diamond unlock","[Y]" if ev.get("milestone_results",{}).get("collect_diamond") else "[N]"),
    ]
    for i, (label, value) in enumerate(rows):
        y = 0.95 - i * 0.09
        ax.text(0.02, y, f"{label}:", transform=ax.transAxes,
                fontsize=10, color="#9090bb", va="top")
        ax.text(0.45, y, value, transform=ax.transAxes,
                fontsize=10, color="#e0e0ff", va="top", fontweight="bold")
    ax.set_title(f"Run Summary — {run_id}", color="#e0e0ff", fontsize=11, pad=10)
    _save(fig, f"{run_id}_summary_card.png")


# ── Combined Markdown table ───────────────────────────────────────────────────
def update_combined_table(run_id: str, data: dict) -> None:
    table_path = OUTPUT_DIR / "combined_table.md"
    ev  = data.get("evaluation", {})
    tr  = data.get("training",   {})
    cfg = data.get("config",     {})
    ms  = ev.get("milestone_results", {})

    HEADER = (
        "| Run ID | k | Z | N | Causal | Steps | "
        "Wood | Stone | Coal | Iron | Diamond | "
        "Crafter Score | Progress | Ctx Util | Train SPS | Eval SPS |\n"
        "|--------|---|---|---|--------|-------|"
        "-----|-------|------|------|---------|"
        "--------------|----------|----------|-----------|----------|\n"
    )

    yn = lambda v: "Y" if v else "N"
    row = (
        f"| {run_id} | {cfg.get('beam_width')} | {cfg.get('z_step')} "
        f"| {cfg.get('n_layers')} | {yn(cfg.get('is_causal'))} "
        f"| {cfg.get('train_steps',0):,} "
        f"| {yn(ms.get('collect_wood'))} | {yn(ms.get('collect_stone'))} "
        f"| {yn(ms.get('collect_coal'))} | {yn(ms.get('collect_iron'))} "
        f"| {yn(ms.get('collect_diamond'))} "
        f"| {ev.get('crafter_score',0):.4f} "
        f"| {ev.get('mean_progress_rate',0):.4f} "
        f"| {ev.get('context_util_ratio',0):.3f} "
        f"| {tr.get('final_sps',0):,.0f} "
        f"| {ev.get('eval_sps',0):,.0f} |\n"
    )
    if not table_path.exists():
        table_path.write_text(
            "# Phase II Combined Results\n\n"
            "**All scores from real GPU inference. No synthetic data.**\n\n"
            + HEADER,
            encoding="utf-8",
        )
    with open(table_path, "a", encoding="utf-8") as f:
        f.write(row)
    print(f"[TABLE] {table_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def single_run(cfg: dict, fetch_only: bool = False, plot_only: bool = False) -> None:
    run_id  = cfg["run_id"]
    log_dir = OUTPUT_DIR / run_id    # live logs stream here during polling

    if not (fetch_only or plot_only):
        build_notebook(cfg)
        push_kernel()
        success = poll_with_live_logs(run_id, log_dir)
        if not success:
            print(f"[RUN] Kernel errored. Fetching outputs for post-mortem.")

    if not plot_only:
        out_dir = fetch_outputs(run_id)
    else:
        out_dir = OUTPUT_DIR / run_id

    if fetch_only:
        # For fetch-only, poll first if kernel not yet done
        raw    = kaggle("kernels", "status", KERNEL_ID, check=False)
        status = raw.lower()
        if not ("complete" in status or "error" in status):
            print(f"[FETCH-ONLY] Kernel still running ({status.strip()!r}). Waiting...")
            success = poll_with_live_logs(run_id, log_dir)
            if not success:
                print("[FETCH-ONLY] Kernel errored. Fetching outputs for post-mortem.")
        out_dir = fetch_outputs(run_id)

    weights = load_weights(out_dir, run_id)
    data    = analyze_results(out_dir, run_id)
    if data:
        plot_results(out_dir, run_id, data)
        update_combined_table(run_id, data)

    print(f"\n[DONE] {run_id}")
    print(f"  Outputs : {out_dir}/")
    print(f"  Plots   : {out_dir}/plots/")
    print(f"  Table   : {OUTPUT_DIR / 'combined_table.md'}")
    if weights is not None:
        print(f"  Weights : loaded ({type(weights).__name__})")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Phase II Kaggle Automation Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-id",      default=None)
    p.add_argument("--n-layers",    type=int,   nargs="+", default=[6])
    p.add_argument("--beam-width",  type=int,   nargs="+", default=[5])
    p.add_argument("--z-step",      type=int,   nargs="+", default=[64])
    p.add_argument("--causal",      action="store_true",   default=True)
    p.add_argument("--no-causal",   dest="causal", action="store_false")
    p.add_argument("--train-steps", type=int,   nargs="+", default=[1_000_000])
    p.add_argument("--eval-steps",  type=int,   default=500)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--d-model",     type=int,   default=512)
    p.add_argument("--num-heads",   type=int,   default=8)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--noise-prob",  type=float, default=0.15)
    p.add_argument("--sweep",       action="store_true",
                   help="Run all combinations of list-valued args")
    p.add_argument("--fetch-only",  action="store_true",
                   help="Skip build/push; poll if needed, then fetch+analyze+plot")
    p.add_argument("--plot-only",   action="store_true",
                   help="Re-plot from already-fetched output_remote/<run-id>/")
    return p.parse_args(argv)


def _ts(): return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    args = parse_args()

    if args.sweep:
        i = 1
        for n in args.n_layers:
            for k in args.beam_width:
                for z in args.z_step:
                    for steps in args.train_steps:
                        rid = (args.run_id or
                               f"phase2_{_ts()}_{i:03d}_N{n}_k{k}_Z{z}_{steps//1000}K")
                        single_run({
                            "run_id": rid, "n_layers": n, "beam_width": k,
                            "z_step": z, "is_causal": args.causal,
                            "train_steps": steps, "eval_steps": args.eval_steps,
                            "seed": args.seed, "d_model": args.d_model,
                            "num_heads": args.num_heads, "lr": args.lr,
                            "noise_prob": args.noise_prob,
                        }, fetch_only=args.fetch_only, plot_only=args.plot_only)
                        i += 1
    else:
        n, k, z, steps = args.n_layers[0], args.beam_width[0], args.z_step[0], args.train_steps[0]
        rid = args.run_id or f"phase2_{_ts()}_N{n}_k{k}_Z{z}_{steps//1000}K"
        single_run({
            "run_id": rid, "n_layers": n, "beam_width": k,
            "z_step": z, "is_causal": args.causal,
            "train_steps": steps, "eval_steps": args.eval_steps,
            "seed": args.seed, "d_model": args.d_model,
            "num_heads": args.num_heads, "lr": args.lr,
            "noise_prob": args.noise_prob,
        }, fetch_only=args.fetch_only, plot_only=args.plot_only)
