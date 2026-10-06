"""TASK-024 equal-effort check (P9): the gru512 baseline of TASK-023 (reference recipe point `ref`) with the same warm-up knob that helped Truck.
  python scripts/run_baseline_warmup_check.py --seeds 2000-2002 --warmup 0.05 --deadline 2026-10-07T12:00
Tuning seeds only (>= 1000); results in output/phase1/truck_scale_t024/baseline/."""
import argparse, datetime as dt, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_phase1_recipe_sweep as rs

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="2000-2002")
ap.add_argument("--warmup", type=float, default=0.05)
ap.add_argument("--deadline", required=True)
ap.add_argument("--role", default="tune", choices=["tune", "final"])
a = ap.parse_args()
lo, _, hi = a.seeds.partition("-")
seeds = range(int(lo), int(hi or lo) + 1)
tag = f"gru512_warm{int(a.warmup * 100)}"
jobs = []
for s in seeds:
    out = f"output/phase1/truck_scale_t024/{'baseline' if a.role == 'tune' else 'final'}/{tag}__s{s}.json"
    cmd = rs.cli("ppo_gru", {"width": 512}, rs.ANCHORS["ref"], s, a.role, out, 1_000_000, "EP-A-tuning" if a.role == "tune" else "EP-A",
                 "TASK-024 equal-effort check: gru512 at the TASK-023 ref point + warm-up", offset=382 if a.role == "final" else 0)
    cmd += ["--warmup", str(a.warmup)]
    jobs.append(dict(name=f"{tag}__s{s}", out=out, seed=s, mem=2500, sec=400, env={}, extra=[], cmd=cmd))
prov = rs.host_provenance()
rs.Scheduler(jobs, 1, dt.datetime.fromisoformat(a.deadline), "output/phase1/truck_scale_t024/baseline", prov).run()
