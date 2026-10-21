"""TASK-20261022-026: sample-efficiency screening on tuning seeds 4000-4999 (lr schedule, reward coefficient, model/recipe parameters).

Reuses the T025 harness (configs, command line, job list); only the output root, the extra configs and the new run_epa_mini flags differ.
  SWEEP_MODULE=run_t026_sweep python scripts/kaggle_t025.py build --kernel t026-a --stage R1 --configs G4TP1,TP1h --seeds 4000-4003
Pre-registration: docs/experiments/2026-10-22_sample_efficiency/README.md.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_t025_sweep as sw  # noqa: E402

sw.ROOT = "output/phase1/push_t026"
sw.rs.VRAM_BUDGET_MIB = 7000
base, truck = sw.base, sw.truck
TP = dict(tp_coef=1.0)
X = ("lr_sched", "lr_knee", "lr_knee_frac", "rew_small")   # point keys that exist only in TASK-026

CONFIGS = {
    # --- LR schedule (idea 1): faster decay after the steep phase, higher peak -------------------------------------------------
    "G_lr2": base(arm=TP, lr=2e-3), "G_lr3": base(arm=TP, lr=3e-3), "G_lr05": base(arm=TP, lr=5e-4),
    "G_cos1": base(arm=TP, lr_sched="cosine"), "G_cos2": base(arm=TP, lr=2e-3, lr_sched="cosine"),
    "G_kn2a": base(arm=TP, lr=2e-3, lr_sched="knee", lr_knee=0.25, lr_knee_frac=0.3),
    "G_kn3": base(arm=TP, lr=3e-3, lr_sched="knee", lr_knee=0.15, lr_knee_frac=0.2),
    "G_kn1": base(arm=TP, lr=1e-3, lr_sched="knee", lr_knee=0.25, lr_knee_frac=0.3),
    "G_kn2b": base(arm=TP, lr=2e-3, lr_sched="knee", lr_knee=0.4, lr_knee_frac=0.4),
    "T_lr2": truck(arm=dict(tp_coef=1.0, tp_src="head"), lr=2e-3),
    "T_kn2a": truck(arm=dict(tp_coef=1.0, tp_src="head"), lr=2e-3, lr_sched="knee", lr_knee=0.25, lr_knee_frac=0.3),
    "T_cos2": truck(arm=dict(tp_coef=1.0, tp_src="head"), lr=2e-3, lr_sched="cosine"),
    # --- reward coefficient on the +-0.1 health terms (idea 2; P5-contested diagnostic, never gate evidence) -----------------
    "G_rs0": base(arm=TP, rew_small=0.0), "G_rs05": base(arm=TP, rew_small=0.5),
    "G_rs2": base(arm=TP, rew_small=2.0), "G_rs4": base(arm=TP, rew_small=4.0),
    # --- model / recipe parameters (idea 3) --------------------------------------------------------------------------------------
    "G_w512": base(arm=dict(tp_coef=1.0, width=512)), "G_w128": base(arm=dict(tp_coef=1.0, width=128)),
    "G_tph256": base(arm=dict(tp_coef=1.0, tp_hidden=256)),
    "G_ent005": base(arm=TP, ent=0.005), "G_ent02": base(arm=TP, ent=0.02),
    "G_lam7": base(arm=TP, lam=0.7), "G_lam9": base(arm=TP, lam=0.9),
    "G_g95": base(arm=TP, gamma=0.95), "G_g99": base(arm=TP, gamma=0.99),
    "G_mb16": base(arm=TP, minibatches=16), "G_mb4": base(arm=TP, minibatches=4),
    "G_n128": base(arm=TP, n=128, t=32), "G_n32": base(arm=TP, n=32, t=64),
    "G_ep6": base(arm=TP, epochs=6), "G_clip3": base(arm=TP, clip=0.3), "G_gn1": base(arm=TP, max_grad_norm=1.0),
}
sw.CONFIGS.update(CONFIGS)
_command = sw.command


def command(name, seed, out, role, **kw):
    cmd = _command(name, seed, out, role, **kw)
    p = sw.CONFIGS[name]["point"]
    if p.get("lr_sched", "linear") != "linear":
        cmd += ["--lr-sched", p["lr_sched"]]
        if p["lr_sched"] == "knee":
            cmd += ["--lr-knee", str(p["lr_knee"]), "--lr-knee-frac", str(p["lr_knee_frac"])]
    if p.get("rew_small", 1.0) != 1.0:
        cmd += ["--rew-small", str(p["rew_small"])]
    return cmd


sw.command = command
CONFIGS = sw.CONFIGS
ROOT = sw.ROOT
TEST_SEED_OFFSET = sw.TEST_SEED_OFFSET
jobs_for, parse_seeds, load_stage, report = sw.jobs_for, sw.parse_seeds, sw.load_stage, sw.report
rs = sw.rs

if __name__ == "__main__":
    sw.main()
