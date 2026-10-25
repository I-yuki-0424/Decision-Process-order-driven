"""TASK-20261025-028: data-scaling diagnostic (GRU-TP at 0.5M / 1M / 2M / 5M env steps) and data reuse at 1M (replay of the last K rollouts).

Reuses the T025/T026 harness (recipe point, command line, Kaggle builder); only the step budget, the replay flags and the protocol labels are new.
  SWEEP_MODULE=run_t028_sweep KERNEL_ROOT=kaggle_kernel_t028 python scripts/kaggle_t025.py build --kernel t028-a --stage S --configs S5M,S05M,S2M --seeds 7000-7003
  SWEEP_MODULE=run_t028_sweep python scripts/run_t028_sweep.py report --stage S
Pre-registration: docs/experiments/2026-10-25_data_scaling_reuse/README.md.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_t026_sweep as t026  # noqa: E402

sw = t026.sw
sw.ROOT = "output/phase1/scaling_t028"
base = sw.base
TP = dict(tp_coef=1.0)
M = 1_000_000
EVAL_AT = [500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000]
SEC_1M = 400   # measured TASK-027: ~390 s training per 1M-step GRU-TP job on one Kaggle T4

CONFIGS = {
    # --- A: data scaling, the P5-clean GRU-TP config (TASK-025 G4TP1) unchanged except the step budget ---------------------------
    "S05M": dict(base(arm=TP), steps=M // 2, sec=SEC_1M // 2),
    "S1M": dict(base(arm=TP), steps=M, sec=SEC_1M),
    "S2M": dict(base(arm=TP), steps=2 * M, sec=2 * SEC_1M),
    "S5M": dict(base(arm=TP), steps=5 * M, sec=5 * SEC_1M + 200, eval_at=EVAL_AT),
    # --- B: data reuse at 1M (control = S1M on the same seeds); K = rollouts per update, each transition is used in K updates ------
    "RP2": dict(base(arm=TP), steps=M, sec=2 * SEC_1M, replay_k=2),
    "RP4": dict(base(arm=TP), steps=M, sec=4 * SEC_1M, replay_k=4),
    "RP4e2": dict(base(arm=TP, epochs=2), steps=M, sec=2 * SEC_1M, replay_k=4),
}
sw.CONFIGS.update(CONFIGS)
_command = sw.command   # TASK-026/027 wrapper (lr shape, reward coefficient) over the TASK-025 command line


def protocol_for(steps):
    if steps > M:
        return "diag-data-scaling"   # beyond the EP-A budget: never EP-A, never a gate, never next to paper numbers
    return "EP-A-tuning" if steps == M else "EP-A-mini"


def command(name, seed, out, role, **kw):
    c = sw.CONFIGS[name]
    if "steps" not in c:
        return _command(name, seed, out, role, **kw)
    steps = c["steps"]
    cmd = _command(name, seed, out, "diag" if steps > M else role, steps=steps, protocol=protocol_for(steps),
                   budget="TASK-028 diagnostic: config fixed in advance (TASK-025 G4TP1), no selection")
    if c.get("replay_k"):
        cmd += ["--replay-k", str(c["replay_k"])]
    if c.get("eval_at"):
        cmd += ["--eval-at", ",".join(str(x) for x in c["eval_at"])]
    return cmd


def jobs_for(stage, names, seeds, role="tune", offset=0, protocol=None, budget=None):
    if offset or role != "tune":
        raise SystemExit("TASK-028 runs tuning/diagnostic seeds only")
    jobs = []
    for s in seeds:
        for n in names:
            out = f"{sw.ROOT}/{stage}/{n}__s{s}.json"
            jobs.append(dict(name=f"{n}__s{s}", out=out, seed=s, mem=4400, sec=sw.CONFIGS[n]["sec"], env={}, extra=[],
                             cmd=command(n, s, out, role)))
    return sorted(jobs, key=lambda j: -j["sec"])   # longest first: the two GPUs of a kernel finish close together


sw.command = command
sw.jobs_for = jobs_for
CONFIGS = sw.CONFIGS
ROOT = sw.ROOT
TEST_SEED_OFFSET = sw.TEST_SEED_OFFSET
parse_seeds, load_stage, report = sw.parse_seeds, sw.load_stage, sw.report
rs = sw.rs

if __name__ == "__main__":
    sw.main()
