# Candidate convergence experiment (2026-09-25)

Index. Split files: [01_setup.md](01_setup.md) · [02_results.md](02_results.md) · [03_findings.md](03_findings.md) · [04_modifications.md](04_modifications.md).
Raw data / plots: `output/experiments/2026-09-25_convergence/` (`local/`, `kaggle/`, `plots/`, `results_table.md`).

**Bottom line:** no candidate demonstrated an effectiveness advantage. After 2,000 training episodes each (<50,000 cap), every trained
policy scores at or below the uniform-random policy (eval return 1.46). Only the plain-logit candidates (transformer_branch, variant_5_2)
moved from untrained (~0.5) to ~1.1–1.4; that is escape from a degenerate deterministic policy, not evidence of learning beyond random.
