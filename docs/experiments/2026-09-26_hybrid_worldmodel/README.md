# Hybrid world model (physics presets + Hamiltonian core + Dreamer-lite latent), evaluation and iteration, 2026-09-26

Files: [01_setup.md](01_setup.md) · [02_results.md](02_results.md) · [03_findings.md](03_findings.md) ·
[04_recommendations.md](04_recommendations.md) · [00_log.md](00_log.md) (chronological log of the autonomous session, including a
mistake and its correction). Raw data, per-run JSON and tables: `output/experiments/2026-09-26_hybrid/`.

**Bottom line.**
1. The proposal as stated (known equations + HN + DreamerV3-like part, fused by a Transformer or a sum) did **not** beat a plain
   structured Hamiltonian model; the Dreamer-lite recurrent-latent part was the least reliable component (unstable long rollouts,
   6-90x more parameters). A learned Transformer fusion was mixed, not better than a sum.
2. What did work, robustly across tasks and in planning: **give the known equation to the learned model as an input feature
   (optionally plus a learned-scale skip), not as a hard fixed term.** A wrong hard preset is worse than no preset; the same wrong
   preset as a feature gets within 5% of the true-physics planning ceiling with 8 training trajectories.
3. Model structure, not data or parameter count, drove quality: 325-parameter structured models matched 130K-parameter ones; the plain
   MLP got worse with more parameters.
4. Nothing that is learned extrapolates outside the training support unless the correct functional form is supplied.
