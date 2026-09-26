# Recommendations and next experiments (my judgement, based on the evidence in 03_findings.md)

**Redesign of the proposed architecture (candidate, not yet validated beyond these toy physics tasks)**
1. Core: (J-R) grad H with the known kinetic term (existing `HamiltonianOps`), which gave most of the gain in every conservative or lightly damped task.
2. Known physics enters as **features** of the learned potential; add a learned-scale direct skip *only* for presets whose functional form you trust (gravity, Newton),
   start it open (a=1) for trusted forms and closed (a=0) for uncertain ones. Do not use fixed hard preset terms unless verified.
3. Replace Dreamer-lite/Transformer fusion by a small, unregularised residual MLP on the force (dp/dt) only when the domain has clear unmodelled dissipation; keep it optional.
4. Use recurrent/latent machinery only where partial observability requires it, and test it against a learned filter first: the estimator dominated error in the partial-observation test.
5. Judge world models by **planning return in the true environment**, not only by rollout error: success rate saturated (100%) while returns differed 3x.

**Next experiments I would run (not done)**
- Full DreamerV3 (RSSM + symlog heads) as the learned part on a pixel or high-dimensional physics task, where its representation learning could matter.
- Self-supervised state estimation (no momentum labels) under partial observation.
- A domain with real physics plus a decision problem at larger scale (e.g. orbital transfer with unmodelled perturbations) and 5+ seeds, comparing hn / skip_feat0 / phys_feat by planning return.
- An automatic trust rule for the skip gate (e.g. validation loss on held-out longer rollouts) instead of choosing a=0 or a=1 per task.
- Iterating on the `WorldModel` class integrator (identified earlier as the reason idea6_full underperformed its own components).
