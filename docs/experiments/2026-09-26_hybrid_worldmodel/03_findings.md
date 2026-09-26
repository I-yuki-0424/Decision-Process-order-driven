# Findings

Evidence strength: prediction tables use 3-5 seeds; planning 3 seeds x 24 episodes; one network family at about 5K parameters unless stated; short
training (2,000 steps). Differences smaller than the min-max ranges in the tables should not be over-read.

## About the operator's proposal (physics layer + HN layer + DreamerV3-like layer, fused)
1. **The Dreamer-lite part did not earn its cost.** Stand-alone it is unstable on long Kepler rollouts (mean nRMSE 2.6-3.5 vs 0.13 for a plain
   Hamiltonian model, worse than persistence), and hyb_sum_rssm inherits that (1.3-3.2). It is only competitive on fall_drag (0.24-0.34, as a
   stand-alone). It costs 6-90x the parameters of the structured models and its accuracy did not improve with size (fall_drag 0.239 -> 0.442 from 3.3K to 344K params).
   Caveat: this is a reduced RSSM on low-dimensional noiseless-ish state; DreamerV3's real strengths (pixels, stochastic partial observability at scale) were not tested.
2. **Residual fusion made things worse where the structure was already sufficient.** On Kepler, hyb_sum_mlp (0.40) is about as bad as a plain MLP (0.36) versus
   0.13 for hn; the free residual absorbs dynamics and breaks the structure. Penalising the residual (lam 0.1, 1) made fall_drag worse (ID 0.12 -> 0.67 -> 1.30):
   restricting the learned part hurts when the physics is incomplete. Force-only residual made no difference.
3. **Transformer fusion vs sum:** the Transformer (hyb_tf_rssm) was better than the sum on Kepler (0.31 vs 3.2) and pendulum OOD (0.98 vs 3.6), worse on fall_drag
   (0.68 vs 0.24). No clear winner; neither beat the simple structured models. I saw no evidence that a Transformer is needed for fusion.
4. **The proposed division of labour (physics/HN = linear and known structure; Dreamer = nonlinear and extrapolation) is not what the data show.** Extrapolation outside
   the training support came only from a correct functional form supplied to the model; the learned parts (MLP, GRU-latent, Transformer) never extrapolated
   (pendulum OOD amplitude, fall_drag OOD height). The learned part is useful for what the equations omit *within* the data support (drag).

## What worked
5. **Preset as a feature, not a hard term.** With the wrong (small-angle) preset on the pendulum, planning return: hard term -480..-530, none (hn) -254..-271, MLP -576..-600,
   **feature -204..-207** (true-model ceiling -196; true preset -196..-200) with only 8 training trajectories. Prediction on large-angle OOD: feature 0.59 vs hard 3.9 vs hn 2.8.
6. **A learned-scale direct skip helps when the preset has the right form and hurts when it does not.** skip_feat: Kepler ID 0.124-0.135 (with wrong mu as well), OOD 0.10-0.12 (best of all),
   fall_drag ID 0.141, but pendulum OOD 4.0 and planning -454..-491 (it locks the wrong shape). skip_feat0 (skip starts closed) recovers pendulum OOD (0.36) and planning (-227..-234)
   but loses the fall_drag gain (0.51 ID). There is no single setting that wins everywhere: it is a bet on how much to trust the preset.
7. **Known form with unknown constant is easy to repair:** learnable scale on the preset (phys_hn_mu) fixes a wrong mu (kepler_wrongmu ID 0.35 -> 0.19).
8. **Structure beats data and parameters.** phys_hn had the same error at 325 and 66,565 parameters (fall_drag 0.13/0.12); the MLP got worse with size (0.72 -> 0.97);
   planning quality was almost flat from N=8 to N=128 trajectories. A wrong hard preset is worse than none.

## Other results
9. **State estimation dominates under partial observation.** With only noisy position observed, error is set by the momentum estimate: finite difference is 2-5x worse than a learned filter
   (Kepler 1.48-2.99 vs 0.38-0.56) and the ranking among dynamics models is unchanged. The learned filter used true-momentum labels from the simulator, which are not available in
   genuinely partial-observation data; a self-supervised estimator was not tested.
10. **A preset can slow optimisation** (singular 1/r potential): Kepler noise-free, 2000 -> 6000 steps: phys_hn 0.129 -> 0.051 and hn 0.029 -> 0.013, so most of the gap to pure hn was
    convergence speed rather than a hard limit. hn still wins at equal steps.

## Things I could not establish
- Whether a full DreamerV3 (with its representation learning) changes point 1. Whether these results hold at larger scale or in a decision problem beyond swing-up.
- Statistical significance: seeds are few; several gaps (e.g. skip_feat vs hn on Kepler) are within seed ranges.
- Relevance to Craftax or other non-physical domains: none of this transfers without an actual physical prior; the earlier Craftax study found no benefit.
