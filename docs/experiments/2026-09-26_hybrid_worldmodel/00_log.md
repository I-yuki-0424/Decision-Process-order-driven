# Autonomous session log (operator authorised ~5 h of independent work; started 2026-09-26 07:39)
Rules I follow: real runs only; document each modification; commit at checkpoints; report negative results.
- 07:39 launched main run (4 tasks x 8 arms x N{64,512} x noise 0.02 x 3 seeds, 2000 steps).
- 08:15 Kepler results INVALID: every arm (even parameter-free physics) was worse than persistence. Cause found by direct simulation: my
  perturbation (+0.4/r^5 attractive, drag 0.05) made 30-34% of test orbits plunge (min r 0.02-0.05) or eject (max r 1e5). Task design bug,
  not a model result. Kept in `main_invalid_kepler_v1/` for transparency. Fixed to repulsive +0.25/r^5 and drag 0.02 (min r >= 0.49 checked); rerunning Kepler tasks.
  fall_drag and pendulum results were unaffected and are kept.
- 08:50 sweep (hidden 16-256) done; 09:00 valid Kepler results in: pure hn (0.13) beats all hybrids; wrong preset (mu=0.7) hurts; RSSM-lite unstable on long Kepler rollouts.
- 09:05 launched lam_res 0.1/1.0 + gated fusion (lam/) and iteration 2: phys_hn_mu (known form, learnable constant) and hyb_force_mlp (residual only on dp/dt) (iter2/).
- 09:15 iter2 + lam results: (a) learnable constant on the known equation (phys_hn_mu) repairs the WRONG-constant preset (kepler_wrongmu 0.35 -> 0.19) but pure hn still best on Kepler (0.13);
  (b) forcing the residual to act only on dp/dt (hyb_force_mlp) ~ no change; (c) penalising the residual (lam 0.1, 1.0) monotonically HURTS fall_drag ID (0.12 -> 0.67 -> 1.30):
  restricting the learned part is counterproductive when the physics is incomplete; (d) gated fusion (init closed): fall_drag OOD 0.049 (best OOD seen) but ID 0.63.
- 09:20 next: partial observation (q only) to test the role the recurrent/Dreamer part is supposed to play (state estimation); it3 = Kepler noise-free/longer-training check on why preset hurts.
- 09:30 it3 (Kepler, noise-free): pure hn best and still improving (0.029 -> 0.013 at 2000 -> 6000 steps); phys_hn 0.129 -> 0.051; phys_hn_mu 0.172 -> 0.050; hyb_sum_mlp 0.28 -> 0.13.
  So the preset's cost is largely slower optimisation (singular 1/r potential) not a hard asymptotic limit; hybrid residual still worst.
- 09:30 partial observation (q only; 216 runs): state-estimation error dominates. Finite-difference p estimate is 2-5x worse than a learned filter; ranking of dynamics arms unchanged.
  Caveat: the learned filter is trained with true p labels from the simulator (not available under real partial observation).
- 09:33 planning study: first version (random shooting, horizon 25) could not solve the task even with the TRUE model (success 0) -> uninformative, replaced by CEM horizon 60, 400 steps;
  true-model MPC now succeeds (return -196), null model fails (-799). Launched full run (5 arms x N{8,32,128} x 3 seeds).
- 09:46 planning (pendulum swing-up, CEM-MPC, true env; 24 episodes; N offline trajectories 8/32/128): success saturates at 100% for every model, so RETURN (speed) is the informative metric.
  true-model planner -196; oracle_phys (true potential as hard preset) -196; hn (no preset) -254..-271; phys_hn (wrong small-angle preset as HARD constraint) -480..-530; hyb -460..-520; mlp -576..-600.
  More data barely matters (N=8 ~ N=128): structure, not data, drives quality; a wrong hard preset is worse than none.
- 09:57 NEW IDEA from that: use the preset as an input FEATURE of the learned potential instead of a hard term. phys_feat (still the WRONG harmonic preset): planning -204..-207 (near the -196 ceiling with N=8!),
  pendulum OOD 0.65-0.84 (best learned arm; hn 2.1-3.3, phys_hn 3.9). But Kepler 0.17-0.21 (worse than hn 0.12) and fall_drag 0.48-0.60 ID / 2.4 OOD (worse than phys_hn 0.14) because a tanh MLP cannot represent a linear/1/r term
  exactly or extrapolate it. => next: add a learned-scale direct skip (skip_feat = MLP([q,Vp]) + a*Vp).
- 10:05 skip_feat (a init 1): best Kepler ID/OOD, fall_drag ID good, but pendulum OOD 4.0 and planning -454..-491 (locks in the wrong preset). skip_feat0 (a init 0): pendulum OOD 0.36, planning -227..-234, fall_drag ID 0.51.
- 10:23 5-seed pooling done (seeds 0-4 for the key arms); final tables generated. Stopped exploring at ~2h45 of the 5h authorised because the main questions were answered; remaining ideas listed in 04_recommendations.md.
