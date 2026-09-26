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
- 09:31 partial observation (q only; 216 runs): state-estimation error dominates. Finite-difference p estimate is 2-5x worse than a learned filter; ranking of dynamics arms unchanged.
  Caveat: the learned filter is trained with true p labels from the simulator (not available under real partial observation).
- 09:45 planning study: first version (random shooting, horizon 25) could not solve the task even with the TRUE model (success 0) -> uninformative, replaced by CEM horizon 60, 400 steps;
  true-model MPC now succeeds (return -196), null model fails (-799). Launched full run (5 arms x N{8,32,128} x 3 seeds).
