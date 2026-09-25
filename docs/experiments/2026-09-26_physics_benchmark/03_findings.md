# Findings

1. **Conservative, canonical systems: the Hamiltonian prior is a large, real win.** Kepler, mean error over the rollout
   (N=128, clean): mlp 0.66, hn_gen 0.10, hn_sep 0.075. The MLP drifts in energy by 44% over the rollout, hn_sep by 0.6%. On
   unseen orbits (OOD) hn_sep 0.26 vs mlp 1.08. Sine (ID): mlp 0.055 vs hn_sep 0.020. The known kinetic term (hn_sep) beats a fully
   generic H (hn_gen), so partial known structure helps beyond structure alone. Data efficiency: hn_sep at N=16 (0.23 end error)
   beats the MLP at N=1024 (1.7).
2. **Dissipative nonlinear systems: no benefit, sometimes harm.** Falling body with quadratic drag: hn_gen (1.6 ID, 3.2 OOD)
   and hn_sep (0.80 ID, 2.6 OOD) do worse than or no better than the MLP (0.75 ID, 0.11 OOD). A constant learned R cannot
   express |v|v drag (port_sep 0.38 ID, 2.3 OOD), and even the oracle-potential arm is only 0.14 / 0.30. The Hamiltonian family
   is the wrong structure here; the generic model wins OOD.
3. **The structure does not give extrapolation by itself.** Sine OOD (amplitude 2-3x): every learned arm fails (about 2.9-3.1
   vs persistence 3.06 in mean error) because V(q) is a tanh MLP that cannot extrapolate q². Only the oracle preset (exact
   V) and, partly, `idea6_full` (0.79) handle it. The benefit of the prior depends on the *form* of the unknown part.
4. **`idea6_full` (the `WorldModel` class as implemented) does not realise the gain.** On Kepler it is at persistence level ID
   (1.27) and diverges OOD (12.0), while hn_sep with the same kind of Hamiltonian (RK4) reaches 0.075 / 0.26. The differences are
   its semi-implicit Euler integrator (8 substeps), the gate, and the zero-init residual MLP; this benchmark does not separate
   which one is responsible. Its residual does help where the physics is unknown (fall_drag: best learned arm ID at 0.30; best
   learned OOD after mlp: 0.74). Fixing the integrator is the first obvious improvement.
5. **Noise.** At 2% observation noise the Kepler advantage shrinks but remains (hn_sep end error 0.57-0.71 vs mlp 1.45-1.7); the
   end-point metric is noise-limited even for the oracle (about 1.0-1.4). Sine and fall_drag are barely affected.
6. **Harness sanity.** `oracle_port` reaches about 0.001-0.002 error (Kepler/sine clean), so the data, integrator and metrics
   are consistent. Its perfect result is by construction (given the true potential) and is not evidence for the method.
7. **Limits.** 3 seeds; one network size (about 5K params); short training (3,000 steps); fixed hyper-parameters not tuned
   per arm (the MLP could gain from tuning); sine end-point metric is degenerate (see setup). The `idea6_full` gap could be
   partly tuning. Nothing here tests discrete or game-like domains.

## What this says about the architecture question
- Idea 6's Hamiltonian core is effective **exactly when the theory says it should be**: conservative, smooth, canonical
  coordinates. Craftax-Classic vitals and text-like domains are not that, consistent with the earlier null result.
- For dissipative/nonlinear or unknown physics, the useful part is the generic residual, not the Hamiltonian structure.
- The path to a useful Idea 6 is (a) correct the integrator/residual design of `WorldModel`, (b) extend the dissipation model
  beyond constant R (state-dependent R(x) or a learned force term), and (c) show the same advantage on a domain with real
  physics plus a decision problem (e.g. orbital transfer or control), which is where the world model would then feed action
  selection.
