# Closed-form physics benchmark for the Idea-6 world-model family (2026-09-26)

Files: [01_setup.md](01_setup.md) · [02_results.md](02_results.md) · [03_findings.md](03_findings.md).
Raw data, full tables and plots: `output/experiments/2026-09-25_physics/` (`results_tables.md`, `plots/`).

**Bottom line.** Where the physics really is conservative and the state is canonical (Kepler two-body, harmonic oscillator), a
Hamiltonian prior with a known kinetic term is decisively better than a generic neural ODE, including on unseen orbits and in
long rollouts. Where the system is dissipative and nonlinear (falling body with quadratic drag), the Hamiltonian family does
not help and a generic MLP is competitive or better. The **specific `WorldModel` class as implemented** (idea6_full) does not
inherit the Hamiltonian benefit: it is worse than its own components on Kepler. Effectiveness is real but conditional on the
domain, which matches the theoretical reservations already stated.
