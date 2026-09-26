# Idea 6 (World Model + Hamiltonian NN) verification, 2026-09-25

Files: [01_setup.md](01_setup.md) · [02_results.md](02_results.md) · [03_findings.md](03_findings.md).
Data/plots: `output/experiments/2026-09-25_idea6/`.

**Bottom line:** the world model learns Craftax "do-nothing" vitals dynamics (Stage A, real signal), but feeding its predictions
to the action Transformer gave **no measurable benefit** (Stage B): all arms 2.9-3.2 return, within seed noise, and an *untrained*
world model scores the same as the trained one. The large improvement over random (about 3.0 vs 1.46) comes from fixing the
adapter, not from Idea 6.
