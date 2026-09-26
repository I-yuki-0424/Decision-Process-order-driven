# Results

Plots: `output/experiments/2026-09-25_physics/plots/data_efficiency.png`, `error_vs_horizon.png`. Full tables (all data sizes and noise levels, ID and OOD): `output/experiments/2026-09-25_physics/results_tables.md`. Below: N=128, clean, median of 3 seeds.

## Mean nRMSE over the whole rollout (lower is better)

| task / split | persistence | mlp | hn_gen | hn_sep | port_sep | idea6_full | oracle_port |
|---|---|---|---|---|---|---|---|
| kepler id | 1.375 | 0.660 | 0.101 | 0.075 | 0.241 | 1.271 | 0.001 |
| kepler ood | 1.660 | 1.079 | 0.459 | 0.259 | 0.291 | 12.038 | 0.000 |
| fall_drag id | 2.726 | 0.753 | 1.616 | 0.802 | 0.384 | 0.296 | 0.142 |
| fall_drag ood | 2.725 | 0.108 | 3.178 | 2.620 | 2.280 | 0.742 | 0.304 |
| sine id | 1.319 | 0.055 | 0.041 | 0.020 | 0.023 | 0.035 | 0.000 |
| sine ood | 3.059 | 3.086 | 3.132 | 2.932 | 2.896 | 0.789 | 0.000 |

## Short-horizon (h=10 = training horizon) error, ID, noisy, N=128

| task | mlp | hn_gen | hn_sep | port_sep | idea6_full | oracle_port |
|---|---|---|---|---|---|---|
| kepler | 0.040 | 0.022 | 0.023 | 0.023 | 0.078 | 0.023 |
| fall_drag | 0.015 | 0.088 | 0.151 | 0.048 | 0.027 | 0.047 |
| sine | 0.018 | 0.017 | 0.015 | 0.016 | 0.017 | 0.016 |

## Drift of TRUE invariants at end of rollout (ID, clean, N=128; median |dI|/|I0|; Kepler: energy, angular momentum)

| task | mlp | hn_gen | hn_sep | port_sep | idea6_full | oracle_port |
|---|---|---|---|---|---|---|
| kepler | 0.44, 0.31 | 0.0075, 0.011 | 0.0059, 0.0051 | 0.083, 0.042 | 0.49, 0.19 | 0.00017, 8.3e-05 |
| fall_drag | 1.1 | 0.88 | 1.9 | 1.6 | 1.6 | 1.9 |
| sine | 0.0057 | 0.00089 | 0.00051 | 0.045 | 0.005 | 0.00015 |

