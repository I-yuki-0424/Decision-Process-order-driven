# Results

See plots/learning_curves.png and plots/final_eval_return.png in the output dir.

Eval: 64 episodes, T=200 steps, Craftax-Classic. Return = env reward sum. Crafter score from achievement rates.

| run | src | params | episodes | wall s | first-10 train ret | last-10 train ret | eval ret (untrained→trained) | random ret | crafter (trained/greedy/random) | last entropy | skipped updates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mdp_branch__reinforce_normlogits | local | 6175 | 2000 | 413 | 0.97 | 0.98 | 1.07→0.90 | 1.46 | 1.26/0.31/1.85 | 1.86 | 0 |
| transformer_branch__argmax_legacy | local | 6175 | 2000 | 366 | 0.74 | 1.20 | 0.55→1.24 | 1.46 | 1.67/0.65/1.85 | 2.58 | 0 |
| transformer_branch__reinforce | local | 6175 | 2000 | 383 | 0.67 | 1.27 | 0.54→1.11 | 1.46 | 2.04/0.53/1.85 | 2.51 | 0 |
| variant_5_1__reinforce_normlogits | local | 8223 | 2000 | 628 | -0.58 | -0.45 | -0.42→-0.42 | 1.46 | 0.14/0.00/1.85 | nan | 0 |
| variant_5_2__argmax_legacy | local | 5583 | 2000 | 530 | 0.66 | 1.23 | 0.46→1.41 | 1.46 | 1.74/0.70/1.85 | 2.74 | 0 |
| variant_5_2__reinforce | local | 5583 | 2000 | 532 | 0.70 | 1.10 | 0.46→0.82 | 1.46 | 1.50/0.81/1.85 | 2.29 | 0 |
| variant_5_3__reinforce | local | 2479 | 2000 | 521 | -0.06 | 0.12 | -0.32→0.11 | 1.46 | 0.88/0.14/1.85 | 1.45 | 0 |
| variant_5_4__reinforce_normlogits | local | 7231 | 2000 | 482 | 0.50 | -0.45 | 0.65→-0.45 | 1.46 | 0.19/0.09/1.85 | nan | 0 |
| mdp_branch__reinforce | kaggle | 6175 | 2000 | 243 | -0.03 | -0.17 | 0.07→-0.10 | 1.46 | 0.47/0.35/1.85 | nan | 0 |
| variant_5_1__reinforce | kaggle | 8223 | 2000 | 698 | -0.37 | -0.44 | -0.32→-0.50 | 1.46 | 0.19/0.19/1.85 | nan | 0 |
| variant_5_4__reinforce | kaggle | 7231 | 2000 | 296 | -0.60 | -0.52 | -0.52→-0.54 | 1.46 | 0.10/0.08/1.85 | nan | 0 |
