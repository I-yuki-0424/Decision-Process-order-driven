# Setup (operator spec followed: passive dynamics first, then the action Transformer)

**Stage A - world model of "what happens if nothing is done".** Real noop transitions from the simulator, used as offline
data only (256 random-policy episodes x 200 steps; each visited state branched with 8 noop steps; 40.8k train / 10.4k test
samples, split by episode).
- State q = standardised vitals (health, food, drink, energy).
- p = learned momentum encoder of an observation summary (fills the DPOD.ipynb "no canonical coordinates" gap).
- Exogenous "wind" = learned code of the observation summary, held constant during the rollout.
- Models: persistence; `hn` = `WorldModel.step/rollout` from `src/model/candidates/world_model.py` UNCHANGED (2 gated
  port-Hamiltonian modes, dissipation R, zero-init residual; 16.5K params); `mlp` = generic dx = MLP(q, p, wind) with the same
  encoder (8.4K params). Open-loop 8-step rollouts, multi-step MSE loss, 6,000 Adam steps.
- **Physics content:** all FormulaPresets masked OFF. Gravity/drag/buoyancy/ground contact have no Craftax counterpart, and
  hand-writing hunger/thirst rules from the Craftax source would be oracle leakage. Only learned H, learned dissipation R and
  the residual remain.

**Stage B - policy with the frozen world model.** `transformer_branch`, d_model=512 (875,590 params; 6K at d=32 was a toy),
REINFORCE as in the previous experiment, 300 updates x 16 episodes = 4,800 episodes/run (<50,000), T=250, 2 seeds per arm,
eval on 64 episodes. Arms differ only in 8 extra state features (predicted passive change of vitals at +1 and +8 steps,
clipped to [-5, 5]): `base` (zeros), `wm_hn` (trained HN), `wm_mlp` (trained MLP), `wm_hn_untrained` (random-init HN control).

Compute in parallel: local WSL2 Docker RTX 3060 Ti (base, wm_hn) and Kaggle GPU kernel `bfloat16/dpod-candidates-convergence`
(wm_mlp, wm_hn_untrained).

**Required adapter change** (`src/environment/craftax_obs_adapter.py`; original adapter untouched): `CraftaxEnvAdapter` drops
the symbolic observation (map, inventory, mobs) and redraws action features/costs randomly each episode. The new adapter
exposes 39 summary features of the real observation (vitals, inventory, light/sleep, tile-type counts in view, mob counts)
in `SystemState.resource_levels` and uses fixed action features.

Code: `src/model/candidates/passive_world_model.py`, `scripts/run_worldmodel_experiment.py`, `scripts/run_idea6_policy.py`,
`scripts/plot_idea6_summary.py`.
