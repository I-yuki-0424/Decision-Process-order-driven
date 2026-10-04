# TASK-20261004-023: reference recipe, Truck variants, diagnostics (pre-registration, nothing run yet)

Accepted by the operator on 2026-10-04 ("modify the code and schedule it; do not begin training yet"). Everything below was
fixed **before** any run; the driver is `scripts/run_phase1_recipe_sweep.py` and no stage starts on its own.

## Why (from the TASK-017/020 data)

* G1.0 fails by ~10 reward_pct (best tuned baseline 32.45 / 8.16 vs 42.7 / 9.6). TASK-020 tuned 16 points at the noise floor and
  did not help the baseline. Its space only had γ ∈ {0.99, 0.995}, λ ∈ {0.8, 0.95}, no value-target normalization, grad-norm 1.0
  (Craftax's 1B-step defaults). λ = 0.95 was among the worst points for every arm, and moving γ or λ up hurt the GRU in TASK-017.
* Both baselines that reach ~47 % at 1M steps use a different recipe. Dedieu et al. 2025 (arXiv 2502.01591, Table 3):
  48 × 96, 4 epochs × 8 minibatches, lr 4.5e-4 annealed, γ 0.925, λ 0.625, TD coef 1.0, grad-norm 0.5, EMA-standardized value
  targets (α 0.95), GAE standardized over the batch. Moon et al. 2023 (arXiv 2307.03486): γ 0.95, λ 0.65, value normalization,
  LayerNorm before every layer, grad-norm 0.5, 3 × 8, lr 3e-4.
* Agents die at the first nightfall. Mean evaluation episode ≈ 150–230 steps for every arm. In Craftax-Classic it is dark from
  about step 148 to 272, zombie spawning rises up to 6×, and a zombie hit does 7 damage to a sleeper. Thirst alone needs about 333
  steps to kill, and plants need 600 steps to ripen, so eat_plant = 0 %.
* Stone tools ≈ 0 % (Truck 0.08 / 0.31 %) although stone 55 % and furnace 49 %. They need ≥1 wood + ≥1 stone next to a table
  (a table costs 2 wood; the furnace needs no table in Classic).

## Code (default off = earlier configs run bit-identically; checked against the previous commit on 264 arrays)

| piece | where |
|---|---|
| `value_norm` (EMA decay of value-target mean/std), `adv_norm` ("minibatch" / "batch"), CLI `--clip --vf --max-grad-norm --value-norm --adv-norm` | `src/pipeline/epa_harness.py`, `scripts/run_epa_mini.py` |
| GRU baseline `ln` (LayerNorm before every dense layer) and `skip` (heads read [h, embedding]); same for `cnn_gru` | `src/model/epa_policies.py` `GRUArm`, `CNNGRUArm` |
| Truck options: `mem_token` (h_{t-1} as an extra Transformer token, so candidate-action tokens see memory), `prev_act` (previous-action embedding into the GRU input), `head_skip` (heads read [LN(h), CLS]), `cand=False` (no candidate tokens: attribution control) | `TFGRUArm` |
| `--save-params`, death-cause / near-table diagnostic replaying the headline evaluator exactly | `run_epa_mini.py`, `src/pipeline/epa_diagnostics.py`, `scripts/diagnose_phase1_deaths.py` |
| provenance in every result file; final runs refuse a dirty tree | `run_provenance()` in `epa_harness.py`, `run_epa_mini.py` |

## Stages (run in this order, each after the operator's go-ahead)

0. **docker build** `docker build -t dpod-local:latest -f docker/Dockerfile .` (the image was deleted), then `run_phase1_recipe_sweep.py env`
   to record `pip freeze` + GPU. Requirements are unpinned: pin to the recorded versions if stage S0 shows a version effect.
1. **S0 `replicate` (~4 GPU-h).** (a) Determinism: Truck seed 1000 twice without and twice with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, GRU twice
   without (100k steps, compare bit for bit). (b) Replication of the TASK-017 finals: Truck and GRU, default config (t00), seeds 0–9
   (role `replication`, never a result). Decision rule: Truck mean more than 2 combined SE below 39.19 → TASK-017's number is
   declared non-reproducible; within → TASK-020's tuned config is the cause of 34.19.
2. **S1 `diag` (~0.4 GPU-h).** Truck and GRU, t00, tuning seed 1002, params saved, then the diagnostic (cause of death, night/sleep at
   death, wood + stone next to a table). Read before S3 is interpreted; not used for selection.
3. **S2 `tune` + `tune2` + `select` (~12 GPU-h).** Arms `ppo_gru`, `ppo_gru_ln` (= gru256+ln+skip), `cnn_gru`, `tf_gru` (Truck).
   Identical points for every arm: anchors **ref** (Dedieu Table 3), **moon** (Moon et al.; rollout 4096 = 64 × 64, env count not from the paper), **t00**
   (TASK-017/020 default) × 3 tuning seeds (1000–1002); 6 random points (RNG 20261004; γ ∈ {0.925, 0.95, 0.99}, λ ∈ {0.625, 0.65, 0.8},
   lr log-U[2e-4, 2e-3], epochs {3, 4}, minibatches {4, 8}, ent {0.003, 0.01}, vf {0.5, 1}, value-norm {0.95, 0.99}, adv-norm {minibatch, batch},
   grad-norm 0.5; shapes 48×96 / 64×64 / 32×128) × seed 1000, top-2 per arm → seeds 1001, 1002. 19 jobs per arm. Selection = max mean
   (reward_pct + score_pct) over 3 tuning seeds. G1.0 is read from S4, not from tuning numbers.
4. **S3 `arch` + `select-arch` (~8 GPU-h).** At tf_gru's selected point: Truck +mem, +pa, +hs, +mem+pa+hs, +hs−nocand. At the better
   baseline's point: gru256+ln, gru256+skip, gru384+ln+skip, gru512+ln+skip (≈3.3M params, near the 4.0M reference), gru512. 5 per family
   × 3 tuning seeds (P9: equal count). Winner per family by the same metric.
5. **S4 `final` (~7 GPU-h).** Truck-family winner, baseline-family winner, `cnn_gru`: seeds 52–61, test 434–443 (replicates k = 10..19:
   42–51 were already looked at), EP-A, 256 envs, final params, clean tree. Then `python scripts/mlflow_ingest.py <files>` and
   `python scripts/audit_phase1_results.py`.

Total ≈ 31 GPU-h on the RTX 3060 Ti (one GPU-saturating job at a time). Cheapest cut if needed: drop `cnn_gru` from S2/S4 (−5 h).

## Critical review (recorded before implementing)

1. Recipe changes are shared by every arm (P9). They can close G1.0 and lift Truck's absolute level, but they are **not evidence
   for the architecture**; only S3 (equal-count variants) and S4 can show a Truck-specific gain.
2. Gains from scaling the baseline (gru512 ≈ 3.3M) are calibration, not a claim. Truck stays ≤ 0.5M params; the "accuracy without
   scaling" claim needs Truck ≥ the 3.3M baseline at equal tuning effort.
3. The memory token adds a recurrent path through the Transformer (deeper BPTT); if it destabilizes training, that is a result, not a
   reason to add unregistered fixes.
4. Not implemented (need a ruling or later): Achievement-Distillation-style objective (its policy-side "memory" feeds a
   reward-selected state to the policy, a grey area under the EP-A observation rule); Dyna-style data reuse (adds a world model to
   params_total and compute; any gain would be compute, not architecture).
5. Not changed: the env reward. Setting the +0.3 net reward of a zombie hit on a sleeper (+1 wake_up − 0.7 health) to 0 would violate
   EP-A ("reward unchanged") and P5 (no reward shaping). S1 measures how often agents die asleep instead.
