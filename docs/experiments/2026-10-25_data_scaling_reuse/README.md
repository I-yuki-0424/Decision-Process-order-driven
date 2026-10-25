# TASK-20261025-028: data scaling (0.5M / 1M / 2M / 5M) and data reuse at 1M (pre-registration, written 2026-10-25 before any run of this task)

Operator request (2026-10-25): measure the difference between about 5M env steps (maximum) and 1M, and find what mainly drives the gain; add a
0.5M run to tell whether the problem is only the step count; in parallel, run a data-reuse experiment. Run on Kaggle if it finishes within 4 h,
otherwise prepare and schedule it.

## 1. Questions and fixed reading rules

Arm: GRU-TP exactly as TASK-025 G4TP1 (`Base_O02_D04_S00`, gru256+ln+skip + TP auxiliary, 1,254,358 params, env reward unchanged, P5-clean), recipe point of
TASK-024/025 (64 envs x 64 steps, 4 epochs, 8 minibatches, lr 1e-3 linear anneal to 0 over the run's own budget, 5 % warm-up, gamma 0.97, lambda 0.625,
entropy 0.01, value norm 0.99, batch advantage norm). Seeds **7000-7003** (unused before: checked against STATE.yaml and the experiment READMEs), test seed = train seed
(tuning convention), 256 fresh envs, sampled policy, final parameters.

* **Q1 (scaling).** Separate runs with budget 0.5M / 1M / 2M / 5M (each anneals lr over its own budget). Report reward_pct, score_pct, per-achievement rates,
  episode length, mean and SE over 4 seeds.
* **Q2 (what drives the gain).** Per-achievement contribution to delta reward_pct = delta rate / 22, between 1M and 5M (and 1M -> 2M). The ranking is reported for
  all 22 achievements; the "main drivers" are the achievements that together make up >= 70 % of the gain, listed largest first.
  The 5M run also evaluates its current parameters at 0.5M, 1M, 2M, 3M, 4M steps (`--eval-at`, same evaluator, never fed back) to show the order in which
  achievements appear; these mid-run numbers have a not-yet-annealed lr and are compared with the budget-matched runs only descriptively.
* **Q3 (only the step count?).** If reward rises roughly linearly in log2(steps) from 0.5M through 2M with no visible change of slope, more data of the
  same kind is the binding constraint at 1M. If 0.5M -> 1M is flat and 1M -> 2M is steep, or the gain at 2M/5M is concentrated in achievements that are 0 % at 1M,
  the limit is reaching new states (exploration) rather than learning from the states already seen. This reading is stated before the runs; the numbers decide.
* **Q4 (data reuse).** At the 1M budget (control S1M, same seeds, same platform), PPO trains on the last K rollouts per update (`src/pipeline/epa_replay.py`):
  RP2 (K=2), RP4 (K=4), RP4e2 (K=4, 2 epochs). Each env transition is used in K updates; no extra env steps.
  A reuse arm is a candidate if its 4-seed mean reward is >= S1M + 2.0 and the paired difference exceeds 2 SE. A candidate needs a fresh-seed confirmation
  and an EP-A block in a later task; nothing here closes a gate.

## 2. Critical review (before running)

* 2M and 5M exceed the EP-A budget: protocol `diag-data-scaling`, role diag, never EP-A, never next to a paper number. 0.5M is `EP-A-mini`; 1M runs are `EP-A-tuning`.
* TASK-026 already measured 2M / 3M / 4M (57.7 / 66.3 / 65.5, CPU, 2 seeds, seeds 4000-4001). This task repeats the curve on Kaggle T4 with 4 seeds, adds 0.5M and 5M,
  per-achievement attribution, and mid-run evaluations. Platforms are not mixed: TASK-026 numbers are quoted as context only.
* Replay is not "more epochs": TASK-026 found 6-8 epochs neutral/harmful and TASK-025 found more, smaller updates (K1/G4K1) neutral. Here new and old rollouts are mixed,
  and the PPO ratio is taken against the policy at the start of each update (proximal policy, Hilton et al. 2021), with truncated importance weights in the
  advantages (V-trace form, Espeholt et al. 2018, rho_bar = c_bar = 1). With K=1 it reproduces plain PPO (tests/test_epa_replay.py).
* Data integrity: the buffer holds only executed transitions (P4), env steps are unchanged (P10), params_total is unchanged (P11), no reward change (P5).
  Compute per env step grows about K x: train_seconds is reported, and any gain is "more compute per sample", not "free".
* Known approximation: an old rollout restarts from the GRU state stored at collection time (computed by older parameters), with no burn-in.
* The same 4 seeds serve every arm, so arm differences are paired. 4 seeds give SE about 0.5-1.0 reward (TASK-027 spread); differences below ~2 are noise.

## 3. Compute plan (measured basis: TASK-027, ~390 s training per 1M-step GRU-TP job on one Kaggle T4; a kernel has 2 T4)

| kernel | jobs (4 seeds each) | estimated GPU time | wall (2 GPUs) |
|---|---|---|---|
| t028-a | S5M (+5 mid-run evals), S2M, S05M | 12,800 s | ~1.8 h |
| t028-b | S1M, RP2, RP4, RP4e2 (replay cost assumed ~K x training) | 14,400 s | ~2.0 h |

Both kernels run in parallel (Kaggle allows 2 GPU sessions). Expected Kaggle wall time ~2-2.3 h including installation, below the 4 h limit, so the runs are started now.
Jobs run longest first. If a kernel has not finished 4 h after its push, the remaining jobs are reported as not run (no partial result is aggregated).

## 4. Outputs

Result JSONs: `output/phase1/scaling_t028/S/` (LFS; no git-lfs on this VPS) with plain copies in `results/` here; summary in `RESULTS.md`.
MLflow: `python scripts/mlflow_ingest.py output/phase1/scaling_t028` in this checkout (registry: `Base_O02_D08_S00/S01` for the replay arms).
