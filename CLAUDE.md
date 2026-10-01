# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research prototype for a novel decision/task-generation model combining Transformers with Markov Decision Processes, written in JAX (JIT, `vmap`, `lax.scan`). The primary target is CUDA, with TPU compatibility. The goal is high accuracy **without** scaling up model size. Design ideas are in Japanese under `docs/JP-ideas/` (1st–5th idea, plus `architecture_review.md`). The code implements the "4th idea" (ADR-001) and the "5th idea" hierarchical extension.

## Critical review of proposed architectures (highest priority)

The operator is searching for groundbreaking ideas and **may propose architectures that are logically flawed**. Treat every proposal as a hypothesis, not a spec. Before implementing, reduce the idea to plain terms and check whether it can work as a model at all. If it clearly cannot, **say so plainly and refute it**, even when it is written in elaborate terms. Do not quietly implement it, soften the objection, or "make it work" by adding hidden assumptions.

Check in particular for these problems:
- **Answer leakage / circularity:** the model's inputs, masks, beam-search scores, rewards, or "engines" already contain the correct answer. Examples: the target action, the ground-truth next state, true goal progress from the env, or a hand-coded solver doing the real work. Once the idea is simplified, nothing is being learned or predicted.
- **Oracle components at inference time:** anything that only exists during training or evaluation, such as the env simulator, true transition matrix `W_res`, or labels, but that the design relies on at test time.
- **Unfalsifiable evaluation:** metrics that cannot fail. Examples: comparing the model against itself, scoring with the same function used to select actions, or baselines handicapped by construction.
- **Claims that don't follow:** for example, a claim that accuracy improves "without scaling" when the gain actually comes from extra compute (beam width, macro/micro steps) or from extra information.

The "strictness" rule below means "don't invent components the operator didn't specify." It does **not** mean "implement flawed specs without comment." When refuting, state the flaw concisely, show the simplified form that exposes it, and propose a fix or an experiment that would decide the question if one exists. Record the objection in the task's `STATE.yaml` entry.

## Operating protocol (`docs/core/MASTER_GUIDANCE.xml`, v11)

The role is a careful senior engineer working autonomously: correct first, minimal diff, verifiable, and ask only when genuinely blocked.

- **STATE.yaml:** it is the only file under `docs/core/` besides the guidance. At session start read only its metadata and the task entries relevant to the current task. Do not load or summarize unrelated entries unless a dependency, a status conflict, or an explicit audit needs them. Every task has the fields `id`, `created_at`, `status` (queued/in_progress/blocked/done), `description`, `spec_ref`, `validation_commands`, `result`, and `blockers`. Update status and result the moment either changes; do not batch. Record material assumptions in the entry. Quote fields directly and report missing ones as missing. Store specs too long to inline as a path/URL in `spec_ref` and read only the needed sections.
- **Strictness:** do not implement unconfirmed components based on your own assumptions. Follow the idea docs, the ADRs in `docs/DECISIONS/`, and `docs/LOGS/DESIGN_HINTS_AND_FAILURE_LOG.md`, subject to the critical review section above.
- **Verification:** run the task's own `validation_commands` (supplied by the operator) and record the result. No default data source or smoke path exists.
- **Git:** commit at natural checkpoints, not per minor edit, and push after each commit. For the scip-neo4j graph to reflect the latest code, work on a branch and commit and push per task set.
- **Compute:** use Docker only when a task needs an isolated or reproducible environment. Request CUDA only for tasks that use it, and never require a GPU to start a session. Keep logging at INFO and low-noise unless debugging.
- **Authorization:** long GPU training runs need operator authorization. ADR-002 authorizes Phase II runs of 1M–10M steps on Craftax-Classic via Kaggle. Its conditions are checkpoints every 100K steps, resumability, and a 24h limit per config.
- **Stop when** the task is done, no further work is implied, or external input is required. Record the blocker and stop.

### Roadmap (`<roadmap>` in MASTER_GUIDANCE, added in v10)

Read the XML for the exact gates before any roadmap task. Structure and rules that are easy to get wrong:
- **Only the gates of `current_phase` (currently 1) are pursued.** Phases 2–4 are locked, and their thresholds are fixed in advance so they cannot be adjusted after seeing results. Phase 1 gates are G1.0 (baseline calibration), G1.1 and G1.2 (surpass the reference on both metrics), and G1.S (EP-B screening). A phase closes only when every gate passes under EP-A. Record the gate id and the quoted numbers in the task's `result`.
- **Two protocols are never mixed.** EP-A is gating and is the only one comparable to papers: Craftax-Classic with default episode length and no T cap, one declared input modality (63x63x3 image or 1345-d symbolic; the 39-d adapters are *not* EP-A observations), at most 1M env steps counted across every copy including passive and offline data, 10 seeds with every seed listed, sampled policy over at least 256 fresh envs per seed, final checkpoint only, and tuning on disjoint seeds with the same budget as the baseline. EP-B is internal screening only (39-d adapter, T=250, PPO actor-critic; tuned base return 4.36) and never closes a gate or gets quoted next to paper numbers.
- **Metrics:** `reward_pct` is the mean of (distinct achievements / 22) x 100, counted via `masked_achievements()` in `src/environment/craftax_env_adapter.py`, not the env return. `score_pct` is the Crafter score, exp(mean ln(1+s_i)) − 1, implemented as `calculate_crafter_score` in the same file. Before the first gate run, verify both in code and record it (`metric_check`). Results from before the 2026-09-29 auto-reset fix are ineligible.
- **Prohibited (P1–P12):** no fabricated or constant result fields, no untrained weights presented as trained, no simulator access or counterfactual simulator-branching labels at decision time (oracle arms only as labelled upper-bound controls, never counting toward a gate), no hand-specified game knowledge (achievement hierarchy, tech tree, reward shaping) in gate runs, no use of quarantined or pre-fix results, no training/tuning/selection on evaluation seeds or episodes, no unequal effort against the baseline, no uncounted env steps, no parameter shortcuts (gates use `params_total`), and no paper number without its source id and caveat ids.
- **Reference values** are quoted from cited papers (Dedieu et al. 2025 Table 1 and others). Do not edit them from memory. Changing one needs a new citation. Values marked `basis="to_measure"` are missing and must not be invented.
- Every result file must record the fields listed in the roadmap's `<reporting>` block (phase, gate_id, protocol, commits, modality, `env_steps_total`, seeds, per-seed metrics with mean and SE, `params_total`, `params_deployed`, tuning budget).

### Experiment tracking (`<experiment_tracking>` in MASTER_GUIDANCE, added in v11)

Local MLflow (SQLite `mlflow_local/`, gitignored) indexes result files; the files under `output/` stay the source of truth. After every execution run `python scripts/mlflow_ingest.py <result file or dir>`.
- One Experiment per phase (`phase-N`; pre-roadmap runs in `phase-0-legacy`). One run per training execution (per seed), named `<model_id>_<YYYYMMDD-HHMMSS>` (UTC).
- `model_id` = `<IdeaName>_O<NN>_D<NN>_S<NN>` from `docs/experiments/MODEL_REGISTRY.yaml` (register before the first run; agents may add `status: provisional` entries). **Proper names are operator-only; never write or propose `proper_name`.**
- Parameters use the official names in `docs/experiments/MLFLOW_PARAM_NAMES.yaml` (`train.learning_rate`, not `lr`/`LR`); add new ones there before the run that uses them.
- Seeds: train 42, test 0424 (stored as 424); replicate k = 42+k / 424+k. Tuning seeds disjoint from 42-51 and 424-433.
- Keep logs minimal (final scalars, <= 50 curve points, no artifacts). Record every failed/interrupted run with `mlflow_ingest.py --record-failure ...`; such runs are never aggregated or quoted as results.
- **Pitfalls (a past mistake left Phase-1 results invisible):**
  - `mlflow_local/` is gitignored, so a store built in a worktree or another checkout does not exist elsewhere. After producing or pulling results, run the ingest **in the checkout being used** (`output/` must be smudged LFS, not pointers) and confirm with `mlflow_report.py summary`. A STATE.yaml "backfilled" note is not proof the store exists here.
  - Always pass the store explicitly: `mlflow ui --backend-store-uri sqlite:///mlflow_local/mlflow.db`. A bare `mlflow ...` or `sqlite:///mlflow.db` uses/creates an empty `./mlflow.db` at the repo root (gitignored) and looks like "nothing was recorded". Never ingest into it.
  - New result arms: register them in `MODEL_REGISTRY.yaml` (provisional) before ingesting; check `summary` shows no `UNREGISTERED` phase-1 rows. Registry edits need `rm -rf mlflow_local` + re-ingest.
  - In pandas aggregation code (`mlflow_report.py`), use `groupby(..., dropna=False)`: arms lack some param columns (e.g. ChunkPPO has no `train.rollout_length`), and the default silently drops those rows. After changing the report, check every arm in the registry appears (compare the run count per `model_id` with the table).
- `python scripts/mlflow_report.py summary|top|ach|models|arch NAME|size`. Headline metrics have the same names in every experiment: `score_pct` (Crafter score = geometric mean of the 22 rates) and `reward_pct`; per-achievement metrics are `ach_rate_pct/<NN>_<name>`. The Models tab is rebuilt from `MODEL_REGISTRY.yaml` on every ingest (one version per configuration, numbers are a derived snapshot; `proper_name` is only copied). Changing metric names needs `rm -rf mlflow_local` + re-ingest (stop `mlflow ui` first: it locks the DB on Windows). If the store reaches 1 GiB, report it to the operator (next step: Cloudflare D1, not on your own).

### Dependency impact checks (scip-neo4j MCP server)

Run these checks before modifying any **existing** symbol (function, class, or method). `symbol_id` is the SCIP symbol string.
1. `check_symbol_constraints`: call it on every symbol you intend to modify. `ALLOWED` means proceed. `LOCKED_CORE_THEORY` means the symbol is an immutable core formula. Do not edit it. Follow `diagnostic_feedback` and look for the cause in its callers instead.
2. `analyze_change_impact`: before changing a signature or behavior, list all transitive dependents and review or update each one in the same task.
3. `get_symbol_dependencies`: use it to explore call structure (direction `upstream|downstream|both`, `max_depth` defaults to 2).

The graph reflects the last ingestion run, not the working tree. If a result is empty or looks wrong, treat it as incomplete and read the code, and do not assume the symbol is safe to change. If the server can't be reached, record that as a blocker in `STATE.yaml`. Continue only when you can confirm the change's scope by reading the code, and never modify suspected core-theory code unchecked. Add a one-line impact summary (dependents checked, locked symbols found) to the task's `result`.

## Data-integrity rules (enforced by `scripts/pre_commit_ban_check.py`)

- All metrics must come from real forward passes or env steps. Never use synthetic scores such as `np.random`.
- Do not put the literal string `VERIFIED_SUCCESS` in code.
- Any `.py` file whose name contains `retrieve`, `verify`, or `real` must perform real I/O, an env step, or a model forward pass.
- A dict assigned to a name containing `metrics`, `results`, `rollout`, or `benchmark` may hold at most 2 numeric literals, unless the lines are annotated with `# SOURCE:`.

## Commands

Run everything from the repo root. The scripts insert `.` into `sys.path` and import `src.*`.

```bash
python -m unittest tests.test_architecture_fixes                                   # all tests
python -m unittest tests.test_architecture_fixes.TestArchitectureFixes.<test_name> # single test
python scripts/pre_commit_ban_check.py                                             # data-integrity lint
python -m src.main                                   # synthetic DecisionProcessEnv train/eval demo
python run_standalone.py                             # Craftax-Classic benchmark suite (writes output/)
docker compose -f docker/compose.yaml up --build     # CUDA 12 + JAX container, runs run_standalone.py
```

Dependencies (see `docker/Dockerfile`): `jax[cuda12]`, `flax`, `optax`, `gymnax`, `craftax`, `matplotlib`, `numpy`, `pandas`, and `kaggle`.

### Kaggle remote GPU workflow

Heavy training runs on Kaggle, on kernel `bfloat16/craftax-classic-1000-episode-rl-benchmark`.
- `build_kaggle_kernel.py` base64-embeds the whole local `src/` tree into `kaggle_kernel/decision_process_benchmark.ipynb`, together with a `config.json`. The notebook then runs `kaggle_kernel/phase2_runner.py`. A `src/` change only reaches Kaggle after you rebuild the notebook.
- `scripts/kaggle_run.py` is the end-to-end orchestrator: build, push, poll with live logs, fetch into `output_remote/<run-id>/`, analyze, and plot. It supports `--sweep`, `--fetch-only`, and `--plot-only`.
- `output_remote/*/src` (untracked) are **snapshots** of `src/` that ran remotely. Do not edit them. Edit `src/`.

## Architecture (`src/`)

All data structures are `NamedTuple` PyTrees, defined in `src/model/types.py`. Model code is pure functions of the form `init_*_parameters(key, ...)` → params plus `forward_*(params, input, ...)`. It is not Flax modules.

- **Input/output contract:** the env produces `InputContextN`. It contains `ActionsData` (the candidate action set A), `SystemState` (S), `ActionHistory` (H), and `TransitionTarget` (T). The model outputs `DecisionVectorD`, or `HierarchicalDecisionVectorD` for the hierarchical model. State transitions follow `S_{t+1} = S_t + W_res[A_t]`, and they are computed **after** action selection. This is required: see FAILURE-002.
- **`model/channel_encoder.py`:** channel-independent projection of the A/S/H/T channels, each with its own channel positional embedding. It prevents scale oversmoothing (FAILURE-003).
- **`model/transformer_decision_core.py`:** the 4th-idea core. It has a block attention mask (bidirectional over the action set, causal over history), noise injection into history during training (p≈0.15), and a validity head V∈[0,1].
- **`model/hierarchical_transformer.py`:** the 5th idea. It adds a macro cluster head and a micro action head (toggled by `use_hierarchical`), restricted local sliding-window attention, and working-memory compression of history (`z_step`).
- **`model/beam_search.py`:** fixed-width beam search (K is static and inactive beams are masked) holding explicit (S_t, A_t) pairs. It has flat and hierarchical variants.
- **`environment/gymnax_decision_env.py`:** a synthetic `DecisionProcessEnv`. It is pure JAX and must stay `jit`/`vmap`-safe (use `jnp.where`, not Python `if`).
- **`environment/craftax_env_adapter.py`:** wraps `CraftaxClassicSymbolicEnv` into the same `InputContextN` interface and computes the Crafter score.
- **`pipeline/`:** training and eval drivers. `trainer.py` handles the synthetic env. `craftax_benchmark.py` handles the Craftax RL suite. `off_policy_trainer.py` and `benchmark.py` handle the off-policy/abstraction benchmarks. `hierarchical_pipeline.py` runs the macro/micro engine with `lax.scan`. `grid_search_benchmark.py` covers grid search. `plotter.py` writes to `output/plots/`.

JAX constraints: keep shapes static inside JIT (fixed beam width, KV cache pre-allocated to N_max=1524), and never call `int()` on tracers.

## Repo layout

- Root: entry points only (`run_standalone.py`, `build_kaggle_kernel*.py`), `CLAUDE.md`, config.
- `src/` code; `tests/` unit tests; `scripts/` runnable tools, indexed in `scripts/README.md` (flat: they import each other as siblings); `scripts/adhoc/` throwaway checks.
- `docs/` specs (`core/`, `DECISIONS/`, `JP-ideas/`, `LOGS/`, `experiments/`, `archive/`); `logs/` old run logs; `docker/` incl. `jupyter-python314/`.
- `kaggle_kernel*/` generated notebooks; `output/` results, all in LFS, indexed in `output/README.md`; `output_remote/` is the untracked raw Kaggle-fetch cache (snapshots, do not edit). Kaggle dirs stay at root because kernel paths are hardcoded.

## Repo notes

- `output/**` is stored in Git LFS, with the LFS remote on Hugging Face (`.lfsconfig`).
- `kaggle.json` credentials are gitignored. Docker sets `KAGGLE_CONFIG_DIR=/workspace`.
