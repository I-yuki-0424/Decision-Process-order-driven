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

## Operating protocol (`docs/core/MASTER_GUIDANCE.xml`, v9)

- Read `docs/core/STATE.yaml` at session start. Every task is tracked there with the fields `id`, `created_at`, `status` (queued/in_progress/blocked/done), `description`, `spec_ref`, `validation_commands`, `result`, and `blockers`. Update status and result as soon as either changes. Record material assumptions in the task entry. Quote fields directly and report missing ones as missing.
- **Strictness:** do not implement unconfirmed components based on your own assumptions. Follow the idea docs, the ADRs in `docs/DECISIONS/`, and `docs/LOGS/DESIGN_HINTS_AND_FAILURE_LOG.md`, subject to the critical review section above.
- Verification means running the task's own `validation_commands`. There is no default smoke path.
- Commit at natural checkpoints and push after each commit.
- Use Docker only when a task needs an isolated or reproducible environment. Never require a GPU to start a session.
- Long GPU training runs need operator authorization. ADR-002 authorizes Phase II runs of 1M–10M steps on Craftax-Classic via Kaggle. Its conditions are checkpoints every 100K steps, resumability, and a 24h limit per config.

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
- `output_remote/*/src` and `kaggle_logs/src` are **snapshots** of `src/` that ran remotely. Do not edit them. Edit `src/`.

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

## Repo notes

- `output/**` is stored in Git LFS, with the LFS remote on Hugging Face (`.lfsconfig`).
- `kaggle.json` credentials are gitignored. Docker sets `KAGGLE_CONFIG_DIR=/workspace`.
