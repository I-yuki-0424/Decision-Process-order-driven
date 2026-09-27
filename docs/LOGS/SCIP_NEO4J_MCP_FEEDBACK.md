# scip-neo4j MCP server: usage feedback log

Purpose: a running, separate record of how well the scip-neo4j MCP tools (`check_symbol_constraints`,
`analyze_change_impact`, `get_symbol_dependencies`, `find_semantic_neighbors`) actually perform on this repo, kept
apart from `docs/core/STATE.yaml` task entries per the operator's request (2026-09-28), since the operator noted the
server is still under development and its results are often not useful. Each entry cites the task that produced it.

## Summary so far

The server is reachable and `check_symbol_constraints` reliably returns `ALLOWED` / not-locked status for ordinary
functions. Its usefulness beyond that single boolean has been consistently poor for this repo:

- `analyze_change_impact` has returned an **empty dependent list on every symbol tested so far, including symbols with
  real, verifiable callers** (confirmed by grep every time). This makes its output indistinguishable from "the index
  doesn't cover this code" vs. "this symbol truly has no callers" — the empty-list case has never yet been
  trustworthy on its own here, per CLAUDE.md's fallback rule (treat empty/wrong results as incomplete, verify with
  code/grep, never treat empty as "safe").
- `find_semantic_neighbors` returned no relevant matches for `transformer_decision_core.py`-family symbols
  (TASK-20260927-010), i.e. it did not surface known-related code that a grep search or manual read immediately
  finds.
- No case so far where a scip-neo4j result changed a decision that manual grep/code-read did not already settle
  correctly by itself. It has added a confirmation step, not new information.

## Entries

### 2026-09-25, TASK-20260925-001-dpod-candidates
`check_symbol_constraints` on `src.model.checkpoint.save_model_checkpoint` → `ALLOWED`.
`analyze_change_impact` on `save_model_checkpoint` and `load_model_checkpoint` → empty dependent list for both.
Confirmed by grep: only `src/model/checkpoint.py` itself references them, so the empty result happened to be correct
here — but this was verified independently, not trusted from the tool alone.

### 2026-09-25, TASK-20260925-002-candidates-smoke-verify
`check_symbol_constraints` on `shared_stages.build_mdp_transition_array` and
`plotter.plot_predicted_transition_distribution` → both `ALLOWED`.
`analyze_change_impact` on `build_mdp_transition_array` → empty dependent list, explicitly logged as "treated as
incomplete per protocol, not as safe." Grep then found its **real** callers: `mdp_branch.py`,
`variant_5_1_shared_attention.py`, `variant_5_4_single_shot.py` — three real dependents that the tool missed
entirely.

### 2026-09-27, TASK-20260927-010-learner-power-budget-check
`find_semantic_neighbors` on the `transformer_decision_core.py` action-scoring path → no relevant matches returned.
Index treated as incomplete for this file; the actual question (does a state-channel signal reach action scores in
one hop) was answered by direct code read instead, per CLAUDE.md's fallback.

### 2026-09-27, TASK-20260927-010 (impact-check note carried into TASK-011 handoff)
Recorded explicitly as a standing caution for the next task (actor-critic learner rewrite of
`candidate_experiment.build_fns`/`loss_fn`): expect the same empty/irrelevant-result pattern and fall back to grep
without skipping the check.

### 2026-09-28, TASK-20260928-013-latency-goal-chunking-calibration (this task)
`check_symbol_constraints` on `craftax_calibration_config` (before editing its `num_envs` default from 64 to 1024,
see `docs/core/STATE.yaml`) → `ALLOWED`.
`analyze_change_impact` on `craftax_calibration_config` → **empty dependent list again**, despite two real,
grep-confirmed callers that both actually invoke it: `scripts/run_latency_experiment.py` (`build_configs`, calls
`craftax_calibration_config(**common)`) and `tests/test_latency_chunking.py`
(`TestCraftaxCalibrationWiring.test_calibration_config_dry_run`, calls
`craftax_calibration_config(num_envs=2, cycles_per_update=4, num_minibatches=2)`). This is the same failure mode as
the 2026-09-25 entry above: a function called from two real, easily-grep-able sites was reported as having zero
dependents. The empty-list output continues to carry no reliable information here — always confirm with grep before
treating a symbol as safe to change.

## Working conclusion (revisit as more data comes in)

Treat `check_symbol_constraints` `ALLOWED`/`LOCKED_CORE_THEORY` as informative (it is a direct read of a stored flag,
not an index traversal). Treat `analyze_change_impact`, `get_symbol_dependencies`, and `find_semantic_neighbors`
results as **advisory only** for this repo until the underlying ingestion index is confirmed current — every
non-trivial check so far has needed a manual grep pass regardless of what the tool returned, so the manual pass is
the actual safety net, not the tool. Keep running the checks per CLAUDE.md (a locked-core-theory hit would still be
real signal), but do not let an empty or clean-looking result skip the grep-based verification.
