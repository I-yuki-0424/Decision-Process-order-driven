# scip-neo4j MCP server: usage feedback log

Purpose: a running, separate record of how well the scip-neo4j MCP tools (`check_symbol_constraints`,
`analyze_change_impact`, `get_symbol_dependencies`, `find_semantic_neighbors`) actually perform on this repo, kept
apart from `docs/core/STATE.yaml` task entries per the operator's request (2026-09-28), since the operator noted the
server is still under development and its results are often not useful. Each entry cites the task that produced it.
The operator asked (2026-09-28, second pass) for a full re-review covering all four tools' correctness, ease of use,
and reliability, noting the server may have improved server-side even if this session's results look the same.

## Per-tool assessment (as of 2026-09-28, second pass)

### `check_symbol_constraints`
- **Correctness:** Cannot be verified as correct, because it has never once returned anything other than
  `{"status": "ALLOWED", "diagnostic_feedback": ""}` across every symbol tried in this repo's history (see entries
  below) -- including a symbol whose name doesn't exist anywhere in the codebase (`this_symbol_does_not_exist_xyz123`,
  tested today). It does not appear to validate that the symbol was actually found in the index before answering
  "ALLOWED." That means an "ALLOWED" response carries no assurance the tool even located the symbol, which is a
  real correctness gap for a gate whose entire job is to say "safe to edit" vs "locked."
- **Ease of use:** Trivial to call (one string argument), fast, no observed errors on malformed/unknown input (it
  just answers ALLOWED instead of erroring, which is itself part of the problem above).
- **Reliability:** Never seen it flag `LOCKED_CORE_THEORY` on anything, including plausible core-theory candidates
  tried today (`DecisionProcessEnv.step`, which implements the CLAUDE.md-cited `S_{t+1} = S_t + W_res[A_t]` exact
  transition). Either nothing in this repo is tagged core-theory yet, or the tag path has never fired in this
  session's testing -- can't distinguish those from the outside, which limits how much this gate can be trusted as
  a safety net right now.

### `analyze_change_impact`
- **Correctness:** Poor so far. Tally across all sessions: **7 symbols tested with real, grep-confirmed callers,
  and 6 of them came back with an empty dependent list** (`save_model_checkpoint`/`load_model_checkpoint` -- empty
  but happened to be correct since those specific two truly have no external callers; `build_mdp_transition_array`,
  `craftax_calibration_config`, `sample_chunk`, `DecisionProcessEnv.step`, `forward_decision_transformer` -- all
  empty despite real callers found by grep in the same session). The **one** case where an empty result was
  actually correct (`build_configs`, a script-local helper with genuinely zero external callers) shows the tool can
  be right when a symbol truly is a dead end -- but there is no way to tell that case apart from a false negative
  by looking at the tool's own output; every empty result still needs a grep pass regardless.
- **Ease of use:** Trivial to call, fast, no errors.
- **Reliability:** Low for this repo. An empty list must be treated as "incomplete," never as "safe to change," per
  CLAUDE.md's own fallback rule -- which in practice means this tool has not yet saved a grep step in this project,
  only added a preliminary (and usually misleading) one.

### `get_symbol_dependencies`
- **Correctness:** Same pattern as `analyze_change_impact`, tested today on `sample_chunk`, `forward_decision_transformer`
  (both `direction=upstream` and `direction=downstream` separately, not just `both`), and `build_mdp_transition_array`
  -- all four returned "No dependencies found" despite each having real callers/callees confirmed by grep or by
  this session's own code reading. Direction did not matter; all directions came back empty identically.
- **Ease of use:** Trivial to call, `max_depth` and `direction` params are self-explanatory, fast, no errors.
- **Reliability:** Low so far, same caveat as `analyze_change_impact` -- not yet observed to return a populated,
  correct call tree for anything in this repo.

### `find_semantic_neighbors`
- **Correctness:** Mixed, and this is the most informative tool of the four because its own docstring is honest
  about being "similarity-ranked guesses, not resolved edges." Two free-text/file-path queries today
  (`"greedy policy evaluation return bias truncated episode window"` and the literal path `src/pipeline/chunk_ppo.py`)
  returned **zero** results from `src/pipeline/`, `src/model/chunk_policy.py`, `src/model/factored_world_model.py`,
  or `src/environment/latency_envs.py` -- the entire TASK-012/013 codebase this session is working in -- and instead
  surfaced unrelated older scripts (`scripts/adhoc/diagnose_mdp_nan.py`, `kaggle_kernel/phase2_runner.py`,
  `scripts/real_end_to_end_pipeline.py`, etc.). Checked whether those returned files are just stale/deleted: they
  all still exist and are not particularly old (`scripts/aggregate_pfirst.py` and `scripts/kaggle_push_pwm.py` were
  both last touched the day before this test), so the index is not simply ancient -- it specifically appears not to
  have ingested the TASK-012/013 files yet (created/committed earlier today), consistent with CLAUDE.md's own note
  that "the graph reflects the last ingestion run, not the working tree." A third query, the bare symbol name
  `"build_mdp_transition_array"` (an older, definitely-should-be-ingested symbol from 2026-09-25), returned one
  plausibly-relevant hit (`candidate_smoke_runner.py`'s `transitions_by_model`, score 0.62) buried among four
  unrelated local-variable-level matches (`arr`, `arr2`, `seed`, `inp` from a diagnostic script) -- so the tool is
  not purely noise, but the signal-to-noise ratio is low and query wording matters a lot.
- **Ease of use:** Trivial to call (query + top_k), fast, no errors, and its description sets expectations
  correctly up front (unlike the other three tools, which don't warn that "no dependents" might just mean
  "not indexed").
- **Reliability:** Low-to-moderate. Best used, as its own docstring suggests, for finding conceptually-related code
  a call-graph tool cannot see at all -- not as a substitute for grep, and not yet reliable enough to skip a manual
  search when the embedding coverage for the exact file/symbol in question is unknown.

## Overall, across all four tools

Nothing tested in this second pass changed the working conclusion from the first pass: every result still needs a
grep/code-read to confirm, for exactly the reasons CLAUDE.md's fallback rule anticipates. The operator noted the
server has reportedly improved server-side since the first pass; this session's results look unchanged, which is
consistent with either "the improvement hasn't reached this repo's ingestion yet" or "the ingestion for this repo's
newest files (TASK-012/013, committed today) simply hasn't run since they were added" -- both point at the same
practical action either way: re-run this same test matrix in a fresh session after the next ingestion, rather than
assuming today's unchanged results mean the server-side fix didn't work.

## Entries (chronological)

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

### 2026-09-28, TASK-20260928-013-latency-goal-chunking-calibration
`check_symbol_constraints` on `craftax_calibration_config` (before editing its `num_envs` default from 64 to 1024)
→ `ALLOWED`.
`analyze_change_impact` on `craftax_calibration_config` → **empty dependent list again**, despite two real,
grep-confirmed callers that both actually invoke it: `scripts/run_latency_experiment.py` (`build_configs`, calls
`craftax_calibration_config(**common)`) and `tests/test_latency_chunking.py`
(`TestCraftaxCalibrationWiring.test_calibration_config_dry_run`). Same failure mode as the 2026-09-25 entry above.

### 2026-09-28, TASK-20260928-014 (this task, second full-tool review requested by the operator)
Repeated `check_symbol_constraints`/`analyze_change_impact` on `craftax_calibration_config` again as asked, to check
for server-side improvement within the same session → identical result to before (`ALLOWED`, empty list) — no
change observed in this session, consistent with the operator's own expectation that a server-side fix might not
show up mid-session.
New tests this round, all against real code in this repo:
- `check_symbol_constraints` on a **nonexistent** symbol name (`this_symbol_does_not_exist_xyz123`) → also
  `ALLOWED`. This is a new, more serious finding than earlier entries: the tool does not appear to distinguish "not
  found in the index" from "found and unrestricted," so an `ALLOWED` answer alone cannot be trusted as evidence the
  symbol was actually resolved.
- `check_symbol_constraints` + `analyze_change_impact` on `DecisionProcessEnv.step` (implements the CLAUDE.md-cited
  locked-core-theory-candidate transition `S_{t+1} = S_t + W_res[A_t]`) and on `sample_chunk`, `forward_decision_transformer`
  → all `ALLOWED` / empty, despite `forward_decision_transformer` and `sample_chunk` both having real,
  grep/test-confirmed callers.
- `get_symbol_dependencies` (direction `upstream`, `downstream`, and `both` tried separately) on the same three
  symbols plus `build_mdp_transition_array` → "No dependencies found" in every direction, every symbol.
- `analyze_change_impact` on `build_configs` (a script-local helper in `scripts/run_latency_experiment.py` with
  genuinely zero external callers, confirmed by grep) → empty list, and this time **correctly** empty — the one
  clean result found across all testing so far, useful as a reminder that the tool isn't universally wrong, just
  not distinguishable from wrong without checking.
- `find_semantic_neighbors` with two different query styles (free-text description of the eval-window-bias bug
  under investigation, and the literal path `src/pipeline/chunk_ppo.py`) → zero relevant results; every match came
  from files outside `src/pipeline/`, `src/model/chunk_policy.py`, `src/model/factored_world_model.py`, and
  `src/environment/latency_envs.py` (the entire TASK-012/013 codebase), despite those files existing and having
  been committed hours before this test — strong evidence the embedding index has not been re-ingested since
  TASK-012 landed, matching CLAUDE.md's "reflects the last ingestion run, not the working tree" caveat.
- `find_semantic_neighbors("build_mdp_transition_array")` (an older, 2026-09-25 symbol that should be well within
  any reasonable ingestion window) → one plausibly-relevant hit (`candidate_smoke_runner.py`'s `transitions_by_model`)
  among otherwise-noisy local-variable matches — the clearest evidence yet that this specific tool has *some* real
  signal, just at a low signal-to-noise ratio, and query wording/specificity matters.

## Working conclusion (revisit again after the next server-side ingestion)

Treat `check_symbol_constraints`'s `ALLOWED` as "not explicitly locked," never as "confirmed found and safe" —
today's nonexistent-symbol test shows it does not gate on the symbol actually resolving. Treat
`analyze_change_impact` and `get_symbol_dependencies` results as **advisory only** for this repo: across 8 total
real-caller test cases (2026-09-25 through 2026-09-28), only 1 empty result was correct, and it is not possible to
tell the correct case apart from the 7 wrong ones from the tool's output alone — a grep/code-read pass is still the
actual safety net, not the tool. `find_semantic_neighbors` is the one tool that occasionally adds real signal (a
plausible neighbor a grep for the exact symbol name would miss), but it currently has no visibility at all into
this session's newest source files, so it should not be relied on for anything committed very recently. None of
this rules out the operator's expectation that server-side changes will eventually show up here — it just means
this specific session, on this specific (partly very-recently-changed) repo state, did not observe a difference.
