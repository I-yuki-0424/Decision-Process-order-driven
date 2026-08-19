# Architecture Review: `Decision-Process-order-driven` (4th-Idea JAX Architecture)

**Repository:** `sigure-0424/Decision-Process-order-driven` (branch `main`)
**Reviewed:** 2026-07-27
**Method:** Direct reading of source code fetched from the repository archive (`src/`, `core/`). No code was executed; this is a static review.

**Summary:** The design documents (`core/DECISIONS/ADR-001_4TH_IDEA_JAX_ARCHITECTURE.md`, `core/GOAL.md`) describe a coherent set of design intentions. However, several claims in those documents are not supported by the actual implementation, and one core mechanism (state transition) is mathematically unjustified. The project has never been executed or trained — verification is limited to a Python syntax check.

---

## 1. Causal mask applied to a non-sequential token set

**Location:** `src/model/transformer_decision_core.py`, lines 214–216 (mask construction), combined with `src/model/channel_encoder.py`, line 102 (token concatenation order).

```python
# channel_encoder.py, line 102
all_tokens = jnp.concatenate([state_token, target_token, act_tokens, hist_tokens], axis=0)
```

```python
# transformer_decision_core.py, lines 214-216
mask = jnp.tril(jnp.ones((seq_len, kv_seq_len)))
attn_weights = jnp.where(mask[None, :, :] == 1, attn_weights, -1e9)
```

**Problem:** A lower-triangular (causal) mask is applied to the entire concatenated token sequence `[State, Target, Actions..., History...]`. `Actions` is defined in `src/model/types.py` (lines 18–32, `ActionsData`) as an unordered **set** of `num_actions` candidate choices, not a time series. Under the causal mask, action token `i` can only attend to action tokens `0..i-1`. This imposes an arbitrary ordering constraint (based on array index, not time or causality) on a set that has no intrinsic order. There is no stated justification for this in the design documents, and it is inconsistent with treating `Actions` as a set of simultaneous candidates.

**Why it matters:** The model's output for a given action can depend on its position in an arbitrary index array, which is not a meaningful inductive bias and is not something the documentation claims or defends.

---

## 2. State transition function is a dimensional hack, not a defined function

**Location:** Same code duplicated in four places:
- `src/model/beam_search.py`, lines 145–148
- `src/pipeline/off_policy_trainer.py`, line 57
- `src/pipeline/benchmark.py`, lines 97 and 210

```python
delta_res = jax.vmap(lambda c: jnp.tile(c, (num_res // num_costs + 1,))[:num_res])(selected_costs)
updated_resource_levels = new_beams_parent.state.resource_levels + delta_res
```

**Problem:** `selected_costs` has shape `(num_costs,)` (e.g., money, time — see `ActionsData.costs` in `types.py`), and `resource_levels` has shape `(num_resources,)`. These are declared as distinct physical quantities. The code reconciles the shape mismatch by tiling the cost vector cyclically and truncating it to `num_resources` length, then adding it directly to resource levels.

**Problem in terms of the ADR's own claims:** `core/DECISIONS/ADR-001...md` states, as the fix for `[FAILURE-002] Action-State Causality Inversion`:

> "State update $S_{t+1} = f(S_t, A_t)$ must be computed strictly after action selection."

The implementation does not define any such function `f`. Instead it performs an unmotivated tile-and-truncate operation that maps one physical quantity onto another with no semantic correspondence (e.g., a "time cost" value could end up added to a "material" resource, or vice versa, depending on the arbitrary index alignment produced by `tile`). This is not a learned or specified transition function — it is a shape-compatibility workaround.

**Why it matters:** This is the core state-update mechanism used both in beam search rollout and in the benchmark/off-policy training pipeline. If the mapping is not physically meaningful, downstream metrics computed from `resource_levels` (progress rate, cost violation) are not meaningful either.

---

## 3. "Validity score" does not implement the claimed token-level noise discounting

**Location:** `src/pipeline/trainer.py`, lines 58–60.

```python
target_validity = 1.0 - jnp.mean(input_n.history.noise_mask.astype(jnp.float32))
validity_loss = - (target_validity * jnp.log(decision_d.validity_score + 1e-6) +
                   (1.0 - target_validity) * jnp.log(1.0 - decision_d.validity_score + 1e-6))
```

**Problem:** ADR-001 describes the noise-injection mechanism as follows:

> "Attention heads learn to assign low weight to noise tokens and redirect attention to valid paths. Multi-task head includes a validity prediction score $V \in [0,1]$ regularizing recovery."

This implies per-token discrimination of noisy vs. clean tokens via attention. What is actually implemented is a single scalar `validity_score` (from `transformer_decision_core.py`, line 286, computed only from the final pooled token) regressed against the **aggregate fraction of clean tokens in the whole history**. There is no loss term, gradient path, or architectural component that ties `validity_score` to per-token attention weights. The mechanism is a weak sequence-level regularizer at best, not the token-level noise-discounting mechanism described in the design document.

---

## 4. JAX/Gymnax claims are not supported by the code

**Location:** `core/GOAL.md`, line 5; `core/DECISIONS/ADR-001...md`, "Decision Drivers" section; `src/environment/gymnax_decision_env.py`, lines 1–52.

**Problem 1 — No JIT compilation anywhere:**
`GOAL.md` states: "Use JAX as the primary framework, maximizing execution speed via JIT compilation wherever possible." ADR-001 lists "Leverage JAX JIT compilation and `vmap` primitives for high-throughput vectorized simulation" as a decision driver. A repository-wide search (`grep -rn "jax.jit" src/`) returns **zero matches**. No function in the codebase is JIT-compiled. `vmap` is used in only two places (`beam_search.py`, and the duplicated `delta_res` computation from Section 2 above), not throughout the pipeline as implied.

**Problem 2 — No actual Gymnax dependency:**
`src/environment/gymnax_decision_env.py` describes itself as "Gymnax-Compatible" (line 2) and its class as "Gymnax-style" (line 49), but:
- There is no `import gymnax` anywhere in `src/`.
- `DecisionProcessEnv` does not subclass any `gymnax` base class (e.g., `gymnax.environments.environment.Environment`).
- No `requirements.txt` or `pyproject.toml` exists in the repository to declare `gymnax` (or `jax`, `optax`) as a dependency.

**Why it matters:** The framework choice is presented as a load-bearing design decision (throughput via JIT/vmap, compatibility with the Gymnax ecosystem), but neither claim is verifiable from the code as it stands.

---

## 5. "Verification" claims overstate what was actually checked

**Location:** `core/STATE.yaml`, lines 14–17; `core/LOGS/DESIGN_HINTS_AND_FAILURE_LOG.md`, "Static Verification Log" section (lines 42–47).

```yaml
smoke_status:
  last_run: "2026-07-25"
  command: "python -m py_compile src/main.py"
  result: "Static analysis passed"
```

**Problem:** The only recorded verification is a Python syntax check (`py_compile`), which confirms the file parses but does not execute any code path, does not check tensor shapes at runtime, and does not validate any of the model's numerical or learning behavior. `DESIGN_HINTS_AND_FAILURE_LOG.md` lists four "Verified" static-analysis items (PyTree registration, channel separation, noise regularization, modular API), all of which are code-inspection-level claims, not execution results.

**Consequence:** There is no training run, benchmark result, checkpoint, or log file anywhere in the repository (`src/`, `kaggle_kernel/`) demonstrating that the model trains, converges, or achieves the stated goal of "≥80% success rate over N > 100 step sequences" (`ADR-001`, "Decision Drivers"). The architecture is unvalidated by any empirical evidence currently in the repository.

---

## Notes on provenance

- `core/ACTIVITY_SUMMARY.md` attributes the implementation to an agent identified as `gemini`, under task `TASK-20260725-001-4th-idea-jax-architecture`, dated 2026-07-25 (two days before this review).
- `ADR-001` lists its authors as "Antigravity AI & System Architecture Team."
- These indicate the codebase and its design documents were produced by an AI agent, not hand-verified by a human engineer, which is consistent with the gap observed between the documents' claims and the actual code (Sections 1–5 above).

---

## Summary Table

| # | Issue | Files | Severity |
|---|-------|-------|----------|
| 1 | Causal mask applied to unordered action set | `transformer_decision_core.py:214-216`, `channel_encoder.py:102` | Design flaw — unjustified inductive bias |
| 2 | State transition is a dimension-tiling hack, not `f(S_t, A_t)` | `beam_search.py:145-148`, `off_policy_trainer.py:57`, `benchmark.py:97,210` | Core correctness issue |
| 3 | Validity score is sequence-level, not token-level as claimed | `trainer.py:58-60` | Documentation/implementation mismatch |
| 4 | No `jax.jit` usage; no real `gymnax` dependency | repo-wide; `gymnax_decision_env.py:1-52` | Documentation/implementation mismatch |
| 5 | "Verification" = syntax check only; no execution evidence | `STATE.yaml:14-17`, `DESIGN_HINTS_AND_FAILURE_LOG.md:42-47` | Unvalidated claims |
