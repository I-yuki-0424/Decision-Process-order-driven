# ADR-002: Phase II Training Scale Authorization (1M–10M Steps)

- **Status**: Approved
- **Date**: 2026-07-30
- **Authors**: User (Operator) + Antigravity AI
- **Supersedes**: MASTER_GUIDANCE.md § 10.3 default prohibition (for this scope only)

---

## Context & Problem Statement

`MASTER_GUIDANCE.md § 10.3` prohibits full-scale training on large datasets and long GPU runs
without explicit operator authorization, due to cost and failure-mode risk. Phase II Model
Limit Testing requires training 4th-Idea Decision Transformer models for 1M–10M steps on the
Craftax-Classic environment to move beyond local rule-based logic and validate learned
model-based transitions.

---

## Operator Authorization

**Explicitly granted** by operator instruction on 2026-07-30.

Permission scope:
- Training scale: **1M to 10M+ steps** on Craftax-Classic (22 achievements, gymnax JAX environment)
- Hardware target: A100/H100 GPU clusters (via Kaggle GPU containers)
- Framework: JAX + gymnax + `jax.lax.scan` + `jax.vmap`
- Zero synthetic data: all metrics must come from real model inference

---

## Decision Outcome

Full-scale Phase II training and grid search are authorized under the following constraints:

### Hard Limits & Safeguards

| Constraint | Requirement |
|---|---|
| Checkpointing | Checkpoint every 100K steps using `src/model/checkpoint.py` |
| Resumability | Training must be resumable from any checkpoint (`--resume` flag) |
| Time limit | Hard wall-clock limit: 24 hours per grid configuration |
| Throughput target | ≥ 0.07s/1M steps on A100 (ensured by JIT + lax.scan) |
| Memory bound | KV cache pre-allocated to N_max=1524; no dynamic growth |
| Data policy | **Zero `np.random` synthetic scores**; all metrics from real forward passes |

### Grid Search Configuration (Authorized)

| Hyperparameter | Values | Notes |
|---|---|---|
| Layer Count N | 4, 6, 8 | Centered on N=6 per Phase II spec |
| Training Steps | 1M, 5M, 10M | Sequential; earlier configs inform later |
| Z-Compression | 32, 64, 128 | Z=32 most aggressive; 128 most faithful to history |
| Beam Width k | 5, 8, 16 | k=16 reserved for A100-scale only |
| Attention Mask | Causal / Non-Causal | Resolves Mohammadi et al. (2025) masking question |

### Execution Platform

| Platform | Command |
|---|---|
| Local smoke (CPU) | `python -m src.pipeline.grid_search_benchmark --steps 200` |
| Kaggle GPU (A100) | `python -m src.pipeline.grid_search_benchmark --steps 1000000` |
| Docker (local CUDA) | `docker compose run dev python -m src.pipeline.grid_search_benchmark --steps 10000` |

---

## Achievement Milestone Mapping (Decision 2)

"Gold Harvesting" is not present in Craftax-Classic's 22-achievement set.
Approved replacement:

| Protocol Milestone | Craftax Achievement | Index |
|--------------------|---------------------|-------|
| Wood Collection    | collect_wood        | 0     |
| Stone Acquisition  | collect_stone       | 9     |
| **Coal Harvesting** (replaces Gold) | collect_coal | 17 |
| Iron Smelting      | collect_iron        | 18    |
| Diamond Collection | collect_diamond     | 19    |

This mapping is encoded in `src/pipeline/grid_search_benchmark.py::PHASE2_MILESTONES`.

---

## Required Recording

- [x] ADR created (this document)
- [ ] STATE.yaml `active_tasks` updated with Phase II task ID
- [ ] ACTIVITY_SUMMARY.md updated after each training run
- [ ] Checkpoint artifacts stored in `output/checkpoints/phase2/`

---

## Termination Conditions

Stop training a configuration early if:
1. Diamond achievement unlocked (terminal milestone reached)
2. Crafter score plateaus for 500K steps (< 0.5% improvement)
3. Wall-clock limit of 24h exceeded
4. OOM or CUDA error — restart from last checkpoint

---

## References

- Bellman (1957) — MDP Optimality Principle
- Mohammadi et al. (2025) — Progress Rate guidelines
- Nie et al. (2023) — PatchTST Channel Independence
- Hessel et al. (2021) — Anakin sub-architecture (policy + environment on accelerator)
