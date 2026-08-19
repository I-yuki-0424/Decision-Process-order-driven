# ADR-001: 4th-Idea JAX Decision Transformer & Gymnax Framework Architecture

- **Status**: Approved
- **Date**: 2026-07-25
- **Authors**: Antigravity AI & System Architecture Team

## Context & Problem Statement

Prior research and iterations (1st-Idea through 3rd-Idea) established the mathematical foundation of Markov Decision Processes (MDP) combined with Transformer self-attention for order-driven process generation. However, single-pass Greedy autoregressive inference in 3rd-Idea encountered critical failure modes:
1. **Exposure Bias**: Training on clean teacher-forced sequences caused severe error compounding during test-time autoregressive generation.
2. **Causality Mismatch**: The model predicted state transitions without determining explicit actions first, breaking Bellman's principle of optimality ($f_N(i) = \max_q [b_i(q) + \sum a_{ij}(q) f_{N-1}(j)]$).
3. **Oversmoothing across Heterogeneous Channels**: Scalar variables with different physical scales (time, cost, resource capacities) were mixed in dense layers, leading to feature degradation.

## Decision Drivers

- **Goal Achievement**: Maintain target goal success rate $\ge 80\%$ over long process generation sequences ($N > 100$ steps).
- **Execution Throughput**: Leverage JAX JIT compilation and `vmap` primitives for high-throughput vectorized simulation on accelerators (GPUs/TPUs).
- **Modularity**: Design the decision model with generic, reusable interfaces similar to standard Transformer modules.

## Considered Options

1. **Sequential Greedy Inference (3rd-Idea)**: Low compute per step, but prone to catastrophic error accumulation and local minima.
2. **Hierarchical Goal Inference**: High-level temporal abstraction, but vulnerable to target non-stationarity and feasibility failures.
3. **4th-Idea (Integrated Noise Injection & Beam Search with KV Cache)**:
   - Noise Injection during training to teach context recovery from invalid history states.
   - Vectorized Beam Search maintaining action-state $(S_t, A_t)$ pairs.
   - Channel Independence preserving variable-specific dynamics.

## Decision Outcome

Chosen Option: **4th-Idea Architecture**.

### Key Architectural Components

1. **Channel Independence (`src/model/channel_encoder.py`)**:
   - Variables are processed in distinct channels (Action channel $A$, State channel $S$, History channel $H$, Target channel $T$).
   - Positional embeddings are added per channel before Multi-Head Attention, preventing oversmoothing.

2. **Noise Injection Module (`src/model/transformer_decision_core.py`)**:
   - During training (`is_training=True`), a Bernoulli noise mask injects sub-optimal or invalid choices into history $H$.
   - Attention heads learn to assign low weight to noise tokens and redirect attention to valid paths.
   - Multi-task head includes a validity prediction score $V \in [0, 1]$ regularizing recovery.

3. **Bellman-Causal Beam Search (`src/model/beam_search.py`)**:
   - Branching maintains $K$ concurrent candidates holding explicit $(S_t, A_t)$ pairs.
   - Pruning evaluates candidate scores balancing action log-probability, Goal Progress Rate, and multi-dimensional cost penalties.

4. **Gymnax Vectorization (`src/environment/gymnax_decision_env.py`)**:
   - Fully JAX PyTree compatible environment supporting `vmap` and `jit` execution.

## Verification & Impact

- Static analysis confirms PyTree compatibility across $N$, $d$, KVCacheState, and BeamSearchState.
- Modular design allows `DecisionTransformerCore` to be imported and used generically in any JAX training/reinforcement learning pipeline.
