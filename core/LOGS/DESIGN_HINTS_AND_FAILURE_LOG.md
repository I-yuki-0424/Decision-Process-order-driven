# Design Hints, Failure Log & Engineering Notes

This document captures historical failure patterns, critical design hints, edge cases, and static analysis notes for the **4th-Idea Decision Process System**.

---

## 1. Historical Failure Modes (1st to 3rd Idea Analysis)

### [FAILURE-001] Autoregressive Error Accumulation (Exposure Bias)
- **Symptom**: In 3rd-Idea greedy inference, the model achieved $>90\%$ step accuracy during training, but collapsed after ~20-30 steps during test-time autoregressive generation.
- **Root Cause**: Training used 100% clean ground-truth histories (Teacher Forcing). Minor inference prediction errors led to out-of-distribution input context $H_t$, triggering cascading decision failures.
- **4th-Idea Mitigation Hint**: Maintain Noise Injection rate ($p \approx 0.15$) in history during training. The attention mechanism must explicitly learn to discount noisy context tokens ($V < 0.5$).

### [FAILURE-002] Action-State Causality Inversion
- **Symptom**: Model generated state transitions $S_{t+1}$ that were physically unachievable by any valid choice in $A$.
- **Root Cause**: Transformer policy head estimated state transitions $S_{t+1}$ independently of the selected action index $A_t$.
- **4th-Idea Mitigation Hint**: Enforce explicit $(S_t, A_t)$ pair state tracking in Beam Search candidates. State update $S_{t+1} = f(S_t, A_t)$ must be computed strictly after action selection.

### [FAILURE-003] Channel Oversmoothing
- **Symptom**: Features with large numeric scales (e.g. monetary cost budget $= 1000$) dominated features with small numeric scales (e.g. status flag $= 0/1$).
- **Root Cause**: Early concatenation of raw heterogeneous feature vectors into a single dense vector prior to attention layers.
- **4th-Idea Mitigation Hint**: Use Channel Independence (`src/model/channel_encoder.py`). Linearly project each channel independently and tag with `channel_pos_embed`.

---

## 2. JAX & Gymnax Implementation Hints

### [HINT-001] JAX Static Shape Constraints in Beam Search
- **Issue**: JAX JIT compilation requires fixed tensor shapes. Dynamic beam pruning with variable beam counts will cause re-compilation triggers.
- **Design Requirement**: Always pad or fix the beam width $K$ (e.g., $K=5$). Mask inactive beams using `active_mask` boolean arrays rather than dynamically resizing arrays.

### [HINT-002] KV Cache Memory Bounds for $N > 1000$ Sequences
- **Issue**: Autoregressive KV cache growth over $100+$ steps can consume significant memory if pre-allocated conservatively.
- **Design Requirement**: Pre-allocate KV cache arrays to max sequence length $N_{max} = 1524$. Use `current_len` index updates to slice active key-values during matrix multiplication.

### [HINT-003] Gymnax `vmap` Compatibility
- **Issue**: Environment step functions must accept batched states without Python side-effects or control flows (`if/else`).
- **Design Requirement**: Use `jnp.where` for conditional state transitions and logic inside `DecisionProcessEnv.step`.

---

## 3. Static Verification Log

- [x] **PyTree Registration**: Verified all dataclasses in `src/model/types.py` inherit from `NamedTuple` for seamless JAX PyTree serialization.
- [x] **Channel Separation**: Verified token embedding sequence in `src/model/channel_encoder.py` tags individual channel vectors.
- [x] **Noise Regularization**: Verified loss computation in `src/pipeline/trainer.py` penalizes validity predictions on noisy sequences.
- [x] **Modular Transformer API**: Verified `forward_decision_transformer` signature is stateless, pure, and reusable.
