"""Candidate decision-model architectures ported from docs/DPOD.ipynb.

Each module implements one notebook candidate as `init_*_parameters(key, ...) ->
params` plus `forward_*(params, input_n: InputContextN, ...) -> DecisionVectorD`,
following the same convention as src/model/transformer_decision_core.py and
src/model/hierarchical_transformer.py (NamedTuple PyTree params, pure functions,
no Flax modules, channel-encoded A/S/H/T tokens from src/model/channel_encoder.py).

See docs/core/STATE.yaml for the critical-review objections these candidates
carry over from the notebook (mean-pool discretization defect, the
Variant5_1Bace shape bug, WorldModel's deferred/non-wired status, and the
dropped dead-code utilities/dispatcher).
"""
