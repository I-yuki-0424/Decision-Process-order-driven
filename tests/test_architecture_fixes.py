"""
Unit and Integration Tests verifying Architecture Fixes & Hierarchical Macro/Micro Pipeline.

Verifies:
1. Issue 1 Fix: Block Attention Mask (Action set bidirectional attention + History causal mask).
2. Issue 2 Fix: State transition S_{t+1} = S_t + W_res[A_t] semantic matrix alignment.
3. Issue 3 Fix: Token-level validity prediction head and loss.
4. Issue 4 & 5 Fix: JAX JIT compilation and shape preservation.
"""

import unittest
import jax
import jax.numpy as jnp

from src.model.types import (
    ActionsData,
    SystemState,
    ActionHistory,
    TransitionTarget,
    InputContextN,
)
from src.model.transformer_decision_core import (
    init_model_parameters,
    forward_decision_transformer,
    build_decision_attention_mask,
)
from src.model.hierarchical_transformer import (
    init_hierarchical_model_parameters,
    forward_hierarchical_transformer,
)
from src.model.beam_search import (
    beam_search_init,
    beam_search_step,
)
from src.environment.gymnax_decision_env import DecisionProcessEnv, EnvParams
from src.pipeline.hierarchical_pipeline import HierarchicalExecutionEngine, HierarchicalConfig


class TestArchitectureFixes(unittest.TestCase):

    def setUp(self):
        self.rng = jax.random.PRNGKey(42)
        self.num_actions = 16
        self.num_costs = 4
        self.num_resources = 8
        self.hist_len = 32

    def test_build_decision_attention_mask(self):
        """Test Issue #1 Fix: Block Attention Mask structure."""
        num_actions = 16
        num_hist = 32
        mask = build_decision_attention_mask(num_actions, num_hist)
        seq_len = 2 + num_actions + num_hist

        self.assertEqual(mask.shape, (seq_len, seq_len))
        
        # 1. State/Target (idx 0,1) attend to State/Target
        self.assertEqual(float(mask[0, 0]), 1.0)
        self.assertEqual(float(mask[1, 0]), 1.0)

        # 2. Action tokens (idx 2 .. 2+N_act-1) attend bidirectionally to ALL action tokens
        act_start = 2
        act_end = 2 + num_actions
        # Action 0 attends to Action N_act-1 (bidirectional set attention)
        self.assertEqual(float(mask[act_start, act_end - 1]), 1.0)
        self.assertEqual(float(mask[act_end - 1, act_start]), 1.0)

        # 3. History sequence tokens maintain causal lower-triangular mask
        hist_start = 2 + num_actions
        # History step 5 should attend to History step 2
        self.assertEqual(float(mask[hist_start + 5, hist_start + 2]), 1.0)
        # History step 2 should NOT attend to future History step 5
        self.assertEqual(float(mask[hist_start + 2, hist_start + 5]), 0.0)

    def test_semantic_state_transition(self):
        """Test Issue #2 Fix: Exact transition S_{t+1} = S_t + W_res[A_t]."""
        env = DecisionProcessEnv(EnvParams(num_actions=16, num_resources=8))
        obs, env_state, actions_data = env.reset(self.rng)

        self.assertIsNotNone(actions_data.resource_effects)
        self.assertEqual(actions_data.resource_effects.shape, (16, 8))

        # Perform one step with action index 3
        action_idx = 3
        expected_next_resources = env_state.resource_levels + actions_data.resource_effects[action_idx]
        
        next_obs, next_env_state, _, _, _ = env.step(self.rng, env_state, action_idx, actions_data)
        
        # Verify exact mathematical resource state matching
        np_diff = jnp.max(jnp.abs(next_env_state.resource_levels - expected_next_resources))
        self.assertLess(float(np_diff), 1e-5)

    def test_token_validity_logits(self):
        """Test Issue #3 Fix: Per-token validity prediction head."""
        params = init_model_parameters(self.rng, num_actions=self.num_actions, num_resources=self.num_resources)
        env = DecisionProcessEnv(EnvParams(num_actions=self.num_actions, num_resources=self.num_resources, history_len=self.hist_len))
        obs, _, _ = env.reset(self.rng)

        decision_d, _ = forward_decision_transformer(params, obs, rng_key=self.rng, is_training=True)

        self.assertIsNotNone(decision_d.token_validity_logits)
        self.assertEqual(decision_d.token_validity_logits.shape, (self.hist_len,))

    def test_jit_compilation(self):
        """Test Issue #4 Fix: Verify JAX JIT compilation across model and environment."""
        params = init_model_parameters(self.rng, num_actions=self.num_actions, num_resources=self.num_resources)
        env = DecisionProcessEnv(EnvParams(num_actions=self.num_actions, num_resources=self.num_resources, history_len=self.hist_len))
        obs, env_state, actions_data = env.reset(self.rng)

        # JIT compile forward pass
        jit_forward = jax.jit(forward_decision_transformer)
        decision_d, _ = jit_forward(params, obs)
        self.assertEqual(decision_d.action_logits.shape, (self.num_actions,))

        # JIT compile beam search step
        beam_state = beam_search_init(obs.state, obs, beam_width=3, num_costs=self.num_costs)
        jit_beam_step = jax.jit(lambda p, bs, ad, t: beam_search_step(p, bs, ad, t, beam_width=3, num_actions=self.num_actions))
        new_beam_state = jit_beam_step(params, beam_state, actions_data, obs.target)
        self.assertEqual(new_beam_state.step_count, 1)

    def test_hierarchical_macro_micro_engine(self):
        """Test Directive #3: Hierarchical Execution Engine."""
        config = HierarchicalConfig(macro_steps=5, micro_steps_per_macro=10, total_micro_steps=50)
        engine = HierarchicalExecutionEngine(config)
        h_params = init_hierarchical_model_parameters(
            self.rng,
            num_actions=2000,
            num_macro_clusters=50,
            num_fine_actions=40,
            num_resources=8,
        )

        final_env_state, total_reward, total_micro_steps = engine.run_macro_episode(h_params, self.rng)
        
        self.assertEqual(int(total_micro_steps), 50)
        self.assertTrue(jnp.isfinite(total_reward))


if __name__ == "__main__":
    unittest.main()
