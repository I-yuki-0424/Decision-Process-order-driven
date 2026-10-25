"""Tests for the replay learner of TASK-20261025-028 (needs craftax).

  python -m unittest tests.test_epa_replay
"""
import unittest

import numpy as np

try:
    import jax
    from src.model.epa_policies import GRUArm, MLPArm
    from src.pipeline import epa_harness as eh
    from src.pipeline.epa_replay import ReplayTrainer
    HAVE = True
except ImportError:   # pragma: no cover - host without craftax
    HAVE = False

CFG = dict(total_steps=3 * 8 * 8, num_envs=8, num_steps=8, minibatches=2, epochs=2, value_norm=0.99, adv_norm="batch", warmup=0.05)


@unittest.skipUnless(HAVE, "craftax/jax not importable")
class TestReplay(unittest.TestCase):
    def test_k1_equals_plain_ppo(self):
        """K = 1: same rollout, ratio 1, V-trace GAE = GAE, same minibatches -> same parameters after two updates."""
        arm = GRUArm(16, ln=True, skip=True, tp_coef=1.0)
        cfg = eh.PPOConfig(**CFG)
        base, rep = eh.Trainer(arm, cfg), ReplayTrainer(arm, cfg, replay_k=1)
        sb, sr = base.init(jax.random.PRNGKey(3)), rep.init(jax.random.PRNGKey(3))
        for _ in range(2):
            sb, _ = base.update(sb)
            sr, _ = rep.update(sr)
        for x, y in zip(jax.tree_util.tree_leaves(sb["params"]), jax.tree_util.tree_leaves(sr["params"])):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=2e-4, atol=2e-5)

    def test_k2_buffer_and_finite_update(self):
        arm = GRUArm(16, ln=True, skip=True, tp_coef=1.0)
        rep = ReplayTrainer(arm, eh.PPOConfig(**CFG), replay_k=2)
        self.assertEqual(rep.cfg.minibatches, 4)   # same envs per minibatch, twice as many minibatches
        st = rep.init(jax.random.PRNGKey(0))
        st, s1 = rep.update(st)
        np.testing.assert_array_equal(np.asarray(st["rb"]["obs"][0]), np.asarray(st["rb"]["obs"][1]))   # first update fills both slots
        first = np.asarray(st["rb"]["obs"][0])
        st, s2 = rep.update(st)
        np.testing.assert_array_equal(np.asarray(st["rb"]["obs"][1]), first)   # the old rollout moved to slot 1
        self.assertFalse(np.array_equal(np.asarray(st["rb"]["obs"][0]), first))
        for leaf in jax.tree_util.tree_leaves(st["params"]):
            self.assertTrue(np.isfinite(np.asarray(leaf)).all())
        self.assertTrue(np.isfinite(np.asarray(s2)).all())

    def test_rejects_flat_arms(self):
        with self.assertRaises(ValueError):
            ReplayTrainer(MLPArm(), eh.PPOConfig(**CFG), replay_k=2)


if __name__ == "__main__":
    unittest.main()
