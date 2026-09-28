"""Regression tests for the Crafter-score auto-reset leak (TASK-20260928-015, item 3).

CraftaxClassicSymbolicEnv.step() auto-resets on `done`. Reading achievements from the state it returns after a death
either yields zeros (final-state readers) or credits the NEXT life to this episode (running-max readers).
"""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.environment.craftax_env_adapter import NUM_ACHIEVEMENTS, calculate_crafter_score, masked_achievements

try:
    from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
except Exception:  # craftax is only installed in the Docker/Kaggle environments
    CraftaxClassicSymbolicEnv = None


class TestMaskedAchievements(unittest.TestCase):
    def test_mask_semantics(self):
        a = jnp.ones((NUM_ACHIEVEMENTS,))
        self.assertEqual(float(masked_achievements(a, jnp.array(False), 1.0).sum()), NUM_ACHIEVEMENTS)
        self.assertEqual(float(masked_achievements(a, jnp.array(True), 1.0).sum()), 0.0)     # terminal / reset state
        self.assertEqual(float(masked_achievements(a, jnp.array(False), 0.0).sum()), 0.0)    # already-dead episode
        self.assertEqual(float(masked_achievements(a, False).sum()), NUM_ACHIEVEMENTS)        # python bool accepted


@unittest.skipIf(CraftaxClassicSymbolicEnv is None, "craftax not installed")
class TestAutoResetLeak(unittest.TestCase):
    def setUp(self):
        self.env = CraftaxClassicSymbolicEnv()
        self.params = self.env.default_params

    def _rollout(self, key, T):
        k0, k1 = jax.random.split(key)
        _, s = self.env.reset(k0, self.params)

        def body(c, t):
            s, alive, ach_masked, ach_leaky = c
            ka, ke = jax.random.split(jax.random.fold_in(k1, t))
            a = jax.random.randint(ka, (), 0, 17)
            _, s2, _, d, _ = self.env.step(ke, s, a, self.params)
            ach = s2.achievements.astype(jnp.float32)
            ach_masked = jnp.maximum(ach_masked, masked_achievements(ach, d, alive))
            ach_leaky = jnp.maximum(ach_leaky, ach)                      # the old, buggy accumulation
            return (s2, alive * (1.0 - d.astype(jnp.float32)), ach_masked, ach_leaky), (d, ach)

        z = jnp.zeros(NUM_ACHIEVEMENTS)
        (_, _, am, al), (ds, achs) = jax.lax.scan(body, (s, jnp.array(1.0), z, z), jnp.arange(T))
        return am, al, ds, achs

    def test_state_returned_on_done_is_a_fresh_episode(self):
        _, _, ds, achs = jax.jit(jax.vmap(lambda k: self._rollout(k, 400)))(jax.random.split(jax.random.PRNGKey(0), 16))
        ds, achs = np.asarray(ds), np.asarray(achs)
        self.assertTrue(ds.any(), "random policy should die within 400 ticks in at least one env")
        self.assertEqual(float(achs[ds].sum()), 0.0, "auto-reset assumption of masked_achievements no longer holds")

    def test_masked_accumulation_never_exceeds_leaky_and_is_lower_after_deaths(self):
        am, al, ds, _ = jax.jit(jax.vmap(lambda k: self._rollout(k, 600)))(jax.random.split(jax.random.PRNGKey(1), 48))
        am, al = np.asarray(am), np.asarray(al)
        self.assertTrue((am <= al).all())
        rates = lambda x: [float(x[:, i].mean() * 100) for i in range(NUM_ACHIEVEMENTS)]
        self.assertLess(calculate_crafter_score(rates(am)), calculate_crafter_score(rates(al)))
        self.assertLess(am.sum(1).mean(), al.sum(1).mean())


if __name__ == "__main__":
    unittest.main()
