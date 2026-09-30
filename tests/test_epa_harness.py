"""Tests for the Phase-1 harness (needs craftax; run inside the dpod-local image).

  python -m unittest tests.test_epa_harness
"""
import unittest

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from src.model.epa_policies import GRUArm, MLPArm, TFArm, TFGRUArm, CNNGRUArm, N_ACT, OBS_DIM
    from src.pipeline import epa_harness as eh
    HAVE = True
except ImportError:   # pragma: no cover - host without craftax
    HAVE = False


@unittest.skipUnless(HAVE, "craftax/jax not importable")
class TestMetrics(unittest.TestCase):
    def test_reward_and_score_definitions(self):
        full = np.ones((8, 22))
        out = eh.summarize_achievements(full)
        self.assertAlmostEqual(out["reward_pct"], 100.0)
        self.assertAlmostEqual(out["score_pct"], 100.0, places=6)
        half = np.zeros((8, 22))
        half[:, :11] = 1.0    # 11 achievements in every episode
        out = eh.summarize_achievements(half)
        self.assertAlmostEqual(out["reward_pct"], 50.0)
        self.assertAlmostEqual(out["score_pct"], float(np.sqrt(101.0) - 1.0), places=6)   # exp(11/22 ln 101) - 1
        mixed = np.zeros((4, 22))
        mixed[:2, 0] = 1.0    # achievement 0 in 50% of episodes only
        out = eh.summarize_achievements(mixed)
        self.assertAlmostEqual(out["reward_pct"], 100.0 / 22 * 0.5)
        self.assertAlmostEqual(out["score_pct"], float(np.exp(np.log(51.0) / 22) - 1.0), places=6)


@unittest.skipUnless(HAVE, "craftax/jax not importable")
class TestArms(unittest.TestCase):
    def setUp(self):
        self.obs, _ = eh.env_reset(jax.random.split(jax.random.PRNGKey(0), 3))

    def test_shapes(self):
        for arm in (MLPArm(64, 2), GRUArm(32), TFGRUArm(d=32, layers=1, width=32), CNNGRUArm(32, 8, 8), TFArm(d=32, layers=1, hist=2), TFArm(d=32, layers=1, hist=2, wm_mode="trained")):
            p = arm.init(jax.random.PRNGKey(1))
            logits, value, _ = arm.step(p, arm.init_carry(3), self.obs)
            self.assertEqual(logits.shape, (3, N_ACT))
            self.assertEqual(value.shape, (3,))
            total, dep = arm.param_counts(p)
            self.assertTrue(0 < dep <= total)

    def test_separate_critic_and_wmq_variants(self):
        for arm in (TFArm(d=32, layers=1, hist=2, critic="mlp"),
                    TFArm(d=32, layers=1, hist=2, wm_mode="trained", wm_hidden=32, critic="mlp", wmq=True),
                    TFArm(d=32, layers=1, hist=2, wm_mode="random", wm_hidden=32, critic="mlp", wmq=True)):
            p = arm.init(jax.random.PRNGKey(1))
            logits, value, _ = arm.step(p, arm.init_carry(3), self.obs)
            self.assertEqual((logits.shape, value.shape), ((3, N_ACT), (3,)))
            total, dep = arm.param_counts(p)
            self.assertLess(dep, total)          # the critic is not deployed
        # world-model features are stop-gradient: the policy-gradient path must not move the WM or (via features) the critic
        arm = TFArm(d=32, layers=1, hist=2, wm_mode="trained", wm_hidden=32, critic="mlp", wmq=True)
        p = arm.init(jax.random.PRNGKey(2))
        g = jax.grad(lambda q: arm.step(q, arm.init_carry(3), self.obs)[0].sum())(p)
        self.assertEqual(max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(g["wm"])), 0.0)
        self.assertEqual(max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(g["pi"]["critic"])), 0.0)

    def test_candidate_token_aux_loss_reaches_trunk(self):
        arm = TFArm(d=32, layers=1, hist=2, aux_coef=0.5)
        cfg = eh.PPOConfig(total_steps=2 * 4 * 4, num_envs=4, num_steps=4, minibatches=2)
        tr = eh.Trainer(arm, cfg)
        st = tr.init(jax.random.PRNGKey(4))
        _, _, _, _, traj, _ = tr._rollout(st["params"], st, jax.random.PRNGKey(5))
        mb = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), dict(traj))

        def wl(p):
            return arm.train_forward(p, mb)[2]

        val, g = jax.value_and_grad(wl)(st["params"])
        self.assertTrue(float(val) > 0.0 and np.isfinite(float(val)))
        # the auxiliary loss trains the shared trunk (it is not a stop-gradient side channel) ...
        self.assertGreater(float(jnp.abs(g["pi"]["blocks"][0]["qkv"]["w"]).max()), 0.0)
        self.assertGreater(float(jnp.abs(g["pi"]["aux_obs"]["w"]).max()), 0.0)
        # ... and only the executed action's token supplies the target (one action per state, no counterfactual labels)
        self.assertEqual(float(jnp.abs(g["pi"]["value"]["w"]).max()), 0.0)

    def test_history_window_and_reset(self):
        arm = TFArm(d=32, layers=1, hist=3)
        c = arm.init_carry(2)
        act = jnp.array([1, 2])
        for t in range(2):
            c = arm.advance(c, self.obs[:2] + t, act, jnp.array([False, False]))
        self.assertEqual(c[2].tolist(), [[False, True, True]] * 2)
        c = arm.advance(c, self.obs[:2], act, jnp.array([True, False]))
        self.assertEqual(c[2].tolist(), [[False, False, False], [True, True, True]])   # episode end clears history

    def test_wm_modes_update_rules(self):
        cfg = eh.PPOConfig(total_steps=2 * 4 * 4, num_envs=4, num_steps=4, minibatches=2)
        for mode, wm_changes in (("trained", True), ("random", False)):
            arm = TFArm(d=32, layers=1, hist=2, wm_mode=mode, wm_hidden=32)
            tr = eh.Trainer(arm, cfg)
            st0 = tr.init(jax.random.PRNGKey(3))
            st1, _ = tr.update(st0)
            d = lambda a, b: max(float(jnp.abs(x - y).max()) for x, y in
                                 zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))
            self.assertGreater(d(st0["params"]["pi"], st1["params"]["pi"]), 0.0)
            self.assertEqual(d(st0["params"]["wm"], st1["params"]["wm"]) > 0.0, wm_changes, mode)


@unittest.skipUnless(HAVE, "craftax/jax not importable")
class TestRecompute(unittest.TestCase):
    """The stored context must reproduce the behaviour policy exactly: logp recomputed in the update == logp at rollout."""

    def _check(self, arm):
        cfg = eh.PPOConfig(total_steps=8 * 12, num_envs=4, num_steps=12, minibatches=2)
        tr = eh.Trainer(arm, cfg)
        st = tr.init(jax.random.PRNGKey(5))
        _, _, _, _, traj, _ = tr._rollout(st["params"], st, jax.random.PRNGKey(6))
        if arm.recurrent:
            mb = dict(traj, carry=traj["carry"][0:1])

            def body(h, x):
                obs, act, done = x
                logits, value, h2 = arm.step(st["params"], h, obs)
                return arm.advance(h2, obs, act, done), logits

            _, logits = jax.lax.scan(body, traj["carry"][0], (traj["obs"], traj["act"], traj["done"]))
        else:
            flat = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), traj)
            logits, _, _ = arm.step(st["params"], flat["carry"], flat["obs"])
            logits = logits.reshape(12, 4, -1)
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits), traj["act"][..., None], -1)[..., 0]
        np.testing.assert_allclose(np.asarray(logp), np.asarray(traj["logp"]), atol=2e-4)
        return traj

    def test_mlp(self):
        self._check(MLPArm(64, 2))

    def test_gru_with_resets(self):
        self._check(GRUArm(32))

    def test_transformer_history(self):
        self._check(TFArm(d=32, layers=1, hist=3))

    def test_cnn_gru_with_resets(self):
        self._check(CNNGRUArm(32, 8, 8))

    def test_transformer_gru_with_resets(self):
        self._check(TFGRUArm(d=32, layers=1, width=32))


@unittest.skipUnless(HAVE, "craftax/jax not importable")
class TestEvaluation(unittest.TestCase):
    def test_untrained_eval_is_bounded_and_first_episode_only(self):
        arm = MLPArm(32, 1)
        p = arm.init(jax.random.PRNGKey(0))
        ach, length, censored = eh.evaluate_first_episodes(arm, p, jax.random.PRNGKey(1), n_envs=16)
        self.assertEqual(ach.shape, (16, 22))
        self.assertTrue(((ach == 0) | (ach == 1)).all())
        self.assertTrue((length > 0).all())
        self.assertEqual(censored, 0)
        # the episode limit is the environment default, not a T cap
        self.assertEqual(eh.EPISODE_LIMIT, 10000)


if __name__ == "__main__":
    unittest.main()
