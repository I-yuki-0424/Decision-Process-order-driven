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

    def test_recurrent_world_model_update_rules(self):
        cfg = eh.PPOConfig(total_steps=2 * 4 * 4, num_envs=4, num_steps=4, minibatches=2)
        for mode, wm_changes in (("trained", True), ("random", False)):
            arm = TFGRUArm(d=32, layers=1, width=32, wm_mode=mode, wm_hidden=32)
            tr = eh.Trainer(arm, cfg)
            st0 = tr.init(jax.random.PRNGKey(3))
            st1, stats = tr.update(st0)
            d = lambda a, b: max(float(jnp.abs(x - y).max()) for x, y in
                                 zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))
            self.assertGreater(d(st0["params"]["pi"], st1["params"]["pi"]), 0.0)
            self.assertEqual(d(st0["params"]["wm"], st1["params"]["wm"]) > 0.0, wm_changes, mode)
            self.assertEqual(float(stats[3]) > 0.0, wm_changes, mode)   # logged WM loss

    def test_cnn_gru_with_resets(self):
        self._check(CNNGRUArm(32, 8, 8))

    def test_transformer_gru_with_resets(self):
        self._check(TFGRUArm(d=32, layers=1, width=32))


@unittest.skipUnless(HAVE, "craftax/jax not importable")
class TestRecipeAndVariants(unittest.TestCase):
    """TASK-20261004-023: recipe options of the reference baselines and the Truck / GRU-baseline variants."""

    def setUp(self):
        self.obs, _ = eh.env_reset(jax.random.split(jax.random.PRNGKey(0), 3))

    def test_value_norm_stats(self):
        mu, sd = eh._vn_stats(jnp.zeros(3))
        self.assertEqual((float(mu), float(sd)), (0.0, 1.0))          # identity before the first update
        vn = jnp.zeros(3)
        for r in (jnp.array([1.0, 3.0]), jnp.array([1.0, 3.0])):
            vn = eh._vn_update(vn, r, 0.9)
        mu, sd = eh._vn_stats(vn)                                       # debiased EMA of a constant batch = its mean/std
        self.assertAlmostEqual(float(mu), 2.0, places=5)
        self.assertAlmostEqual(float(sd), 1.0, places=4)

    def test_lr_schedules_and_reward_coefficient(self):   # TASK-026
        base = dict(total_steps=40 * 4 * 8, num_envs=4, num_steps=8, minibatches=2, epochs=1, lr=1.0)
        n_opt = 40 * 1 * 2
        for sched in ("linear", "cosine", "knee"):
            eh.Trainer(GRUArm(16), eh.PPOConfig(**base, lr_sched=sched, lr_knee=0.25, lr_knee_frac=0.3))   # builds
        # the knee schedule: 1 -> 0.3 over the first quarter, then linear to 0 (checked on the same interp the trainer builds)
        kn = int(0.25 * n_opt)
        f = lambda c: float(jnp.interp(jnp.float32(c), jnp.array([0.0, kn, n_opt], jnp.float32), jnp.array([1.0, 0.3, 0.0], jnp.float32)))
        self.assertAlmostEqual(f(0), 1.0)
        self.assertAlmostEqual(f(kn), 0.3, places=5)
        self.assertAlmostEqual(f(n_opt), 0.0, places=5)
        with self.assertRaises(ValueError):
            eh.Trainer(GRUArm(16), eh.PPOConfig(**base, lr_sched="exp"))
        # defaults leave the training reward unchanged; rew_small scales only |r| < 0.5 terms and is a training-only change
        cfg0 = eh.PPOConfig(total_steps=2 * 4 * 8, num_envs=4, num_steps=8, minibatches=2)
        cfg1 = cfg0._replace(rew_small=0.0)
        out = []
        for cfg in (cfg0, cfg1):
            tr = eh.Trainer(GRUArm(16), cfg)
            st = tr.init(jax.random.PRNGKey(0))
            _, _, _, _, traj, _ = tr._rollout(st["params"], st, jax.random.PRNGKey(5))
            out.append(np.asarray(traj["rew"]))
        big = np.abs(out[0]) >= 0.5
        np.testing.assert_array_equal(out[0][big], out[1][big])
        self.assertTrue((out[1][~big] == 0.0).all())

    def test_latent_prediction_auxiliary(self):   # TASK-027
        from src.model.epa_policies import init_lp_head, lp_loss
        T, N, F = 8, 3, 16
        feat = jax.random.normal(jax.random.PRNGKey(0), (T, N, F))
        act = jax.random.randint(jax.random.PRNGKey(1), (T, N), 0, N_ACT)
        done = jnp.zeros((T, N), bool)
        head = init_lp_head(jax.random.PRNGKey(2), F, 8)
        l0 = lp_loss(head, feat, act, done, 3)
        self.assertTrue(0.0 <= float(l0) <= 2.0)
        g = jax.grad(lambda h, f: lp_loss(h, f, act, done, 3), argnums=(0, 1))(head, feat)
        self.assertGreater(max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(g[0])), 0.0)   # the predictor receives gradient
        done_all = jnp.ones((T, N), bool)
        self.assertEqual(float(lp_loss(head, feat, act, done_all, 3)), 0.0)   # every path crosses an episode end: no pairs
        cfg = eh.PPOConfig(total_steps=2 * 4 * 8, num_envs=4, num_steps=8, minibatches=2)
        tr = eh.Trainer(GRUArm(32, ln=True, skip=True, tp_coef=1.0, lp_coef=0.5, lp_k=3), cfg)
        st0 = tr.init(jax.random.PRNGKey(3))
        st1, stats = tr.update(st0)
        self.assertTrue(np.isfinite(np.asarray(stats)).all())
        self.assertIn("lp", st1["params"]["pi"])
        base = GRUArm(32, ln=True, skip=True, tp_coef=1.0)   # lp off: identical parameters to the TASK-025 arm
        a, b = base.init(jax.random.PRNGKey(4)), GRUArm(32, ln=True, skip=True, tp_coef=1.0, lp_coef=0.5).init(jax.random.PRNGKey(4))
        b_without_lp = {"pi": {k: v for k, v in b["pi"].items() if k != "lp"}}
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b_without_lp)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    def test_recipe_options_train(self):
        for adv in ("minibatch", "batch"):
            cfg = eh.PPOConfig(total_steps=2 * 4 * 8, num_envs=4, num_steps=8, minibatches=2, value_norm=0.95, adv_norm=adv,
                               max_grad_norm=0.5, vf=1.0, gamma=0.925, lam=0.625)
            tr = eh.Trainer(GRUArm(32, ln=True, skip=True), cfg)
            st0 = tr.init(jax.random.PRNGKey(3))
            st1, stats = tr.update(st0)
            st2, _ = tr.update(st1)
            self.assertTrue(np.isfinite(np.asarray(stats)).all())
            self.assertGreater(float(st2["vn"][2]), float(st1["vn"][2]))   # target statistics are being tracked
            d = max(float(jnp.abs(x - y).max()) for x, y in zip(jax.tree_util.tree_leaves(st0["params"]), jax.tree_util.tree_leaves(st2["params"])))
            self.assertGreater(d, 0.0)
        with self.assertRaises(ValueError):
            eh.Trainer(GRUArm(32), eh.PPOConfig(total_steps=64, num_envs=4, num_steps=8, minibatches=2, adv_norm="episode"))

    def test_variant_shapes_names_counts(self):
        cases = {GRUArm(32, ln=True, skip=True): "gru32+ln+skip", CNNGRUArm(32, 8, 8, ln=True, skip=True): "cnngru8-8w32+ln+skip",
                 TFGRUArm(d=32, layers=1, width=32, mem_token=True): "tfgru32x1w32+mem",
                 TFGRUArm(d=32, layers=1, width=32, prev_act=True): "tfgru32x1w32+pa",
                 TFGRUArm(d=32, layers=1, width=32, head_skip=True): "tfgru32x1w32+hs",
                 TFGRUArm(d=32, layers=1, width=32, mem_token=True, prev_act=True, head_skip=True): "tfgru32x1w32+mem+pa+hs",
                 TFGRUArm(d=32, layers=1, width=32, head_skip=True, cand=False): "tfgru32x1w32+hs-nocand"}
        for arm, name in cases.items():
            self.assertEqual(arm.name, name)
            p = arm.init(jax.random.PRNGKey(1))
            logits, value, caux = arm.step(p, arm.init_carry(3), self.obs)
            self.assertEqual((logits.shape, value.shape), ((3, N_ACT), (3,)))
            self.assertTrue(np.isfinite(np.asarray(logits)).all())
            total, dep = arm.param_counts(p)
            self.assertTrue(0 < dep < total)                            # the value head is never deployed
        p = TFGRUArm(d=32, layers=1, width=32, cand=False).init(jax.random.PRNGKey(1))
        self.assertNotIn("cand", p["pi"])                               # no unused parameters are counted
        self.assertNotIn("logit", p["pi"])

    def test_default_flags_are_the_original_arms(self):
        for a, b in ((GRUArm(32), GRUArm(32, ln=False, skip=False)), (TFGRUArm(d=32, layers=1, width=32),
                     TFGRUArm(d=32, layers=1, width=32, mem_token=False, prev_act=False, head_skip=False, cand=True))):
            pa, pb = a.init(jax.random.PRNGKey(7)), b.init(jax.random.PRNGKey(7))
            self.assertEqual(jax.tree_util.tree_structure(pa), jax.tree_util.tree_structure(pb))
            self.assertEqual(sorted(pa["pi"]), sorted(pb["pi"]))
            self.assertEqual(a.name, b.name)

    def test_memory_token_feeds_candidate_tokens(self):
        """With mem_token the candidate-token logits depend on the memory; without it only the Dense(h) bias does."""
        for mem, expect in ((True, True), (False, False)):
            arm = TFGRUArm(d=32, layers=1, width=32, mem_token=mem)
            p = arm.init(jax.random.PRNGKey(2))
            p["pi"]["h_logit"] = jax.tree_util.tree_map(jnp.zeros_like, p["pi"]["h_logit"])   # logits = candidate tokens only
            h0 = jax.random.normal(jax.random.PRNGKey(3), (3, 32))
            g = jax.grad(lambda h: arm.step(p, h, self.obs)[0].sum())(h0)
            self.assertEqual(float(jnp.abs(g).max()) > 0.0, expect, mem)

    def test_previous_action_carry(self):
        arm = TFGRUArm(d=32, layers=1, width=32, prev_act=True)
        c = arm.init_carry(2)
        self.assertEqual(c.shape, (2, 32 + N_ACT + 1))
        self.assertEqual(c[:, 32 + N_ACT].tolist(), [1.0, 1.0])        # 'none' at episode start
        c2 = arm.advance(jnp.ones((2, 32)), self.obs[:2], jnp.array([5, 7]), jnp.array([False, True]))
        self.assertEqual(int(jnp.argmax(c2[0, 32:])), 5)
        self.assertEqual(int(jnp.argmax(c2[1, 32:])), N_ACT)            # episode ended: memory cleared, previous action 'none'
        self.assertEqual(float(jnp.abs(c2[1, :32]).max()), 0.0)

    def test_recompute_variants_with_resets(self):
        for arm in (GRUArm(32, ln=True, skip=True), CNNGRUArm(32, 8, 8, ln=True, skip=True),
                    TFGRUArm(d=32, layers=1, width=32, mem_token=True, prev_act=True, head_skip=True),
                    TFGRUArm(d=32, layers=1, width=32, head_skip=True, cand=False)):
            TestRecompute()._check(arm)


@unittest.skipUnless(HAVE, "craftax/jax not importable")
class TestDiagnosticsAndProvenance(unittest.TestCase):
    def test_diagnostic_replays_the_headline_evaluator(self):
        from src.pipeline.epa_diagnostics import CAUSES, diagnose_first_episodes, summarize
        arm = TFGRUArm(d=32, layers=1, width=32, prev_act=True)
        p = arm.init(jax.random.PRNGKey(0))
        key = jax.random.PRNGKey(11)
        ach, length, censored = eh.evaluate_first_episodes(arm, p, key, n_envs=8)
        diag = diagnose_first_episodes(arm, p, key, n_envs=8)
        np.testing.assert_array_equal(diag["length"], length)          # same trajectories, step for step
        np.testing.assert_array_equal(diag["ach_masked"], ach)          # same headline achievements
        self.assertTrue((diag["ach_full"] >= diag["ach_masked"]).all())  # + what was unlocked on the death tick
        s, causes = summarize(diag)
        self.assertEqual(sum(s["cause_counts"].values()), 8)
        self.assertTrue(set(causes) <= set(CAUSES))
        self.assertEqual(s["censored"], censored)

    def test_provenance_fields(self):
        prov = eh.run_provenance()
        for k in ("git_commit", "git_dirty", "evaluator_commit", "code_sha256", "packages", "backend", "xla_flags"):
            self.assertIn(k, prov)
        self.assertEqual(len(prov["code_sha256"]), 64)
        rec = eh.make_record("a", "a", {}, {}, [dict(seed=1000, reward_pct=1.0, score_pct=1.0)], "tune", "x", 1, 2, 1, "b",
                             prov["git_commit"], provenance=prov)
        self.assertEqual(rec["evaluator_commit"], prov["evaluator_commit"])
        self.assertNotIn("provenance", eh.make_record("a", "a", {}, {}, [dict(seed=1000, reward_pct=1.0, score_pct=1.0)], "tune",
                                                      "x", 1, 2, 1, "b", "c"))   # old callers (run_epa_chunkppo) unchanged


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
