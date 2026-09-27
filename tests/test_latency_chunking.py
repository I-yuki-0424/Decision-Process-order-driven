"""Tests for the latency / goal-chunking code path (TASK-20260928-012). No training: forward passes, one rollout,
loss and gradient evaluation only; parameters are never updated."""
import importlib.util
import unittest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from src.environment.latency_envs import LatencyEnv, charged_latency, make_intercept, make_pendulum_goal
from src.model import chunk_policy as cp
from src.model import factored_world_model as fwm
from src.pipeline.chunk_ppo import ARMS, ChunkPPO, ChunkPPOConfig


def _tiny(**kw):
    base = dict(env="intercept", arm="augment", delta=2, k=4, hist_len=3, d_model=16, n_layers=1, n_heads=2,
                critic_width=32, num_envs=4, cycles_per_update=4, num_minibatches=2, wm_hidden=32, wm_window=4,
                wm_batch=8, passive_envs=4, passive_ticks_per_env=16, passive_pretrain_steps=1, eval_cycles=4)
    base.update(kw)
    return ChunkPPOConfig(**base)


class TestLatencyEnv(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(0)

    def test_prefix_executes_before_chunk_and_tail_is_committed(self):
        env = LatencyEnv(make_pendulum_goal(), delta=2, k=4, hist_len=3)   # never terminates early
        ws = env.reset(self.key)
        ws = ws._replace(prefix=jnp.array([3, 4], jnp.int32))
        ws2, out = env.cycle(jax.random.PRNGKey(1), ws, jnp.array([1, 2, 1, 2], jnp.int32))
        np.testing.assert_array_equal(out.tick_act, [3, 4, 1, 2])
        np.testing.assert_array_equal(ws2.prefix, [1, 2])
        np.testing.assert_array_equal(ws2.hist_act, [4, 1, 2])   # last 3 executed ticks
        self.assertTrue(bool(ws2.hist_valid.all()))

    def test_oracle_forecast_is_exact_state_at_plan_start(self):
        for core in (make_pendulum_goal(noise=0.5), make_intercept(turn_noise=1.0)):
            env = LatencyEnv(core, delta=3, k=5)
            ws = env.reset(self.key)._replace(prefix=jnp.array([1, 0, 1], jnp.int32))
            kc = jax.random.PRNGKey(7)
            fc = env.oracle_forecast(kc, ws)
            _, out = env.cycle(kc, ws, jnp.zeros((5,), jnp.int32))
            np.testing.assert_allclose(fc, out.tick_obs[3], atol=1e-6)
            # a different key gives a different exogenous future -> the oracle really is perfect foresight
            self.assertFalse(np.allclose(env.oracle_forecast(jax.random.PRNGKey(8), ws), out.tick_obs[3]))

    def test_zero_delta_oracle_is_observation(self):
        env = LatencyEnv(make_pendulum_goal(), delta=0, k=3)
        ws = env.reset(self.key)
        np.testing.assert_allclose(env.oracle_forecast(self.key, ws), env.observe(ws))

    def test_episode_end_freezes_ticks_and_resets(self):
        env = LatencyEnv(make_pendulum_goal(max_ticks=5), delta=1, k=4, hist_len=2)
        ws = env.reset(self.key)
        ws, out1 = env.cycle(jax.random.PRNGKey(1), ws, jnp.ones((4,), jnp.int32))
        self.assertFalse(bool(out1.done))
        ws, out2 = env.cycle(jax.random.PRNGKey(2), ws, jnp.ones((4,), jnp.int32))
        np.testing.assert_array_equal(out2.tick_alive, [True, False, False, False])
        np.testing.assert_array_equal(out2.tick_end, [True, False, False, False])
        self.assertTrue(bool(out2.done))
        self.assertAlmostEqual(float(out2.finished_return), float(out1.tick_reward.sum() + out2.tick_reward.sum()),
                               places=5)
        self.assertEqual(int(ws.tick), 0)
        np.testing.assert_array_equal(ws.prefix, [make_pendulum_goal().ref_action])
        self.assertFalse(bool(ws.hist_valid.any()))

    def test_charged_latency(self):
        self.assertEqual(charged_latency(2, 8, None), 2)
        self.assertEqual(charged_latency(2, 8, 3), 5)


class TestFactoredWorldModel(unittest.TestCase):
    def setUp(self):
        self.D, self.A = 6, 5
        self.p = fwm.init_world_model_parameters(jax.random.PRNGKey(0), self.D, self.A, hidden=16)
        # perturb output layers so the model is non-trivial
        self.p = jax.tree_util.tree_map(lambda x: x + 0.05 * jnp.ones_like(x), self.p)
        self.stats = fwm.identity_stats(self.D)
        self.s = jnp.linspace(-1, 1, self.D)

    def test_untrained_model_is_persistence(self):
        p0 = fwm.init_world_model_parameters(jax.random.PRNGKey(0), self.D, self.A, hidden=16)
        out = fwm.forecast_state(p0, self.stats, self.s, jnp.array([1, 2, 3]), 0, 1.0)
        np.testing.assert_allclose(out, self.s)

    def test_reference_action_has_zero_effect(self):
        ref = 0
        with_e = fwm.forward_world_model_step(self.p, self.stats, self.s, ref, ref, 0.1, use_effect=True)
        without_e = fwm.forward_world_model_step(self.p, self.stats, self.s, ref, ref, 0.1, use_effect=False)
        np.testing.assert_allclose(with_e, without_e, atol=1e-6)
        other = fwm.forward_world_model_step(self.p, self.stats, self.s, 3, ref, 0.1, use_effect=True)
        self.assertFalse(np.allclose(other, without_e))

    def test_passive_forecast_ignores_committed_actions(self):
        a = fwm.forecast_state(self.p, self.stats, self.s, jnp.array([1, 2]), 0, 0.1, use_effect=False)
        b = fwm.forecast_state(self.p, self.stats, self.s, jnp.array([3, 4]), 0, 0.1, use_effect=False)
        np.testing.assert_allclose(a, b)

    def test_loss_grads_and_frozen_f(self):
        B, L = 3, 4
        ow = jax.random.normal(jax.random.PRNGKey(1), (B, L + 1, self.D))
        aw = jax.random.randint(jax.random.PRNGKey(2), (B, L), 0, self.A)
        v = jnp.array([True, True, False])
        g = jax.grad(fwm.world_model_loss)(self.p, self.stats, ow, aw, v, 0, 0.1)
        self.assertTrue(all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(g)))
        g_frozen = jax.grad(fwm.world_model_loss)(self.p, self.stats, ow, aw, v, 0, 0.1, True, False)
        self.assertEqual(max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(g_frozen["f"])), 0.0)
        self.assertEqual(fwm.world_model_skill(self.p, self.stats, ow, aw, v, 0, 0.1).shape, (L,))

    def test_windows_do_not_cross_episode_end(self):
        N, T, D = 1, 6, 2
        obs = jnp.arange(N * T * D, dtype=jnp.float32).reshape(N, T, D)
        alive = jnp.ones((N, T), jnp.bool_)
        end = jnp.zeros((N, T), jnp.bool_).at[0, 2].set(True)
        ow, aw, v = fwm.extract_windows(jax.random.PRNGKey(0), obs, obs + 1, jnp.zeros((N, T), jnp.int32), alive,
                                        end, 3, 64)
        starts = np.asarray(ow[:, 0, 0]) / D
        ok = np.asarray(v)
        # windows [0,3) end at tick 2 on their last tick (valid); [1,4), [2,5) contain the end before their last tick
        for s, o in zip(starts, ok):
            self.assertEqual(bool(o), int(s) in (0, 3))


class TestChunkPolicy(unittest.TestCase):
    def setUp(self):
        self.cfg = cp.ChunkPolicyConfig(obs_dim=5, goal_dim=2, num_actions=4, k=4, dt=0.1, d_model=16, n_layers=2,
                                        n_heads=2)
        self.p = cp.init_chunk_policy_parameters(jax.random.PRNGKey(0), self.cfg)
        self.p["head"]["w"] = self.p["head"]["w"] * 100.0   # sharpen so position dependence is visible
        self.ai = cp.ActorInput(jnp.ones((5,)), jnp.array([1.0, 0.0]), jnp.ones((3, 5)), jnp.array([0, 1, 2]),
                                jnp.array([False, True, True]), jnp.array([1, 3]), jnp.asarray(0.2), jnp.asarray(0.2))
        self.fn = lambda p, ai, a: cp.forward_chunk_policy(p, self.cfg, ai, a)

    def test_causal_plan_positions(self):
        a1 = jnp.array([0, 1, 2, 3])
        a2 = a1.at[2].set(0)
        l1, l2 = self.fn(self.p, self.ai, a1), self.fn(self.p, self.ai, a2)
        np.testing.assert_allclose(l1[:3], l2[:3], atol=1e-5)   # position j depends only on a_<j
        self.assertFalse(np.allclose(l1[3], l2[3]))

    def test_sampled_logp_matches_teacher_forced(self):
        acts, logp = cp.sample_chunk(self.fn, self.p, self.ai, jax.random.PRNGKey(3), self.cfg.k)
        lp, ent = cp.chunk_log_probs(self.fn, self.p, self.ai, acts)
        np.testing.assert_allclose(logp, lp, atol=1e-5)
        self.assertEqual(ent.shape, (self.cfg.k,))

    def test_invalid_history_is_masked(self):
        other = self.ai._replace(hist_obs=self.ai.hist_obs.at[0].set(99.0), hist_act=self.ai.hist_act.at[0].set(3))
        a = jnp.array([0, 1, 2, 3])
        np.testing.assert_allclose(self.fn(self.p, self.ai, a), self.fn(self.p, other, a), atol=1e-5)

    def test_distance_head_ignores_plan_and_uses_goal(self):
        d1 = cp.forward_temporal_distance(self.p, self.cfg, self.ai)
        d2 = cp.forward_temporal_distance(self.p, self.cfg, self.ai._replace(goal=jnp.array([0.0, 1.0])))
        self.assertTrue(bool(jnp.isfinite(d1)))
        self.assertNotAlmostEqual(float(d1), float(d2), places=6)


class TestChunkPPOWiring(unittest.TestCase):
    def test_arms_differ_only_in_actor_state(self):
        runners = {arm: ChunkPPO(_tiny(arm=arm)) for arm in ARMS}
        r0 = runners["augment"]
        ws = r0.env.reset(jax.random.PRNGKey(0))._replace(prefix=jnp.array([1, 3], jnp.int32))
        obs = r0.env.observe(ws)
        sentinel = jnp.full_like(obs, 123.0)
        for arm, r in runners.items():
            ai = r.actor_input(ws, obs, sentinel)
            uses_forecast = arm in ("wm", "wm_passive", "oracle")
            np.testing.assert_allclose(ai.state, sentinel if uses_forecast else obs)
            self.assertEqual(ai.prefix.shape[0], 2 if arm == "augment" else 0)
            np.testing.assert_allclose(r.critic_input(ws, obs), r0.critic_input(ws, obs))

    def test_only_oracle_forecast_depends_on_env_key(self):
        for arm in ("wm", "wm_passive", "oracle"):
            r = ChunkPPO(_tiny(env="pendulum_goal", arm=arm))
            st = r.init(jax.random.PRNGKey(0))
            st["wm"] = jax.tree_util.tree_map(lambda x: x + 0.05, st["wm"])
            ws = r.env.reset(jax.random.PRNGKey(1))._replace(prefix=jnp.array([0, 4], jnp.int32))
            obs = r.env.observe(ws)
            f1 = r._forecast(st["wm"], st["stats"], jax.random.PRNGKey(2), ws, obs)
            f2 = r._forecast(st["wm"], st["stats"], jax.random.PRNGKey(3), ws, obs)
            self.assertEqual(not np.allclose(f1, f2), arm == "oracle", arm)

    def test_zero_delta_all_arms_see_the_observation(self):
        for arm in ARMS:
            r = ChunkPPO(_tiny(arm=arm, delta=0))
            st = r.init(jax.random.PRNGKey(0))
            ws = r.env.reset(jax.random.PRNGKey(1))
            obs = r.env.observe(ws)
            ai = r.actor_input(ws, obs, r._forecast(st["wm"], st["stats"], jax.random.PRNGKey(2), ws, obs))
            np.testing.assert_allclose(ai.state, obs, atol=1e-6)

    def test_gae_matches_manual(self):
        r = ChunkPPO(_tiny(k=2, delta=1, gamma=0.9, gae_lambda=0.5))
        C, N, k = 3, 1, 2
        rew = jnp.array([[[1.0, 0.0]], [[0.0, 2.0]], [[1.0, 1.0]]])
        done = jnp.array([[False], [True], [False]])
        val = jnp.array([[0.5], [0.2], [0.1]])

        class O:  # minimal stand-ins for the fields gae reads
            pass
        traj = O(); traj.out = O(); traj.out.tick_reward = rew; traj.out.done = done; traj.value = val
        adv, _ = r.gae(traj, jnp.array([0.3]))
        g, gk, lam = 0.9, 0.81, 0.5
        R = [1.0, 0.0 + g * 2.0, 1.0 + g * 1.0]
        td2 = R[2] + gk * 0.3 - 0.1
        td1 = R[1] - 0.2
        td0 = R[0] + gk * 0.2 - 0.5
        a2, a1 = td2, td1
        a0 = td0 + gk * lam * a1
        np.testing.assert_allclose(np.asarray(adv)[:, 0], [a0, a1, a2], rtol=1e-5)

    def test_hindsight_pairs_respect_episode_boundaries(self):
        r = ChunkPPO(_tiny(env="pendulum_goal", arm="augment", k=2, delta=1, dist_max_gap=2))

        class O:
            pass
        C, N = 5, 1
        traj = O(); traj.out = O()
        traj.achieved = jnp.arange(C, dtype=jnp.float32)[:, None, None] * jnp.ones((C, N, 2))
        traj.out.done = jnp.array([False, True, False, False, False])[:, None]
        for seed in range(5):
            goal, tgt, valid = r.hindsight(jax.random.PRNGKey(seed), traj)
            for c in range(C):
                j = int(round(float(np.expm1(tgt[c, 0])))) // 2
                crosses = c <= 1 < c + j
                self.assertEqual(bool(valid[c, 0]), (c + j < C) and not crosses, (seed, c, j))
                if bool(valid[c, 0]):
                    self.assertEqual(float(goal[c, 0, 0]), float(c + j))

    def test_dry_run_every_arm(self):
        for env in ("intercept", "pendulum_goal"):
            for arm in ARMS:
                info = ChunkPPO(_tiny(env=env, arm=arm)).dry_run()
                for key in ("loss", "grad_norm", "pg", "vl", "entropy"):
                    self.assertTrue(np.isfinite(info[key]), (env, arm, key))
                if arm == "oracle":
                    self.assertAlmostEqual(info["forecast_skill"], 1.0, places=5)
                if arm in ("ignore", "augment"):
                    self.assertAlmostEqual(info["forecast_skill"], 0.0, places=6)
                if arm in ("wm", "wm_passive"):
                    self.assertTrue(np.isfinite(info["wm_loss"]))

    def test_history_times_are_relative_to_snapshot(self):
        r_wm, r_aug = ChunkPPO(_tiny(arm="wm")), ChunkPPO(_tiny(arm="augment"))
        ws = r_wm.env.reset(jax.random.PRNGKey(0))
        obs = r_wm.env.observe(ws)
        a_wm, a_aug = r_wm.actor_input(ws, obs, obs), r_aug.actor_input(ws, obs, obs)
        dt = r_wm.core.dt
        self.assertAlmostEqual(float(a_wm.state_age), 0.0)          # forecast is AT plan start
        self.assertAlmostEqual(float(a_wm.obs_age), 2 * dt)         # ...but the history is delta ticks older
        self.assertAlmostEqual(float(a_aug.state_age), 2 * dt)

    def test_update_and_wm_paths_trace(self):
        """Abstract evaluation (jax.eval_shape): traces the PPO update, WM training and passive collection without
        executing them, so no parameter is ever updated."""
        r = ChunkPPO(_tiny(env="pendulum_goal", arm="wm"))
        st = r.init(jax.random.PRNGKey(0))
        _, _, traj, last_v = r.collect(st["pp"], st["wm"], st["stats"], r.reset_envs(jax.random.PRNGKey(1)),
                                       jax.random.PRNGKey(2), n_cycles=r.cfg.cycles_per_update)
        pp, _, aux = jax.eval_shape(r._update, st["pp"], st["opt"], traj, last_v, jax.random.PRNGKey(3))
        self.assertEqual(jax.tree_util.tree_structure(pp), jax.tree_util.tree_structure(st["pp"]))
        self.assertEqual(aux.shape, (6,))
        streams = r.streams(traj)
        wm, _, loss = jax.eval_shape(partial(r._wm_train, n_steps=2, train_f=True), st["wm"], st["wm_opt"],
                                     st["stats"], jax.random.PRNGKey(4), *streams)
        self.assertEqual(jax.tree_util.tree_structure(wm), jax.tree_util.tree_structure(st["wm"]))
        passive = jax.eval_shape(r._collect_passive, jax.random.PRNGKey(5))
        self.assertEqual(passive[0].shape, (4, 16, 3))
        ev = r.evaluate(st, jax.random.PRNGKey(6), greedy=True)      # forward-only rollout
        self.assertIn("mean_return", ev)

    def test_checkpoint_roundtrip_preserves_structure(self):
        import tempfile
        from src.model.checkpoint import load_pytree_checkpoint, save_pytree_checkpoint
        r = ChunkPPO(_tiny(arm="wm"))
        st = r.init(jax.random.PRNGKey(0))
        with tempfile.TemporaryDirectory() as d:
            path = save_pytree_checkpoint({"st": st, "key": jax.random.PRNGKey(1)}, f"{d}/step_1.pkl", 1)
            back = load_pytree_checkpoint(path)["params"]
        self.assertEqual(jax.tree_util.tree_structure(back["st"]), jax.tree_util.tree_structure(st))

    def test_config_guards(self):
        with self.assertRaises(ValueError):
            ChunkPPO(_tiny(actor="mlp", k=4))              # chunks need the autoregressive actor
        with self.assertRaises(ValueError):
            ChunkPPO(_tiny(arm="wm_passive", passive_pretrain_steps=0))
        with self.assertRaises(ValueError):
            ChunkPPO(_tiny(delta=5, k=4))
        r = ChunkPPO(_tiny(arm="wm", total_env_ticks=10_000))
        self.assertEqual(r.n_updates, (10_000 - 4 * 16) // (4 * 4 * 4))   # passive ticks charged to the budget


@unittest.skipUnless(importlib.util.find_spec("craftax"), "craftax not installed")
class TestCraftaxCalibrationWiring(unittest.TestCase):
    def test_calibration_config_dry_run(self):
        from src.pipeline.chunk_ppo import craftax_calibration_config
        info = ChunkPPO(craftax_calibration_config(num_envs=2, cycles_per_update=4, num_minibatches=2)).dry_run()
        self.assertTrue(np.isfinite(info["loss"]))
        self.assertEqual(info["chunk_shape"], [4, 2, 1])


if __name__ == "__main__":
    unittest.main()
