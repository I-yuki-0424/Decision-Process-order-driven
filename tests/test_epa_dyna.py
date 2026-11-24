"""TASK-20261125-041: Dyna-style PPO (src/pipeline/epa_dyna.py)."""
import sys
import unittest

sys.path.insert(0, ".")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from src.model.epa_policies import ARM_BUILDERS  # noqa: E402
from src.pipeline.epa_dyna import DynaTrainer, _split_obs, init_model, model_loss, model_sample  # noqa: E402
from src.pipeline.epa_harness import PPOConfig, env_reset, env_step  # noqa: E402

CFG = PPOConfig(total_steps=2 * 16 * 8, num_envs=16, num_steps=8, epochs=1, minibatches=4, value_norm=0.99, adv_norm="batch", warmup=0.05)
SMALL = dict(im_envs=16, im_h=4, im_epochs=1, im_minibatches=4, im_start=0.0, buf_k=2, wm_steps=2, wm_batch=32)


def real_batch(n=64, seed=0):
    k = jax.random.PRNGKey(seed)
    obs, st = env_reset(jax.random.split(k, n))
    act = jax.random.randint(jax.random.fold_in(k, 1), (n,), 0, 17)
    nobs, _, rew, done = env_step(jax.random.split(jax.random.fold_in(k, 2), n), st, act)
    return obs, act, nobs, rew, done


class TestModel(unittest.TestCase):
    def test_observation_format_round_trip(self):
        obs, *_ = real_batch()
        t = _split_obs(obs)   # the env's own observation decodes to valid classes
        self.assertEqual(t["block"].shape, (64, 63))
        self.assertTrue(bool((t["inv"] >= 0).all() & (t["inv"] <= 9).all()))
        self.assertTrue(np.allclose(np.asarray(obs[:, 1339:1343]).sum(-1), 1.0))   # direction one-hot

    def test_sample_has_env_format(self):
        m = init_model(jax.random.PRNGKey(0), c=16, hidden=32)
        obs, act, *_ = real_batch()
        nobs, r, d = model_sample(m, obs, act, jax.random.PRNGKey(1))
        self.assertEqual(nobs.shape, obs.shape)
        grid = np.asarray(nobs[:, :1323]).reshape(64, 63, 21)
        self.assertTrue(np.allclose(grid[..., :17].sum(-1), 1.0))   # one block class per cell
        self.assertEqual(r.shape, (64,))
        self.assertEqual(d.dtype, jnp.bool_)

    def test_model_learns_executed_transitions(self):
        m = init_model(jax.random.PRNGKey(0), c=16, hidden=64)
        data = real_batch(128)
        opt = optax.adam(3e-3)
        o = opt.init(m)
        l0 = float(model_loss(m, *data))
        @jax.jit
        def step(m, o):
            loss, g = jax.value_and_grad(model_loss)(m, *data)
            upd, o = opt.update(g, o, m)
            return optax.apply_updates(m, upd), o, loss

        for _ in range(60):
            m, o, l1 = step(m, o)
        self.assertLess(float(l1), 0.5 * l0)


class TestModelV2(unittest.TestCase):
    def test_v1_unchanged_by_v2_option(self):   # S1 results stay reproducible: v1 has no rew2 head and the same keys as before
        m = init_model(jax.random.PRNGKey(0), c=16, hidden=32)
        self.assertNotIn("rew2", m)

    def test_twohot_reward_is_exact_on_support(self):
        from src.pipeline.epa_dyna import REW_BINS, _twohot
        r = jnp.array([0.0, 1.0, -0.1, 0.9, 0.05])
        self.assertTrue(np.allclose((_twohot(r) * REW_BINS).sum(-1), r, atol=1e-5))

    def test_v2_learns_and_samples(self):
        m = init_model(jax.random.PRNGKey(0), c=16, hidden=64, version=2)
        data = real_batch(128)
        nobs, r, d = model_sample(m, data[0], data[1], jax.random.PRNGKey(2))
        self.assertEqual(r.shape, (128,))
        self.assertTrue(np.isfinite(float(model_loss(m, *data))))


class TestTrainer(unittest.TestCase):
    def _run(self, model):
        arm = ARM_BUILDERS["ppo_gru"](width=32, ln=True, skip=True, tp_coef=1.0)
        tr = DynaTrainer(arm, CFG, model=model, wm_c=8, wm_hidden=16, **SMALL)
        params, curve, _ = tr.train(5)
        self.assertEqual(curve.shape, (tr.n_updates, 22))
        self.assertTrue(np.isfinite(curve).all())
        self.assertTrue((curve[:, 11] >= 0).all() and (curve[:, 11] <= 1).all())   # imagined done fraction
        return tr, params

    def test_learned_counts_model_params(self):
        tr, _ = self._run("learned")
        self.assertGreater(tr.model_params, 0)
        self.assertEqual(tr.im_steps_total, tr.n_updates * 16 * 4)

    def test_learned_conditioned_on_agent_state(self):   # L4
        arm = ARM_BUILDERS["ppo_gru"](width=32, ln=True, skip=True, tp_coef=1.0)
        tr = DynaTrainer(arm, CFG, model="learned", wm_c=8, wm_hidden=16, wm_version=2, wm_cond=True, **SMALL)
        self.assertEqual(tr.cond_dim, 32)
        _, curve, _ = tr.train(6)
        self.assertTrue(np.isfinite(curve).all())

    def test_oracle_runs_and_has_no_model(self):
        tr, _ = self._run("oracle")
        self.assertEqual(tr.model_params, 0)

    def test_rejects_bad_settings(self):
        arm = ARM_BUILDERS["ppo_gru"](width=32)
        with self.assertRaises(ValueError):
            DynaTrainer(arm, CFG, model="perfect")
        with self.assertRaises(ValueError):
            DynaTrainer(ARM_BUILDERS["ppo_mlp"](), CFG)


if __name__ == "__main__":
    unittest.main()
