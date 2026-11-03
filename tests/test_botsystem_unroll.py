"""The large-matrix unroll (GoalWM.unroll: one chunked encoder pass over all B x T observations, a scan over the recurrence only)
gives the values and gradients of T successive step() calls, and the fused PPO + self-imitation pass equals the two separate ones.
Side track (forge1201-MODED-AIPlayer, owner 2026-11-03): nothing here counts toward a roadmap gate."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


class TestUnroll(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import botsystem_goal_ppo as gp
        import test_botsystem_goal as tg
        cls.jax, cls.gp, cls.tg = jax, gp, tg
        cls.sizes = tg.sizes_from_hello(tg.HELLO)
        cls.model = tg.GoalWM(cls.sizes)
        cls.params = cls.model.init(jax.random.PRNGKey(1))
        rng = np.random.default_rng(0)
        cls.gs = gp.GoalSpace(cls.sizes, tg.MILESTONES)
        cls.B, cls.T = 3, 5
        rows = []
        for _ in range(cls.B * cls.T):
            o = tg.make_obs(rng, {2: 1}, n_valid=6)
            rows.append({**gp.to_arrays(o, cls.sizes), **cls.gs.encode({"tokens": [(("item", 2), 1.0, 1.0)], "t0": 0, "task": False}, cls.gs.values(o), 0, 10, 2)})
        cls.rows = rows
        cls.obs = {k: np.stack([r[k] for r in rows]).reshape((cls.B, cls.T) + np.shape(rows[0][k])) for k in rows[0]}
        cls.h0 = rng.normal(0, 0.3, (cls.B, cls.model.width)).astype(np.float32)

    def reference(self, P):
        jnp = self.jax.numpy

        def body(h, o):
            logits, value, h2, _ = self.model.step(P, h, o)
            return h2, (logits, value)
        _, (lg, v) = self.jax.lax.scan(body, self.h0, {k: jnp.swapaxes(v, 0, 1) for k, v in self.obs.items()})
        return jnp.swapaxes(lg, 0, 1), jnp.swapaxes(v, 0, 1)

    def test_values_match_stepwise(self):
        ref_l, ref_v = self.reference(self.params)
        for chunk in (4, 7, 512):                       # several chunks with padding, and one chunk bigger than the batch
            lg, v = self.model.unroll(self.params, self.h0, self.obs, chunk)
            np.testing.assert_allclose(np.asarray(lg), np.asarray(ref_l), rtol=2e-4, atol=2e-4)
            np.testing.assert_allclose(np.asarray(v), np.asarray(ref_v), rtol=2e-4, atol=2e-4)

    def test_gradients_match_stepwise(self):
        jax, jnp = self.jax, self.jax.numpy
        mask = jnp.asarray(self.obs["cand_mask"])
        f_ref = lambda P: (jnp.where(mask, self.reference(P)[0], 0.0).sum() + self.reference(P)[1].sum())
        f_new = lambda P: (jnp.where(mask, self.model.unroll(P, self.h0, self.obs, 4)[0], 0.0).sum() + self.model.unroll(P, self.h0, self.obs, 4)[1].sum())
        g1, g2 = jax.grad(f_ref)(self.params), jax.grad(f_new)(self.params)
        for a, b in zip(jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g2)):
            scale = max(float(jnp.abs(a).max()), 1e-3)
            self.assertLess(float(jnp.abs(a - b).max()) / scale, 2e-3)

    def test_fused_ppo_and_sil_equal_separate(self):
        jnp = self.jax.numpy
        gp = self.gp
        a = gp.parse_args(["--root", "/nonexistent", "--base", "x", "--enc-chunk", "5", "--seg", "5"])
        lr = gp.GoalLearner(a, self.model, self.params)
        rng = np.random.default_rng(3)
        zero = np.zeros((self.B, self.T), np.float32)
        b = {"obs": self.obs, "h0": self.h0, "act": np.zeros((self.B, self.T), np.int32), "logp": zero - 2.0, "adv": rng.normal(0, 1, zero.shape).astype(np.float32),
             "ret": rng.normal(0, 1, zero.shape).astype(np.float32), "valid": np.ones_like(zero), "tch": np.full(zero.shape, -1, np.int32), "src": zero}
        sil = [{"h0": self.h0[i], "obs": [self.rows[i * self.T + t] for t in range(self.T)], "act": [0] * self.T, "ret": list(rng.normal(0, 1, self.T))} for i in range(self.B)]
        sb = lr.sil_batch(sil, 1.0, self.rows[0], rng)
        loss, m = lr._loss(self.params, b, sb, 0.01, 0.0)
        sep, sm = lr._sil(self.params, sb)
        for k in ("sil_pol", "sil_val", "sil_gap"):
            self.assertAlmostEqual(float(m[k]), float(sm[k]), places=4, msg=k)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertLessEqual(float(m["expl_var"]), 1.0)

    def test_skipping_the_empty_plan_batch_changes_nothing(self):
        """In list mode the plan batch is the all-masked dummy: its loss is exactly 0, so leaving it out (use_plan=False) gives the
        same auxiliary loss and the same gradients, minus one encoder pass."""
        jax, gp = self.jax, self.gp
        a = gp.parse_args(["--root", "/nonexistent", "--base", "x"])
        lr = gp.GoalLearner(a, self.model, self.params)
        x = {k: v for k, v in self.rows[0].items()}
        ao, al = gp.dummy_aux(x, 4)
        po, pl = gp.dummy_plan(x, 3, self.model.L)
        on, off = lr._aux(self.params, ao, al, po, pl, True), lr._aux(self.params, ao, al, po, pl, False)
        self.assertAlmostEqual(float(on[0]), float(off[0]), places=5)
        g1 = jax.grad(lambda P: lr._aux(P, ao, al, po, pl, True)[0])(self.params)
        g2 = jax.grad(lambda P: lr._aux(P, ao, al, po, pl, False)[0])(self.params)
        for u, v in zip(jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g2)):
            self.assertLess(float(abs(u - v).max()), 1e-5)

    def test_scale_flag_only_moves_defaults(self):
        gp = self.gp
        a = gp.parse_args(["--root", "/x", "--base", "x", "--scale", "8", "--aux-batch", "100"])
        self.assertEqual((a.batch_segs, a.aux_batch, a.sil_batch, a.plan_batch), (128, 100, 32, 256))


if __name__ == "__main__":
    unittest.main()
