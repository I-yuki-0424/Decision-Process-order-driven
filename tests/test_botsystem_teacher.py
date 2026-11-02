"""BOTSYSTEM scripted teacher: the recipe recursion on synthetic observations, and the fade-out rule. Side track (nothing counts toward a gate)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import botsystem_teacher as bt  # noqa: E402

ITEMS = ["", "oak_log", "oak_planks", "stick", "crafting_table", "wooden_pickaxe", "cobblestone", "stone", "coal_ore", "dirt"]
IX = {n: i for i, n in enumerate(ITEMS)}


def obs(cands, inv=None, K=16):
    """cands: [(cmd, item, dist, dy)]; inv: {item: count}."""
    o = {"cand_cmd": np.zeros(K, np.int32), "cand_item": np.zeros(K, np.int32), "cand_mask": np.zeros(K, bool),
         "cand_feat": np.zeros((K, 8), np.float32), "inv_item": np.zeros(36, np.int32), "inv_count": np.zeros(36, np.float32)}
    for k, (cmd, item, dist, dy) in enumerate(cands):
        o["cand_cmd"][k], o["cand_item"][k], o["cand_mask"][k] = bt.CMDS.index(cmd), IX.get(item, 0), True
        o["cand_feat"][k][3], o["cand_feat"][k][1] = dist, dy
    for s, (n, c) in enumerate((inv or {}).items()):
        o["inv_item"][s], o["inv_count"][s] = IX[n], c / 64
    return o


class TestTeacher(unittest.TestCase):
    def setUp(self):
        self.t = bt.Teacher(ITEMS)

    def test_wood_mines_the_nearest_log(self):
        o = obs([("wait", "", 0, 0), ("mine", "oak_log", 0.5, 0), ("mine", "oak_log", 0.2, 0), ("mine", "dirt", 0.1, 0)])
        self.assertEqual(self.t.choose(o, "collect_wood"), 2)

    def test_planks_need_a_log_first_then_craft(self):
        self.assertEqual(self.t.choose(obs([("mine", "oak_log", 0.3, 0), ("wait", "", 0, 0)]), "craft_planks"), 0)
        o = obs([("craft", "oak_planks", 0, 0), ("mine", "oak_log", 0.3, 0)], {"oak_log": 1})
        self.assertEqual(self.t.choose(o, "craft_planks"), 0)

    def test_table_chain(self):
        o = obs([("craft", "crafting_table", 0, 0), ("mine", "oak_log", 0.3, 0)], {"oak_planks": 4})
        self.assertEqual(self.t.choose(o, "craft_crafting_table"), 0)
        o = obs([("mine", "oak_log", 0.3, 0)], {"oak_planks": 2})                  # 2 planks: gather a log -> more planks later
        self.assertEqual(self.t.choose(o, "craft_crafting_table"), 0)
        o = obs([("place", "crafting_table", 0, 0)], {"crafting_table": 1})
        self.assertEqual(self.t.choose(o, "place_crafting_table"), 0)

    def test_pickaxe_without_a_table_in_reach_places_one(self):
        inv = {"oak_planks": 3, "stick": 2, "crafting_table": 1}
        o = obs([("place", "crafting_table", 0, 0), ("wait", "", 0, 0)], inv)
        self.assertEqual(self.t.choose(o, "craft_wooden_pickaxe"), 0)
        o = obs([("craft", "wooden_pickaxe", 0, 0), ("place", "crafting_table", 0, 0)], inv)
        self.assertEqual(self.t.choose(o, "craft_wooden_pickaxe"), 0)

    def test_stone_needs_a_pickaxe_then_stone_else_digs_down(self):
        o = obs([("mine", "stone", 0.2, 0), ("wait", "", 0, 0)], {})
        self.assertEqual(self.t.choose(o, "collect_stone"), -1)                   # no pickaxe, no log, no craft: no opinion
        inv = {"wooden_pickaxe": 1}
        self.assertEqual(self.t.choose(obs([("mine", "stone", 0.2, 0)], inv), "collect_stone"), 0)
        o = obs([("tunnel", "dirt", 0.1, 0.0), ("tunnel", "dirt", 0.1, -0.06), ("wait", "", 0, 0)], inv)
        self.assertEqual(self.t.choose(o, "collect_stone"), 1)

    def test_unknown_goal_and_masked_candidates(self):
        self.assertEqual(self.t.choose(obs([("mine", "oak_log", 0.1, 0)]), "collect_diamond"), -1)
        o = obs([("mine", "oak_log", 0.1, 0)])
        o["cand_mask"][0] = False
        self.assertEqual(self.t.choose(o, "collect_wood"), -1)


class TestTeaching(unittest.TestCase):
    def fake(self, **kw):
        import types
        a = dict(teacher_agree=0.55, teacher_fade=0.9, teacher_min=0.02, teacher_min_n=10, pair_prune=0.25, pair_min_eps=2)
        a.update(kw)
        return types.SimpleNamespace(a=types.SimpleNamespace(**a), teach={"level": 1.0, "agree_ema": None, "hit": 0, "n": 0, "ex": 0, "lab": 0},
                                     pair_key=lambda e: ("pair", e["p"]), pair_score={})

    def test_fade_waits_for_agreement_then_only_goes_down(self):
        import botsystem_goal_ppo as gp
        s = self.fake()
        s.teach.update(hit=2, n=20)                                            # 10 % agreement: not settled, level stays
        self.assertEqual(gp.GoalServer.fade_teacher(s)["level"], 1.0)
        s.teach.update(hit=18, n=20)                                           # EMA 0.7*0.1+0.3*0.9 = 0.34: still below 0.55
        self.assertEqual(gp.GoalServer.fade_teacher(s)["level"], 1.0)
        for _ in range(3):
            s.teach.update(hit=19, n=20)
            out = gp.GoalServer.fade_teacher(s)
        self.assertLess(out["level"], 1.0)
        low = out["level"]
        s.teach.update(hit=0, n=20)                                            # the policy drifts: no re-teaching
        self.assertLessEqual(gp.GoalServer.fade_teacher(s)["level"], low)
        for _ in range(200):
            s.teach.update(hit=20, n=20)
            out = gp.GoalServer.fade_teacher(s)
        self.assertEqual(out["level"], 0.0)

    def test_pair_pruning_drops_the_bottom_share_only_with_enough_evidence(self):
        import botsystem_goal_ppo as gp
        s = self.fake()
        e = lambda p: {"p": p}
        s.pair_score = {("pair", i): {"ema": float(i), "n": 3} for i in range(8)}
        self.assertFalse(gp.GoalServer.pair_ok(s, e(0)))                       # bottom quartile
        self.assertTrue(gp.GoalServer.pair_ok(s, e(7)))
        s.pair_score[("pair", 0)]["n"] = 1                                     # too few episodes: kept
        self.assertTrue(gp.GoalServer.pair_ok(s, e(0)))
        s.pair_score = {("pair", i): {"ema": float(i), "n": 3} for i in range(3)}
        self.assertTrue(gp.GoalServer.pair_ok(s, e(0)))                        # fewer than 4 pairs known: no pruning

    def test_behaviour_cloning_positive_control(self):
        """With only the cloning term the policy moves its probability onto the teacher's candidate (a null would be uninformative
        without this); steps the teacher executed carry no policy gradient."""
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import jax
        import jax.numpy as jnp
        import optax
        import botsystem_goal_ppo as gp
        import test_botsystem_goal as tg
        sizes = tg.sizes_from_hello(tg.HELLO)
        a = gp.parse_args(["--root", "/nonexistent", "--base", "x", "--sil-coef", "0"])
        model = tg.GoalWM(sizes)
        params = model.init(jax.random.PRNGKey(0))
        lr = gp.GoalLearner(a, model, params)
        rng = np.random.default_rng(0)
        gs = gp.GoalSpace(sizes, tg.MILESTONES)
        B, T = 4, 6
        rows = []
        for _ in range(B * T):
            o = tg.make_obs(rng, {2: 1}, n_valid=6)
            rows.append({**gp.to_arrays(o, sizes), **gs.encode({"tokens": [(("item", 2), 1.0, 1.0)], "t0": 0, "task": False}, gs.values(o), 0, 10, 2)})
        obs = {k: np.stack([r[k] for r in rows]).reshape((B, T) + np.shape(rows[0][k])) for k in rows[0]}
        zero = np.zeros((B, T), np.float32)
        b = {"obs": obs, "h0": np.zeros((B, model.width), np.float32), "act": np.zeros((B, T), np.int32), "logp": zero, "adv": zero,
             "ret": zero, "valid": np.ones((B, T), np.float32), "tch": np.full((B, T), 2, np.int32), "src": np.zeros((B, T), np.float32)}
        sb = lr.sil_batch([], 1.0, rows[0], rng)
        opt = optax.adam(3e-3)
        st = opt.init(params)
        p0 = float(jnp.exp(jax.nn.log_softmax(lr._unroll(params, b["h0"], b["obs"])[0], -1)[..., 2]).mean())
        loss = lambda p, w: lr._loss(p, b, sb, 0.0, w)[0]
        g = jax.jit(jax.grad(loss))
        p = params
        for _ in range(60):
            u, st = opt.update(g(p, 5.0), st, p)
            p = optax.apply_updates(p, u)
        p1 = float(jnp.exp(jax.nn.log_softmax(lr._unroll(p, b["h0"], b["obs"])[0], -1)[..., 2]).mean())
        self.assertLess(p0, 0.4)
        self.assertGreater(p1, 0.8, (p0, p1))
        # executed-by-teacher steps: the policy-gradient term is masked out
        b2 = {**b, "src": np.ones((B, T), np.float32), "adv": np.ones((B, T), np.float32), "logp": zero - 5.0}
        m_on = lr._loss(params, b2, sb, 0.0, 0.0)[1]["pg"]
        self.assertEqual(float(m_on), 0.0)


if __name__ == "__main__":
    unittest.main()
