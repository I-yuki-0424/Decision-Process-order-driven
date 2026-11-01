"""Tests of the goal-conditioned world-model learner (scripts/botsystem_goal_wm.py, scripts/botsystem_goal_ppo.py). Needs JAX.
Side track (TASK-20261101-031): nothing here counts toward a roadmap gate.

  python -m unittest tests.test_botsystem_goal
Positive controls: (1) the world model learns success and inventory effects from synthetic transitions, and its predicted
inventory never goes below zero; (2) end to end over the real file API, a goal-only reward ("+1 of item X", X drawn from what
the bots have raised) is learned: the policy must pick the candidate whose item is the goal's.
"""
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import botsystem_goal_ppo as gp  # noqa: E402
from botsystem_goal_wm import GK, GoalWM, held_vector, plan_beam, sizes_from_hello  # noqa: E402

K, NI, NE = 16, 50, 20
MILESTONES, NOTABLE = ["minecraft:end/kill_dragon", "minecraft:story/root"], ["ender_dragon"]
P = len(MILESTONES) + len(NOTABLE) + 2
VS = (2, 4)
R = VS[0] * VS[1]
FIELDS = [("grid", [405], "int"), ("self", [20], "float"), ("inv_item", [36], "int"), ("inv_count", [36], "float"),
          ("ent_type", [8], "int"), ("ent_feat", [8, 5], "float"), ("progress", [P], "float"), ("view_cat", [R], "int"),
          ("view_item", [R], "int"), ("view_dist", [R], "float"), ("killed", [NE], "float"), ("clock", [2], "float"),
          ("cand_cmd", [K], "int"), ("cand_item", [K], "int"), ("cand_ent", [K], "int"), ("cand_feat", [K, 8], "float"),
          ("cand_mask", [K], "bool"), ("last_reason", [], "int"), ("dt_ticks", [], "float")]
TERMS = ["survival", "kills", "notable", "mined", "achievements", "invalid"]
HELLO = {"env_version": "test-v2", "action_spec": {"name": "skills_v1", "version": 1, "ticks_per_step": 1,
                                                   "components": [{"kind": "select", "name": "candidate", "n": K}], "rules": []},
         "obs_spec": {"fields": [{"name": n, "shape": s, "dtype": d} for n, s, d in FIELDS]},
         "vocab_sizes": {"item": NI, "entity": NE, "cmd": 13, "reason": 18, "category": 11}, "milestones": MILESTONES,
         "notable": NOTABLE, "reward_terms": TERMS, "view_shape": list(VS),
         "vocab": {"items": ["none"] + [f"i{j}" for j in range(1, NI)], "entities": ["none"] + [f"e{j}" for j in range(1, NE)]},
         "stages": {"dragon": {t: 1.0 for t in TERMS}, "zero": {t: 0.0 for t in TERMS}}}


def make_obs(rng, inv: dict, items=None, n_valid=8, kills=None, nights=0.0, reason=0):
    items = items if items is not None else rng.integers(1, 11, n_valid)
    slots = sorted(inv.items())[:36]
    prog = [0.0] * P
    prog[-2] = nights
    killed = [0.0] * NE
    for j, k in (kills or {}).items():
        killed[j] = k / 8
    return {"grid": rng.integers(0, 11, 405).tolist(), "self": rng.random(20).round(3).tolist(),
            "inv_item": [i for i, _ in slots] + [0] * (36 - len(slots)), "inv_count": [c / 64 for _, c in slots] + [0.0] * (36 - len(slots)),
            "ent_type": [0] * 8, "ent_feat": [[0.0] * 5] * 8, "progress": prog, "view_cat": rng.integers(0, 11, R).tolist(),
            "view_item": rng.integers(0, NI, R).tolist(), "view_dist": rng.random(R).round(3).tolist(), "killed": killed, "clock": [0.0, 0.1],
            "cand_cmd": [5] * n_valid + [0] * (K - n_valid), "cand_item": [int(x) for x in items] + [0] * (K - n_valid), "cand_ent": [0] * K,
            "cand_feat": rng.random((K, 8)).round(3).tolist(), "cand_mask": [i < n_valid for i in range(K)], "last_reason": reason, "dt_ticks": 20.0}


class TestGoalSpace(unittest.TestCase):
    def setUp(self):
        self.gs = gp.GoalSpace(sizes_from_hello(HELLO), MILESTONES)

    def test_values_reached_encode(self):
        rng = np.random.default_rng(0)
        o = make_obs(rng, {3: 2, 7: 1}, kills={4: 2}, nights=0.1)
        o["progress"][1] = 1.0
        v = self.gs.values(o)
        self.assertEqual((v[("item", 3)], v[("item", 7)], v[("kill", 4)], v[("adv", 1)], v[("night", 0)]), (2, 1, 2, 1.0, 1))
        g = {"tokens": [(("item", 3), 2.0, 1.0)], "t0": 10, "task": False}
        self.assertFalse(self.gs.reached(g, v))
        self.assertTrue(self.gs.reached(g, {**v, ("item", 3): 3}))
        enc = self.gs.encode(g, {**v, ("item", 3): 2}, step=40, horizon=60, G=2)
        self.assertEqual(list(enc["goal_kind"]), [GK["item"], 0]), self.assertEqual(list(enc["goal_mask"]), [True, False])
        np.testing.assert_allclose(enc["goal_feat"][0], [np.log1p(1), 0.0, 0.5], rtol=1e-6)
        self.gs.note_raised(v, {**v, ("item", 3): 3, ("item", 9): 1})
        self.assertEqual(set(self.gs.archive), {("item", 3), ("item", 9)})
        self.assertEqual(self.gs.task_dims({"diamond": 5}), [("night", 0), ("item", 5), ("adv", 0)])

    def test_goal_bonus_owner_rule_trivial_results_worth_less_than_a_log(self):
        # owner 2026-11-02: thousands of trivial results must be worth less than one log (reward-v2: oak_log 0.2)
        trivial = sum(gp.goal_value(0.5, 0.95, n) for n in range(1000))     # a goal reached 95 % of the time, 1000x per episode
        self.assertLess(trivial, 0.2)
        self.assertAlmostEqual(gp.goal_value(0.5, 0.0, 0), 0.5)               # a goal never reached before pays the full bonus
        self.assertAlmostEqual(gp.goal_value(0.5, 0.2, 2), 0.5 * 0.8 * 0.25)

    def test_hindsight_labels_are_true(self):
        rng = np.random.default_rng(1)
        now = {("item", 1): 2, ("night", 0): 0}
        future = [{("item", 1): 2}, {("item", 1): 3, ("kill", 2): 1}, {("item", 1): 1}]
        h = gp.hindsight(now, future, [("item", 1), ("item", 5), ("adv", 0), ("kill", 2)], rng)
        lab = {(int(k), int(i)): float(l) for k, i, l, m in zip(h["her_kind"], h["her_id"], h["her_label"], h["her_mask"]) if m}
        self.assertEqual(lab[(GK["item"], 1)], 1.0), self.assertEqual(lab[(GK["kill"], 2)], 1.0)
        self.assertEqual(len(lab), 4)
        self.assertTrue(all(v == 0.0 for (k, i), v in lab.items() if (k, i) not in ((GK["item"], 1), (GK["kill"], 2))))


class TestPlan(unittest.TestCase):
    def test_dim_index_roundtrip_and_hindsight_plans(self):
        s = sizes_from_hello(HELLO)
        for d in [("item", 0), ("item", NI - 1), ("kill", 3), ("adv", 1), ("adv", P - 1), ("night", 0)]:
            self.assertEqual(gp.index_dim(gp.dim_index(d, s), s), d)
        self.assertIsNone(gp.index_dim(0, s))
        firsts = [(3, ("item", 4)), (7, ("item", 5)), (9, ("kill", 2)), (12, ("item", 6))]
        rng = np.random.default_rng(0)
        out = gp.plan_samples(firsts, [(0, {"a": 1}), (8, {"a": 2}), (20, {"a": 3})], s, 8, rng)
        self.assertEqual(len(out), 2)                                     # the anchor at 20 has nothing after it
        for x, lab in out:
            seq = [gp.index_dim(int(j), s) for j, m in zip(lab["seq"], lab["mask"]) if m]
            self.assertIsNone(seq[-1])                                    # END closes the plan
            self.assertEqual(seq[-2], gp.index_dim(int(lab["task"]), s))  # T is the last step
            order = [d for _, d in firsts]
            self.assertEqual(seq[:-1], sorted(seq[:-1], key=order.index))  # in the order they were first raised
        long = [(i, ("item", i + 1)) for i in range(1, 20)]
        x, lab = gp.plan_samples(long, [(0, {})], s, 8, np.random.default_rng(5))[0]
        self.assertLessEqual(int(lab["mask"].sum()), 8)

    def test_plan_decoder_positive_control_learns_orders_and_beam_reproduces_them(self):
        """Hidden 'tech chains': reaching item T needs items in a fixed order (two chains). Trained on hindsight samples
        only, beam search for T must return its chain."""
        sizes = sizes_from_hello(HELLO)
        m = GoalWM(sizes)
        params = m.init(jax.random.PRNGKey(0))
        chains = {9: [2, 3, 5, 9], 8: [4, 6, 8]}
        rng = np.random.default_rng(0)
        o0 = gp.to_arrays(make_obs(rng, {}), sizes)

        def batch(n):
            obs, task, seq, mask = [], [], [], []
            for _ in range(n):
                T = int(rng.choice([9, 8]))
                firsts = [(t + 1, ("item", d)) for t, d in enumerate(chains[T])]
                x, lab = gp.plan_samples(firsts, [(0, o0)], sizes, m.L, rng)[0]
                if gp.index_dim(int(lab["task"]), sizes) != ("item", T):     # keep samples whose target is the chain's end
                    continue
                obs.append(x), task.append(lab["task"]), seq.append(lab["seq"]), mask.append(lab["mask"])
            st = lambda L: np.stack(L)
            return {k: st([o[k] for o in obs]) for k in obs[0]}, st(task), st(seq), st(mask)

        import optax
        opt = optax.adam(3e-3)
        os_ = opt.init(params)

        @jax.jit
        def step(p, s_, o, t, q, mk):
            (loss, acc), g = jax.value_and_grad(lambda pp: m.plan_loss(pp, o, t, q, mk), has_aux=True)(p)
            u, s_ = opt.update(g, s_, p)
            return optax.apply_updates(p, u), s_, loss, acc
        for _ in range(300):
            params, os_, loss, acc = step(params, os_, *batch(32))
        enc = jax.jit(lambda P_, o: m.encode(P_["enc"], o)[:2])
        dec = jax.jit(jax.vmap(m.plan_logits, in_axes=(None, None, None, None, 0)))
        for T, chain in chains.items():
            best = plan_beam(m, params, enc, dec, o0, gp.dim_index(("item", T), sizes), beam=4)[0][0]
            self.assertEqual([gp.index_dim(j, sizes) for j in best], [("item", d) for d in chain], f"T={T}")
        print(f"plan decoder positive control: teacher-forced accuracy {float(acc):.3f}, beam reproduces both chains")


class TestModel(unittest.TestCase):
    def test_shapes_masking_and_inventory_balance(self):
        sizes = sizes_from_hello(HELLO)
        m = GoalWM(sizes)
        P_ = m.init(jax.random.PRNGKey(0))
        rng = np.random.default_rng(0)
        gs = gp.GoalSpace(sizes, MILESTONES)
        obs = []
        for i in range(3):
            o = make_obs(rng, {2: 1}, n_valid=3 + i)
            x = gp.to_arrays(o, sizes)
            obs.append({**x, **gs.encode({"tokens": [(("item", 2), 1.0, 1.0)], "t0": 0, "task": False}, gs.values(o), 0, 10, 2)})
        batch = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        logits, value, h, cls = m.step(P_, m.init_carry(3), batch)
        self.assertEqual((logits.shape, value.shape, h.shape, cls.shape), ((3, K), (3,), (3, 256), (3, 64)))
        for i in range(3):
            self.assertTrue(bool(jnp.all(logits[i, 3 + i:] <= -1e8)) and bool(jnp.all(logits[i, :3 + i] > -1e8)))
        held = held_vector(jnp.asarray(obs[0]["inv_item"]), jnp.asarray(obs[0]["inv_count"]), NI)
        z = jax.random.normal(jax.random.PRNGKey(1), (64,)) * 10
        self.assertTrue(bool(jnp.all(m.d_inv(P_, z, held, 2) >= -held)))       # never lose more than held
        self.assertTrue(bool(jnp.all(m.d_inv(P_, -z, held, 2) >= -held)))
        tot, dep = m.param_counts(P_)
        self.assertLess(dep, tot)

    def test_v13_observation_is_accepted_with_view_masked(self):
        h13 = {**HELLO, "obs_spec": {"fields": [f for f in HELLO["obs_spec"]["fields"] if f["name"] not in ("view_cat", "view_item", "view_dist", "killed", "clock")]}}
        s = sizes_from_hello(h13)
        self.assertFalse(s["has_view"] or s["has_killed"])
        o = make_obs(np.random.default_rng(0), {})
        for k in ("view_cat", "view_item", "view_dist", "killed", "clock"):
            del o[k]
        x = gp.to_arrays(o, s)
        self.assertFalse(bool(x["view_valid"]))
        self.assertEqual(x["view_cat"].shape, (R,))

    def test_world_model_positive_control(self):
        """Synthetic physics: a candidate succeeds iff its first feature > 0.5; crafting item X (succeeding) gives +1 X and
        costs 2 of item 3 when held. The aux loss alone must learn both."""
        sizes = sizes_from_hello(HELLO)
        m = GoalWM(sizes)
        params = m.init(jax.random.PRNGKey(0))
        rng = np.random.default_rng(0)

        def sample(n):
            xs, labs = [], []
            for _ in range(n):
                inv = {3: int(rng.integers(0, 4))} if rng.random() < 0.7 else {}
                inv = {k: v for k, v in inv.items() if v}
                o = make_obs(rng, inv)
                a = int(rng.integers(0, 8))
                ok = o["cand_feat"][a][0] > 0.5
                inv2 = dict(inv)
                if ok:
                    it = o["cand_item"][a]
                    inv2[it] = inv2.get(it, 0) + 1
                    if inv2.get(3, 0) >= 2 and it != 3:
                        inv2[3] -= 2
                    inv2 = {k: v for k, v in inv2.items() if v}
                o2 = make_obs(rng, inv2, reason=0 if ok else 3)
                x, x2 = gp.to_arrays(o, sizes), gp.to_arrays(o2, sizes)
                lab = {**gp.wm_labels(x2, o2, False, a, True), **gp.hindsight({}, [], [], rng)}
                xs.append(x), labs.append(lab)
            st = lambda L: {k: np.stack([d[k] for d in L]) for k in L[0]}
            return st(xs), st(labs)

        import optax
        opt = optax.adam(3e-3)
        os_ = opt.init(params)

        @jax.jit
        def step(p, s, o, l):
            (loss, (terms, _)), g = jax.value_and_grad(lambda q: (lambda r: (r[0], (r[1], r[2])))(m.aux_loss(q, o, l)), has_aux=True)(p)
            u, s = opt.update(g, s, p)
            return optax.apply_updates(p, u), s, loss
        for i in range(800):
            o, l = sample(32)
            params, os_, loss = step(params, os_, o, l)
        o, l = sample(256)
        _, terms, out = m.aux_loss(params, o, l)
        pk, y = np.asarray(out["_ok_p"]), l["ok"]
        auroc = np.mean([(pk[y == 1][:, None] > pk[y == 0][None, :]).mean()])
        gained = float(np.asarray(out["_gain_hit"]).sum() / max(np.asarray(out["_gain_has"]).sum(), 1))
        self.assertGreater(auroc, 0.85, f"success AUROC {auroc:.3f}")
        self.assertGreater(gained, 0.7, f"gained-item top-1 {gained:.3f}")
        print(f"world-model positive control: success AUROC {auroc:.3f}, gained-item top-1 {gained:.3f}")


class FakeEnv:
    """Environment side of the file API for two bots. Each step offers 8 candidates with random items 1..10; choosing one
    adds that item to the bot's inventory. The environment pays nothing: only the learner's goal bonus rewards."""

    def __init__(self, root: Path, base: str, run: str, seed=0, ids=(0, 1)):
        from integrations.botsystem.file_channel import LineReader, LineWriter
        d = root / run
        d.mkdir(parents=True)
        (root / f"{base}.current").write_text(run + "\n")
        self.ids, self.w = ids, LineWriter(d / "env_to_policy.jsonl")
        self.r = LineReader(d / "policy_to_env.jsonl", run_id=run, allowed=("hello_ack", "act"))
        self.run, self.seq, self.rng = run, 0, np.random.default_rng(seed)

    def send(self, env_id, type_, **kw):
        self.w.send({"v": 1, "type": type_, "run_id": self.run, "env_id": env_id, "seq": self.seq, "t": time.time(), **kw})
        self.seq += 1

    def wait(self, pred, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            for m in self.r.poll():
                if pred(m):
                    return m
            time.sleep(0.003)
        raise TimeoutError

    def play(self, steps, ep_len=40, srv=None):
        for e in self.ids:
            self.send(e, "hello", **HELLO)
            assert self.wait(lambda m: m["type"] == "hello_ack" and m["env_id"] == e)["accepted"]
        st = {e: {"episode": 0, "step": 0, "inv": {}} for e in self.ids}
        zero = {t: 0.0 for t in TERMS}
        for i in range(steps):
            while srv is not None and i % 8 == 0 and len(srv.queue) >= srv.a.batch_segs:
                time.sleep(0.05)
            for e in self.ids:
                s = st[e]
                done = s["step"] >= ep_len
                items = self.rng.permutation(np.arange(1, 11))[:8]
                self.send(e, "obs", episode=s["episode"], step=s["step"], obs=make_obs(self.rng, s["inv"], items=items),
                          reward=0.0, reward_terms=zero, terminated=False, truncated=done)
                if done:
                    st[e] = {"episode": s["episode"] + 1, "step": 0, "inv": {}}
                    continue
                act = self.wait(lambda m: m["type"] == "act" and m["env_id"] == e and m["step"] == s["step"] and m["episode"] == s["episode"])
                it = int(items[act["action"]["candidate"]])
                s["inv"][it] = s["inv"].get(it, 0) + 1
                s["step"] += 1


class TestEndToEnd(unittest.TestCase):
    def test_goal_positive_control_learns_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ["--root", tmp, "--base", "gc", "--seg", "8", "--batch-segs", "8", "--epochs", "4", "--minibatches", "2",
                    "--lr", "1e-3", "--n-max", "2", "--poll-s", "0.002", "--goal-horizon", "1", "--p-task", "0", "--p-pair", "0",
                    "--ext-stage", "zero", "--her-horizon", "4", "--replay", "2000", "--aux-batch", "32", "--aux-lr", "3e-3", "--plan", "0"]
            srv = gp.GoalServer(gp.parse_args(args))
            th = threading.Thread(target=srv.loop, daemon=True)
            th.start()
            env = FakeEnv(root, "gc", "gc-1")
            env.play(3200, srv=srv)
            time.sleep(1.0)
            srv.stop.set(), th.join(10), srv.learn_thread.join(120)
            log = [json.loads(x) for x in (root / "gc" / "train" / "train_log.jsonl").read_text().splitlines()]
            ups = [x for x in log if x.get("event") == "update"]
            self.assertGreater(len(ups), 10)
            self.assertTrue(all(np.isfinite(x["loss"]) for x in ups))
            # goals reached per goal assigned (horizon 1: one step per goal; chance = 1/8, ceiling ~0.8: the goal's item is
            # among the 8 offered of 10), first updates against the last third of the run
            g0, gm, g1 = ups[2]["goals"], ups[2 * len(ups) // 3]["goals"], ups[-1]["goals"]
            early = g0["reached"] / max(g0["assigned"], 1)
            late = (g1["reached"] - gm["reached"]) / max(g1["assigned"] - gm["assigned"], 1)
            self.assertGreater(ups[-1]["replay"], 100)
            self.assertGreater(ups[-1]["u_gain"], 0.1, "policy does not use the world model's predicted goal gain")
            self.assertLess(ups[-1]["aux_inv"], 0.1, "world model did not learn the inventory effect")
            # Significance bar, not an effect-size target: ~2,000 assignments in the last third give chance (0.125) an SE of
            # ~0.0074, so 0.18 is > 7 SE above chance (set 2026-11-02 after runs at 0.22-0.34 under CPU contention; 0.25 failed
            # on scheduling noise while the controls of the same runs clearly learned).
            self.assertGreater(late, early + 0.05)
            self.assertGreater(late, 0.18, f"goal positive control did not learn: early {early:.2f}, late {late:.2f}")
            n_up = srv.glearner.updates
            srv2 = gp.GoalServer(gp.parse_args(args))
            srv2.follow()
            for m in srv2.reader.poll():
                if m["type"] == "hello":
                    srv2.on_hello(m)
                    break
            self.assertEqual(srv2.glearner.updates, n_up)
            self.assertEqual(len(srv2.goals.archive), len(srv.goals.archive))
            srv2.stop.set()
            print(f"goal positive control: reached/assigned early {early:.3f}, late {late:.3f} (chance 0.125), {n_up} updates")

    def test_plan_mode_end_to_end_smoke(self):
        """Plans are batch-generated at episode start, executed step by step, logged, and hindsight plan samples train."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ["--root", tmp, "--base", "pm", "--seg", "8", "--batch-segs", "4", "--epochs", "1", "--minibatches", "1",
                    "--n-max", "2", "--poll-s", "0.002", "--goal-horizon", "6", "--p-task", "0", "--ext-stage", "zero",
                    "--her-horizon", "4", "--replay", "500", "--aux-batch", "8", "--plan", "1", "--p-explore", "0.2",
                    "--plan-batch", "4", "--plan-anchor", "8"]
            srv = gp.GoalServer(gp.parse_args(args))
            th = threading.Thread(target=srv.loop, daemon=True)
            th.start()
            FakeEnv(root, "pm", "pm-1").play(300, ep_len=30, srv=srv)
            time.sleep(1.0)
            srv.stop.set(), th.join(10), srv.learn_thread.join(120)
            log = [json.loads(x) for x in (root / "pm" / "train" / "train_log.jsonl").read_text().splitlines()]
            plans = [x for x in log if x.get("event") == "plan"]
            ups = [x for x in log if x.get("event") == "update"]
            self.assertTrue(plans and all(p["steps"] and p["steps"][-1] == p["target"] for p in plans), plans[:2])
            self.assertGreater(ups[-1]["goals"]["plans"], 0)
            self.assertGreater(ups[-1]["plan_replay"], 0)
            self.assertTrue(all(np.isfinite(x["aux_plan"]) for x in ups))

    def test_v13_hello_is_refused_by_the_online_learner(self):
        with tempfile.TemporaryDirectory() as tmp:
            srv = gp.GoalServer(gp.parse_args(["--root", tmp, "--base", "x"]))
            h13 = {**HELLO, "obs_spec": {"fields": [f for f in HELLO["obs_spec"]["fields"] if f["name"] not in ("view_cat", "view_item", "view_dist", "killed", "clock")]}}
            sent = []
            srv.send = lambda env_id, type_, run=None, **kw: sent.append(kw)
            srv.runs = {"x.current": {}}
            srv.on_hello({**h13, "env_id": 0}, "x.current")
            self.assertFalse(sent[0]["accepted"])
            self.assertIn("v2.0", sent[0]["reason"])


if __name__ == "__main__":
    unittest.main()
