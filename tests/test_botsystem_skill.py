"""Tests of the BOTSYSTEM skill-level learner (scripts/botsystem_skill_truck.py, scripts/botsystem_skill_ppo.py).
Needs JAX. Side track: nothing here counts toward a roadmap gate.

  python -m unittest tests.test_botsystem_skill
The end-to-end test is the learner's positive control: a fake environment over the real file API where one candidate per step
(at a random index, marked only by its command embedding) pays reward 1; the server must learn to pick it.
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

import botsystem_skill_ppo as sp  # noqa: E402
from botsystem_skill_truck import SkillTruck, sizes_from_hello  # noqa: E402

K, P = 48, 4
MILESTONES, NOTABLE = ["minecraft:end/kill_dragon", "minecraft:story/root"], ["ender_dragon"]
P = len(MILESTONES) + len(NOTABLE) + 2
FIELDS = [("grid", [405], "int"), ("self", [20], "float"), ("inv_item", [36], "int"), ("inv_count", [36], "float"),
          ("ent_type", [8], "int"), ("ent_feat", [8, 5], "float"), ("progress", [P], "float"), ("cand_cmd", [K], "int"),
          ("cand_item", [K], "int"), ("cand_ent", [K], "int"), ("cand_feat", [K, 8], "float"), ("cand_mask", [K], "bool"),
          ("last_reason", [], "int"), ("dt_ticks", [], "float")]
HELLO = {"env_version": "test", "action_spec": {"name": "skills_v1", "version": 1, "ticks_per_step": 1,
                                                "components": [{"kind": "select", "name": "candidate", "n": K}], "rules": []},
         "obs_spec": {"fields": [{"name": n, "shape": s, "dtype": d} for n, s, d in FIELDS]},
         "vocab_sizes": {"item": 50, "entity": 20, "cmd": 12, "reason": 18, "category": 11}, "milestones": MILESTONES,
         "notable": NOTABLE, "reward_terms": ["survival", "kills", "notable", "mined", "achievements"],
         "stages": {"survive": {"survival": 1.0, "kills": 0.1, "notable": 0.1, "mined": 0.1, "achievements": 0.1},
                    "dragon": {"survival": 1.0, "kills": 1.0, "notable": 1.0, "mined": 1.0, "achievements": 1.0}}}


def make_obs(rng, n_valid=8, good=None, nights=0.0):
    cmd = np.zeros(K, int)
    cmd[:n_valid] = rng.integers(1, 12, n_valid)
    cmd[:n_valid][cmd[:n_valid] == 2] = 3
    if good is not None:
        cmd[good] = 2
    prog = [0.0] * P
    prog[-2] = nights
    return {"grid": rng.integers(0, 11, 405).tolist(), "self": rng.random(20).round(3).tolist(), "inv_item": [0] * 36,
            "inv_count": [0.0] * 36, "ent_type": [0] * 8, "ent_feat": [[0.0] * 5] * 8, "progress": prog,
            "cand_cmd": cmd.tolist(), "cand_item": [0] * K, "cand_ent": [0] * K, "cand_feat": rng.random((K, 8)).round(3).tolist(),
            "cand_mask": [i < n_valid for i in range(K)], "last_reason": 0, "dt_ticks": 20.0}


class TestMath(unittest.TestCase):
    def test_gae_semi_mdp(self):
        rew, val, dt, done = [1.0, 0.0], [0.5, 0.25], [20.0, 40.0], [False, True]
        adv, ret = sp.gae(rew, val, 9.0, dt, done, 0.9, 1.0)
        # step 1 is terminal: delta = 0 - 0.25; step 0: delta = 1 + 0.9 * 0.25 - 0.5, adv = delta + 0.9 * adv1
        self.assertAlmostEqual(float(adv[1]), -0.25, places=6)
        self.assertAlmostEqual(float(adv[0]), 1 + 0.9 * 0.25 - 0.5 + 0.9 * -0.25, places=6)
        _, ret2 = sp.gae([0.0], [0.0], 1.0, [40.0], [False], 0.9, 0.95)   # 40 ticks = 2 s -> 0.81
        self.assertAlmostEqual(float(ret2[0]), 0.81, places=6)

    def test_stage_reward(self):
        self.assertAlmostEqual(sp.stage_reward({"survival": 2, "mined": 10}, HELLO["stages"]["survive"]), 3.0)

    def test_curriculum(self):
        a = sp.parse_args(["--root", "x", "--base", "b", "--stage-window", "4", "--stage-windows", "3", "--stage1-max-steps", "10**9".replace("**9", "00000")])
        c = sp.Curriculum(a, MILESTONES, NOTABLE)
        self.assertEqual(list(c.target()), [0, 0, 0, 1, 1])
        for n in [0, 0, 0, 0, 2, 2, 2, 2, 1, 2, 2, 2]:
            self.assertIsNone(c.end_episode(n))   # unstable (0 then 2), then the first two stable blocks
        whys = [c.end_episode(n) for n in [2, 2, 2, 1.8, 1.8, 2, 2, 2]]
        why = next(w for w in whys if w)
        self.assertEqual(c.stage, 1)
        self.assertIn("stable", why)
        self.assertEqual(list(c.target()), [1, 0, 1, 0, 0])
        c2 = sp.Curriculum(sp.parse_args(["--root", "x", "--base", "b", "--stage1-max-steps", "5"]), MILESTONES, NOTABLE)
        c2.steps[0] = 6
        self.assertIn("budget", c2.end_episode(0.0))


class TestModel(unittest.TestCase):
    def test_masking_and_shapes(self):
        sizes = sizes_from_hello(HELLO)
        m = SkillTruck(sizes)
        params = m.init(jax.random.PRNGKey(0))
        rng = np.random.default_rng(0)
        obs = [sp.to_arrays(make_obs(rng, n_valid=3 + i), np.zeros(P, np.float32)) for i in range(4)]
        batch = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        logits, value, h = m.step(params, m.init_carry(4), batch, jnp.array([0, 1, 0, 1]))
        self.assertEqual(logits.shape, (4, K)), self.assertEqual(value.shape, (4,)), self.assertEqual(h.shape, (4, 256))
        for i in range(4):
            self.assertTrue(bool(jnp.all(logits[i, 3 + i:] <= -1e8)) and bool(jnp.all(logits[i, :3 + i] > -1e8)))
        tot, dep = m.param_counts(params)
        self.assertLess(tot, 600_000)
        self.assertLess(dep, tot)


class FakeEnv:
    """Writes the environment side of the file API for two bots; candidate cmd 2 at a random index pays reward 1."""

    def __init__(self, root: Path, base: str, run: str, seed=0):
        from integrations.botsystem.file_channel import LineReader, LineWriter
        d = root / run
        d.mkdir(parents=True)
        (root / f"{base}.current").write_text(run + "\n")
        self.w = LineWriter(d / "env_to_policy.jsonl")
        self.r = LineReader(d / "policy_to_env.jsonl", run_id=run, allowed=("hello_ack", "act"))
        self.run, self.seq, self.rng = run, 0, np.random.default_rng(seed)
        self.hits = []

    def send(self, env_id, type_, **kw):
        self.w.send({"v": 1, "type": type_, "run_id": self.run, "env_id": env_id, "seq": self.seq, "t": time.time(), **kw})
        self.seq += 1

    def wait(self, pred, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            for m in self.r.poll():
                if pred(m):
                    return m
            time.sleep(0.005)
        raise TimeoutError

    def play(self, steps, ep_len=40, srv=None):
        for e in (0, 1):
            self.send(e, "hello", **HELLO)
            assert self.wait(lambda m: m["type"] == "hello_ack" and m["env_id"] == e)["accepted"]
        state = {e: {"episode": 0, "step": 0, "good": None, "reward": 0.0} for e in (0, 1)}
        terms0 = {"survival": 0.0, "kills": 0.0, "notable": 0.0, "mined": 0.0, "achievements": 0.0}
        for i in range(steps):
            while srv is not None and i % 16 == 0 and len(srv.queue) >= srv.a.batch_segs:   # let the learner keep up
                time.sleep(0.05)
            for e in (0, 1):
                s = state[e]
                done = s["step"] >= ep_len
                good = int(self.rng.integers(0, 8))
                self.send(e, "obs", episode=s["episode"], step=s["step"], obs=make_obs(self.rng, good=good, nights=0.1),
                          reward=s["reward"], reward_terms={**terms0, "survival": s["reward"]}, terminated=False, truncated=done)
                if done:
                    state[e] = {"episode": s["episode"] + 1, "step": 0, "good": None, "reward": 0.0}
                    continue
                act = self.wait(lambda m: m["type"] == "act" and m["env_id"] == e and m["step"] == s["step"] and m["episode"] == s["episode"])
                hit = act["action"]["candidate"] == good
                self.hits.append(hit)
                s.update(step=s["step"] + 1, reward=1.0 if hit else 0.0)


class TestEndToEnd(unittest.TestCase):
    def test_positive_control_learns_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ["--root", tmp, "--base", "pc", "--seg", "8", "--batch-segs", "8", "--epochs", "4", "--minibatches", "2",
                    "--lr", "1e-3", "--n-max", "2", "--poll-s", "0.002", "--stage1-max-steps", "10000000"]
            srv = sp.Server(sp.parse_args(args))
            th = threading.Thread(target=srv.loop, daemon=True)
            th.start()
            env = FakeEnv(root, "pc", "pc-1")
            env.play(1600, srv=srv)
            srv.stop.set(), th.join(10), srv.learn_thread.join(60)
            early, late = np.mean(env.hits[:300]), np.mean(env.hits[-300:])
            self.assertGreater(srv.learner.updates, 10)
            self.assertLess(early, 0.35)               # chance is 1/8
            self.assertGreater(late, 0.5, f"positive control did not learn: early {early:.2f}, late {late:.2f}")
            log = [json.loads(x) for x in (root / "pc" / "train" / "train_log.jsonl").read_text().splitlines()]
            ups = [x for x in log if x.get("event") == "update"]
            self.assertTrue(all(np.isfinite(x["loss"]) for x in ups))
            eps = [json.loads(x) for x in (root / "pc" / "train" / "episodes.jsonl").read_text().splitlines()]
            self.assertTrue(eps and all(e["stage"] == "survive" and e["steps"] == 40 for e in eps))
            n_up = srv.learner.updates
            srv2 = sp.Server(sp.parse_args(args))
            srv2.follow()
            for m in srv2.reader.poll():
                if m["type"] == "hello":
                    srv2.on_hello(m)
                    break
            self.assertEqual(srv2.learner.updates, n_up)
            self.assertEqual(srv2.total_steps, srv.total_steps)
            srv2.stop.set()
            print(f"positive control: chance 0.125, first 300 {early:.3f}, last 300 {late:.3f}, {n_up} updates")


if __name__ == "__main__":
    unittest.main()
