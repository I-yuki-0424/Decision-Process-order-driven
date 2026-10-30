"""Contract tests for integrations/botsystem: action spec, file channel, policy server, and the research/integration boundary.

A fake environment plays BOTSYSTEM's side by appending lines to env_to_policy.jsonl. Run:
    python -m unittest tests.test_botsystem_interface
"""
import ast
import json
import tempfile
import unittest
from pathlib import Path

from integrations.botsystem.action_spec import (ActionError, ActionSpec, Axis, Key, Rule, proposal_full,
                                                proposal_min, proposal_min_pitch)
from integrations.botsystem.file_channel import ChannelError, LineReader
from integrations.botsystem.protocol import ENV_TYPES
from integrations.botsystem.serve import NoopPolicy, PolicyError, PolicyServer, UniformRandomPolicy

ROOT = Path(__file__).resolve().parent.parent
OBS_SPEC = {"fields": [{"name": "pos", "shape": [3], "dtype": "float"}, {"name": "on_ground", "shape": [], "dtype": "bool"}]}


class TestActionSpec(unittest.TestCase):
    def test_min_pitch_is_min_plus_pitch(self):
        a, m = proposal_min_pitch(), proposal_min()
        self.assertEqual(a.components[:-1], m.components)
        self.assertEqual(a.rules, m.rules)
        self.assertEqual(len(list(a.iter_valid())), 90 * 3)
        # cross-language vector: BOTSYSTEM's JS implementation (src/rl/action_spec.js) must produce the same hash
        self.assertEqual(a.spec_hash(), "cd36d198ee6accff8732129686184d68d873d5b3d6625cb7ed2359c6ae341a2c")

    def test_sizes_and_valid_count(self):
        full, small = proposal_full(), proposal_min()
        self.assertEqual(full.size, 3 ** 9 * 5 * 3 * 10)
        with self.assertRaises(ActionError):
            list(full.iter_valid())
        valid = list(small.iter_valid())
        # forward 3 x jump 3 x sprint 2 x attack 2 x yaw 3 = 108 combinations
        self.assertEqual(small.size, 108)
        self.assertEqual(len(valid), 108 - 3 * 2 * 3)  # minus sprint=hold with forward=off: 3 jump x 2 attack x 3 yaw
        self.assertTrue(all(small.is_valid(a) for a in valid))

    def test_round_trip_and_rules(self):
        s = proposal_full()
        a = s.noop()
        a.update(forward="hold", sprint="hold", jump="tap", yaw_delta_deg=15.0, hotbar=3)
        self.assertEqual(s.decode(s.encode(a)), a)
        for bad in ({"forward": "hold", "back": "tap"}, {"sprint": "hold"}, {"sprint": "hold", "forward": "hold", "sneak": "hold"},
                    {"jump": "press"}, {"yaw_delta_deg": 14.0}, {"hotbar": 10}, {"jump": True}):
            with self.assertRaises(ActionError, msg=str(bad)):
                s.validate({**s.noop(), **bad})
        with self.assertRaises(ActionError):
            s.validate({"forward": "hold"})  # missing components

    def test_tap_is_first_tick_only(self):
        s = proposal_full(ticks_per_step=3)
        a = {**s.noop(), "jump": "tap", "forward": "hold"}
        sched = s.key_schedule(a)
        self.assertEqual([t["jump"] for t in sched], [True, False, False])
        self.assertEqual([t["forward"] for t in sched], [True, True, True])
        self.assertEqual([t["attack"] for t in sched], [False, False, False])

    def test_tap_rejected_at_one_tick(self):
        with self.assertRaises(ActionError):
            proposal_full(ticks_per_step=1)
        ActionSpec("x", 0, 1, (Key("jump", ("off", "hold")),))

    def test_wire_round_trip_and_hash(self):
        s = proposal_full()
        wire = json.loads(json.dumps(s.to_wire()))
        self.assertEqual(ActionSpec.from_wire(wire), s)
        wire["components"][-2]["values"] = [-15, 0, 15]  # ints instead of floats, as another language may write them
        again = ActionSpec.from_wire(wire)
        self.assertEqual(again.spec_hash(), ActionSpec.from_wire(json.loads(json.dumps(wire))).spec_hash())
        self.assertNotEqual(s.spec_hash(), proposal_min().spec_hash())
        for broken in (None, {}, {**s.to_wire(), "components": [{"kind": "wheel", "name": "x"}]},
                       {**s.to_wire(), "rules": [{"kind": "exclusive", "a": "forward", "b": "nope"}]}):
            with self.assertRaises(ActionError):
                ActionSpec.from_wire(broken)

    def test_rule_must_name_keys(self):
        with self.assertRaises(ActionError):
            ActionSpec("x", 0, 2, (Axis("yaw", (0.0,)),), (Rule("exclusive", "yaw", "yaw"),))


class TestChannel(unittest.TestCase):
    def test_partial_line_and_bad_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.jsonl"
            good = {"v": 1, "type": "bye", "run_id": "r", "env_id": 0, "seq": 0}
            r = LineReader(p, run_id="r", allowed=ENV_TYPES)
            self.assertEqual(r.poll(), [])
            line = json.dumps(good).encode()
            p.write_bytes(b"not json\n" + b'{"v":1,"type":"bye","run_id":"r","env_id":0,"seq":NaN}\n'
                          + b'{"v":1,"type":"act","run_id":"r","env_id":0,"seq":1}\n' + line[:10])
            self.assertEqual(r.poll(), [])        # the tail has no newline yet
            self.assertEqual(r.rejected, 3)       # garbage, NaN, wrong direction (act is not an env message)
            with open(p, "ab") as f:
                f.write(line[10:] + b"\n")
            self.assertEqual(r.poll(), [good])
            self.assertEqual(r.rejected, 3)
            p.write_bytes(b"")
            with self.assertRaises(ChannelError):
                r.poll()


class FakeEnv:
    """BOTSYSTEM's side, reduced to what the contract needs."""
    def __init__(self, root: Path, run_id: str = "r1"):
        self.dir, self.run_id, self.seq = root / run_id, run_id, 0
        self.dir.mkdir(parents=True)
        self.env_file, self.pol_file = self.dir / "env_to_policy.jsonl", self.dir / "policy_to_env.jsonl"

    def send(self, type_, **f):
        msg = {"v": 1, "type": type_, "run_id": self.run_id, "env_id": 0, "seq": self.seq, "t": 0.0, **f}
        with open(self.env_file, "ab") as fh:
            fh.write(json.dumps(msg).encode() + b"\n")
        self.seq += 1

    def hello(self, spec):
        self.send("hello", action_spec=spec.to_wire(), obs_spec=OBS_SPEC, ticks_per_step=spec.ticks_per_step, env_version="fake")

    def obs(self, episode, step, *, terminated=False, truncated=False, reward=0.0, **over):
        o = {"pos": [0.0, 64.0, 0.0], "on_ground": True}
        o.update(over)
        self.send("obs", episode=episode, step=step, obs=o, reward=reward, terminated=terminated, truncated=truncated)

    def replies(self):
        if not self.pol_file.exists():
            return []
        return [json.loads(x) for x in self.pol_file.read_text().splitlines()]


class TestServer(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.env = FakeEnv(self.root)
        self.spec = proposal_full()

    def tearDown(self):
        self._tmp.cleanup()

    def server(self, policy=None, **kw):
        s = PolicyServer(self.root, "r1", policy or UniformRandomPolicy(0), **kw)
        self.addCleanup(s.close)
        return s

    def test_episode_flow(self):
        srv = self.server(expected_spec=self.spec)
        self.env.hello(self.spec)
        for step in range(3):
            self.env.obs(0, step)
        self.env.obs(0, 3, truncated=True, reward=1.5)
        self.assertEqual(srv.poll_once(), 3)
        out = self.env.replies()
        self.assertEqual([m["type"] for m in out], ["hello_ack", "act", "act", "act"])
        self.assertTrue(out[0]["accepted"])
        self.assertEqual(out[0]["action_spec_hash"], self.spec.spec_hash())
        self.assertEqual([(m["episode"], m["step"]) for m in out[1:]], [(0, 0), (0, 1), (0, 2)])
        self.assertEqual([m["seq"] for m in out], [0, 1, 2, 3])
        self.assertTrue(all(self.spec.is_valid(m["action"]) for m in out[1:]))
        self.assertTrue(all(m["obs_seq"] == m["step"] + 1 for m in out[1:]))

    def test_spec_mismatch_is_refused(self):
        srv = self.server(expected_spec=proposal_min())
        self.env.hello(self.spec)
        self.env.obs(0, 0)
        self.assertEqual(srv.poll_once(), 0)
        out = self.env.replies()
        self.assertEqual([m["type"] for m in out], ["hello_ack"])
        self.assertFalse(out[0]["accepted"])
        self.assertIn("differs", out[0]["reason"])

    def test_restart_does_not_answer_twice(self):
        s1 = self.server(NoopPolicy())
        self.env.hello(self.spec)
        self.env.obs(0, 0)
        s1.poll_once()
        s2 = self.server(NoopPolicy())          # new process over the same files
        self.assertEqual(s2.poll_once(), 0)     # replays hello (re-acked) and obs 0 (skipped)
        self.env.obs(0, 1)
        self.assertEqual(s2.poll_once(), 1)
        acts = [(m["episode"], m["step"]) for m in self.env.replies() if m["type"] == "act"]
        self.assertEqual(acts, [(0, 0), (0, 1)])

    def test_undeclared_or_malformed_obs_is_rejected(self):
        srv = self.server()
        self.env.hello(self.spec)
        self.env.obs(0, 0, true_goal_dir=[1.0, 0.0, 0.0])   # field not declared in hello: possible leak
        self.env.obs(0, 1, pos=[0.0, 64.0])                  # wrong shape
        self.env.obs(0, 2, on_ground=1)                      # int where bool is declared
        self.env.obs(0, 3)
        self.assertEqual(srv.poll_once(), 1)
        self.assertEqual([m["step"] for m in self.env.replies() if m["type"] == "act"], [3])

    def test_obs_before_hello_is_ignored(self):
        srv = self.server()
        self.env.obs(0, 0)
        self.assertEqual(srv.poll_once(), 0)
        self.assertEqual(self.env.replies(), [])

    def test_invalid_policy_action_stops_the_server(self):
        class Bad(NoopPolicy):
            def act(self, msg):
                return {**self._spec.noop(), "sprint": "hold"}   # sprint without forward
        srv = self.server(Bad())
        self.env.hello(self.spec)
        self.env.obs(0, 0)
        with self.assertRaises(PolicyError):
            srv.poll_once()
        self.assertEqual([m["type"] for m in self.env.replies()], ["hello_ack"])

    def test_bye_ends_run(self):
        srv = self.server()
        self.env.hello(self.spec)
        self.env.send("bye")
        srv.run(poll_s=0.001, idle_timeout_s=5)
        self.assertTrue(srv.finished)


class TestBoundary(unittest.TestCase):
    def test_research_code_does_not_import_integrations(self):
        offenders = []
        for py in sorted(ROOT.glob("src/**/*.py")):
            for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
                mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                        else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                if any(m.split(".")[0] == "integrations" for m in mods):
                    offenders.append(str(py.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_interface_core_is_standard_library_only(self):
        allowed = {"__future__", "argparse", "dataclasses", "hashlib", "itertools", "json", "logging", "math", "os", "pathlib",
                   "random", "time", "typing"}
        for py in sorted((ROOT / "integrations").glob("**/*.py")):
            for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    roots = {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    roots = {(node.module or "").split(".")[0]}
                else:
                    continue
                self.assertLessEqual(roots, allowed, f"{py.name} imports outside the standard library")


if __name__ == "__main__":
    unittest.main()
