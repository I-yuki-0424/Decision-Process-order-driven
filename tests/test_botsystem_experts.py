"""Router + goal-specific skill policies (integrations/botsystem/experts). Standard library only; no environment, no JAX.
Run: python -m unittest tests.test_botsystem_experts"""
import ast
import json
import pathlib
import random
import unittest

from integrations.botsystem.experts import escalation
from integrations.botsystem.experts.goal_spec import EXAMPLE_GOALS, GoalSpec, ItemRange, SpecError, encode, vocabulary
from integrations.botsystem.experts.registry import BudgetExceeded, ExpertRegistry, ExpertSpec, RegistryError, StepBudget
from integrations.botsystem.experts.router import TableRouter
from integrations.botsystem.protocol import ProtocolError

SPECS = list(EXAMPLE_GOALS)
IRON = next(s for s in SPECS if s.goal_id == "collect_iron")


class Fake:
    def __init__(self, tag):
        self.tag, self.seen = tag, []

    def act(self, x):
        self.seen.append(x)
        return self.tag


def registry(max_resident=2, budget=None):
    reg = ExpertRegistry(SPECS, max_resident=max_resident, budget=budget)
    made = {}
    for s in SPECS:
        reg.register(ExpertSpec(f"x_{s.goal_id}", s.goal_id, s.spec_hash(), lambda g=s.goal_id: made.setdefault(g, Fake(g))))
    return reg, made


class TestGoalSpec(unittest.TestCase):
    def test_sampling_is_deterministic_varied_and_startable(self):
        a = [IRON.sample(random.Random(5)).expert_input() for _ in range(3)]
        self.assertEqual(a[0], a[1])
        insts = [IRON.sample(random.Random(i)) for i in range(300)]
        self.assertTrue(all(i.inventory.get("stone_pickaxe", 0) >= 1 for i in insts), "`requires` is always held at the start")
        self.assertTrue(all(IRON.amount[0] <= i.amount <= IRON.amount[1] for i in insts))
        self.assertGreater(len({i.amount for i in insts}), 20, "the requested amount varies")
        self.assertGreater(len({tuple(sorted(i.inventory)) for i in insts}), 8, "the held items vary: many different starts for one goal")
        share = sum("coal" in i.inventory for i in insts) / len(insts)
        self.assertAlmostEqual(share, 0.6, delta=0.1)

    def test_expert_input_carries_no_provenance(self):
        inst = IRON.sample(random.Random(1))
        self.assertEqual(set(inst.expert_input()), {"goal_id", "amount", "inventory"})
        self.assertIn("seed", inst.meta); self.assertIn("handed_out", inst.meta)
        self.assertNotIn("seed", json.dumps(inst.expert_input()))

    def test_validation_is_closed(self):
        for bad in (lambda: ItemRange("Bad Item", 1, 2), lambda: ItemRange("dirt", 0, 2), lambda: ItemRange("dirt", 3, 2), lambda: ItemRange("dirt", 1, 2, 1.5),
                    lambda: GoalSpec("Bad", "dirt", (1, 2)), lambda: GoalSpec("g1", "dirt; op", (1, 2)), lambda: GoalSpec("g1", "dirt", (5, 2)),
                    lambda: GoalSpec("g1", "dirt", (1, 99999)), lambda: GoalSpec("g1", "dirt", (1, 2), initial=(ItemRange("a", 1, 2), ItemRange("a", 1, 3)))):
            with self.assertRaises(SpecError):
                bad()

    def test_hash_changes_with_the_distribution_only(self):
        same = GoalSpec(IRON.goal_id, IRON.target, IRON.amount, IRON.requires, tuple(reversed(IRON.initial)), IRON.max_steps)
        self.assertEqual(same.spec_hash(), IRON.spec_hash(), "order of the initial items does not matter")
        other = GoalSpec(IRON.goal_id, IRON.target, (1, 31), IRON.requires, IRON.initial, IRON.max_steps)
        self.assertNotEqual(other.spec_hash(), IRON.spec_hash())

    def test_goal_vector_has_a_fixed_length_and_drops_unknown_items(self):
        goals, items = vocabulary(SPECS)
        v1 = encode(SPECS, {"goal_id": "collect_iron", "amount": 8, "inventory": {"coal": 4, "totally_unknown_item": 9}})
        v2 = encode(SPECS, {"goal_id": "collect_wood", "amount": 64, "inventory": {}})
        self.assertEqual(len(v1), len(v2)); self.assertEqual(len(v1), len(goals) + 1 + len(items))
        self.assertEqual(sum(v1[:len(goals)]), 1.0)
        with self.assertRaises(SpecError):
            encode(SPECS, {"goal_id": "dig_to_bedrock", "amount": 1, "inventory": {}})


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.reg, _ = registry()
        self.r = TableRouter(SPECS, self.reg.expert_for())

    def test_routes_by_goal_and_refuses_what_it_cannot_serve(self):
        self.assertEqual(self.r.route("collect_iron", 8, {"stone_pickaxe": 1}), ("x_collect_iron", (), "ok"))
        self.assertEqual(self.r.route("collect_iron", 8, {}).reason, "missing_preconditions:stone_pickaxe")
        self.assertEqual(self.r.route("collect_iron", 33, {"stone_pickaxe": 1}).reason, "amount_out_of_range")
        self.assertEqual(self.r.route("collect_iron", 0, {"stone_pickaxe": 1}).reason, "amount_out_of_range")
        self.assertEqual(self.r.route("fly", 1, {}).reason, "unknown_goal")
        self.assertEqual(TableRouter(SPECS, {}).route("collect_wood", 1, {}).reason, "no_expert:collect_wood")

    def test_the_route_depends_on_the_request_only(self):
        # same request -> same answer, whatever was routed before (no hidden state, no outcome feedback)
        a = self.r.route("collect_iron", 8, {"stone_pickaxe": 1})
        for g in ("collect_wood", "collect_stone"):
            self.r.route(g, 2, {"wooden_pickaxe": 1})
        self.assertEqual(self.r.route("collect_iron", 8, {"stone_pickaxe": 1}), a)

    def test_setup_delegation_is_off_by_default_bounded_and_cycle_safe(self):
        ex = self.reg.expert_for()
        self.assertEqual(self.r.route("collect_iron", 2, {}).setup, ())
        d = TableRouter(SPECS, ex, delegate_setup=True)
        # stone_pickaxe is not the target of any goal here, so nothing provides it: still refused
        self.assertEqual(d.route("collect_iron", 2, {}).reason, "missing_preconditions:stone_pickaxe")
        chain = [GoalSpec("mine_cobble", "cobblestone", (1, 8)), GoalSpec("make_tool", "stone_pickaxe", (1, 1), requires=("cobblestone",)),
                 GoalSpec("get_iron", "raw_iron", (1, 8), requires=("stone_pickaxe",))]
        ex2 = {s.goal_id: f"x_{s.goal_id}" for s in chain}
        d2 = TableRouter(chain, ex2, delegate_setup=True, max_setup_depth=2)
        self.assertEqual(d2.route("get_iron", 3, {}), ("x_get_iron", ("mine_cobble", "make_tool"), "ok"), "deepest first")
        self.assertEqual(d2.route("get_iron", 3, {"cobblestone": 4}).setup, ("make_tool",))
        self.assertEqual(TableRouter(chain, ex2, delegate_setup=True, max_setup_depth=1).route("get_iron", 3, {}).reason, "missing_preconditions:cobblestone")
        loop = [GoalSpec("a_goal", "item_a", (1, 2), requires=("item_b",)), GoalSpec("b_goal", "item_b", (1, 2), requires=("item_a",))]
        self.assertEqual(TableRouter(loop, {s.goal_id: "e" for s in loop}, delegate_setup=True, max_setup_depth=5).route("a_goal", 1, {}).reason, "setup_cycle")

    def test_construction_errors(self):
        with self.assertRaises(ValueError):
            TableRouter(SPECS, {"nonexistent": "x"})
        with self.assertRaises(ValueError):
            TableRouter(SPECS + [SPECS[0]], {})


class TestRegistry(unittest.TestCase):
    def test_only_the_needed_expert_is_loaded_and_least_recently_used_leave(self):
        reg, made = registry(max_resident=2)
        self.assertEqual(made, {}, "nothing is loaded at registration")
        reg.act("x_collect_iron", {"goal_id": "collect_iron", "amount": 3, "inventory": {}})
        self.assertEqual(sorted(made), ["collect_iron"]); self.assertEqual(reg.loads, 1)
        reg.act("x_collect_iron", {"goal_id": "collect_iron", "amount": 4, "inventory": {}}); self.assertEqual(reg.loads, 1, "resident: no second load")
        reg.act("x_collect_wood", {"goal_id": "collect_wood", "amount": 4, "inventory": {}})
        reg.act("x_collect_stone", {"goal_id": "collect_stone", "amount": 4, "inventory": {}})
        self.assertEqual(reg.resident(), ["x_collect_wood", "x_collect_stone"]); self.assertEqual(reg.loads, 3)

    def test_expert_input_is_exactly_the_declared_keys(self):
        reg, made = registry()
        for bad in ({"goal_id": "collect_iron", "amount": 3, "inventory": {}, "world": [1]}, {"goal_id": "collect_iron", "amount": 3},
                    {"goal_id": "collect_iron", "amount": 3, "inventory": {}, "label": "success"}):
            with self.assertRaises(RegistryError):
                reg.act("x_collect_iron", bad)
        with self.assertRaises(RegistryError):
            reg.act("x_collect_iron", {"goal_id": "collect_wood", "amount": 3, "inventory": {}})
        self.assertEqual(made, {}, "refused calls load nothing")

    def test_an_expert_trained_on_another_distribution_is_refused(self):
        reg = ExpertRegistry(SPECS)
        changed = GoalSpec(IRON.goal_id, IRON.target, (1, 16), IRON.requires, IRON.initial)
        with self.assertRaises(RegistryError):
            reg.register(ExpertSpec("old", IRON.goal_id, changed.spec_hash(), lambda: Fake("x")))
        reg.register(ExpertSpec("ok", IRON.goal_id, IRON.spec_hash(), lambda: Fake("x")))
        with self.assertRaises(RegistryError):
            reg.register(ExpertSpec("ok", IRON.goal_id, IRON.spec_hash(), lambda: Fake("x")))
        reg.register(ExpertSpec("second", IRON.goal_id, IRON.spec_hash(), lambda: Fake("y")))
        with self.assertRaises(RegistryError):
            reg.expert_for()                                   # two experts for one goal is a v1 error

    def test_one_budget_counts_every_expert(self):
        b = StepBudget(limit=1000)
        b.charge("a", 600); b.charge("b", 400)
        self.assertEqual(b.total(), 1000); self.assertEqual(b.by_expert, {"a": 600, "b": 400})
        with self.assertRaises(BudgetExceeded):
            b.charge("a", 1)
        for bad in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                b.charge("a", bad)


class TestEscalation(unittest.TestCase):
    def setUp(self):
        reg, _ = registry()
        self.router = TableRouter(SPECS, reg.expert_for())

    def line(self, **over):
        m = {"v": 1, "type": "escalate", "run_id": "r1", "seq": 4, "goal_id": "collect_iron", "amount": 12, "inventory": {"stone_pickaxe": 1, "coal": 3}, "why": "amount too large for gather"}
        m.update(over)
        return (json.dumps(m) + "\n").encode()

    def test_round_trip(self):
        out = json.loads(escalation.answer(self.router, self.line(), run_id="r1"))
        self.assertEqual(out, {"v": 1, "type": "route", "run_id": "r1", "seq": 4, "expert": "x_collect_iron", "setup": [], "reason": "ok"})
        out = json.loads(escalation.answer(self.router, self.line(inventory={}), run_id="r1"))
        self.assertEqual((out["expert"], out["reason"]), (None, "missing_preconditions:stone_pickaxe"))

    def test_hostile_requests_are_refused(self):
        bad = [self.line(run_id="other"), self.line(type="act"), self.line(seq=-1), self.line(seq=True), self.line(goal_id="Bad Goal"), self.line(amount=0),
               self.line(amount=1.5), self.line(amount=10 ** 6), self.line(inventory={"x": -1}), self.line(inventory={"bad name": 1}), self.line(inventory={"a": 1.5}),
               self.line(inventory=[1]), self.line(inventory={f"i{i}": 1 for i in range(600)}), self.line(why="x" * 500), self.line(world_state=[1, 2]),
               b"not json\n", b'{"v":1,"type":"escalate","run_id":"r1","seq":1,"goal_id":"collect_iron","amount":NaN,"inventory":{}}\n', b"x" * (5 * 1024 * 1024)]
        for line in bad:
            with self.assertRaises(ProtocolError, msg=line[:80]):
                escalation.parse_request(line, run_id="r1")


class TestBoundary(unittest.TestCase):
    def test_standard_library_only(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        std = {"__future__", "collections", "dataclasses", "hashlib", "json", "random", "re", "typing"}
        for f in (root / "integrations/botsystem/experts").glob("*.py"):
            for node in ast.walk(ast.parse(f.read_text())):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
                for m in mods:
                    self.assertTrue(m.split(".")[0] in std, f"{f.name} imports {m}")
        # src/ never importing integrations/ is enforced for the whole package by tests.test_botsystem_interface.TestBoundary


if __name__ == "__main__":
    unittest.main()
