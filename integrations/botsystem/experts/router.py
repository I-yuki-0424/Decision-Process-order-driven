"""The router: a request from the BOT mechanism (a goal it cannot carry out with its command sequences) -> which expert answers.

v1 is a TABLE, not a model. How the router should be learned is an open question (DESIGN.md); the interface below is what a learned router
has to keep. Nothing in `route` reads the world, an episode label or the outcome of any expert: only the request (goal, amount, the
inventory the bot reports). A router trained on "which expert would have succeeded here" would need counterfactual branching of the
environment, which the project rules forbid at decision time (P3) and which would make the router an oracle.
"""
from __future__ import annotations

from typing import NamedTuple

from .goal_spec import GoalSpec


class Route(NamedTuple):
    expert: str | None            # expert id to call now, or None
    setup: tuple[str, ...]        # goal ids to run FIRST (delegated precondition set-up), in order; empty unless delegation is on
    reason: str                   # 'ok' | 'unknown_goal' | 'amount_out_of_range' | 'missing_preconditions:<items>' | 'no_expert:<goal>' | 'setup_cycle'


class TableRouter:
    def __init__(self, specs: list[GoalSpec], expert_for: dict[str, str], *, delegate_setup: bool = False, max_setup_depth: int = 2):
        self.specs = {s.goal_id: s for s in specs}
        if len(self.specs) != len(specs):
            raise ValueError("a goal id twice")
        for g in expert_for:
            if g not in self.specs:
                raise ValueError(f"expert for unknown goal {g!r}")
        self.expert_for = dict(expert_for)
        self.delegate_setup = delegate_setup              # off by default: the open question "may the router order set-up?" is the owner's
        self.max_setup_depth = max_setup_depth
        self._provides = {s.target: s.goal_id for s in specs}

    def route(self, goal_id: str, amount: int, inventory: dict[str, int]) -> Route:
        spec = self.specs.get(goal_id)
        if spec is None:
            return Route(None, (), "unknown_goal")
        if not spec.amount[0] <= amount <= spec.amount[1]:
            return Route(None, (), "amount_out_of_range")             # the caller splits the request; the expert was not trained for it
        if goal_id not in self.expert_for:
            return Route(None, (), f"no_expert:{goal_id}")
        missing = tuple(r for r in spec.requires if inventory.get(r, 0) < 1)
        if not missing:
            return Route(self.expert_for[goal_id], (), "ok")
        if not self.delegate_setup:
            return Route(None, (), "missing_preconditions:" + ",".join(missing))
        chain: list[str] = []
        ok, why = self._plan_setup(missing, inventory, chain, 0, {goal_id})
        if not ok:
            return Route(None, (), why)
        return Route(self.expert_for[goal_id], tuple(chain), "ok")

    def _plan_setup(self, missing: tuple[str, ...], inv: dict[str, int], chain: list[str], depth: int, seen: set[str]) -> tuple[bool, str]:
        """Goals (by their target item) that provide what is missing, deepest first. Bounded depth; a goal needed by itself is a cycle."""
        if depth >= self.max_setup_depth:
            return False, "missing_preconditions:" + ",".join(missing)
        for item in missing:
            g = self._provides.get(item)
            if g is None or g not in self.expert_for:
                return False, f"missing_preconditions:{item}"
            if g in seen:
                return False, "setup_cycle"
            have = {self.specs[x].target for x in chain}
            sub = tuple(r for r in self.specs[g].requires if inv.get(r, 0) < 1 and r not in have)
            if sub:
                ok, why = self._plan_setup(sub, inv, chain, depth + 1, seen | {g})
                if not ok:
                    return False, why
            if g not in chain:
                chain.append(g)
        return True, "ok"
