"""Experts on demand: only the expert a request needs is loaded, a few stay resident, and every env step any expert takes is counted in one
budget (EP-style counting across ALL copies, MASTER_GUIDANCE P7/P10: a router with N experts must not look cheaper than one policy by
hiding the steps of N-1 of them).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .goal_spec import GoalSpec


class Expert(Protocol):
    def act(self, expert_input: dict[str, Any]) -> Any: ...      # one decision from the goal and the inventory (nothing else)


class BudgetExceeded(RuntimeError):
    pass


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ExpertSpec:
    expert_id: str
    goal_id: str
    spec_hash: str                     # GoalSpec.spec_hash() of the start distribution it was trained on
    loader: Callable[[], Expert]       # loads the weights; called at most once while the expert stays resident


class StepBudget:
    """Env steps over every expert, trained or evaluated, in one count. `limit=None` only counts."""

    def __init__(self, limit: int | None = None):
        self.limit = limit
        self.by_expert: dict[str, int] = {}

    def charge(self, expert_id: str, steps: int) -> None:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if self.limit is not None and self.total() + steps > self.limit:
            raise BudgetExceeded(f"{self.total()} + {steps} steps exceed the budget of {self.limit}")
        self.by_expert[expert_id] = self.by_expert.get(expert_id, 0) + steps

    def total(self) -> int:
        return sum(self.by_expert.values())


class ExpertRegistry:
    def __init__(self, specs: list[GoalSpec], *, max_resident: int = 2, budget: StepBudget | None = None):
        if max_resident < 1:
            raise ValueError("max_resident >= 1")
        self.goals = {s.goal_id: s for s in specs}
        self.max_resident = max_resident
        self.budget = budget or StepBudget()
        self._specs: dict[str, ExpertSpec] = {}
        self._loaded: "OrderedDict[str, Expert]" = OrderedDict()
        self.loads = 0
        self.calls = 0

    def register(self, e: ExpertSpec) -> None:
        g = self.goals.get(e.goal_id)
        if g is None:
            raise RegistryError(f"expert {e.expert_id}: unknown goal {e.goal_id!r}")
        if e.spec_hash != g.spec_hash():
            raise RegistryError(f"expert {e.expert_id} was trained on start distribution {e.spec_hash}, the goal now says {g.spec_hash()}: "
                                "retrain, or register it with the hash it was trained on and decide on purpose")
        if e.expert_id in self._specs:
            raise RegistryError(f"expert {e.expert_id} twice")
        self._specs[e.expert_id] = e

    def expert_for(self) -> dict[str, str]:
        """goal id -> expert id, for the router (one expert per goal in v1)."""
        out: dict[str, str] = {}
        for e in self._specs.values():
            if e.goal_id in out:
                raise RegistryError(f"two experts for goal {e.goal_id}")
            out[e.goal_id] = e.expert_id
        return out

    def get(self, expert_id: str) -> Expert:
        spec = self._specs.get(expert_id)
        if spec is None:
            raise RegistryError(f"no expert {expert_id!r}")
        if expert_id in self._loaded:
            self._loaded.move_to_end(expert_id)
            return self._loaded[expert_id]
        while len(self._loaded) >= self.max_resident:
            self._loaded.popitem(last=False)             # the least recently used leaves memory
        self._loaded[expert_id] = spec.loader()
        self.loads += 1
        return self._loaded[expert_id]

    def act(self, expert_id: str, expert_input: dict[str, Any]) -> Any:
        """One decision. Only the declared keys reach the expert: a caller that adds the world state, a label or the outcome gets an error, not a leak."""
        if set(expert_input) != {"goal_id", "amount", "inventory"}:
            raise RegistryError(f"expert input must be exactly goal_id, amount, inventory; got {sorted(expert_input)}")
        if expert_input["goal_id"] != self._specs[expert_id].goal_id:
            raise RegistryError(f"expert {expert_id} answers goal {self._specs[expert_id].goal_id}, not {expert_input['goal_id']!r}")
        self.calls += 1
        return self.get(expert_id).act(expert_input)

    def resident(self) -> list[str]:
        return list(self._loaded)
