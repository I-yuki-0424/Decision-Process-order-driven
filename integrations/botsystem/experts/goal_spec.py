"""One goal = one skill policy's job, with a STANDARDISED distribution of start conditions.

The owner's point (forge1201 LLM-records 2026-11-06, benefit 3): a goal such as iron is normally trained after earlier goals, so the held
items are almost always one pattern and the preconditions are never set up for training. Here a goal declares the distribution of
(requested amount, held items) its expert is trained on, so the same goal can be run again and again from many different starts.

What the policy may see is `Instance.expert_input()`: the goal, the requested amount and the inventory it actually holds. The sampling seed,
the spec and the fact that items were handed out at the start are provenance, kept in `Instance.meta` and never given to the policy
(DESIGN.md, review point 1).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

NAME = re.compile(r"^([a-z0-9_.-]{1,32}:)?[a-z0-9_/]{1,64}$")     # same closed form as the bot's item names (src/skills TYPES.item)
GOAL_ID = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
MAX_AMOUNT = 2304                                                  # the action layer's `count` limit


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class ItemRange:
    """`item` is present with probability `p`, its count uniform in [lo, hi]."""
    item: str
    lo: int
    hi: int
    p: float = 1.0

    def __post_init__(self):
        if not NAME.match(self.item):
            raise SpecError(f"bad item name {self.item!r}")
        if not (isinstance(self.lo, int) and isinstance(self.hi, int)) or isinstance(self.lo, bool) or not 1 <= self.lo <= self.hi <= MAX_AMOUNT:
            raise SpecError(f"{self.item}: need 1 <= lo <= hi <= {MAX_AMOUNT}")
        if not 0.0 <= self.p <= 1.0:
            raise SpecError(f"{self.item}: p must be in [0, 1]")


@dataclass(frozen=True)
class GoalSpec:
    goal_id: str
    target: str                       # the item whose count in the inventory is the goal's progress
    amount: tuple[int, int]           # inclusive range of requested counts
    requires: tuple[str, ...] = ()    # items that must be held to start (read from the inventory, nothing else)
    initial: tuple[ItemRange, ...] = ()
    max_steps: int = 600              # episode cap of the expert, in policy steps
    provisional: bool = True          # the numbers are a first design, owner to set (DESIGN.md, open questions)

    def __post_init__(self):
        if not GOAL_ID.match(self.goal_id):
            raise SpecError(f"bad goal id {self.goal_id!r}")
        if not NAME.match(self.target):
            raise SpecError(f"bad target {self.target!r}")
        lo, hi = self.amount
        if not (isinstance(lo, int) and isinstance(hi, int)) or not 1 <= lo <= hi <= MAX_AMOUNT:
            raise SpecError(f"{self.goal_id}: amount needs 1 <= lo <= hi <= {MAX_AMOUNT}")
        for r in self.requires:
            if not NAME.match(r):
                raise SpecError(f"bad required item {r!r}")
        if not 1 <= self.max_steps <= 100000:
            raise SpecError("max_steps 1..100000")
        names = [r.item for r in self.initial]
        if len(set(names)) != len(names):
            raise SpecError(f"{self.goal_id}: an item twice in `initial`")

    def canonical(self) -> dict[str, Any]:
        return {"goal_id": self.goal_id, "target": self.target, "amount": list(self.amount), "requires": sorted(self.requires),
                "initial": [[r.item, r.lo, r.hi, r.p] for r in sorted(self.initial, key=lambda r: r.item)], "max_steps": self.max_steps}

    def spec_hash(self) -> str:
        """What an expert was trained on. The registry refuses to serve an expert under a different hash (no silent change of the
        precondition distribution between training and deployment)."""
        return hashlib.sha256(json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    def sample(self, rng: random.Random) -> "Instance":
        """One training start. The inventory holds `requires` (at least one of each, so the goal is startable) plus the `initial` draw."""
        inv: dict[str, int] = {}
        for r in self.initial:
            if rng.random() < r.p:
                inv[r.item] = rng.randint(r.lo, r.hi)
        for item in self.requires:
            inv.setdefault(item, 1)
        seed = rng.getrandbits(32)
        return Instance(self.goal_id, rng.randint(*self.amount), inv, {"seed": seed, "spec_hash": self.spec_hash(), "handed_out": sorted(inv)})


@dataclass(frozen=True)
class Instance:
    goal_id: str
    amount: int
    inventory: dict[str, int]
    meta: dict[str, Any] = field(default_factory=dict)       # provenance: NEVER part of the expert's input

    def expert_input(self) -> dict[str, Any]:
        return {"goal_id": self.goal_id, "amount": self.amount, "inventory": dict(sorted(self.inventory.items()))}


def vocabulary(specs: list[GoalSpec]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(goal ids, item names) in a fixed order: the common goal-vector schema every expert and the router share."""
    goals = tuple(sorted(s.goal_id for s in specs))
    items = tuple(sorted({s.target for s in specs} | {i for s in specs for i in s.requires} | {r.item for s in specs for r in s.initial}))
    return goals, items


def encode(specs: list[GoalSpec], inp: dict[str, Any]) -> list[float]:
    """Goal vector: one-hot goal | amount / the largest amount of any goal | inventory counts (clipped at 64, /64) in vocabulary order.
    Fixed length for a given set of specs; an item outside the vocabulary is dropped (the policy cannot use what it has no name for)."""
    goals, items = vocabulary(specs)
    if inp.get("goal_id") not in goals:
        raise SpecError(f"unknown goal {inp.get('goal_id')!r}")
    top = max(s.amount[1] for s in specs)
    inv = inp.get("inventory", {})
    return ([1.0 if g == inp["goal_id"] else 0.0 for g in goals] + [min(1.0, float(inp["amount"]) / top)]
            + [min(int(inv.get(i, 0)), 64) / 64.0 for i in items])


# First design of three goals. PROVISIONAL: the numbers are placeholders for the owner's decision (DESIGN.md, open question 4); they
# only make the interface testable and show the form. Item names are Minecraft 1.20.1 ids as the bot uses them.
EXAMPLE_GOALS: tuple[GoalSpec, ...] = (
    GoalSpec("collect_wood", "oak_log", (1, 64)),
    GoalSpec("collect_stone", "cobblestone", (1, 64), requires=("wooden_pickaxe",), initial=(ItemRange("stick", 1, 16, 0.5),)),
    GoalSpec("collect_iron", "raw_iron", (1, 32), requires=("stone_pickaxe",),
             initial=(ItemRange("cobblestone", 1, 64, 0.8), ItemRange("coal", 1, 16, 0.6), ItemRange("torch", 1, 16, 0.5), ItemRange("bread", 1, 8, 0.5))),
)
