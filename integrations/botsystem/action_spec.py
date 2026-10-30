"""Closed, data-defined action space for the BOTSYSTEM (Mineflayer / Forge 1.20.1) environment.

Why data-defined: the action set is not DPOD's 17 Craftax actions and is not final. BOTSYSTEM publishes its spec in the
`hello` message (PROTOCOL.md); DPOD parses it with `ActionSpec.from_wire` and refuses a spec it does not expect. Nothing
else in this package hard-codes an action name.

One action = one value per component, held for `ticks_per_step` game ticks:
  Key     off | tap | hold   a key or mouse button. `tap` is down on the first tick only, `hold` on every tick.
                             Joint use of consecutive steps ("held across steps") is just `hold` repeated.
  Axis    one of `values`    a discrete delta applied once, at the first tick (look yaw / pitch, degrees).
  Select  0 .. n-1           a choice applied once, at the first tick (hotbar slot; 0 = leave unchanged).
Rules (`exclusive`, `requires`) remove joint combinations the game cannot or should not receive. A key counts as pressed
when its mode is not `off`.

The full space is a product of components (millions of combinations), so the primary representation is factored
(`nvec`, like a MultiDiscrete). `iter_valid` enumerates a flat list only for small specs (`proposal_min`).

Standard library only, so the BOTSYSTEM side and the tests can import it without JAX.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from typing import Any, Iterator

KEY_MODES = ("off", "tap", "hold")
RULE_KINDS = ("exclusive", "requires")


class ActionError(ValueError):
    """A spec or an action violates the closed action-space definition."""


@dataclass(frozen=True)
class Key:
    name: str
    modes: tuple[str, ...] = KEY_MODES


@dataclass(frozen=True)
class Axis:
    name: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class Select:
    name: str
    n: int


@dataclass(frozen=True)
class Rule:
    kind: str  # exclusive: not (a and b); requires: a implies b. Both name Key components.
    a: str
    b: str


def _domain(c: Key | Axis | Select) -> tuple:
    if isinstance(c, Key):
        return c.modes
    if isinstance(c, Axis):
        return c.values
    return tuple(range(c.n))


@dataclass(frozen=True)
class ActionSpec:
    name: str
    version: int
    ticks_per_step: int
    components: tuple[Key | Axis | Select, ...]
    rules: tuple[Rule, ...] = ()

    def __post_init__(self) -> None:
        if self.ticks_per_step < 1:
            raise ActionError("ticks_per_step must be >= 1")
        names = [c.name for c in self.components]
        if not names or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
            raise ActionError("component names must be non-empty and unique")
        for c in self.components:
            if isinstance(c, Key):
                if "off" not in c.modes or len(set(c.modes)) != len(c.modes) or not set(c.modes) <= set(KEY_MODES):
                    raise ActionError(f"{c.name}: modes must be unique, contain 'off', and be within {KEY_MODES}")
                if "tap" in c.modes and self.ticks_per_step == 1:
                    raise ActionError(f"{c.name}: 'tap' equals 'hold' at ticks_per_step=1; remove it")
            elif isinstance(c, Axis):
                if not c.values or len(set(c.values)) != len(c.values) or not all(math.isfinite(v) for v in c.values):
                    raise ActionError(f"{c.name}: axis values must be non-empty, unique and finite")
            elif c.n < 1:
                raise ActionError(f"{c.name}: select needs n >= 1")
        keys = {c.name for c in self.components if isinstance(c, Key)}
        for r in self.rules:
            if r.kind not in RULE_KINDS or r.a not in keys or r.b not in keys or r.a == r.b:
                raise ActionError(f"bad rule {r}")

    # --- factored view -------------------------------------------------------------------------------------------
    @property
    def nvec(self) -> tuple[int, ...]:
        return tuple(len(_domain(c)) for c in self.components)

    @property
    def size(self) -> int:
        """Size of the product space, rule-violating combinations included."""
        return math.prod(self.nvec)

    def encode(self, action: dict[str, Any]) -> tuple[int, ...]:
        self.validate(action)
        return tuple(_domain(c).index(action[c.name]) for c in self.components)

    def decode(self, idx: tuple[int, ...], *, check: bool = True) -> dict[str, Any]:
        """Index tuple -> action. `check=False` skips the rule check (for rejection sampling over the product space)."""
        if len(idx) != len(self.components) or any(not 0 <= i < n for i, n in zip(idx, self.nvec)):
            raise ActionError(f"index {idx} outside nvec {self.nvec}")
        action = {c.name: _domain(c)[i] for c, i in zip(self.components, idx)}
        if check:
            self.validate(action)
        return action

    def noop(self) -> dict[str, Any]:
        """Keys released, axes at 0 (or the first value if 0 is absent), selects at 0. Always rule-satisfying."""
        return {c.name: "off" if isinstance(c, Key) else (0.0 if isinstance(c, Axis) and 0.0 in c.values else _domain(c)[0])
                for c in self.components}

    def validate(self, action: dict[str, Any]) -> None:
        if set(action) != {c.name for c in self.components}:
            raise ActionError(f"action keys {sorted(action)} != components {[c.name for c in self.components]}")
        for c in self.components:
            v = action[c.name]
            if isinstance(v, bool) or v not in _domain(c):
                raise ActionError(f"{c.name}={v!r} not in {_domain(c)}")
        for r in self.rules:
            a, b = action[r.a] != "off", action[r.b] != "off"
            if (r.kind == "exclusive" and a and b) or (r.kind == "requires" and a and not b):
                raise ActionError(f"rule {r.kind}({r.a}, {r.b}) violated")

    def is_valid(self, action: dict[str, Any]) -> bool:
        try:
            self.validate(action)
        except ActionError:
            return False
        return True

    def iter_valid(self, max_size: int = 200_000) -> Iterator[dict[str, Any]]:
        """Every rule-satisfying action, in a stable order. Refuses specs whose product exceeds `max_size`."""
        if self.size > max_size:
            raise ActionError(f"{self.name}: product space {self.size} > {max_size}; use the factored view")
        names = [c.name for c in self.components]
        for combo in itertools.product(*(_domain(c) for c in self.components)):
            action = dict(zip(names, combo))
            if self.is_valid(action):
                yield action

    # --- reference semantics for the environment side --------------------------------------------------------------
    def key_schedule(self, action: dict[str, Any]) -> list[dict[str, bool]]:
        """Per tick, which keys are down. This is what the environment must feed to setControlState."""
        self.validate(action)
        keys = [c.name for c in self.components if isinstance(c, Key)]
        down = {"off": lambda t: False, "tap": lambda t: t == 0, "hold": lambda t: True}
        return [{k: down[action[k]](t) for k in keys} for t in range(self.ticks_per_step)]

    # --- wire form -------------------------------------------------------------------------------------------------
    def to_wire(self) -> dict[str, Any]:
        comps = []
        for c in self.components:
            if isinstance(c, Key):
                comps.append({"kind": "key", "name": c.name, "modes": list(c.modes)})
            elif isinstance(c, Axis):
                comps.append({"kind": "axis", "name": c.name, "values": list(c.values)})
            else:
                comps.append({"kind": "select", "name": c.name, "n": c.n})
        return {"name": self.name, "version": self.version, "ticks_per_step": self.ticks_per_step,
                "components": comps, "rules": [{"kind": r.kind, "a": r.a, "b": r.b} for r in self.rules]}

    @classmethod
    def from_wire(cls, d: Any) -> "ActionSpec":
        try:
            comps: list[Key | Axis | Select] = []
            for c in d["components"]:
                if c["kind"] == "key":
                    comps.append(Key(c["name"], tuple(c["modes"])))
                elif c["kind"] == "axis":
                    comps.append(Axis(c["name"], tuple(float(v) for v in c["values"])))
                elif c["kind"] == "select":
                    comps.append(Select(c["name"], int(c["n"])))
                else:
                    raise ActionError(f"unknown component kind {c['kind']!r}")
            rules = tuple(Rule(r["kind"], r["a"], r["b"]) for r in d["rules"])
            return cls(str(d["name"]), int(d["version"]), int(d["ticks_per_step"]), tuple(comps), rules)
        except (KeyError, TypeError, ValueError) as e:  # ActionError is a ValueError
            raise ActionError(f"malformed action spec: {e}") from e

    def spec_hash(self) -> str:
        """sha256 of a canonical text form that is easy to reproduce in any language (axis values as `%.6g`)."""
        w = self.to_wire()
        for c in w["components"]:
            if c["kind"] == "axis":
                c["values"] = [format(v, ".6g") for v in c["values"]]
        return hashlib.sha256(json.dumps(w, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_MOVE_RULES = (Rule("exclusive", "forward", "back"), Rule("exclusive", "left", "right"),
               Rule("requires", "sprint", "forward"), Rule("exclusive", "sprint", "sneak"))


def proposal_full(ticks_per_step: int = 4) -> ActionSpec:
    """PROPOSAL, not agreed with BOTSYSTEM. Every control Mineflayer exposes as a control state, plus look and hotbar.

    `ticks_per_step=4` is a placeholder: the real value follows from the measured server/bot speed (BOTSYSTEM task 007).
    The product space is about 3 million combinations, so policies must use factored heads, not a flat softmax.
    """
    keys = tuple(Key(n) for n in ("forward", "back", "left", "right", "jump", "sprint", "sneak", "attack", "use"))
    return ActionSpec("botsystem_full", 0, ticks_per_step,
                      keys + (Axis("yaw_delta_deg", (-45.0, -15.0, 0.0, 15.0, 45.0)),
                              Axis("pitch_delta_deg", (-15.0, 0.0, 15.0)), Select("hotbar", 10)),
                      _MOVE_RULES)


def proposal_min(ticks_per_step: int = 4) -> ActionSpec:
    """PROPOSAL. Small enough to enumerate flat (DPOD's candidate-set interface) for first end-to-end runs."""
    keys = (Key("forward"), Key("jump"), Key("sprint", ("off", "hold")), Key("attack", ("off", "hold")))
    return ActionSpec("botsystem_min", 0, ticks_per_step,
                      keys + (Axis("yaw_delta_deg", (-30.0, 0.0, 30.0)),),
                      (Rule("requires", "sprint", "forward"),))
