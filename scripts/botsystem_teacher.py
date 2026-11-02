"""Scripted teacher for the BOTSYSTEM side track (owner 2026-11-02: "give a small amount of teacher data at the beginning ... then fade it out").

Hand-specified game knowledge, on purpose and labelled: nothing run through integrations/botsystem counts toward a DPOD gate (P4 does not
apply). It answers one question: given the observation and the progress-list entry the pair is after, which of the offered candidates
is the next step of the usual recipe? -1 = no opinion (the policy acts alone). It reads only what the policy reads (candidate command /
item ids, inventory, feat), never the world or the server, so it is an expert on the same interface, not an oracle.

Coverage: the wood and stone tier (log ... furnace, coal, torch). Everything above (iron, bed, diamond) is left to the policy.
"""
from __future__ import annotations

import numpy as np

CMDS = ["pad", "move_to", "mine", "attack", "pickup", "craft", "smelt", "place", "activate", "equip", "eat", "wait", "tunnel"]
LOGS = ("_log", "_stem")
# result -> materials (name -> count); a crafting table is needed for everything with "table": True
NEED = {
    "planks": ({"log": 1}, False), "stick": ({"planks": 2}, False), "crafting_table": ({"planks": 4}, False),
    "wooden_pickaxe": ({"planks": 3, "stick": 2}, True), "wooden_sword": ({"planks": 2, "stick": 1}, True),
    "stone_pickaxe": ({"cobblestone": 3, "stick": 2}, True), "stone_sword": ({"cobblestone": 2, "stick": 1}, True),
    "furnace": ({"cobblestone": 8}, True), "torch": ({"coal": 1, "stick": 1}, False),
}
# progress-list entry -> (kind, what)
GOALS = {
    "collect_wood": ("have", "log"), "craft_planks": ("craft", "planks"), "craft_crafting_table": ("craft", "crafting_table"),
    "place_crafting_table": ("place", "crafting_table"), "craft_stick": ("craft", "stick"),
    "craft_wooden_pickaxe": ("craft", "wooden_pickaxe"), "craft_wooden_sword": ("craft", "wooden_sword"),
    "collect_stone": ("have", "cobblestone"), "craft_stone_pickaxe": ("craft", "stone_pickaxe"),
    "craft_stone_sword": ("craft", "stone_sword"), "craft_furnace": ("craft", "furnace"), "place_furnace": ("place", "furnace"),
    "collect_coal": ("have", "coal"), "craft_torch": ("craft", "torch"),
}
PICKS = ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "golden_pickaxe")


def _is_log(n: str) -> bool:
    return n.endswith(LOGS)


def _is_planks(n: str) -> bool:
    return n.endswith("_planks")


class Teacher:
    def __init__(self, items: list[str]):
        self.items = items

    def name(self, i: int) -> str:
        return self.items[i] if 0 < i < len(self.items) else ""

    def inventory(self, obs: dict) -> dict:
        inv: dict = {}
        for i, c in zip(obs["inv_item"], obs["inv_count"]):
            n = self.name(int(i))
            if n:
                inv[n] = inv.get(n, 0) + int(round(float(c) * 64))
        out = dict(inv)
        out["log"] = sum(v for k, v in inv.items() if _is_log(k))
        out["planks"] = sum(v for k, v in inv.items() if _is_planks(k))
        out["coal"] = inv.get("coal", 0) + inv.get("charcoal", 0)
        out["cobblestone"] = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
        return out

    def cands(self, obs: dict) -> list[tuple[int, str, str, np.ndarray]]:
        """(index, command, item name, feat) of every offered candidate."""
        out = []
        for k in range(len(obs["cand_mask"])):
            if obs["cand_mask"][k] and int(obs["cand_cmd"][k]) > 0:
                out.append((k, CMDS[int(obs["cand_cmd"][k])], self.name(int(obs["cand_item"][k])), np.asarray(obs["cand_feat"][k])))
        return out

    @staticmethod
    def nearest(cs):
        return min(cs, key=lambda c: float(c[3][3]))[0] if cs else -1   # feat[3] = distance

    # -- the recipe, as a recursion over what is missing --
    def acquire(self, what: str, n: int, inv: dict, cs: list, depth: int = 0) -> int:
        """Next step towards holding n of `what` (a class name: log / planks / cobblestone / coal or an item)."""
        if depth > 4:
            return -1
        if inv.get(what, 0) >= n:
            return -1
        mine = lambda pred: self.nearest([c for c in cs if c[1] == "mine" and pred(c[2])])
        if what == "log":
            return mine(_is_log)
        if what == "coal":
            return mine(lambda s: s == "coal_ore" or s == "deepslate_coal_ore")
        if what == "cobblestone":
            if not any(inv.get(p, 0) for p in PICKS):
                return self.acquire("wooden_pickaxe", 1, inv, cs, depth + 1)
            j = mine(lambda s: s in ("stone", "cobblestone", "deepslate", "andesite", "diorite", "granite"))
            if j >= 0:
                return j
            down = [c for c in cs if c[1] == "tunnel" and float(c[3][1]) < 0]         # dig a stair down until stone is in reach
            return min(down, key=lambda c: float(c[3][1]))[0] if down else -1
        if what == "planks":
            crafts = [c for c in cs if c[1] == "craft" and _is_planks(c[2])]
            if crafts and inv.get("log", 0):
                return crafts[0][0]
            return self.acquire("log", 1, inv, cs, depth + 1)
        mats, table = NEED.get(what, ({}, False))
        if not mats:
            return -1
        crafts = [c for c in cs if c[1] == "craft" and c[2] == what]
        if crafts:
            return crafts[0][0]
        for m, c in mats.items():                                                    # missing materials first
            if inv.get(m, 0) < c:
                return self.acquire(m, c, inv, cs, depth + 1)
        if table:                                                                    # all held, no craft offered: no table in reach
            put = [c for c in cs if c[1] == "place" and c[2] == "crafting_table"]
            if put:
                return put[0][0]
            return self.acquire("crafting_table", 1, inv, cs, depth + 1)
        return -1

    def choose(self, obs: dict, goal: str) -> int:
        """Candidate index for progress-list entry `goal`, or -1."""
        spec = GOALS.get(goal)
        if spec is None:
            return -1
        kind, what = spec
        inv, cs = self.inventory(obs), self.cands(obs)
        if kind == "have":
            return self.acquire(what, inv.get(what, 0) + 1, inv, cs)
        if kind == "place":
            put = [c for c in cs if c[1] == "place" and c[2] == what]
            if put:
                return put[0][0]
            return self.acquire(what, 1, inv, cs)
        return self.acquire(what, inv.get(what, 0) + 1, inv, cs)
