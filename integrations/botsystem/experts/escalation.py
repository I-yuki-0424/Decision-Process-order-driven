"""The escalation messages (BOT mechanism <-> router), one JSON object per line like the rest of the file API (protocol.py).

    request  {v, type: "escalate", run_id, seq, goal_id, amount, inventory: {item: count}, why}   bot mechanism -> router
    decision {v, type: "route",    run_id, seq, expert, setup: [goal_id], reason}                  router -> bot mechanism

`why` is free text for the log only and never reaches an expert. Everything in a request is untrusted (the bot's inventory comes from a
game server): sizes capped, names closed, counts bounded integers.
"""
from __future__ import annotations

import json
from typing import Any

from ..protocol import MAX_LINE_BYTES, ProtocolError, VERSION, dumps
from .goal_spec import GOAL_ID, MAX_AMOUNT, NAME
from .router import Route, TableRouter

MAX_INVENTORY_ITEMS = 512
MAX_WHY = 200


def parse_request(line: bytes, *, run_id: str) -> dict[str, Any]:
    if len(line) > MAX_LINE_BYTES:
        raise ProtocolError("line too long")
    try:
        msg = json.loads(line, parse_constant=lambda c: (_ for _ in ()).throw(ProtocolError(f"non-finite number {c}")))
    except (ValueError, RecursionError) as e:
        raise ProtocolError(f"not JSON: {e}") from e
    if not isinstance(msg, dict) or msg.get("v") != VERSION or msg.get("type") != "escalate" or msg.get("run_id") != run_id:
        raise ProtocolError("not a v1 escalate message of this run")
    if set(msg) - {"v", "type", "run_id", "seq", "goal_id", "amount", "inventory", "why"}:
        raise ProtocolError(f"undeclared fields {sorted(set(msg) - {'v', 'type', 'run_id', 'seq', 'goal_id', 'amount', 'inventory', 'why'})}")
    seq, amount = msg.get("seq"), msg.get("amount")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise ProtocolError("seq must be a non-negative integer")
    if not isinstance(msg.get("goal_id"), str) or not GOAL_ID.match(msg["goal_id"]):
        raise ProtocolError("bad goal_id")
    if not isinstance(amount, int) or isinstance(amount, bool) or not 1 <= amount <= MAX_AMOUNT:
        raise ProtocolError(f"amount must be an integer in 1..{MAX_AMOUNT}")
    inv = msg.get("inventory")
    if not isinstance(inv, dict) or len(inv) > MAX_INVENTORY_ITEMS:
        raise ProtocolError("inventory must be an object of at most %d items" % MAX_INVENTORY_ITEMS)
    for k, v in inv.items():
        if not NAME.match(k) or not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 1_000_000:
            raise ProtocolError(f"bad inventory entry {k!r}")
    why = msg.get("why", "")
    if not isinstance(why, str) or len(why) > MAX_WHY:
        raise ProtocolError("why must be a string of at most %d characters" % MAX_WHY)
    return msg


def decision(run_id: str, seq: int, route: Route) -> bytes:
    return dumps({"v": VERSION, "type": "route", "run_id": run_id, "seq": seq, "expert": route.expert, "setup": list(route.setup), "reason": route.reason})


def answer(router: TableRouter, line: bytes, *, run_id: str) -> bytes:
    """One request line in, one decision line out. A malformed request is a ProtocolError for the caller to log, never a guess."""
    msg = parse_request(line, run_id=run_id)
    return decision(run_id, msg["seq"], router.route(msg["goal_id"], msg["amount"], msg["inventory"]))
