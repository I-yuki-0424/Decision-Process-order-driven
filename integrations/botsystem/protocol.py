"""Message schema of the file API (v1). The normative description is PROTOCOL.md; this module enforces it.

Everything read from the environment is untrusted data: sizes are capped, NaN/Infinity are rejected, observation fields
must be exactly the ones declared in `hello` (an undeclared field is a possible leak, so it is refused, not ignored).
"""
from __future__ import annotations

import json
import math
from typing import Any

VERSION = 1
MAX_LINE_BYTES = 4 * 1024 * 1024
ENV_TYPES = ("hello", "obs", "bye")       # environment -> policy
POLICY_TYPES = ("hello_ack", "act")       # policy -> environment
DTYPES = ("float", "int", "bool")


class ProtocolError(ValueError):
    pass


def dumps(msg: dict[str, Any]) -> bytes:
    """One message as one newline-terminated JSON line. NaN/Infinity are refused so the line stays portable."""
    return (json.dumps(msg, separators=(",", ":"), allow_nan=False, ensure_ascii=True) + "\n").encode()


def _no_constant(name: str) -> Any:
    raise ProtocolError(f"non-finite number {name}")


def loads(line: bytes, *, run_id: str, allowed: tuple[str, ...]) -> dict[str, Any]:
    if len(line) > MAX_LINE_BYTES:
        raise ProtocolError(f"line of {len(line)} bytes exceeds {MAX_LINE_BYTES}")
    try:
        msg = json.loads(line, parse_constant=_no_constant)
    except (ValueError, RecursionError) as e:
        raise ProtocolError(f"not JSON: {e}") from e
    if not isinstance(msg, dict) or msg.get("v") != VERSION:
        raise ProtocolError(f"not a v{VERSION} object")
    if msg.get("type") not in allowed:
        raise ProtocolError(f"type {msg.get('type')!r} not in {allowed}")
    if msg.get("run_id") != run_id:
        raise ProtocolError(f"run_id {msg.get('run_id')!r} != {run_id!r}")
    for k in ("seq", "env_id"):
        if not isinstance(msg.get(k), int) or isinstance(msg.get(k), bool) or msg[k] < 0:
            raise ProtocolError(f"{k} must be a non-negative integer")
    return msg


def _shape_ok(value: Any, shape: tuple[int, ...], dtype: str) -> bool:
    if not shape:
        if dtype == "bool":
            return isinstance(value, bool)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return math.isfinite(value) and (dtype == "float" or isinstance(value, int))
    return (isinstance(value, list) and len(value) == shape[0]
            and all(_shape_ok(v, shape[1:], dtype) for v in value))


def parse_obs_spec(d: Any) -> dict[str, tuple[tuple[int, ...], str]]:
    """`{"fields": [{"name", "shape", "dtype"}]}` -> {name: (shape, dtype)}."""
    try:
        out: dict[str, tuple[tuple[int, ...], str]] = {}
        for f in d["fields"]:
            shape = tuple(f["shape"])
            if (f["dtype"] not in DTYPES or not f["name"] or f["name"] in out
                    or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in shape)):
                raise ProtocolError(f"bad observation field {f!r}")
            out[f["name"]] = (shape, f["dtype"])
        if not out:
            raise ProtocolError("observation spec has no fields")
        return out
    except (KeyError, TypeError) as e:
        raise ProtocolError(f"malformed observation spec: {e}") from e


def check_obs(msg: dict[str, Any], obs_spec: dict[str, tuple[tuple[int, ...], str]]) -> None:
    for k in ("episode", "step"):
        if not isinstance(msg.get(k), int) or isinstance(msg.get(k), bool) or msg[k] < 0:
            raise ProtocolError(f"{k} must be a non-negative integer")
    for k in ("terminated", "truncated"):
        if not isinstance(msg.get(k), bool):
            raise ProtocolError(f"{k} must be a boolean")
    reward = msg.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ProtocolError("reward must be a finite number")
    obs = msg.get("obs")
    if not isinstance(obs, dict) or set(obs) != set(obs_spec):
        raise ProtocolError(f"obs fields {sorted(obs) if isinstance(obs, dict) else obs!r} != declared {sorted(obs_spec)}")
    for name, (shape, dtype) in obs_spec.items():
        if not _shape_ok(obs[name], shape, dtype):
            raise ProtocolError(f"obs field {name!r} does not match shape {shape} dtype {dtype}")
    terms = msg.get("reward_terms", {})
    if not isinstance(terms, dict) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                                              for v in terms.values()):
        raise ProtocolError("reward_terms must map names to finite numbers")
