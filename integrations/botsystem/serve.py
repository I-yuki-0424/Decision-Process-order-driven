"""Policy side of the file API: read the environment's observations, write actions.

    <root>/<run_id>/env_to_policy.jsonl   written by BOTSYSTEM: hello, obs, bye
    <root>/<run_id>/policy_to_env.jsonl   written here:         hello_ack, act

The server is restartable: on start it reads its own output and skips every (episode, step) it already answered, so a
crash or a reboot never produces a second, different action for the same observation.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Protocol

from .action_spec import ActionError, ActionSpec
from .file_channel import LineReader, LineWriter
from .protocol import ENV_TYPES, POLICY_TYPES, VERSION, ProtocolError, check_obs, parse_obs_spec

log = logging.getLogger("dpod.botsystem")


class PolicyError(RuntimeError):
    """The policy produced an action outside the spec. Stops the server instead of sending a guess."""


class Policy(Protocol):
    def bind(self, spec: ActionSpec) -> None: ...
    def act(self, msg: dict[str, Any]) -> dict[str, Any]: ...   # msg: the validated obs message
    def end_episode(self, msg: dict[str, Any]) -> None: ...     # msg: the terminal obs message


class NoopPolicy:
    """Release everything. Control for environment smoke tests."""
    def bind(self, spec: ActionSpec) -> None:
        self._spec = spec

    def act(self, msg: dict[str, Any]) -> dict[str, Any]:
        return self._spec.noop()

    def end_episode(self, msg: dict[str, Any]) -> None:
        pass


class UniformRandomPolicy(NoopPolicy):
    """Uniform over rule-satisfying actions (rejection sampling). The reference every learned policy must beat."""
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def act(self, msg: dict[str, Any]) -> dict[str, Any]:
        for _ in range(1000):
            a = self._spec.decode(tuple(self._rng.randrange(n) for n in self._spec.nvec), check=False)
            if self._spec.is_valid(a):
                return a
        return self._spec.noop()

    def end_episode(self, msg: dict[str, Any]) -> None:
        pass


class PolicyServer:
    def __init__(self, root: Path, run_id: str, policy: Policy, *, expected_spec: ActionSpec | None = None) -> None:
        d = root / run_id
        self.run_id, self.policy, self.expected = run_id, policy, expected_spec
        self._in = LineReader(d / "env_to_policy.jsonl", run_id=run_id, allowed=ENV_TYPES)
        out = d / "policy_to_env.jsonl"
        self._answered = {(m["episode"], m["step"]) for m in
                          LineReader(out, run_id=run_id, allowed=POLICY_TYPES).poll() if m["type"] == "act"}
        self._out = LineWriter(out)
        self._seq = 0
        self.spec: ActionSpec | None = None
        self._obs_spec: dict | None = None
        self.refused: str | None = None
        self.finished = False

    def _send(self, env_id: int, type_: str, **fields: Any) -> None:
        self._out.send({"v": VERSION, "type": type_, "run_id": self.run_id, "env_id": env_id, "seq": self._seq,
                        "t": round(time.time(), 3), **fields})
        self._seq += 1

    def poll_once(self) -> int:
        """Handle every message that has arrived. Returns how many acts were written."""
        acts = 0
        for msg in self._in.poll():
            if msg["type"] == "bye":
                self.finished = True
            elif msg["type"] == "hello":
                self._on_hello(msg)
            elif self.spec is None:
                log.warning("obs seq=%d before an accepted hello; ignored", msg["seq"])
            else:
                acts += self._on_obs(msg)
        return acts

    def _on_hello(self, msg: dict[str, Any]) -> None:
        try:
            spec = ActionSpec.from_wire(msg.get("action_spec"))
            obs_spec = parse_obs_spec(msg.get("obs_spec"))
            if self.expected is not None and spec != self.expected:
                raise ProtocolError(f"action spec {spec.name} v{spec.version} differs from expected "
                                    f"{self.expected.name} v{self.expected.version}")
        except ValueError as e:  # ActionError and ProtocolError are ValueErrors
            self.refused = str(e)
            log.error("hello refused: %s", e)
            self._send(msg["env_id"], "hello_ack", accepted=False, reason=str(e))
            return
        self.spec, self._obs_spec, self.refused = spec, obs_spec, None
        self.policy.bind(spec)
        log.info("hello accepted: %s v%d, %d ticks/step, nvec=%s", spec.name, spec.version, spec.ticks_per_step, spec.nvec)
        self._send(msg["env_id"], "hello_ack", accepted=True, action_spec_hash=spec.spec_hash())

    def _on_obs(self, msg: dict[str, Any]) -> int:
        assert self.spec is not None and self._obs_spec is not None
        try:
            check_obs(msg, self._obs_spec)
        except ProtocolError as e:
            self._in.rejected += 1
            log.warning("obs seq=%d rejected: %s", msg["seq"], e)
            return 0
        key = (msg["episode"], msg["step"])
        if msg["terminated"] or msg["truncated"]:
            if key not in self._answered:
                self._answered.add(key)
                self.policy.end_episode(msg)
            return 0
        if key in self._answered:
            return 0
        t0 = time.perf_counter()
        action = self.policy.act(msg)
        try:
            self.spec.validate(action)
        except ActionError as e:
            raise PolicyError(f"{type(self.policy).__name__} produced an invalid action at {key}: {e}") from e
        self._answered.add(key)
        self._send(msg["env_id"], "act", episode=msg["episode"], step=msg["step"], obs_seq=msg["seq"], action=action,
                   latency_ms=round((time.perf_counter() - t0) * 1000, 3))
        return 1

    def run(self, *, poll_s: float = 0.02, idle_timeout_s: float | None = None) -> None:
        last = time.monotonic()
        while not self.finished:
            if self.poll_once():
                last = time.monotonic()
            elif idle_timeout_s is not None and time.monotonic() - last > idle_timeout_s:
                log.warning("no observation for %.0fs; stopping", idle_timeout_s)
                return
            else:
                time.sleep(poll_s)
        log.info("bye received")

    def close(self) -> None:
        self._out.close()
