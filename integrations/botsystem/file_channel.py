"""Append-only JSON-lines channel: one writer, one reader, a file each way. Also the audit trail (`tail -f` works).

The writer emits each message with a single `os.write` on an `O_APPEND` descriptor, so a reader sees either no part or
the whole line on a local filesystem. The reader only consumes bytes up to the last newline, so a half-written line is
picked up on the next poll. A line that fails validation is counted and skipped, never fatal; a file that shrinks is
an error (`ChannelError`) because the offset would be meaningless.

Limits: this is for one machine, or a filesystem that keeps append and read-after-write order. A synced folder
(Nextcloud, SMB) does not promise that; use an HTTP transport with the same message schema there (PROTOCOL.md).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .protocol import MAX_LINE_BYTES, ProtocolError, dumps, loads

log = logging.getLogger("dpod.botsystem")


class ChannelError(RuntimeError):
    pass


class LineWriter:
    def __init__(self, path: Path, *, fsync: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
        self._fsync = fsync

    def send(self, msg: dict[str, Any]) -> None:
        data = dumps(msg)
        if len(data) > MAX_LINE_BYTES:
            raise ProtocolError(f"outgoing message of {len(data)} bytes exceeds {MAX_LINE_BYTES}")
        if os.write(self._fd, data) != len(data):  # regular files do not short-write; refuse to continue if one does
            raise ChannelError("short write")
        if self._fsync:
            os.fsync(self._fd)

    def close(self) -> None:
        os.close(self._fd)


class LineReader:
    def __init__(self, path: Path, *, run_id: str, allowed: tuple[str, ...]) -> None:
        self.path, self._run_id, self._allowed = path, run_id, allowed
        self._offset = 0
        self.rejected = 0

    def poll(self) -> list[dict[str, Any]]:
        """New valid messages since the last call (possibly none)."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return []
        if size < self._offset:
            raise ChannelError(f"{self.path} shrank from {self._offset} to {size} bytes")
        if size == self._offset:
            return []
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            chunk = f.read(size - self._offset)
        end = chunk.rfind(b"\n") + 1
        out = []
        for line in chunk[:end].splitlines():
            if not line.strip():
                continue
            try:
                out.append(loads(line, run_id=self._run_id, allowed=self._allowed))
            except ProtocolError as e:
                self.rejected += 1
                log.warning("rejected line in %s at ~%d: %s | %r", self.path.name, self._offset, e, line[:120])
        self._offset += end
        if end == 0 and len(chunk) > MAX_LINE_BYTES:
            raise ChannelError(f"{self.path}: no newline within {MAX_LINE_BYTES} bytes")
        return out
