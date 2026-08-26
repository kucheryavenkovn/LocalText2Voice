"""Append-only JSONL event sink.

Each event is written and flushed immediately so the journal stays usable
after an abrupt process death (segfault, ``kill -9``, power loss). Writes are
serialised by a process-wide lock to keep the file coherent when worker
threads emit concurrently.

A single :class:`JsonlEventSink` is bound as a contextvar by the service at the
start of an operation; deeply nested code (cue loop, FFmpeg runner) emits
events through :func:`emit_event` without receiving the sink explicitly.
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO

from .context import current_context
from .redaction import redact_value

# Force a flush on these event categories even if batched writes were added.
_FLUSH_LEVELS = {"error", "cancelled", "cue.persisted", "cue.cancelled", "cue.failed"}

_CURRENT_SINK: contextvars.ContextVar["JsonlEventSink | None"] = contextvars.ContextVar(
    "current_event_sink", default=None
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + ".%03dZ" % (
        int(time.time() * 1000) % 1000
    )


def set_event_sink(sink: "JsonlEventSink | None"):
    """Bind ``sink`` as the active sink for the current context."""
    return _CURRENT_SINK.set(sink)


@contextmanager
def use_event_sink(sink: "JsonlEventSink | None") -> Iterator[None]:
    """Bind ``sink`` for the duration of a ``with`` block (token-safe)."""
    token = _CURRENT_SINK.set(sink)
    try:
        yield
    finally:
        _CURRENT_SINK.reset(token)


def emit_event(
    event_type: str,
    *,
    payload: dict[str, Any] | None = None,
    sink: "JsonlEventSink | None" = None,
    force_flush: bool | None = None,
) -> None:
    """Emit one structured event to the active sink (if any).

    Never raises: diagnostic failures must not crash the pipeline.
    """
    target = sink or _CURRENT_SINK.get()
    if target is None:
        return
    try:
        flush = bool(force_flush) if force_flush is not None else event_type in _FLUSH_LEVELS
        target.write(event_type, payload or {}, force_flush=flush)
    except Exception:  # pragma: no cover - diagnostics must never raise
        pass


class JsonlEventSink:
    """Append-only ``events.jsonl`` writer.

    Each :meth:`write` call appends one JSON object followed by ``\\n`` and
    flushes the OS buffer so the record reaches disk. The current diagnostic
    context (run/operation/cue ids, thread name, monotonic clock) is merged
    into every record automatically.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._start_monotonic = time.monotonic()
        self._fh: TextIO | None = None
        self._open()

    def _open(self) -> None:
        # `` buffering=1`` = line buffered; combined with explicit flush this
        # keeps records durable across abnormal termination.
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)

    def write(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        force_flush: bool = False,
    ) -> None:
        record: dict[str, Any] = {
            "ts": _utc_now(),
            "elapsed_ms": int((time.monotonic() - self._start_monotonic) * 1000),
            "event": event_type,
            "thread": _current_thread_name(),
        }
        record.update(current_context())
        # Redact defensively so a stray secret in a payload never lands on disk.
        record["data"] = redact_value(payload) if payload else {}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            fh = self._fh
            if fh is None:
                self._open()
                fh = self._fh
            if fh is None:  # pragma: no cover
                return
            fh.write(line + "\n")
            if force_flush:
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except (OSError, ValueError):
                    pass

    def flush(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                finally:
                    self._fh.close()
                    self._fh = None

    def __enter__(self) -> "JsonlEventSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class SummaryWriter:
    """Writes ``summary.json`` for a run directory atomically."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}

    def update(self, **fields: Any) -> None:
        with self._lock:
            self._data.update(fields)

    def increment(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._data[key] = int(self._data.get(key, 0)) + amount

    def write(self) -> None:
        with self._lock:
            data = dict(self._data)
        tmp = self.path.with_suffix(".json.part")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            os.replace(tmp, self.path)
        except OSError:  # pragma: no cover - Windows share fallback
            try:
                self.path.unlink(missing_ok=True)
                os.replace(tmp, self.path)
            except OSError:
                tmp.replace(self.path)


def _current_thread_name() -> str:
    import threading as _t

    return _t.current_thread().name


__all__ = [
    "JsonlEventSink",
    "SummaryWriter",
    "emit_event",
    "set_event_sink",
    "use_event_sink",
]
