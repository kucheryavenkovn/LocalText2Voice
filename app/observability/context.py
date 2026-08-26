"""Diagnostic context propagated through the pipeline via ``contextvars``.

Context flows from the UI thread into worker threads explicitly: the caller
captures a snapshot with :func:`capture_context` and re-applies it inside the
worker with :func:`bind_context`. Every event emitted anywhere downstream then
carries ``run_id``, ``operation_id``, ``cue_id`` and so on without threading
those values through every function signature.
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol


_APP_SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "app_session_id", default=""
)
_PROJECT_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "project_id", default=""
)
_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="")
_OPERATION_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "operation_id", default=""
)
_PARENT_OPERATION_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "parent_operation_id", default=""
)
_CUE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("cue_id", default="")
_CUE_SEQUENCE: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "cue_sequence", default=None
)
_ENGINE_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "engine_id", default=""
)
_ENGINE_CLASS: contextvars.ContextVar[str] = contextvars.ContextVar(
    "engine_class", default=""
)
_ATTEMPT: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "attempt", default=None
)
_STAGE: contextvars.ContextVar[str] = contextvars.ContextVar("stage", default="")

_VAR_MAP = {
    "app_session_id": _APP_SESSION_ID,
    "project_id": _PROJECT_ID,
    "run_id": _RUN_ID,
    "operation_id": _OPERATION_ID,
    "parent_operation_id": _PARENT_OPERATION_ID,
    "cue_id": _CUE_ID,
    "cue_sequence": _CUE_SEQUENCE,
    "engine_id": _ENGINE_ID,
    "engine_class": _ENGINE_CLASS,
    "attempt": _ATTEMPT,
    "stage": _STAGE,
}


class _Rng(Protocol):
    def hex(self) -> str: ...


_id_lock = threading.Lock()


def new_session_id() -> str:
    """Stable id for one application process (set once at startup)."""
    from uuid import uuid4

    return uuid4().hex


def new_operation_id() -> str:
    from uuid import uuid4

    return uuid4().hex


def set_app_session_id(session_id: str) -> None:
    _APP_SESSION_ID.set(session_id or "")


@dataclass
class DiagnosticContext:
    """A serialisable snapshot of the active diagnostic context."""

    app_session_id: str = ""
    project_id: str = ""
    run_id: str = ""
    operation_id: str = ""
    parent_operation_id: str = ""
    cue_id: str = ""
    cue_sequence: Any = None
    engine_id: str = ""
    engine_class: str = ""
    attempt: Any = None
    stage: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key, var in _VAR_MAP.items():
            value = var.get()
            if value in (None, "", []):
                continue
            data[key] = value
        if self.extra:
            for key, value in self.extra.items():
                if value not in (None, "", []):
                    data.setdefault(key, value)
        return data


def capture_context() -> dict[str, Any]:
    """Return the currently-bound context as a plain dict.

    The result is safe to hand to another thread (e.g. a ``QThread`` worker)
    and re-apply there with :func:`bind_context`.
    """
    data: dict[str, Any] = {}
    for key, var in _VAR_MAP.items():
        value = var.get()
        if value not in (None, "", []):
            data[key] = value
    return data


def current_context() -> dict[str, Any]:
    """Alias of :func:`capture_context` for readability at call sites."""
    return capture_context()


@contextmanager
def bind(**values: Any) -> Iterator[None]:
    """Bind diagnostic values for the duration of the ``with`` block.

    Restores the previous values on exit so spans nest cleanly. Returns a list
    of ``contextvars`` tokens used to reset state.
    """
    tokens: list[tuple[contextvars.ContextVar[Any], contextvars.Token[Any]]] = []
    try:
        for key, value in values.items():
            var = _VAR_MAP.get(key)
            if var is None:
                continue
            tokens.append((var, var.set(value)))
        yield
    finally:
        for var, token in reversed(tokens):
            try:
                var.reset(token)
            except (ValueError, LookupError):
                var.set(var.default)


@contextmanager
def bind_context(snapshot: dict[str, Any]) -> Iterator[None]:
    """Re-apply a snapshot captured by :func:`capture_context` (cross-thread)."""
    with bind(**{k: v for k, v in snapshot.items() if k in _VAR_MAP}) as _:
        yield


__all__ = [
    "DiagnosticContext",
    "bind",
    "bind_context",
    "capture_context",
    "current_context",
    "new_operation_id",
    "new_session_id",
    "set_app_session_id",
]
