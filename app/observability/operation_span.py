"""Operation spans — structured replacements for ``_begin_operation``/``_end_operation``.

Usage::

    with operation_span("render_narration", project_id=..., run_id=...):
        ...

A span automatically records ``started``/``completed``/``failed``/``cancelled``
events with duration, exception type, message and full traceback. Spans nest:
a child span inherits the parent's ``operation_id``.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Any, Iterator

from .context import bind, new_operation_id
from .events import emit_event

_log = logging.getLogger("video_dubbing.spans")

# Per-thread span stack so nested spans know their parent without threading an
# explicit parent id through every call.
_span_stack = threading.local()


def _current_operation_id() -> str:
    stack = getattr(_span_stack, "stack", None)
    if stack:
        return stack[-1]
    return ""


@contextmanager
def operation_span(
    name: str,
    *,
    project_id: str | None = None,
    run_id: str | None = None,
    stage: str | None = None,
    extra: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> Iterator[dict[str, Any]]:
    """Open a diagnostic span around ``name``.

    Yields a mutable ``info`` dict the body can enrich (e.g. result counts).
    On normal exit emits ``<name>.completed``; on :class:`Exception` it emits
    ``<name>.failed`` with the full traceback and re-raises. Subclasses of
    cancellation are reported as ``<name>.cancelled``.
    """

    log = logger or _log
    operation_id = new_operation_id()
    parent_operation_id = _current_operation_id()
    stack = getattr(_span_stack, "stack", None)
    if stack is None:
        stack = []
        _span_stack.stack = stack
    stack.append(operation_id)

    bindings: dict[str, Any] = {
        "operation_id": operation_id,
        "parent_operation_id": parent_operation_id,
        "stage": stage or name,
    }
    if project_id is not None:
        bindings["project_id"] = project_id
    if run_id is not None:
        bindings["run_id"] = run_id
    started_payload = {"name": name, "operation_id": operation_id}
    if extra:
        started_payload.update(extra)

    t0 = time.perf_counter()
    log.info("OPERATION START: %s %s", name, _fmt_kv(extra))
    emit_event(f"{name}.started", payload=started_payload, force_flush=False)
    info: dict[str, Any] = {"name": name, "operation_id": operation_id}
    cancelled = False
    try:
        with bind(**bindings):
            yield info
    except Exception as exc:  # noqa: BLE001 - span must classify every failure
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        cancelled = _is_cancellation(exc)
        event_type = f"{name}.cancelled" if cancelled else f"{name}.failed"
        payload = {
            "name": name,
            "operation_id": operation_id,
            "duration_ms": elapsed_ms,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": tb_text,
        }
        if info:
            payload.update({k: v for k, v in info.items() if k not in {"name", "operation_id"}})
        emit_event(event_type, payload=payload, force_flush=True)
        if cancelled:
            log.warning("OPERATION CANCELLED: %s (%d ms)", name, elapsed_ms)
        else:
            log.error("OPERATION FAILED: %s (%d ms): %s", name, elapsed_ms, exc)
        raise
    else:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        info["duration_ms"] = elapsed_ms
        payload = {"name": name, "operation_id": operation_id, "duration_ms": elapsed_ms}
        payload.update({k: v for k, v in info.items() if k not in {"name", "operation_id"}})
        emit_event(f"{name}.completed", payload=payload, force_flush=False)
        log.info("OPERATION END: %s (%d ms)", name, elapsed_ms)
        return info
    finally:
        stack.pop()


def _is_cancellation(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"GenerationCancelled", "CueGenerationCancelled", "VideoMuxCancelled",
                "AudioMixerCancelled", "PreviewRenderCancelled", "TimelineRenderCancelled",
                "TTSCancelled", "FFmpegCancelled"}:
        return True
    msg = str(exc).casefold()
    return "cancel" in msg


def _fmt_kv(extra: dict[str, Any] | None) -> str:
    if not extra:
        return ""
    return " ".join(f"{k}={v}" for k, v in extra.items())


__all__ = ["operation_span"]
