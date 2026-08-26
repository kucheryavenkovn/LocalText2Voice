"""Observability layer for the video-dubbing pipeline.

This package provides a cohesive diagnostic infrastructure:

* :mod:`context`           - request/operation context propagated via contextvars
* :mod:`operation_span`    - context-manager spans replacing ``_begin_operation``
* :mod:`events`            - append-only JSONL event sink (crash-safe)
* :mod:`logging_config`    - rotating application log + crash dump bootstrap
* :mod:`subprocess_diagnostics` - FFmpeg/FFprobe runner with full stderr capture
* :mod:`resource_snapshot` - RSS/RAM/disk/GPU snapshots
* :mod:`redaction`         - secret + subtitle-text redaction
* :mod:`crash_reporter`    - excepthooks, faulthandler, operation watchdog
* :mod:`diagnostic_bundle` - per-run directory layout (events.jsonl, subprocess/)

The goal is a single, structured source of truth for diagnostics so the
pipeline never resorts to scattered ``print()``/``log_callback()`` calls for
forensic information.
"""

from __future__ import annotations

from .context import (
    DiagnosticContext,
    bind,
    bind_context,
    capture_context,
    current_context,
    new_operation_id,
    new_session_id,
    set_app_session_id,
)
from .crash_reporter import (
    OperationWatchdog,
    install_crash_handlers,
)
from .diagnostic_bundle import RunDirectory
from .events import JsonlEventSink, SummaryWriter, emit_event, set_event_sink, use_event_sink
from .logging_config import configure_observability_logging
from .operation_span import operation_span
from .redaction import redact_command, redact_text, sha256_text
from .resource_snapshot import ResourceLimit, log_resource_snapshot, snapshot_resources
from .subprocess_diagnostics import DiagnosedSubprocess, SubprocessResult

__all__ = [
    "DiagnosticContext",
    "RunDirectory",
    "JsonlEventSink",
    "SummaryWriter",
    "OperationWatchdog",
    "DiagnosedSubprocess",
    "SubprocessResult",
    "ResourceLimit",
    "bind",
    "bind_context",
    "capture_context",
    "configure_observability_logging",
    "current_context",
    "emit_event",
    "install_crash_handlers",
    "log_resource_snapshot",
    "new_operation_id",
    "new_session_id",
    "operation_span",
    "redact_command",
    "redact_text",
    "set_app_session_id",
    "set_event_sink",
    "sha256_text",
    "snapshot_resources",
    "use_event_sink",
]
