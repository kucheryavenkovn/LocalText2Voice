"""Crash reporting + operation watchdog.

This module replaces the old ``logging_utils`` bootstrap which called
``faulthandler.dump_traceback_later(30)`` unconditionally — that produced a
spurious stack dump every 30 s even during normal long-running operations.

Instead we install:

* ``faulthandler.enable`` (native segfault → crash file + stderr);
* signal handlers for SIGSEGV/SIGFPE/SIGABRT/... (where available);
* ``sys.excepthook`` + ``threading.excepthook`` routed through ``logger``;
* a Qt message handler (installed separately by the caller after QApplication).

The long-stall detection becomes an **opt-in watchdog**
(:class:`OperationWatchdog`) armed only for active operations, fed by
heartbeats and cancelled on completion.
"""

from __future__ import annotations

import faulthandler
import logging
import signal
import sys
import threading
import traceback
from pathlib import Path
from typing import TextIO

from .events import emit_event

_log = logging.getLogger("crash")


def install_crash_handlers(
    crash_file: Path | None = None,
    *,
    enable_faulthandler: bool = True,
) -> None:
    """Install crash diagnostics.

    ``crash_file`` is opened in append mode and used both for ``faulthandler``
    native dumps and for uncaught exception tracebacks.
    """
    crash_handle: TextIO | None = None
    if crash_file is not None:
        crash_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            crash_handle = crash_file.open("a", encoding="utf-8", buffering=1)
        except OSError:
            crash_handle = None

    if enable_faulthandler:
        try:
            faulthandler.enable(file=crash_handle if crash_handle else sys.__stderr__)
        except Exception:  # pragma: no cover
            pass
        _register_fatal_signals(crash_handle)

    _install_python_excepthooks(crash_handle)


def _register_fatal_signals(crash_handle: TextIO | None) -> None:
    target = crash_handle if crash_handle else sys.__stderr__
    for signame in ("SIGABRT", "SIGFPE", "SIGILL", "SIGSEGV", "SIGBUS"):
        signum = getattr(signal, signame, None)
        if signum is None:
            continue
        try:
            faulthandler.register(signum, file=target, all_threads=True, chain=False)
        except (ValueError, OSError, RuntimeError, AttributeError):
            pass


def _install_python_excepthooks(crash_handle: TextIO | None) -> None:
    def _write_crash(marker: str, text: str) -> None:
        _log.critical("%s:\n%s", marker, text)
        if crash_handle is not None:
            try:
                crash_handle.write(f"\n===== {marker} =====\n" + text + "\n")
                crash_handle.flush()
            except OSError:  # pragma: no cover
                pass

    def excepthook(exc_type, exc_value, exc_tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        _write_crash("UNCAUGHT EXCEPTION", text)

    def threading_excepthook(args) -> None:
        text = "".join(
            traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
        )
        thread_name = getattr(args.thread, "name", "?")
        _write_crash(f"UNCAUGHT EXCEPTION (thread {thread_name})", text)
        emit_event(
            "crash.thread_exception",
            payload={
                "thread": thread_name,
                "exception_type": str(getattr(args.exc_type, "__name__", args.exc_type)),
                "exception_message": str(args.exc_value),
                "traceback": text,
            },
            force_flush=True,
        )

    sys.excepthook = excepthook
    try:
        threading.excepthook = threading_excepthook
    except AttributeError:  # pragma: no cover
        pass


class OperationWatchdog:
    """Detects operation stalls *only* while armed.

    Usage::

        watchdog = OperationWatchdog(timeout_seconds=120, on_stall=callback)
        with watchdog:
            for chunk in work:
                do(chunk)
                watchdog.heartbeat()  # resets the stall timer

    ``on_stall`` defaults to dumping the traceback of all threads to stderr
    and emitting an ``operation.stall_detected`` event. The watchdog never
    kills the process: a Python thread cannot be safely killed; it merely
    surfaces the stall diagnostically. It is a deliberate replacement for the
    previous unconditional ``faulthandler.dump_traceback_later(30)``.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float,
        on_stall=None,
        label: str = "operation",
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.label = label
        self._on_stall = on_stall
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._armed = False

    def arm(self) -> None:
        with self._lock:
            self._armed = True
            self._schedule()

    def heartbeat(self) -> None:
        with self._lock:
            if not self._armed:
                return
            self._schedule()

    def cancel(self) -> None:
        with self._lock:
            self._armed = False
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def __enter__(self) -> "OperationWatchdog":
        self.arm()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cancel()
        return False

    def _schedule(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        timer = threading.Timer(self.timeout_seconds, self._fire)
        timer.daemon = True
        timer.start()
        self._timer = timer

    def _fire(self) -> None:
        with self._lock:
            if not self._armed:
                return
            # Stay armed: a single dump does not disengage the watchdog.
        try:
            dump = self._dump_traceback()
            emit_event(
                "operation.stall_detected",
                payload={
                    "label": self.label,
                    "timeout_seconds": self.timeout_seconds,
                    "traceback": dump,
                },
                force_flush=True,
            )
            _log.error(
                "operation.stall_detected: %s (no heartbeat for %ss)",
                self.label,
                self.timeout_seconds,
            )
            if self._on_stall is not None:
                try:
                    self._on_stall()
                except Exception:  # pragma: no cover
                    pass
        except Exception:  # pragma: no cover - watchdog must never crash
            pass
        # Re-arm so the watchdog keeps firing until cancelled/heartbeat.
        with self._lock:
            if self._armed:
                self._schedule()

    @staticmethod
    def _dump_traceback() -> str:
        import io

        buf = io.StringIO()
        try:
            faulthandler.dump_traceback(file=buf, all_threads=True)
        except Exception:  # pragma: no cover
            pass
        return buf.getvalue()


__all__ = [
    "OperationWatchdog",
    "install_crash_handlers",
]
