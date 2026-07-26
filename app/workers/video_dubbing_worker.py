from __future__ import annotations

import logging
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal, Slot

from app.core.video_dubbing.generation import (
    GenerationCancelled,
    GenerationRunResult,
    GenerationRunStatus,
)
from app.core.video_dubbing.service import (
    VideoDubbingService,
    VideoDubbingServiceError,
)
from app.observability import bind_context, capture_context, emit_event

_log = logging.getLogger("video_dubbing.worker")


class VideoDubbingWorker(QObject):
    """Runs a VideoDubbingService operation on a background thread.

    The operation is any callable taking the service and returning a value.
    Progress, per-cue updates, logs and cancel are wired through Qt signals so
    the UI stays responsive and is never touched from the worker thread.
    """

    progress = Signal(str, int, int, str)
    cue_updated = Signal(int, str, int, int, int)  # sequence, status, raw, fitted, index
    log = Signal(str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        service: VideoDubbingService,
        operation: Callable[[VideoDubbingService], Any],
    ) -> None:
        super().__init__()
        self.service = service
        self.operation = operation
        self._cancel_requested = False
        # Capture the diagnostic context (run_id/operation_id/project_id/...)
        # in the UI thread so it propagates into this worker thread. contextvars
        # do not cross thread boundaries automatically.
        self._context_snapshot = capture_context()
        # Wire the service callbacks to Qt signals (thread-safe queued connections).
        self.service.progress_callback = self.progress.emit
        self.service.log_callback = self.log.emit
        self.service.cue_updated_callback = self.cue_updated.emit

    @Slot()
    def run(self) -> None:
        # Re-apply the captured diagnostic context inside the worker thread.
        with bind_context(self._context_snapshot):
            emit_event("worker.started", payload={"thread": _current_thread_name()})
            try:
                if self._cancel_requested:
                    self.service.cancel()
                result = self.operation(self.service)
                if isinstance(result, GenerationRunResult):
                    if result.status == GenerationRunStatus.CANCELLED:
                        emit_event(
                            "worker.cancelled",
                            payload={"run_id": result.run_id},
                            force_flush=True,
                        )
                        self.cancelled.emit()
                        return
                    if result.status == GenerationRunStatus.FAILED:
                        emit_event(
                            "worker.failed",
                            payload={
                                "run_id": result.run_id,
                                "error": result.error_message,
                            },
                            force_flush=True,
                        )
                        self.failed.emit(
                            result.error_message or "Generation stopped due to error"
                        )
                        return
                self.finished.emit(result)
            except GenerationCancelled:
                emit_event("worker.cancelled", payload={}, force_flush=True)
                self.cancelled.emit()
            except VideoDubbingServiceError as exc:
                if "cancelled" in str(exc).casefold():
                    emit_event("worker.cancelled", payload={"reason": str(exc)}, force_flush=True)
                    self.cancelled.emit()
                else:
                    # logger.exception records the full traceback + diag context.
                    _log.exception("Video dubbing worker error")
                    emit_event(
                        "worker.failed",
                        payload={
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        force_flush=True,
                    )
                    self.failed.emit(str(exc))
            except Exception as exc:  # pragma: no cover - defensive
                # Never swallow worker exceptions silently: the full traceback
                # and active run_id are written to the JSONL + application log.
                _log.exception("Unexpected video dubbing worker error")
                emit_event(
                    "worker.crashed",
                    payload={
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    force_flush=True,
                )
                self.failed.emit(f"Unexpected video dubbing error: {exc}")
            finally:
                emit_event("worker.finished", payload={})

    def request_cancel(self) -> None:
        self._cancel_requested = True
        try:
            self.service.cancel()
        except Exception as exc:  # pragma: no cover - defensive
            # cancel() failures must never be swallowed without a record.
            _log.exception("worker.request_cancel failed")
            emit_event(
                "worker.cancel_error",
                payload={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
                force_flush=True,
            )


def _current_thread_name() -> str:
    import threading as _t

    return _t.current_thread().name
