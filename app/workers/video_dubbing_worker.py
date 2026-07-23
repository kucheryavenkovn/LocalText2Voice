from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal, Slot

from app.core.video_dubbing.service import (
    VideoDubbingService,
    VideoDubbingServiceError,
)


class VideoDubbingWorker(QObject):
    """Runs a VideoDubbingService operation on a background thread.

    The operation is any callable taking the service and returning a value.
    Progress/log/cancel are wired through Qt signals so the UI stays responsive.
    """

    progress = Signal(str, int, int, str)
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
        self.service.progress_callback = self.progress.emit
        self.service.log_callback = self.log.emit

    @Slot()
    def run(self) -> None:
        try:
            if self._cancel_requested:
                self.service.cancel()
            result = self.operation(self.service)
            self.finished.emit(result)
        except VideoDubbingServiceError as exc:
            if "cancelled" in str(exc).casefold():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            traceback.print_exc()
            self.failed.emit(f"Unexpected video dubbing error: {exc}")

    def request_cancel(self) -> None:
        self._cancel_requested = True
        try:
            self.service.cancel()
        except Exception:  # pragma: no cover - defensive
            pass
