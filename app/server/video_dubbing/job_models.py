"""Dubbing job model + statuses.

Heavy dubbing operations (generate/refit/render/export/optimize) run as jobs so
the MCP/HTTP request returns immediately. The statuses mirror the spec.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

# Terminal statuses.
TERMINAL = frozenset(
    {
        "completed",
        "completed_with_errors",
        "failed",
        "cancelled",
        "interrupted",
    }
)


class DubbingJobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


@dataclass
class DubbingJob:
    job_id: str
    job_type: str
    project_id: str
    status: str = DubbingJobStatus.QUEUED
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    stage: str = ""
    progress_current: int = 0
    progress_total: int = 0
    message: str = ""
    run_id: str = ""
    operation_id: str = ""
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cancel_requested: bool = False
    # Internal: callable to actually cancel the running operation.
    cancel_handler: Callable[[], None] | None = field(default=None, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        percent = round((self.progress_current / self.progress_total) * 100, 2) if self.progress_total else 0.0
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "project_id": self.project_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": {
                "stage": self.stage,
                "current": self.progress_current,
                "total": self.progress_total,
                "percent": percent,
                "message": self.message,
            },
            "run_id": self.run_id,
            "operation_id": self.operation_id,
            "result": self.result,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
