"""Structured errors for the video-dubbing control plane.

These are raised by the facade and translated into safe JSON responses by the
MCP/HTTP layers. Every user-facing error carries a stable ``code`` and an
optional ``diagnostic_id`` linking to observability.
"""

from __future__ import annotations

import uuid
from typing import Any


class DubbingError(RuntimeError):
    """Base class. ``code`` is a stable machine identifier."""

    code = "operation_failed"
    http_status = 400

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        diagnostic_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message or code or self.code)
        self.message = message or self.code
        if code is not None:
            self.code = code
        self.diagnostic_id = diagnostic_id or uuid.uuid4().hex
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.code,
            "message": self.message,
            "diagnostic_id": self.diagnostic_id,
        }
        if self.details:
            payload.update(self.details)
        return payload


class DubbingNotFoundError(DubbingError):
    code = "not_found"
    http_status = 404


class DubbingValidationError(DubbingError):
    code = "validation_failed"
    http_status = 400


class DubbingProjectBusyError(DubbingError):
    code = "project_busy"
    http_status = 409

    def __init__(self, message: str = "Project is busy.", *, active_job_id: str = "", active_operation: str = "") -> None:
        super().__init__(
            message,
            details={
                "active_job_id": active_job_id,
                "active_operation": active_operation,
            },
        )


class DubbingProjectChangedError(DubbingError):
    code = "project_changed"
    http_status = 409

    def __init__(
        self,
        message: str = "Project revision changed.",
        *,
        expected_revision: int,
        current_revision: int,
    ) -> None:
        super().__init__(
            message,
            details={
                "expected_revision": expected_revision,
                "current_revision": current_revision,
            },
        )


class DubbingFaultNotAvailable(DubbingError):
    code = "fault_injection_unavailable"
    http_status = 403
