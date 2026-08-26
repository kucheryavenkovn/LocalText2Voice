"""Per-project locks.

* One mutating job may run per project at a time.
* Read-only operations acquire a shared read lock (non-blocking against the
  write lock holder).
* Lost updates are prevented by ``expected_revision`` checks at the facade
  level; the lock serialises writers.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from .errors import DubbingProjectBusyError


class ProjectLock:
    """Reentrant write lock + readers-writer semantics for one project."""

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self._write_lock = threading.RLock()
        self._readers = threading.Semaphore(0)
        self._reader_count = 0
        self._state_lock = threading.Lock()
        self.active_job_id = ""
        self.active_operation = ""
        self._holder = ""  # "ui" | "mcp" | "test" | "system"

    @contextmanager
    def write_lock(
        self,
        *,
        initiator: str = "mcp",
        active_job_id: str = "",
        active_operation: str = "",
    ) -> Iterator[None]:
        with self._write_lock:
            self._holder = initiator
            if active_job_id:
                self.active_job_id = active_job_id
                self.active_operation = active_operation
            try:
                yield
            finally:
                if active_job_id:
                    self.active_job_id = ""
                    self.active_operation = ""
                self._holder = ""

    @property
    def holder(self) -> str:
        return self._holder

    def is_busy(self) -> bool:
        # A held write lock with an active job means mutations are blocked.
        return bool(self.active_job_id)


class ProjectLockRegistry:
    """Owns one :class:`ProjectLock` per project id."""

    def __init__(self) -> None:
        self._locks: dict[str, ProjectLock] = {}
        self._registry_lock = threading.Lock()

    def get(self, project_id: str) -> ProjectLock:
        with self._registry_lock:
            lock = self._locks.get(project_id)
            if lock is None:
                lock = ProjectLock(project_id)
                self._locks[project_id] = lock
            return lock

    def ensure_idle(self, project_id: str) -> None:
        lock = self.get(project_id)
        if lock.is_busy():
            raise DubbingProjectBusyError(
                f"Project {project_id} has an active job: {lock.active_operation}.",
                active_job_id=lock.active_job_id,
                active_operation=lock.active_operation,
            )

    def active_job(self, project_id: str) -> dict[str, str]:
        lock = self.get(project_id)
        return {
            "active_job_id": lock.active_job_id,
            "active_operation": lock.active_operation,
        }
