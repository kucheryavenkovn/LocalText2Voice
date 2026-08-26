"""Dubbing job manager.

A thread-pool job runner specialised for video-dubbing operations. It is
distinct from :class:`LocalServerJobManager` (audiobook jobs) but follows the
same patterns: persistent state in SQLite, progress callbacks, cooperative
cancellation via a callable + cancel event, and a bounded worker pool.

The manager records project_id + job_type + run_id so the facade can enforce
"one mutating job per project".
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from app.observability import bind, emit_event, new_operation_id
from app.utils.paths import app_data_root

from .job_models import DubbingJob, DubbingJobStatus, TERMINAL, utc_now
from .project_locks import ProjectLockRegistry


ProgressFn = Callable[[str, int, int, str], None]
JobFn = Callable[[DubbingJob, ProgressFn, threading.Event], Any]


class DubbingJobManager:
    DB_SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: Path | None = None,
        max_parallel_jobs: int = 1,
        locks: ProjectLockRegistry | None = None,
    ) -> None:
        self.db_path = db_path or app_data_root() / "server" / "video_dubbing_jobs.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_parallel_jobs = max(1, int(max_parallel_jobs))
        self.locks = locks or ProjectLockRegistry()
        self._db_lock = threading.RLock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._workers: list[threading.Thread] = []
        self._jobs: dict[str, DubbingJob] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._cancel_handlers: dict[str, Callable[[], None]] = {}
        self._active = threading.RLock()
        self._ensure_schema()
        self._ensure_workers()

    # ------------------------------------------------------------------ submit

    def submit(
        self,
        project_id: str,
        job_type: str,
        fn: JobFn,
        *,
        run_id: str | None = None,
        operation_id: str | None = None,
    ) -> DubbingJob:
        job_id = uuid.uuid4().hex
        operation_id = operation_id or new_operation_id()
        run_id = run_id or uuid.uuid4().hex
        job = DubbingJob(
            job_id=job_id,
            job_type=job_type,
            project_id=project_id,
            created_at=utc_now(),
            run_id=run_id,
            operation_id=operation_id,
        )
        with self._active:
            self._jobs[job_id] = job
            self._cancel_events[job_id] = job.cancel_event
        self._persist(job)
        emit_event(
            "job.queued",
            payload={"job_id": job_id, "job_type": job_type, "project_id": project_id, "run_id": run_id},
            force_flush=False,
        )
        self._queue.put(job_id)
        return job

    # ------------------------------------------------------------------ query

    def get_job(self, job_id: str) -> DubbingJob | None:
        with self._active:
            if job_id in self._jobs:
                return self._jobs[job_id]
        row = self._load_row(job_id)
        return self._row_to_job(row) if row else None

    def list_jobs(
        self,
        *,
        project_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[DubbingJob]:
        limit = max(1, min(500, int(limit)))
        # Combine in-memory + persisted, de-duped, newest first.
        seen: set[str] = set()
        out: list[DubbingJob] = []
        with self._active:
            for job in sorted(
                self._jobs.values(), key=lambda j: j.created_at, reverse=True
            ):
                if project_id and job.project_id != project_id:
                    continue
                if status and job.status != status:
                    continue
                seen.add(job.job_id)
                out.append(job)
        for row in self._load_rows(project_id=project_id, status=status, limit=limit * 2):
            jid = str(row["job_id"])
            if jid in seen:
                continue
            seen.add(jid)
            out.append(self._row_to_job(row))
        out.sort(key=lambda j: j.created_at, reverse=True)
        return out[:limit]

    # ------------------------------------------------------------------ cancel

    def cancel(self, job_id: str) -> DubbingJob | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        if job.status in TERMINAL:
            return job
        job.cancel_requested = True
        if job.status == DubbingJobStatus.QUEUED:
            self._finish(job, DubbingJobStatus.CANCELLED, message="Cancelled before start.")
            return job
        with self._active:
            event = self._cancel_events.get(job_id)
            handler = self._cancel_handlers.get(job_id)
        if event is not None:
            event.set()
        if handler is not None:
            try:
                handler()
            except Exception:
                pass
        emit_event(
            "job.cancel_requested",
            payload={"job_id": job_id, "job_type": job.job_type, "project_id": job.project_id},
            force_flush=True,
        )
        return self.get_job(job_id)

    def register_cancel_handler(self, job_id: str, handler: Callable[[], None]) -> None:
        with self._active:
            self._cancel_handlers[job_id] = handler

    def wait_for_job(self, job_id: str, timeout_seconds: float = 3600.0) -> DubbingJob | None:
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while time.monotonic() < deadline:
            job = self.get_job(job_id)
            if job is None or job.status in TERMINAL:
                return job
            time.sleep(0.1)
        return self.get_job(job_id)

    # ------------------------------------------------------------------ workers

    def _ensure_workers(self) -> None:
        if self._workers and any(t.is_alive() for t in self._workers):
            return
        self._stop.clear()
        self._workers = [
            threading.Thread(
                target=self._worker_loop,
                name=f"DubbingJobWorker-{i}",
                daemon=True,
            )
            for i in range(self.max_parallel_jobs)
        ]
        for thread in self._workers:
            thread.start()

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job_id is None:
                break
            try:
                self._run(job_id)
            finally:
                self._queue.task_done()

    def _run(self, job_id: str) -> None:
        with self._active:
            job = self._jobs.get(job_id)
            if job is None:
                return
            cancel_event = self._cancel_events.get(job_id)
        if job is None:
            return
        if job.cancel_requested:
            self._finish(job, DubbingJobStatus.CANCELLED, message="Cancelled before start.")
            return
        job.status = DubbingJobStatus.RUNNING
        job.started_at = utc_now()
        self._persist(job)
        emit_event(
            "job.started",
            payload={"job_id": job_id, "job_type": job.job_type, "project_id": job.project_id},
        )

        def progress(stage: str, current: int, total: int, message: str) -> None:
            job.stage = stage
            job.progress_current = max(0, int(current))
            job.progress_total = max(0, int(total))
            job.message = message
            self._persist(job)
            emit_event(
                "job.progress",
                payload={
                    "job_id": job_id,
                    "stage": stage,
                    "current": job.progress_current,
                    "total": job.progress_total,
                },
                force_flush=False,
            )

        try:
            with bind(job_id=job_id, run_id=job.run_id, operation_id=job.operation_id):
                result = self._dispatch(job, progress, cancel_event)
            if cancel_event.is_set():
                self._finish(job, DubbingJobStatus.CANCELLED, message="Cancelled.")
            else:
                job.result = result if isinstance(result, dict) else {"value": result}
                # Caller may signal partial success with errors.
                final_status = DubbingJobStatus.COMPLETED
                if isinstance(result, dict) and result.get("status") == "completed_with_errors":
                    final_status = DubbingJobStatus.COMPLETED_WITH_ERRORS
                self._finish(job, final_status, message="Completed.")
        except _JobCancelled:
            self._finish(job, DubbingJobStatus.CANCELLED, message="Cancelled.")
        except Exception as exc:
            traceback.print_exc()
            job.error = {
                "code": "operation_failed",
                "message": _safe_message(exc),
                "exception_type": type(exc).__name__,
            }
            self._finish(job, DubbingJobStatus.FAILED, message=_safe_message(exc))
        finally:
            with self._active:
                self._cancel_handlers.pop(job_id, None)

    def _dispatch(self, job: DubbingJob, progress: ProgressFn, cancel_event: threading.Event) -> Any:
        fn = getattr(job, "_fn", None)
        if fn is None:
            raise RuntimeError("Job function missing.")
        return fn(job, progress, cancel_event)

    def _finish(self, job: DubbingJob, status: str, *, message: str = "") -> None:
        job.status = status
        job.finished_at = utc_now()
        job.message = message or job.message
        if status == DubbingJobStatus.CANCELLED:
            job.progress_total = max(job.progress_total, job.progress_current)
        self._persist(job)
        event = "job.cancelled" if status == DubbingJobStatus.CANCELLED else (
            "job.failed" if status == DubbingJobStatus.FAILED else "job.completed"
        )
        emit_event(event, payload={"job_id": job.job_id, "status": status}, force_flush=True)

    # ------------------------------------------------------------------ storage

    def submit_with_fn(
        self,
        project_id: str,
        job_type: str,
        fn: JobFn,
        *,
        run_id: str | None = None,
        operation_id: str | None = None,
        register_cancel: Callable[[], None] | None = None,
    ) -> DubbingJob:
        """Submit and attach the callable to the job object (workers read it)."""
        job = self.submit(project_id, job_type, fn, run_id=run_id, operation_id=operation_id)
        # Stash the fn on the in-memory job so the worker can find it.
        with self._active:
            stored = self._jobs.get(job.job_id)
            if stored is not None:
                setattr(stored, "_fn", fn)
                if register_cancel is not None:
                    self._cancel_handlers[job.job_id] = register_cancel
        return job

    def shutdown(self) -> None:
        self._stop.set()
        for _ in self._workers:
            self._queue.put(None)
        for thread in self._workers:
            if thread.is_alive():
                thread.join(timeout=5)
        self._workers = []

    # ------------------------------------------------------------------ sqlite

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dubbing_jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    stage TEXT NOT NULL DEFAULT '',
                    progress_current INTEGER NOT NULL DEFAULT 0,
                    progress_total INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    run_id TEXT NOT NULL DEFAULT '',
                    operation_id TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error_json TEXT NOT NULL DEFAULT '{}',
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def _persist(self, job: DubbingJob) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO dubbing_jobs (
                    job_id, job_type, project_id, status, created_at,
                    started_at, finished_at, stage, progress_current,
                    progress_total, message, run_id, operation_id,
                    result_json, error_json, cancel_requested
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    stage=excluded.stage,
                    progress_current=excluded.progress_current,
                    progress_total=excluded.progress_total,
                    message=excluded.message,
                    result_json=excluded.result_json,
                    error_json=excluded.error_json,
                    cancel_requested=excluded.cancel_requested
                """,
                (
                    job.job_id,
                    job.job_type,
                    job.project_id,
                    job.status,
                    job.created_at,
                    job.started_at,
                    job.finished_at,
                    job.stage,
                    job.progress_current,
                    job.progress_total,
                    job.message,
                    job.run_id,
                    job.operation_id,
                    json.dumps(job.result or {}, default=str),
                    json.dumps(job.error or {}, default=str),
                    1 if job.cancel_requested else 0,
                ),
            )

    def _load_row(self, job_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM dubbing_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()

    def _load_rows(
        self,
        *,
        project_id: str | None,
        status: str | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        query = "SELECT * FROM dubbing_jobs"
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            return connection.execute(query, params).fetchall()

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> DubbingJob:
        try:
            result = json.loads(row["result_json"] or "{}")
        except json.JSONDecodeError:
            result = {}
        try:
            error = json.loads(row["error_json"] or "{}")
        except json.JSONDecodeError:
            error = {}
        return DubbingJob(
            job_id=str(row["job_id"]),
            job_type=str(row["job_type"]),
            project_id=str(row["project_id"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            started_at=str(row["started_at"]),
            finished_at=str(row["finished_at"]),
            stage=str(row["stage"]),
            progress_current=int(row["progress_current"]),
            progress_total=int(row["progress_total"]),
            message=str(row["message"]),
            run_id=str(row["run_id"]),
            operation_id=str(row["operation_id"]),
            result=result,
            error=error if error else None,
            cancel_requested=bool(row["cancel_requested"]),
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._db_lock:
            connection = sqlite3.connect(self.db_path, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA busy_timeout = 30000")
            try:
                yield connection
                connection.commit()
            finally:
                connection.close()


class _JobCancelled(Exception):
    pass


def _safe_message(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__
