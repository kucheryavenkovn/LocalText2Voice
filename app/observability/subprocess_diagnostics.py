"""Subprocess runner with full forensic diagnostics.

Replaces the ad-hoc ``FFmpegRunner`` truncation (last 3000 chars of stderr)
with a runner that:

* writes the **full** stderr to ``<run_dir>/subprocess/<op>.stderr.log``;
* records pid, timing, exit code, timeout, cancellation, redacted command;
* returns a short tail for UI messages;
* supports cooperative cancellation via a ``threading.Event``.

The UI message keeps a tail; the full stderr survives on disk.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .context import current_context, new_operation_id
from .events import emit_event
from .redaction import redact_command

_log = logging.getLogger("video_dubbing.subprocess")

_DEFAULT_TAIL = 3000


class SubprocessError(RuntimeError):
    pass


class SubprocessCancelled(SubprocessError):
    pass


class SubprocessTimeout(SubprocessError):
    pass


@dataclass
class SubprocessResult:
    operation_id: str
    executable: str
    command: list[str]
    cwd: str
    pid: int | None
    exit_code: int | None
    started_at: float
    finished_at: float
    duration_ms: int
    cancelled: bool = False
    timed_out: bool = False
    stderr_tail: str = ""
    stderr_path: Path | None = None
    stdout_path: Path | None = None
    input_files: list[str] = field(default_factory=list)
    output_file: str | None = None
    input_sizes: dict[str, int] = field(default_factory=dict)
    output_size: int | None = None
    timeout_seconds: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.cancelled and not self.timed_out

    def to_event_payload(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "executable": self.executable,
            "command": self.command,
            "cwd": self.cwd,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "cancelled": self.cancelled,
            "timed_out": self.timed_out,
            "timeout_seconds": self.timeout_seconds,
            "stderr_tail": self.stderr_tail,
            "stderr_path": str(self.stderr_path) if self.stderr_path else None,
            "stdout_path": str(self.stdout_path) if self.stdout_path else None,
            "input_files": self.input_files,
            "output_file": self.output_file,
            "input_sizes": self.input_sizes,
            "output_size": self.output_size,
        }


class DiagnosedSubprocess:
    """Run an external process with full diagnostics."""

    def __init__(
        self,
        stderr_dir: Path | None = None,
        *,
        tail_length: int = _DEFAULT_TAIL,
    ) -> None:
        self.stderr_dir = Path(stderr_dir) if stderr_dir else None
        if self.stderr_dir is not None:
            self.stderr_dir.mkdir(parents=True, exist_ok=True)
        self.tail_length = tail_length
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._cancel_requested = threading.Event()

    def cancel_current(self) -> None:
        self._cancel_requested.set()
        with self._lock:
            process = self._process
        if process is not None:
            _terminate(process)

    @property
    def cancellation_requested(self) -> bool:
        return self._cancel_requested.is_set()

    def reset_cancel(self) -> None:
        self._cancel_requested.clear()

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        input_files: Sequence[str | Path] | None = None,
        output_file: str | Path | None = None,
        label: str | None = None,
        poll_interval: float = 0.2,
    ) -> SubprocessResult:
        if self._cancel_requested.is_set():
            raise SubprocessCancelled("Generation cancelled before subprocess start.")
        operation_id = new_operation_id()
        command_list = [str(c) for c in command]
        safe_command = redact_command(command_list)
        executable = safe_command[0] if safe_command else ""
        input_strs = [str(p) for p in (input_files or [])]
        output_str = str(output_file) if output_file is not None else None
        input_sizes = _file_sizes(input_strs)

        stderr_path = self._sink_path(label or operation_id, ".stderr.log")
        stdout_path = self._sink_path(label or operation_id, ".stdout.log")
        started = time.monotonic()
        started_wall = time.time()

        creation_flags = (
            subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        )
        ctx = current_context()
        emit_event(
            "subprocess.started",
            payload={
                "operation_id": operation_id,
                "executable": executable,
                "command": safe_command,
                "cwd": str(cwd) if cwd else None,
                "timeout_seconds": timeout,
                "input_files": input_strs,
                "input_sizes": input_sizes,
                "output_file": output_str,
                "context": ctx,
            },
        )
        _log.info(
            "subprocess.start pid=? exe=%s op=%s inputs=%d",
            executable,
            operation_id,
            len(input_strs),
        )

        stderr_fh = stdout_fh = None
        try:
            if stderr_path is not None:
                stderr_fh = open(stderr_path, "wb")
            if stdout_path is not None:
                stdout_fh = open(stdout_path, "wb")
            try:
                process = subprocess.Popen(
                    command_list,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(cwd) if cwd else None,
                    creationflags=creation_flags,
                )
            except OSError as exc:
                raise SubprocessError(f"Could not start {executable}: {exc}") from exc

            with self._lock:
                self._process = process
            pid = process.pid
            emit_event(
                "subprocess.pid",
                payload={"operation_id": operation_id, "pid": pid},
            )

            timed_out = False
            cancelled = False
            try:
                while True:
                    if self._cancel_requested.is_set():
                        _terminate(process)
                        cancelled = True
                        break
                    try:
                        stdout, stderr = process.communicate(timeout=poll_interval)
                        break
                    except subprocess.TimeoutExpired:
                        if timeout is not None and (time.monotonic() - started) > timeout:
                            _terminate(process)
                            timed_out = True
                            try:
                                stdout, stderr = process.communicate(timeout=2)
                            except subprocess.TimeoutExpired:
                                stdout, stderr = b"", b""
                            break
                        continue
            finally:
                with self._lock:
                    if self._process is process:
                        self._process = None

            if stderr_fh is not None:
                stderr_fh.write(stderr or b"")
            if stdout_fh is not None:
                stdout_fh.write(stdout or b"")

            exit_code = process.returncode
            stderr_text = (stderr or b"").decode("utf-8", errors="replace")
            tail = stderr_text[-self.tail_length:].strip() if stderr_text else ""
            output_size = _file_size(output_str) if output_str else None
            finished = time.monotonic()
            result = SubprocessResult(
                operation_id=operation_id,
                executable=executable,
                command=safe_command,
                cwd=str(cwd) if cwd else "",
                pid=pid,
                exit_code=exit_code,
                started_at=started_wall,
                finished_at=time.time(),
                duration_ms=int((finished - started) * 1000),
                cancelled=cancelled,
                timed_out=timed_out,
                stderr_tail=tail,
                stderr_path=stderr_path,
                stdout_path=stdout_path,
                input_files=input_strs,
                output_file=output_str,
                input_sizes=input_sizes,
                output_size=output_size,
                timeout_seconds=timeout,
            )

            payload = result.to_event_payload()
            emit_event(
                "subprocess.completed" if result.succeeded else "subprocess.failed",
                payload=payload,
                force_flush=not result.succeeded,
            )
            _log.info(
                "subprocess.done op=%s pid=%s exit=%s dur=%dms cancelled=%s timeout=%s",
                operation_id,
                pid,
                exit_code,
                result.duration_ms,
                cancelled,
                timed_out,
            )

            if cancelled:
                raise SubprocessCancelled(f"Subprocess cancelled: {executable}")
            if timed_out:
                raise SubprocessTimeout(
                    f"Subprocess timed out after {timeout}s: {executable}\n{tail}"
                )
            if exit_code != 0:
                raise SubprocessError(
                    f"{executable} failed with exit code {exit_code}:\n"
                    f"{tail or 'No error details were returned.'}"
                )
            return result
        finally:
            for fh in (stderr_fh, stdout_fh):
                if fh is not None:
                    try:
                        fh.flush()
                    except Exception:  # pragma: no cover
                        pass
                    try:
                        os.fsync(fh.fileno())
                    except (OSError, ValueError):
                        pass
                    fh.close()

    def _sink_path(self, label: str, suffix: str) -> Path | None:
        if self.stderr_dir is None:
            return None
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
        return self.stderr_dir / f"{safe}{suffix}"


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.poll() is not None:
            return
    except Exception:  # pragma: no cover
        pass
    try:
        process.terminate()
    except Exception:  # pragma: no cover
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except Exception:  # pragma: no cover
            pass


def _file_size(path: str | None) -> int | None:
    if not path:
        return None
    try:
        return Path(path).stat().st_size
    except OSError:
        return None


def _file_sizes(paths: Sequence[str]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for path in paths:
        size = _file_size(path)
        if size is not None:
            sizes[path] = size
    return sizes


__all__ = [
    "DiagnosedSubprocess",
    "SubprocessCancelled",
    "SubprocessError",
    "SubprocessResult",
    "SubprocessTimeout",
]
