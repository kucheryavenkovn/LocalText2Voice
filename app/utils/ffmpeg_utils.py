from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Sequence

from .paths import resolve_app_path

_log = logging.getLogger("video_dubbing.ffmpeg")


class FFmpegError(RuntimeError):
    pass


class FFmpegCancelled(FFmpegError):
    pass


def find_ffmpeg(configured_path: str | Path) -> Path:
    configured = resolve_app_path(configured_path)
    if configured.is_file():
        return configured

    path_match = shutil.which("ffmpeg")
    if path_match:
        return Path(path_match)

    raise FFmpegError(
        "FFmpeg was not found. Place ffmpeg.exe in the ffmpeg folder, "
        "set ffmpeg_path in config.json, or add FFmpeg to PATH."
    )


class FFmpegRunner:
    def __init__(
        self,
        executable: Path,
        *,
        stderr_dump_dir: Path | str | None = None,
        tail_length: int = 3000,
    ) -> None:
        self.executable = executable
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._cancel_requested = threading.Event()
        self.stderr_dump_dir = Path(stderr_dump_dir) if stderr_dump_dir else None
        if self.stderr_dump_dir is not None:
            self.stderr_dump_dir.mkdir(parents=True, exist_ok=True)
        self.tail_length = tail_length
        # Full stderr of the last run (not truncated) — available to callers that
        # want the complete FFmpeg diagnostics instead of only the tail.
        self.last_full_stderr: str = ""
        self.last_command: list[str] = []
        self.last_exit_code: int | None = None

    def run(self, arguments: Sequence[str], *, label: str | None = None) -> None:
        if self._cancel_requested.is_set():
            raise FFmpegCancelled("Generation cancelled.")

        creation_flags = (
            subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0
        )
        command = [str(self.executable), *arguments]
        self.last_command = command
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags,
            )
        except OSError as exc:
            raise FFmpegError(f"Could not start FFmpeg: {exc}") from exc

        with self._lock:
            self._process = process

        stdout = b""
        stderr = b""
        started = time.monotonic()
        try:
            while True:
                if self._cancel_requested.is_set():
                    self._terminate(process)
                    raise FFmpegCancelled("Generation cancelled.")
                try:
                    stdout, stderr = process.communicate(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None

        full_stderr = stderr.decode("utf-8", errors="replace")
        self.last_full_stderr = full_stderr
        self.last_exit_code = process.returncode
        # Persist the FULL stderr to disk so forensic information is never lost
        # to the tail truncation used in the user-facing error message.
        self._dump_stderr(full_stderr, command, label, process.returncode, started)

        if process.returncode != 0:
            error_text = full_stderr.strip()
            if len(error_text) > self.tail_length:
                error_text = error_text[-self.tail_length:]
            raise FFmpegError(
                f"FFmpeg failed with exit code {process.returncode}:\n"
                f"{error_text or 'No error details were returned.'}"
            )

    def _dump_stderr(
        self,
        full_stderr: str,
        command: list[str],
        label: str | None,
        exit_code: int | None,
        started: float,
    ) -> None:
        if self.stderr_dump_dir is None:
            return
        try:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (label or "ffmpeg"))
            name = f"{stamp}_{safe}_{exit_code}.stderr.log"
            path = self.stderr_dump_dir / name
            header = (
                f"# command: {' '.join(command)}\n"
                f"# exit_code: {exit_code}\n"
                f"# duration_ms: {int((time.monotonic() - started) * 1000)}\n"
            )
            path.write_text(header + full_stderr, encoding="utf-8")
        except OSError:  # pragma: no cover - diagnostics must never crash
            pass

    def cancel_current(self) -> None:
        self._cancel_requested.set()
        with self._lock:
            process = self._process
        if process is not None:
            self._terminate(process)

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
