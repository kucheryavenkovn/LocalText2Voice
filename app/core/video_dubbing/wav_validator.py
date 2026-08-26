from __future__ import annotations

import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WavValidationResult:
    valid: bool
    duration_ms: int | None
    sample_rate: int | None
    channels: int | None
    frame_count: int | None
    file_size: int
    error_code: str | None = None
    error_message: str | None = None


class WavArtifactValidator:
    """Validates that a path is a usable PCM WAV artifact."""

    MIN_BYTES = 44
    MIN_DURATION_MS = 1

    def validate(self, path: Path | str) -> WavValidationResult:
        file_path = Path(path)
        if not file_path.exists():
            return WavValidationResult(
                valid=False,
                duration_ms=None,
                sample_rate=None,
                channels=None,
                frame_count=None,
                file_size=0,
                error_code="missing",
                error_message=f"File not found: {file_path}",
            )
        if not file_path.is_file():
            return WavValidationResult(
                valid=False,
                duration_ms=None,
                sample_rate=None,
                channels=None,
                frame_count=None,
                file_size=0,
                error_code="not_file",
                error_message=f"Not a regular file: {file_path}",
            )
        try:
            file_size = file_path.stat().st_size
        except OSError as exc:
            return WavValidationResult(
                valid=False,
                duration_ms=None,
                sample_rate=None,
                channels=None,
                frame_count=None,
                file_size=0,
                error_code="stat_failed",
                error_message=str(exc),
            )
        if file_size < self.MIN_BYTES:
            return WavValidationResult(
                valid=False,
                duration_ms=None,
                sample_rate=None,
                channels=None,
                frame_count=None,
                file_size=file_size,
                error_code="too_small",
                error_message=f"WAV too small ({file_size} bytes)",
            )
        try:
            with wave.open(str(file_path), "rb") as audio:
                channels = audio.getnchannels()
                sample_width = audio.getsampwidth()
                sample_rate = audio.getframerate()
                frame_count = audio.getnframes()
                # Attempt to read a small chunk to catch truncated data.
                if frame_count > 0:
                    audio.readframes(min(frame_count, sample_rate or 1))
        except (wave.Error, EOFError, OSError) as exc:
            return WavValidationResult(
                valid=False,
                duration_ms=None,
                sample_rate=None,
                channels=None,
                frame_count=None,
                file_size=file_size,
                error_code="invalid_wav",
                error_message=str(exc),
            )
        if channels is None or channels <= 0:
            return self._fail(file_size, "bad_channels", "Invalid channel count")
        if sample_rate is None or sample_rate <= 0:
            return self._fail(file_size, "bad_sample_rate", "Invalid sample rate")
        if sample_width is None or sample_width not in {1, 2, 3, 4}:
            return self._fail(file_size, "bad_sample_width", "Invalid sample width")
        if frame_count is None or frame_count <= 0:
            return self._fail(file_size, "empty_audio", "WAV has zero frames")
        expected_data = frame_count * channels * sample_width
        # Header is typically 44 bytes; allow some extra chunks.
        if file_size + 8 < expected_data:
            return WavValidationResult(
                valid=False,
                duration_ms=None,
                sample_rate=sample_rate,
                channels=channels,
                frame_count=frame_count,
                file_size=file_size,
                error_code="truncated",
                error_message="File size smaller than WAV header claims",
            )
        duration_ms = int(round(frame_count / sample_rate * 1000))
        if duration_ms < self.MIN_DURATION_MS:
            return WavValidationResult(
                valid=False,
                duration_ms=duration_ms,
                sample_rate=sample_rate,
                channels=channels,
                frame_count=frame_count,
                file_size=file_size,
                error_code="too_short",
                error_message=f"Duration too short: {duration_ms} ms",
            )
        return WavValidationResult(
            valid=True,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            channels=channels,
            frame_count=frame_count,
            file_size=file_size,
        )

    @staticmethod
    def _fail(file_size: int, code: str, message: str) -> WavValidationResult:
        return WavValidationResult(
            valid=False,
            duration_ms=None,
            sample_rate=None,
            channels=None,
            frame_count=None,
            file_size=file_size,
            error_code=code,
            error_message=message,
        )


def atomic_replace(src: Path, dst: Path) -> None:
    """Atomically replace ``dst`` with ``src`` (best-effort fsync first).

    On Windows the destination is frequently opened by ffmpeg probes /
    preview playback for the cue being regenerated, which makes a single
    ``os.replace`` fail with ``ERROR_ACCESS_DENIED``. Retry with backoff
    (handles transient handles) and fall back to delete-then-rename, which
    occasionally succeeds where replace does not.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with src.open("rb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        pass

    last_error: OSError | None = None
    # ~5s total budget; ffmpeg probe handles release in milliseconds, while
    # an active preview player usually releases within a second or two.
    backoff_steps = (
        [0.05] * 4 + [0.1] * 4 + [0.2] * 4 + [0.3] * 4 + [0.5] * 6
    )
    for attempt, delay in enumerate(backoff_steps):
        try:
            os.replace(str(src), str(dst))
            return
        except PermissionError as exc:
            last_error = exc
            # On Windows, replacing a locked file fails; deleting it first
            # then renaming sometimes succeeds against a read-locked handle.
            try:
                if dst.exists():
                    os.remove(str(dst))
                os.rename(str(src), str(dst))
                return
            except OSError:
                pass
            time.sleep(delay)
    raise last_error


def cleanup_part_files(directory: Path, *, move_to: Path | None = None) -> list[Path]:
    """Remove or quarantine leftover ``*.part`` files under ``directory``."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found: list[Path] = []
    patterns = ("*.part", "*.part.wav", "*.prefit.wav")
    seen: set[Path] = set()
    for pattern in patterns:
        for path in directory.rglob(pattern):
            if path in seen:
                continue
            seen.add(path)
            found.append(path)
            try:
                if move_to is not None:
                    move_to.mkdir(parents=True, exist_ok=True)
                    target = move_to / path.name
                    os.replace(str(path), str(target))
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                pass
    return found
