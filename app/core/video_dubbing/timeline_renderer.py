from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.utils.ffmpeg_utils import (
    FFmpegCancelled,
    FFmpegError,
    FFmpegRunner,
    find_ffmpeg,
)

from .duration_fitter import DurationFitter
from .models import Alignment, DubbingCue, DubbingProject


ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str], None]


class TimelineRenderError(RuntimeError):
    pass


class TimelineRenderCancelled(TimelineRenderError):
    pass


@dataclass(frozen=True)
class PlacedSegment:
    cue_id: str
    sequence: int
    audio_path: Path
    absolute_start_ms: int
    duration_ms: int


class TimelineRenderer:
    """Builds the single narration track of exact video duration.

    Each fitted cue is placed at its absolute SRT timecode (start_ms +
    placement_offset_ms). Silence fills the gaps. Cues are never concatenated
    sequentially, so there is no cumulative drift.

    Rendering is split into fixed time windows so thousands of cues do not
    exceed the Windows command-line length limit. A cue crossing a window
    boundary is split deterministically across the windows it intersects, then
    reassembled by window concatenation.
    """

    DEFAULT_WINDOW_SECONDS = 300.0

    def __init__(
        self,
        ffmpeg_path: str | Path,
        sample_rate: int = 48000,
        channels: int = 2,
        window_seconds: float | None = None,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.window_seconds = (
            window_seconds if window_seconds and window_seconds > 0
            else self.DEFAULT_WINDOW_SECONDS
        )
        self.progress_callback = progress_callback or (lambda c, t, msg: None)
        self.log_callback = log_callback or (lambda msg: None)
        self._cancel_requested = threading.Event()
        self._runner: FFmpegRunner | None = None

    def cancel(self) -> None:
        self._cancel_requested.set()
        runner = self._runner
        if runner is not None:
            runner.cancel_current()

    def _check_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise TimelineRenderCancelled("Timeline render cancelled.")

    def _runner_instance(self) -> FFmpegRunner:
        if self._runner is None:
            self._runner = FFmpegRunner(find_ffmpeg(self.ffmpeg_path))
        return self._runner

    def _audio_format_filter(self) -> str:
        layout = "stereo" if self.channels == 2 else "mono"
        return (
            f"aresample={self.sample_rate},"
            f"aformat=sample_fmts=fltp:channel_layouts={layout}"
        )

    def _silence_lavfi(self) -> str:
        layout = "stereo" if self.channels == 2 else "mono"
        return f"anullsrc=channel_layout={layout}:sample_rate={self.sample_rate}"

    def placed_segments(
        self,
        cues: list[DubbingCue],
        alignment: Alignment = Alignment.START,
    ) -> list[PlacedSegment]:
        segments: list[PlacedSegment] = []
        for cue in cues:
            if not cue.enabled:
                continue
            fitted = cue.fitted_audio_path
            duration_ms = cue.fitted_duration_ms
            if fitted is None or duration_ms is None or duration_ms <= 0:
                continue
            if not Path(fitted).is_file():
                continue
            offset = DurationFitter.placement_offset(
                duration_ms, cue.duration_budget_ms, alignment
            )
            segments.append(
                PlacedSegment(
                    cue_id=cue.cue_id,
                    sequence=cue.sequence,
                    audio_path=Path(fitted),
                    absolute_start_ms=cue.start_ms + offset,
                    duration_ms=duration_ms,
                )
            )
        segments.sort(key=lambda seg: seg.absolute_start_ms)
        return segments

    def render(
        self,
        project: DubbingProject,
        output_path: Path,
        alignment: Alignment = Alignment.START,
        temp_dir: Path | None = None,
    ) -> Path:
        duration_ms = project.duration_ms
        if duration_ms <= 0:
            raise TimelineRenderError(
                "Project has no video duration; cannot size narration track."
            )
        segments = self.placed_segments(project.cues, alignment=alignment)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        work_dir = temp_dir or project.temp_dir()
        work_dir.mkdir(parents=True, exist_ok=True)

        if not segments:
            self.progress_callback(0, 1, "Rendering silent narration track...")
            self._render_silence(output_path, duration_ms / 1000.0)
            self.progress_callback(1, 1, "Narration track created (silence).")
            return output_path

        windows = self._plan_windows(duration_ms, segments)
        window_paths: list[Path] = []
        total = len(windows) + 1
        for index, window in enumerate(windows, start=1):
            self._check_cancelled()
            self.progress_callback(
                index - 1,
                total,
                f"Rendering timeline window {index}/{len(windows)}...",
            )
            window_path = work_dir / f"window_{index:04d}.wav"
            self._render_window(window, window_path)
            window_paths.append(window_path)

        self._check_cancelled()
        self.progress_callback(total - 1, total, "Concatenating timeline windows...")
        self._concat_and_trim(window_paths, output_path, duration_ms / 1000.0)
        self.progress_callback(total, total, "Narration track assembled.")
        self.log_callback(
            f"Narration track rendered: {duration_ms} ms "
            f"({len(segments)} cue(s), {len(windows)} window(s))."
        )
        return output_path

    def _plan_windows(
        self,
        duration_ms: int,
        segments: list[PlacedSegment],
    ) -> list[_Window]:
        window_ms = max(1000, int(round(self.window_seconds * 1000)))
        windows: list[_Window] = []
        cursor = 0
        while cursor < duration_ms:
            end = min(duration_ms, cursor + window_ms)
            intersecting: list[PlacedSegment] = []
            for seg in segments:
                seg_start = seg.absolute_start_ms
                seg_end = seg_start + seg.duration_ms
                if seg_start < end and seg_end > cursor:
                    intersecting.append(seg)
            windows.append(_Window(start_ms=cursor, end_ms=end, segments=intersecting))
            cursor = end
        return windows

    def _render_window(self, window: _Window, output_path: Path) -> None:
        window_duration_s = (window.end_ms - window.start_ms) / 1000.0
        fmt = self._audio_format_filter()
        arguments: list[str] = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-t",
            f"{window_duration_s:.3f}",
            "-i",
            self._silence_lavfi(),
        ]
        prepared: list[tuple[int, int, float, int]] = []
        input_index = 1
        for seg in window.segments:
            seg_start = seg.absolute_start_ms
            cue_offset_start_s = max(0.0, (window.start_ms - seg_start) / 1000.0)
            cue_offset_end_s = min(
                seg.duration_ms, window.end_ms - seg_start
            ) / 1000.0
            if cue_offset_end_s <= cue_offset_start_s:
                continue
            local_position_ms = max(0, seg_start - window.start_ms)
            segment_duration_ms = int(round((cue_offset_end_s - cue_offset_start_s) * 1000))
            arguments.extend(["-ss", f"{cue_offset_start_s:.3f}"])
            arguments.extend(["-t", f"{segment_duration_ms / 1000.0:.3f}"])
            arguments.extend(["-i", str(seg.audio_path)])
            prepared.append((input_index, local_position_ms, cue_offset_start_s, segment_duration_ms))
            input_index += 1

        filters: list[str] = []
        filters.append(
            f"[0:a]{fmt},atrim=duration={window_duration_s:.3f},asetpts=N/SR/TB[base]"
        )
        labels = ["[base]"]
        for index, (input_idx, local_position_ms, _cue_offset, segment_duration_ms) in enumerate(
            prepared, start=1
        ):
            label = f"seg{index}"
            chain = [
                fmt,
                f"atrim=duration={segment_duration_ms / 1000.0:.3f}",
                "asetpts=N/SR/TB",
            ]
            if local_position_ms > 0:
                chain.append(f"adelay={local_position_ms}:all=1")
            filters.append(f"[{input_idx}:a]" + ",".join(chain) + f"[{label}]")
            labels.append(f"[{label}]")
        mix_inputs = "".join(labels)
        filters.append(
            f"{mix_inputs}amix=inputs={len(labels)}:duration=first:normalize=0,"
            f"atrim=duration={window_duration_s:.3f},asetpts=N/SR/TB[out]"
        )
        arguments.extend(["-filter_complex", ";".join(filters), "-map", "[out]"])
        arguments.extend(["-codec:a", "pcm_s16le", str(output_path)])
        try:
            self._runner_instance().run(arguments)
        except FFmpegCancelled as exc:
            raise TimelineRenderCancelled(str(exc)) from exc
        except FFmpegError as exc:
            raise TimelineRenderError(f"Could not render window: {exc}") from exc

    def _render_silence(self, output_path: Path, duration_seconds: float) -> None:
        arguments = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-t",
            f"{max(0.01, duration_seconds):.3f}",
            "-i",
            self._silence_lavfi(),
            "-ar",
            str(self.sample_rate),
            "-ac",
            str(self.channels),
            "-codec:a",
            "pcm_s16le",
            str(output_path),
        ]
        try:
            self._runner_instance().run(arguments)
        except FFmpegCancelled as exc:
            raise TimelineRenderCancelled(str(exc)) from exc
        except FFmpegError as exc:
            raise TimelineRenderError(f"Could not render silence: {exc}") from exc

    def _concat_and_trim(
        self,
        window_paths: list[Path],
        output_path: Path,
        total_seconds: float,
    ) -> None:
        arguments: list[str] = ["-y", "-hide_banner", "-loglevel", "error"]
        for path in window_paths:
            arguments.extend(["-i", str(path)])
        fmt = self._audio_format_filter()
        filter_parts = [
            f"[{index}:a]{fmt}[a{index}]"
            for index in range(len(window_paths))
        ]
        concat_inputs = "".join(f"[a{index}]" for index in range(len(window_paths)))
        filter_parts.append(
            f"{concat_inputs}concat=n={len(window_paths)}:v=0:a=1[concat]"
        )
        filter_parts.append(
            f"[concat]atrim=0:{total_seconds:.3f},asetpts=N/SR/TB[out]"
        )
        arguments.extend(["-filter_complex", ";".join(filter_parts), "-map", "[out]"])
        arguments.extend(
            [
                "-codec:a",
                "pcm_s16le",
                "-ar",
                str(self.sample_rate),
                "-ac",
                str(self.channels),
                str(output_path),
            ]
        )
        try:
            self._runner_instance().run(arguments)
        except FFmpegCancelled as exc:
            raise TimelineRenderCancelled(str(exc)) from exc
        except FFmpegError as exc:
            raise TimelineRenderError(f"Could not concatenate windows: {exc}") from exc


@dataclass
class _Window:
    start_ms: int
    end_ms: int
    segments: list[PlacedSegment]
