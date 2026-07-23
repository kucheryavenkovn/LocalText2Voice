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


# Conservative per-command limits to stay well under the Windows 8191-char
# command-line ceiling even with long Unicode paths.
MAX_CUE_INPUTS_PER_BATCH = 24
MAX_BATCH_MIX_INPUTS = 16
SAFE_COMMAND_CHAR_BUDGET = 7000


class TimelineRenderer:
    """Builds the single narration track of exact video duration.

    Each fitted cue is placed at its absolute SRT timecode. Cues are never
    concatenated sequentially, so there is no cumulative drift.

    Rendering is split into fixed time windows, and each window is rendered in
    bounded input batches (``MAX_CUE_INPUTS_PER_BATCH``). Large filter graphs
    are written to a ``-filter_complex_script`` file and windows are joined via
    the concat demuxer with a list file, so neither the command line nor the
    filter graph can exceed the Windows limit (no ``[WinError 206]``).
    """

    DEFAULT_WINDOW_SECONDS = 300.0

    def __init__(
        self,
        ffmpeg_path: str | Path,
        sample_rate: int = 48000,
        channels: int = 2,
        window_seconds: float | None = None,
        max_cue_inputs_per_batch: int = MAX_CUE_INPUTS_PER_BATCH,
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
        self.max_cue_inputs_per_batch = max(2, int(max_cue_inputs_per_batch))
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

    # ------------------------------------------------------------------ render

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
            self._render_window(window, window_path, work_dir, index)
            window_paths.append(window_path)

        self._check_cancelled()
        self.progress_callback(total - 1, total, "Concatenating timeline windows...")
        self._concat_windows(window_paths, output_path, duration_ms / 1000.0, work_dir)
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

    # ------------------------------------------------------------------ window

    def _render_window(
        self,
        window: _Window,
        output_path: Path,
        work_dir: Path,
        window_index: int,
    ) -> None:
        window_duration_s = (window.end_ms - window.start_ms) / 1000.0
        clipped = self._clip_segments_to_window(window)
        if not clipped:
            self._render_silence(output_path, window_duration_s)
            return

        batch_dir = work_dir / f"window_{window_index:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batches = [
            clipped[i : i + self.max_cue_inputs_per_batch]
            for i in range(0, len(clipped), self.max_cue_inputs_per_batch)
        ]
        batch_paths: list[Path] = []
        for batch_index, batch in enumerate(batches, start=1):
            self._check_cancelled()
            batch_path = batch_dir / f"batch_{batch_index:04d}.wav"
            self._render_batch(batch, window_duration_s, batch_path, batch_dir)
            batch_paths.append(batch_path)

        # Mix the (bounded number of) batch tracks into the final window WAV.
        if len(batch_paths) == 1:
            batch_paths[0].replace(output_path)
        else:
            self._mix_batches(batch_paths, output_path, window_duration_s, batch_dir)

    def _clip_segments_to_window(
        self, window: _Window
    ) -> list[tuple[PlacedSegment, int, int, float]]:
        """Return (segment, local_position_ms, segment_duration_ms, input_seek_s)
        clipped to the window. A cue crossing a boundary contributes only its
        in-window portion; the remainder is handled by the neighbouring window.
        ``input_seek_s`` is where to start reading inside the source cue file.
        """
        clipped: list[tuple[PlacedSegment, int, int, float]] = []
        for seg in window.segments:
            seg_start = seg.absolute_start_ms
            in_window_start_ms = max(window.start_ms, seg_start)
            in_window_end_ms = min(window.end_ms, seg_start + seg.duration_ms)
            if in_window_end_ms <= in_window_start_ms:
                continue
            local_position_ms = in_window_start_ms - window.start_ms
            segment_duration_ms = in_window_end_ms - in_window_start_ms
            input_seek_s = max(0.0, (in_window_start_ms - seg_start) / 1000.0)
            clipped.append(
                (seg, local_position_ms, segment_duration_ms, input_seek_s)
            )
        return clipped

    def _render_batch(
        self,
        batch: list[tuple[PlacedSegment, int, int, float]],
        window_duration_s: float,
        output_path: Path,
        work_dir: Path,
    ) -> None:
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
        prepared: list[tuple[int, int, int]] = []
        input_index = 1
        for seg, local_position_ms, segment_duration_ms, input_seek_s in batch:
            arguments.extend(["-ss", f"{input_seek_s:.3f}"])
            arguments.extend(["-t", f"{segment_duration_ms / 1000.0:.3f}"])
            arguments.extend(["-i", str(seg.audio_path)])
            prepared.append((input_index, local_position_ms, segment_duration_ms))
            input_index += 1

        filters: list[str] = []
        filters.append(
            f"[0:a]{fmt},atrim=duration={window_duration_s:.3f},asetpts=N/SR/TB[base]"
        )
        labels = ["[base]"]
        for index, (input_idx, local_position_ms, segment_duration_ms) in enumerate(
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
        graph = ";".join(filters)

        use_script, script_path = self._maybe_write_filter_script(graph, work_dir)
        if use_script:
            arguments.extend(["-filter_complex_script", str(script_path)])
        else:
            arguments.extend(["-filter_complex", graph])
        arguments.extend(["-map", "[out]", "-codec:a", "pcm_s16le", str(output_path)])

        self._assert_safe_command(arguments)
        try:
            self._runner_instance().run(arguments)
        except FFmpegCancelled as exc:
            raise TimelineRenderCancelled(str(exc)) from exc
        except FFmpegError as exc:
            raise TimelineRenderError(
                f"Could not render timeline batch: {exc}"
            ) from exc

    def _mix_batches(
        self,
        batch_paths: list[Path],
        output_path: Path,
        window_duration_s: float,
        work_dir: Path,
    ) -> None:
        fmt = self._audio_format_filter()
        arguments: list[str] = ["-y", "-hide_banner", "-loglevel", "error"]
        for path in batch_paths:
            arguments.extend(["-i", str(path)])
        filters: list[str] = []
        labels: list[str] = []
        for index in range(len(batch_paths)):
            label = f"b{index}"
            filters.append(f"[{index}:a]{fmt}[{label}]")
            labels.append(f"[{label}]")
        mix_inputs = "".join(labels)
        filters.append(
            f"{mix_inputs}amix=inputs={len(labels)}:duration=longest:normalize=0,"
            f"atrim=duration={window_duration_s:.3f},asetpts=N/SR/TB[out]"
        )
        graph = ";".join(filters)
        use_script, script_path = self._maybe_write_filter_script(graph, work_dir)
        if use_script:
            arguments.extend(["-filter_complex_script", str(script_path)])
        else:
            arguments.extend(["-filter_complex", graph])
        arguments.extend(["-map", "[out]", "-codec:a", "pcm_s16le", str(output_path)])
        self._assert_safe_command(arguments)
        try:
            self._runner_instance().run(arguments)
        except FFmpegCancelled as exc:
            raise TimelineRenderCancelled(str(exc)) from exc
        except FFmpegError as exc:
            raise TimelineRenderError(f"Could not mix timeline batches: {exc}") from exc

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

    # ------------------------------------------------------------------ concat

    def _concat_windows(
        self,
        window_paths: list[Path],
        output_path: Path,
        total_seconds: float,
        work_dir: Path,
    ) -> None:
        if len(window_paths) == 1:
            self._trim_to_duration(window_paths[0], output_path, total_seconds)
            return
        # Use the concat demuxer with a list file (avoids one -i per window).
        list_path = work_dir / "windows_concat.txt"
        lines = []
        for path in window_paths:
            escaped = str(path).replace("'", r"\'")
            lines.append(f"file '{escaped}'")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        arguments = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            str(self.channels),
            str(work_dir / "windows_concat.wav"),
        ]
        self._assert_safe_command(arguments)
        try:
            self._runner_instance().run(arguments)
        except FFmpegCancelled as exc:
            raise TimelineRenderCancelled(str(exc)) from exc
        except FFmpegError as exc:
            raise TimelineRenderError(f"Could not concatenate windows: {exc}") from exc
        self._trim_to_duration(
            work_dir / "windows_concat.wav", output_path, total_seconds
        )

    def _trim_to_duration(
        self, source: Path, output_path: Path, total_seconds: float
    ) -> None:
        fmt = self._audio_format_filter()
        graph = (
            f"[0:a]{fmt},atrim=0:{total_seconds:.3f},asetpts=N/SR/TB[out]"
        )
        arguments = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-codec:a",
            "pcm_s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            str(self.channels),
            str(output_path),
        ]
        try:
            self._runner_instance().run(arguments)
        except FFmpegCancelled as exc:
            raise TimelineRenderCancelled(str(exc)) from exc
        except FFmpegError as exc:
            raise TimelineRenderError(f"Could not trim narration: {exc}") from exc

    # ------------------------------------------------------------------ safety

    @staticmethod
    def _maybe_write_filter_script(
        graph: str, work_dir: Path
    ) -> tuple[bool, Path | None]:
        if len(graph) <= 1500:
            return False, None
        script_path = work_dir / f"filter_{abs(hash(graph)) & 0xFFFFFFFF:08x}.txt"
        script_path.write_text(graph, encoding="utf-8")
        return True, script_path

    @staticmethod
    def _assert_safe_command(arguments: list[str]) -> None:
        approx = sum(len(arg) + 3 for arg in arguments)
        if approx > SAFE_COMMAND_CHAR_BUDGET:
            raise TimelineRenderError(
                f"FFmpeg command too long for Windows ({approx} chars > "
                f"{SAFE_COMMAND_CHAR_BUDGET}). Reduce batch size."
            )


@dataclass
class _Window:
    start_ms: int
    end_ms: int
    segments: list[PlacedSegment]
