from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from app.utils.ffmpeg_utils import (
    FFmpegCancelled,
    FFmpegError,
    FFmpegRunner,
    find_ffmpeg,
)

from .models import DubbingCue, DubbingProject, OriginalAudioMode
from .duration_fitter import DurationFitter


ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str], None]


class PreviewRenderError(RuntimeError):
    pass


class PreviewRenderCancelled(PreviewRenderError):
    pass


class PreviewRenderer:
    """Produces fast, in-app preview files.

    * ``cue preview`` — a short clip around a single cue (start - pre_roll to
      end + post_roll) with the video stream copied, the original audio ducked
      during the cue, and the fitted narration laid on top. Built quickly per
      selected cue.
    * ``full preview`` — the whole video remuxed with the current Dubbed Mix
      (or narration) as the audio. The video stream is copied, so this is much
      faster than a final export and avoids re-encoding on every edit.

    Previews are internal-only artifacts; the original video and final render
    are never modified.
    """

    def __init__(
        self,
        ffmpeg_path: str | Path,
        sample_rate: int = 48000,
        channels: int = 2,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.sample_rate = sample_rate
        self.channels = channels
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
            raise PreviewRenderCancelled("Preview render cancelled.")

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

    def render_cue_preview(
        self,
        project: DubbingProject,
        cue: DubbingCue,
        output_path: Path | None = None,
    ) -> Path:
        if project.video_path is None or not Path(project.video_path).is_file():
            raise PreviewRenderError("Project has no source video.")
        if cue.fitted_audio_path is None or not Path(cue.fitted_audio_path).is_file():
            raise PreviewRenderError(
                f"Cue #{cue.sequence} has no fitted audio; generate it first."
            )
        preview = project.settings.preview
        duration_ms = project.duration_ms or cue.end_ms + preview.post_roll_ms
        preview_start_ms = max(0, cue.start_ms - preview.pre_roll_ms)
        preview_end_ms = min(duration_ms, cue.end_ms + preview.post_roll_ms)
        if preview_end_ms <= preview_start_ms:
            raise PreviewRenderError("Cue preview window is empty.")
        if output_path is None:
            output_path = project.cue_preview_path(cue.sequence)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        window_seconds = (preview_end_ms - preview_start_ms) / 1000.0
        offset = DurationFitter.placement_offset(
            cue.fitted_duration_ms or 0,
            cue.duration_budget_ms,
            preview.alignment,
        )
        narration_offset_ms = max(0, cue.start_ms - preview_start_ms) + offset
        duck = project.settings.ducking
        during_gain = max(0.0, duck.original_during_percent / 100.0)
        outside_gain = max(0.0, duck.original_outside_percent / 100.0)

        fitted_path = Path(cue.fitted_audio_path)
        arguments: list[str] = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{preview_start_ms / 1000.0:.3f}",
            "-t",
            f"{window_seconds:.3f}",
            "-i",
            str(project.video_path),
            "-i",
            str(fitted_path),
        ]
        fmt = self._audio_format_filter()
        rel_duck_start = max(0.0, (cue.start_ms - preview_start_ms) / 1000.0)
        rel_duck_end = min(
            window_seconds,
            (cue.end_ms + max(0, cue.overflow_ms) - preview_start_ms) / 1000.0,
        )
        if duck.mode == OriginalAudioMode.NARRATION_ONLY or duck.mode == OriginalAudioMode.REPLACE:
            original_chain = f"{fmt},volume=-96dB"
        elif duck.mode == OriginalAudioMode.CONSTANT:
            const_db = 20 * _log_gain(during_gain)
            original_chain = f"{fmt},volume={const_db:.2f}dB"
        else:
            outside_db = 20 * _log_gain(outside_gain)
            during_db = 20 * _log_gain(during_gain)
            original_chain = (
                f"{fmt},volume={outside_db:.2f}dB,"
                f"volume={during_db:.2f}dB:"
                f"enable='between(t,{rel_duck_start:.3f},{rel_duck_end:.3f})'"
            )
        original_chain_full = (
            original_chain
            + f",apad=whole_dur={window_seconds:.3f},"
            f"atrim=0:{window_seconds:.3f},asetpts=N/SR/TB"
        )
        filters: list[str] = [f"[0:a]{original_chain_full}[orig]"]
        narr_chain = [fmt]
        if narration_offset_ms > 0:
            narr_chain.append(f"adelay={narration_offset_ms}:all=1")
        narr_chain.append(f"apad=whole_dur={window_seconds:.3f}")
        narr_chain.append(f"atrim=0:{window_seconds:.3f}")
        narr_chain.append("asetpts=N/SR/TB")
        filters.append("[1:a]" + ",".join(narr_chain) + "[narr]")
        filters.append(
            "[orig][narr]amix=inputs=2:duration=longest:normalize=0,"
            f"atrim=0:{window_seconds:.3f},asetpts=N/SR/TB[out]"
        )
        arguments.extend(["-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[out]"])
        arguments.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-t",
                f"{window_seconds:.3f}",
                str(output_path),
            ]
        )
        try:
            self._runner_instance().run(arguments)
        except FFmpegCancelled as exc:
            raise PreviewRenderCancelled(str(exc)) from exc
        except FFmpegError as exc:
            raise PreviewRenderError(f"Could not render cue preview: {exc}") from exc
        project.cue_preview_cache[str(cue.sequence)] = output_path
        self.log_callback(
            f"Cue #{cue.sequence} preview rendered "
            f"({preview_start_ms}-{preview_end_ms} ms)."
        )
        return output_path

    def render_full_preview(
        self,
        project: DubbingProject,
        output_path: Path | None = None,
        mix_wav: Path | None = None,
    ) -> Path:
        if project.video_path is None or not Path(project.video_path).is_file():
            raise PreviewRenderError("Project has no source video.")
        audio_source = mix_wav or project.dubbed_mix_wav or project.narration_wav
        if audio_source is None or not Path(audio_source).is_file():
            raise PreviewRenderError(
                "No mix/narration track available; build one first."
            )
        if output_path is None:
            output_path = project.preview_dir() / "full_preview.mkv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arguments = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(project.video_path),
            "-i",
            str(audio_source),
            "-map",
            "0:v",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(output_path),
        ]
        try:
            self._runner_instance().run(arguments)
        except FFmpegCancelled as exc:
            raise PreviewRenderCancelled(str(exc)) from exc
        except FFmpegError as exc:
            raise PreviewRenderError(f"Could not render full preview: {exc}") from exc
        project.full_preview_path = output_path
        self.log_callback(f"Full preview rendered: {output_path}")
        return output_path


def _log_gain(gain: float) -> float:
    if gain <= 0:
        return -96.0
    import math

    return math.log10(max(1e-5, gain)) * 20
