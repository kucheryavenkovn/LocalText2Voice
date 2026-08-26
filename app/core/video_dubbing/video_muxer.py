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
from app.utils.ffprobe_utils import parse_video_probe, probe_media

from .models import DubbingProject, OutputContainer


ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str], None]


class VideoMuxError(RuntimeError):
    pass


class VideoMuxCancelled(VideoMuxError):
    pass


@dataclass(frozen=True)
class MuxResult:
    output_path: Path
    container: OutputContainer
    tracks: list[str]


# Video codecs each container can hold without re-encoding.
_CONTAINER_NATIVE_VIDEO = {
    OutputContainer.MP4: {"h264", "hevc", "h265", "av1", "mpeg4", "vp9"},
    OutputContainer.MKV: {
        "h264",
        "hevc",
        "h265",
        "av1",
        "mpeg4",
        "vp9",
        "vp8",
        "theora",
        "mjpeg",
    },
}


class VideoMuxer:
    """Assembles the final container: video stream (copied by default) plus
    multiple audio tracks (Original, Dubbed Mix, optional Narration Only).

    Dubbed Mix is marked as the default audio track so consumer players pick it
    automatically. The video stream is copied (-c:v copy) whenever the codec is
    compatible with the target container; otherwise a controlled re-encode is
    used or MKV is recommended. ``-shortest`` is never used so the output keeps
    the original video duration.
    """

    def __init__(
        self,
        ffmpeg_path: str | Path,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
        stderr_dump_dir: Path | str | None = None,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.progress_callback = progress_callback or (lambda c, t, msg: None)
        self.log_callback = log_callback or (lambda msg: None)
        self._cancel_requested = threading.Event()
        self._runner: FFmpegRunner | None = None
        self.stderr_dump_dir = stderr_dump_dir

    def cancel(self) -> None:
        self._cancel_requested.set()
        runner = self._runner
        if runner is not None:
            runner.cancel_current()

    def _check_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise VideoMuxCancelled("Video mux cancelled.")

    def _runner_instance(self) -> FFmpegRunner:
        if self._runner is None:
            self._runner = FFmpegRunner(
                find_ffmpeg(self.ffmpeg_path), stderr_dump_dir=self.stderr_dump_dir
            )
        return self._runner

    def can_copy_video(
        self,
        video_path: Path,
        container: OutputContainer,
    ) -> tuple[bool, str]:
        try:
            data = probe_media(video_path, self.ffmpeg_path)
        except FFmpegError:
            return True, ""
        _duration, video_stream, _audio = parse_video_probe(data)
        if video_stream is None:
            return True, ""
        codec = video_stream.codec_name.casefold()
        native = _CONTAINER_NATIVE_VIDEO.get(container, set())
        if codec in native:
            return True, codec
        return False, codec

    def mux(
        self,
        project: DubbingProject,
        output_path: Path | None = None,
        narration_only_wav: Path | None = None,
        srt_path: Path | None = None,
    ) -> MuxResult:
        if project.video_path is None or not Path(project.video_path).is_file():
            raise VideoMuxError("Project has no source video.")
        if project.dubbed_mix_wav is None or not Path(project.dubbed_mix_wav).is_file():
            raise VideoMuxError("Dubbed mix track is missing; render it first.")

        settings = project.settings.export
        container = settings.container
        video_path = Path(project.video_path)
        if output_path is None:
            suffix = ".mp4" if container == OutputContainer.MP4 else ".mkv"
            output_path = project.render_dir() / f"dubbed_video{suffix}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        copy_video, codec_name = self.can_copy_video(video_path, container)
        if not copy_video and not settings.force_video_reencode:
            raise VideoMuxError(
                f"Video codec '{codec_name}' is not natively supported by "
                f"{container.value}. Use MKV or enable forced re-encode."
            )

        self._check_cancelled()
        self.progress_callback(0, 2, "Muxing video and audio tracks...")
        has_source_audio, audio_probe_error = self._source_has_audio(video_path)
        if audio_probe_error and not has_source_audio:
            # ffprobe failed: do NOT silently assume the video has audio and
            # copy a non-existent stream. Surface a clear diagnostic instead.
            raise VideoMuxError(
                f"Could not probe source audio for {video_path.name}: "
                f"{audio_probe_error}. The video track will not be copied blindly."
            )
        arguments = self._build_arguments(
            project=project,
            video_path=video_path,
            output_path=output_path,
            container=container,
            copy_video=copy_video,
            narration_only_wav=narration_only_wav,
            srt_path=srt_path,
            has_source_audio=has_source_audio,
        )
        try:
            self._runner_instance().run(arguments, label="video_mux")
        except FFmpegCancelled as exc:
            raise VideoMuxCancelled(str(exc)) from exc
        except FFmpegError as exc:
            runner = self._runner
            full_stderr = getattr(runner, "last_full_stderr", "") if runner else ""
            raise VideoMuxError(
                f"Could not mux final video: {exc}"
                + (f"\n[full stderr retained: {len(full_stderr)} chars]" if full_stderr else "")
            ) from exc

        self.progress_callback(2, 2, "Final video assembled.")
        tracks = self._expected_tracks(project, narration_only_wav, srt_path, has_source_audio)
        self.log_callback(
            f"Final video: {output_path} ({container.value}, "
            f"{'copy' if copy_video else 'reencode'} video, {len(tracks)} tracks)."
        )
        return MuxResult(output_path=output_path, container=container, tracks=tracks)

    def _expected_tracks(
        self,
        project: DubbingProject,
        narration_only_wav: Path | None,
        srt_path: Path | None,
        has_source_audio: bool = True,
    ) -> list[str]:
        tracks = ["Dubbed Mix"]
        if has_source_audio:
            tracks = ["Original", "Dubbed Mix"]
        if (
            project.settings.export.include_narration_only
            and narration_only_wav is not None
        ):
            tracks.append("Narration Only")
        if project.settings.export.embed_subtitles and srt_path is not None:
            tracks.append("Subtitles")
        return tracks

    def _build_arguments(
        self,
        project: DubbingProject,
        video_path: Path,
        output_path: Path,
        container: OutputContainer,
        copy_video: bool,
        narration_only_wav: Path | None,
        srt_path: Path | None,
        has_source_audio: bool = True,
    ) -> list[str]:
        settings = project.settings.export
        dubbed_mix = Path(project.dubbed_mix_wav)
        include_narration = (
            settings.include_narration_only and narration_only_wav is not None
        )
        embed_srt = settings.embed_subtitles and srt_path is not None

        arguments: list[str] = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(dubbed_mix),
        ]
        # Input index map: 0=video(original audio also here), 1=dubbed mix,
        # 2=narration only (optional), 3=srt (optional).
        narration_input_index: int | None = None
        if include_narration and narration_only_wav is not None:
            narration_input_index = 2
            arguments.extend(["-i", str(narration_only_wav)])

        srt_input_index: int | None = None
        if embed_srt and srt_path is not None:
            srt_input_index = narration_input_index + 1 if narration_input_index else 2
            arguments.extend(["-i", str(srt_path)])

        # Map video stream (copied).
        if copy_video:
            arguments.extend(["-map", "0:v", "-c:v", "copy"])
        else:
            video_codec = (
                settings.video_codec_override
                or ("libx264" if container == OutputContainer.MP4 else "libx264")
            )
            arguments.extend(["-map", "0:v", "-c:v", video_codec])

        # Audio track 0: Original (first audio stream of the source) — ONLY when
        # the source actually has an audio stream. Mapping 0:a:0 on a video with
        # no audio makes FFmpeg fail with "Could not find tag ... in stream".
        track_index = 1
        if has_source_audio:
            arguments.extend(["-map", "0:a:0"])
            # Audio track 1: Dubbed Mix (default).
            arguments.extend(["-map", "1:a:0"])
        else:
            # Source has no audio: export Dubbed Mix as the only/primary track.
            arguments.extend(["-map", "1:a:0"])
            track_index = 0
        if narration_input_index is not None:
            arguments.extend(["-map", f"{narration_input_index}:a:0"])
            track_index += 1

        # One codec for all audio streams (AAC for broad compatibility).
        arguments.extend(["-c:a", "aac", "-b:a", settings.audio_bitrate])

        # Subtitles.
        if srt_input_index is not None:
            if container == OutputContainer.MP4:
                arguments.extend(
                    ["-map", f"{srt_input_index}:s", "-c:s", "mov_text"]
                )
            else:
                arguments.extend(["-map", f"{srt_input_index}:s", "-c:s", "srt"])

        language = project.settings.language or "und"
        if has_source_audio:
            # Track metadata + default flags. MP4 stores the readable track name
            # in handler_name; MKV uses title. Set both so either container keeps
            # a human-readable label.
            arguments.extend(["-metadata:s:a:0", "title=Original"])
            arguments.extend(["-metadata:s:a:0", "handler_name=Original"])
            arguments.extend(["-metadata:s:a:1", "title=Dubbed Mix"])
            arguments.extend(["-metadata:s:a:1", "handler_name=Dubbed Mix"])
            arguments.extend(["-metadata:s:a:1", f"language={language}"])
            # Mark Dubbed Mix (a:1) as default; original (a:0) not default.
            arguments.extend(["-disposition:a:0", "0", "-disposition:a:1", "default"])
            narration_meta_index = 2
        else:
            # No source audio: Dubbed Mix is track 0 and the default track.
            arguments.extend(["-metadata:s:a:0", "title=Dubbed Mix"])
            arguments.extend(["-metadata:s:a:0", "handler_name=Dubbed Mix"])
            arguments.extend(["-metadata:s:a:0", f"language={language}"])
            arguments.extend(["-disposition:a:0", "default"])
            narration_meta_index = 1
        if narration_input_index is not None:
            arguments.extend(
                [
                    f"-metadata:s:a:{narration_meta_index}",
                    "title=Narration Only",
                    f"-metadata:s:a:{narration_meta_index}",
                    "handler_name=Narration Only",
                ]
            )
        arguments.extend(["-map_metadata", "0"])
        arguments.append(str(output_path))
        return arguments

    def _source_has_audio(self, video_path: Path) -> tuple[bool, str | None]:
        """Return ``(has_audio, probe_error)``.

        ``probe_error`` is set when ffprobe itself failed so the caller can
        refuse to mux instead of blindly assuming an audio stream exists.
        """
        try:
            data = probe_media(video_path, self.ffmpeg_path)
        except FFmpegError as exc:
            return False, str(exc)
        _duration, _video, audio = parse_video_probe(data)
        return bool(audio), None

    @staticmethod
    def _audio_codec(container: OutputContainer, bitrate: str) -> str:
        if container == OutputContainer.MP4:
            return "aac"
        return "aac"
