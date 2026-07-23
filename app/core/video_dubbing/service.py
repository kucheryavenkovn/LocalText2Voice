from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.tts.base import BaseTTSEngine
from app.utils import ffprobe_utils
from app.utils.ffmpeg_utils import FFmpegError

from .audio_mixer import AudioMixer
from .cue_generator import CueGenerator, CueGenerationError
from .duration_fitter import DurationFitter
from .models import (
    Alignment,
    CueStatus,
    DubbingCue,
    DubbingProject,
    DubbingProjectSettings,
    SyncMode,
    VideoProbeInfo,
)
from .preview_renderer import PreviewRenderer
from .project_store import DubbingProjectStore
from .reports import build_report, write_reports
from .srt_parser import (
    SrtParseError,
    SrtWarning,
    cues_from_parsed,
    parse_srt,
    parse_srt_file,
    validate_cues,
)
from .stale_state import (
    invalidate_for_container_change,
    invalidate_for_ducking_change,
    invalidate_for_narration_change,
    invalidate_for_preview_change,
    invalidate_for_text_change,
    mark_clean,
)
from .timeline_renderer import TimelineRenderer
from .video_muxer import VideoMuxError, VideoMuxer


ProgressCallback = Callable[[str, int, int, str], None]
LogCallback = Callable[[str], None]


_module_logger = logging.getLogger("video_dubbing")


class VideoDubbingServiceError(RuntimeError):
    pass


@dataclass
class ImportSrtResult:
    cues: list[DubbingCue]
    warnings: list[SrtWarning]


class VideoDubbingService:
    """Orchestrates the video dubbing pipeline.

    The service is the only layer the UI / CLI / API talks to. It wires the TTS
    engine, FFmpeg helpers, project store, fitter, timeline renderer, audio
    mixer, video muxer and preview renderer, and keeps stale-state coherent.
    Business logic never lives in the UI.
    """

    STAGES = (
        "analysis",
        "tts",
        "fitting",
        "narration",
        "ducking_mask",
        "mixing",
        "preview",
        "muxing",
        "report",
    )

    def __init__(
        self,
        tts_engine: BaseTTSEngine,
        store: DubbingProjectStore | None = None,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.tts_engine = tts_engine
        self.store = store or DubbingProjectStore()
        self.progress_callback = progress_callback or (lambda stage, c, t, msg: None)
        self.log_callback = log_callback or (lambda msg: None)
        self._cancel_requested = threading.Event()
        self._active_cue_generator: CueGenerator | None = None
        self._active_timeline: TimelineRenderer | None = None
        self._active_mixer: AudioMixer | None = None
        self._active_muxer: VideoMuxer | None = None
        self._active_preview: PreviewRenderer | None = None
        self._current_operation: str = ""

    def _log(self, message: str, level: int = logging.INFO) -> None:
        self.log_callback(message)
        _module_logger.log(level, message)

    def _begin_operation(self, name: str, **context: Any) -> None:
        self._current_operation = name
        detail = " ".join(f"{k}={v}" for k, v in context.items())
        _module_logger.info("OPERATION START: %s %s", name, detail)

    def _end_operation(self, name: str, **context: Any) -> None:
        self._current_operation = ""
        detail = " ".join(f"{k}={v}" for k, v in context.items())
        _module_logger.info("OPERATION END: %s %s", name, detail)

    @property
    def current_operation(self) -> str:
        return self._current_operation

    def cancel(self) -> None:
        self._cancel_requested.set()
        for component in (
            self._active_cue_generator,
            self._active_timeline,
            self._active_mixer,
            self._active_muxer,
            self._active_preview,
        ):
            if component is not None:
                try:
                    component.cancel()
                except Exception:  # pragma: no cover - defensive
                    pass

    def _check_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise VideoDubbingServiceError("Operation cancelled.")

    # ------------------------------------------------------------------ create

    def create_project(
        self,
        project_dir: Path,
        title: str = "Video Dubbing",
        settings: DubbingProjectSettings | None = None,
    ) -> DubbingProject:
        project_id = self.store.create_project(project_dir, title=title)
        project = DubbingProject(
            project_id=project_id,
            project_dir=project_dir,
            title=title,
            settings=settings or DubbingProjectSettings(),
        )
        project.ensure_directories()
        self.store.save_project(project)
        self.log_callback(f"Created video dubbing project: {project_dir}")
        return project

    def load_project(self, project_id: str) -> DubbingProject:
        project = self.store.load_project(project_id)
        if project is None:
            raise VideoDubbingServiceError(f"Project not found: {project_id}")
        return project

    def open_project_manifest(self, manifest_path: Path) -> DubbingProject:
        return self.store.load_project_from_manifest(manifest_path)

    def list_projects(self) -> list[dict[str, Any]]:
        return self.store.list_projects()

    def delete_project(self, project_id: str) -> None:
        self.store.delete_project(project_id)

    def save_project(self, project: DubbingProject) -> None:
        self.store.save_project(project)

    # ------------------------------------------------------------------ import

    def attach_video(self, project: DubbingProject, video_path: Path) -> VideoProbeInfo:
        if not video_path.is_file():
            raise VideoDubbingServiceError(f"Video not found: {video_path}")
        project.video_path = video_path
        data = ffprobe_utils.probe_media(video_path, project.settings.ffmpeg_path)
        duration_ms, video_stream, audio_streams = ffprobe_utils.parse_video_probe(data)
        probe = VideoProbeInfo(
            duration_ms=duration_ms,
            video_codec=video_stream.codec_name if video_stream else "",
            audio_codec=audio_streams[0].codec_name if audio_streams else "",
            audio_tracks=len(audio_streams),
            sample_rate=audio_streams[0].sample_rate if audio_streams else 0,
            channels=audio_streams[0].channels if audio_streams else 0,
            fps=video_stream.fps() if video_stream else 0.0,
            container_format=str(data.get("format", {}).get("format_name", "")),
            width=video_stream.width if video_stream else 0,
            height=video_stream.height if video_stream else 0,
        )
        project.video_probe = probe
        invalidate_for_narration_change(project.stale)
        self.store.save_project(project)
        self.log_callback(
            f"Video attached: {video_path.name}, duration={duration_ms} ms, "
            f"video={probe.video_codec}, audio_tracks={probe.audio_tracks}."
        )
        return probe

    def import_srt(
        self,
        project: DubbingProject,
        srt_path: Path,
    ) -> ImportSrtResult:
        if not srt_path.is_file():
            raise VideoDubbingServiceError(f"SRT not found: {srt_path}")
        parsed, parse_warnings = parse_srt_file(srt_path)
        video_duration_ms = project.duration_ms or None
        warnings = validate_cues(parsed, video_duration_ms=video_duration_ms)
        cues = cues_from_parsed(parsed)
        for cue in cues:
            cue.raw_audio_path = project.cue_raw_path(cue.sequence)
            cue.fitted_audio_path = project.cue_fitted_path(cue.sequence)
        project.cues = cues
        project.srt_source_path = srt_path
        project.srt_source_text = srt_path.read_text(encoding="utf-8-sig")
        # persist a copy of the imported SRT
        project.ensure_directories()
        target = project.source_dir() / "subtitles.srt"
        target.write_text(project.srt_source_text, encoding="utf-8")
        invalidate_for_text_change(project.stale)
        self.store.save_project(project)
        self.log_callback(
            f"SRT imported: {len(cues)} cue(s), "
            f"{len(warnings)} validation warning(s)."
        )
        return ImportSrtResult(cues=cues, warnings=warnings + parse_warnings)

    # ------------------------------------------------------------------ analyze

    def analyze_project(self, project: DubbingProject) -> list[SrtWarning]:
        warnings: list[SrtWarning] = []
        parsed, _ = parse_srt(project.srt_source_text or "")
        warnings.extend(validate_cues(parsed, video_duration_ms=project.duration_ms or None))
        for cue in project.cues:
            if not cue.spoken_text.strip():
                warnings.append(
                    SrtWarning(
                        code="empty_text",
                        message=f"Cue #{cue.sequence} has empty text.",
                        sequence=cue.sequence,
                    )
                )
        self.log_callback(f"Analysis complete: {len(warnings)} warning(s).")
        return warnings

    # ------------------------------------------------------------------ generate

    def _resolve_voice_config(self, project: DubbingProject) -> dict[str, Any]:
        config = dict(project.settings.voice_config)
        config.setdefault("engine", project.settings.tts_engine)
        return config

    def generate_cue(
        self,
        project: DubbingProject,
        cue: DubbingCue,
    ) -> DubbingCue:
        voice_config = self._resolve_voice_config(project)
        generator = CueGenerator(
            self.tts_engine,
            ffmpeg_path=project.settings.ffmpeg_path,
            progress_callback=lambda c, t, msg: self.progress_callback("tts", c, t, msg),
            log_callback=self.log_callback,
        )
        self._active_cue_generator = generator
        try:
            generator.generate_raw(cue, voice_config)
            fitter = DurationFitter(project.settings)
            result = fitter.evaluate(cue)
            fitter.apply_result(cue, result)
            generator.apply_fitting(cue, result)
            invalidate_for_narration_change(project.stale)
        except CueGenerationError as exc:
            cue.status = CueStatus.FAILED.value
            cue.error_message = str(exc)
            raise VideoDubbingServiceError(str(exc)) from exc
        finally:
            self._active_cue_generator = None
        self.store.save_project(project)
        return cue

    def generate_selected(
        self,
        project: DubbingProject,
        sequences: list[int],
    ) -> list[DubbingCue]:
        targets = [c for c in project.cues if c.sequence in sequences and c.enabled]
        self._begin_operation("generate_selected", count=len(targets))
        try:
            return self._generate_many(project, targets)
        finally:
            self._end_operation("generate_selected", count=len(targets))

    def generate_all(self, project: DubbingProject) -> list[DubbingCue]:
        targets = [c for c in project.cues if c.enabled]
        self._begin_operation("generate_all", cues=len(targets), total=len(project.cues))
        try:
            return self._generate_many(project, targets)
        finally:
            self._end_operation("generate_all", cues=len(targets))

    def _generate_many(
        self,
        project: DubbingProject,
        cues: list[DubbingCue],
    ) -> list[DubbingCue]:
        voice_config = self._resolve_voice_config(project)
        fitter = DurationFitter(project.settings)
        generator = CueGenerator(
            self.tts_engine,
            ffmpeg_path=project.settings.ffmpeg_path,
            progress_callback=lambda c, t, msg: self.progress_callback("tts", c, t, msg),
            log_callback=self.log_callback,
        )
        self._active_cue_generator = generator
        total = len(cues)
        try:
            for index, cue in enumerate(cues, start=1):
                self._check_cancelled()
                self.progress_callback("tts", index - 1, total, f"Cue #{cue.sequence}")
                try:
                    generator.generate_raw(cue, voice_config)
                    result = fitter.evaluate(cue)
                    fitter.apply_result(cue, result)
                    generator.apply_fitting(cue, result)
                except CueGenerationError as exc:
                    cue.status = CueStatus.FAILED.value
                    cue.error_message = str(exc)
                    self.log_callback(f"Cue #{cue.sequence} failed: {exc}")
        finally:
            self._active_cue_generator = None
        invalidate_for_narration_change(project.stale)
        self.store.save_project(project)
        return cues

    # ------------------------------------------------------------------ edit

    def update_cue_text(self, project: DubbingProject, sequence: int, text: str) -> None:
        cue = self._find_cue(project, sequence)
        if cue is None:
            raise VideoDubbingServiceError(f"Cue #{sequence} not found.")
        cue.spoken_text = text
        cue.mark_stale()
        invalidate_for_text_change(project.stale)
        self.store.save_project(project)

    def update_cue_timing(
        self,
        project: DubbingProject,
        sequence: int,
        start_ms: int,
        end_ms: int,
    ) -> None:
        cue = self._find_cue(project, sequence)
        if cue is None:
            raise VideoDubbingServiceError(f"Cue #{sequence} not found.")
        if end_ms <= start_ms:
            raise VideoDubbingServiceError("End time must be after start time.")
        cue.start_ms = start_ms
        cue.end_ms = end_ms
        cue.duration_budget_ms = end_ms - start_ms
        invalidate_for_narration_change(project.stale)
        self.store.save_project(project)

    def set_cue_enabled(
        self,
        project: DubbingProject,
        sequence: int,
        enabled: bool,
    ) -> None:
        cue = self._find_cue(project, sequence)
        if cue is None:
            raise VideoDubbingServiceError(f"Cue #{sequence} not found.")
        cue.enabled = enabled
        cue.status = CueStatus.DISABLED.value if not enabled else CueStatus.PENDING.value
        invalidate_for_narration_change(project.stale)
        self.store.save_project(project)

    def merge_cues(
        self,
        project: DubbingProject,
        first_sequence: int,
        second_sequence: int,
    ) -> DubbingCue:
        first = self._find_cue(project, first_sequence)
        second = self._find_cue(project, second_sequence)
        if first is None or second is None:
            raise VideoDubbingServiceError("Both cues must exist to merge.")
        if second.start_ms < first.start_ms:
            first, second = second, first
        first.end_ms = second.end_ms
        first.duration_budget_ms = first.end_ms - first.start_ms
        first.spoken_text = f"{first.spoken_text} {second.spoken_text}".strip()
        first.source_text = first.spoken_text
        first.mark_stale()
        project.cues = [c for c in project.cues if c.sequence != second.sequence]
        invalidate_for_text_change(project.stale)
        self.store.save_project(project)
        return first

    @staticmethod
    def _find_cue(project: DubbingProject, sequence: int) -> DubbingCue | None:
        return next((c for c in project.cues if c.sequence == sequence), None)

    # ------------------------------------------------------------------ render

    def render_narration(
        self,
        project: DubbingProject,
        output_path: Path | None = None,
    ) -> Path:
        self._begin_operation("render_narration", duration_ms=project.duration_ms)
        if output_path is None:
            output_path = project.render_dir() / "narration.wav"
        renderer = TimelineRenderer(
            ffmpeg_path=project.settings.ffmpeg_path,
            sample_rate=project.settings.sample_rate,
            channels=project.settings.channels,
            progress_callback=lambda c, t, msg: self.progress_callback(
                "narration", c, t, msg
            ),
            log_callback=self.log_callback,
        )
        self._active_timeline = renderer
        try:
            renderer.render(
                project,
                output_path,
                alignment=project.settings.preview.alignment,
            )
        finally:
            self._active_timeline = None
        project.narration_wav = output_path
        project.stale.narration = False
        # encode narration mp3
        mixer = self._build_mixer(project, stage="narration")
        narration_mp3 = project.render_dir() / "narration.mp3"
        try:
            mixer.encode_narration_mp3(output_path, narration_mp3)
            project.narration_mp3 = narration_mp3
        except Exception as exc:
            self.log_callback(f"Narration MP3 encoding skipped: {exc}")
        self.store.save_project(project)
        return output_path

    def render_dubbed_mix(
        self,
        project: DubbingProject,
        output_path: Path | None = None,
    ) -> Path:
        self._begin_operation("render_dubbed_mix")
        if project.narration_wav is None or not Path(project.narration_wav).is_file():
            self.render_narration(project)
        if output_path is None:
            output_path = project.render_dir() / "dubbed_mix.wav"
        mixer = self._build_mixer(project, stage="mixing")
        self._active_mixer = mixer
        try:
            mixer.render_dubbed_mix(
                project,
                Path(project.narration_wav),
                output_path,
            )
        finally:
            self._active_mixer = None
        project.dubbed_mix_wav = output_path
        project.stale.mix = False
        invalidate_for_preview_change(project.stale)
        project.stale.video = True
        self.store.save_project(project)
        return output_path

    def render_cue_preview(
        self,
        project: DubbingProject,
        sequence: int,
    ) -> Path:
        self._begin_operation("render_cue_preview", sequence=sequence)
        cue = self._find_cue(project, sequence)
        if cue is None:
            raise VideoDubbingServiceError(f"Cue #{sequence} not found.")
        renderer = self._build_preview(project)
        self._active_preview = renderer
        try:
            path = renderer.render_cue_preview(project, cue)
        finally:
            self._active_preview = None
        self.store.save_project(project)
        return path

    def render_full_preview(self, project: DubbingProject) -> Path:
        self._begin_operation("render_full_preview")
        if project.dubbed_mix_wav is None or not Path(project.dubbed_mix_wav).is_file():
            self.render_dubbed_mix(project)
        renderer = self._build_preview(project)
        self._active_preview = renderer
        try:
            path = renderer.render_full_preview(project)
        finally:
            self._active_preview = None
        project.stale.preview = False
        self.store.save_project(project)
        return path

    def export_video(
        self,
        project: DubbingProject,
        output_path: Path | None = None,
    ) -> Path:
        self._begin_operation(
            "export_video",
            container=project.settings.export.container.value,
        )
        if project.video_path is None:
            raise VideoDubbingServiceError("No source video attached.")
        if project.dubbed_mix_wav is None or not Path(project.dubbed_mix_wav).is_file():
            self.render_dubbed_mix(project)
        muxer = VideoMuxer(
            ffmpeg_path=project.settings.ffmpeg_path,
            progress_callback=lambda c, t, msg: self.progress_callback(
                "muxing", c, t, msg
            ),
            log_callback=self.log_callback,
        )
        self._active_muxer = muxer
        narration_only = (
            project.narration_wav
            if project.settings.export.include_narration_only
            else None
        )
        srt_path = (
            project.source_dir() / "subtitles.srt"
            if project.settings.export.embed_subtitles
            else None
        )
        if srt_path is not None and not srt_path.is_file():
            srt_path = None
        try:
            result = muxer.mux(
                project,
                output_path=output_path,
                narration_only_wav=narration_only,
                srt_path=srt_path,
            )
        except VideoMuxError as exc:
            raise VideoDubbingServiceError(str(exc)) from exc
        finally:
            self._active_muxer = None
        project.final_video_path = result.output_path
        project.stale.video = False
        report = build_report(project, audio_tracks=result.tracks)
        write_reports(project, report)
        mark_clean(project.stale)
        self.store.save_project(project)
        self.log_callback(f"Final video exported: {result.output_path}")
        return result.output_path

    def write_report(self, project: DubbingProject) -> tuple[Path, Path]:
        report = build_report(project)
        return write_reports(project, report)

    # ------------------------------------------------------------------ helpers

    def _build_mixer(self, project: DubbingProject, stage: str) -> AudioMixer:
        return AudioMixer(
            ffmpeg_path=project.settings.ffmpeg_path,
            sample_rate=project.settings.sample_rate,
            channels=project.settings.channels,
            progress_callback=lambda c, t, msg: self.progress_callback(stage, c, t, msg),
            log_callback=self.log_callback,
        )

    def _build_preview(self, project: DubbingProject) -> PreviewRenderer:
        return PreviewRenderer(
            ffmpeg_path=project.settings.ffmpeg_path,
            sample_rate=project.settings.sample_rate,
            channels=project.settings.channels,
            progress_callback=lambda c, t, msg: self.progress_callback(
                "preview", c, t, msg
            ),
            log_callback=self.log_callback,
        )

    def update_ducking_settings(
        self,
        project: DubbingProject,
        **kwargs: Any,
    ) -> None:
        ducking = project.settings.ducking
        for key, value in kwargs.items():
            if hasattr(ducking, key):
                setattr(ducking, key, value)
        invalidate_for_ducking_change(project.stale)
        self.store.save_project(project)

    def update_export_container(
        self,
        project: DubbingProject,
        container_value: str,
    ) -> None:
        from .models import OutputContainer

        try:
            project.settings.export.container = OutputContainer(container_value)
        except ValueError as exc:
            raise VideoDubbingServiceError(f"Unknown container: {container_value}") from exc
        invalidate_for_container_change(project.stale)
        self.store.save_project(project)

    def can_export(self, project: DubbingProject) -> tuple[bool, list[str]]:
        blockers: list[str] = []
        if project.settings.sync_mode == SyncMode.STRICT:
            for cue in project.cues:
                if not cue.enabled:
                    continue
                if cue.status == CueStatus.NEEDS_SHORTENING.value:
                    blockers.append(
                        f"Cue #{cue.sequence} needs text shortening "
                        f"(overflow {cue.overflow_ms} ms)."
                    )
                if cue.status == CueStatus.FAILED.value:
                    blockers.append(f"Cue #{cue.sequence} failed: {cue.error_message}")
        if project.video_path is None:
            blockers.append("No source video attached.")
        if project.dubbed_mix_wav is None:
            blockers.append("Dubbed mix not rendered.")
        return (len(blockers) == 0, blockers)
