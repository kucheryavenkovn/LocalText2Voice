from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from app.observability import (
    OperationWatchdog,
    RunDirectory,
    bind,
    emit_event,
    log_resource_snapshot,
    new_operation_id,
    new_session_id,
    operation_span,
    set_app_session_id,
    sha256_text,
)
from app.observability.redaction import safe_voice_snapshot
from app.tts.base import BaseTTSEngine
from app.utils import ffprobe_utils

from .audio_mixer import AudioMixer, AudioMixerCancelled
from .cue_generator import CueGenerator, CueGenerationError
from .duration_fitter import DurationFitter
from .elastic_timing import ElasticTimingPlanner
from .tempo_smoothing import TempoSmoothingPlanner
from .fingerprints import (
    extract_reference_path,
    file_content_hash,
    fit_fingerprint,
    fit_settings_fingerprint,
    legacy_combined_fingerprint,
    raw_generation_fingerprint,
)
from .generation import (
    CueErrorPolicy,
    GenerationCancelled,
    GenerationContext,
    GenerationRunResult,
    GenerationRunStatus,
)
from .models import (
    BLOCKING_STATUSES,
    READY_STATUSES,
    CueStatus,
    CueTimingMode,
    DubbingCue,
    DubbingProject,
    DubbingProjectSettings,
    VideoProbeInfo,
)
from .preview_renderer import PreviewRenderer, PreviewRenderCancelled
from .project_store import DubbingProjectStore
from .reports import build_report, write_reports
from .srt_parser import (
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
from .timeline_renderer import TimelineRenderer, TimelineRenderCancelled
from .video_muxer import VideoMuxCancelled, VideoMuxError, VideoMuxer
from .wav_validator import WavArtifactValidator, cleanup_part_files


ProgressCallback = Callable[[str, int, int, str], None]
LogCallback = Callable[[str], None]
CueUpdatedCallback = Callable[[int, str, int, int, int], None]

_module_logger = logging.getLogger("video_dubbing")


def _fingerprint(cue: DubbingCue, settings: DubbingProjectSettings) -> str:
    """Backward-compatible raw provenance fingerprint."""
    return legacy_combined_fingerprint(cue, settings)


def _rel_to_project(path: Path | str | None, project: DubbingProject) -> str:
    """Return ``path`` relative to the project dir (or its name if outside).

    Used by diagnostics so events carry stable, portable paths instead of
    absolute user paths.
    """
    if not path:
        return ""
    try:
        return str(Path(path).relative_to(project.project_dir)).replace("\\", "/")
    except (ValueError, TypeError):
        return Path(path).name


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
        cue_updated_callback: CueUpdatedCallback | None = None,
    ) -> None:
        self.tts_engine = tts_engine
        self.store = store or DubbingProjectStore()
        self.progress_callback = progress_callback or (lambda stage, c, t, msg: None)
        self.log_callback = log_callback or (lambda msg: None)
        self.cue_updated_callback = cue_updated_callback or (
            lambda seq, status, raw, fitted, index: None
        )
        self._cancel_requested = threading.Event()
        self._save_lock = threading.RLock()
        # True only while the worker thread is actively writing cue rows
        # (generate / re-fit). While True, save_project() degrades to
        # update_project_metadata() so the UI autosave can never DELETE+reinsert
        # the cue table and roll back a cue the worker just checkpointed.
        self._generation_active = False
        self._active_cue_generator: CueGenerator | None = None
        self._active_timeline: TimelineRenderer | None = None
        self._active_mixer: AudioMixer | None = None
        self._active_muxer: VideoMuxer | None = None
        self._active_preview: PreviewRenderer | None = None
        self._current_operation: str = ""
        self._active_generation_context: GenerationContext | None = None
        self._wav_validator = WavArtifactValidator()
        self._generation_log: list[dict[str, Any]] = []
        # Observability: a stable session id for the whole process, plus an
        # optional run directory bound while a generation/heavy op is active.
        self._app_session_id = new_session_id()
        set_app_session_id(self._app_session_id)
        self._active_run_dir: RunDirectory | None = None
        self._watchdog: OperationWatchdog | None = None

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

    # ------------------------------------------------------------------ observability

    @contextmanager
    def _diagnostic_run(
        self,
        project: DubbingProject | None,
        run_id: str | None = None,
    ) -> Iterator[RunDirectory | None]:
        """Open a per-run diagnostic directory and bind it as the event sink.

        Events emitted anywhere in the body (cue loop, FFmpeg runner, resource
        snapshots) flow into ``<project>/logs/run_<run_id>/events.jsonl``. If
        the project has no directory yet (early create/load), events degrade to
        the application log only — they are never lost silently.
        """
        rid = run_id or new_operation_id()
        project_id = project.project_id if project is not None else ""
        if project is None or not Path(project.project_dir).exists():
            with bind(app_session_id=self._app_session_id, project_id=project_id, run_id=rid):
                yield None
            return
        run_dir = RunDirectory.create(project.project_dir, rid)
        self._active_run_dir = run_dir
        try:
            with run_dir.bind(), bind(
                app_session_id=self._app_session_id,
                project_id=project_id,
                run_id=rid,
            ):
                yield run_dir
        finally:
            try:
                run_dir.finalize()
            except Exception:  # pragma: no cover
                pass
            if self._active_run_dir is run_dir:
                self._active_run_dir = None

    def _snapshot_resources(self, project: DubbingProject | None, *, label: str) -> None:
        """Record a resource snapshot at an operation boundary."""
        try:
            log_resource_snapshot(
                project.project_dir if project is not None else None,
                label=label,
            )
        except Exception:  # pragma: no cover - diagnostics must never crash
            pass

    def _tts_snapshot(
        self,
        context: GenerationContext,
        *,
        text_length: int,
        text_sha256: str,
    ) -> dict[str, Any]:
        """Redacted TTS configuration snapshot emitted before each synthesis.

        Never contains API keys, bearer tokens, full subtitle text or full
        reference transcripts — only their hashes / lengths.
        """
        ref_path = context.reference_voice_path
        snapshot = {
            "engine_id": context.engine_id,
            "engine_class": type(self.tts_engine).__name__,
            "model_id": context.model_id,
            "voice_id": context.voice_id,
            "language": context.language,
            "sample_rate": context.sample_rate,
            "channels": context.channels,
            "reference_voice_filename": Path(ref_path).name if ref_path else None,
            "reference_voice_sha256": context.reference_voice_content_hash,
            "text_length": text_length,
            "text_sha256": text_sha256,
            "speed": context.speed,
            "voice_config": safe_voice_snapshot(context.voice_config),
        }
        emit_event("tts.snapshot", payload=snapshot, force_flush=False)
        return snapshot

    @property
    def active_run_directory(self) -> RunDirectory | None:
        return self._active_run_dir

    def _arm_watchdog(self, label: str, *, timeout_seconds: float = 180.0) -> None:
        """Arm a stall watchdog for an active long-running operation."""
        self._cancel_watchdog()
        try:
            self._watchdog = OperationWatchdog(
                timeout_seconds=timeout_seconds, label=label
            )
            self._watchdog.arm()
        except Exception:  # pragma: no cover
            self._watchdog = None

    def _heartbeat_watchdog(self) -> None:
        if self._watchdog is not None:
            try:
                self._watchdog.heartbeat()
            except Exception:  # pragma: no cover
                pass

    def _cancel_watchdog(self) -> None:
        if self._watchdog is not None:
            try:
                self._watchdog.cancel()
            except Exception:  # pragma: no cover
                pass
            self._watchdog = None

    @property
    def current_operation(self) -> str:
        return self._current_operation

    def cancel(self) -> None:
        self._cancel_requested.set()
        emit_event("operation.cancel_requested", force_flush=True)
        self._cancel_watchdog()
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
                except Exception as exc:  # pragma: no cover - defensive
                    # cancel() failures must never be swallowed silently.
                    _module_logger.exception(
                        "cancel component %s failed: %s", type(component).__name__, exc
                    )

    def reset_cancel(self) -> None:
        """Allow a new operation after the previous one was cancelled."""
        self._cancel_requested.clear()
        # Engines keep a sticky cancel flag; clear it or every later TTS is a no-op.
        engine = self.tts_engine
        if engine is not None:
            clearer = getattr(engine, "clear_cancel", None)
            if callable(clearer):
                try:
                    clearer()
                except Exception:
                    pass
            else:
                event = getattr(engine, "_cancel_requested", None)
                clear = getattr(event, "clear", None)
                if callable(clear):
                    try:
                        clear()
                    except Exception:
                        pass

    @staticmethod
    def _engine_class_family(engine_name: str) -> str:
        """Map an engine class name to its canonical family id."""
        engine_name = engine_name.casefold()
        class_map = {
            "piperttsengine": "piper",
            "omnivoicettsengine": "omnivoice",
            "chatterboxttsengine": "chatterbox",
            "kokoropythonttsengine": "kokoro",
            "kokorottsengine": "kokoro",
            "qwenttsengine": "qwen",
        }
        if engine_name in class_map:
            return class_map[engine_name]
        for key, family in class_map.items():
            if key in engine_name:
                return family
        return ""

    _ENGINE_FAMILY_ALIASES = {
        "piper": "piper",
        "omnivoice": "omnivoice",
        "chatterbox": "chatterbox",
        "kokoro": "kokoro",
        "kokoro_python": "kokoro",
        "qwen": "qwen",
    }

    def _engine_mismatch_message(
        self,
        engine: BaseTTSEngine | None,
        expected_engine_id: str,
    ) -> str | None:
        """Return an error message iff we have *positive* evidence of mismatch.

        Identity resolution order:

        1. ``engine.engine_id`` — authoritative when set (covers API engines and
           in-process runtimes that declare their id). A direct equality check
           avoids false mismatches for engines whose class name does not encode
           the family (e.g. test doubles, ``HttpTTSEngine``).
        2. Class-name family mapping for engines without ``engine_id``.

        An engine whose family cannot be determined (unrecognised class name and
        no ``engine_id``) is **not** treated as a mismatch: ``validate()`` is
        the authority there, and hard-failing would make every third-party or
        test double mass-fail with a misleading "Engine mismatch".
        """
        if engine is None:
            return "No TTS engine bound to the service."
        expected = str(expected_engine_id or "").casefold()
        if not expected:
            return None
        want = self._ENGINE_FAMILY_ALIASES.get(expected, expected)
        explicit = str(getattr(engine, "engine_id", "") or "").casefold()
        if explicit:
            resolved = self._ENGINE_FAMILY_ALIASES.get(explicit, explicit)
            if resolved == want or want in resolved or resolved in want:
                return None
            return (
                f"Engine mismatch: project wants '{expected_engine_id}' but the "
                f"bound engine identifies as '{explicit}' "
                f"({type(engine).__name__}). Re-select the TTS engine."
            )
        engine_name = type(engine).__name__
        resolved = self._engine_class_family(engine_name)
        if not resolved:
            # Unrecognised engine: cannot prove a mismatch — let validate() decide.
            return None
        if resolved == want or want in resolved or resolved in want:
            return None
        return (
            f"Engine mismatch: project wants '{expected_engine_id}' but service "
            f"has {engine_name} (family '{resolved}'). Re-select the TTS engine."
        )

    def _check_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise GenerationCancelled("Operation cancelled.")

    @property
    def is_generation_active(self) -> bool:
        """True while the worker is actively writing cue rows.

        The UI uses this (and the page uses ``_generation_busy``) to route
        autosave through metadata-only persistence so a full save cannot roll
        back a cue the worker just checkpointed.
        """
        return self._generation_active

    @property
    def active_generation_context(self) -> GenerationContext | None:
        return self._active_generation_context

    def build_generation_context(
        self, project: DubbingProject
    ) -> GenerationContext:
        settings = project.settings
        voice_config = dict(self._resolve_voice_config(project))
        ref_path = extract_reference_path(voice_config)
        ref_hash = file_content_hash(ref_path) if ref_path else None
        # Keep speed unset when absent so fingerprints match legacy payloads.
        speed_raw = voice_config.get("speed")
        speed = float(speed_raw) if speed_raw is not None else 1.0
        speed_for_fp = float(speed_raw) if speed_raw is not None else None
        seed_fp = raw_generation_fingerprint(
            DubbingCue("seed", 0, 0, 1, 1, "", ""),
            engine_id=settings.tts_engine,
            voice_id=settings.voice,
            voice_config=voice_config,
            language=settings.language,
            sample_rate=settings.sample_rate,
            channels=settings.channels,
            reference_voice_content_hash=ref_hash,
            speed=speed_for_fp,
        )
        return GenerationContext(
            run_id=GenerationContext.new_run_id(),
            engine_id=settings.tts_engine,
            model_id=str(voice_config.get("model") or voice_config.get("model_id") or "")
            or None,
            voice_id=settings.voice or None,
            voice_display_name=settings.voice or None,
            voice_config=voice_config,
            reference_voice_path=ref_path,
            reference_voice_content_hash=ref_hash,
            language=settings.language,
            speed=speed,
            compress_internal_pauses=settings.compress_internal_pauses,
            internal_pause_keep_ms=settings.internal_pause_keep_ms,
            sample_rate=settings.sample_rate,
            channels=settings.channels,
            preferred_speed_limit=settings.preferred_speed_limit,
            hard_speed_limit=settings.hard_speed_limit,
            guard_gap_ms=settings.guard_gap_ms,
            sync_mode=settings.sync_mode.value,
            ffmpeg_path=settings.ffmpeg_path,
            raw_generation_fingerprint_seed=seed_fp,
            fit_settings_fingerprint=fit_settings_fingerprint(settings),
        )

    def recover_project_artifacts(self, project: DubbingProject) -> list[Path]:
        """Drop leftover .part files so they are never treated as ready audio."""
        recovered = cleanup_part_files(
            project.cues_dir(),
            move_to=project.temp_dir() / "recovery",
        )
        if recovered:
            self._log(
                f"Recovered {len(recovered)} leftover .part artifact(s).",
                level=logging.WARNING,
            )
        return recovered

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
        self.recover_project_artifacts(project)
        self.backfill_legacy_fingerprints(project)
        for cue in project.cues:
            cue.ensure_source_timing()
        return project

    def open_project_manifest(self, manifest_path: Path) -> DubbingProject:
        project = self.store.load_project_from_manifest(manifest_path)
        self.recover_project_artifacts(project)
        self.backfill_legacy_fingerprints(project)
        for cue in project.cues:
            cue.ensure_source_timing()
        return project

    def backfill_legacy_fingerprints(self, project: DubbingProject) -> None:
        """Flag legacy cues whose raw audio exists but whose provenance cannot
        be verified. Does NOT invent a fingerprint — confirmed fingerprints are
        recorded only after a real synthesis. The flag drives the UI warning
        and forces TTS when voice/text later change."""
        changed = False
        for cue in project.cues:
            if not cue.generation_fingerprint and self._raw_intact(cue):
                if not cue.legacy_audio_unverified:
                    cue.legacy_audio_unverified = True
                    changed = True
                if not cue.legacy_observed_fingerprint:
                    cue.legacy_observed_fingerprint = _fingerprint(cue, project.settings)
                    changed = True
        if changed:
            self.store.save_project(project)

    def list_projects(self) -> list[dict[str, Any]]:
        return self.store.list_projects()

    def delete_project(self, project_id: str) -> None:
        self.store.delete_project(project_id)

    def save_project(self, project: DubbingProject) -> None:
        """Persist the project.

        During active generation a full save would ``DELETE``+reinsert every
        cue row from the (possibly stale) UI snapshot, racing the worker's
        per-cue ``upsert_cue`` and rolling back just-finished cues. When
        generation is active we therefore persist **metadata only** — cue rows
        are owned by the worker thread until it finishes.
        """
        with self._save_lock:
            if self._generation_active:
                self._log(
                    "save_project during active generation: metadata only "
                    "(cue rows preserved by worker)."
                )
                self.store.update_project_metadata(project)
                return
            self.store.save_project(project)

    def save_project_metadata(self, project: DubbingProject) -> None:
        """Persist project metadata without touching cue rows (autosave-safe)."""
        with self._save_lock:
            self.store.update_project_metadata(project)

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
            cue.ensure_source_timing()
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

    def _cue_is_complete(self, cue: DubbingCue) -> bool:
        """True if the cue has intact raw+fitted WAV and recorded durations."""
        if cue.status not in READY_STATUSES and cue.status != CueStatus.FITTED.value:
            return False
        if not self._raw_intact(cue):
            return False
        if not self._fitted_intact(cue):
            return False
        return True

    def _raw_intact(self, cue: DubbingCue) -> bool:
        if cue.raw_audio_path is None:
            return False
        path = Path(cue.raw_audio_path)
        if not path.is_file():
            return False
        result = self._wav_validator.validate(path)
        if not result.valid:
            return False
        if cue.raw_duration_ms is None or cue.raw_duration_ms <= 0:
            cue.raw_duration_ms = result.duration_ms
        return True

    def _fitted_intact(self, cue: DubbingCue) -> bool:
        if cue.fitted_audio_path is None:
            return False
        path = Path(cue.fitted_audio_path)
        if not path.is_file():
            return False
        result = self._wav_validator.validate(path)
        if not result.valid:
            return False
        if cue.fitted_duration_ms is None or cue.fitted_duration_ms <= 0:
            cue.fitted_duration_ms = result.duration_ms
        return True

    def _raw_fp_for(
        self,
        cue: DubbingCue,
        context: GenerationContext | None = None,
        project: DubbingProject | None = None,
    ) -> str:
        if context is not None:
            speed_raw = context.voice_config.get("speed")
            speed_for_fp = float(speed_raw) if speed_raw is not None else None
            return raw_generation_fingerprint(
                cue,
                engine_id=context.engine_id,
                voice_id=context.voice_id,
                voice_config=context.voice_config,
                language=context.language,
                sample_rate=context.sample_rate,
                channels=context.channels,
                reference_voice_content_hash=context.reference_voice_content_hash,
                speed=speed_for_fp,
            )
        assert project is not None
        return _fingerprint(cue, project.settings)

    def generation_plan(
        self,
        project: DubbingProject,
        force: bool = False,
        context: GenerationContext | None = None,
    ) -> dict[str, Any]:
        """Classify every enabled cue into exact reuse categories."""
        ready_without_changes: list[int] = []
        needs_refit_from_raw: list[int] = []
        needs_tts_generation: list[int] = []
        missing_raw: list[int] = []
        missing_fitted: list[int] = []
        legacy_audio_unverified: list[int] = []
        failed: list[int] = []
        disabled: list[int] = []

        for cue in project.cues:
            if not cue.enabled:
                disabled.append(cue.sequence)
                continue
            if cue.status == CueStatus.FAILED.value:
                failed.append(cue.sequence)

            current_fp = self._raw_fp_for(cue, context=context, project=project)
            raw_intact = self._raw_intact(cue)
            fitted_intact = self._fitted_intact(cue)
            if not raw_intact:
                missing_raw.append(cue.sequence)
                needs_tts_generation.append(cue.sequence)
                continue
            if not fitted_intact:
                missing_fitted.append(cue.sequence)

            provenance_verified = bool(cue.generation_fingerprint) and (
                cue.generation_fingerprint == current_fp
            )
            raw_fp_for_fit = cue.generation_fingerprint or current_fp
            expected_fit_fp = fit_fingerprint(
                cue,
                raw_generation_fp=raw_fp_for_fit,
                raw_wav_hash=cue.raw_wav_hash or file_content_hash(cue.raw_audio_path),
                settings=project.settings,
            )
            fit_verified = bool(cue.fit_fingerprint) and cue.fit_fingerprint == expected_fit_fp

            if force:
                needs_tts_generation.append(cue.sequence)
                continue

            if provenance_verified:
                if fitted_intact and cue.status in READY_STATUSES and fit_verified:
                    ready_without_changes.append(cue.sequence)
                else:
                    needs_refit_from_raw.append(cue.sequence)
                continue

            if not cue.generation_fingerprint:
                if cue.legacy_observed_fingerprint == current_fp:
                    legacy_audio_unverified.append(cue.sequence)
                    if fitted_intact and cue.fit_fingerprint and fit_verified:
                        # Legacy already migrated once via successful refit.
                        ready_without_changes.append(cue.sequence)
                    else:
                        needs_refit_from_raw.append(cue.sequence)
                else:
                    needs_tts_generation.append(cue.sequence)
                continue

            needs_tts_generation.append(cue.sequence)

        return {
            "ready_without_changes": ready_without_changes,
            "needs_refit_from_raw": needs_refit_from_raw,
            "needs_tts_generation": needs_tts_generation,
            "missing_raw": missing_raw,
            "missing_fitted": missing_fitted,
            "legacy_audio_unverified": legacy_audio_unverified,
            "failed": failed,
            "disabled": disabled,
            "ready": ready_without_changes,
            "will_reuse": needs_refit_from_raw,
            "will_generate": needs_tts_generation,
            "stale": needs_refit_from_raw,
            "missing": missing_raw,
        }

    def generate_cue(
        self,
        project: DubbingProject,
        cue: DubbingCue,
        force: bool = False,
    ) -> DubbingCue:
        result = self._generate_many(project, [cue], force=force)
        if result.status == GenerationRunStatus.CANCELLED:
            raise GenerationCancelled(result.error_message or "Cancelled.")
        if result.status == GenerationRunStatus.FAILED and result.error_message:
            raise VideoDubbingServiceError(result.error_message)
        return cue

    def generate_selected(
        self,
        project: DubbingProject,
        sequences: list[int],
        force: bool = True,
        error_policy: CueErrorPolicy | str | None = None,
    ) -> GenerationRunResult:
        targets = [c for c in project.cues if c.sequence in sequences and c.enabled]
        self._begin_operation("generate_selected", count=len(targets), force=force)
        try:
            return self._generate_many(
                project, targets, force=force, error_policy=error_policy
            )
        finally:
            self._end_operation("generate_selected", count=len(targets))

    def generate_all(
        self,
        project: DubbingProject,
        force: bool = False,
        error_policy: CueErrorPolicy | str | None = None,
    ) -> GenerationRunResult:
        context = self.build_generation_context(project)
        with self._diagnostic_run(project, run_id=context.run_id), bind(
            engine_id=context.engine_id,
            engine_class=type(self.tts_engine).__name__,
            run_id=context.run_id,
        ), operation_span(
            "generate_all",
            project_id=project.project_id,
            run_id=context.run_id,
            extra={"force": force, "engine_id": context.engine_id},
        ):
            plan = self.generation_plan(project, force=force, context=context)
            emit_event(
                "generation.plan",
                payload={
                    "ready": len(plan["ready_without_changes"]),
                    "needs_refit": len(plan["needs_refit_from_raw"]),
                    "needs_tts": len(plan["needs_tts_generation"]),
                    "legacy_unverified": len(plan["legacy_audio_unverified"]),
                    "missing_raw": len(plan["missing_raw"]),
                    "failed": len(plan["failed"]),
                    "force": force,
                },
            )
            if force:
                targets = [c for c in project.cues if c.enabled]
            else:
                need = set(plan["needs_tts_generation"]) | set(plan["needs_refit_from_raw"])
                targets = [c for c in project.cues if c.sequence in need]
            self._log(
                f"Ready: {len(plan['ready_without_changes'])} | "
                f"Refit-from-raw: {len(plan['needs_refit_from_raw'])} | "
                f"TTS: {len(plan['needs_tts_generation'])} | "
                f"Legacy unverified: {len(plan['legacy_audio_unverified'])} | "
                f"Missing raw: {len(plan['missing_raw'])} | "
                f"Failed: {len(plan['failed'])}"
            )
            self._begin_operation(
                "generate_all", cues=len(targets), total=len(project.cues), force=force
            )
            self._log(
                "GenerationContext: "
                f"engine={context.engine_id} voice={context.voice_id} "
                f"ref={Path(context.reference_voice_path).name if context.reference_voice_path else '—'} "
                f"ref_hash={(context.reference_voice_content_hash or '')[:12] or '—'} "
                f"lang={context.language}"
            )
            try:
                result = self._generate_many(
                    project,
                    targets,
                    force=force,
                    error_policy=error_policy,
                    context=context,
                )
                # Only post-process when real cue audio was produced.
                if (
                    result.status
                    in {
                        GenerationRunStatus.COMPLETED,
                        GenerationRunStatus.COMPLETED_WITH_ERRORS,
                    }
                    and result.completed_count > 0
                ):
                    tempo_n = self.apply_tempo_smoothing(project)
                    if tempo_n:
                        self._log(f"Tempo smoothing updated {tempo_n} cue(s).")
                    smoothed = self.auto_smooth_neighbors(project)
                    if smoothed:
                        self._log(f"Auto-smoothed {smoothed} elastic group(s).")
                return result
            finally:
                self._end_operation("generate_all", cues=len(targets))

    def re_fit_existing(self, project: DubbingProject) -> GenerationRunResult:
        """Re-fit existing raw WAV under the current speed policy WITHOUT TTS."""
        targets = [
            c
            for c in project.cues
            if c.enabled and self._raw_intact(c)
        ]
        self._begin_operation("re_fit", cues=len(targets))
        context = self.build_generation_context(project)
        self._active_generation_context = context
        fitter = DurationFitter(project.settings, video_duration_ms=project.duration_ms)
        generator = self._make_generator_from_context(context)
        self._active_cue_generator = generator
        cues_sorted = sorted(project.cues, key=lambda c: c.effective_start_ms())
        completed = 0
        failed_ids: list[str] = []
        cancelled_count = 0
        last_id: str | None = None
        status = GenerationRunStatus.COMPLETED
        error_message: str | None = None
        self._generation_active = True
        try:
            for index, cue in enumerate(targets, start=1):
                self._check_cancelled()
                self.progress_callback(
                    "fitting", index - 1, len(targets), f"Cue #{cue.sequence}"
                )
                next_cue = self._next_cue(cues_sorted, cue)
                try:
                    self._refit_one(cue, fitter, next_cue, project, generator, context)
                    self.store.upsert_cue(project.project_id, cue)
                    self.cue_updated_callback(
                        cue.sequence,
                        cue.status,
                        cue.raw_duration_ms or 0,
                        cue.fitted_duration_ms or 0,
                        index,
                    )
                    completed += 1
                    last_id = cue.cue_id
                except GenerationCancelled:
                    cancelled_count = 1
                    if cue.status == CueStatus.RENDERING.value:
                        cue.status = CueStatus.CANCELLED.value
                    self.store.upsert_cue(project.project_id, cue)
                    status = GenerationRunStatus.CANCELLED
                    error_message = "Generation cancelled."
                    raise
                except CueGenerationError as exc:
                    cue.status = CueStatus.FAILED.value
                    cue.error_message = str(exc)
                    failed_ids.append(cue.cue_id)
                    self.store.upsert_cue(project.project_id, cue)
        except GenerationCancelled:
            status = GenerationRunStatus.CANCELLED
            error_message = error_message or "Generation cancelled."
        finally:
            self._active_cue_generator = None
            self._active_generation_context = None
            self._generation_active = False
            self._end_operation("re_fit", cues=len(targets))
        invalidate_for_narration_change(project.stale)
        msg = {
            GenerationRunStatus.CANCELLED: "Re-fit cancelled",
            GenerationRunStatus.COMPLETED_WITH_ERRORS: "Re-fit completed with errors",
            GenerationRunStatus.FAILED: "Re-fit failed",
        }.get(status, "Re-fit complete")
        self.progress_callback("fitting", len(targets), len(targets), msg)
        self.store.write_manifest_atomic(project)
        if failed_ids and status == GenerationRunStatus.COMPLETED:
            status = GenerationRunStatus.COMPLETED_WITH_ERRORS
        if status in {
            GenerationRunStatus.COMPLETED,
            GenerationRunStatus.COMPLETED_WITH_ERRORS,
        }:
            tempo_n = self.apply_tempo_smoothing(project)
            if tempo_n:
                self._log(f"Tempo smoothing updated {tempo_n} cue(s) after re-fit.")
            smoothed = self.auto_smooth_neighbors(project)
            if smoothed:
                self._log(f"Auto-smoothed {smoothed} elastic group(s) after re-fit.")
        return GenerationRunResult(
            status=status,
            total_count=len(targets),
            completed_count=completed,
            failed_count=len(failed_ids),
            cancelled_count=cancelled_count,
            skipped_count=0,
            failed_cue_ids=tuple(failed_ids),
            last_processed_cue_id=last_id,
            error_message=error_message,
            run_id=context.run_id,
        )

    def _resolve_error_policy(
        self,
        project: DubbingProject,
        error_policy: CueErrorPolicy | str | None,
    ) -> CueErrorPolicy:
        if error_policy is None:
            raw = project.settings.cue_error_policy or CueErrorPolicy.CONTINUE.value
        else:
            raw = error_policy.value if isinstance(error_policy, CueErrorPolicy) else str(error_policy)
        try:
            return CueErrorPolicy(raw)
        except ValueError:
            return CueErrorPolicy.CONTINUE

    def _generate_many(
        self,
        project: DubbingProject,
        cues: list[DubbingCue],
        force: bool = False,
        error_policy: CueErrorPolicy | str | None = None,
        context: GenerationContext | None = None,
    ) -> GenerationRunResult:
        import time as _time

        context = context or self.build_generation_context(project)
        policy = self._resolve_error_policy(project, error_policy)
        # Force stops on first hard failure so silent mass-fail is visible —
        # but an explicit CONTINUE policy (set by the user/project) is honoured
        # even under force, so a single bad cue does not abort a forced run.
        if force and error_policy is None and policy != CueErrorPolicy.CONTINUE:
            policy = CueErrorPolicy.STOP
        self.reset_cancel()
        self._active_generation_context = context
        # Fail fast if the live engine object does not match the project engine.
        mismatch_msg = self._engine_mismatch_message(self.tts_engine, context.engine_id)
        if mismatch_msg is not None:
            self._log(mismatch_msg, level=logging.ERROR)
            return GenerationRunResult(
                status=GenerationRunStatus.FAILED,
                total_count=len(cues),
                completed_count=0,
                failed_count=0,
                cancelled_count=0,
                skipped_count=len(cues),
                error_message=mismatch_msg,
                run_id=context.run_id,
            )
        # Validate speaker reference once up front (fail fast, not 149 silent errors).
        if cues:
            ref = context.reference_voice_path
            if context.engine_id in {"omnivoice", "chatterbox"}:
                if not ref or not Path(ref).is_file():
                    msg = (
                        f"Reference voice missing for {context.engine_id}: "
                        f"{ref or '(empty)'}. Select the voice again on the Voice tab."
                    )
                    self._log(msg, level=logging.ERROR)
                    return GenerationRunResult(
                        status=GenerationRunStatus.FAILED,
                        total_count=len(cues),
                        completed_count=0,
                        failed_count=len(cues),
                        cancelled_count=0,
                        skipped_count=0,
                        failed_cue_ids=tuple(c.cue_id for c in cues),
                        last_processed_cue_id=None,
                        error_message=msg,
                        run_id=context.run_id,
                    )
            try:
                self.tts_engine.validate(dict(context.voice_config))
            except Exception as exc:
                msg = (
                    f"TTS engine ({type(self.tts_engine).__name__}) rejected "
                    f"voice config for '{context.engine_id}': {exc}"
                )
                self._log(msg, level=logging.ERROR)
                return GenerationRunResult(
                    status=GenerationRunStatus.FAILED,
                    total_count=len(cues),
                    completed_count=0,
                    failed_count=0,
                    cancelled_count=0,
                    skipped_count=len(cues),
                    error_message=msg,
                    run_id=context.run_id,
                )
        fitter = DurationFitter(project.settings, video_duration_ms=project.duration_ms)
        generator = self._make_generator_from_context(context)
        self._active_cue_generator = generator
        cues_sorted = sorted(project.cues, key=lambda c: c.effective_start_ms())
        total = len(cues)
        completed = 0
        failed_ids: list[str] = []
        cancelled_count = 0
        last_id: str | None = None
        status = GenerationRunStatus.COMPLETED
        error_message: str | None = None
        stopped_on_error = False
        batch_started = _time.perf_counter()
        self._generation_active = True
        self._snapshot_resources(project, label="before_generation")
        self._arm_watchdog("generate", timeout_seconds=300.0)
        try:
            for index, cue in enumerate(cues, start=1):
                self._check_cancelled()
                self._heartbeat_watchdog()
                with bind(cue_id=cue.cue_id, cue_sequence=cue.sequence, attempt=index):
                    self.progress_callback("tts", index - 1, total, f"Cue #{cue.sequence}")
                    self.cue_updated_callback(
                        cue.sequence, CueStatus.RENDERING.value, 0, 0, index
                    )
                    emit_event(
                        "cue.started",
                        payload={
                            "sequence": cue.sequence,
                            "cue_id": cue.cue_id,
                            "text_length": len(cue.spoken_text or ""),
                            "text_sha256": sha256_text(cue.spoken_text),
                            "duration_budget_ms": cue.duration_budget_ms,
                            "attempt": index,
                        },
                    )
                    started_at = __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat()
                    cue_t0 = _time.perf_counter()
                    try:
                        self._synthesize_and_fit(
                            generator,
                            project,
                            cue,
                            context,
                            force=force,
                            fitter=fitter,
                            cues_sorted=cues_sorted,
                        )
                        elapsed_ms = int((_time.perf_counter() - cue_t0) * 1000)
                        self._check_cancelled()
                        self.store.upsert_cue(project.project_id, cue)
                        emit_event(
                            "cue.persisted",
                            payload={
                                "sequence": cue.sequence,
                                "cue_id": cue.cue_id,
                                "status": cue.status,
                                "raw_duration_ms": cue.raw_duration_ms,
                                "fitted_duration_ms": cue.fitted_duration_ms,
                                "required_speed_factor": cue.required_speed_factor,
                                "applied_speed_factor": cue.applied_speed_factor,
                                "elapsed_ms": elapsed_ms,
                                "raw_path": _rel_to_project(cue.raw_audio_path, project),
                                "fitted_path": _rel_to_project(cue.fitted_audio_path, project),
                            },
                            force_flush=True,
                        )
                        self.cue_updated_callback(
                            cue.sequence,
                            cue.status,
                            cue.raw_duration_ms or 0,
                            cue.fitted_duration_ms or 0,
                            index,
                        )
                        completed += 1
                        last_id = cue.cue_id
                        self._log(
                            f"OK cue #{cue.sequence} in {elapsed_ms} ms "
                            f"raw={cue.raw_duration_ms}ms status={cue.status}"
                        )
                        # Real OmniVoice is never <80ms end-to-end; guard against no-op TTS.
                        if force and elapsed_ms < 50 and (cue.raw_duration_ms or 0) > 0:
                            self._log(
                                f"WARNING cue #{cue.sequence}: suspiciously fast TTS "
                                f"({elapsed_ms} ms) — check engine is really synthesizing.",
                                level=logging.WARNING,
                            )
                        self._generation_log.append(
                            {
                                "run_id": context.run_id,
                                "cue_id": cue.cue_id,
                                "engine_id": context.engine_id,
                                "voice_id": context.voice_id,
                                "reference_voice_hash": context.reference_voice_content_hash,
                                "raw_generation_fingerprint": cue.generation_fingerprint,
                                "started_at": started_at,
                                "finished_at": __import__("datetime").datetime.now(
                                    __import__("datetime").timezone.utc
                                ).isoformat(),
                                "result": "completed",
                                "elapsed_ms": elapsed_ms,
                            }
                        )
                    except GenerationCancelled:
                        cancelled_count = 1
                        if generator is not None:
                            generator.cleanup_partial_artifacts()
                        if cue.status in {
                            CueStatus.RENDERING.value,
                            CueStatus.PENDING.value,
                        } or cue.status == CueStatus.CANCELLED.value:
                            cue.status = CueStatus.CANCELLED.value
                            cue.error_message = "Cancelled."
                        self.store.upsert_cue(project.project_id, cue)
                        emit_event(
                            "cue.cancelled",
                            payload={
                                "sequence": cue.sequence,
                                "cue_id": cue.cue_id,
                                "status": cue.status,
                            },
                            force_flush=True,
                        )
                        self.cue_updated_callback(
                            cue.sequence, cue.status, 0, 0, index
                        )
                        last_id = cue.cue_id
                        status = GenerationRunStatus.CANCELLED
                        error_message = "Generation cancelled."
                        self._log(
                            f"CANCELLED at cue #{cue.sequence}",
                            level=logging.WARNING,
                        )
                        self._generation_log.append(
                            {
                                "run_id": context.run_id,
                                "cue_id": cue.cue_id,
                                "engine_id": context.engine_id,
                                "voice_id": context.voice_id,
                                "reference_voice_hash": context.reference_voice_content_hash,
                                "raw_generation_fingerprint": cue.generation_fingerprint,
                                "started_at": started_at,
                                "finished_at": __import__("datetime").datetime.now(
                                    __import__("datetime").timezone.utc
                                ).isoformat(),
                                "result": "cancelled",
                            }
                        )
                        break
                    except CueGenerationError as exc:
                        elapsed_ms = int((_time.perf_counter() - cue_t0) * 1000)
                        cue.status = CueStatus.FAILED.value
                        cue.error_message = str(exc)
                        failed_ids.append(cue.cue_id)
                        self.store.upsert_cue(project.project_id, cue)
                        emit_event(
                            "cue.failed",
                            payload={
                                "sequence": cue.sequence,
                                "cue_id": cue.cue_id,
                                "status": cue.status,
                                "error": str(exc),
                                "elapsed_ms": elapsed_ms,
                            },
                            force_flush=True,
                        )
                        self.cue_updated_callback(
                            cue.sequence, cue.status, 0, 0, index
                        )
                        self.log_callback(f"Cue #{cue.sequence} failed: {exc}")
                        self._log(
                            f"FAIL cue #{cue.sequence} in {elapsed_ms} ms: {exc}",
                            level=logging.ERROR,
                        )
                        last_id = cue.cue_id
                        self._generation_log.append(
                            {
                                "run_id": context.run_id,
                                "cue_id": cue.cue_id,
                                "engine_id": context.engine_id,
                                "voice_id": context.voice_id,
                                "reference_voice_hash": context.reference_voice_content_hash,
                                "raw_generation_fingerprint": cue.generation_fingerprint,
                                "started_at": started_at,
                                "finished_at": __import__("datetime").datetime.now(
                                    __import__("datetime").timezone.utc
                                ).isoformat(),
                                "result": "failed",
                                "error": str(exc),
                                "elapsed_ms": elapsed_ms,
                            }
                        )
                        if policy == CueErrorPolicy.STOP:
                            stopped_on_error = True
                            status = GenerationRunStatus.FAILED
                            error_message = f"Cue #{cue.sequence}: {exc}"
                            self.log_callback(
                                f"Generation stopped at cue #{cue.sequence} "
                                f"({total - index} cue(s) pending)."
                            )
                            break
                    self._snapshot_resources(project, label=f"after_cue_{cue.sequence}")
        except GenerationCancelled:
            status = GenerationRunStatus.CANCELLED
            error_message = "Generation cancelled."
            cancelled_count = max(1, cancelled_count)
            if generator is not None:
                generator.cleanup_partial_artifacts()
        finally:
            self._active_cue_generator = None
            self._active_generation_context = None
            self._generation_active = False
            self._cancel_watchdog()
            self._snapshot_resources(project, label="after_generation")
            if status == GenerationRunStatus.CANCELLED:
                done_msg = "Generation cancelled"
            elif stopped_on_error:
                done_msg = "Generation stopped due to error"
            elif failed_ids:
                done_msg = "Generation completed with errors"
                if status == GenerationRunStatus.COMPLETED:
                    status = GenerationRunStatus.COMPLETED_WITH_ERRORS
            else:
                done_msg = "Generation completed"
            self.progress_callback("tts", total, total, done_msg)
        batch_ms = int((_time.perf_counter() - batch_started) * 1000)
        self._log(
            f"Batch done: status={status.value} completed={completed} "
            f"failed={len(failed_ids)} cancelled={cancelled_count} "
            f"total={total} elapsed_ms={batch_ms} force={force}"
        )
        if force and completed == 0 and cancelled_count == 0 and total > 0:
            status = GenerationRunStatus.FAILED
            error_message = error_message or (
                "Force regenerate produced 0 successful cues. "
                "Check TTS engine / reference voice. First error: "
                + (failed_ids[0] if failed_ids else "unknown")
            )
            self._log(error_message, level=logging.ERROR)
        invalidate_for_narration_change(project.stale)
        self.store.write_manifest_atomic(project)
        return GenerationRunResult(
            status=status,
            total_count=total,
            completed_count=completed,
            failed_count=len(failed_ids),
            cancelled_count=cancelled_count,
            skipped_count=max(0, total - completed - len(failed_ids) - cancelled_count),
            failed_cue_ids=tuple(failed_ids),
            last_processed_cue_id=last_id,
            error_message=error_message,
            run_id=context.run_id,
        )

    def _synthesize_and_fit(
        self,
        generator: CueGenerator,
        project: DubbingProject,
        cue: DubbingCue,
        context: GenerationContext,
        force: bool,
        fitter: DurationFitter | None = None,
        cues_sorted: list[DubbingCue] | None = None,
    ) -> None:
        cue.ensure_source_timing()
        current_fp = self._raw_fp_for(cue, context=context)
        raw_intact = self._raw_intact(cue)
        provenance_verified = bool(cue.generation_fingerprint) and (
            cue.generation_fingerprint == current_fp
        )
        emit_event(
            "cue.plan",
            payload={
                "sequence": cue.sequence,
                "cue_id": cue.cue_id,
                "raw_intact": raw_intact,
                "provenance_verified": provenance_verified,
                "legacy_audio_unverified": cue.legacy_audio_unverified,
                "force": force,
            },
        )
        if force:
            needs_tts = True
        elif not raw_intact:
            needs_tts = True
        elif provenance_verified:
            needs_tts = False
        elif cue.legacy_audio_unverified or not cue.generation_fingerprint:
            needs_tts = cue.legacy_observed_fingerprint != current_fp
        else:
            needs_tts = True

        if needs_tts:
            # Do NOT delete existing raw before a successful new synthesis.
            # Deleting first made force-runs that failed leave the project silent
            # and looked like "regeneration never started".
            voice_cfg = dict(context.voice_config)
            self._tts_snapshot(
                context,
                text_length=len(cue.spoken_text or ""),
                text_sha256=sha256_text(cue.spoken_text),
            )
            emit_event("cue.tts.started", payload={"sequence": cue.sequence})
            tts_t0 = __import__("time").perf_counter()
            self._log(
                f"TTS cue #{cue.sequence} force={force} "
                f"voice={context.voice_id or '—'} "
                f"ref={Path(context.reference_voice_path).name if context.reference_voice_path else '—'} "
                f"ref_in_cfg={bool(voice_cfg.get('reference_audio_path'))} "
                f"engine={type(self.tts_engine).__name__}"
            )
            try:
                generator.generate_raw(cue, voice_cfg)
            except Exception:
                # Fatal TTS error: the engine instance is suspect and must not
                # be silently reused for the next cue.
                try:
                    setattr(self.tts_engine, "_ltv_unusable", True)
                except Exception:  # pragma: no cover
                    pass
                emit_event(
                    "cue.tts.failed",
                    payload={
                        "sequence": cue.sequence,
                        "elapsed_ms": int((__import__("time").perf_counter() - tts_t0) * 1000),
                    },
                    force_flush=True,
                )
                raise
            tts_elapsed_ms = int((__import__("time").perf_counter() - tts_t0) * 1000)
            # Fingerprint comes from the immutable run snapshot, never live UI.
            cue.generation_fingerprint = current_fp
            cue.legacy_audio_unverified = False
            cue.legacy_observed_fingerprint = ""
            cue.raw_wav_hash = file_content_hash(cue.raw_audio_path) or ""
            wav_result = self._wav_validator.validate(Path(cue.raw_audio_path))
            emit_event(
                "cue.tts.completed",
                payload={
                    "sequence": cue.sequence,
                    "elapsed_ms": tts_elapsed_ms,
                    "raw_duration_ms": cue.raw_duration_ms,
                    "raw_wav_hash": cue.raw_wav_hash,
                    "raw_path": _rel_to_project(cue.raw_audio_path, project),
                    "raw_size": Path(cue.raw_audio_path).stat().st_size
                    if cue.raw_audio_path and Path(cue.raw_audio_path).is_file()
                    else None,
                },
            )
            emit_event(
                "cue.wav.validation",
                payload={
                    "sequence": cue.sequence,
                    "valid": wav_result.valid,
                    "duration_ms": wav_result.duration_ms,
                    "error_code": wav_result.error_code,
                    "error_message": wav_result.error_message,
                },
            )
        elif not cue.raw_wav_hash and cue.raw_audio_path:
            cue.raw_wav_hash = file_content_hash(cue.raw_audio_path) or ""

        if fitter is None:
            fitter = DurationFitter(project.settings, video_duration_ms=project.duration_ms)
        if cues_sorted is None:
            cues_sorted = sorted(project.cues, key=lambda c: c.effective_start_ms())
        next_cue = self._next_cue(cues_sorted, cue)
        result = fitter.evaluate(cue, next_cue=next_cue)
        emit_event(
            "cue.fit.evaluated",
            payload={
                "sequence": cue.sequence,
                "strategy": str(result.strategy.value if hasattr(result.strategy, "value") else result.strategy),
                "applied_speed_factor": result.applied_speed_factor,
                "fitted_duration_ms": result.fitted_duration_ms,
                "blocking": result.blocking,
            },
        )
        fitter.apply_result(cue, result)
        generator.apply_fitting(
            cue,
            result,
            compress_internal_pauses=context.compress_internal_pauses,
            internal_pause_keep_ms=context.internal_pause_keep_ms,
        )
        cue.fit_fingerprint = fit_fingerprint(
            cue,
            raw_generation_fp=cue.generation_fingerprint or current_fp,
            raw_wav_hash=cue.raw_wav_hash,
            settings=project.settings,
        )
        cue.fit_pipeline_version = 1
        emit_event(
            "cue.fit.completed",
            payload={
                "sequence": cue.sequence,
                "applied_speed_factor": cue.applied_speed_factor,
                "fitted_duration_ms": cue.fitted_duration_ms,
                "fitted_path": _rel_to_project(cue.fitted_audio_path, project),
            },
        )
        if cue.legacy_audio_unverified and not needs_tts:
            # Successful legacy refit: keep legacy raw flag but record fit so
            # reopening does not loop forever.
            cue.legacy_observed_fingerprint = current_fp

    def _refit_one(
        self,
        cue: DubbingCue,
        fitter: DurationFitter,
        next_cue: DubbingCue | None,
        project: DubbingProject,
        generator: CueGenerator | None = None,
        context: GenerationContext | None = None,
    ) -> None:
        cue.ensure_source_timing()
        result = fitter.evaluate(cue, next_cue=next_cue)
        fitter.apply_result(cue, result)
        if generator is None:
            generator = self._make_generator(project)
        compress = (
            context.compress_internal_pauses
            if context is not None
            else project.settings.compress_internal_pauses
        )
        keep = (
            context.internal_pause_keep_ms
            if context is not None
            else project.settings.internal_pause_keep_ms
        )
        generator.apply_fitting(
            cue,
            result,
            compress_internal_pauses=compress,
            internal_pause_keep_ms=keep,
        )
        raw_fp = cue.generation_fingerprint or self._raw_fp_for(
            cue, context=context, project=project
        )
        if not cue.raw_wav_hash and cue.raw_audio_path:
            cue.raw_wav_hash = file_content_hash(cue.raw_audio_path) or ""
        cue.fit_fingerprint = fit_fingerprint(
            cue,
            raw_generation_fp=raw_fp,
            raw_wav_hash=cue.raw_wav_hash,
            settings=project.settings,
        )
        cue.fit_pipeline_version = 1
        if cue.legacy_audio_unverified:
            cue.legacy_observed_fingerprint = raw_fp

    @staticmethod
    def _next_cue(cues_sorted: list[DubbingCue], cue: DubbingCue) -> DubbingCue | None:
        for index, other in enumerate(cues_sorted):
            if other.sequence == cue.sequence:
                for candidate in cues_sorted[index + 1:]:
                    if candidate.enabled and candidate.sequence != cue.sequence:
                        return candidate
        return None

    def _make_generator(self, project: DubbingProject) -> CueGenerator:
        from .cue_generator import CueGenerationConfig

        config = CueGenerationConfig(
            sample_rate=project.settings.sample_rate,
            channels=project.settings.channels,
            compress_internal_pauses=project.settings.compress_internal_pauses,
            internal_pause_keep_ms=project.settings.internal_pause_keep_ms,
        )
        return CueGenerator(
            self.tts_engine,
            ffmpeg_path=project.settings.ffmpeg_path,
            config=config,
            progress_callback=lambda c, t, msg: self.progress_callback("tts", c, t, msg),
            log_callback=self.log_callback,
        )

    def _make_generator_from_context(self, context: GenerationContext) -> CueGenerator:
        from .cue_generator import CueGenerationConfig

        config = CueGenerationConfig(
            sample_rate=context.sample_rate,
            channels=context.channels,
            compress_internal_pauses=context.compress_internal_pauses,
            internal_pause_keep_ms=context.internal_pause_keep_ms,
        )
        return CueGenerator(
            self.tts_engine,
            ffmpeg_path=context.ffmpeg_path,
            config=config,
            progress_callback=lambda c, t, msg: self.progress_callback("tts", c, t, msg),
            log_callback=self.log_callback,
        )

    def apply_tempo_smoothing(self, project: DubbingProject) -> int:
        """Independent tempo smoothing (no timecode shifts in strict mode)."""
        settings = project.settings.tempo_smoothing
        if not settings.auto_apply_after_generate:
            return 0
        planner = TempoSmoothingPlanner(settings)
        plans = planner.plan(project.cues)
        if not plans:
            return 0
        changed = planner.apply(project.cues, plans)
        if not changed:
            return 0
        # Refit cues that received a planned speed different from current applied.
        context = self.build_generation_context(project)
        generator = self._make_generator_from_context(context)
        fitter = DurationFitter(project.settings, video_duration_ms=project.duration_ms)
        cues_sorted = sorted(project.cues, key=lambda c: c.effective_start_ms())
        for plan in plans:
            for seq in plan.sequences:
                cue = next((c for c in project.cues if c.sequence == seq), None)
                if cue is None or not self._raw_intact(cue):
                    continue
                next_cue = self._next_cue(cues_sorted, cue)
                try:
                    self._refit_one(cue, fitter, next_cue, project, generator, context)
                    self.store.upsert_cue(project.project_id, cue)
                except Exception as exc:
                    self._log(f"Tempo-smooth refit cue #{seq} failed: {exc}")
            self._log(
                f"Tempo {plan.group_id}: cues {plan.sequences[0]}–{plan.sequences[-1]} "
                f"({plan.reason})"
            )
        invalidate_for_narration_change(project.stale)
        self.store.write_manifest_atomic(project)
        return changed

    def _elastic_smoothing_allowed(self, project: DubbingProject) -> bool:
        settings = project.settings.elastic_timing
        return bool(
            settings.enabled
            or settings.auto_smooth_neighbors
            or project.settings.cue_timing_mode == CueTimingMode.ELASTIC_GROUP
        )

    def _cue_needs_smoothing(self, project: DubbingProject, cue: DubbingCue) -> bool:
        if not cue.enabled or not cue.raw_duration_ms or cue.raw_duration_ms <= 0:
            return False
        if cue.timing_locked:
            return False
        threshold = project.settings.elastic_timing.smooth_speed_threshold
        if threshold <= 0:
            threshold = project.settings.preferred_speed_limit
        # Estimate required factor against the source window.
        cue.ensure_source_timing()
        budget = max(
            1,
            (cue.source_end_ms or cue.end_ms) - (cue.source_start_ms or cue.start_ms),
        )
        required = float(cue.raw_duration_ms) / float(budget)
        applied = float(cue.applied_speed_factor or 1.0)
        return required > threshold + 1e-6 or applied > threshold + 1e-6

    def plan_elastic_groups(self, project: DubbingProject) -> list[Any]:
        """Compute elastic groups for cues that need help. Does not mutate SRT."""
        if not self._elastic_smoothing_allowed(project):
            return []
        settings = project.settings.elastic_timing
        planner = ElasticTimingPlanner(settings, video_duration_ms=project.duration_ms)
        enabled = sorted(
            [c for c in project.cues if c.enabled],
            key=lambda c: c.source_start_ms if c.source_start_ms is not None else c.start_ms,
        )
        plans = []
        covered: set[int] = set()
        index = 0
        while index < len(enabled):
            cue = enabled[index]
            if cue.sequence in covered:
                index += 1
                continue
            if not self._cue_needs_smoothing(project, cue):
                index += 1
                continue
            plan = planner.choose_group_for_cue(enabled, index)
            if plan is not None and len(plan.sequences) >= 1:
                if plan.common_speed_factor > 1.0 or plan.max_shift_ms > 0:
                    plans.append(plan)
                    covered.update(plan.sequences)
                    index += len(plan.sequences)
                    continue
            index += 1
        return plans

    def auto_smooth_neighbors(self, project: DubbingProject) -> int:
        """Automatically form elastic groups so neighboring cues share tempo.

        Runs after generate/refit when auto_smooth_neighbors is enabled (default),
        even if the UI timing mode is still "strict". Source SRT is never mutated.
        """
        settings = project.settings.elastic_timing
        if not settings.auto_apply_after_generate and not settings.enabled:
            return 0
        if not self._elastic_smoothing_allowed(project):
            return 0
        plans = self.plan_elastic_groups(project)
        if not plans:
            return 0
        affected_sequences: set[int] = set()
        for plan in plans:
            self.apply_elastic_plan(project, plan)
            affected_sequences.update(plan.sequences)
            self._log(
                f"Elastic {plan.group_id}: cues {plan.sequences[0]}–{plan.sequences[-1]}, "
                f"common×{plan.common_speed_factor:.3f}, max shift {plan.max_shift_ms} ms"
            )
        # Re-fit only the cues that received a new planned timeline.
        if affected_sequences:
            context = self.build_generation_context(project)
            generator = self._make_generator_from_context(context)
            fitter = DurationFitter(project.settings, video_duration_ms=project.duration_ms)
            cues_sorted = sorted(project.cues, key=lambda c: c.effective_start_ms())
            for cue in project.cues:
                if cue.sequence not in affected_sequences:
                    continue
                if not self._raw_intact(cue):
                    continue
                next_cue = self._next_cue(cues_sorted, cue)
                try:
                    self._refit_one(cue, fitter, next_cue, project, generator, context)
                    self.store.upsert_cue(project.project_id, cue)
                except Exception as exc:
                    self._log(f"Auto-smooth refit cue #{cue.sequence} failed: {exc}")
            invalidate_for_narration_change(project.stale)
            self.store.write_manifest_atomic(project)
        return len(plans)

    def apply_elastic_plan(self, project: DubbingProject, plan: Any) -> None:
        planner = ElasticTimingPlanner(
            project.settings.elastic_timing, video_duration_ms=project.duration_ms
        )
        planner.apply_plan(project.cues, plan)
        invalidate_for_narration_change(project.stale)
        self.store.save_project(project)

    def clear_elastic_group(self, project: DubbingProject, group_id: str) -> None:
        planner = ElasticTimingPlanner(
            project.settings.elastic_timing, video_duration_ms=project.duration_ms
        )
        planner.clear_group(project.cues, group_id)
        invalidate_for_narration_change(project.stale)
        self.store.save_project(project)

    def export_adjusted_srt(self, project: DubbingProject, output_path: Path | None = None) -> Path:
        from .srt_parser import cues_to_srt

        if output_path is None:
            output_path = project.reports_dir() / "adjusted_subtitles.srt"
        # Temporarily map planned times into start/end for export without
        # mutating persisted source timing permanently.
        original: list[tuple[int, int]] = []
        for cue in project.cues:
            original.append((cue.start_ms, cue.end_ms))
            if cue.planned_start_ms is not None:
                cue.start_ms = cue.planned_start_ms
            if cue.planned_end_ms is not None:
                cue.end_ms = cue.planned_end_ms
        try:
            text = cues_to_srt(project.cues)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
        finally:
            for cue, (start, end) in zip(project.cues, original):
                cue.start_ms = start
                cue.end_ms = end
        return output_path

    def export_elastic_report(self, project: DubbingProject) -> Path:
        import json

        groups: dict[str, dict[str, Any]] = {}
        for cue in project.cues:
            if not cue.timing_group_id:
                continue
            entry = groups.setdefault(
                cue.timing_group_id,
                {
                    "group_id": cue.timing_group_id,
                    "cue_ids": [],
                    "common_speed_factor": cue.common_speed_factor,
                    "source_start_ms": cue.source_start_ms,
                    "planned_end_ms": cue.planned_end_ms,
                    "borrowed_right_ms": 0,
                    "max_shift_ms": 0,
                },
            )
            entry["cue_ids"].append(cue.cue_id)
            entry["borrowed_right_ms"] = max(
                entry["borrowed_right_ms"], cue.borrowed_right_ms
            )
            entry["max_shift_ms"] = max(entry["max_shift_ms"], cue.start_shift_ms)
            if cue.planned_end_ms is not None:
                prev = entry.get("planned_end_ms") or 0
                entry["planned_end_ms"] = max(prev, cue.planned_end_ms)
            if cue.source_start_ms is not None:
                prev_s = entry.get("source_start_ms")
                if prev_s is None or cue.source_start_ms < prev_s:
                    entry["source_start_ms"] = cue.source_start_ms
        path = project.reports_dir() / "elastic_groups.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(list(groups.values()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def invalidate_reference_voice(self, project: DubbingProject) -> None:
        """Mark all raw/fit/downstream stale after reference voice change."""
        for cue in project.cues:
            if cue.enabled:
                cue.mark_stale()
                cue.generation_fingerprint = ""
                cue.fit_fingerprint = ""
                cue.raw_wav_hash = ""
        invalidate_for_text_change(project.stale)
        self.store.save_project(project)

    def set_engine(
        self,
        project: DubbingProject,
        engine: BaseTTSEngine,
        tts_engine_id: str,
        voice_config: dict[str, Any],
    ) -> None:
        """Swap the TTS engine. Existing cues are marked stale (not deleted) so
        they can be regenerated on demand; ready cues are not re-synthesized
        until their inputs actually differ."""
        self.tts_engine = engine
        project.settings.tts_engine = tts_engine_id
        project.settings.voice_config = dict(voice_config)
        project.settings.voice = str(voice_config.get("voice") or voice_config.get("speaker") or "")
        invalidate_for_text_change(project.stale)
        self.store.save_project(project)

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
        with self._diagnostic_run(project) as run_dir, operation_span(
            "render_narration",
            project_id=project.project_id,
            run_id=getattr(run_dir, "run_id", None),
            extra={"duration_ms": project.duration_ms},
        ):
            self._begin_operation("render_narration", duration_ms=project.duration_ms)
            self._snapshot_resources(project, label="before_narration")
            if output_path is None:
                output_path = project.render_dir() / "narration.wav"
            renderer = TimelineRenderer(
                ffmpeg_path=project.settings.ffmpeg_path,
                sample_rate=project.settings.sample_rate,
                channels=project.settings.channels,
                progress_callback=lambda c, t, msg: (
                    self.progress_callback("narration", c, t, msg),
                    self._heartbeat_watchdog(),
                ),
                log_callback=self.log_callback,
            )
            self._active_timeline = renderer
            self._arm_watchdog("render_narration")
            try:
                renderer.render(
                    project,
                    output_path,
                    alignment=project.settings.preview.alignment,
                )
            except TimelineRenderCancelled as exc:
                raise GenerationCancelled(str(exc)) from exc
            finally:
                self._active_timeline = None
                self._cancel_watchdog()
            self._check_cancelled()
            project.narration_wav = output_path
            project.stale.narration = False
            # encode narration mp3
            self._check_cancelled()
            mixer = self._build_mixer(project, stage="narration")
            narration_mp3 = project.render_dir() / "narration.mp3"
            try:
                mixer.encode_narration_mp3(output_path, narration_mp3)
                project.narration_mp3 = narration_mp3
            except AudioMixerCancelled as exc:
                raise GenerationCancelled(str(exc)) from exc
            except Exception as exc:
                self.log_callback(f"Narration MP3 encoding skipped: {exc}")
            self._snapshot_resources(project, label="after_narration")
            self.save_project(project)
            return output_path

    def render_dubbed_mix(
        self,
        project: DubbingProject,
        output_path: Path | None = None,
    ) -> Path:
        with self._diagnostic_run(project) as run_dir, operation_span(
            "render_dubbed_mix",
            project_id=project.project_id,
            run_id=getattr(run_dir, "run_id", None),
        ):
            self._begin_operation("render_dubbed_mix")
            self._snapshot_resources(project, label="before_mix")
            # Cancel checkpoint BEFORE narration: if the user cancelled during a
            # prior stage, mix must not start (test: cancel between narration
            # and mix does not launch mix).
            self._check_cancelled()
            if project.narration_wav is None or not Path(project.narration_wav).is_file():
                self.render_narration(project)
            # Cancel checkpoint AFTER narration, BEFORE mix.
            self._check_cancelled()
            if output_path is None:
                output_path = project.render_dir() / "dubbed_mix.wav"
            mixer = self._build_mixer(project, stage="mixing")
            self._active_mixer = mixer
            self._arm_watchdog("render_dubbed_mix")
            try:
                mixer.render_dubbed_mix(
                    project,
                    Path(project.narration_wav),
                    output_path,
                )
            except AudioMixerCancelled as exc:
                raise GenerationCancelled(str(exc)) from exc
            finally:
                self._active_mixer = None
                self._cancel_watchdog()
            project.dubbed_mix_wav = output_path
            project.stale.mix = False
            invalidate_for_preview_change(project.stale)
            project.stale.video = True
            self._snapshot_resources(project, label="after_mix")
            self.save_project(project)
            return output_path

    def render_cue_preview(
        self,
        project: DubbingProject,
        sequence: int,
    ) -> Path:
        with self._diagnostic_run(project) as run_dir, operation_span(
            "render_cue_preview",
            project_id=project.project_id,
            run_id=getattr(run_dir, "run_id", None),
            extra={"sequence": sequence},
        ):
            self._begin_operation("render_cue_preview", sequence=sequence)
            self._check_cancelled()
            cue = self._find_cue(project, sequence)
            if cue is None:
                raise VideoDubbingServiceError(f"Cue #{sequence} not found.")
            renderer = self._build_preview(project)
            self._active_preview = renderer
            try:
                path = renderer.render_cue_preview(project, cue)
            except PreviewRenderCancelled as exc:
                raise GenerationCancelled(str(exc)) from exc
            finally:
                self._active_preview = None
            self.save_project(project)
            return path

    def render_full_preview(self, project: DubbingProject) -> Path:
        with self._diagnostic_run(project) as run_dir, operation_span(
            "render_full_preview",
            project_id=project.project_id,
            run_id=getattr(run_dir, "run_id", None),
        ):
            self._begin_operation("render_full_preview")
            self._check_cancelled()
            if project.dubbed_mix_wav is None or not Path(project.dubbed_mix_wav).is_file():
                self.render_dubbed_mix(project)
            self._check_cancelled()
            renderer = self._build_preview(project)
            self._active_preview = renderer
            try:
                path = renderer.render_full_preview(project)
            except PreviewRenderCancelled as exc:
                raise GenerationCancelled(str(exc)) from exc
            finally:
                self._active_preview = None
            project.stale.preview = False
            self.save_project(project)
            return path

    def export_video(
        self,
        project: DubbingProject,
        output_path: Path | None = None,
    ) -> Path:
        with self._diagnostic_run(project) as run_dir, operation_span(
            "export_video",
            project_id=project.project_id,
            run_id=getattr(run_dir, "run_id", None),
            extra={"container": project.settings.export.container.value},
        ):
            self._begin_operation(
                "export_video",
                container=project.settings.export.container.value,
            )
            self._snapshot_resources(project, label="before_mux")
            # Cancel checkpoint BEFORE mux.
            self._check_cancelled()
            if project.video_path is None:
                raise VideoDubbingServiceError("No source video attached.")
            if project.dubbed_mix_wav is None or not Path(project.dubbed_mix_wav).is_file():
                self.render_dubbed_mix(project)
            # Cancel checkpoint AFTER mix, BEFORE mux (test: cancel between mix
            # and mux does not launch mux).
            self._check_cancelled()
            muxer = VideoMuxer(
                ffmpeg_path=project.settings.ffmpeg_path,
                progress_callback=lambda c, t, msg: (
                    self.progress_callback("muxing", c, t, msg),
                    self._heartbeat_watchdog(),
                ),
                log_callback=self.log_callback,
                stderr_dump_dir=getattr(self._active_run_dir, "subprocess_dir", None),
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
            self._arm_watchdog("export_video")
            try:
                result = muxer.mux(
                    project,
                    output_path=output_path,
                    narration_only_wav=narration_only,
                    srt_path=srt_path,
                )
            except VideoMuxCancelled as exc:
                raise GenerationCancelled(str(exc)) from exc
            except VideoMuxError as exc:
                raise VideoDubbingServiceError(str(exc)) from exc
            finally:
                self._active_muxer = None
                self._cancel_watchdog()
            self._check_cancelled()
            project.final_video_path = result.output_path
            project.stale.video = False
            self._check_cancelled()
            report = build_report(project, audio_tracks=result.tracks)
            write_reports(project, report)
            mark_clean(project.stale)
            self._snapshot_resources(project, label="after_mux")
            self.save_project(project)
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
        for cue in project.cues:
            if not cue.enabled:
                continue
            if cue.status in BLOCKING_STATUSES:
                blockers.append(
                    f"Cue #{cue.sequence}: {cue.status}"
                    + (f" ({cue.error_message})" if cue.error_message else "")
                )
            elif not self._cue_has_playable_audio(cue):
                blockers.append(f"Cue #{cue.sequence}: fitted audio missing/unreadable.")
        if project.video_path is None:
            blockers.append("No source video attached.")
        if project.dubbed_mix_wav is None:
            blockers.append("Dubbed mix not rendered.")
        return (len(blockers) == 0, blockers)

    @staticmethod
    def _cue_has_playable_audio(cue: DubbingCue) -> bool:
        return (
            cue.fitted_audio_path is not None
            and Path(cue.fitted_audio_path).is_file()
            and cue.fitted_duration_ms is not None
            and cue.fitted_duration_ms > 0
        )
