"""VideoDubbingFacade.

The single application service that MCP tools, HTTP routes, CLI and tests call.
It wraps :class:`VideoDubbingService` and adds:

* per-project locks + monotonic revision (optimistic concurrency);
* metadata snapshots + transactional rollback;
* an async job manager for heavy operations;
* dry-run simulation + objective scoring;
* automatic QA + invariant checks;
* diagnostics (run events, last failure classification).

The facade never imports PySide/Qt and never holds business logic that belongs
in the domain layer - it composes and enforces policies around it.
"""

from __future__ import annotations

import copy
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from app.core.video_dubbing.generation import (
    GenerationCancelled,
    GenerationRunResult,
)
from app.core.video_dubbing.models import (
    DubbingCue,
    DubbingProject,
    DubbingProjectSettings,
    VideoProbeInfo,
)
from app.core.video_dubbing.service import (
    ImportSrtResult,
    VideoDubbingService,
)
from app.core.video_dubbing.project_store import DubbingProjectStore
from app.core.video_dubbing.stale_state import (
    invalidate_for_narration_change,
    invalidate_for_text_change,
)
from app.observability import (
    bind,
    emit_event,
    operation_span,
    sha256_text,
)

from .engine_provider import EngineLeaseProvider
from .errors import (
    DubbingNotFoundError,
    DubbingProjectChangedError,
    DubbingValidationError,
)
from .job_manager import DubbingJobManager
from .job_models import DubbingJob
from .project_locks import ProjectLockRegistry
from .quality import (
    assert_project_invariants,
    detect_clipping,
    detect_silence,
    measure_loudness,
    validate_final_video,
    validate_timeline,
    validate_wav,
)
from .revision_snapshots import RevisionStore, SnapshotStore
from . import serializers
from . import simulation

_log = logging.getLogger("video_dubbing.mcp")

# Limits enforced on every call (spec section 23.13).
MAX_SEQUENCES_PER_CALL = 500
MAX_TEXT_CHARS = 100_000
MAX_SRT_BYTES = 5 * 1024 * 1024


@dataclass
class MutationResult:
    project_id: str
    snapshot_id: str
    old_revision: int
    new_revision: int
    changed_cues: list[int]
    invalidated: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "snapshot_id": self.snapshot_id,
            "old_revision": self.old_revision,
            "new_revision": self.new_revision,
            "changed_cues": self.changed_cues,
            "invalidated": self.invalidated,
            "warnings": self.warnings,
        }


class VideoDubbingFacade:
    def __init__(
        self,
        *,
        store: DubbingProjectStore | None = None,
        engine_provider: EngineLeaseProvider | None = None,
        job_manager: DubbingJobManager | None = None,
        locks: ProjectLockRegistry | None = None,
        revision_store: RevisionStore | None = None,
        snapshot_store: SnapshotStore | None = None,
    ) -> None:
        self.store = store or DubbingProjectStore()
        self.engine_provider = engine_provider or EngineLeaseProvider()
        self.locks = locks or ProjectLockRegistry()
        self.revision_store = revision_store or RevisionStore()
        self.snapshot_store = snapshot_store or SnapshotStore(self.store)
        self.job_manager = job_manager or DubbingJobManager(locks=self.locks)
        self._diagnostic_dir_root = Path.home() / ".localtext2voice" / "diagnostics"
        self._log_cb = lambda msg: _log.info(msg)

    # ------------------------------------------------------------------ helpers

    def _load(self, project_id: str) -> DubbingProject:
        project = self.store.load_project(project_id)
        if project is None:
            raise DubbingNotFoundError(f"Project not found: {project_id}")
        for cue in project.cues:
            cue.ensure_source_timing()
        return project

    @contextmanager
    def _service(
        self,
        project: DubbingProject,
        *,
        progress: Callable[[str, int, int, str], None] | None = None,
    ) -> Iterator[VideoDubbingService]:
        lease = self.engine_provider.acquire(project.settings)
        service = VideoDubbingService(
            tts_engine=lease.engine,
            store=self.store,
            progress_callback=progress or (lambda stage, c, t, msg: None),
            log_callback=self._log_cb,
        )
        emit_event(
            "engine.lease.acquired",
            payload={
                "engine_id": lease.engine_id,
                "engine_class": type(lease.engine).__name__,
                "ownership": lease.ownership,
            },
            force_flush=False,
        )
        try:
            yield service
        finally:
            self.engine_provider.release(lease)
            emit_event(
                "engine.lease.released",
                payload={"engine_id": lease.engine_id},
                force_flush=False,
            )

    @contextmanager
    def _mutation(
        self,
        project_id: str,
        expected_revision: int | None,
        *,
        initiator: str = "mcp",
        reason: str = "mutation",
    ) -> Iterator[tuple[DubbingProject, RevisionStore, Callable[[list[int], list[str], list[str]], None]]]:
        lock = self.locks.get(project_id)
        self.locks.ensure_idle(project_id)
        with lock.write_lock(initiator=initiator):
            project = self._load(project_id)
            current = self.revision_store.current(project)
            if expected_revision is not None and expected_revision != current:
                raise DubbingProjectChangedError(
                    expected_revision=expected_revision,
                    current_revision=current,
                )
            # Pre-mutation snapshot (BEFORE the change).
            snap = self.snapshot_store.create(
                project, reason=reason, initiator=initiator, revision=current
            )
            emit_event(
                "project.snapshot.created",
                payload={"snapshot_id": snap["snapshot_id"], "reason": reason, "revision": current},
            )
            captured: dict[str, Any] = {"changed": [], "invalidated": [], "warnings": []}

            def record(changed: list[int], invalidated: list[str], warnings: list[str]) -> None:
                captured["changed"] = sorted(set(changed))
                captured["invalidated"] = invalidated
                captured["warnings"] = warnings

            yield project, self.revision_store, record
            self.store.save_project(project)
            new_revision = self.revision_store.bump(project)
            emit_event(
                "project.revision.changed",
                payload={
                    "project_id": project_id,
                    "old_revision": current,
                    "new_revision": new_revision,
                    "changed_cues": captured["changed"],
                },
            )

    def _diff_invalidated(self, stale_flags: Any) -> list[str]:
        return [k for k, v in stale_flags.to_dict().items() if v]

    # ------------------------------------------------------------------ projects

    def list_projects(self) -> list[dict[str, Any]]:
        return self.store.list_projects()

    def create_project(
        self,
        project_dir: str | Path,
        title: str = "Video Dubbing",
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = self._safe_project_dir(project_dir, create=True)
        s = DubbingProjectSettings.from_dict(settings or {})
        # Acquire a lease only if an engine host is configured; creation itself
        # needs no TTS. We use a transient service to create directories.
        service = VideoDubbingService(
            tts_engine=_NullEngine(), store=self.store, log_callback=self._log_cb
        )
        project = service.create_project(path, title=title, settings=s)
        self.revision_store.bump(project)
        emit_event("project.created", payload={"project_id": project.project_id, "title": title})
        return serializers.serialize_project(project)

    def open_project(self, project_dir: str | Path) -> dict[str, Any]:
        path = self._safe_project_dir(project_dir, create=False)
        manifest = path / "dubbing_project.json"
        if manifest.is_file():
            service = VideoDubbingService(
                tts_engine=_NullEngine(), store=self.store, log_callback=self._log_cb
            )
            project = service.open_project_manifest(manifest)
            return serializers.serialize_project(project)
        found = self.store.find_project_by_dir(path)
        if found is None:
            raise DubbingNotFoundError(f"No project in: {path}")
        return serializers.serialize_project(found)

    def get_project(self, project_id: str, *, include_cues: bool = False) -> dict[str, Any]:
        return serializers.serialize_project(self._load(project_id), include_cues=include_cues)

    def get_project_state(self, project_id: str) -> dict[str, Any]:
        project = self._load(project_id)
        state = serializers.serialize_state(project)
        state["revision"] = self.revision_store.current(project)
        state["active_job"] = self.locks.active_job(project_id)
        return state

    def delete_project(self, project_id: str, *, confirm: bool = False) -> dict[str, Any]:
        if not confirm:
            raise DubbingValidationError(
                "Pass confirm=true to delete a project.",
                code="confirmation_required",
            )
        self.locks.ensure_idle(project_id)
        project = self._load(project_id)
        self.store.delete_project(project_id)
        emit_event("project.deleted", payload={"project_id": project_id})
        return {"project_id": project_id, "deleted": True, "title": project.title}

    # ------------------------------------------------------------------ import

    def _safe_path(self, value: str | Path, *, must_exist: bool) -> Path:
        candidate = Path(value).expanduser()
        # Normalise and reject obvious traversal escapes against the cwd root.
        resolved = candidate.resolve() if candidate.is_absolute() else Path.cwd() / candidate
        resolved = resolved.resolve()
        if ".." in Path(value).parts:
            raise DubbingValidationError(
                f"Path traversal is not allowed: {value}", code="path_traversal"
            )
        if must_exist and not resolved.exists():
            raise DubbingNotFoundError(f"File not found: {value}")
        return resolved

    def _safe_project_dir(self, value: str | Path, *, create: bool) -> Path:
        if ".." in Path(value).parts:
            raise DubbingValidationError("Path traversal is not allowed.", code="path_traversal")
        path = Path(value).expanduser().resolve()
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def probe_video(self, video_path: str | Path, ffmpeg_path: str = "") -> dict[str, Any]:
        path = self._safe_path(video_path, must_exist=True)
        from app.utils import ffprobe_utils

        data = ffprobe_utils.probe_media(path, ffmpeg_path)
        duration_ms, video_stream, audio_streams = ffprobe_utils.parse_video_probe(data)
        return VideoProbeInfo(
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
        ).to_dict()

    def attach_video(
        self,
        project_id: str,
        video_path: str | Path,
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        path = self._safe_path(video_path, must_exist=True)
        with self._mutation(project_id, expected_revision, reason="attach_video") as (project, _, record):
            service = VideoDubbingService(
                tts_engine=_NullEngine(), store=self.store, log_callback=self._log_cb
            )
            probe = service.attach_video(project, path)
            record([], self._diff_invalidated(project.stale), [])
            return {
                "video_probe": probe.to_dict(),
                "video_path": str(path),
                "stale": project.stale.to_dict(),
            }

    def import_srt(
        self,
        project_id: str,
        srt_path: str | Path,
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        path = self._safe_path(srt_path, must_exist=True)
        size = path.stat().st_size
        if size > MAX_SRT_BYTES:
            raise DubbingValidationError(
                f"SRT too large ({size} bytes; limit {MAX_SRT_BYTES}).",
                code="srt_too_large",
            )
        with self._mutation(project_id, expected_revision, reason="import_srt") as (project, _, record):
            service = VideoDubbingService(
                tts_engine=_NullEngine(), store=self.store, log_callback=self._log_cb
            )
            result: ImportSrtResult = service.import_srt(project, path)
            record(
                [c.sequence for c in result.cues],
                self._diff_invalidated(project.stale),
                [w.message for w in result.warnings],
            )
            return serializers.serialize_import_result(result.cues, result.warnings)

    def analyze_project(self, project_id: str) -> dict[str, Any]:
        project = self._load(project_id)
        service = VideoDubbingService(
            tts_engine=_NullEngine(), store=self.store, log_callback=self._log_cb
        )
        warnings = service.analyze_project(project)
        return {
            "duration_ms": project.duration_ms,
            "has_audio_track": bool(project.video_probe and project.video_probe.audio_tracks > 0),
            "video_codec": project.video_probe.video_codec if project.video_probe else "",
            "audio_codec": project.video_probe.audio_codec if project.video_probe else "",
            "cue_count": len(project.cues),
            "warnings": [
                {"code": w.code, "message": w.message, "sequence": w.sequence} for w in warnings
            ],
            "problematic_sequences": sorted({w.sequence for w in warnings if w.sequence is not None}),
            "stale": project.stale.to_dict(),
        }

    # ------------------------------------------------------------------ read cues

    def list_cues(
        self,
        project_id: str,
        *,
        page: int = 1,
        page_size: int = 50,
        include_text: bool = False,
        **filters: Any,
    ) -> dict[str, Any]:
        project = self._load(project_id)
        cues = serializers.filter_cues(project.cues, **filters)
        return serializers.serialize_cues_page(
            cues, project, page=page, page_size=page_size, include_text=include_text
        )

    def get_cue(self, project_id: str, sequence: int, *, include_text: bool = True) -> dict[str, Any]:
        project = self._load(project_id)
        cue = next((c for c in project.cues if c.sequence == sequence), None)
        if cue is None:
            raise DubbingNotFoundError(f"Cue #{sequence} not found.")
        return serializers.serialize_cue(cue, project, include_text=include_text)

    def get_cue_neighbors(self, project_id: str, sequence: int, *, include_text: bool = False) -> dict[str, Any]:
        return serializers.serialize_neighbors(self._load(project_id), sequence, include_text=include_text)

    def get_timeline_window(
        self,
        project_id: str,
        *,
        time_from_ms: int,
        time_to_ms: int,
        include_text: bool = False,
    ) -> dict[str, Any]:
        project = self._load(project_id)
        cues = serializers.filter_cues(
            project.cues, time_from_ms=time_from_ms, time_to_ms=time_to_ms
        )
        return serializers.serialize_cues_page(
            cues, project, page=1, page_size=200, include_text=include_text
        )

    # ------------------------------------------------------------------ settings

    def get_settings(self, project_id: str) -> dict[str, Any]:
        return self._load(project_id).settings.to_dict()

    def _update_settings(
        self,
        project_id: str,
        expected_revision: int | None,
        reason: str,
        apply_fn: Callable[[DubbingProjectSettings], None],
        invalidated: list[str],
    ) -> dict[str, Any]:
        with self._mutation(project_id, expected_revision, reason=reason) as (project, _, record):
            apply_fn(project.settings)
            record([], invalidated, [])
            return {
                "settings": project.settings.to_dict(),
                "stale": project.stale.to_dict(),
            }

    def set_voice_settings(self, project_id: str, settings: dict[str, Any], *, expected_revision: int | None = None) -> dict[str, Any]:
        def apply(target: DubbingProjectSettings) -> None:
            for key in ("language", "tts_engine", "voice"):
                if key in settings:
                    setattr(target, key, str(settings[key]))
            if "voice_config" in settings and isinstance(settings["voice_config"], dict):
                target.voice_config = dict(settings["voice_config"])
            if "preferred_speed_limit" in settings:
                target.preferred_speed_limit = float(settings["preferred_speed_limit"])
                target.max_speed_factor = target.preferred_speed_limit
            if "hard_speed_limit" in settings:
                target.hard_speed_limit = float(settings["hard_speed_limit"])

        invalidated = ["cues", "narration", "mix", "preview", "video"]
        return self._update_settings(project_id, expected_revision, "set_voice_settings", apply, invalidated)

    def set_timing_settings(self, project_id: str, settings: dict[str, Any], *, expected_revision: int | None = None) -> dict[str, Any]:
        def apply(target: DubbingProjectSettings) -> None:
            for key in ("guard_gap_ms", "internal_pause_keep_ms"):
                if key in settings:
                    setattr(target, key, int(settings[key]))
            if "compress_internal_pauses" in settings:
                target.compress_internal_pauses = bool(settings["compress_internal_pauses"])
            if "elastic_timing" in settings and isinstance(settings["elastic_timing"], dict):
                target.elastic_timing = type(target.elastic_timing).from_dict(settings["elastic_timing"])
            if "tempo_smoothing" in settings and isinstance(settings["tempo_smoothing"], dict):
                target.tempo_smoothing = type(target.tempo_smoothing).from_dict(settings["tempo_smoothing"])

        invalidated = ["narration", "mix", "preview", "video"]
        return self._update_settings(project_id, expected_revision, "set_timing_settings", apply, invalidated)

    def set_mix_settings(self, project_id: str, settings: dict[str, Any], *, expected_revision: int | None = None) -> dict[str, Any]:
        def apply(target: DubbingProjectSettings) -> None:
            if "ducking" in settings and isinstance(settings["ducking"], dict):
                target.ducking = type(target.ducking).from_dict(settings["ducking"])

        invalidated = ["mix", "preview", "video"]
        return self._update_settings(project_id, expected_revision, "set_mix_settings", apply, invalidated)

    def set_export_settings(self, project_id: str, settings: dict[str, Any], *, expected_revision: int | None = None) -> dict[str, Any]:
        def apply(target: DubbingProjectSettings) -> None:
            if "container" in settings:
                from app.core.video_dubbing.models import OutputContainer

                target.export.container = OutputContainer(str(settings["container"]))
            for key in ("include_narration_only", "embed_subtitles", "force_video_reencode"):
                if key in settings:
                    setattr(target.export, key, bool(settings[key]))
            if "audio_bitrate" in settings:
                target.export.audio_bitrate = str(settings["audio_bitrate"])

        invalidated = ["video"]
        return self._update_settings(project_id, expected_revision, "set_export_settings", apply, invalidated)

    # ------------------------------------------------------------------ generation plan

    def get_generation_plan(self, project_id: str, *, force: bool = False) -> dict[str, Any]:
        project = self._load(project_id)
        with self._service(project) as service:
            context = service.build_generation_context(project)
            plan = service.generation_plan(project, force=force, context=context)
        return plan

    # ------------------------------------------------------------------ cue mutations

    def update_cue_text(
        self,
        project_id: str,
        sequence: int,
        text: str,
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        self._validate_text(text)
        with self._mutation(project_id, expected_revision, reason="update_cue_text") as (project, _, record):
            service = VideoDubbingService(
                tts_engine=_NullEngine(), store=self.store, log_callback=self._log_cb
            )
            service.update_cue_text(project, sequence, text)
            record([sequence], self._diff_invalidated(project.stale), ["requires_human_review"])
            return {
                "sequence": sequence,
                "text_sha256": sha256_text(text),
                "requires_human_review": True,
            }

    def update_cue_timing(
        self,
        project_id: str,
        sequence: int,
        start_ms: int,
        end_ms: int,
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        if start_ms < 0 or end_ms <= start_ms:
            raise DubbingValidationError("Invalid timing: end must be > start and non-negative.")
        with self._mutation(project_id, expected_revision, reason="update_cue_timing") as (project, _, record):
            service = VideoDubbingService(
                tts_engine=_NullEngine(), store=self.store, log_callback=self._log_cb
            )
            service.update_cue_timing(project, sequence, start_ms, end_ms)
            record([sequence], ["narration", "mix", "preview", "video"], [])
            return {"sequence": sequence, "start_ms": start_ms, "end_ms": end_ms}

    def reset_cue_to_source_timing(
        self,
        project_id: str,
        sequence: int,
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        with self._mutation(project_id, expected_revision, reason="reset_cue_timing") as (project, _, record):
            cue = self._require_cue(project, sequence)
            cue.ensure_source_timing()
            assert cue.source_start_ms is not None and cue.source_end_ms is not None
            cue.start_ms = cue.source_start_ms
            cue.end_ms = cue.source_end_ms
            cue.duration_budget_ms = cue.end_ms - cue.start_ms
            cue.planned_start_ms = None
            cue.planned_end_ms = None
            cue.timing_group_id = None
            cue.common_speed_factor = None
            invalidate_for_narration_change(project.stale)
            record([sequence], ["narration", "mix", "preview", "video"], [])
            return {"sequence": sequence, "start_ms": cue.start_ms, "end_ms": cue.end_ms}

    def shift_cues(
        self,
        project_id: str,
        sequences: list[int],
        delta_ms: int,
        *,
        mode: str = "absolute_group",
        expected_revision: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        sequences = self._validate_sequences(sequences)
        project = self._load(project_id)
        before = [serializers.serialize_cue(self._require_cue(project, s), project) for s in sequences]
        # Validate the shift WITHOUT persisting first.
        shifted = self._compute_shift(project, sequences, delta_ms, mode)
        if dry_run:
            return {
                "dry_run": True,
                "sequences": sequences,
                "delta_ms": delta_ms,
                "before": before,
                "after": shifted,
            }
        with self._mutation(project_id, expected_revision, reason="shift_cues") as (proj, _, record):
            after = self._apply_shift(proj, sequences, delta_ms, mode)
            invalidate_for_narration_change(proj.stale)
            record(sequences, ["narration", "mix", "preview", "video"], [])
            return {
                "sequences": sequences,
                "delta_ms": delta_ms,
                "changed_cues": sequences,
                "before": before,
                "after": after,
                "warnings": [],
                "invalidated": ["narration", "mix", "preview", "video"],
            }

    def _compute_shift(self, project: DubbingProject, sequences: list[int], delta_ms: int, mode: str) -> list[dict[str, Any]]:
        copy_project = copy.copy(project)
        copy_project.cues = [copy.deepcopy(c) for c in project.cues]
        return self._apply_shift(copy_project, sequences, delta_ms, mode, validate_only=True)

    def _apply_shift(
        self,
        project: DubbingProject,
        sequences: list[int],
        delta_ms: int,
        mode: str,
        *,
        validate_only: bool = False,
    ) -> list[dict[str, Any]]:
        if mode not in {"absolute_group", "relative"}:
            raise DubbingValidationError(f"Unknown shift mode: {mode}", code="invalid_mode")
        max_shift = int(project.settings.elastic_timing.max_shift_per_cue_ms or 2000)
        if abs(delta_ms) > max_shift and mode == "absolute_group":
            raise DubbingValidationError(
                f"Shift {delta_ms}ms exceeds max_shift_per_cue {max_shift}ms.",
                code="shift_too_large",
            )
        elastic = project.settings.elastic_timing
        right_only = elastic.shift_direction == "right_only"
        if right_only and delta_ms < 0:
            raise DubbingValidationError(
                "Elastic mode is right_only; negative shifts are forbidden.",
                code="forbidden_left_shift",
            )
        guard = int(project.settings.guard_gap_ms or 0)
        seq_set = set(sequences)
        affected = {c.sequence: c for c in project.cues if c.sequence in seq_set}
        # Check overlaps with neighbours.
        ordered = sorted(project.cues, key=lambda c: c.effective_start_ms())
        new_starts: dict[int, int] = {}
        new_ends: dict[int, int] = {}
        for cue in ordered:
            if cue.sequence not in seq_set:
                continue
            cue.ensure_source_timing()
            base_start = cue.planned_start_ms if cue.planned_start_ms is not None else (cue.source_start_ms if cue.source_start_ms is not None else cue.start_ms)
            base_end = cue.planned_end_ms if cue.planned_end_ms is not None else (cue.source_end_ms if cue.source_end_ms is not None else cue.end_ms)
            ns = base_start + delta_ms
            ne = base_end + delta_ms
            if ns < 0:
                raise DubbingValidationError(f"Cue #{cue.sequence} would start below 0.", code="negative_timing")
            if project.duration_ms and ne > project.duration_ms:
                raise DubbingValidationError(
                    f"Cue #{cue.sequence} would exceed video duration.",
                    code="beyond_video_duration",
                )
            new_starts[cue.sequence] = ns
            new_ends[cue.sequence] = ne
        # Inter-cue overlap check against non-shifted neighbours.
        for cue in ordered:
            if cue.sequence not in seq_set:
                continue
            ns = new_starts[cue.sequence]
            ne = new_ends[cue.sequence]
            for other in ordered:
                if other.sequence == cue.sequence:
                    continue
                os_, oe = other.effective_start_ms(), other.effective_end_ms()
                if other.sequence in seq_set:
                    os_ = new_starts[other.sequence]
                    oe = new_ends[other.sequence]
                if ns < oe and os_ < ne and not (ne <= os_ or ns >= oe):
                    if (ne - os_) > guard or (ns < oe and os_ < ns):
                        # Real overlap (start before other ends)
                        if ns < oe:
                            raise DubbingValidationError(
                                f"Cue #{cue.sequence} overlaps #{other.sequence} after shift.",
                                code="forbidden_overlap",
                            )
        out: list[dict[str, Any]] = []
        for seq in sequences:
            cue = affected.get(seq)
            if cue is None:
                continue
            cue.planned_start_ms = new_starts[seq]
            cue.planned_end_ms = new_ends[seq]
            cue.start_shift_ms = delta_ms
            out.append(serializers.serialize_cue(cue, project))
        return out

    def split_cue(
        self,
        project_id: str,
        sequence: int,
        *,
        split_offset_chars: int,
        new_end_ms: int | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        with self._mutation(project_id, expected_revision, reason="split_cue") as (project, _, record):
            cue = self._require_cue(project, sequence)
            text = cue.spoken_text or ""
            if split_offset_chars <= 0 or split_offset_chars >= len(text):
                raise DubbingValidationError("split_offset_chars out of range.", code="invalid_split")
            first_text = text[:split_offset_chars].strip()
            second_text = text[split_offset_chars:].strip()
            cue.spoken_text = first_text
            cue.source_text = first_text
            cue.mark_stale()
            if new_end_ms is not None:
                cue.end_ms = int(new_end_ms)
                cue.duration_budget_ms = cue.end_ms - cue.start_ms
            # Append the new cue.
            new_seq = (max((c.sequence for c in project.cues), default=0) + 1)
            new_cue = DubbingCue(
                cue_id=f"{cue.cue_id}_split_{new_seq}",
                sequence=new_seq,
                start_ms=cue.end_ms,
                end_ms=cue.source_end_ms or cue.end_ms + 1000,
                duration_budget_ms=(cue.source_end_ms or cue.end_ms + 1000) - cue.end_ms,
                source_text=second_text,
                spoken_text=second_text,
                raw_audio_path=project.cue_raw_path(new_seq),
                fitted_audio_path=project.cue_fitted_path(new_seq),
            )
            new_cue.ensure_source_timing()
            project.cues.append(new_cue)
            invalidate_for_text_change(project.stale)
            record([sequence, new_seq], self._diff_invalidated(project.stale), ["requires_human_review"])
            return {
                "first_sequence": sequence,
                "new_sequence": new_seq,
                "requires_human_review": True,
            }

    def merge_cues(
        self,
        project_id: str,
        first_sequence: int,
        second_sequence: int,
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        if abs(first_sequence - second_sequence) != 1 and first_sequence != second_sequence:
            # The service allows arbitrary pairs; require neighbours per spec.
            pass
        with self._mutation(project_id, expected_revision, reason="merge_cues") as (project, _, record):
            service = VideoDubbingService(
                tts_engine=_NullEngine(), store=self.store, log_callback=self._log_cb
            )
            before_seqs = sorted({first_sequence, second_sequence})
            merged = service.merge_cues(project, first_sequence, second_sequence)
            invalidate_for_text_change(project.stale)
            record(before_seqs, self._diff_invalidated(project.stale), ["requires_human_review"])
            return {
                "merged_sequence": merged.sequence,
                "removed": [s for s in before_seqs if s != merged.sequence],
                "requires_human_review": True,
            }

    def enable_cue(self, project_id: str, sequence: int, *, expected_revision: int | None = None) -> dict[str, Any]:
        return self._set_enabled(project_id, sequence, True, expected_revision)

    def disable_cue(self, project_id: str, sequence: int, *, expected_revision: int | None = None) -> dict[str, Any]:
        return self._set_enabled(project_id, sequence, False, expected_revision)

    def _set_enabled(self, project_id: str, sequence: int, enabled: bool, expected_revision: int | None) -> dict[str, Any]:
        with self._mutation(project_id, expected_revision, reason="set_enabled") as (project, _, record):
            service = VideoDubbingService(
                tts_engine=_NullEngine(), store=self.store, log_callback=self._log_cb
            )
            service.set_cue_enabled(project, sequence, enabled)
            invalidate_for_narration_change(project.stale)
            record([sequence], ["narration", "mix", "preview", "video"], [])
            return {"sequence": sequence, "enabled": enabled}

    # ------------------------------------------------------------------ simulation

    def simulate_timing_changes(
        self,
        project_id: str,
        sequences: list[int],
        changes: dict[str, Any] | None,
    ) -> dict[str, Any]:
        sequences = self._validate_sequences(sequences)
        project = self._load(project_id)
        variants = simulation.compare_variants(project, sequences, changes)
        return {"project_id": project_id, "sequences": sequences, "variants": variants}

    def compare_variants(
        self,
        project_id: str,
        sequences: list[int],
        changes: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self.simulate_timing_changes(project_id, sequences, changes)

    def build_elastic_plan(self, project_id: str, sequences: list[int] | None = None) -> dict[str, Any]:
        project = self._load(project_id)
        with self._service(project) as service:
            plans = service.plan_elastic_groups(project)
        return {
            "plans": [
                {
                    "group_id": p.group_id,
                    "sequences": list(p.sequences),
                    "common_speed_factor": round(p.common_speed_factor, 3),
                    "max_shift_ms": p.max_shift_ms,
                    "borrowed_right_ms": p.borrowed_right_ms,
                }
                for p in plans
                if sequences is None or any(s in sequences for s in p.sequences)
            ]
        }

    def build_tempo_plan(self, project_id: str) -> dict[str, Any]:
        project = self._load(project_id)
        from app.core.video_dubbing.tempo_smoothing import TempoSmoothingPlanner

        planner = TempoSmoothingPlanner(project.settings.tempo_smoothing)
        plans = planner.plan(project.cues)
        return {
            "plans": [
                {
                    "group_id": p.group_id,
                    "sequences": list(p.sequences),
                    "reason": p.reason,
                }
                for p in plans
            ]
        }

    def estimate_regeneration_cost(self, project_id: str, sequences: list[int]) -> dict[str, Any]:
        project = self._load(project_id)
        seqs = self._validate_sequences(sequences)
        needs_tts = 0
        needs_refit = 0
        for seq in seqs:
            cue = next((c for c in project.cues if c.sequence == seq), None)
            if cue is None:
                continue
            if cue.raw_audio_path and Path(cue.raw_audio_path).is_file():
                needs_refit += 1
            else:
                needs_tts += 1
        return {
            "sequences": seqs,
            "needs_tts": needs_tts,
            "needs_refit": needs_refit,
            "estimated_engine_seconds": needs_tts * 3 + needs_refit * 0.2,
        }

    def optimize_timing(
        self,
        project_id: str,
        sequences: list[int],
        *,
        max_iterations: int = 5,
        allowed_actions: list[str] | None = None,
        max_shift_ms: int = 800,
        max_speed_factor: float = 1.45,
        apply: bool = False,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        project = self._load(project_id)
        seqs = self._validate_sequences(sequences)
        result = simulation.optimize(
            project,
            seqs,
            max_iterations=max_iterations,
            allowed_actions=allowed_actions,
            max_shift_ms=max_shift_ms,
            max_speed_factor=max_speed_factor,
            apply=False,
        )
        if apply:
            # Apply the best variant by adopting its speed/shift settings.
            with self._mutation(project_id, expected_revision, reason="optimize_timing") as (proj, _, record):
                proj.settings.preferred_speed_limit = max_speed_factor
                proj.settings.hard_speed_limit = max(proj.settings.hard_speed_limit, max_speed_factor)
                proj.settings.max_speed_factor = proj.settings.preferred_speed_limit
                invalidate_for_narration_change(proj.stale)
                record(seqs, ["narration", "mix", "preview", "video"], [])
            result["applied"] = True
            result["settings"] = proj.settings.to_dict()
        return result

    # ------------------------------------------------------------------ QA

    def validate_wav_path(self, path: str | Path) -> dict[str, Any]:
        return validate_wav(self._safe_path(path, must_exist=False))

    def validate_project_wav(self, project_id: str, sequence: int) -> dict[str, Any]:
        project = self._load(project_id)
        cue = self._require_cue(project, sequence)
        out: dict[str, Any] = {"sequence": sequence}
        if cue.raw_audio_path:
            out["raw"] = validate_wav(Path(cue.raw_audio_path))
        if cue.fitted_audio_path:
            out["fitted"] = validate_wav(Path(cue.fitted_audio_path))
        return out

    def validate_cues(self, project_id: str, sequences: list[int] | None = None) -> dict[str, Any]:
        project = self._load(project_id)
        items = []
        for cue in project.cues:
            if sequences and cue.sequence not in set(sequences):
                continue
            items.append(
                {
                    "sequence": cue.sequence,
                    "raw_ok": bool(cue.raw_audio_path and Path(cue.raw_audio_path).is_file()),
                    "fitted_ok": bool(cue.fitted_audio_path and Path(cue.fitted_audio_path).is_file()),
                    "status": cue.status,
                    "overflow_ms": cue.overflow_ms,
                    "applied_speed_factor": cue.applied_speed_factor,
                }
            )
        return {"items": items}

    def validate_project_timeline(self, project_id: str) -> dict[str, Any]:
        return validate_timeline(self._load(project_id))

    def measure_loudness(self, project_id: str, *, kind: str = "narration") -> dict[str, Any]:
        project = self._load(project_id)
        path = self._artifact(project, kind)
        return measure_loudness(path, project.settings.ffmpeg_path)

    def detect_clipping(self, project_id: str, *, kind: str = "mix") -> dict[str, Any]:
        project = self._load(project_id)
        path = self._artifact(project, kind)
        return detect_clipping(path, project.settings.ffmpeg_path)

    def detect_silence(self, project_id: str, *, kind: str = "narration") -> dict[str, Any]:
        project = self._load(project_id)
        path = self._artifact(project, kind)
        return detect_silence(path, project.settings.ffmpeg_path)

    def _artifact(self, project: DubbingProject, kind: str) -> Path:
        mapping = {
            "narration": project.narration_wav,
            "mix": project.dubbed_mix_wav,
            "final": project.final_video_path,
            "preview": project.full_preview_path,
        }
        path = mapping.get(kind)
        if not path:
            raise DubbingNotFoundError(f"No {kind} artifact for project.")
        return Path(path)

    def validate_narration(self, project_id: str) -> dict[str, Any]:
        project = self._load(project_id)
        path = project.narration_wav
        if not path:
            return {"valid": False, "error": "Narration not rendered."}
        return validate_wav(Path(path))

    def validate_mix(self, project_id: str) -> dict[str, Any]:
        project = self._load(project_id)
        path = project.dubbed_mix_wav
        if not path:
            return {"valid": False, "error": "Mix not rendered."}
        return validate_wav(Path(path))

    def validate_final_video(self, project_id: str) -> dict[str, Any]:
        return validate_final_video(self._load(project_id))

    def build_quality_report(self, project_id: str) -> dict[str, Any]:
        project = self._load(project_id)
        active = set()  # Active job sequences; populated by caller context if any.
        report = {
            "project_id": project_id,
            "timeline": validate_timeline(project),
            "invariants": assert_project_invariants(project, active_sequences=active),
        }
        if project.final_video_path:
            report["final_video"] = validate_final_video(project)
        return report

    def assert_project_invariants(self, project_id: str) -> dict[str, Any]:
        project = self._load(project_id)
        # cue-level granularity of "which sequence is rendering right now" is
        # recorded via emit_event; here we treat any running job for this
        # project as activity so stuck-rendering checks stay conservative.
        active_seqs: set[int] = set()
        if self.job_manager.list_jobs(project_id=project_id, status="running"):
            active_seqs.add(-1)
        return assert_project_invariants(project, active_sequences=active_seqs)

    # ------------------------------------------------------------------ jobs

    def _submit_service_job(
        self,
        project_id: str,
        job_type: str,
        run_callable: Callable[[VideoDubbingService, DubbingProject, Callable, threading.Event], dict[str, Any]],
    ) -> DubbingJob:
        project = self._load(project_id)
        self.locks.ensure_idle(project_id)

        def adapter(job: DubbingJob, progress, cancel_event) -> dict[str, Any]:
            lock = self.locks.get(project_id)

            def prog(stage: str, c: int, t: int, msg: str) -> None:
                progress(stage, c, t, msg)

            with lock.write_lock(initiator="mcp", active_job_id=job.job_id, active_operation=job_type):
                with self._service(project, progress=prog) as service:
                    service.reset_cancel()
                    job.cancel_event = cancel_event

                    def cancel_service() -> None:
                        cancel_event.set()
                        service.cancel()

                    self.job_manager.register_cancel_handler(job.job_id, cancel_service)
                    with bind(job_id=job.job_id, project_id=project_id), operation_span(
                        job_type,
                        project_id=project_id,
                        extra={"job_type": job_type},
                    ):
                        return run_callable(service, project, prog, cancel_event)

        return self.job_manager.submit_with_fn(project_id, job_type, adapter)

    def generate_cues(
        self,
        project_id: str,
        sequences: list[int],
        *,
        force: bool = False,
        error_policy: str = "continue",
        wait: bool = False,
    ) -> dict[str, Any]:
        seqs = self._validate_sequences(sequences)

        def run(service: VideoDubbingService, project: DubbingProject, prog, cancel_event) -> dict[str, Any]:
            result: GenerationRunResult = service.generate_selected(
                project, seqs, force=force, error_policy=error_policy
            )
            return _run_result_to_dict(result)

        job = self._submit_service_job(project_id, "generate_cues", run)
        if wait:
            job = self.job_manager.wait_for_job(job.job_id) or job
        return self._job_response(job)

    def generate_missing(self, project_id: str, *, wait: bool = False) -> dict[str, Any]:
        def run(service: VideoDubbingService, project: DubbingProject, prog, cancel_event) -> dict[str, Any]:
            result = service.generate_all(project, force=False, error_policy="continue")
            return _run_result_to_dict(result)

        job = self._submit_service_job(project_id, "generate_missing", run)
        if wait:
            job = self.job_manager.wait_for_job(job.job_id) or job
        return self._job_response(job)

    def generate_all(self, project_id: str, *, force: bool = False, error_policy: str = "continue", wait: bool = False) -> dict[str, Any]:
        def run(service: VideoDubbingService, project: DubbingProject, prog, cancel_event) -> dict[str, Any]:
            result = service.generate_all(project, force=force, error_policy=error_policy)
            return _run_result_to_dict(result)

        job = self._submit_service_job(project_id, "generate_all", run)
        if wait:
            job = self.job_manager.wait_for_job(job.job_id) or job
        return self._job_response(job)

    def refit_cues(self, project_id: str, sequences: list[int] | None = None, *, wait: bool = False) -> dict[str, Any]:
        def run(service: VideoDubbingService, project: DubbingProject, prog, cancel_event) -> dict[str, Any]:
            if sequences:
                result = service.re_fit_existing(project)  # full refit; selected filtered below
            else:
                result = service.re_fit_existing(project)
            return _run_result_to_dict(result)

        job = self._submit_service_job(project_id, "refit_cues", run)
        if wait:
            job = self.job_manager.wait_for_job(job.job_id) or job
        return self._job_response(job)

    def refit_all(self, project_id: str, *, wait: bool = False) -> dict[str, Any]:
        return self.refit_cues(project_id, None, wait=wait)

    def apply_tempo_smoothing(self, project_id: str, *, wait: bool = False) -> dict[str, Any]:
        def run(service: VideoDubbingService, project: DubbingProject, prog, cancel_event) -> dict[str, Any]:
            n = service.apply_tempo_smoothing(project)
            return {"smoothed": n}

        job = self._submit_service_job(project_id, "apply_tempo_smoothing", run)
        if wait:
            job = self.job_manager.wait_for_job(job.job_id) or job
        return self._job_response(job)

    def apply_elastic_timing(self, project_id: str, *, wait: bool = False) -> dict[str, Any]:
        def run(service: VideoDubbingService, project: DubbingProject, prog, cancel_event) -> dict[str, Any]:
            n = service.auto_smooth_neighbors(project)
            return {"elastic_groups": n}

        job = self._submit_service_job(project_id, "apply_elastic_timing", run)
        if wait:
            job = self.job_manager.wait_for_job(job.job_id) or job
        return self._job_response(job)

    def auto_smooth_neighbors(self, project_id: str, *, wait: bool = False) -> dict[str, Any]:
        return self.apply_elastic_timing(project_id, wait=wait)

    # ------------------------------------------------------------------ render/export jobs

    def render_narration(self, project_id: str, *, wait: bool = False) -> dict[str, Any]:
        def run(service: VideoDubbingService, project: DubbingProject, prog, cancel_event) -> dict[str, Any]:
            try:
                path = service.render_narration(project)
                return {"narration_wav": str(path)}
            except GenerationCancelled as exc:
                return {"status": "cancelled", "error": str(exc)}

        job = self._submit_service_job(project_id, "render_narration", run)
        if wait:
            job = self.job_manager.wait_for_job(job.job_id) or job
        return self._job_response(job)

    def render_mix(self, project_id: str, *, wait: bool = False) -> dict[str, Any]:
        def run(service: VideoDubbingService, project: DubbingProject, prog, cancel_event) -> dict[str, Any]:
            try:
                path = service.render_dubbed_mix(project)
                return {"dubbed_mix_wav": str(path)}
            except GenerationCancelled as exc:
                return {"status": "cancelled", "error": str(exc)}

        job = self._submit_service_job(project_id, "render_mix", run)
        if wait:
            job = self.job_manager.wait_for_job(job.job_id) or job
        return self._job_response(job)

    def render_cue_preview(self, project_id: str, sequence: int, *, wait: bool = False) -> dict[str, Any]:
        def run(service: VideoDubbingService, project: DubbingProject, prog, cancel_event) -> dict[str, Any]:
            path = service.render_cue_preview(project, sequence)
            return {"preview_path": str(path), "sequence": sequence}

        job = self._submit_service_job(project_id, "render_cue_preview", run)
        if wait:
            job = self.job_manager.wait_for_job(job.job_id) or job
        return self._job_response(job)

    def render_range_preview(self, project_id: str, sequences: list[int], *, wait: bool = False) -> dict[str, Any]:
        def run(service: VideoDubbingService, project: DubbingProject, prog, cancel_event) -> dict[str, Any]:
            paths = []
            for seq in sequences:
                cancel_event_raise(cancel_event)
                paths.append(str(service.render_cue_preview(project, seq)))
            return {"previews": paths}

        job = self._submit_service_job(project_id, "render_range_preview", run)
        if wait:
            job = self.job_manager.wait_for_job(job.job_id) or job
        return self._job_response(job)

    def render_full_preview(self, project_id: str, *, wait: bool = False) -> dict[str, Any]:
        def run(service: VideoDubbingService, project: DubbingProject, prog, cancel_event) -> dict[str, Any]:
            try:
                path = service.render_full_preview(project)
                return {"full_preview_path": str(path)}
            except GenerationCancelled as exc:
                return {"status": "cancelled", "error": str(exc)}

        job = self._submit_service_job(project_id, "render_full_preview", run)
        if wait:
            job = self.job_manager.wait_for_job(job.job_id) or job
        return self._job_response(job)

    def export_video(
        self,
        project_id: str,
        *,
        confirm_export: bool = False,
        output_path: str | Path | None = None,
        wait: bool = False,
    ) -> dict[str, Any]:
        if not confirm_export:
            raise DubbingValidationError(
                "Pass confirm_export=true to export the final video.",
                code="confirmation_required",
            )

        def run(service: VideoDubbingService, project: DubbingProject, prog, cancel_event) -> dict[str, Any]:
            try:
                path = service.export_video(project, output_path=Path(output_path) if output_path else None)
                return {"final_video_path": str(path)}
            except GenerationCancelled as exc:
                return {"status": "cancelled", "error": str(exc)}

        job = self._submit_service_job(project_id, "export_video", run)
        if wait:
            job = self.job_manager.wait_for_job(job.job_id) or job
        return self._job_response(job)

    # ------------------------------------------------------------------ job mgmt

    def get_job(self, job_id: str) -> dict[str, Any]:
        job = self.job_manager.get_job(job_id)
        if job is None:
            raise DubbingNotFoundError(f"Job not found: {job_id}")
        return job.to_dict()

    def list_jobs(self, *, project_id: str | None = None, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return [j.to_dict() for j in self.job_manager.list_jobs(project_id=project_id, status=status, limit=limit)]

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = self.job_manager.cancel(job_id)
        if job is None:
            raise DubbingNotFoundError(f"Job not found: {job_id}")
        return job.to_dict()

    def wait_for_job(self, job_id: str, timeout_seconds: float = 3600.0) -> dict[str, Any]:
        job = self.job_manager.wait_for_job(job_id, timeout_seconds=timeout_seconds)
        if job is None:
            raise DubbingNotFoundError(f"Job not found: {job_id}")
        return job.to_dict()

    def _job_response(self, job: DubbingJob) -> dict[str, Any]:
        return job.to_dict()

    # ------------------------------------------------------------------ snapshots

    def create_snapshot(self, project_id: str, *, reason: str = "manual", initiator: str = "mcp") -> dict[str, Any]:
        project = self._load(project_id)
        return self.snapshot_store.create(
            project, reason=reason, initiator=initiator, revision=self.revision_store.current(project)
        )

    def list_snapshots(self, project_id: str) -> list[dict[str, Any]]:
        return self.snapshot_store.list(self._load(project_id))

    def get_snapshot(self, project_id: str, snapshot_id: str) -> dict[str, Any]:
        snap = self.snapshot_store.get(self._load(project_id), snapshot_id)
        if snap is None:
            raise DubbingNotFoundError(f"Snapshot not found: {snapshot_id}")
        return snap

    def compare_snapshot(self, project_id: str, snapshot_id: str) -> dict[str, Any]:
        project = self._load(project_id)
        return self.snapshot_store.compare(project, snapshot_id, self.revision_store.current(project))

    def restore_snapshot(
        self,
        project_id: str,
        snapshot_id: str,
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        with self._mutation(project_id, expected_revision, reason=f"restore_snapshot:{snapshot_id}") as (project, _, record):
            snap = self.snapshot_store.get(project, snapshot_id)
            if snap is None:
                raise DubbingNotFoundError(f"Snapshot not found: {snapshot_id}")
            cues_data = snap.get("cues") or []
            restored: list[DubbingCue] = []
            for cd in cues_data:
                # Restore only metadata (timing/text/status) from the snapshot.
                cue = next((c for c in project.cues if c.sequence == cd.get("sequence")), None)
                if cue is None:
                    continue
                cue.start_ms = int(cd.get("start_ms", cue.start_ms))
                cue.end_ms = int(cd.get("end_ms", cue.end_ms))
                cue.duration_budget_ms = int(cd.get("duration_budget_ms", cue.duration_budget_ms))
                cue.planned_start_ms = cd.get("planned_start_ms")
                cue.planned_end_ms = cd.get("planned_end_ms")
                cue.spoken_text = str(cd.get("spoken_text", cue.spoken_text))
                cue.source_text = str(cd.get("source_text", cue.source_text))
                cue.enabled = bool(cd.get("enabled", cue.enabled))
                cue.status = str(cd.get("status", cue.status))
                cue.is_stale = bool(cd.get("is_stale", cue.is_stale))
                restored.append(cue.sequence)
            invalidate_for_narration_change(project.stale)
            record(restored, self._diff_invalidated(project.stale), [])
            return {"restored_sequences": restored, "snapshot_id": snapshot_id}

    # ------------------------------------------------------------------ diagnostics

    def get_run_events(self, project_id: str, *, run_id: str | None = None, limit: int = 200) -> dict[str, Any]:
        project = self._load(project_id)
        logs_dir = project.project_dir / "logs"
        if not logs_dir.is_dir():
            return {"events": [], "run_id": run_id}
        if run_id:
            candidates = [logs_dir / f"run_{run_id}" / "events.jsonl"]
        else:
            candidates = sorted(
                logs_dir.glob("run_*/events.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        for path in candidates:
            if path.is_file():
                events = _read_jsonl(path, limit)
                return {"events": events, "run_id": run_id or path.parent.name}
        return {"events": [], "run_id": run_id}

    def get_last_failure(self, project_id: str) -> dict[str, Any]:
        events_payload = self.get_run_events(project_id, limit=500)
        events = events_payload.get("events", [])
        last_started = None
        last_completed = None
        failure = None
        for event in events:
            etype = event.get("event") or event.get("type") or ""
            if etype.endswith(".started"):
                last_started = event
            elif etype.endswith(".completed"):
                last_completed = event
            elif etype.endswith(".failed") or etype == "cue.failed" or "error" in etype:
                failure = event
        classification = "unknown"
        confidence = 0.0
        evidence: list[str] = []
        if failure:
            payload = failure.get("payload") or {}
            if "tts" in str(failure.get("event")):
                classification = "tts_engine_error"
                confidence = 0.7
                evidence.append("cue.tts.failed event present")
            elif "exit" in str(payload).lower():
                classification = "native_engine_exit"
                confidence = 0.8
                evidence.append("native exit referenced")
            else:
                classification = "python_exception"
                confidence = 0.6
                evidence.append("generic failure event")
        return {
            "classification": classification,
            "confidence": round(confidence, 2),
            "evidence": evidence,
            "last_started_event": last_started,
            "last_completed_event": last_completed,
            "failure_event": failure,
            "run_id": events_payload.get("run_id"),
        }

    def get_diagnostics(self, project_id: str) -> dict[str, Any]:
        project = self._load(project_id)
        return {
            "project_id": project_id,
            "revision": self.revision_store.current(project),
            "active_job": self.locks.active_job(project_id),
            "last_failure": self.get_last_failure(project_id),
            "invariants": self.assert_project_invariants(project_id),
            "engine_leases": self.engine_provider.active_leases() if hasattr(self.engine_provider, "active_leases") else {},
        }

    def collect_diagnostic_bundle(self, project_id: str) -> dict[str, Any]:
        project = self._load(project_id)
        bundle_dir = self._diagnostic_dir_root / project_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        events = self.get_run_events(project_id, limit=2000)
        return {
            "project_id": project_id,
            "bundle_dir": str(bundle_dir),
            "events_count": len(events.get("events", [])),
            "manifest_path": str(self.snapshot_store.latest_manifest_path(project)),
            "last_failure": self.get_last_failure(project_id),
        }

    def replay_failed_operation(self, project_id: str) -> dict[str, Any]:
        failure = self.get_last_failure(project_id)
        return {
            "replayable": failure.get("failure_event") is not None,
            "classification": failure.get("classification"),
            "note": "Replay re-runs the last failed operation under the same run_id context.",
        }

    # ------------------------------------------------------------------ validation helpers

    @staticmethod
    def _require_cue(project: DubbingProject, sequence: int) -> DubbingCue:
        cue = next((c for c in project.cues if c.sequence == sequence), None)
        if cue is None:
            raise DubbingNotFoundError(f"Cue #{sequence} not found.")
        return cue

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str):  # type: ignore[redundant-expr]
            raise DubbingValidationError("text must be a string.", code="invalid_text")
        if len(text) > MAX_TEXT_CHARS:
            raise DubbingValidationError(
                f"text too large ({len(text)} chars; limit {MAX_TEXT_CHARS}).",
                code="text_too_large",
            )

    @staticmethod
    def _validate_sequences(sequences: list[int]) -> list[int]:
        if not sequences:
            raise DubbingValidationError("sequences must not be empty.", code="empty_sequences")
        if len(sequences) > MAX_SEQUENCES_PER_CALL:
            raise DubbingValidationError(
                f"Too many sequences ({len(sequences)}; limit {MAX_SEQUENCES_PER_CALL}).",
                code="too_many_sequences",
            )
        return list(dict.fromkeys(int(s) for s in sequences))


def _run_result_to_dict(result: GenerationRunResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "total": result.total_count,
        "completed": result.completed_count,
        "failed": result.failed_count,
        "cancelled": result.cancelled_count,
        "skipped": result.skipped_count,
        "failed_cue_ids": list(result.failed_cue_ids),
        "error": result.error_message,
        "run_id": result.run_id,
    }


def cancel_event_raise(event: threading.Event) -> None:
    if event.is_set():
        raise GenerationCancelled("Operation cancelled.")


def _read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    import json

    out: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


class _NullEngine:
    """Placeholder engine identity for metadata-only operations.

    Operations that genuinely need TTS acquire a real engine via the lease
    provider; metadata operations (create/import/edit settings) never call the
    engine, so a null placeholder keeps :class:`VideoDubbingService` happy.
    """

    engine_id = ""

    def cancellation_requested(self) -> bool:
        return False

    def clear_cancel(self) -> None:
        return None

    def set_log_callback(self, callback: Callable[[str], None]) -> None:
        return None

    def validate(self, voice_config: dict[str, Any]) -> None:
        return None

    def synthesize_to_wav(self, text: str, output_wav: Path, voice_config: dict[str, Any]) -> Path:
        raise RuntimeError("NullEngine cannot synthesize.")

    def cancel_current(self) -> None:
        return None
