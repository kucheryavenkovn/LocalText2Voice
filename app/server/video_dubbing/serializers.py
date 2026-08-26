"""DTO serializers for video-dubbing projects.

The facade never returns raw domain dataclasses to MCP/HTTP. It returns plain
dicts produced here, so cue text can be omitted/hashed where appropriate and
paths are normalised relative to the project directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from app.core.video_dubbing.models import DubbingCue, DubbingProject
from app.observability import sha256_text

# Cue statuses that imply the cue still needs work before it can be exported.
_NEEDS_WORK = {
    "pending",
    "rendering",
    "stale",
    "failed",
    "missing_audio",
    "cancelled",
    "needs_text_shortening",
    "elastic_group_failed",
    "extreme_speed_required",
    "disabled",
}


def _rel(path: Path | str | None, project: DubbingProject) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).relative_to(project.project_dir)).replace("\\", "/")
    except (ValueError, TypeError):
        return Path(path).name


def _abs(path: Path | str | None) -> str:
    if not path:
        return ""
    return str(path)


def serialize_project(
    project: DubbingProject,
    *,
    include_cues: bool = False,
    include_text: bool = False,
    max_cues: int = 500,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "project_id": project.project_id,
        "title": project.title,
        "project_dir": _abs(project.project_dir),
        "video_path": _abs(project.video_path),
        "video_probe": project.video_probe.to_dict() if project.video_probe else None,
        "duration_ms": project.duration_ms,
        "srt_source_path": _abs(project.srt_source_path),
        "settings": project.settings.to_dict(),
        "stale": project.stale.to_dict(),
        "cue_count": len(project.cues),
        "enabled_cue_count": sum(1 for c in project.cues if c.enabled),
        "narration_wav": _abs(project.narration_wav),
        "dubbed_mix_wav": _abs(project.dubbed_mix_wav),
        "final_video_path": _abs(project.final_video_path),
        "full_preview_path": _abs(project.full_preview_path),
        "created_at": project.created_at,
        "updated_at": project.updated_at,
    }
    if include_cues:
        cues = list(project.cues)[:max_cues]
        payload["cues"] = [serialize_cue(c, project, include_text=include_text) for c in cues]
    return payload


def serialize_state(project: DubbingProject) -> dict[str, Any]:
    """Compact pipeline-state snapshot (no cue text, no full cue list)."""
    by_status: dict[str, int] = {}
    total_overflow = 0
    max_speed = 1.0
    needs_tts = 0
    needs_refit = 0
    for cue in project.cues:
        by_status[cue.status] = by_status.get(cue.status, 0) + 1
        total_overflow += max(0, int(cue.overflow_ms or 0))
        max_speed = max(max_speed, float(cue.applied_speed_factor or 1.0))
        if cue.status in _NEEDS_WORK:
            if cue.raw_audio_path and Path(cue.raw_audio_path).is_file():
                needs_refit += 1
            else:
                needs_tts += 1
    return {
        "project_id": project.project_id,
        "revision_token": _revision_token(project),
        "stale": project.stale.to_dict(),
        "status_counts": by_status,
        "total_overflow_ms": total_overflow,
        "max_speed_factor": round(max_speed, 3),
        "needs_tts": needs_tts,
        "needs_refit": needs_refit,
        "has_video": project.video_path is not None,
        "has_srt": bool(project.srt_source_text),
        "narration_ready": bool(
            project.narration_wav and Path(project.narration_wav).is_file()
        ),
        "mix_ready": bool(
            project.dubbed_mix_wav and Path(project.dubbed_mix_wav).is_file()
        ),
        "final_video_ready": bool(
            project.final_video_path and Path(project.final_video_path).is_file()
        ),
    }


def _revision_token(project: DubbingProject) -> str:
    """A content-derived token used as a fallback revision source.

    The authoritative monotonic revision lives in :class:`RevisionStore`; this
    token is only used to surface a revision-like value in read-only payloads
    when the store is unavailable.
    """
    return project.updated_at or ""


def serialize_cue(
    cue: DubbingCue,
    project: DubbingProject | None = None,
    *,
    include_text: bool = False,
) -> dict[str, Any]:
    text = cue.spoken_text or ""
    payload: dict[str, Any] = {
        "cue_id": cue.cue_id,
        "sequence": cue.sequence,
        "enabled": cue.enabled,
        "start_ms": cue.start_ms,
        "end_ms": cue.end_ms,
        "duration_budget_ms": cue.duration_budget_ms,
        "source_start_ms": cue.source_start_ms,
        "source_end_ms": cue.source_end_ms,
        "planned_start_ms": cue.planned_start_ms,
        "planned_end_ms": cue.planned_end_ms,
        "raw_duration_ms": cue.raw_duration_ms,
        "fitted_duration_ms": cue.fitted_duration_ms,
        "required_speed_factor": cue.required_speed_factor,
        "planned_speed_factor": cue.planned_speed_factor,
        "applied_speed_factor": cue.applied_speed_factor,
        "overflow_ms": cue.overflow_ms,
        "status": cue.status,
        "fitting_strategy": cue.fitting_strategy,
        "warning_codes": list(cue.warning_codes),
        "error_message": cue.error_message,
        "is_stale": cue.is_stale,
        "text_sha256": sha256_text(text),
        "text_length": len(text),
        "generation_fingerprint": cue.generation_fingerprint,
        "fit_fingerprint": cue.fit_fingerprint,
        "legacy_audio_unverified": cue.legacy_audio_unverified,
        "timing_group_id": cue.timing_group_id,
        "common_speed_factor": cue.common_speed_factor,
        "start_shift_ms": cue.start_shift_ms,
        "borrowed_right_ms": cue.borrowed_right_ms,
        "timing_locked": cue.timing_locked,
        "smoothing_group_id": cue.smoothing_group_id,
    }
    if include_text:
        payload["spoken_text"] = text
        payload["source_text"] = cue.source_text
    if project is not None:
        payload["raw_audio_rel"] = _rel(cue.raw_audio_path, project)
        payload["fitted_audio_rel"] = _rel(cue.fitted_audio_path, project)
        payload["raw_audio_exists"] = bool(
            cue.raw_audio_path and Path(cue.raw_audio_path).is_file()
        )
        payload["fitted_audio_exists"] = bool(
            cue.fitted_audio_path and Path(cue.fitted_audio_path).is_file()
        )
    return payload


def serialize_cues_page(
    cues: Iterable[DubbingCue],
    project: DubbingProject,
    *,
    page: int = 1,
    page_size: int = 50,
    include_text: bool = False,
) -> dict[str, Any]:
    items = list(cues)
    page = max(1, int(page))
    page_size = max(1, min(200, int(page_size)))
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    slice_ = items[start:end]
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": [serialize_cue(c, project, include_text=include_text) for c in slice_],
    }


def serialize_neighbors(
    project: DubbingProject, sequence: int, *, include_text: bool = False
) -> dict[str, Any]:
    ordered = sorted(project.cues, key=lambda c: c.sequence)
    target = None
    before = None
    after = None
    for index, cue in enumerate(ordered):
        if cue.sequence == sequence:
            target = cue
            if index > 0:
                before = ordered[index - 1]
            if index + 1 < len(ordered):
                after = ordered[index + 1]
            break
    if target is None:
        return {"found": False}
    return {
        "found": True,
        "cue": serialize_cue(target, project, include_text=include_text),
        "previous": serialize_cue(before, project, include_text=include_text) if before else None,
        "next": serialize_cue(after, project, include_text=include_text) if after else None,
    }


def filter_cues(
    cues: Iterable[DubbingCue],
    *,
    status: str | None = None,
    enabled: bool | None = None,
    has_error: bool | None = None,
    has_overflow: bool | None = None,
    needs_tts: bool | None = None,
    needs_refit: bool | None = None,
    sequence_from: int | None = None,
    sequence_to: int | None = None,
    time_from_ms: int | None = None,
    time_to_ms: int | None = None,
) -> list[DubbingCue]:
    result: list[DubbingCue] = []
    for cue in cues:
        if status is not None and cue.status != status:
            continue
        if enabled is not None and cue.enabled != enabled:
            continue
        if has_error is True and not cue.error_message:
            continue
        if has_error is False and cue.error_message:
            continue
        if has_overflow is True and (cue.overflow_ms or 0) <= 0:
            continue
        if has_overflow is False and (cue.overflow_ms or 0) > 0:
            continue
        if sequence_from is not None and cue.sequence < sequence_from:
            continue
        if sequence_to is not None and cue.sequence > sequence_to:
            continue
        if time_from_ms is not None and cue.end_ms < time_from_ms:
            continue
        if time_to_ms is not None and cue.start_ms > time_to_ms:
            continue
        if needs_tts is True and (
            cue.raw_audio_path and Path(cue.raw_audio_path).is_file()
        ):
            continue
        if needs_tts is False and not (
            cue.raw_audio_path and Path(cue.raw_audio_path).is_file()
        ):
            continue
        if needs_refit is True and cue.status not in _NEEDS_WORK:
            continue
        if needs_refit is False and cue.status in _NEEDS_WORK:
            continue
        result.append(cue)
    return result


def serialize_import_result(cues: list[DubbingCue], warnings: list[Any]) -> dict[str, Any]:
    return {
        "imported_cues": len(cues),
        "warnings": [
            {"code": getattr(w, "code", ""), "message": getattr(w, "message", str(w)),
             "sequence": getattr(w, "sequence", None)}
            for w in warnings
        ],
    }
