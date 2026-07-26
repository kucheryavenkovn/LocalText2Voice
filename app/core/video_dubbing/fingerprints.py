from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .models import DubbingCue, DubbingProject, DubbingProjectSettings, ElasticTimingSettings


def _stable_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_content_hash(path: Path | str | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def reference_voice_identity(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.is_file():
        return {"path": str(file_path)}
    stat = file_path.stat()
    return {
        "path": str(file_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)),
        "content_hash": file_content_hash(file_path),
    }


def extract_reference_path(voice_config: Mapping[str, Any] | None) -> str | None:
    if not voice_config:
        return None
    for key in ("reference_audio_path", "reference_voice_path", "ref_audio"):
        value = voice_config.get(key)
        if value:
            return str(value)
    return None


def raw_generation_fingerprint(
    cue: DubbingCue,
    *,
    engine_id: str,
    voice_id: str | None,
    voice_config: Mapping[str, Any] | None,
    language: str,
    sample_rate: int,
    channels: int,
    reference_voice_content_hash: str | None = None,
    speed: float | None = None,
) -> str:
    config = dict(voice_config or {})
    # Drop runtime-only keys that must not affect provenance.
    config.pop("_dubbing_sequence", None)
    ref_hash = reference_voice_content_hash
    if ref_hash is None:
        ref_path = extract_reference_path(config)
        ref_hash = file_content_hash(ref_path) if ref_path else None
    payload = {
        "kind": "raw_generation",
        "spoken_text": cue.spoken_text,
        "engine_id": engine_id,
        "model_id": config.get("model") or config.get("model_id"),
        "voice_id": voice_id or config.get("voice") or config.get("speaker") or "",
        "voice_config": config,
        "reference_voice_content_hash": ref_hash or "",
        "language": language,
        "sample_rate": sample_rate,
        "channels": channels,
        "speed": speed if speed is not None else config.get("speed"),
        "seed": config.get("seed"),
        "temperature": config.get("temperature"),
    }
    return _sha256_text(_stable_dumps(payload))


def fit_settings_fingerprint(
    settings: DubbingProjectSettings,
    elastic: ElasticTimingSettings | None = None,
) -> str:
    elastic = elastic or settings.elastic_timing
    payload = {
        "kind": "fit_settings",
        "preferred_speed_limit": settings.preferred_speed_limit,
        "hard_speed_limit": settings.hard_speed_limit,
        "exact_timing": settings.exact_timing,
        "guard_gap_ms": settings.guard_gap_ms,
        "compress_internal_pauses": settings.compress_internal_pauses,
        "internal_pause_keep_ms": settings.internal_pause_keep_ms,
        "sample_rate": settings.sample_rate,
        "channels": settings.channels,
        "sync_mode": settings.sync_mode.value
        if hasattr(settings.sync_mode, "value")
        else str(settings.sync_mode),
        "elastic": elastic.to_dict() if elastic is not None else {},
        "tempo_smoothing": settings.tempo_smoothing.to_dict()
        if getattr(settings, "tempo_smoothing", None) is not None
        else {},
        "fit_pipeline_version": 1,
    }
    return _sha256_text(_stable_dumps(payload))


def fit_fingerprint(
    cue: DubbingCue,
    *,
    raw_generation_fp: str,
    raw_wav_hash: str | None,
    settings: DubbingProjectSettings,
    target_duration_ms: int | None = None,
    common_speed_factor: float | None = None,
    planned_start_ms: int | None = None,
    planned_end_ms: int | None = None,
    timing_group_id: str | None = None,
) -> str:
    payload = {
        "kind": "fit",
        "raw_generation_fingerprint": raw_generation_fp,
        "raw_wav_hash": raw_wav_hash or "",
        "fit_settings": fit_settings_fingerprint(settings),
        "target_duration_ms": target_duration_ms
        if target_duration_ms is not None
        else cue.target_duration_ms,
        "max_speed_factor": settings.hard_speed_limit,
        "compress_internal_pauses": settings.compress_internal_pauses,
        "internal_pause_keep_ms": settings.internal_pause_keep_ms,
        "common_speed_factor": common_speed_factor
        if common_speed_factor is not None
        else cue.common_speed_factor,
        "planned_speed_factor": cue.planned_speed_factor,
        "smoothing_group_id": cue.smoothing_group_id,
        "timing_group_id": timing_group_id
        if timing_group_id is not None
        else cue.timing_group_id,
        "planned_start_ms": planned_start_ms
        if planned_start_ms is not None
        else cue.planned_start_ms,
        "planned_end_ms": planned_end_ms
        if planned_end_ms is not None
        else cue.planned_end_ms,
        "force_fit": cue.force_fit,
        "hard_speed_override": cue.hard_speed_override,
        "sample_rate": settings.sample_rate,
        "channels": settings.channels,
        "fit_pipeline_version": 1,
    }
    return _sha256_text(_stable_dumps(payload))


def timeline_fingerprint(project: DubbingProject) -> str:
    cues_payload = []
    for cue in project.cues:
        if not cue.enabled:
            continue
        cues_payload.append(
            {
                "cue_id": cue.cue_id,
                "sequence": cue.sequence,
                "fitted_fingerprint": cue.fit_fingerprint,
                "planned_start_ms": cue.planned_start_ms
                if cue.planned_start_ms is not None
                else cue.start_ms,
                "planned_end_ms": cue.planned_end_ms
                if cue.planned_end_ms is not None
                else cue.end_ms,
                "fitted_duration_ms": cue.fitted_duration_ms,
                "status": cue.status,
            }
        )
    payload = {
        "kind": "timeline",
        "video_duration_ms": project.duration_ms,
        "sample_rate": project.settings.sample_rate,
        "channels": project.settings.channels,
        "cues": cues_payload,
    }
    return _sha256_text(_stable_dumps(payload))


def mix_fingerprint(project: DubbingProject, narration_fp: str = "") -> str:
    ducking = project.settings.ducking
    video_identity = ""
    if project.video_path and Path(project.video_path).is_file():
        video_identity = file_content_hash(project.video_path) or str(project.video_path)
    payload = {
        "kind": "mix",
        "narration_fingerprint": narration_fp or timeline_fingerprint(project),
        "original_audio_identity": video_identity,
        "mode": ducking.mode.value,
        "original_outside_percent": ducking.original_outside_percent,
        "original_during_percent": ducking.original_during_percent,
        "narration_volume_percent": ducking.narration_volume_percent,
        "attack_ms": ducking.attack_ms,
        "release_ms": ducking.release_ms,
        "normalize": ducking.normalize,
        "target_lufs": ducking.target_lufs,
        "true_peak_db": ducking.true_peak_db,
    }
    return _sha256_text(_stable_dumps(payload))


def export_fingerprint(project: DubbingProject, mix_fp: str = "") -> str:
    export = project.settings.export
    video_identity = ""
    if project.video_path and Path(project.video_path).is_file():
        video_identity = file_content_hash(project.video_path) or str(project.video_path)
    payload = {
        "kind": "export",
        "video_identity": video_identity,
        "mix_fingerprint": mix_fp or mix_fingerprint(project),
        "container": export.container.value,
        "include_narration_only": export.include_narration_only,
        "embed_subtitles": export.embed_subtitles,
        "audio_bitrate": export.audio_bitrate,
        "video_codec_override": export.video_codec_override,
        "force_video_reencode": export.force_video_reencode,
    }
    return _sha256_text(_stable_dumps(payload))


def legacy_combined_fingerprint(
    cue: DubbingCue, settings: DubbingProjectSettings
) -> str:
    """Backward-compatible single fingerprint used by older projects/tests."""
    return raw_generation_fingerprint(
        cue,
        engine_id=settings.tts_engine,
        voice_id=settings.voice,
        voice_config=settings.voice_config,
        language=settings.language,
        sample_rate=settings.sample_rate,
        channels=settings.channels,
        speed=(settings.voice_config or {}).get("speed"),
    )
