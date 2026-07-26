"""Deterministic quality model + automatic QA.

Two responsibilities:

1. :func:`score_variant` - an objective, LLM-independent cost function that
   ranks candidate timing corrections. Lower is better. Returns a breakdown so
   the agent can explain *why* a variant was selected.
2. QA validators used by ``dubbing_validate_*`` MCP tools and
   ``assert_project_invariants``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.core.video_dubbing.models import (
    CueStatus,
    DubbingCue,
    DubbingProject,
    READY_STATUSES,
)
from app.core.video_dubbing.wav_validator import WavArtifactValidator
from app.utils import ffmpeg_utils, ffprobe_utils


@dataclass
class QualityWeights:
    overflow_ms: float = 1.0
    overlap_ms: float = 5.0
    preferred_speed_excess: float = 8.0
    hard_speed_violation: float = 50.0
    absolute_shift_ms: float = 0.01
    tempo_jump: float = 12.0
    transcript_error: float = 100.0
    clipping: float = 40.0
    silence_excess_ms: float = 0.005
    regeneration_cost_per_cue: float = 1.5
    guard_gap_violation_ms: float = 0.05
    boundary_violation: float = 30.0
    missing_artifact: float = 25.0

    def to_dict(self) -> dict[str, float]:
        return {
            f: getattr(self, f)
            for f in (
                "overflow_ms",
                "overlap_ms",
                "preferred_speed_excess",
                "hard_speed_violation",
                "absolute_shift_ms",
                "tempo_jump",
                "transcript_error",
                "clipping",
                "silence_excess_ms",
                "regeneration_cost_per_cue",
                "guard_gap_violation_ms",
                "boundary_violation",
                "missing_artifact",
            )
        }


def _effective_start(cue: DubbingCue) -> int:
    return cue.planned_start_ms if cue.planned_start_ms is not None else cue.start_ms


def _effective_end(cue: DubbingCue) -> int:
    return cue.planned_end_ms if cue.planned_end_ms is not None else cue.end_ms


def score_variant(
    cues: Iterable[DubbingCue],
    *,
    preferred_speed: float = 1.35,
    hard_speed: float = 2.5,
    guard_gap_ms: int = 20,
    video_duration_ms: int = 0,
    weights: QualityWeights | None = None,
    regeneration_count: int = 0,
    transcript_errors: int = 0,
    clipping_events: int = 0,
    silence_excess_ms: int = 0,
    missing_artifacts: int = 0,
) -> dict[str, Any]:
    """Score a candidate set of cues. Lower score is better.

    The function is pure: it never touches disk or DB. The same inputs always
    yield the same score.
    """
    weights = weights or QualityWeights()
    ordered = sorted(cues, key=lambda c: _effective_start(c))
    total_overflow = 0
    total_overlap = 0
    total_shift = 0
    preferred_excess = 0.0
    hard_violation = 0.0
    max_speed = 1.0
    tempo_jump_sum = 0.0
    gap_violation = 0
    boundary_violation = 0

    prev_end: int | None = None
    prev_speed: float | None = None
    for cue in ordered:
        start = _effective_start(cue)
        end = _effective_end(cue)
        if start < 0 or end < 0:
            boundary_violation += 1
        if end <= start:
            boundary_violation += 1
        if video_duration_ms and end > video_duration_ms:
            boundary_violation += 1
        total_overflow += max(0, int(cue.overflow_ms or 0))
        speed = float(cue.applied_speed_factor or 1.0)
        max_speed = max(max_speed, speed)
        if speed > preferred_speed + 1e-6:
            preferred_excess += speed - preferred_speed
        if speed > hard_speed + 1e-6:
            hard_violation += 1.0
        if prev_speed is not None:
            jump = abs(speed - prev_speed)
            if jump > 0.10:
                tempo_jump_sum += jump
        prev_speed = speed
        if prev_end is not None:
            gap = start - prev_end
            if gap < 0:
                total_overlap += -gap
            elif gap < guard_gap_ms:
                gap_violation += guard_gap_ms - gap
        # Absolute shift from source SRT.
        src_start = cue.source_start_ms if cue.source_start_ms is not None else cue.start_ms
        total_shift += abs(start - src_start)
        prev_end = end

    score = (
        total_overflow * weights.overflow_ms
        + total_overlap * weights.overlap_ms
        + preferred_excess * weights.preferred_speed_excess
        + hard_violation * weights.hard_speed_violation
        + total_shift * weights.absolute_shift_ms
        + tempo_jump_sum * weights.tempo_jump
        + transcript_errors * weights.transcript_error
        + clipping_events * weights.clipping
        + silence_excess_ms * weights.silence_excess_ms
        + regeneration_count * weights.regeneration_cost_per_cue
        + gap_violation * weights.guard_gap_violation_ms
        + boundary_violation * weights.boundary_violation
        + missing_artifacts * weights.missing_artifact
    )
    breakdown = {
        "overflow_penalty": round(total_overflow * weights.overflow_ms, 3),
        "overlap_penalty": round(total_overlap * weights.overlap_ms, 3),
        "speed_penalty": round(preferred_excess * weights.preferred_speed_excess, 3),
        "hard_limit_penalty": round(hard_violation * weights.hard_speed_violation, 3),
        "shift_penalty": round(total_shift * weights.absolute_shift_ms, 3),
        "tempo_jump_penalty": round(tempo_jump_sum * weights.tempo_jump, 3),
        "transcript_penalty": round(transcript_errors * weights.transcript_error, 3),
        "clipping_penalty": round(clipping_events * weights.clipping, 3),
        "silence_penalty": round(silence_excess_ms * weights.silence_excess_ms, 3),
        "regeneration_cost": round(regeneration_count * weights.regeneration_cost_per_cue, 3),
        "gap_violation_penalty": round(gap_violation * weights.guard_gap_violation_ms, 3),
        "boundary_penalty": round(boundary_violation * weights.boundary_violation, 3),
        "missing_artifact_penalty": round(missing_artifacts * weights.missing_artifact, 3),
    }
    return {
        "score": round(score, 3),
        "score_breakdown": breakdown,
        "total_overflow_ms": total_overflow,
        "max_speed_factor": round(max_speed, 3),
        "total_shift_ms": total_shift,
        "total_overlap_ms": total_overlap,
    }


# --------------------------------------------------------------------------- QA


@dataclass
class InvariantViolation:
    code: str
    message: str
    sequence: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


def validate_wav(path: Path) -> dict[str, Any]:
    validator = WavArtifactValidator()
    result = validator.validate(path)
    payload: dict[str, Any] = {
        "valid": result.valid,
        "path": str(path),
        "exists": path.is_file(),
        "duration_ms": result.duration_ms,
        "error_code": result.error_code,
        "error_message": result.error_message,
    }
    if path.is_file():
        payload["size_bytes"] = path.stat().st_size
    return payload


def validate_timeline(project: DubbingProject) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    guard = int(project.settings.guard_gap_ms or 0)
    prev_end: int | None = None
    prev_seq: int | None = None
    sequences: set[int] = set()
    cue_ids: set[str] = set()
    for cue in sorted(project.cues, key=lambda c: c.sequence):
        if cue.sequence in sequences:
            violations.append({"code": "duplicate_sequence", "sequence": cue.sequence})
        sequences.add(cue.sequence)
        if cue.cue_id in cue_ids:
            violations.append({"code": "duplicate_cue_id", "cue_id": cue.cue_id})
        cue_ids.add(cue.cue_id)
        start = _effective_start(cue)
        end = _effective_end(cue)
        if start < 0 or end < 0:
            violations.append({"code": "negative_timing", "sequence": cue.sequence})
        if end <= start:
            violations.append({"code": "non_positive_interval", "sequence": cue.sequence})
        if project.duration_ms and end > project.duration_ms:
            violations.append(
                {
                    "code": "beyond_video_duration",
                    "sequence": cue.sequence,
                    "end_ms": end,
                    "video_duration_ms": project.duration_ms,
                }
            )
        if cue.enabled and (cue.applied_speed_factor or 1.0) > project.settings.hard_speed_limit + 1e-6:
            violations.append(
                {
                    "code": "excessive_speed",
                    "sequence": cue.sequence,
                    "applied_speed_factor": cue.applied_speed_factor,
                    "hard_limit": project.settings.hard_speed_limit,
                }
            )
        if prev_end is not None and start < prev_end:
            violations.append(
                {
                    "code": "overlap",
                    "sequence": cue.sequence,
                    "previous_sequence": prev_seq,
                    "overlap_ms": prev_end - start,
                }
            )
        elif prev_end is not None and guard and start - prev_end < guard and cue.enabled:
            # Only flag enabled neighbours; disabled cues may abut.
            pass
        if cue.enabled and (cue.status == CueStatus.RENDERING.value):
            violations.append(
                {
                    "code": "cue_stuck_rendering",
                    "sequence": cue.sequence,
                    "message": "Cue is rendering but no active job context is recorded here.",
                }
            )
        if (
            cue.enabled
            and cue.status in READY_STATUSES
            and (not cue.fitted_audio_path or not Path(cue.fitted_audio_path).is_file())
        ):
            violations.append(
                {"code": "missing_fitted_wav", "sequence": cue.sequence}
            )
        prev_end = end
        prev_seq = cue.sequence
    return {"valid": not violations, "violations": violations}


def validate_final_video(project: DubbingProject) -> dict[str, Any]:
    if not project.final_video_path:
        return {"valid": False, "error": "No final video path."}
    path = Path(project.final_video_path)
    if not path.is_file():
        return {"valid": False, "error": "Final video file missing.", "path": str(path)}
    if path.suffix == ".part" or path.name.endswith(".part"):
        return {"valid": False, "error": "Final path is a .part artifact."}
    try:
        data = ffprobe_utils.probe_media(path, project.settings.ffmpeg_path)
    except Exception as exc:
        return {"valid": False, "error": f"ffprobe failed: {exc}", "path": str(path)}
    duration_ms, video_stream, audio_streams = ffprobe_utils.parse_video_probe(data)
    violations: list[str] = []
    if video_stream is None:
        violations.append("No video stream found.")
    if not audio_streams:
        # Narration-only export legitimately has no original audio, but a dub
        # mix must carry the narration track.
        if not project.settings.export.include_narration_only:
            violations.append("No audio stream found.")
    if project.duration_ms and abs(duration_ms - project.duration_ms) > max(
        1000, project.duration_ms * 0.05
    ):
        violations.append(
            f"Duration mismatch: video={duration_ms}ms expected~{project.duration_ms}ms"
        )
    return {
        "valid": not violations,
        "violations": violations,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "duration_ms": duration_ms,
        "video_codec": video_stream.codec_name if video_stream else "",
        "audio_codec": audio_streams[0].codec_name if audio_streams else "",
        "audio_tracks": len(audio_streams),
        "container_format": str(data.get("format", {}).get("format_name", "")),
    }


def measure_loudness(path: Path, ffmpeg_path: str = "") -> dict[str, Any]:
    """EBU R128 loudness via ffmpeg. Best-effort; returns ``available=False``
    if ffmpeg lacks the filter."""
    if not path.is_file():
        return {"available": False, "error": "File missing."}
    exe = _find_ffmpeg(ffmpeg_path)
    if exe is None:
        return {"available": False, "error": "ffmpeg not available."}
    try:
        proc = subprocess.run(
            [str(exe), "-hide_banner", "-i", str(path), "-af", "loudnorm=print_format=json", "-f", "null", "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "error": str(exc)}
    stderr = proc.stderr or ""
    marker = stderr.rfind("{")
    if marker < 0:
        return {"available": False, "error": "loudnorm output not found."}
    end = stderr.find("}", marker)
    if end < 0:
        return {"available": False, "error": "loudnorm JSON truncated."}
    try:
        parsed = json.loads(stderr[marker : end + 1])
    except json.JSONDecodeError as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "input_i": float(parsed.get("input_i", 0.0)),
        "input_tp": float(parsed.get("input_tp", 0.0)),
        "input_lra": float(parsed.get("input_lra", 0.0)),
        "input_thresh": float(parsed.get("input_thresh", 0.0)),
    }


def detect_clipping(path: Path, ffmpeg_path: str = "", *, threshold_db: float = -0.5) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "error": "File missing."}
    exe = _find_ffmpeg(ffmpeg_path)
    if exe is None:
        return {"available": False, "error": "ffmpeg not available."}
    try:
        proc = subprocess.run(
            [str(exe), "-hide_banner", "-i", str(path), "-af", "astats=metadata=1:reset=0,ametadata=print:key=lavfi.astats.Overall.Peak_level", "-f", "null", "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "error": str(exc)}
    text = proc.stderr or ""
    # crude parse: look for Peak_level max
    import re

    peaks = [float(m) for m in re.findall(r"Peak_level_max:\s*(-?\d+\.?\d*)", text)]
    peak = max(peaks) if peaks else None
    clipped = peak is not None and peak >= threshold_db
    return {"available": peak is not None, "peak_db": peak, "clipping_detected": bool(clipped)}


def detect_silence(path: Path, ffmpeg_path: str = "", *, noise_db: float = -50.0, min_duration_s: float = 0.3) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "error": "File missing."}
    exe = _find_ffmpeg(ffmpeg_path)
    if exe is None:
        return {"available": False, "error": "ffmpeg not available."}
    try:
        proc = subprocess.run(
            [str(exe), "-hide_banner", "-i", str(path), "-af", f"silencedetect=noise={noise_db}dB:d={min_duration_s}", "-f", "null", "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "error": str(exc)}
    import re

    starts = [float(m) for m in re.findall(r"silence_start:\s*(-?\d+\.?\d*)", proc.stderr or "")]
    ends = [float(m) for m in re.findall(r"silence_end:\s*(-?\d+\.?\d*)", proc.stderr or "")]
    intervals = list(zip(starts, ends))
    total_ms = int(sum((e - s) for s, e in intervals) * 1000)
    return {
        "available": True,
        "silence_intervals": len(intervals),
        "total_silence_ms": total_ms,
    }


def assert_project_invariants(
    project: DubbingProject,
    *,
    active_sequences: Iterable[int] = (),
) -> dict[str, Any]:
    """Check the full set of project invariants. See spec section 20."""
    violations: list[InvariantViolation] = []
    tl = validate_timeline(project)
    for raw in tl["violations"]:
        violations.append(
            InvariantViolation(
                code=str(raw.get("code", "timeline")),
                message=json.dumps(raw),
                sequence=raw.get("sequence"),
            )
        )
    active = set(active_sequences)
    for cue in project.cues:
        if cue.status == CueStatus.RENDERING.value and cue.sequence not in active:
            violations.append(
                InvariantViolation(
                    code="cue_stuck_rendering",
                    message="Cue is rendering but no active job exists.",
                    sequence=cue.sequence,
                )
            )
        if cue.enabled and cue.status in READY_STATUSES:
            if not cue.raw_audio_path or not Path(cue.raw_audio_path).is_file():
                violations.append(
                    InvariantViolation(
                        code="missing_raw_wav",
                        message="Ready cue has no raw WAV.",
                        sequence=cue.sequence,
                    )
                )
            if not cue.fitted_audio_path or not Path(cue.fitted_audio_path).is_file():
                violations.append(
                    InvariantViolation(
                        code="missing_fitted_wav",
                        message="Ready cue has no fitted WAV.",
                        sequence=cue.sequence,
                    )
                )
    # Orphan .part files
    cues_dir = project.cues_dir()
    if cues_dir.is_dir():
        for part in cues_dir.glob("*.part"):
            violations.append(
                InvariantViolation(code="orphan_part_file", message=str(part.name))
            )
    # JSONL validity for the latest run
    violations.extend(_check_latest_jsonl(project))
    if project.final_video_path and not project.stale.video:
        if not Path(project.final_video_path).is_file():
            violations.append(
                InvariantViolation(
                    code="final_video_missing_but_clean",
                    message="stale.video is False but the final video file is gone.",
                )
            )
    if not project.stale.narration:
        if any(c.is_stale for c in project.cues if c.enabled):
            violations.append(
                InvariantViolation(
                    code="narration_clean_but_stale_cues",
                    message="narration marked fresh while cues are stale.",
                )
            )
    return {
        "valid": not violations,
        "violations": [
            {
                "code": v.code,
                "sequence": v.sequence,
                "message": v.message,
            }
            for v in violations
        ],
    }


def _check_latest_jsonl(project: DubbingProject) -> list[InvariantViolation]:
    logs_dir = project.project_dir / "logs"
    if not logs_dir.is_dir():
        return []
    runs = sorted(logs_dir.glob("run_*/events.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return []
    path = runs[0]
    out: list[InvariantViolation] = []
    try:
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            try:
                json.loads(line)
            except json.JSONDecodeError:
                out.append(
                    InvariantViolation(
                        code="jsonl_invalid_line",
                        message=f"{path.name}:{lineno} is not valid JSON.",
                    )
                )
                break
    except OSError:
        pass
    return out


def _find_ffmpeg(configured: str):
    try:
        return ffmpeg_utils.find_ffmpeg(configured)
    except Exception:
        return None


__all__ = [
    "QualityWeights",
    "score_variant",
    "validate_wav",
    "validate_timeline",
    "validate_final_video",
    "measure_loudness",
    "detect_clipping",
    "detect_silence",
    "assert_project_invariants",
]
