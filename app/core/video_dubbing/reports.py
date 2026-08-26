from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import CueStatus, DubbingProject


@dataclass
class CueReportRow:
    sequence: int
    start_ms: int
    end_ms: int
    duration_budget_ms: int
    text: str
    raw_duration_ms: int | None
    fitted_duration_ms: int | None
    required_speed_factor: float | None
    applied_speed_factor: float
    fitting_strategy: str | None
    overflow_ms: int
    status: str
    warnings: list[str]
    error: str | None


@dataclass
class ProjectReportSummary:
    total_cues: int = 0
    rendered: int = 0
    speed_up: int = 0
    needs_shortening: int = 0
    failed: int = 0
    overlaps: int = 0
    disabled: int = 0
    max_speed_factor: float = 0.0
    average_speed_factor: float = 0.0
    video_duration_ms: int = 0
    narration_duration_ms: int = 0
    mix_duration_ms: int = 0
    ducking_mode: str = ""
    final_container: str = ""
    audio_tracks: list[str] = field(default_factory=list)


@dataclass
class TimingReport:
    cues: list[CueReportRow]
    summary: ProjectReportSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "cues": [asdict(row) for row in self.cues],
            "summary": asdict(self.summary),
        }


def build_report(project: DubbingProject, audio_tracks: list[str] | None = None) -> TimingReport:
    cues_rows: list[CueReportRow] = []
    speed_factors: list[float] = []
    overlaps = sum(
        1 for cue in project.cues if "overlap" in (cue.warning_codes or [])
    )
    for cue in project.cues:
        if cue.applied_speed_factor and cue.applied_speed_factor > 1.001:
            speed_factors.append(cue.applied_speed_factor)
        cues_rows.append(
            CueReportRow(
                sequence=cue.sequence,
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                duration_budget_ms=cue.duration_budget_ms,
                text=cue.spoken_text,
                raw_duration_ms=cue.raw_duration_ms,
                fitted_duration_ms=cue.fitted_duration_ms,
                required_speed_factor=cue.required_speed_factor,
                applied_speed_factor=cue.applied_speed_factor,
                fitting_strategy=cue.fitting_strategy,
                overflow_ms=cue.overflow_ms,
                status=cue.status,
                warnings=list(cue.warning_codes),
                error=cue.error_message,
            )
        )
    summary = ProjectReportSummary(
        total_cues=len(project.cues),
        rendered=sum(1 for c in project.cues if c.status == CueStatus.RENDERED.value),
        speed_up=sum(1 for c in project.cues if c.status == CueStatus.SPEED_UP.value),
        needs_shortening=sum(
            1 for c in project.cues if c.status == CueStatus.NEEDS_SHORTENING.value
        ),
        failed=sum(1 for c in project.cues if c.status == CueStatus.FAILED.value),
        overlaps=overlaps,
        disabled=sum(1 for c in project.cues if not c.enabled),
        max_speed_factor=max(speed_factors) if speed_factors else 0.0,
        average_speed_factor=(
            sum(speed_factors) / len(speed_factors) if speed_factors else 0.0
        ),
        video_duration_ms=project.duration_ms,
        narration_duration_ms=_track_duration_ms(project.narration_wav),
        mix_duration_ms=_track_duration_ms(project.dubbed_mix_wav),
        ducking_mode=project.settings.ducking.mode.value,
        final_container=project.settings.export.container.value,
        audio_tracks=list(audio_tracks or []),
    )
    return TimingReport(cues=cues_rows, summary=summary)


def write_reports(
    project: DubbingProject,
    report: TimingReport,
) -> tuple[Path, Path]:
    project.reports_dir().mkdir(parents=True, exist_ok=True)
    json_path = project.reports_dir() / "timing_report.json"
    csv_path = project.reports_dir() / "timing_report.csv"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    field_names = [
        "sequence",
        "start_ms",
        "end_ms",
        "duration_budget_ms",
        "text",
        "raw_duration_ms",
        "fitted_duration_ms",
        "required_speed_factor",
        "applied_speed_factor",
        "fitting_strategy",
        "overflow_ms",
        "status",
        "warnings",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in report.cues:
            writer.writerow(
                {
                    "sequence": row.sequence,
                    "start_ms": row.start_ms,
                    "end_ms": row.end_ms,
                    "duration_budget_ms": row.duration_budget_ms,
                    "text": row.text,
                    "raw_duration_ms": row.raw_duration_ms,
                    "fitted_duration_ms": row.fitted_duration_ms,
                    "required_speed_factor": row.required_speed_factor,
                    "applied_speed_factor": row.applied_speed_factor,
                    "fitting_strategy": row.fitting_strategy,
                    "overflow_ms": row.overflow_ms,
                    "status": row.status,
                    "warnings": "|".join(row.warnings),
                    "error": row.error or "",
                }
            )
    return json_path, csv_path


def _track_duration_ms(path: Path | None) -> int:
    if path is None or not Path(path).is_file():
        return 0
    try:
        import wave

        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            return int(round(audio.getnframes() / rate * 1000)) if rate else 0
    except (OSError, Exception):
        return 0
