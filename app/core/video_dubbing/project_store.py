from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.utils.paths import app_data_root

from .models import (
    CueStatus,
    DubbingCue,
    DubbingProject,
    DubbingProjectSettings,
    StaleFlags,
    VideoProbeInfo,
)


CURRENT_DB_SCHEMA_VERSION = 1
DUBBING_MANIFEST_NAME = "dubbing_project.json"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _path_to_str(value: Path | None) -> str:
    if value is None:
        return ""
    return str(value)


def _str_to_path(value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text)


class DubbingProjectStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.root_dir = app_data_root() / "video_dubbing"
        self.db_path = db_path or self.root_dir / "video_dubbing.sqlite3"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dubbing_projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    video_path TEXT DEFAULT '',
                    video_probe_json TEXT DEFAULT '{}',
                    srt_source_path TEXT DEFAULT '',
                    srt_source_text TEXT DEFAULT '',
                    settings_json TEXT DEFAULT '{}',
                    stale_json TEXT DEFAULT '{}',
                    project_dir TEXT NOT NULL,
                    narration_wav TEXT DEFAULT '',
                    narration_mp3 TEXT DEFAULT '',
                    dubbed_mix_wav TEXT DEFAULT '',
                    final_video_path TEXT DEFAULT '',
                    full_preview_path TEXT DEFAULT '',
                    cue_preview_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dubbing_cues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL REFERENCES dubbing_projects(id)
                        ON DELETE CASCADE,
                    cue_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    start_ms INTEGER NOT NULL,
                    end_ms INTEGER NOT NULL,
                    duration_budget_ms INTEGER NOT NULL,
                    source_text TEXT NOT NULL,
                    spoken_text TEXT NOT NULL,
                    raw_audio_path TEXT DEFAULT '',
                    fitted_audio_path TEXT DEFAULT '',
                    raw_duration_ms INTEGER,
                    fitted_duration_ms INTEGER,
                    required_speed_factor REAL,
                    applied_speed_factor REAL DEFAULT 1.0,
                    placement_offset_ms INTEGER DEFAULT 0,
                    overflow_ms INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    fitting_strategy TEXT,
                    warning_codes_json TEXT DEFAULT '[]',
                    error_message TEXT DEFAULT '',
                    enabled INTEGER DEFAULT 1,
                    is_stale INTEGER DEFAULT 0,
                    attempt_count INTEGER DEFAULT 0,
                    native_speed_factor REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (project_id, cue_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dubbing_cues_project
                ON dubbing_cues(project_id, sequence)
                """
            )

    def create_project(
        self,
        project_dir: Path | None,
        title: str = "Video Dubbing",
    ) -> str:
        project_uuid = str(uuid.uuid4())
        resolved_dir = project_dir or (self.root_dir / project_uuid)
        resolved_dir.mkdir(parents=True, exist_ok=True)
        now = _now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO dubbing_projects (
                    uuid, title, project_dir, settings_json, stale_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_uuid,
                    title,
                    str(resolved_dir),
                    "{}",
                    "{}",
                    now,
                    now,
                ),
            )
            _ = cursor
        return project_uuid

    def save_project(self, project: DubbingProject) -> None:
        now = _now_iso()
        project.updated_at = now
        if not project.created_at:
            project.created_at = now
        project.ensure_directories()
        settings_json = json.dumps(
            project.settings.to_dict(), ensure_ascii=False
        )
        stale_json = json.dumps(project.stale.to_dict(), ensure_ascii=False)
        video_probe_json = (
            project.video_probe.to_dict() if project.video_probe else {}
        )
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM dubbing_projects WHERE uuid = ?",
                (project.project_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO dubbing_projects (
                        uuid, title, video_path, video_probe_json,
                        srt_source_path, srt_source_text, settings_json,
                        stale_json, project_dir, narration_wav, narration_mp3,
                        dubbed_mix_wav, final_video_path, full_preview_path,
                        cue_preview_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project.project_id,
                        project.title,
                        _path_to_str(project.video_path),
                        json.dumps(video_probe_json, ensure_ascii=False),
                        _path_to_str(project.srt_source_path),
                        project.srt_source_text,
                        settings_json,
                        stale_json,
                        str(project.project_dir),
                        _path_to_str(project.narration_wav),
                        _path_to_str(project.narration_mp3),
                        _path_to_str(project.dubbed_mix_wav),
                        _path_to_str(project.final_video_path),
                        _path_to_str(project.full_preview_path),
                        json.dumps(
                            {
                                str(k): _path_to_str(v)
                                for k, v in project.cue_preview_cache.items()
                            },
                            ensure_ascii=False,
                        ),
                        project.created_at or now,
                        now,
                    ),
                )
                project_db_id = connection.execute(
                    "SELECT id FROM dubbing_projects WHERE uuid = ?",
                    (project.project_id,),
                ).fetchone()["id"]
            else:
                project_db_id = existing["id"]
                connection.execute(
                    """
                    UPDATE dubbing_projects
                    SET title = ?, video_path = ?, video_probe_json = ?,
                        srt_source_path = ?, srt_source_text = ?,
                        settings_json = ?, stale_json = ?, project_dir = ?,
                        narration_wav = ?, narration_mp3 = ?,
                        dubbed_mix_wav = ?, final_video_path = ?,
                        full_preview_path = ?, cue_preview_json = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        project.title,
                        _path_to_str(project.video_path),
                        json.dumps(video_probe_json, ensure_ascii=False),
                        _path_to_str(project.srt_source_path),
                        project.srt_source_text,
                        settings_json,
                        stale_json,
                        str(project.project_dir),
                        _path_to_str(project.narration_wav),
                        _path_to_str(project.narration_mp3),
                        _path_to_str(project.dubbed_mix_wav),
                        _path_to_str(project.final_video_path),
                        _path_to_str(project.full_preview_path),
                        json.dumps(
                            {
                                str(k): _path_to_str(v)
                                for k, v in project.cue_preview_cache.items()
                            },
                            ensure_ascii=False,
                        ),
                        now,
                        project_db_id,
                    ),
                )
            connection.execute(
                "DELETE FROM dubbing_cues WHERE project_id = ?",
                (project_db_id,),
            )
            for cue in project.cues:
                self._insert_cue(connection, project_db_id, cue, now)
        self._write_manifest(project)

    def _insert_cue(
        self,
        connection: sqlite3.Connection,
        project_db_id: int,
        cue: DubbingCue,
        now: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO dubbing_cues (
                project_id, cue_id, sequence, start_ms, end_ms,
                duration_budget_ms, source_text, spoken_text,
                raw_audio_path, fitted_audio_path, raw_duration_ms,
                fitted_duration_ms, required_speed_factor,
                applied_speed_factor, placement_offset_ms, overflow_ms,
                status, fitting_strategy, warning_codes_json,
                error_message, enabled, is_stale, attempt_count,
                native_speed_factor, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_db_id,
                cue.cue_id,
                cue.sequence,
                cue.start_ms,
                cue.end_ms,
                cue.duration_budget_ms,
                cue.source_text,
                cue.spoken_text,
                _path_to_str(cue.raw_audio_path),
                _path_to_str(cue.fitted_audio_path),
                cue.raw_duration_ms,
                cue.fitted_duration_ms,
                cue.required_speed_factor,
                cue.applied_speed_factor,
                cue.placement_offset_ms,
                cue.overflow_ms,
                cue.status,
                cue.fitting_strategy,
                json.dumps(cue.warning_codes, ensure_ascii=False),
                cue.error_message or "",
                1 if cue.enabled else 0,
                1 if cue.is_stale else 0,
                cue.attempt_count,
                cue.native_speed_factor,
                now,
                now,
            ),
        )

    def load_project(self, project_id: str) -> DubbingProject | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT uuid, title, video_path, video_probe_json,
                       srt_source_path, srt_source_text, settings_json,
                       stale_json, project_dir, narration_wav, narration_mp3,
                       dubbed_mix_wav, final_video_path, full_preview_path,
                       cue_preview_json, created_at, updated_at
                FROM dubbing_projects
                WHERE uuid = ?
                """,
                (project_id,),
            ).fetchone()
            if row is None:
                return None
            cue_rows = connection.execute(
                """
                SELECT cue_id, sequence, start_ms, end_ms, duration_budget_ms,
                       source_text, spoken_text, raw_audio_path,
                       fitted_audio_path, raw_duration_ms, fitted_duration_ms,
                       required_speed_factor, applied_speed_factor,
                       placement_offset_ms, overflow_ms, status,
                       fitting_strategy, warning_codes_json, error_message,
                       enabled, is_stale, attempt_count, native_speed_factor
                FROM dubbing_cues
                WHERE project_id = (SELECT id FROM dubbing_projects WHERE uuid = ?)
                ORDER BY sequence
                """,
                (project_id,),
            ).fetchall()
        return self._row_to_project(row, cue_rows)

    def list_projects(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT uuid, title, video_path, updated_at
                FROM dubbing_projects
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [
            {
                "project_id": str(row["uuid"]),
                "title": str(row["title"]),
                "video_path": str(row["video_path"] or ""),
                "updated_at": str(row["updated_at"] or ""),
            }
            for row in rows
        ]

    def delete_project(self, project_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM dubbing_projects WHERE uuid = ?",
                (project_id,),
            )

    def _row_to_project(
        self,
        row: sqlite3.Row,
        cue_rows: list[sqlite3.Row],
    ) -> DubbingProject:
        settings_data = json.loads(row["settings_json"] or "{}")
        video_probe_data = json.loads(row["video_probe_json"] or "{}")
        stale_data = json.loads(row["stale_json"] or "{}")
        cue_preview_data = json.loads(row["cue_preview_json"] or "{}")
        project = DubbingProject(
            project_id=str(row["uuid"]),
            project_dir=Path(str(row["project_dir"])),
            title=str(row["title"] or "Video Dubbing"),
            video_path=_str_to_path(str(row["video_path"] or "")),
            video_probe=(
                VideoProbeInfo.from_dict(video_probe_data)
                if video_probe_data
                else None
            ),
            srt_source_path=_str_to_path(str(row["srt_source_path"] or "")),
            srt_source_text=str(row["srt_source_text"] or ""),
            settings=DubbingProjectSettings.from_dict(
                settings_data if isinstance(settings_data, dict) else {}
            ),
            stale=StaleFlags.from_dict(stale_data if isinstance(stale_data, dict) else {}),
            narration_wav=_str_to_path(str(row["narration_wav"] or "")),
            narration_mp3=_str_to_path(str(row["narration_mp3"] or "")),
            dubbed_mix_wav=_str_to_path(str(row["dubbed_mix_wav"] or "")),
            final_video_path=_str_to_path(str(row["final_video_path"] or "")),
            full_preview_path=_str_to_path(str(row["full_preview_path"] or "")),
            cue_preview_cache={
                str(key): Path(str(value))
                for key, value in (cue_preview_data.items() if isinstance(cue_preview_data, dict) else {})
                if value
            },
            created_at=str(row["created_at"] or ""),
            updated_at=str(row["updated_at"] or ""),
        )
        for cue_row in cue_rows:
            project.cues.append(self._row_to_cue(cue_row))
        return project

    def _row_to_cue(self, row: sqlite3.Row) -> DubbingCue:
        try:
            warnings = json.loads(row["warning_codes_json"] or "[]")
        except json.JSONDecodeError:
            warnings = []
        if not isinstance(warnings, list):
            warnings = []
        return DubbingCue(
            cue_id=str(row["cue_id"]),
            sequence=int(row["sequence"]),
            start_ms=int(row["start_ms"]),
            end_ms=int(row["end_ms"]),
            duration_budget_ms=int(row["duration_budget_ms"]),
            source_text=str(row["source_text"]),
            spoken_text=str(row["spoken_text"]),
            raw_audio_path=_str_to_path(str(row["raw_audio_path"] or "")),
            fitted_audio_path=_str_to_path(str(row["fitted_audio_path"] or "")),
            raw_duration_ms=(
                int(row["raw_duration_ms"])
                if row["raw_duration_ms"] is not None
                else None
            ),
            fitted_duration_ms=(
                int(row["fitted_duration_ms"])
                if row["fitted_duration_ms"] is not None
                else None
            ),
            required_speed_factor=(
                float(row["required_speed_factor"])
                if row["required_speed_factor"] is not None
                else None
            ),
            applied_speed_factor=float(row["applied_speed_factor"] or 1.0),
            placement_offset_ms=int(row["placement_offset_ms"] or 0),
            overflow_ms=int(row["overflow_ms"] or 0),
            status=str(row["status"] or CueStatus.PENDING.value),
            fitting_strategy=(
                str(row["fitting_strategy"])
                if row["fitting_strategy"] is not None
                else None
            ),
            warning_codes=[str(item) for item in warnings],
            error_message=str(row["error_message"] or "") or None,
            enabled=bool(row["enabled"]),
            is_stale=bool(row["is_stale"]),
            attempt_count=int(row["attempt_count"] or 0),
            native_speed_factor=(
                float(row["native_speed_factor"])
                if row["native_speed_factor"] is not None
                else None
            ),
        )

    def _write_manifest(self, project: DubbingProject) -> None:
        project.ensure_directories()
        manifest_path = project.project_dir / DUBBING_MANIFEST_NAME
        data: dict[str, Any] = {
            "manifest_version": CURRENT_DB_SCHEMA_VERSION,
            "project_id": project.project_id,
            "title": project.title,
            "video_path": _path_to_str(project.video_path),
            "video_probe": (
                project.video_probe.to_dict() if project.video_probe else None
            ),
            "srt_source_path": _path_to_str(project.srt_source_path),
            "settings": project.settings.to_dict(),
            "stale": project.stale.to_dict(),
            "project_dir": str(project.project_dir),
            "narration_wav": _path_to_str(project.narration_wav),
            "narration_mp3": _path_to_str(project.narration_mp3),
            "dubbed_mix_wav": _path_to_str(project.dubbed_mix_wav),
            "final_video_path": _path_to_str(project.final_video_path),
            "full_preview_path": _path_to_str(project.full_preview_path),
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "cues": [self._cue_to_manifest(cue) for cue in project.cues],
        }
        manifest_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _cue_to_manifest(self, cue: DubbingCue) -> dict[str, Any]:
        return {
            "cue_id": cue.cue_id,
            "sequence": cue.sequence,
            "start_ms": cue.start_ms,
            "end_ms": cue.end_ms,
            "duration_budget_ms": cue.duration_budget_ms,
            "source_text": cue.source_text,
            "spoken_text": cue.spoken_text,
            "raw_audio_path": _path_to_str(cue.raw_audio_path),
            "fitted_audio_path": _path_to_str(cue.fitted_audio_path),
            "raw_duration_ms": cue.raw_duration_ms,
            "fitted_duration_ms": cue.fitted_duration_ms,
            "required_speed_factor": cue.required_speed_factor,
            "applied_speed_factor": cue.applied_speed_factor,
            "placement_offset_ms": cue.placement_offset_ms,
            "overflow_ms": cue.overflow_ms,
            "status": cue.status,
            "fitting_strategy": cue.fitting_strategy,
            "warning_codes": list(cue.warning_codes),
            "error_message": cue.error_message,
            "enabled": cue.enabled,
            "is_stale": cue.is_stale,
            "attempt_count": cue.attempt_count,
            "native_speed_factor": cue.native_speed_factor,
        }

    def load_project_from_manifest(
        self,
        manifest_path: Path,
    ) -> DubbingProject:
        data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("Invalid video dubbing project manifest.")
        project_id = str(data.get("project_id") or uuid.uuid4())
        project_dir = manifest_path.parent
        project = DubbingProject(
            project_id=project_id,
            project_dir=project_dir,
            title=str(data.get("title") or "Video Dubbing"),
            video_path=_str_to_path(str(data.get("video_path") or "")),
            video_probe=(
                VideoProbeInfo.from_dict(data["video_probe"])
                if isinstance(data.get("video_probe"), dict)
                else None
            ),
            srt_source_path=_str_to_path(str(data.get("srt_source_path") or "")),
            settings=DubbingProjectSettings.from_dict(
                data.get("settings", {}) if isinstance(data.get("settings"), dict) else {}
            ),
            stale=StaleFlags.from_dict(
                data.get("stale", {}) if isinstance(data.get("stale"), dict) else {}
            ),
            narration_wav=_str_to_path(str(data.get("narration_wav") or "")),
            narration_mp3=_str_to_path(str(data.get("narration_mp3") or "")),
            dubbed_mix_wav=_str_to_path(str(data.get("dubbed_mix_wav") or "")),
            final_video_path=_str_to_path(str(data.get("final_video_path") or "")),
            full_preview_path=_str_to_path(str(data.get("full_preview_path") or "")),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )
        cues_data = data.get("cues", [])
        if isinstance(cues_data, list):
            for cue_data in cues_data:
                if not isinstance(cue_data, dict):
                    continue
                project.cues.append(self._manifest_to_cue(cue_data))
        # Persist the imported manifest into SQLite so it is tracked.
        self.save_project(project)
        return project

    def _manifest_to_cue(self, data: dict[str, Any]) -> DubbingCue:
        return DubbingCue(
            cue_id=str(data.get("cue_id") or data.get("sequence") or ""),
            sequence=int(data.get("sequence") or 0),
            start_ms=int(data.get("start_ms") or 0),
            end_ms=int(data.get("end_ms") or 0),
            duration_budget_ms=int(data.get("duration_budget_ms") or 0),
            source_text=str(data.get("source_text") or ""),
            spoken_text=str(data.get("spoken_text") or ""),
            raw_audio_path=_str_to_path(str(data.get("raw_audio_path") or "")),
            fitted_audio_path=_str_to_path(str(data.get("fitted_audio_path") or "")),
            raw_duration_ms=(
                int(data.get("raw_duration_ms"))
                if data.get("raw_duration_ms") is not None
                else None
            ),
            fitted_duration_ms=(
                int(data.get("fitted_duration_ms"))
                if data.get("fitted_duration_ms") is not None
                else None
            ),
            required_speed_factor=(
                float(data.get("required_speed_factor"))
                if data.get("required_speed_factor") is not None
                else None
            ),
            applied_speed_factor=float(data.get("applied_speed_factor") or 1.0),
            placement_offset_ms=int(data.get("placement_offset_ms") or 0),
            overflow_ms=int(data.get("overflow_ms") or 0),
            status=str(data.get("status") or CueStatus.PENDING.value),
            fitting_strategy=(
                str(data.get("fitting_strategy"))
                if data.get("fitting_strategy") is not None
                else None
            ),
            warning_codes=[
                str(item)
                for item in (data.get("warning_codes") or [])
                if isinstance(item, (str, int, float))
            ],
            error_message=(
                str(data.get("error_message") or "") or None
            ),
            enabled=bool(data.get("enabled", True)),
            is_stale=bool(data.get("is_stale", False)),
            attempt_count=int(data.get("attempt_count") or 0),
            native_speed_factor=(
                float(data.get("native_speed_factor"))
                if data.get("native_speed_factor") is not None
                else None
            ),
        )
