"""Monotonic revision store + metadata snapshots.

The revision is the concurrency-control token for mutating operations. It lives
in a sidecar JSON file per project (``<project_dir>/snapshots/revision.json``)
so the domain layer (``DubbingProject``) stays untouched and existing
uncommitted work is not disturbed.

Snapshots are *metadata* snapshots: a copy of the project manifest's cue
metadata + settings + stale flags + revision at a point in time. They never
copy heavy WAV/video files; artifacts stay content-addressed on disk and the
snapshot references their relative paths.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.video_dubbing.models import DubbingProject
from app.core.video_dubbing.project_store import DubbingProjectStore, DUBBING_MANIFEST_NAME

REVISION_FILE = "revision.json"
SNAPSHOT_PREFIX = "snap_"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _snapshots_dir(project: DubbingProject) -> Path:
    directory = project.project_dir / "snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _manifest_dict(store: DubbingProjectStore, project: DubbingProject) -> dict[str, Any]:
    return store._build_manifest_dict(project)


class RevisionStore:
    """Monotonic, per-project revision counter persisted to JSON."""

    def __init__(self) -> None:
        self._cache: dict[str, int] = {}
        self._lock = threading.RLock()

    def current(self, project: DubbingProject) -> int:
        path = project.project_dir / "snapshots" / REVISION_FILE
        with self._lock:
            if not path.is_file():
                return 0
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return int(data.get("revision", 0))
            except (json.JSONDecodeError, ValueError, OSError):
                return 0

    def bump(self, project: DubbingProject) -> int:
        with self._lock:
            current = self.current(project)
            nxt = current + 1
            directory = project.project_dir / "snapshots"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / REVISION_FILE
            payload = {"project_id": project.project_id, "revision": nxt, "updated_at": _utc()}
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            try:
                os.replace(tmp, path)
            except OSError:
                path.replace(tmp)
            self._cache[project.project_id] = nxt
            return nxt


class SnapshotStore:
    """Metadata-only snapshots stored as JSON files."""

    def __init__(self, store: DubbingProjectStore) -> None:
        self.store = store
        self._lock = threading.RLock()

    def create(
        self,
        project: DubbingProject,
        *,
        reason: str = "mutation",
        initiator: str = "mcp",
        revision: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            directory = _snapshots_dir(project)
            snapshot_id = uuid.uuid4().hex[:12]
            manifest = _manifest_dict(self.store, project)
            # Keep only metadata fields; drop heavy inline text where huge.
            payload = {
                "snapshot_id": snapshot_id,
                "project_id": project.project_id,
                "reason": reason,
                "initiator": initiator,
                "created_at": _utc(),
                "revision": revision if revision is not None else RevisionStore().current(project),
                "title": manifest.get("title"),
                "settings": manifest.get("settings"),
                "stale": manifest.get("stale"),
                "video_path": manifest.get("video_path"),
                "narration_wav": manifest.get("narration_wav"),
                "dubbed_mix_wav": manifest.get("dubbed_mix_wav"),
                "final_video_path": manifest.get("final_video_path"),
                "cues": manifest.get("cues"),
            }
            path = directory / f"{SNAPSHOT_PREFIX}{snapshot_id}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return {
                "snapshot_id": snapshot_id,
                "reason": reason,
                "initiator": initiator,
                "created_at": payload["created_at"],
                "revision": payload["revision"],
                "cue_count": len(payload.get("cues") or []),
            }

    def list(self, project: DubbingProject) -> list[dict[str, Any]]:
        directory = _snapshots_dir(project)
        if not directory.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for path in sorted(directory.glob(f"{SNAPSHOT_PREFIX}*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            items.append(
                {
                    "snapshot_id": data.get("snapshot_id"),
                    "reason": data.get("reason"),
                    "initiator": data.get("initiator"),
                    "created_at": data.get("created_at"),
                    "revision": data.get("revision"),
                    "cue_count": len(data.get("cues") or []),
                }
            )
        return items

    def get(self, project: DubbingProject, snapshot_id: str) -> dict[str, Any] | None:
        path = _snapshots_dir(project) / f"{SNAPSHOT_PREFIX}{snapshot_id}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def compare(
        self, project: DubbingProject, snapshot_id: str, current_revision: int
    ) -> dict[str, Any]:
        snap = self.get(project, snapshot_id)
        if snap is None:
            return {"found": False}
        snap_cues = {c["sequence"]: c for c in (snap.get("cues") or [])}
        cur_cues = {c.sequence: c for c in project.cues}
        changed: list[int] = []
        for seq, snap_cue in snap_cues.items():
            cur_cue = cur_cues.get(seq)
            if cur_cue is None:
                changed.append(seq)
                continue
            for field in ("start_ms", "end_ms", "spoken_text", "status", "enabled"):
                if getattr(cur_cue, field) != snap_cue.get(field):
                    changed.append(seq)
                    break
        for seq in cur_cues:
            if seq not in snap_cues:
                changed.append(seq)
        return {
            "found": True,
            "snapshot_revision": snap.get("revision"),
            "current_revision": current_revision,
            "changed_cues": sorted(set(changed)),
            "snapshot_cue_count": len(snap_cues),
            "current_cue_count": len(cur_cues),
        }

    def latest_manifest_path(self, project: DubbingProject) -> Path:
        return project.project_dir / DUBBING_MANIFEST_NAME
