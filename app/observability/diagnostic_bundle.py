"""Per-run diagnostic directory layout.

For each generation / heavy operation the service opens a run directory::

    <project>/logs/run_<run_id>/
        events.jsonl
        summary.json
        subprocess/<op>.stderr.log
        subprocess/<op>.stdout.log

The :class:`RunDirectory` owns the :class:`JsonlEventSink` and
:class:`SummaryWriter`. Code binds it as the active event sink via
``with run_dir.bind():`` so nested pipeline stages emit there automatically.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .events import JsonlEventSink, SummaryWriter, use_event_sink


@dataclass
class RunDirectory:
    """A run-scoped diagnostic directory and its writers."""

    root: Path
    run_id: str
    sink: JsonlEventSink
    summary: SummaryWriter

    @classmethod
    def create(cls, project_dir: Path, run_id: str) -> "RunDirectory":
        root = Path(project_dir) / "logs" / f"run_{run_id}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "subprocess").mkdir(parents=True, exist_ok=True)
        sink = JsonlEventSink(root / "events.jsonl")
        summary = SummaryWriter(root / "summary.json")
        summary.update(
            run_id=run_id,
            project_dir=str(project_dir),
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        summary.write()
        return cls(root=root, run_id=run_id, sink=sink, summary=summary)

    @property
    def subprocess_dir(self) -> Path:
        return self.root / "subprocess"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.root / "summary.json"

    @contextmanager
    def bind(self) -> Iterator[None]:
        with use_event_sink(self.sink):
            try:
                yield
            finally:
                try:
                    self.sink.flush()
                except Exception:  # pragma: no cover
                    pass

    def finalize(self, **fields: object) -> None:
        try:
            self.summary.update(
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **fields,
            )
            self.summary.write()
            self.sink.flush()
        except Exception:  # pragma: no cover
            pass

    def close(self) -> None:
        try:
            self.sink.close()
        except Exception:  # pragma: no cover
            pass

    def read_summary(self) -> dict:
        try:
            return json.loads(self.summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def read_events(self) -> list[dict]:
        events: list[dict] = []
        if not self.events_path.is_file():
            return events
        import json as _json

        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(_json.loads(line))
            except json.JSONDecodeError:
                # A truncated final line (process killed mid-write) is tolerated.
                continue
        return events


__all__ = ["RunDirectory"]
