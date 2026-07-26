from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class GenerationRunStatus(str, Enum):
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CueErrorPolicy(str, Enum):
    CONTINUE = "continue"
    STOP = "stop"


class GenerationCancelled(Exception):
    """Raised when the user cancels an in-flight generation batch."""


@dataclass(frozen=True)
class GenerationRunResult:
    status: GenerationRunStatus
    total_count: int
    completed_count: int
    failed_count: int
    cancelled_count: int
    skipped_count: int
    failed_cue_ids: tuple[str, ...] = ()
    last_processed_cue_id: str | None = None
    error_message: str | None = None
    run_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status == GenerationRunStatus.COMPLETED

    def summary_counts(self) -> dict[str, int]:
        return {
            "total": self.total_count,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "cancelled": self.cancelled_count,
            "skipped": self.skipped_count,
        }


@dataclass(frozen=True)
class GenerationContext:
    """Immutable snapshot of settings for one generation run."""

    run_id: str
    engine_id: str
    model_id: str | None
    voice_id: str | None
    voice_display_name: str | None
    voice_config: dict[str, Any]
    reference_voice_path: str | None
    reference_voice_content_hash: str | None
    language: str
    speed: float
    compress_internal_pauses: bool
    internal_pause_keep_ms: int
    sample_rate: int
    channels: int
    preferred_speed_limit: float
    hard_speed_limit: float
    guard_gap_ms: int
    sync_mode: str
    ffmpeg_path: str
    raw_generation_fingerprint_seed: str
    fit_settings_fingerprint: str
    created_at_utc: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @staticmethod
    def new_run_id() -> str:
        return uuid4().hex
