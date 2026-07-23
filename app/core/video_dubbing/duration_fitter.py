from __future__ import annotations

import math
from dataclasses import dataclass

from .models import (
    Alignment,
    CueStatus,
    DubbingCue,
    DubbingProjectSettings,
    FittingStrategy,
    SyncMode,
)


@dataclass
class FittingResult:
    strategy: FittingStrategy
    applied_speed_factor: float
    fitted_duration_ms: int
    overflow_ms: int
    native_speed_factor: float | None = None
    atempo_factor: float | None = None
    status: str = CueStatus.FITTED.value
    warning_codes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warning_codes is None:
            self.warning_codes = []


class DurationFitter:
    def __init__(self, settings: DubbingProjectSettings) -> None:
        self.settings = settings

    def evaluate(self, cue: DubbingCue) -> FittingResult:
        raw_ms = cue.raw_duration_ms
        budget_ms = cue.duration_budget_ms
        if raw_ms is None:
            raise ValueError(
                f"Cue #{cue.sequence} has no raw duration yet."
            )
        if raw_ms <= 0:
            raise ValueError(
                f"Cue #{cue.sequence} has non-positive raw duration {raw_ms}."
            )
        if budget_ms <= 0:
            raise ValueError(
                f"Cue #{cue.sequence} has non-positive budget {budget_ms}."
            )

        required_factor = raw_ms / budget_ms

        if required_factor <= 1.0:
            return FittingResult(
                strategy=FittingStrategy.NONE,
                applied_speed_factor=1.0,
                fitted_duration_ms=raw_ms,
                overflow_ms=0,
                status=CueStatus.RENDERED.value,
            )

        max_factor = self.settings.max_speed_factor
        if required_factor > max_factor:
            return self._handle_overflow(
                cue,
                required_factor=required_factor,
                raw_ms=raw_ms,
                budget_ms=budget_ms,
            )

        return FittingResult(
            strategy=FittingStrategy.ATEMPO,
            applied_speed_factor=required_factor,
            fitted_duration_ms=budget_ms,
            overflow_ms=0,
            native_speed_factor=required_factor,
            atempo_factor=required_factor,
            status=CueStatus.SPEED_UP.value,
        )

    def _handle_overflow(
        self,
        cue: DubbingCue,
        required_factor: float,
        raw_ms: int,
        budget_ms: int,
    ) -> FittingResult:
        max_factor = self.settings.max_speed_factor
        capped_factor = max_factor
        capped_duration_ms = max(
            1, int(math.ceil(raw_ms / capped_factor))
        )
        overflow_ms = max(0, capped_duration_ms - budget_ms)
        warnings = [
            "needs_text_shortening",
            f"required_factor_{required_factor:.2f}",
        ]
        if self.settings.sync_mode == SyncMode.STRICT:
            return FittingResult(
                strategy=FittingStrategy.BEST_EFFORT,
                applied_speed_factor=capped_factor,
                fitted_duration_ms=capped_duration_ms,
                overflow_ms=overflow_ms,
                native_speed_factor=capped_factor,
                atempo_factor=capped_factor,
                status=CueStatus.NEEDS_SHORTENING.value,
                warning_codes=warnings,
            )
        return FittingResult(
            strategy=FittingStrategy.BEST_EFFORT,
            applied_speed_factor=capped_factor,
            fitted_duration_ms=capped_duration_ms,
            overflow_ms=overflow_ms,
            native_speed_factor=capped_factor,
            atempo_factor=capped_factor,
            status=CueStatus.SPEED_UP.value,
            warning_codes=warnings,
        )

    def apply_result(self, cue: DubbingCue, result: FittingResult) -> None:
        cue.applied_speed_factor = round(result.applied_speed_factor, 4)
        cue.fitted_duration_ms = result.fitted_duration_ms
        cue.overflow_ms = result.overflow_ms
        cue.fitting_strategy = result.strategy.value
        cue.native_speed_factor = (
            round(result.native_speed_factor, 4)
            if result.native_speed_factor is not None
            else None
        )
        cue.required_speed_factor = (
            round((cue.raw_duration_ms or 0) / cue.duration_budget_ms, 4)
            if cue.raw_duration_ms and cue.duration_budget_ms
            else None
        )
        cue.status = result.status
        cue.warning_codes = list(result.warning_codes)

    @staticmethod
    def placement_offset(
        fitted_duration_ms: int,
        budget_ms: int,
        alignment: Alignment = Alignment.START,
    ) -> int:
        slack = max(0, budget_ms - fitted_duration_ms)
        if alignment == Alignment.CENTER:
            return slack // 2
        if alignment == Alignment.END:
            return slack
        return 0

    @staticmethod
    def build_atempo_chain(speed_factor: float) -> list[str]:
        remaining = max(0.01, min(100.0, float(speed_factor)))
        filters: list[str] = []
        while remaining > 2.0:
            filters.append("atempo=2.000")
            remaining /= 2.0
        while remaining < 0.5:
            filters.append("atempo=0.500")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.4f}")
        return filters
