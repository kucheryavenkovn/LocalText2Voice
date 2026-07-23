from __future__ import annotations

import math
from dataclasses import dataclass, field

from .models import (
    Alignment,
    CueStatus,
    DubbingCue,
    DubbingProjectSettings,
    FittingStrategy,
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
    warning_codes: list[str] = field(default_factory=list)
    target_duration_ms: int | None = None
    safe_end_ms: int | None = None
    timing_diff_ms: int | None = None
    blocking: bool = False


# Tolerance for the second-pass correction (ms).
TIMING_TOLERANCE_MS = 20


class DurationFitter:
    """Computes the per-cue speed decision under the new fitting policy.

    Each cue is fitted to ``target_duration_ms = safe_end_ms - start_ms`` where
    ``safe_end_ms = min(end_ms, next_cue.start_ms - guard_gap, video_duration)``.

    Bands:
      * <= 1.0            -> no speedup (rendered)
      * 1.0 .. preferred  -> speed_up (blue)
      * preferred .. hard -> strong_speed_up (orange, WARNING ONLY, still fits)
      * > hard            -> extreme_speed_required (red, blocks unless force)
    """

    def __init__(
        self,
        settings: DubbingProjectSettings,
        video_duration_ms: int | None = None,
    ) -> None:
        self.settings = settings
        self.video_duration_ms = video_duration_ms

    # ------------------------------------------------------------------ windows

    def safe_end_ms(
        self,
        cue: DubbingCue,
        next_cue: DubbingCue | None,
    ) -> int:
        end = cue.end_ms
        if next_cue is not None and next_cue.start_ms > cue.start_ms:
            end = min(end, next_cue.start_ms - self.settings.guard_gap_ms)
        if self.video_duration_ms and self.video_duration_ms > 0:
            end = min(end, self.video_duration_ms)
        return max(cue.start_ms + 1, end)

    def target_duration_ms(
        self,
        cue: DubbingCue,
        next_cue: DubbingCue | None,
    ) -> int:
        safe_end = self.safe_end_ms(cue, next_cue)
        return max(1, safe_end - cue.start_ms)

    # ------------------------------------------------------------------ evaluate

    def evaluate(
        self,
        cue: DubbingCue,
        next_cue: DubbingCue | None = None,
    ) -> FittingResult:
        raw_ms = cue.raw_duration_ms
        if raw_ms is None:
            raise ValueError(f"Cue #{cue.sequence} has no raw duration yet.")
        if raw_ms <= 0:
            raise ValueError(
                f"Cue #{cue.sequence} has non-positive raw duration {raw_ms}."
            )

        safe_end = self.safe_end_ms(cue, next_cue)
        target_ms = max(1, safe_end - cue.start_ms)
        required = raw_ms / target_ms

        preferred = self.settings.preferred_speed_limit
        hard = self._hard_limit_for(cue)

        base = FittingResult(
            strategy=FittingStrategy.NONE,
            applied_speed_factor=1.0,
            fitted_duration_ms=raw_ms,
            overflow_ms=0,
            status=CueStatus.RENDERED.value,
            target_duration_ms=target_ms,
            safe_end_ms=safe_end,
            timing_diff_ms=0,
        )

        if required <= 1.0:
            # Fits naturally; fitted duration is the raw (<= target). Overflow
            # is 0 because the cue finishes before the window ends.
            base.timing_diff_ms = max(0, raw_ms - target_ms)
            return base

        # Needs speedup. Choose status by band.
        warnings: list[str] = [f"required_factor_{required:.2f}"]

        if required <= preferred:
            status = CueStatus.SPEED_UP.value
        elif required <= hard:
            status = CueStatus.STRONG_SPEED_UP.value
            warnings.append("strong_speed_up")
        else:
            if cue.force_fit:
                # Explicit user override: fit beyond hard limit with a warning.
                status = CueStatus.STRONG_SPEED_UP.value
                warnings.append("force_fit_beyond_hard_limit")
            else:
                return FittingResult(
                    strategy=FittingStrategy.BEST_EFFORT,
                    applied_speed_factor=hard,
                    fitted_duration_ms=max(1, int(math.ceil(raw_ms / hard))),
                    overflow_ms=max(0, int(math.ceil(raw_ms / hard)) - target_ms),
                    native_speed_factor=hard,
                    atempo_factor=hard,
                    status=CueStatus.EXTREME_SPEED_REQUIRED.value,
                    warning_codes=warnings + ["exceeds_hard_limit"],
                    target_duration_ms=target_ms,
                    safe_end_ms=safe_end,
                    timing_diff_ms=max(0, int(math.ceil(raw_ms / hard)) - target_ms),
                    blocking=True,
                )

        # Fit exactly to the target window.
        fitted_ms = target_ms
        return FittingResult(
            strategy=FittingStrategy.ATEMPO,
            applied_speed_factor=round(required, 4),
            fitted_duration_ms=fitted_ms,
            overflow_ms=0,
            native_speed_factor=round(required, 4),
            atempo_factor=round(required, 4),
            status=status,
            warning_codes=warnings,
            target_duration_ms=target_ms,
            safe_end_ms=safe_end,
            timing_diff_ms=0,
        )

    def _hard_limit_for(self, cue: DubbingCue) -> float:
        if cue.hard_speed_override and cue.hard_speed_override > 0:
            return max(self.settings.preferred_speed_limit, cue.hard_speed_override)
        return max(
            self.settings.preferred_speed_limit,
            self.settings.hard_speed_limit,
        )

    # ------------------------------------------------------------------ apply

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
        if result.target_duration_ms is not None:
            cue.target_duration_ms = result.target_duration_ms
        if result.safe_end_ms is not None:
            cue.safe_end_ms = result.safe_end_ms
        target = result.target_duration_ms or cue.duration_budget_ms
        cue.required_speed_factor = (
            round((cue.raw_duration_ms or 0) / target, 4)
            if cue.raw_duration_ms and target
            else None
        )
        cue.status = result.status
        cue.warning_codes = list(result.warning_codes)
        measured = cue.fitted_duration_ms or 0
        cue.timing_diff_ms = max(0, measured - target)

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
