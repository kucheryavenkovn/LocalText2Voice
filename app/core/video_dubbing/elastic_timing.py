from __future__ import annotations

from dataclasses import dataclass, field

from .models import CueStatus, DubbingCue, ElasticTimingSettings


MAX_BINARY_SEARCH_ITERATIONS = 40


@dataclass(frozen=True)
class ScheduledCue:
    cue_id: str
    sequence: int
    planned_start_ms: int
    planned_end_ms: int
    fitted_duration_ms: int
    start_shift_ms: int
    end_shift_ms: int
    borrowed_right_ms: int


@dataclass(frozen=True)
class GroupScheduleResult:
    fits: bool
    factor: float
    cues: tuple[ScheduledCue, ...] = ()
    total_shift_ms: int = 0
    max_shift_ms: int = 0
    borrowed_right_ms: int = 0
    boundary_end_ms: int = 0
    reason: str = ""


@dataclass
class ElasticGroupPlan:
    group_id: str
    cue_ids: list[str]
    sequences: list[int]
    common_speed_factor: float
    boundary_end_ms: int
    borrowed_right_ms: int
    max_shift_ms: int
    total_shift_ms: int
    schedule: list[ScheduledCue] = field(default_factory=list)
    locked: bool = False


class ElasticTimingPlanner:
    """Plans elastic cue groups with a shared speed factor and right-only shifts."""

    def __init__(
        self,
        settings: ElasticTimingSettings,
        video_duration_ms: int | None = None,
    ) -> None:
        self.settings = settings
        self.video_duration_ms = video_duration_ms or 0

    def group_boundary_end(
        self,
        group: list[DubbingCue],
        next_external: DubbingCue | None,
    ) -> int:
        last = group[-1]
        last.ensure_source_timing()
        source_end = last.source_end_ms if last.source_end_ms is not None else last.end_ms
        candidates = [
            source_end + self.settings.max_group_extension_ms,
        ]
        if self.video_duration_ms > 0:
            candidates.append(self.video_duration_ms)
        if next_external is not None:
            next_external.ensure_source_timing()
            next_start = (
                next_external.source_start_ms
                if next_external.source_start_ms is not None
                else next_external.start_ms
            )
            candidates.append(next_start - self.settings.boundary_guard_ms)
        return max(group[0].start_ms + 1, min(candidates))

    def schedule_group(
        self,
        group: list[DubbingCue],
        factor: float,
        boundary_end_ms: int,
    ) -> GroupScheduleResult:
        if factor <= 0:
            return GroupScheduleResult(fits=False, factor=factor, reason="invalid_factor")
        planned: list[ScheduledCue] = []
        prev_end: int | None = None
        total_shift = 0
        max_shift = 0
        for index, cue in enumerate(group):
            cue.ensure_source_timing()
            raw_ms = int(cue.raw_duration_ms or 0)
            if raw_ms <= 0:
                return GroupScheduleResult(
                    fits=False, factor=factor, reason=f"missing_raw_{cue.sequence}"
                )
            fitted_ms = max(1, int(round(raw_ms / factor)))
            source_start = (
                cue.source_start_ms if cue.source_start_ms is not None else cue.start_ms
            )
            source_end = cue.source_end_ms if cue.source_end_ms is not None else cue.end_ms
            if index == 0 or self.settings.preserve_first_cue_start and index == 0:
                start = source_start
            else:
                min_start = source_start
                if prev_end is not None:
                    min_start = max(
                        source_start, prev_end + self.settings.min_inter_cue_gap_ms
                    )
                start = min_start
            # Right-only: never earlier than source start.
            start = max(start, source_start)
            end = start + fitted_ms
            start_shift = start - source_start
            end_shift = end - source_end
            if start_shift < 0:
                return GroupScheduleResult(
                    fits=False, factor=factor, reason="left_shift_forbidden"
                )
            if start_shift > self.settings.max_shift_per_cue_ms:
                return GroupScheduleResult(
                    fits=False, factor=factor, reason="max_shift_exceeded"
                )
            total_shift += start_shift
            max_shift = max(max_shift, start_shift)
            planned.append(
                ScheduledCue(
                    cue_id=cue.cue_id,
                    sequence=cue.sequence,
                    planned_start_ms=start,
                    planned_end_ms=end,
                    fitted_duration_ms=fitted_ms,
                    start_shift_ms=start_shift,
                    end_shift_ms=end_shift,
                    borrowed_right_ms=max(0, end - source_end),
                )
            )
            prev_end = end
        last_end = planned[-1].planned_end_ms if planned else 0
        first_source_end = (
            group[0].source_end_ms
            if group[0].source_end_ms is not None
            else group[0].end_ms
        )
        borrowed = max(0, last_end - first_source_end)
        fits = last_end <= boundary_end_ms and factor <= self.settings.max_common_speed_factor + 1e-9
        return GroupScheduleResult(
            fits=fits,
            factor=factor,
            cues=tuple(planned),
            total_shift_ms=total_shift,
            max_shift_ms=max_shift,
            borrowed_right_ms=borrowed,
            boundary_end_ms=boundary_end_ms,
            reason="" if fits else "overflow",
        )

    def find_min_common_speed(
        self,
        group: list[DubbingCue],
        boundary_end_ms: int,
    ) -> GroupScheduleResult | None:
        base = self.schedule_group(group, 1.0, boundary_end_ms)
        if base.fits:
            return base
        high_factor = self.settings.max_common_speed_factor
        high = self.schedule_group(group, high_factor, boundary_end_ms)
        if not high.fits:
            return None
        low = 1.0
        best = high
        for _ in range(MAX_BINARY_SEARCH_ITERATIONS):
            mid = (low + high_factor) / 2.0
            result = self.schedule_group(group, mid, boundary_end_ms)
            if result.fits:
                best = result
                high_factor = mid
            else:
                low = mid
        return best

    def choose_group_for_cue(
        self,
        cues: list[DubbingCue],
        start_index: int,
    ) -> ElasticGroupPlan | None:
        """Pick the best elastic group starting at ``start_index``.

        Strategy (default): minimize common_speed_factor, then group size,
        then total shift, then max shift. When prefer_smaller_group is set,
        group size is weighted higher.
        """
        if not self.settings.enabled:
            return None
        enabled = [c for c in cues if c.enabled and not c.timing_locked]
        if start_index < 0 or start_index >= len(enabled):
            return None
        max_size = max(1, min(5, self.settings.max_cues_per_group))
        candidates: list[tuple[tuple, ElasticGroupPlan]] = []
        for size in range(1, max_size + 1):
            end = start_index + size
            if end > len(enabled):
                break
            group = enabled[start_index:end]
            next_external = enabled[end] if end < len(enabled) else None
            # Hard boundary: locked cues act as walls.
            for candidate in cues:
                if candidate.timing_locked and candidate.sequence > group[-1].sequence:
                    if next_external is None or candidate.sequence < next_external.sequence:
                        next_external = candidate
                    break
            boundary = self.group_boundary_end(group, next_external)
            result = self.find_min_common_speed(group, boundary)
            if result is None or not result.fits:
                continue
            plan = ElasticGroupPlan(
                group_id=f"E-{group[0].sequence:03d}",
                cue_ids=[c.cue_id for c in group],
                sequences=[c.sequence for c in group],
                common_speed_factor=result.factor,
                boundary_end_ms=boundary,
                borrowed_right_ms=result.borrowed_right_ms,
                max_shift_ms=result.max_shift_ms,
                total_shift_ms=result.total_shift_ms,
                schedule=list(result.cues),
            )
            size_term = size * (10 if self.settings.prefer_smaller_group else 1)
            score = (
                result.factor,
                size_term,
                result.total_shift_ms,
                result.max_shift_ms,
            )
            candidates.append((score, plan))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def apply_plan(self, cues: list[DubbingCue], plan: ElasticGroupPlan) -> None:
        by_id = {c.cue_id: c for c in cues}
        for position, scheduled in enumerate(plan.schedule):
            cue = by_id.get(scheduled.cue_id)
            if cue is None:
                continue
            cue.ensure_source_timing()
            cue.timing_group_id = plan.group_id
            cue.timing_group_position = position
            cue.common_speed_factor = plan.common_speed_factor
            cue.planned_start_ms = scheduled.planned_start_ms
            cue.planned_end_ms = scheduled.planned_end_ms
            cue.start_shift_ms = scheduled.start_shift_ms
            cue.end_shift_ms = scheduled.end_shift_ms
            cue.borrowed_right_ms = scheduled.borrowed_right_ms
            cue.target_duration_ms = scheduled.fitted_duration_ms
            cue.required_speed_factor = plan.common_speed_factor

    def clear_group(self, cues: list[DubbingCue], group_id: str) -> None:
        for cue in cues:
            if cue.timing_group_id != group_id:
                continue
            cue.timing_group_id = None
            cue.timing_group_position = None
            cue.common_speed_factor = None
            cue.planned_start_ms = None
            cue.planned_end_ms = None
            cue.start_shift_ms = 0
            cue.end_shift_ms = 0
            cue.borrowed_right_ms = 0

    def mark_group_failed(self, group: list[DubbingCue], message: str) -> None:
        for cue in group:
            if cue.status not in {
                CueStatus.FAILED.value,
                CueStatus.CANCELLED.value,
            }:
                cue.status = CueStatus.ELASTIC_GROUP_FAILED.value
                cue.error_message = message
                cue.warning_codes = list(cue.warning_codes) + ["elastic_group_failed"]
