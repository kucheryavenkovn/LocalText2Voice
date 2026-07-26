from __future__ import annotations

from dataclasses import dataclass

from .models import DubbingCue, TempoSmoothingMode, TempoSmoothingSettings


@dataclass(frozen=True)
class SmoothingAssignment:
    cue_id: str
    sequence: int
    required_factor: float
    planned_factor: float
    group_id: str
    reason: str


@dataclass
class SmoothingGroupPlan:
    group_id: str
    sequences: list[int]
    assignments: list[SmoothingAssignment]
    reason: str


class TempoSmoothingPlanner:
    """Smooth speed jumps between neighboring cues without moving timecodes.

    Strict-mode contract: never mutates source/planned start/end windows here.
    Only planned_speed_factor values are produced for the fitter to consume.
    """

    def __init__(self, settings: TempoSmoothingSettings) -> None:
        self.settings = settings

    def required_factor(self, cue: DubbingCue) -> float:
        raw = float(cue.raw_duration_ms or 0)
        if raw <= 0:
            return 1.0
        budget = max(1, int(cue.duration_budget_ms or (cue.end_ms - cue.start_ms)))
        return max(1.0, raw / budget)

    def plan(self, cues: list[DubbingCue]) -> list[SmoothingGroupPlan]:
        if self.settings.mode == TempoSmoothingMode.OFF:
            return []
        enabled = [
            c
            for c in sorted(cues, key=lambda item: item.sequence)
            if c.enabled and (c.raw_duration_ms or 0) > 0
        ]
        if len(enabled) < 2:
            return []

        required = {c.sequence: self.required_factor(c) for c in enabled}
        plans: list[SmoothingGroupPlan] = []
        index = 0
        group_counter = 1
        while index < len(enabled):
            if self._is_boundary(enabled[index]):
                index += 1
                continue
            end = index
            while end + 1 < len(enabled):
                left = enabled[end]
                right = enabled[end + 1]
                if self._is_boundary(right):
                    break
                if end - index + 2 > self.settings.max_cues_per_group:
                    break
                jump = abs(required[left.sequence] - required[right.sequence])
                left_needs = required[left.sequence] > 1.0 + 1e-9
                right_needs = required[right.sequence] > 1.0 + 1e-9
                should_join = (
                    jump >= self.settings.speed_jump_threshold
                    or (left_needs and right_needs)
                    or (left_needs and not right_needs)
                    or (right_needs and not left_needs)
                )
                if not should_join and end == index:
                    break
                if not should_join:
                    break
                end += 1
            if end == index:
                index += 1
                continue
            group = enabled[index : end + 1]
            plan = self._smooth_group(group, required, group_counter)
            if plan is not None:
                plans.append(plan)
                group_counter += 1
            index = end + 1
        return plans

    def apply(self, cues: list[DubbingCue], plans: list[SmoothingGroupPlan]) -> int:
        by_seq = {c.sequence: c for c in cues}
        # Clear previous auto smoothing markers first.
        for cue in cues:
            if cue.smoothing_group_id and str(cue.smoothing_group_id).startswith("S-"):
                cue.smoothing_group_id = None
                cue.smoothing_reason = ""
                cue.planned_speed_factor = None
        changed = 0
        for plan in plans:
            for item in plan.assignments:
                cue = by_seq.get(item.sequence)
                if cue is None:
                    continue
                cue.required_speed_factor = item.required_factor
                cue.planned_speed_factor = item.planned_factor
                cue.smoothing_group_id = item.group_id
                cue.smoothing_reason = item.reason
                changed += 1
        return changed

    def _is_boundary(self, cue: DubbingCue) -> bool:
        if self.settings.preserve_locked_cues and cue.timing_locked:
            return True
        if not cue.enabled:
            return True
        return False

    def _smooth_group(
        self,
        group: list[DubbingCue],
        required: dict[int, float],
        group_no: int,
    ) -> SmoothingGroupPlan | None:
        reqs = [required[c.sequence] for c in group]
        if max(reqs) - min(reqs) < self.settings.speed_jump_threshold and max(reqs) <= 1.0 + 1e-9:
            return None
        group_id = f"S-{group_no:03d}"
        if self.settings.mode == TempoSmoothingMode.COMMON_GROUP_FACTOR:
            factors = self._common_factor(group, reqs)
            reason = "единый темп для группы"
        else:
            factors = self._smooth_factors(group, reqs)
            reason = "сглаживание соседних реплик"
        if factors is None:
            return None
        # Skip no-op plans where nothing moves from required.
        if all(abs(f - r) < 1e-4 for f, r in zip(factors, reqs)):
            # Still useful when neighbors had a jump that required equalization
            if max(reqs) - min(reqs) < self.settings.speed_jump_threshold:
                return None
        assignments = [
            SmoothingAssignment(
                cue_id=cue.cue_id,
                sequence=cue.sequence,
                required_factor=reqs[i],
                planned_factor=factors[i],
                group_id=group_id,
                reason=reason,
            )
            for i, cue in enumerate(group)
        ]
        return SmoothingGroupPlan(
            group_id=group_id,
            sequences=[c.sequence for c in group],
            assignments=assignments,
            reason=reason,
        )

    def _cap_optional(self, required: float, candidate: float) -> float:
        """Limit speedup for cues that already fit naturally."""
        if required <= 1.0 + 1e-9:
            ceiling = 1.0 + max(0.0, self.settings.max_optional_speedup)
            return min(candidate, ceiling, self.settings.max_speed_factor)
        return min(max(candidate, required), self.settings.max_speed_factor)

    def _common_factor(
        self, group: list[DubbingCue], reqs: list[float]
    ) -> list[float] | None:
        common = max(reqs)
        if common > self.settings.max_speed_factor + 1e-9:
            return None
        # Don't yank a natural cue too hard toward a single extreme neighbor.
        natural = min(reqs)
        if common - natural > max(0.25, self.settings.max_optional_speedup + 0.05):
            return None
        factors = []
        for req in reqs:
            factors.append(self._cap_optional(req, common))
        # Re-check common mode still keeps everyone fitting.
        if any(f + 1e-9 < r for f, r in zip(factors, reqs)):
            return None
        return factors

    def _smooth_factors(
        self, group: list[DubbingCue], reqs: list[float]
    ) -> list[float] | None:
        n = len(reqs)
        # Start at required (hard lower bounds), then iteratively pull neighbors.
        factors = [max(self.settings.min_speed_factor, r) for r in reqs]
        delta = max(0.001, self.settings.max_neighbor_speed_delta)
        max_factor = self.settings.max_speed_factor
        for _ in range(40):
            changed = False
            for i in range(1, n):
                left = factors[i - 1]
                right = factors[i]
                gap = right - left
                if abs(gap) <= delta + 1e-9:
                    continue
                # Pull the lower one up first (never below its required).
                if gap > delta:
                    # right much faster than left: raise left toward right-delta
                    target_left = min(max_factor, max(reqs[i - 1], right - delta))
                    target_right = max(reqs[i], min(max_factor, left + delta))
                    # Prefer natural: don't raise more than needed for delta.
                    new_left = max(left, min(target_left, max_factor))
                    new_right = min(right, max(target_right, reqs[i]))
                    # If still too far, raise the slower side.
                    if new_right - new_left > delta:
                        new_left = max(new_left, new_right - delta)
                    new_left = self._cap_optional(reqs[i - 1], new_left)
                    new_right = self._cap_optional(reqs[i], new_right)
                    if abs(new_left - left) > 1e-6 or abs(new_right - right) > 1e-6:
                        factors[i - 1] = new_left
                        factors[i] = new_right
                        changed = True
                else:
                    # left much faster than right
                    target_right = min(max_factor, max(reqs[i], left - delta))
                    target_left = max(reqs[i - 1], min(max_factor, right + delta))
                    new_right = max(right, min(target_right, max_factor))
                    new_left = min(left, max(target_left, reqs[i - 1]))
                    if new_left - new_right > delta:
                        new_right = max(new_right, new_left - delta)
                    new_left = self._cap_optional(reqs[i - 1], new_left)
                    new_right = self._cap_optional(reqs[i], new_right)
                    if abs(new_left - left) > 1e-6 or abs(new_right - right) > 1e-6:
                        factors[i - 1] = new_left
                        factors[i] = new_right
                        changed = True
            # Enforce bounds again.
            for i in range(n):
                factors[i] = min(max_factor, max(reqs[i], factors[i]))
            if not changed:
                break
        # Final neighbor check — if still violated, fall back to local max pairs.
        for i in range(1, n):
            if abs(factors[i] - factors[i - 1]) > delta + 1e-6:
                shared = max(factors[i], factors[i - 1])
                factors[i] = self._cap_optional(reqs[i], shared)
                factors[i - 1] = self._cap_optional(reqs[i - 1], shared)
        for i in range(n):
            if factors[i] + 1e-9 < reqs[i]:
                return None
            if factors[i] > max_factor + 1e-9:
                return None
        return factors
