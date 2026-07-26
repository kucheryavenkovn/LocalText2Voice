from __future__ import annotations

from app.core.video_dubbing.models import (
    DubbingCue,
    TempoSmoothingMode,
    TempoSmoothingSettings,
)
from app.core.video_dubbing.tempo_smoothing import TempoSmoothingPlanner


def _cue(seq: int, start: int, end: int, raw_ms: int) -> DubbingCue:
    cue = DubbingCue(
        cue_id=str(seq),
        sequence=seq,
        start_ms=start,
        end_ms=end,
        duration_budget_ms=end - start,
        source_text=f"t{seq}",
        spoken_text=f"t{seq}",
        raw_duration_ms=raw_ms,
    )
    cue.ensure_source_timing()
    return cue


def test_smooth_reduces_neighbor_jumps_without_moving_timecodes():
    settings = TempoSmoothingSettings(
        mode=TempoSmoothingMode.SMOOTH,
        max_cues_per_group=5,
        speed_jump_threshold=0.08,
        max_neighbor_speed_delta=0.08,
        max_speed_factor=1.35,
        max_optional_speedup=0.25,
    )
    planner = TempoSmoothingPlanner(settings)
    # Individual required: ~1.00, 1.28, 1.02, 1.31, 1.00
    cues = [
        _cue(1, 0, 1000, 1000),
        _cue(2, 1100, 2100, 1280),
        _cue(3, 2200, 3200, 1020),
        _cue(4, 3300, 4300, 1310),
        _cue(5, 4400, 5400, 1000),
    ]
    source = [(c.start_ms, c.end_ms) for c in cues]
    plans = planner.plan(cues)
    assert plans
    planner.apply(cues, plans)
    # Timecodes unchanged (strict contract of tempo smoothing).
    assert [(c.start_ms, c.end_ms) for c in cues] == source
    factors = [
        c.planned_speed_factor or c.required_speed_factor or 1.0 for c in cues if c.planned_speed_factor
    ]
    assert factors
    for i in range(1, len(cues)):
        left = cues[i - 1].planned_speed_factor or cues[i - 1].required_speed_factor or 1.0
        right = cues[i].planned_speed_factor or cues[i].required_speed_factor or 1.0
        # Within smoothed groups, deltas are limited.
        if cues[i].smoothing_group_id and cues[i].smoothing_group_id == cues[i - 1].smoothing_group_id:
            assert abs(left - right) <= settings.max_neighbor_speed_delta + 1e-6


def test_common_group_factor_uses_max_required():
    settings = TempoSmoothingSettings(
        mode=TempoSmoothingMode.COMMON_GROUP_FACTOR,
        max_cues_per_group=3,
        speed_jump_threshold=0.05,
        max_speed_factor=1.35,
        max_optional_speedup=0.30,
    )
    planner = TempoSmoothingPlanner(settings)
    cues = [
        _cue(1, 0, 1000, 1050),   # 1.05
        _cue(2, 1100, 2100, 1220),  # 1.22
        _cue(3, 2200, 3200, 1100),  # 1.10
    ]
    plans = planner.plan(cues)
    assert plans
    planner.apply(cues, plans)
    planned = [c.planned_speed_factor for c in cues]
    assert all(p is not None for p in planned)
    assert max(planned) == max(planner.required_factor(c) for c in cues) or all(
        abs(p - max(planner.required_factor(c) for c in cues)) < 0.05 for p in planned
    )


def test_optional_speedup_cap_for_fitting_cues():
    settings = TempoSmoothingSettings(
        mode=TempoSmoothingMode.SMOOTH,
        max_cues_per_group=3,
        speed_jump_threshold=0.05,
        max_neighbor_speed_delta=0.20,
        max_optional_speedup=0.15,
        max_speed_factor=1.35,
    )
    planner = TempoSmoothingPlanner(settings)
    cues = [
        _cue(1, 0, 1000, 1300),  # needs 1.30
        _cue(2, 1100, 2100, 1000),  # fits 1.00
    ]
    plans = planner.plan(cues)
    planner.apply(cues, plans)
    # Fitting cue may rise a bit, but not beyond 1.15
    if cues[1].planned_speed_factor:
        assert cues[1].planned_speed_factor <= 1.15 + 1e-6


def test_off_mode_no_plans():
    settings = TempoSmoothingSettings(mode=TempoSmoothingMode.OFF)
    planner = TempoSmoothingPlanner(settings)
    cues = [_cue(1, 0, 1000, 1300), _cue(2, 1100, 2100, 1000)]
    assert planner.plan(cues) == []


def test_locked_cue_breaks_group():
    settings = TempoSmoothingSettings(
        mode=TempoSmoothingMode.SMOOTH,
        max_cues_per_group=5,
        speed_jump_threshold=0.05,
        preserve_locked_cues=True,
    )
    planner = TempoSmoothingPlanner(settings)
    cues = [
        _cue(1, 0, 1000, 1280),
        _cue(2, 1100, 2100, 1000),
        _cue(3, 2200, 3200, 1310),
    ]
    cues[1].timing_locked = True
    plans = planner.plan(cues)
    for plan in plans:
        assert 2 not in plan.sequences
