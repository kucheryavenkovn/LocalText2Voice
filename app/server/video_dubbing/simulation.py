"""Dry-run simulation + variant comparison.

Simulation never touches the project, the SQLite DB, the manifest, the
artifacts, the stale flags or the revision. It deep-copies the selected cues,
re-runs the deterministic :class:`DurationFitter` on the copies under modified
settings, and scores the result with :func:`score_variant`.

LLM builds/selects variants; this module computes the objective score.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any

from app.core.video_dubbing.duration_fitter import DurationFitter
from app.core.video_dubbing.models import (
    DubbingCue,
    DubbingProject,
    DubbingProjectSettings,
)

from .quality import QualityWeights, score_variant


@dataclass
class Variant:
    variant_id: str
    label: str
    description: str
    cues: list[DubbingCue]
    settings: DubbingProjectSettings
    requires_tts: bool
    requires_refit: bool
    actions: list[str]


def _copy_cues(project: DubbingProject, sequences: list[int]) -> list[DubbingCue]:
    wanted = set(sequences)
    return [copy.deepcopy(c) for c in project.cues if c.sequence in wanted]


def _project_for_scoring(project: DubbingProject, cues: list[DubbingCue]) -> DubbingProject:
    """Build a lightweight stand-in project carrying the full cue set (copies
    for affected sequences, originals for unaffected) so overlap/shift math
    sees the real neighbours."""
    merged: list[DubbingCue] = []
    replaced = {c.sequence: c for c in cues}
    for cue in project.cues:
        merged.append(replaced.get(cue.sequence, cue))
    proj = copy.copy(project)
    proj.cues = merged
    return proj


def _refit_copies(
    project: DubbingProject,
    copies: list[DubbingCue],
    settings: DubbingProjectSettings,
) -> None:
    """Re-run the fitter on the copied cues only (no disk, no TTS)."""
    fitter = DurationFitter(settings, video_duration_ms=project.duration_ms)
    ordered = sorted(project.cues, key=lambda c: c.effective_start_ms())
    seq_to_next = {}
    for i, cue in enumerate(ordered):
        nxt = None
        for cand in ordered[i + 1 :]:
            if cand.enabled and cand.sequence != cue.sequence:
                nxt = cand
                break
        seq_to_next[cue.sequence] = nxt
    for cue in copies:
        if not cue.raw_duration_ms or cue.raw_duration_ms <= 0:
            continue
        nxt = seq_to_next.get(cue.sequence)
        try:
            result = fitter.evaluate(cue, next_cue=nxt)
            fitter.apply_result(cue, result)
        except Exception:
            continue


def _has_raw(cues: list[DubbingCue]) -> bool:
    return any(c.raw_duration_ms and c.raw_duration_ms > 0 for c in cues)


def build_variants(
    project: DubbingProject,
    sequences: list[int],
    changes: dict[str, Any] | None,
) -> list[Variant]:
    changes = changes or {}
    variants: list[Variant] = []
    base_settings = copy.deepcopy(project.settings)

    preferred = float(changes.get("preferred_speed_limit", base_settings.preferred_speed_limit))
    hard = float(changes.get("hard_speed_limit", base_settings.hard_speed_limit))
    hard = max(hard, preferred)
    max_shift = int(changes.get("max_shift_ms", 0) or 0)
    elastic_mode = str(changes.get("elastic_mode", base_settings.elastic_timing.shift_direction))

    # Baseline (no change) - always present for comparison.
    variants.append(
        Variant(
            variant_id="baseline",
            label="baseline",
            description="No changes (current state).",
            cues=_copy_cues(project, sequences),
            settings=copy.deepcopy(base_settings),
            requires_tts=False,
            requires_refit=False,
            actions=[],
        )
    )

    # Variant A: relax the preferred speed limit (reduce overflow via speed).
    if _has_raw(_copy_cues(project, sequences)):
        s = copy.deepcopy(base_settings)
        s.preferred_speed_limit = preferred
        s.hard_speed_limit = hard
        s.max_speed_factor = preferred
        variants.append(
            Variant(
                variant_id="adjust_speed",
                label="adjust_speed",
                description=f"Raise preferred speed to {preferred} (hard {hard}) and re-fit.",
                cues=_copy_cues(project, sequences),
                settings=s,
                requires_tts=False,
                requires_refit=True,
                actions=["adjust_speed"],
            )
        )

    # Variant B: shift the group right (elastic, right-only) by max_shift.
    if max_shift > 0 and elastic_mode != "left_only":
        copies = _copy_cues(project, sequences)
        ordered = sorted(copies, key=lambda c: c.sequence)
        per = max(1, max_shift // max(1, len(ordered)))
        for cue in ordered:
            cue.ensure_source_timing()
            if cue.planned_start_ms is None:
                cue.planned_start_ms = cue.source_start_ms if cue.source_start_ms is not None else cue.start_ms
            cue.planned_start_ms = int(cue.planned_start_ms) + per
            if cue.planned_end_ms is None:
                cue.planned_end_ms = cue.source_end_ms if cue.source_end_ms is not None else cue.end_ms
            cue.planned_end_ms = int(cue.planned_end_ms) + per
        variants.append(
            Variant(
                variant_id="elastic_shift_right",
                label="elastic_shift_right",
                description=f"Shift group right by ~{per}ms/cue (total budget {max_shift}ms).",
                cues=copies,
                settings=copy.deepcopy(base_settings),
                requires_tts=False,
                requires_refit=True,
                actions=["elastic_shift_right"],
            )
        )

    # Variant C: combine speed + shift (best of both).
    if max_shift > 0 and _has_raw(_copy_cues(project, sequences)) and elastic_mode != "left_only":
        copies = _copy_cues(project, sequences)
        ordered = sorted(copies, key=lambda c: c.sequence)
        per = max(1, max_shift // max(1, len(ordered)))
        for cue in ordered:
            cue.ensure_source_timing()
            if cue.planned_start_ms is None:
                cue.planned_start_ms = cue.source_start_ms if cue.source_start_ms is not None else cue.start_ms
            if cue.planned_end_ms is None:
                cue.planned_end_ms = cue.source_end_ms if cue.source_end_ms is not None else cue.end_ms
            cue.planned_start_ms = int(cue.planned_start_ms) + per
            cue.planned_end_ms = int(cue.planned_end_ms) + per
        s = copy.deepcopy(base_settings)
        s.preferred_speed_limit = preferred
        s.hard_speed_limit = hard
        s.max_speed_factor = preferred
        variants.append(
            Variant(
                variant_id="speed_and_shift",
                label="speed_and_shift",
                description=f"Shift right {per}ms/cue AND raise speed to {preferred}.",
                cues=copies,
                settings=s,
                requires_tts=False,
                requires_refit=True,
                actions=["adjust_speed", "elastic_shift_right"],
            )
        )

    return variants


def evaluate_variant(
    project: DubbingProject,
    variant: Variant,
    *,
    weights: QualityWeights | None = None,
    baseline_score: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before_cues = [c for c in project.cues if c.sequence in {v.sequence for v in variant.cues}]
    if variant.requires_refit:
        _refit_copies(project, variant.cues, variant.settings)
    scoring_proj_before = _project_for_scoring(project, before_cues)
    scoring_proj_after = _project_for_scoring(project, variant.cues)
    before = score_variant(
        scoring_proj_before.cues,
        preferred_speed=project.settings.preferred_speed_limit,
        hard_speed=project.settings.hard_speed_limit,
        guard_gap_ms=project.settings.guard_gap_ms,
        video_duration_ms=project.duration_ms,
        weights=weights,
        regeneration_count=0,
    )
    after = score_variant(
        scoring_proj_after.cues,
        preferred_speed=variant.settings.preferred_speed_limit,
        hard_speed=variant.settings.hard_speed_limit,
        guard_gap_ms=variant.settings.guard_gap_ms,
        video_duration_ms=project.duration_ms,
        weights=weights,
        regeneration_count=len(variant.cues) if variant.requires_refit else 0,
    )
    return {
        "variant_id": variant.variant_id,
        "label": variant.label,
        "description": variant.description,
        "affected_cues": sorted({c.sequence for c in variant.cues}),
        "actions": variant.actions,
        "before": _summary(before),
        "after": _summary(after),
        "score": after["score"],
        "score_breakdown": after["score_breakdown"],
        "requires_tts": variant.requires_tts,
        "requires_refit": variant.requires_refit,
        "warnings": [],
    }


def _summary(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_overflow_ms": score["total_overflow_ms"],
        "max_speed_factor": score["max_speed_factor"],
        "total_shift_ms": score["total_shift_ms"],
        "total_overlap_ms": score["total_overlap_ms"],
        "score": score["score"],
    }


def compare_variants(
    project: DubbingProject,
    sequences: list[int],
    changes: dict[str, Any] | None,
    *,
    weights: QualityWeights | None = None,
) -> list[dict[str, Any]]:
    variants = build_variants(project, sequences, changes)
    baseline = evaluate_variant(project, variants[0], weights=weights)
    out = [baseline]
    for variant in variants[1:]:
        out.append(evaluate_variant(project, variant, weights=weights, baseline_score=baseline))
    return out


def optimize(
    project: DubbingProject,
    sequences: list[int],
    *,
    max_iterations: int = 5,
    allowed_actions: list[str] | None = None,
    max_shift_ms: int = 800,
    max_speed_factor: float = 1.45,
    apply: bool = False,
    weights: QualityWeights | None = None,
) -> dict[str, Any]:
    max_iterations = max(1, min(10, int(max_iterations)))
    allowed = set(allowed_actions or ["adjust_speed", "elastic_shift_right"])
    changes = {
        "preferred_speed_limit": min(max_speed_factor, project.settings.hard_speed_limit),
        "hard_speed_limit": max(project.settings.hard_speed_limit, max_speed_factor),
        "max_shift_ms": max_shift_ms if "elastic_shift_right" in allowed else 0,
        "elastic_mode": project.settings.elastic_timing.shift_direction,
    }
    # Restrict variants to allowed actions.
    all_variants = compare_variants(project, sequences, changes, weights=weights)
    candidates = [v for v in all_variants if not v["actions"] or set(v["actions"]).issubset(allowed)]
    if not candidates:
        candidates = all_variants
    baseline = next((v for v in candidates if not v["actions"]), candidates[0])
    best = min((v for v in candidates if v["actions"]), key=lambda v: v["score"], default=None)
    if best is None:
        best = baseline
    improvement = 0.0
    if baseline["score"] > 0:
        improvement = round(
            max(0.0, (baseline["score"] - best["score"]) / baseline["score"] * 100.0), 2
        )
    alternatives = [v for v in candidates if v["variant_id"] != best["variant_id"]]
    return {
        "best_variant": best,
        "alternatives": alternatives,
        "baseline": baseline,
        "improvement_percent": improvement,
        "applied": False,  # actual application happens in the facade under lock
        "requires_human_review": False,
        "max_iterations": max_iterations,
        "allowed_actions": sorted(allowed),
    }


def new_variant_id() -> str:
    return uuid.uuid4().hex[:10]


__all__ = [
    "Variant",
    "build_variants",
    "evaluate_variant",
    "compare_variants",
    "optimize",
]
