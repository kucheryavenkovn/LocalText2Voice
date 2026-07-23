from __future__ import annotations

from .models import StaleFlags


def invalidate_for_text_change(stale: StaleFlags) -> None:
    stale.cues = True
    stale.narration = True
    stale.mix = True
    stale.preview = True
    stale.video = True


def invalidate_for_ducking_change(stale: StaleFlags) -> None:
    stale.mix = True
    stale.preview = True
    stale.video = True


def invalidate_for_container_change(stale: StaleFlags) -> None:
    stale.video = True


def invalidate_for_preview_change(stale: StaleFlags) -> None:
    stale.preview = True


def invalidate_for_narration_change(stale: StaleFlags) -> None:
    stale.narration = True
    stale.mix = True
    stale.preview = True
    stale.video = True


def mark_clean(stale: StaleFlags) -> None:
    stale.cues = False
    stale.narration = False
    stale.mix = False
    stale.preview = False
    stale.video = False
