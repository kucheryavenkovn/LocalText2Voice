"""Allowlist of domain actions for ``dubbing_execute_action``.

This is a *dispatcher over approved domain actions only*. Arbitrary Python,
shell, SQL or FFmpeg command lines are explicitly forbidden and rejected.
"""

from __future__ import annotations

from app.observability import emit_event
from app.server.video_dubbing.errors import DubbingValidationError

# Approved mutating / job actions.
ALLOWED_ACTIONS = frozenset(
    {
        "update_cue_text",
        "update_cue_timing",
        "shift_cues",
        "split_cue",
        "merge_cues",
        "enable_cue",
        "disable_cue",
        "reset_cue_to_source_timing",
        "apply_elastic_plan",
        "apply_tempo_plan",
        "generate_selected",
        "generate_missing",
        "generate_all",
        "refit_selected",
        "refit_all",
        "render_narration",
        "render_mix",
        "render_preview",
        "render_full_preview",
        "export_video",
    }
)

# Explicitly forbidden capabilities — names that must NEVER be accepted, even
# if a future contributor is tempted to wire them up.
FORBIDDEN_ACTIONS = frozenset(
    {
        "execute_python",
        "execute_shell",
        "execute_sql",
        "run_arbitrary_ffmpeg",
        "eval",
        "exec",
    }
)


def validate_action(action: str) -> str:
    """Return the normalised action name or raise."""
    name = str(action or "").strip()
    if name in FORBIDDEN_ACTIONS:
        emit_event(
            "mcp.action.forbidden",
            payload={"action": name},
            force_flush=True,
        )
        raise DubbingValidationError(
            f"Action '{name}' is forbidden.",
            code="forbidden_action",
        )
    if name not in ALLOWED_ACTIONS:
        raise DubbingValidationError(
            f"Unknown action '{name}'. Allowed: {sorted(ALLOWED_ACTIONS)}",
            code="unknown_action",
        )
    return name


def list_actions() -> list[dict[str, str]]:
    return [{"action": a} for a in sorted(ALLOWED_ACTIONS)]
