"""Static MCP resource payloads for the video-dubbing control plane."""

from __future__ import annotations

import json
from typing import Any

from .action_registry import ALLOWED_ACTIONS
from .quality import QualityWeights

DOCS_URI = "localtext2voice://video-dubbing/docs"
ACTIONS_URI = "localtext2voice://video-dubbing/actions"
SETTINGS_SCHEMA_URI = "localtext2voice://video-dubbing/settings-schema"
QUALITY_MODEL_URI = "localtext2voice://video-dubbing/quality-model"
SCENARIO_SCHEMA_URI = "localtext2voice://video-dubbing/scenario-schema"


def docs_text() -> str:
    return """# Video Dubbing MCP — Agent Guide

## Workflow

1. `dubbing_create_project` -> `dubbing_attach_video` -> `dubbing_import_srt`.
2. `dubbing_analyze_project` to find problematic cues.
3. Read cues with `dubbing_list_cues` (filters: status, has_overflow, needs_tts).
4. Model fixes with `dubbing_simulate_timing_changes` (dry_run never mutates).
5. Compare variants via `dubbing_compare_variants`; pick the lowest `score`.
6. Apply with `dubbing_shift_cues` / `dubbing_update_cue_timing` / `dubbing_optimize_timing` (pass expected_revision).
7. Regenerate only what changed: `dubbing_get_generation_plan` then `dubbing_generate_cues` (force=false reuses raw audio; force=true re-synthesizes).
8. Re-fit without TTS via `dubbing_refit_cues`.
9. Render: `dubbing_render_narration` -> `dubbing_render_mix` -> (optional) `dubbing_render_full_preview`.
10. Export with `dubbing_export_video` (requires `confirm_export=true`), then `dubbing_validate_final_video`.
11. Run `dubbing_build_quality_report` and `dubbing_assert_project_invariants`.

## What requires TTS vs refit

- Changing **text or voice** -> needs TTS (raw WAV must be regenerated).
- Changing **speed limits / timing** only -> needs **refit** (re-uses existing raw WAV, no TTS).
- `dubbing_get_generation_plan` classifies every cue into: ready_without_changes, needs_refit_from_raw, needs_tts_generation, missing_raw, missing_fitted, legacy_audio_unverified, failed, disabled. Do not re-run TTS if `needs_refit_from_raw` suffices.

## What invalidates downstream artifacts

- text/voice change -> cues, narration, mix, preview, video
- timing/fitting change -> narration, mix, preview, video
- ducking/volume change -> mix, preview, video
- container change -> video
- preview setting change -> preview

Use the `stale` flags from `dubbing_get_project_state` to know what must be rebuilt.

## Concurrency: revision & expected_revision

Mutations accept `expected_revision`. If it does not match the current revision
you get `error=project_changed` with `current_revision`. Re-read state and retry.
This prevents lost updates between UI and MCP.

## Jobs & cancellation

Heavy ops (generate/refit/render/export) return a `job_id` immediately. Poll with
`dubbing_get_job`, cancel with `dubbing_cancel_job`. Cancellation is cooperative
(no thread.terminate); the active FFmpeg/TTS is stopped, `.part` files are
recovered, and cues are not left in `rendering`.

## Human review

`dubbing_update_cue_text`, `dubbing_split_cue` and `dubbing_merge_cues` set
`requires_human_review=true` because they alter meaning.

## Dry-run

Pass `dry_run=true` to `dubbing_shift_cues` / `dubbing_simulate_timing_changes`.
Dry-run NEVER touches SQLite, the manifest, cues, artifacts, stale flags or revision.

## Security

Only allowlisted domain actions are accepted (`dubbing_execute_action`).
Arbitrary python/shell/sql/ffmpeg command lines are rejected. Paths are
normalised and path traversal is forbidden.
"""


def actions_text() -> str:
    return "Allowed actions:\n" + "\n".join(f"- {a}" for a in sorted(ALLOWED_ACTIONS))


def settings_schema() -> str:
    schema: dict[str, Any] = {
        "voice": {
            "language": "string",
            "tts_engine": "string (piper|kokoro|chatterbox|qwen|omnivoice|...)",
            "voice": "string",
            "voice_config": "object",
            "preferred_speed_limit": "float (e.g. 1.35)",
            "hard_speed_limit": "float (e.g. 2.5)",
        },
        "timing": {
            "guard_gap_ms": "int",
            "compress_internal_pauses": "bool",
            "internal_pause_keep_ms": "int",
            "elastic_timing": "object (see DubbingProjectSettings)",
            "tempo_smoothing": "object",
        },
        "mix": {"ducking": "object"},
        "export": {
            "container": "mp4|mkv",
            "include_narration_only": "bool",
            "embed_subtitles": "bool",
            "audio_bitrate": "string",
        },
    }
    return json.dumps(schema, indent=2)


def quality_model_text() -> str:
    weights = QualityWeights().to_dict()
    return (
        "Score = sum of weighted penalties. Lower is better.\n\n"
        "Components:\n"
        "  overflow_ms * overflow_weight\n"
        "  overlap_ms * overlap_weight\n"
        "  (speed - preferred) * preferred_speed_excess   (per cue)\n"
        "  hard_limit_violation * hard_speed_violation\n"
        "  absolute_shift_ms * absolute_shift_ms\n"
        "  tempo_jump * tempo_weight\n"
        "  transcript_error * transcript_weight\n"
        "  clipping_events * clipping\n"
        "  silence_excess_ms * silence_excess_ms\n"
        "  regeneration_count * regeneration_cost_per_cue\n"
        "  guard_gap_violation_ms * guard_gap_violation_ms\n"
        "  boundary_violation * boundary_violation\n"
        "  missing_artifacts * missing_artifact\n\n"
        "Default weights:\n" + json.dumps(weights, indent=2)
    )


def scenario_schema() -> str:
    return json.dumps(
        {
            "name": "string",
            "project_id": "string",
            "steps": [
                {
                    "action": "facade_method_name",
                    "arguments": {"key": "value"},
                    "wait_for_event": {"event": "subprocess.started", "where": {"stage": "muxing"}},
                    "assert": {"job_status": "cancelled", "final_video_preserved": True},
                }
            ],
        },
        indent=2,
    )
