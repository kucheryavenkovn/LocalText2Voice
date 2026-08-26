"""Registration of all video-dubbing MCP tools + resources on an existing
FastMCP instance. No new MCP server is created."""

from __future__ import annotations

from typing import Any

from app.observability import bind, emit_event, new_operation_id

from . import resources
from .action_registry import list_actions, validate_action
from .errors import DubbingError
from .facade import VideoDubbingFacade
from .fault_injection import FaultInjector
from .job_manager import DubbingJobManager
from .scenario_runner import ScenarioRunner


def _emit(name: str, payload: dict[str, Any], *, started: bool) -> None:
    emit_event(
        f"mcp.request.{'started' if started else 'completed'}",
        payload={"tool": name, **payload},
        force_flush=not started,
    )


def _wrap(name: str, fn):  # noqa: ANN001
    """Wrap a facade call with request_id/operation_id + observability."""

    def _call(*args: Any, **kwargs: Any) -> Any:
        op_id = new_operation_id()
        with bind(operation_id=op_id, initiator="mcp"):
            _emit(name, {"operation_id": op_id}, started=True)
            try:
                result = fn(*args, **kwargs)
            except DubbingError as exc:
                _emit(
                    name,
                    {"operation_id": op_id, "error": exc.code, "diagnostic_id": exc.diagnostic_id},
                    started=False,
                )
                raise
            except Exception as exc:  # pragma: no cover - defensive
                _emit(name, {"operation_id": op_id, "error": type(exc).__name__}, started=False)
                raise
            _emit(name, {"operation_id": op_id}, started=False)
            return result

    _call.__name__ = name
    _call.__doc__ = fn.__doc__
    return _call


def register_video_dubbing_mcp_tools(
    mcp: Any,
    facade: VideoDubbingFacade,
    job_manager: DubbingJobManager,
    *,
    fault_injector: FaultInjector | None = None,
) -> None:
    """Register every dubbing_* tool and resource on ``mcp``."""

    fault_injector = fault_injector or FaultInjector()

    # ---------------------------------------------------------------- resources

    @mcp.resource(resources.DOCS_URI)
    def dubbing_docs() -> str:
        """Agent guide: workflow, TTS vs refit, invalidation, revision, jobs."""
        return resources.docs_text()

    @mcp.resource(resources.ACTIONS_URI)
    def dubbing_actions_resource() -> str:
        """Allowlist of domain actions accepted by dubbing_execute_action."""
        return resources.actions_text()

    @mcp.resource(resources.SETTINGS_SCHEMA_URI)
    def dubbing_settings_schema_resource() -> str:
        """Strict settings schema reference."""
        return resources.settings_schema()

    @mcp.resource(resources.QUALITY_MODEL_URI)
    def dubbing_quality_model_resource() -> str:
        """Deterministic quality scoring model and default weights."""
        return resources.quality_model_text()

    @mcp.resource(resources.SCENARIO_SCHEMA_URI)
    def dubbing_scenario_schema_resource() -> str:
        """Scenario runner YAML/JSON schema."""
        return resources.scenario_schema()

    # ---------------------------------------------------------------- projects

    @mcp.tool(description="List video-dubbing projects.")
    def dubbing_list_projects() -> list[dict[str, Any]]:
        return facade.list_projects()

    @mcp.tool(description="Create a new video-dubbing project (no UI needed).")
    def dubbing_create_project(
        project_dir: str,
        title: str = "Video Dubbing",
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return facade.create_project(project_dir, title=title, settings=settings)

    @mcp.tool(description="Open a project by directory (manifest or DB).")
    def dubbing_open_project(project_dir: str) -> dict[str, Any]:
        return facade.open_project(project_dir)

    @mcp.tool(description="Get project metadata. Pass include_cues=true for the full cue list.")
    def dubbing_get_project(project_id: str, include_cues: bool = False) -> dict[str, Any]:
        return facade.get_project(project_id, include_cues=include_cues)

    @mcp.tool(description="Compact pipeline state: revision, stale flags, status counts.")
    def dubbing_get_project_state(project_id: str) -> dict[str, Any]:
        return facade.get_project_state(project_id)

    @mcp.tool(description="Delete a project. Requires confirm=true. Blocked if a job is active.")
    def dubbing_delete_project(project_id: str, confirm: bool = False) -> dict[str, Any]:
        return facade.delete_project(project_id, confirm=confirm)

    # ---------------------------------------------------------------- import

    @mcp.tool(description="Probe a video file with ffprobe without attaching it.")
    def dubbing_probe_video(video_path: str, ffmpeg_path: str = "") -> dict[str, Any]:
        return facade.probe_video(video_path, ffmpeg_path=ffmpeg_path)

    @mcp.tool(description="Attach a source video to the project and probe it.")
    def dubbing_attach_video(
        project_id: str,
        video_path: str,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        return facade.attach_video(project_id, video_path, expected_revision=expected_revision)

    @mcp.tool(description="Import an SRT subtitle file as the cue list.")
    def dubbing_import_srt(
        project_id: str,
        srt_path: str,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        return facade.import_srt(project_id, srt_path, expected_revision=expected_revision)

    @mcp.tool(description="Analyze the project: durations, codecs, cue warnings, stale artifacts.")
    def dubbing_analyze_project(project_id: str) -> dict[str, Any]:
        return facade.analyze_project(project_id)

    # ---------------------------------------------------------------- read cues

    @mcp.tool(description="List cues with filters and pagination. Pass include_text only when needed.")
    def dubbing_list_cues(
        project_id: str,
        page: int = 1,
        page_size: int = 50,
        include_text: bool = False,
        status: str | None = None,
        enabled: bool | None = None,
        has_error: bool | None = None,
        has_overflow: bool | None = None,
        needs_tts: bool | None = None,
        needs_refit: bool | None = None,
        sequence_from: int | None = None,
        sequence_to: int | None = None,
        time_from_ms: int | None = None,
        time_to_ms: int | None = None,
    ) -> dict[str, Any]:
        return facade.list_cues(
            project_id,
            page=page,
            page_size=page_size,
            include_text=include_text,
            status=status,
            enabled=enabled,
            has_error=has_error,
            has_overflow=has_overflow,
            needs_tts=needs_tts,
            needs_refit=needs_refit,
            sequence_from=sequence_from,
            sequence_to=sequence_to,
            time_from_ms=time_from_ms,
            time_to_ms=time_to_ms,
        )

    @mcp.tool(description="Read one cue card. Full text only here, not in list_cues.")
    def dubbing_get_cue(project_id: str, sequence: int, include_text: bool = True) -> dict[str, Any]:
        return facade.get_cue(project_id, sequence, include_text=include_text)

    @mcp.tool(description="Get a cue with its previous/next neighbours.")
    def dubbing_get_cue_neighbors(project_id: str, sequence: int, include_text: bool = False) -> dict[str, Any]:
        return facade.get_cue_neighbors(project_id, sequence, include_text=include_text)

    @mcp.tool(description="Get all cues overlapping a [time_from_ms, time_to_ms] window.")
    def dubbing_get_timeline_window(
        project_id: str,
        time_from_ms: int,
        time_to_ms: int,
        include_text: bool = False,
    ) -> dict[str, Any]:
        return facade.get_timeline_window(
            project_id, time_from_ms=time_from_ms, time_to_ms=time_to_ms, include_text=include_text
        )

    # ---------------------------------------------------------------- cue edit

    @mcp.tool(description="Update cue spoken text. Sets requires_human_review=true.")
    def dubbing_update_cue_text(
        project_id: str,
        sequence: int,
        text: str,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        return facade.update_cue_text(project_id, sequence, text, expected_revision=expected_revision)

    @mcp.tool(description="Update a cue's start/end timing (ms).")
    def dubbing_update_cue_timing(
        project_id: str,
        sequence: int,
        start_ms: int,
        end_ms: int,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        return facade.update_cue_timing(project_id, sequence, start_ms, end_ms, expected_revision=expected_revision)

    @mcp.tool(description="Shift a group of cues by delta_ms. Supports dry_run. Respects right_only elastic mode.")
    def dubbing_shift_cues(
        project_id: str,
        sequences: list[int],
        delta_ms: int,
        mode: str = "absolute_group",
        expected_revision: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return facade.shift_cues(
            project_id, sequences, delta_ms, mode=mode, expected_revision=expected_revision, dry_run=dry_run
        )

    @mcp.tool(description="Split a cue at a character offset. Sets requires_human_review=true.")
    def dubbing_split_cue(
        project_id: str,
        sequence: int,
        split_offset_chars: int,
        new_end_ms: int | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        return facade.split_cue(
            project_id,
            sequence,
            split_offset_chars=split_offset_chars,
            new_end_ms=new_end_ms,
            expected_revision=expected_revision,
        )

    @mcp.tool(description="Merge two neighbouring cues. Sets requires_human_review=true.")
    def dubbing_merge_cues(
        project_id: str,
        first_sequence: int,
        second_sequence: int,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        return facade.merge_cues(
            project_id, first_sequence, second_sequence, expected_revision=expected_revision
        )

    @mcp.tool(description="Enable a cue.")
    def dubbing_enable_cue(project_id: str, sequence: int, expected_revision: int | None = None) -> dict[str, Any]:
        return facade.enable_cue(project_id, sequence, expected_revision=expected_revision)

    @mcp.tool(description="Disable a cue (excluded from narration/export).")
    def dubbing_disable_cue(project_id: str, sequence: int, expected_revision: int | None = None) -> dict[str, Any]:
        return facade.disable_cue(project_id, sequence, expected_revision=expected_revision)

    @mcp.tool(description="Reset planned timing back to the original SRT source timing.")
    def dubbing_reset_cue_to_source_timing(
        project_id: str, sequence: int, expected_revision: int | None = None
    ) -> dict[str, Any]:
        return facade.reset_cue_to_source_timing(project_id, sequence, expected_revision=expected_revision)

    # ---------------------------------------------------------------- settings

    @mcp.tool(description="Read the full DubbingProjectSettings.")
    def dubbing_get_settings(project_id: str) -> dict[str, Any]:
        return facade.get_settings(project_id)

    @mcp.tool(description="Update voice/language/speed settings (invalidates TTS+downstream).")
    def dubbing_set_voice_settings(
        project_id: str, settings: dict[str, Any], expected_revision: int | None = None
    ) -> dict[str, Any]:
        return facade.set_voice_settings(project_id, settings, expected_revision=expected_revision)

    @mcp.tool(description="Update timing/elastic/tempo settings (invalidates narration+downstream).")
    def dubbing_set_timing_settings(
        project_id: str, settings: dict[str, Any], expected_revision: int | None = None
    ) -> dict[str, Any]:
        return facade.set_timing_settings(project_id, settings, expected_revision=expected_revision)

    @mcp.tool(description="Update mix/ducking settings (invalidates mix+downstream).")
    def dubbing_set_mix_settings(
        project_id: str, settings: dict[str, Any], expected_revision: int | None = None
    ) -> dict[str, Any]:
        return facade.set_mix_settings(project_id, settings, expected_revision=expected_revision)

    @mcp.tool(description="Update export container settings (invalidates video).")
    def dubbing_set_export_settings(
        project_id: str, settings: dict[str, Any], expected_revision: int | None = None
    ) -> dict[str, Any]:
        return facade.set_export_settings(project_id, settings, expected_revision=expected_revision)

    # ---------------------------------------------------------------- generation

    @mcp.tool(description="Classify every cue into reuse categories (no TTS, no mutation).")
    def dubbing_get_generation_plan(project_id: str, force: bool = False) -> dict[str, Any]:
        return facade.get_generation_plan(project_id, force=force)

    @mcp.tool(description="Generate selected cues as a job. force=true re-synthesizes; force=false reuses raw audio.")
    def dubbing_generate_cues(
        project_id: str,
        sequences: list[int],
        force: bool = False,
        error_policy: str = "continue",
        wait: bool = False,
    ) -> dict[str, Any]:
        return facade.generate_cues(
            project_id, sequences, force=force, error_policy=error_policy, wait=wait
        )

    @mcp.tool(description="Generate only missing/refit-needed cues (no force).")
    def dubbing_generate_missing(project_id: str, wait: bool = False) -> dict[str, Any]:
        return facade.generate_missing(project_id, wait=wait)

    @mcp.tool(description="Generate all enabled cues. Respects CONTINUE policy when explicitly set.")
    def dubbing_generate_all(
        project_id: str, force: bool = False, error_policy: str = "continue", wait: bool = False
    ) -> dict[str, Any]:
        return facade.generate_all(project_id, force=force, error_policy=error_policy, wait=wait)

    @mcp.tool(description="Re-fit existing raw audio under current speed policy WITHOUT TTS.")
    def dubbing_refit_cues(project_id: str, sequences: list[int] | None = None, wait: bool = False) -> dict[str, Any]:
        return facade.refit_cues(project_id, sequences, wait=wait)

    @mcp.tool(description="Re-fit all cues with intact raw audio (no TTS).")
    def dubbing_refit_all(project_id: str, wait: bool = False) -> dict[str, Any]:
        return facade.refit_all(project_id, wait=wait)

    @mcp.tool(description="Apply tempo smoothing plan.")
    def dubbing_apply_tempo_smoothing(project_id: str, wait: bool = False) -> dict[str, Any]:
        return facade.apply_tempo_smoothing(project_id, wait=wait)

    @mcp.tool(description="Apply elastic timing (borrow right time, common group factor).")
    def dubbing_apply_elastic_timing(project_id: str, wait: bool = False) -> dict[str, Any]:
        return facade.apply_elastic_timing(project_id, wait=wait)

    @mcp.tool(description="Auto-smooth neighbour cues so problematic cues borrow right time.")
    def dubbing_auto_smooth_neighbors(project_id: str, wait: bool = False) -> dict[str, Any]:
        return facade.auto_smooth_neighbors(project_id, wait=wait)

    # ---------------------------------------------------------------- render/export

    @mcp.tool(description="Render the narration track (job).")
    def dubbing_render_narration(project_id: str, wait: bool = False) -> dict[str, Any]:
        return facade.render_narration(project_id, wait=wait)

    @mcp.tool(description="Render the dubbed mix (job).")
    def dubbing_render_mix(project_id: str, wait: bool = False) -> dict[str, Any]:
        return facade.render_mix(project_id, wait=wait)

    @mcp.tool(description="Render a single cue preview (job).")
    def dubbing_render_cue_preview(project_id: str, sequence: int, wait: bool = False) -> dict[str, Any]:
        return facade.render_cue_preview(project_id, sequence, wait=wait)

    @mcp.tool(description="Render previews for a range of cues (job).")
    def dubbing_render_range_preview(project_id: str, sequences: list[int], wait: bool = False) -> dict[str, Any]:
        return facade.render_range_preview(project_id, sequences, wait=wait)

    @mcp.tool(description="Render the full preview video (job).")
    def dubbing_render_full_preview(project_id: str, wait: bool = False) -> dict[str, Any]:
        return facade.render_full_preview(project_id, wait=wait)

    @mcp.tool(description="Export the final video. Requires confirm_export=true.")
    def dubbing_export_video(
        project_id: str,
        confirm_export: bool = False,
        output_path: str | None = None,
        wait: bool = False,
    ) -> dict[str, Any]:
        return facade.export_video(
            project_id, confirm_export=confirm_export, output_path=output_path, wait=wait
        )

    # ---------------------------------------------------------------- simulation

    @mcp.tool(description="Dry-run timing/speed changes and score variants. Never mutates the project.")
    def dubbing_simulate_timing_changes(
        project_id: str, sequences: list[int], changes: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return facade.simulate_timing_changes(project_id, sequences, changes)

    @mcp.tool(description="Build and compare several correction variants; returns scored variants.")
    def dubbing_compare_variants(
        project_id: str, sequences: list[int], changes: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return facade.compare_variants(project_id, sequences, changes)

    @mcp.tool(description="Compute the elastic group plan without applying it.")
    def dubbing_build_elastic_plan(project_id: str, sequences: list[int] | None = None) -> dict[str, Any]:
        return facade.build_elastic_plan(project_id, sequences)

    @mcp.tool(description="Compute the tempo smoothing plan without applying it.")
    def dubbing_build_tempo_plan(project_id: str) -> dict[str, Any]:
        return facade.build_tempo_plan(project_id)

    @mcp.tool(description="Estimate how many cues need TTS vs refit.")
    def dubbing_estimate_regeneration_cost(project_id: str, sequences: list[int]) -> dict[str, Any]:
        return facade.estimate_regeneration_cost(project_id, sequences)

    @mcp.tool(description="Automatic optimizing loop: build variants, score, optionally apply the best.")
    def dubbing_optimize_timing(
        project_id: str,
        sequences: list[int],
        max_iterations: int = 5,
        allowed_actions: list[str] | None = None,
        max_shift_ms: int = 800,
        max_speed_factor: float = 1.45,
        apply: bool = False,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        return facade.optimize_timing(
            project_id,
            sequences,
            max_iterations=max_iterations,
            allowed_actions=allowed_actions,
            max_shift_ms=max_shift_ms,
            max_speed_factor=max_speed_factor,
            apply=apply,
            expected_revision=expected_revision,
        )

    # ---------------------------------------------------------------- QA

    @mcp.tool(description="Validate a WAV file (exists, format, duration, sample rate).")
    def dubbing_validate_wav(path: str) -> dict[str, Any]:
        return facade.validate_wav_path(path)

    @mcp.tool(description="Validate raw+fitted WAV artifacts for cues.")
    def dubbing_validate_cues(project_id: str, sequences: list[int] | None = None) -> dict[str, Any]:
        return facade.validate_cues(project_id, sequences)

    @mcp.tool(description="Validate the whole timeline (overlaps, overflow, gaps, boundaries).")
    def dubbing_validate_timeline(project_id: str) -> dict[str, Any]:
        return facade.validate_project_timeline(project_id)

    @mcp.tool(description="Measure EBU R128 loudness of an artifact (narration|mix|final).")
    def dubbing_measure_loudness(project_id: str, kind: str = "narration") -> dict[str, Any]:
        return facade.measure_loudness(project_id, kind=kind)

    @mcp.tool(description="Detect clipping on an artifact (peak >= threshold).")
    def dubbing_detect_clipping(project_id: str, kind: str = "mix") -> dict[str, Any]:
        return facade.detect_clipping(project_id, kind=kind)

    @mcp.tool(description="Detect long silences in an artifact.")
    def dubbing_detect_silence(project_id: str, kind: str = "narration") -> dict[str, Any]:
        return facade.detect_silence(project_id, kind=kind)

    @mcp.tool(description="Validate the narration WAV.")
    def dubbing_validate_narration(project_id: str) -> dict[str, Any]:
        return facade.validate_narration(project_id)

    @mcp.tool(description="Validate the mix WAV.")
    def dubbing_validate_mix(project_id: str) -> dict[str, Any]:
        return facade.validate_mix(project_id)

    @mcp.tool(description="Validate the final video with ffprobe (streams, duration, container).")
    def dubbing_validate_final_video(project_id: str) -> dict[str, Any]:
        return facade.validate_final_video(project_id)

    @mcp.tool(description="Build a full quality report (timeline + invariants + final video).")
    def dubbing_build_quality_report(project_id: str) -> dict[str, Any]:
        return facade.build_quality_report(project_id)

    # ---------------------------------------------------------------- jobs

    @mcp.tool(description="Get one dubbing job by id.")
    def dubbing_get_job(job_id: str) -> dict[str, Any]:
        return facade.get_job(job_id)

    @mcp.tool(description="List dubbing jobs (optionally per project/status).")
    def dubbing_list_jobs(
        project_id: str | None = None, status: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        return facade.list_jobs(project_id=project_id, status=status, limit=limit)

    @mcp.tool(description="Cancel a queued/running dubbing job (cooperative).")
    def dubbing_cancel_job(job_id: str) -> dict[str, Any]:
        return facade.cancel_job(job_id)

    @mcp.tool(description="Block until a job reaches a terminal status (or timeout).")
    def dubbing_wait_for_job(job_id: str, timeout_seconds: float = 3600.0) -> dict[str, Any]:
        return facade.wait_for_job(job_id, timeout_seconds=timeout_seconds)

    # ---------------------------------------------------------------- snapshots

    @mcp.tool(description="Create a metadata snapshot of the current project state.")
    def dubbing_create_snapshot(
        project_id: str, reason: str = "manual", initiator: str = "mcp"
    ) -> dict[str, Any]:
        return facade.create_snapshot(project_id, reason=reason, initiator=initiator)

    @mcp.tool(description="List metadata snapshots.")
    def dubbing_list_snapshots(project_id: str) -> list[dict[str, Any]]:
        return facade.list_snapshots(project_id)

    @mcp.tool(description="Get one metadata snapshot.")
    def dubbing_get_snapshot(project_id: str, snapshot_id: str) -> dict[str, Any]:
        return facade.get_snapshot(project_id, snapshot_id)

    @mcp.tool(description="Restore a snapshot transactionally (bumps revision).")
    def dubbing_restore_snapshot(
        project_id: str, snapshot_id: str, expected_revision: int | None = None
    ) -> dict[str, Any]:
        return facade.restore_snapshot(project_id, snapshot_id, expected_revision=expected_revision)

    @mcp.tool(description="Compare current state to a snapshot (changed cues).")
    def dubbing_compare_snapshot(project_id: str, snapshot_id: str) -> dict[str, Any]:
        return facade.compare_snapshot(project_id, snapshot_id)

    # ---------------------------------------------------------------- diagnostics

    @mcp.tool(description="Read JSONL events for the latest (or a specific) run.")
    def dubbing_get_run_events(
        project_id: str, run_id: str | None = None, limit: int = 200
    ) -> dict[str, Any]:
        return facade.get_run_events(project_id, run_id=run_id, limit=limit)

    @mcp.tool(description="Classify the most recent failure (deterministic, evidence-based).")
    def dubbing_get_last_failure(project_id: str) -> dict[str, Any]:
        return facade.get_last_failure(project_id)

    @mcp.tool(description="Return a diagnostics summary (revision, jobs, invariants, last failure).")
    def dubbing_get_diagnostics(project_id: str) -> dict[str, Any]:
        return facade.get_diagnostics(project_id)

    @mcp.tool(description="Collect a diagnostic bundle for support (events, manifest, failure).")
    def dubbing_collect_diagnostic_bundle(project_id: str) -> dict[str, Any]:
        return facade.collect_diagnostic_bundle(project_id)

    @mcp.tool(description="Re-run the last failed operation under the same run context.")
    def dubbing_replay_failed_operation(project_id: str) -> dict[str, Any]:
        return facade.replay_failed_operation(project_id)

    @mcp.tool(description="Run the project invariant suite (spec section 20).")
    def dubbing_assert_project_invariants(project_id: str) -> dict[str, Any]:
        return facade.assert_project_invariants(project_id)

    # ---------------------------------------------------------------- dispatch

    @mcp.tool(description="Dispatch a single allowlisted domain action. Rejects arbitrary code/ffmpeg.")
    def dubbing_execute_action(
        project_id: str,
        action: str,
        arguments: dict[str, Any] | None = None,
        dry_run: bool = False,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        action = validate_action(action)
        arguments = dict(arguments or {})
        arguments["expected_revision"] = expected_revision
        if dry_run:
            return {"action": action, "dry_run": True, "arguments": arguments}
        method = getattr(facade, action, None)
        if not callable(method):
            return {"error": f"action not implemented: {action}"}
        arguments.pop("project_id", None)
        result = method(project_id=project_id, **arguments)
        return result if isinstance(result, dict) else {"value": result}

    @mcp.tool(description="List the actions accepted by dubbing_execute_action.")
    def dubbing_list_actions() -> list[dict[str, str]]:
        return list_actions()

    @mcp.tool(description="Run a scenario (YAML/JSON) against a project and return assertions.")
    def dubbing_run_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
        runner = ScenarioRunner(facade)
        result = runner.run(scenario)
        return {
            "name": result.name,
            "passed": result.passed,
            "steps_total": result.steps_total,
            "steps_executed": result.steps_executed,
            "assertions": result.assertions,
            "error": result.error,
            "job_ids": result.job_ids,
        }
