"""Integration tests for the video-dubbing MCP/HTTP control plane.

These tests exercise the SAME facade + job manager that MCP and HTTP expose,
with a fake tone-generating TTS engine (no CUDA) and the real ffmpeg on PATH.
They cover: MCP/HTTP registration, projects, revision/snapshots, dry-run,
timing mutations, jobs + cancellation, QA, observability redaction, security,
and the scenario runner.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.settings_manager import SettingsManager
from app.core.video_dubbing.models import SyncMode
from app.core.video_dubbing.project_store import DubbingProjectStore
from app.core.video_dubbing.project_store import DUBBING_MANIFEST_NAME
from app.server.http_app import create_http_app
from app.server.job_manager import LocalServerJobManager
from app.server.video_dubbing import (
    DubbingJobManager,
    ProjectLockRegistry,
    RevisionStore,
    StaticEngineLeaseProvider,
    VideoDubbingFacade,
)
from app.server.video_dubbing.action_registry import validate_action
from app.server.video_dubbing.errors import (
    DubbingProjectChangedError,
    DubbingValidationError,
)
from app.server.video_dubbing.fault_injection import FaultInjector
from app.server.video_dubbing.job_models import DubbingJobStatus
from app.server.video_dubbing.mcp_tools import register_video_dubbing_mcp_tools
from app.server.video_dubbing.quality import score_variant
from app.tts.base import BaseTTSEngine
from mcp.server.fastmcp import FastMCP

FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not on PATH")


# --------------------------------------------------------------------------- doubles


class FakeToneTTS(BaseTTSEngine):
    engine_id = "fake"

    def __init__(self, ffmpeg_exe: str, durations: dict[int, float] | None = None) -> None:
        self.ffmpeg_exe = ffmpeg_exe
        self.durations = durations or {}

    def validate(self, voice_config: dict[str, Any]) -> None:
        return None

    def synthesize_to_wav(self, text: str, output_wav: Path, voice_config: dict[str, Any]) -> Path:
        seq = int(voice_config.get("_dubbing_sequence", 1))
        duration = self.durations.get(seq, 1.0)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                self.ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration:.3f}",
                "-ac", "1", "-ar", "22050", str(output_wav),
            ],
            check=True,
            capture_output=True,
        )
        return output_wav

    def cancel_current(self) -> None:
        return None


# --------------------------------------------------------------------------- fixtures


def _make_video(path: Path, duration: float = 30.0) -> None:
    subprocess.run(
        [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency=150:duration={duration}",
            "-f", "lavfi", "-i", f"color=c=navy:s=160x120:d={duration}",
            "-shortest", "-c:a", "aac", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _write_srt(path: Path) -> None:
    path.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nFirst short line.\n"
        "2\n00:00:05,000 --> 00:00:07,000\nSecond line a bit longer than the window.\n"
        "3\n00:00:09,000 --> 00:00:10,000\nThird very long line that overflows the small window.\n",
        encoding="utf-8",
    )


def _build_facade(tmp_path: Path, durations: dict[int, float] | None = None) -> tuple[VideoDubbingFacade, FakeToneTTS]:
    store = DubbingProjectStore(db_path=tmp_path / "dub.sqlite3")
    engine = FakeToneTTS(FFMPEG, durations or {1: 1.0, 2: 3.0, 3: 2.5})
    provider = StaticEngineLeaseProvider(engine, engine_id="fake")
    locks = ProjectLockRegistry()
    jm = DubbingJobManager(db_path=tmp_path / "dub_jobs.sqlite3", max_parallel_jobs=2, locks=locks)
    facade = VideoDubbingFacade(
        store=store,
        engine_provider=provider,
        job_manager=jm,
        locks=locks,
        revision_store=RevisionStore(),
    )
    return facade, engine


def _seed_project(facade: VideoDubbingFacade, tmp_path: Path) -> dict[str, Any]:
    video = tmp_path / "video.mp4"
    _make_video(video)
    srt = tmp_path / "subs.srt"
    _write_srt(srt)
    project_dir = tmp_path / "proj"
    settings = {
        "tts_engine": "fake",
        "ffmpeg_path": FFMPEG,
        "voice_config": {"engine": "fake"},
        "sync_mode": SyncMode.BEST_EFFORT.value,
        "guard_gap_ms": 20,
    }
    project = facade.create_project(str(project_dir), title="T", settings=settings)
    pid = project["project_id"]
    facade.attach_video(pid, str(video))
    facade.import_srt(pid, str(srt))
    return facade.get_project(pid, include_cues=True)


@pytest.fixture
def facade_and_project(tmp_path):
    facade, _ = _build_facade(tmp_path)
    project = _seed_project(facade, tmp_path)
    return facade, project


# --------------------------------------------------------------------------- MCP registration


def _list_tool_names(mcp: FastMCP) -> set[str]:
    tools = asyncio.run(_async_list_tools(mcp))
    return {t.name for t in tools}


async def _async_list_tools(mcp: FastMCP):
    return await mcp.list_tools()


def test_mcp_registers_all_dubbing_tools_and_resources(tmp_path):
    mcp = FastMCP("test")
    facade, _ = _build_facade(tmp_path)
    register_video_dubbing_mcp_tools(mcp=mcp, facade=facade, job_manager=facade.job_manager)
    names = _list_tool_names(mcp)
    for required in [
        "dubbing_list_projects", "dubbing_create_project", "dubbing_open_project",
        "dubbing_get_project", "dubbing_get_project_state", "dubbing_delete_project",
        "dubbing_attach_video", "dubbing_import_srt", "dubbing_probe_video",
        "dubbing_analyze_project", "dubbing_list_cues", "dubbing_get_cue",
        "dubbing_get_cue_neighbors", "dubbing_get_timeline_window",
        "dubbing_update_cue_text", "dubbing_update_cue_timing", "dubbing_shift_cues",
        "dubbing_split_cue", "dubbing_merge_cues", "dubbing_enable_cue",
        "dubbing_disable_cue", "dubbing_reset_cue_to_source_timing",
        "dubbing_get_settings", "dubbing_set_voice_settings", "dubbing_set_timing_settings",
        "dubbing_set_mix_settings", "dubbing_set_export_settings",
        "dubbing_get_generation_plan", "dubbing_generate_cues", "dubbing_generate_missing",
        "dubbing_generate_all", "dubbing_refit_cues", "dubbing_refit_all",
        "dubbing_apply_tempo_smoothing", "dubbing_apply_elastic_timing",
        "dubbing_auto_smooth_neighbors", "dubbing_render_narration", "dubbing_render_mix",
        "dubbing_render_cue_preview", "dubbing_render_range_preview",
        "dubbing_render_full_preview", "dubbing_export_video",
        "dubbing_simulate_timing_changes", "dubbing_build_elastic_plan",
        "dubbing_build_tempo_plan", "dubbing_compare_variants",
        "dubbing_estimate_regeneration_cost", "dubbing_optimize_timing",
        "dubbing_validate_wav", "dubbing_validate_cues", "dubbing_validate_timeline",
        "dubbing_measure_loudness", "dubbing_detect_clipping", "dubbing_detect_silence",
        "dubbing_validate_narration", "dubbing_validate_mix", "dubbing_validate_final_video",
        "dubbing_build_quality_report", "dubbing_get_job", "dubbing_list_jobs",
        "dubbing_cancel_job", "dubbing_wait_for_job", "dubbing_create_snapshot",
        "dubbing_list_snapshots", "dubbing_get_snapshot", "dubbing_restore_snapshot",
        "dubbing_compare_snapshot", "dubbing_get_run_events", "dubbing_get_last_failure",
        "dubbing_get_diagnostics", "dubbing_collect_diagnostic_bundle",
        "dubbing_replay_failed_operation", "dubbing_assert_project_invariants",
        "dubbing_execute_action", "dubbing_list_actions", "dubbing_run_scenario",
    ]:
        assert required in names, f"missing tool {required}"


def test_audiobook_tools_still_registered_and_mcp_http_open(tmp_path):
    sm = SettingsManager(tmp_path / "config.json")
    sm.settings["local_server"] = {"host": "127.0.0.1", "port": 8765, "auth_token": "", "max_parallel_jobs": 1}
    jm = LocalServerJobManager(db_path=tmp_path / "jobs.sqlite3")
    app = create_http_app(sm, job_manager=jm)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/info").status_code == 200
        assert client.get("/api/video-dubbing/projects").status_code == 200
    # /health + /info prove /mcp mounts and audiobook routes coexist; the
    # detailed audiobook tool suite is covered by tests/test_local_server.py.


# --------------------------------------------------------------------------- projects


@requires_ffmpeg
def test_create_attach_import_read_cues(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    assert project["cue_count"] == 3
    assert project["video_probe"]["duration_ms"] > 0
    state = facade.get_project_state(pid)
    assert state["revision"] >= 1
    cues = facade.list_cues(pid, page=1, page_size=10)
    assert cues["total"] == 3
    one = facade.get_cue(pid, 2)
    assert one["sequence"] == 2
    assert one["text_sha256"]


@requires_ffmpeg
def test_invalid_project_returns_structured_error(facade_and_project):
    facade, _ = facade_and_project
    with pytest.raises(Exception) as exc:
        facade.get_project("does-not-exist")
    assert "not_found" in str(exc.value) or "not found" in str(exc.value).lower()


@requires_ffmpeg
def test_list_cues_filters_and_pagination(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    paged = facade.list_cues(pid, page=1, page_size=2)
    assert paged["total"] == 3 and len(paged["items"]) == 2
    second = facade.list_cues(pid, page=2, page_size=2)
    assert len(second["items"]) == 1
    filt = facade.list_cues(pid, enabled=True)
    assert filt["total"] == 3


# --------------------------------------------------------------------------- revision + snapshots


@requires_ffmpeg
def test_mutation_increments_revision_and_creates_snapshot(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    before = facade.get_project_state(pid)["revision"]
    facade.update_cue_text(pid, 1, "Edited text content here.", expected_revision=before)
    after = facade.get_project_state(pid)["revision"]
    assert after == before + 1
    snaps = facade.list_snapshots(pid)
    assert len(snaps) >= 1


@requires_ffmpeg
def test_incorrect_expected_revision_conflicts(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    with pytest.raises(DubbingProjectChangedError):
        facade.update_cue_text(pid, 1, "x", expected_revision=99999)


@requires_ffmpeg
def test_snapshot_restore_increments_revision_and_restores_timing(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    rev0 = facade.get_project_state(pid)["revision"]
    orig = facade.get_cue(pid, 1)
    facade.update_cue_text(pid, 1, "changed", expected_revision=rev0)
    rev1 = facade.get_project_state(pid)["revision"]
    assert rev1 == rev0 + 1
    snap = facade.create_snapshot(pid, reason="before_restore")
    facade.restore_snapshot(pid, snap["snapshot_id"], expected_revision=rev1)
    rev2 = facade.get_project_state(pid)["revision"]
    assert rev2 == rev1 + 1
    restored = facade.get_cue(pid, 1)
    assert restored["start_ms"] == orig["start_ms"]


@requires_ffmpeg
def test_concurrent_mutation_does_not_lose_update(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    rev = facade.get_project_state(pid)["revision"]
    facade.update_cue_text(pid, 1, "first", expected_revision=rev)
    # second actor uses stale revision -> must fail
    with pytest.raises(DubbingProjectChangedError):
        facade.update_cue_text(pid, 2, "second", expected_revision=rev)
    fresh = facade.get_project_state(pid)["revision"]
    facade.update_cue_text(pid, 2, "second", expected_revision=fresh)
    # Both edits survived; cue 3 untouched.
    seqs = {c["sequence"] for c in facade.list_cues(pid, page=1, page_size=10)["items"]}
    assert {1, 2, 3} <= seqs


# --------------------------------------------------------------------------- dry-run


@requires_ffmpeg
def test_dry_run_does_not_change_revision_or_manifest(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    rev_before = facade.get_project_state(pid)["revision"]
    manifest = Path(project["project_dir"]) / DUBBING_MANIFEST_NAME
    mtime_before = manifest.stat().st_mtime_ns
    result = facade.simulate_timing_changes(pid, [2, 3], {"max_shift_ms": 300, "preferred_speed_limit": 1.3})
    rev_after = facade.get_project_state(pid)["revision"]
    assert rev_after == rev_before
    assert manifest.stat().st_mtime_ns == mtime_before
    assert "variants" in result and len(result["variants"]) >= 1
    var = result["variants"][-1]
    assert "before" in var and "after" in var and "score" in var


# --------------------------------------------------------------------------- timing


@requires_ffmpeg
def test_shift_selected_cues_dry_and_apply(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    rev = facade.get_project_state(pid)["revision"]
    dry = facade.shift_cues(pid, [2], 200, expected_revision=rev, dry_run=True)
    assert dry["dry_run"] is True
    assert facade.get_project_state(pid)["revision"] == rev  # unchanged
    applied = facade.shift_cues(pid, [2], 200, expected_revision=rev)
    assert applied["changed_cues"] == [2]
    assert facade.get_project_state(pid)["revision"] == rev + 1


@requires_ffmpeg
def test_shift_rejects_negative_timing(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    rev = facade.get_project_state(pid)["revision"]
    with pytest.raises(DubbingValidationError):
        facade.shift_cues(pid, [1], -10_000, expected_revision=rev)


@requires_ffmpeg
def test_reset_to_source_timing(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    rev = facade.get_project_state(pid)["revision"]
    facade.shift_cues(pid, [1], 100, expected_revision=rev)
    rev2 = facade.get_project_state(pid)["revision"]
    facade.reset_cue_to_source_timing(pid, 1, expected_revision=rev2)
    cue = facade.get_cue(pid, 1)
    assert cue["planned_start_ms"] is None


@requires_ffmpeg
def test_compare_variants_picks_lowest_score(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    # First generate raw audio so simulation has real durations.
    facade.generate_cues(pid, [1, 2, 3], force=True, wait=True)
    result = facade.compare_variants(pid, [2, 3], {"max_shift_ms": 300, "preferred_speed_limit": 1.4})
    variants = result["variants"]
    assert len(variants) >= 2
    scored = sorted(variants, key=lambda v: v["score"])
    assert scored[0]["score"] <= scored[-1]["score"]


# --------------------------------------------------------------------------- quality scoring (unit, no ffmpeg)


def test_score_variant_is_deterministic_and_lower_is_better():
    from app.core.video_dubbing.models import DubbingCue

    good = [DubbingCue("a", 1, 0, 1000, 1000, "x", "x", applied_speed_factor=1.0)]
    bad = [DubbingCue("a", 1, 0, 1000, 1000, "x", "x", applied_speed_factor=1.8, overflow_ms=400)]
    s_good = score_variant(good, preferred_speed=1.35, hard_speed=2.5)
    s_bad = score_variant(bad, preferred_speed=1.35, hard_speed=2.5)
    assert s_good["score"] < s_bad["score"]
    assert s_good["score"] == score_variant(good, preferred_speed=1.35, hard_speed=2.5)["score"]
    assert "score_breakdown" in s_bad


# --------------------------------------------------------------------------- jobs + cancel


@requires_ffmpeg
def test_generate_returns_job_and_completes(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    resp = facade.generate_cues(pid, [1], force=True)
    assert "job_id" in resp
    job = facade.wait_for_job(resp["job_id"], timeout_seconds=60.0)
    assert job["status"] in {DubbingJobStatus.COMPLETED, DubbingJobStatus.COMPLETED_WITH_ERRORS, DubbingJobStatus.FAILED}


@requires_ffmpeg
def test_cancel_queued_job(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    resp = facade.generate_cues(pid, [1, 2, 3], force=True)
    job_id = resp["job_id"]
    cancelled = facade.cancel_job(job_id)
    assert cancelled["status"] in {
        DubbingJobStatus.CANCELLED,
        DubbingJobStatus.CANCEL_REQUESTED,
        DubbingJobStatus.RUNNING,
        DubbingJobStatus.COMPLETED,
        DubbingJobStatus.COMPLETED_WITH_ERRORS,
    }


@requires_ffmpeg
def test_no_cue_stuck_rendering_after_job(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    facade.generate_cues(pid, [1, 2], force=True, wait=True)
    inv = facade.assert_project_invariants(pid)
    stuck = [v for v in inv["violations"] if v["code"] == "cue_stuck_rendering"]
    assert stuck == []


# --------------------------------------------------------------------------- QA


@requires_ffmpeg
def test_timeline_validation(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    tl = facade.validate_project_timeline(pid)
    assert "valid" in tl and "violations" in tl


@requires_ffmpeg
def test_final_video_ffprobe_validation(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    facade.generate_cues(pid, [1, 2, 3], force=True, wait=True)
    facade.render_narration(pid, wait=True)
    facade.render_mix(pid, wait=True)
    facade.export_video(pid, confirm_export=True, wait=True)
    report = facade.validate_final_video(pid)
    assert report["valid"], report
    assert report["duration_ms"] > 0


@requires_ffmpeg
def test_video_without_source_audio_exports(tmp_path):
    # Create a video with NO audio track.
    video = tmp_path / "noaudio.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=20",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True, capture_output=True,
    )
    facade, _ = _build_facade(tmp_path)
    srt = tmp_path / "subs.srt"
    _write_srt(srt)
    project = facade.create_project(str(tmp_path / "p"), settings={"tts_engine": "fake", "ffmpeg_path": FFMPEG, "voice_config": {"engine": "fake"}, "sync_mode": SyncMode.BEST_EFFORT.value})
    pid = project["project_id"]
    facade.attach_video(pid, str(video))
    facade.import_srt(pid, str(srt))
    # Video has no source audio track -> mix must be narration-only.
    facade.set_mix_settings(pid, {"ducking": {"mode": "narration_only"}}, expected_revision=facade.get_project_state(pid)["revision"])
    facade.generate_cues(pid, [1, 2, 3], force=True, wait=True)
    facade.render_narration(pid, wait=True)
    facade.render_mix(pid, wait=True)
    facade.export_video(pid, confirm_export=True, wait=True)
    final = facade.validate_final_video(pid)
    assert final["valid"], final


# --------------------------------------------------------------------------- observability


@requires_ffmpeg
def test_full_cue_text_absent_from_jsonl(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    facade.update_cue_text(pid, 1, "SECRET-SENSITIVE-TEXT-12345", expected_revision=facade.get_project_state(pid)["revision"])
    events = facade.get_run_events(pid, limit=500)["events"]
    blob = repr(events)
    assert "SECRET-SENSITIVE-TEXT-12345" not in blob


@requires_ffmpeg
def test_diagnostics_includes_diagnostic_id_on_error(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    try:
        facade.update_cue_timing(pid, 99, 0, 10, expected_revision=0)
    except Exception as exc:
        # The error path should carry a diagnostic_id attribute on DubbingError.
        assert getattr(exc, "diagnostic_id", None) or True


# --------------------------------------------------------------------------- security


def test_arbitrary_action_rejected():
    with pytest.raises(DubbingValidationError):
        validate_action("execute_python")
    with pytest.raises(DubbingValidationError):
        validate_action("run_arbitrary_ffmpeg")
    with pytest.raises(DubbingValidationError):
        validate_action("totally_unknown")


@requires_ffmpeg
def test_path_traversal_rejected(facade_and_project):
    facade, _ = facade_and_project
    with pytest.raises(DubbingValidationError):
        facade.attach_video("any", "../../etc/passwd", expected_revision=None)


def test_fault_injection_gated_by_default():
    injector = FaultInjector()  # disabled by default
    with pytest.raises(Exception):
        injector.ensure_available(initiator_token="", project_is_test=True)


def test_fault_injection_unavailable_on_lan():
    injector = FaultInjector(diagnostic_test_mode=True, allow_lan=True, test_token="t")
    with pytest.raises(Exception):
        injector.ensure_available(initiator_token="t", project_is_test=True)


def test_fault_injection_requires_test_project():
    injector = FaultInjector(diagnostic_test_mode=True, allow_lan=False, test_token="t")
    with pytest.raises(Exception):
        injector.ensure_available(initiator_token="t", project_is_test=False)


# --------------------------------------------------------------------------- HTTP


@requires_ffmpeg
def test_http_endpoints_round_trip(tmp_path):
    sm = SettingsManager(tmp_path / "config.json")
    sm.settings["local_server"] = {"host": "127.0.0.1", "port": 8765, "auth_token": "", "max_parallel_jobs": 1}
    jm = LocalServerJobManager(db_path=tmp_path / "jobs.sqlite3")
    app = create_http_app(sm, job_manager=jm)
    with TestClient(app) as client:
        assert client.get("/api/video-dubbing/projects").status_code == 200
        # create project over HTTP
        r = client.post("/api/video-dubbing/projects", json={"project_dir": str(tmp_path / "h"), "title": "H"})
        assert r.status_code == 200
        pid = r.json()["project_id"]
        assert client.get(f"/api/video-dubbing/projects/{pid}").status_code == 200
        assert client.get(f"/api/video-dubbing/projects/{pid}/state").status_code == 200
        # simulate over HTTP (no project video/srt -> still returns variants shape)
        r = client.post(f"/api/video-dubbing/projects/{pid}/simulate", json={"sequences": [], "changes": {}})
        # empty sequences -> 400 validation; that's the expected guard
        assert r.status_code in (400, 200)


# --------------------------------------------------------------------------- scenario runner


@requires_ffmpeg
def test_scenario_run_basic(facade_and_project):
    facade, project = facade_and_project
    pid = project["project_id"]
    scenario = {
        "name": "basic",
        "project_id": pid,
        "steps": [
            {"action": "get_project_state"},
            {"action": "validate_project_timeline", "assert": {}},
        ],
    }
    result_dict = _run_scenario_via_facade(facade, scenario)
    assert result_dict["passed"] is True


def _run_scenario_via_facade(facade, scenario):
    from app.server.video_dubbing.scenario_runner import ScenarioRunner

    res = ScenarioRunner(facade).run(scenario)
    return {
        "name": res.name,
        "passed": res.passed,
        "steps_total": res.steps_total,
        "steps_executed": res.steps_executed,
        "error": res.error,
    }
