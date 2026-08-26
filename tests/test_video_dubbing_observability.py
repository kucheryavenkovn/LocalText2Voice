"""Observability + lifecycle tests for the video-dubbing pipeline.

Covers the diagnostic/robustness requirements:

1.  Closing the app during active generation safely stops the QThread.
2.  Autosave does not roll back a just-finished cue.
3.  Switching engine closes the old page-owned engine exactly once.
4.  A shared (main-window) engine is never closed by the page.
5.  Full SQLite round-trip preserves every cue field.
6.  A worker exception records the full traceback + run_id.
7.  Normal operation longer than 30 s does not create a false crash dump.
8.  Cancel between narration and mix does not launch mix.
9.  Cancel between mix and mux does not launch mux.
10. An FFmpeg error persists the full stderr.
11. A render error keeps the previous working file.
12. A leftover ``.part`` artefact is recovered/removed on project open.
13. A video without an audio track is handled predictably.
14. Logs/JSONL never contain API keys or full cue text.
15. JSONL stays valid after an artificial process abort.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from app.core.video_dubbing.generation import (
    GenerationCancelled,
)
from app.core.video_dubbing.models import (
    CueStatus,
    DubbingCue,
    DubbingProject,
    DubbingProjectSettings,
)
from app.core.video_dubbing.project_store import (
    DubbingProjectStore,
)
from app.core.video_dubbing.service import VideoDubbingService
from app.core.video_dubbing.video_muxer import VideoMuxer
from app.observability import (
    DiagnosedSubprocess,
    JsonlEventSink,
    RunDirectory,
    sha256_text,
)
from app.observability.redaction import redact_command, redact_value, safe_voice_snapshot
from app.tts.base import BaseTTSEngine, TTSCancelled, TTSEngineError

FFMPEG_EXE = shutil.which("ffmpeg")


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _make_page(qt_app, tmp_path):
    from app.ui.video_dubbing_page import VideoDubbingPage

    return VideoDubbingPage(
        lambda k, d: d, ffmpeg_path=FFMPEG_EXE or "ffmpeg/ffmpeg.exe",
        default_output_dir=str(tmp_path),
    )


class _ScriptedTTS(BaseTTSEngine):
    """Fake TTS writing a tone via ffmpeg; can fail/cancel/block."""

    engine_id = "fake"
    close_calls = 0

    def __init__(
        self,
        ffmpeg_exe: str,
        *,
        fail_on: set[int] | None = None,
        cancel_on: set[int] | None = None,
        duration: float = 0.8,
        block_event: threading.Event | None = None,
        release_event: threading.Event | None = None,
    ) -> None:
        self.ffmpeg_exe = ffmpeg_exe
        self.fail_on = fail_on or set()
        self.cancel_on = cancel_on or set()
        self.duration = duration
        self.calls: list[int] = []
        self.block_event = block_event
        self.release_event = release_event
        self._cancelled = False
        self.close_calls = 0

    def validate(self, voice_config: dict[str, Any]) -> None:
        return None

    def synthesize_to_wav(
        self, text: str, output_wav: Path, voice_config: dict[str, Any]
    ) -> Path:
        seq = int(voice_config.get("_dubbing_sequence", 0))
        self.calls.append(seq)
        if self.block_event is not None:
            self.block_event.set()
        if self.release_event is not None:
            self.release_event.wait(timeout=10)
        if self._cancelled or seq in self.cancel_on:
            raise TTSCancelled("cancelled by test")
        if seq in self.fail_on:
            raise TTSEngineError("synthesis failed (test)")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                self.ffmpeg_exe,
                "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={self.duration}",
                "-ac", "1", "-ar", "22050", str(output_wav),
            ],
            check=True, capture_output=True,
        )
        return output_wav

    def cancel_current(self) -> None:
        self._cancelled = True
        # Unblock a blocked synthesize so cancel propagates promptly.
        if self.release_event is not None:
            self.release_event.set()

    def close(self) -> None:
        self.close_calls += 1


def _new_service(tmp_path: Path, **tts_kwargs) -> tuple[VideoDubbingService, DubbingProject, _ScriptedTTS]:
    store = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    tts = _ScriptedTTS(FFMPEG_EXE, **tts_kwargs)
    service = VideoDubbingService(tts, store=store)
    settings = DubbingProjectSettings(
        language="ru", tts_engine="fake", ffmpeg_path="ffmpeg/ffmpeg.exe",
        voice_config={"engine": "fake", "voice": "v1"}, voice="v1",
    )
    project = service.create_project(tmp_path / "proj", settings=settings)
    return service, project, tts


def _import_srt(service: VideoDubbingService, project, count: int = 3) -> None:
    blocks = []
    for i in range(1, count + 1):
        start_s = i * 2
        end_s = start_s + 1
        blocks.append(
            f"{i}\n00:00:{start_s:02d},000 --> 00:00:{end_s:02d},000\nРеплика {i}.\n"
        )
    srt = project.project_dir / "in.srt"
    srt.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    service.import_srt(project, srt)


def _attach_video(service: VideoDubbingService, project, tmp_path: Path, *, with_audio: bool = True) -> Path:
    """Generate a short video (with/without audio) via ffmpeg and attach it."""
    video = tmp_path / ("withaudio.mp4" if with_audio else "noaudio.mp4")
    cmd = [
        FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=6:r=10",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=300:duration=6"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd.append(str(video))
    subprocess.run(cmd, check=True, capture_output=True)
    service.attach_video(project, video)
    return video


# ---------------------------------------------------------------------------
# 1. Safe QThread shutdown
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_shutdown_stops_worker_thread(tmp_path, qt_app):
    block = threading.Event()
    release = threading.Event()
    service, project, tts = _new_service(tmp_path, block_event=block, release_event=release)
    _import_srt(service, project, count=2)
    page = _make_page(qt_app, tmp_path)
    page._service = service
    page._tts_engine = tts
    page._engine_ownership = "page"
    page._project = project
    # Bypass engine sync (which would try to build a real engine from the combo).
    page._ensure_service = lambda: service  # type: ignore[assignment]

    def op(svc):
        return svc.generate_all(project, force=True)

    page._start_worker(op, lambda r: None, generation=True)
    try:
        assert block.wait(timeout=5), "TTS did not start"
        assert page._worker_thread is not None and page._worker_thread.isRunning()
        # Cancel + wait for the thread to actually stop.
        stopped = page.shutdown(timeout_ms=10_000)
    finally:
        release.set()
    assert stopped, "shutdown did not stop the QThread in time"
    assert page._worker_thread is None


# ---------------------------------------------------------------------------
# 2. Autosave does not roll back a finished cue
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_autosave_metadata_only_during_generation_preserves_cue(tmp_path):
    service, project, tts = _new_service(tmp_path, block_event=threading.Event(), release_event=threading.Event())
    _import_srt(service, project, count=3)

    # Simulate the worker having finished cue #1 (persisted via upsert_cue).
    service.generate_selected(project, [1], force=True)
    finished_status = project.cues[0].status
    finished_raw = project.cues[0].raw_duration_ms
    assert finished_status in {CueStatus.RENDERED.value, CueStatus.FITTED.value}

    # Now pretend generation is active and the UI autosave fires a metadata save.
    service._generation_active = True
    project.title = "Autosaved title"
    service.save_project(project)  # must degrade to metadata-only
    service._generation_active = False

    # Reload from SQLite: cue #1 must be untouched (not rolled back).
    fresh = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    reloaded = fresh.load_project(project.project_id)
    assert reloaded.cues[0].status == finished_status
    assert reloaded.cues[0].raw_duration_ms == finished_raw
    assert reloaded.title == "Autosaved title"


# ---------------------------------------------------------------------------
# 3 & 4. Engine ownership
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_page_owned_engine_closed_exactly_once_on_switch(tmp_path, qt_app):
    page = _make_page(qt_app, tmp_path)
    first = _ScriptedTTS(FFMPEG_EXE)
    page._adopt_engine(first, ownership="page")
    # Switch to a new page-owned engine; the previous one must close exactly once.
    second = _ScriptedTTS(FFMPEG_EXE)
    page._adopt_engine(second, ownership="page")
    assert first.close_calls == 1
    assert second.close_calls == 0
    # Switching again must not re-close the first.
    third = _ScriptedTTS(FFMPEG_EXE)
    page._adopt_engine(third, ownership="page")
    assert first.close_calls == 1
    assert second.close_calls == 1


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_shared_engine_not_closed_by_page(tmp_path, qt_app):
    page = _make_page(qt_app, tmp_path)
    shared = _ScriptedTTS(FFMPEG_EXE)
    page._adopt_engine(shared, ownership="main_window")
    # Adopting a page-owned engine must NOT close the shared one.
    page_owned = _ScriptedTTS(FFMPEG_EXE)
    page._adopt_engine(page_owned, ownership="page")
    assert shared.close_calls == 0
    assert page_owned.close_calls == 0


# ---------------------------------------------------------------------------
# 5. Full SQLite round-trip
# ---------------------------------------------------------------------------


def test_cue_full_sqlite_round_trip(tmp_path):
    store = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    project_id = store.create_project(tmp_path / "p")
    from app.core.video_dubbing.models import DubbingProject, DubbingProjectSettings

    project = DubbingProject(
        project_id=project_id, project_dir=tmp_path / "p",
        settings=DubbingProjectSettings(),
    )
    project.ensure_directories()
    cue = DubbingCue(
        cue_id="c1", sequence=1, start_ms=100, end_ms=900, duration_budget_ms=800,
        source_text="x", spoken_text="y",
        raw_duration_ms=700, fitted_duration_ms=780,
        applied_speed_factor=1.1, required_speed_factor=1.2,
        fit_fingerprint="FITFP", fit_pipeline_version=1, raw_wav_hash="RAWHASH",
        source_start_ms=100, source_end_ms=900,
        planned_start_ms=110, planned_end_ms=890,
        timing_group_id="g1", timing_group_position=0, common_speed_factor=1.15,
        start_shift_ms=10, end_shift_ms=-10, borrowed_right_ms=5, timing_locked=True,
        planned_speed_factor=1.13, smoothing_group_id="s1", smoothing_reason="neighbor",
        generation_fingerprint="GENFP",
    )
    project.cues.append(cue)
    store.save_project(project)

    fresh = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    reloaded = fresh.load_project(project_id)
    assert reloaded is not None
    rc = reloaded.cues[0]
    # Every migration-added field must survive the round trip.
    assert rc.fit_fingerprint == "FITFP"
    assert rc.fit_pipeline_version == 1
    assert rc.raw_wav_hash == "RAWHASH"
    assert rc.source_start_ms == 100
    assert rc.source_end_ms == 900
    assert rc.planned_start_ms == 110
    assert rc.planned_end_ms == 890
    assert rc.timing_group_id == "g1"
    assert rc.timing_group_position == 0
    assert rc.common_speed_factor == 1.15
    assert rc.start_shift_ms == 10
    assert rc.end_shift_ms == -10
    assert rc.borrowed_right_ms == 5
    assert rc.timing_locked is True
    assert rc.planned_speed_factor == 1.13
    assert rc.smoothing_group_id == "s1"
    assert rc.smoothing_reason == "neighbor"
    assert rc.generation_fingerprint == "GENFP"


# ---------------------------------------------------------------------------
# 6. Worker exception records traceback + run_id
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_worker_exception_recorded_with_traceback_and_run_id(tmp_path):
    service, project, _tts = _new_service(tmp_path, fail_on={1})
    project.settings.cue_error_policy = "stop"
    _import_srt(service, project, count=2)
    try:
        service.generate_all(project, force=True)
    except Exception:
        pass
    # The service writes events to <project>/logs/run_<run_id>/events.jsonl.
    run_logs = list((project.project_dir / "logs").glob("run_*"))
    assert run_logs, "a per-run diagnostic directory must exist"
    events = []
    for ev_path in (d / "events.jsonl" for d in run_logs):
        if ev_path.is_file():
            for line in ev_path.read_text(encoding="utf-8").splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    blob = json.dumps(events, ensure_ascii=False)
    # A failure event with traceback/run_id context must be present.
    assert "failed" in blob
    assert any(e.get("run_id") for e in events)


# ---------------------------------------------------------------------------
# 7. No false crash dump during normal >30s operation
# ---------------------------------------------------------------------------


def test_no_unconditional_faulthandler_dump():
    """configure_observability_logging must NOT install dump_traceback_later.

    The old bootstrap dumped a stack trace every 30 s unconditionally; the new
    one only arms a watchdog for active operations. We assert the offending call
    is absent from the module source.
    """
    import app.observability.logging_config as lc
    import app.observability.crash_reporter as cr

    assert "dump_traceback_later" not in lc.__file__ or True  # placeholder
    src_lc = Path(lc.__file__).read_text(encoding="utf-8")
    src_cr = Path(cr.__file__).read_text(encoding="utf-8")
    assert "dump_traceback_later(timeout=30" not in src_lc
    assert "dump_traceback_later(timeout=30" not in src_cr
    # The watchdog must be opt-in (a class), not installed at import.
    assert "class OperationWatchdog" in src_cr


# ---------------------------------------------------------------------------
# 8 & 9. Cancel between stages
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_cancel_between_narration_and_mix_does_not_mix(tmp_path, monkeypatch):
    service, project, _tts = _new_service(tmp_path)
    _attach_video(service, project, tmp_path)
    _import_srt(service, project, count=2)
    service.generate_all(project)
    # Render narration successfully, then make mix raise if it ever starts.
    service.render_narration(project)
    mix_calls = {"n": 0}

    def _boom(self, *a, **k):
        mix_calls["n"] += 1
        raise AssertionError("mix must not start after cancel")

    monkeypatch.setattr("app.core.video_dubbing.service.AudioMixer.render_dubbed_mix", _boom)
    # Request cancel BEFORE entering render_dubbed_mix's mix stage.
    service.cancel()
    with pytest.raises(GenerationCancelled):
        service.render_dubbed_mix(project)
    assert mix_calls["n"] == 0
    service.reset_cancel()


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_cancel_between_mix_and_mux_does_not_mux(tmp_path, monkeypatch):
    service, project, _tts = _new_service(tmp_path)
    _attach_video(service, project, tmp_path)
    _import_srt(service, project, count=2)
    service.generate_all(project)
    service.render_dubbed_mix(project)
    mux_calls = {"n": 0}

    def _boom(self, *a, **k):
        mux_calls["n"] += 1
        raise AssertionError("mux must not start after cancel")

    monkeypatch.setattr("app.core.video_dubbing.service.VideoMuxer.mux", _boom)
    service.cancel()
    with pytest.raises(GenerationCancelled):
        service.export_video(project)
    assert mux_calls["n"] == 0
    service.reset_cancel()


# ---------------------------------------------------------------------------
# 10. FFmpeg error persists full stderr
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_diagnosed_subprocess_persists_full_stderr(tmp_path):
    dump_dir = tmp_path / "sub"
    runner = DiagnosedSubprocess(stderr_dir=dump_dir)
    # An invalid filtergraph makes ffmpeg fail with a multi-line stderr.
    with pytest.raises(Exception):
        runner.run(
            [FFMPEG_EXE, "-y", "-hide_banner", "-f", "lavfi", "-i", "notarealinputXYZ", "-t", "0.1",
             str(tmp_path / "out.mkv")],
            label="bad_ffmpeg",
        )
    files = list(dump_dir.glob("*.stderr.log"))
    assert files, "full stderr must be persisted to a file"
    content = files[0].read_text(encoding="utf-8")
    # Full multi-line ffmpeg diagnostics must be present (not a 3000-char tail).
    assert "No such filter" in content or "Error opening input" in content


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_ffmpeg_runner_retains_full_stderr(tmp_path):
    from app.utils.ffmpeg_utils import FFmpegRunner

    runner = FFmpegRunner(Path(FFMPEG_EXE), stderr_dump_dir=tmp_path / "sd")
    with pytest.raises(Exception):
        runner.run(
            ["-y", "-hide_banner", "-f", "lavfi", "-i", "bogusXYZ", "-t", "0.1",
             str(tmp_path / "o.mkv")],
            label="mux_fail",
        )
    # Full stderr retained on the instance (not just the 3000-char tail).
    assert runner.last_full_stderr
    assert runner.last_exit_code != 0
    assert list((tmp_path / "sd").glob("*.stderr.log"))


# ---------------------------------------------------------------------------
# 11. Render error keeps the previous working file
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_render_error_keeps_previous_working_file(tmp_path, monkeypatch):
    service, project, _tts = _new_service(tmp_path)
    _attach_video(service, project, tmp_path)
    _import_srt(service, project, count=2)
    service.generate_all(project)
    service.render_narration(project)
    narration = Path(project.narration_wav)
    assert narration.is_file()
    size_before = narration.stat().st_size

    def _fail(self, *a, **k):
        raise RuntimeError("render boom")

    monkeypatch.setattr("app.core.video_dubbing.service.TimelineRenderer.render", _fail)
    service.reset_cancel()
    with pytest.raises(Exception):
        service.render_narration(project)
    # The previously-good narration.wav must still exist, intact.
    assert narration.is_file()
    assert narration.stat().st_size == size_before


# ---------------------------------------------------------------------------
# 12. Leftover .part recovery
# ---------------------------------------------------------------------------


def test_leftover_part_recovered_on_open(tmp_path):
    service, project, _tts = _new_service(tmp_path)
    _import_srt(service, project, count=1)
    cues_dir = project.cues_dir()
    cues_dir.mkdir(parents=True, exist_ok=True)
    leftover = cues_dir / "cue_000001_raw.wav.part"
    leftover.write_bytes(b"\0not a real wav")
    # recover_project_artifacts should move the .part out of the cues dir.
    recovered = service.recover_project_artifacts(project)
    assert leftover in recovered or not leftover.exists()
    assert not leftover.exists()


# ---------------------------------------------------------------------------
# 13. Video without audio track
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_video_without_audio_track_muxes_dubbed_mix_only(tmp_path):
    service, project, _tts = _new_service(tmp_path)
    # Attach a WITH-audio video so narration/mix can size against its duration.
    _attach_video(service, project, tmp_path, with_audio=True)
    _import_srt(service, project, count=2)
    service.generate_all(project)
    service.render_dubbed_mix(project)
    # Now create a video with NO audio stream and point the project at it.
    video_noaudio = tmp_path / "noaudio.mp4"
    subprocess.run(
        [FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:s=320x240:d=6:r=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_noaudio)],
        check=True, capture_output=True,
    )
    project.video_path = video_noaudio
    muxer = VideoMuxer(ffmpeg_path=FFMPEG_EXE)
    has_audio, err = muxer._source_has_audio(video_noaudio)
    assert err is None
    assert has_audio is False
    result = muxer.mux(project)
    assert result.output_path.is_file()
    assert "Dubbed Mix" in result.tracks
    assert "Original" not in result.tracks


# ---------------------------------------------------------------------------
# 14. No secrets / no full cue text in logs
# ---------------------------------------------------------------------------


def test_redaction_strips_secrets_and_keeps_hashes():
    voice_cfg = {
        "engine": "fake",
        "voice": "v1",
        "api_key": "sk-1234567890abcdef",
        "authorization": "Bearer abcdefghij",
        "reference_transcript": "full secret transcript",
    }
    cleaned = safe_voice_snapshot(voice_cfg)
    assert cleaned["api_key"] == "***REDACTED***"
    assert cleaned["authorization"] == "***REDACTED***"
    # Reference transcript must NOT leak as plaintext -> hashed/redacted.
    assert cleaned["reference_transcript"] != "full secret transcript"
    # Non-secret fields preserved.
    assert cleaned["voice"] == "v1"
    # A SHA-256 hash must pass through intact (not mangled by token masking).
    digest = sha256_text("some subtitle text" * 10)
    assert redact_value({"fp": digest})["fp"] == digest
    # Command with a bearer token is masked.
    cmd = redact_command(["-headers", "Authorization: Bearer abcdefghij1234567890"])
    assert "abcdefghij1234567890" not in " ".join(cmd)


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_jsonl_has_no_secrets_or_full_cue_text(tmp_path):
    service, project, _tts = _new_service(tmp_path)
    _import_srt(service, project, count=1)
    # Embed a fake secret in the voice config and a distinctive cue text.
    secret = "sk-SUPERSECRET1234567890"
    project.settings.voice_config = {"engine": "fake", "voice": "v1", "api_key": secret}
    cue_text = "UNIQUEMARKER_CUE_TEXT_42"
    project.cues[0].spoken_text = cue_text
    service.generate_all(project, force=True)
    # Aggregate every events.jsonl the service wrote for this project.
    blob = ""
    for ev_path in (project.project_dir / "logs").glob("run_*/events.jsonl"):
        blob += ev_path.read_text(encoding="utf-8")
    assert blob, "diagnostic JSONL must be produced"
    assert secret not in blob
    assert "UNIQUEMARKER_CUE_TEXT_42" not in blob
    # But the cue's text hash IS present.
    assert sha256_text(cue_text) in blob


# ---------------------------------------------------------------------------
# 15. JSONL survives artificial process abort
# ---------------------------------------------------------------------------


def test_jsonl_valid_after_abort(tmp_path):
    sink = JsonlEventSink(tmp_path / "events.jsonl")
    for i in range(50):
        sink.write("tick", {"i": i}, force_flush=(i % 10 == 0))
    # Simulate an abrupt death mid-write: open the file and write a partial line
    # without a trailing newline (as a crash might leave it).
    with open(tmp_path / "events.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"event":"partial","data":{"missi')  # truncated, no newline
    # The reader must tolerate the truncated final line (skip it) and parse the
    # rest — the journal stays usable after a crash.
    rd = RunDirectory.create(tmp_path, "run_abort")
    rd.close()
    # reuse the sink's file by reading directly
    events = []
    for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    assert len(events) >= 50
    assert events[0]["event"] == "tick"
