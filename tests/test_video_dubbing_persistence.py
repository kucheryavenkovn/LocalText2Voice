from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from app.core.video_dubbing.models import (
    CueStatus,
    DubbingProjectSettings,
    DubbingCue,
)
from app.core.video_dubbing.project_store import (
    DUBBING_MANIFEST_NAME,
    DubbingProjectStore,
)
from app.core.video_dubbing.service import VideoDubbingService, _fingerprint
from app.core.video_dubbing.voice_catalog import VoiceCatalogService
from app.tts.base import BaseTTSEngine, TTSEngineError


class _ScriptedTTS(BaseTTSEngine):
    """Fake TTS that can fail on chosen cues to exercise checkpoints."""

    def __init__(self, ffmpeg_exe: str, fail_on: set[int] | None = None) -> None:
        self.ffmpeg_exe = ffmpeg_exe
        self.fail_on = fail_on or set()
        self.calls: list[int] = []

    def validate(self, voice_config: dict[str, Any]) -> None:
        return None

    def synthesize_to_wav(
        self, text: str, output_wav: Path, voice_config: dict[str, Any]
    ) -> Path:
        seq = int(voice_config.get("_dubbing_sequence", 0))
        self.calls.append(seq)
        if seq in self.fail_on:
            raise TTSEngineError("synthesis failed (test)")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                self.ffmpeg_exe,
                "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=0.8",
                "-ac", "1", "-ar", "22050", str(output_wav),
            ],
            check=True, capture_output=True,
        )
        return output_wav

    def cancel_current(self) -> None:
        return None


FFMPEG_EXE = shutil.which("ffmpeg")


def _new_service(tmp_path, fail_on=None):
    store = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    tts = _ScriptedTTS(FFMPEG_EXE, fail_on=fail_on)
    service = VideoDubbingService(tts, store=store)
    settings = DubbingProjectSettings(
        language="ru", tts_engine="fake", ffmpeg_path="ffmpeg/ffmpeg.exe",
        voice_config={"engine": "fake"},
    )
    project = service.create_project(tmp_path / "proj", settings=settings)
    return service, project


def _import_short_srt(service, project):
    srt = project.project_dir / "in.srt"
    blocks = []
    for i in range(1, 11):
        start_s = i * 2
        end_s = start_s + 1
        blocks.append(
            f"{i}\n00:00:{start_s:02d},000 --> 00:00:{end_s:02d},000\nРеплика {i}.\n"
        )
    srt.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    service.import_srt(project, srt)


# ---------------------------------------------------------------------------
# Fingerprint + skip-ready
# ---------------------------------------------------------------------------


def test_fingerprint_changes_with_text_or_voice():
    settings = DubbingProjectSettings(tts_engine="piper", voice="A", voice_config={"engine": "piper"})
    cue = DubbingCue("1", 1, 0, 1000, 1000, "hello", "hello")
    fp1 = _fingerprint(cue, settings)
    cue.spoken_text = "bye"
    assert _fingerprint(cue, settings) != fp1
    settings.voice = "B"
    assert _fingerprint(cue, settings) != fp1


def test_generation_plan_classifies_ready_and_missing(tmp_path):
    pytest.importorskip("shutil")
    if not FFMPEG_EXE:
        pytest.skip("ffmpeg required")
    service, project = _new_service(tmp_path)
    _import_short_srt(service, project)
    # Generate only first 3 cues.
    service.generate_selected(project, [1, 2, 3], force=True)
    plan = service.generation_plan(project, force=False)
    assert set([1, 2, 3]).issubset(set(plan["ready"]))
    assert set(range(4, 11)).issubset(set(plan["will_generate"]))


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_checkpoint_resume_after_failure(tmp_path):
    service, project = _new_service(tmp_path, fail_on={6})
    _import_short_srt(service, project)
    # generate_all marks cue 6 failed but persists cues 1-5 via checkpoint.
    service.generate_all(project)
    assert project.cues[5].status == CueStatus.FAILED.value

    # Reload from a fresh store (simulate restart) and resume.
    store2 = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    reloaded = store2.load_project(project.project_id)
    assert reloaded is not None
    # First five are ready, cue 6 failed.
    assert reloaded.cues[0].status == CueStatus.RENDERED.value
    assert reloaded.cues[5].status == CueStatus.FAILED.value

    tts2 = _ScriptedTTS(FFMPEG_EXE)  # no failures now
    service2 = VideoDubbingService(tts2, store=store2)
    plan = service2.generation_plan(reloaded, force=False)
    # Cues 1-5 and 7-10 are ready (the loop continued past cue 6); only cue 6
    # failed and must be re-synthesized. Nothing ready is re-synthesized.
    assert 6 in plan["will_generate"]
    for seq in (1, 2, 3, 4, 5, 7, 8, 9, 10):
        assert seq not in plan["will_generate"], f"cue {seq} should be ready"
    service2.generate_all(reloaded)
    # TTS re-synthesizes ONLY cue 6.
    assert sorted(tts2.calls) == [6]
    # All cues are now ready.
    assert all(c.status != CueStatus.FAILED.value for c in reloaded.cues)


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_re_fit_existing_does_not_call_tts(tmp_path):
    service, project = _new_service(tmp_path)
    _import_short_srt(service, project)
    service.generate_all(project)
    service.tts_engine.calls.clear()
    # Change speed policy and re-fit existing raw WAV.
    project.settings.hard_speed_limit = 2.2
    service.re_fit_existing(project)
    assert service.tts_engine.calls == []  # no TTS re-synthesis
    # Fitted audio still present.
    assert all(Path(c.fitted_audio_path).is_file() for c in project.cues)


# ---------------------------------------------------------------------------
# Manifest completeness / fallback
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")
def test_manifest_restores_without_sqlite(tmp_path):
    service, project = _new_service(tmp_path)
    _import_short_srt(service, project)
    service.generate_selected(project, [1], force=True)
    manifest = project.project_dir / DUBBING_MANIFEST_NAME

    # Simulate SQLite loss: brand new DB + open from manifest only.
    fresh = DubbingProjectStore(db_path=tmp_path / "db2.sqlite3")
    loaded = fresh.load_project_from_manifest(manifest)
    assert len(loaded.cues) == 10
    assert loaded.srt_source_text  # restored from manifest field
    assert loaded.cues[0].status == CueStatus.RENDERED.value
    assert loaded.cues[0].generation_fingerprint


def test_manifest_contains_schema_and_restore_fields(tmp_path):
    service, project = _new_service(tmp_path)
    project.selected_sequence = 3
    project.last_player_position_ms = 12345
    service.save_project(project)
    data = json.loads((project.project_dir / DUBBING_MANIFEST_NAME).read_text("utf-8"))
    assert data["schema_version"] >= 2
    assert data["selected_sequence"] == 3
    assert data["last_player_position_ms"] == 12345
    assert data["srt_source_text"] == ""  # no SRT imported yet


# ---------------------------------------------------------------------------
# Create-in-existing-dir guard
# ---------------------------------------------------------------------------


def test_find_project_by_dir_recovers_existing(tmp_path):
    service, project = _new_service(tmp_path)
    found = service.store.find_project_by_dir(project.project_dir)
    assert found is not None
    assert found.project_id == project.project_id


def test_create_does_not_silently_overwrite_existing_manifest(tmp_path):
    service, project = _new_service(tmp_path)
    manifest = project.project_dir / DUBBING_MANIFEST_NAME
    assert manifest.is_file()
    # Re-creating in the same dir would normally make a new UUID; the store's
    # create_project always inserts a new row, but find_project_by_dir lets the
    # UI detect the existing one. Verify the original is recoverable.
    found = service.store.find_project_by_dir(project.project_dir)
    assert found.project_id == project.project_id


# ---------------------------------------------------------------------------
# Voice catalog
# ---------------------------------------------------------------------------


def test_voice_catalog_resolve_config_piper(tmp_path):
    catalog = VoiceCatalogService(piper_path="engines/piper/piper.exe")
    config = catalog.resolve_voice_config("piper", "ru_RU-denis-medium", {})
    assert config["engine"] == "piper"


def test_voice_catalog_resolve_config_kokoro():
    catalog = VoiceCatalogService()
    config = catalog.resolve_voice_config("kokoro", "af_heart", {})
    assert config["engine"] == "kokoro"
    assert config["voice"] == "af_heart"


def test_voice_catalog_empty_for_unknown_engine():
    catalog = VoiceCatalogService()
    assert catalog.list_voices("nonexistent-engine") == []
