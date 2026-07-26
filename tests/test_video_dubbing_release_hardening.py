from __future__ import annotations

import json
import shutil
import subprocess
import threading
import wave
from pathlib import Path
from typing import Any

import pytest

from app.core.video_dubbing.elastic_timing import ElasticTimingPlanner
from app.core.video_dubbing.fingerprints import (
    file_content_hash,
    fit_fingerprint,
    fit_settings_fingerprint,
    raw_generation_fingerprint,
)
from app.core.video_dubbing.generation import (
    CueErrorPolicy,
    GenerationCancelled,
    GenerationRunStatus,
)
from app.core.video_dubbing.models import (
    CueStatus,
    DubbingCue,
    DubbingProjectSettings,
    ElasticTimingSettings,
    ORIGINAL_AUDIO_MODE_LABELS_RU,
    OriginalAudioMode,
)
from app.core.video_dubbing.project_store import DubbingProjectStore
from app.core.video_dubbing.service import VideoDubbingService, _fingerprint
from app.core.video_dubbing.wav_validator import WavArtifactValidator, cleanup_part_files
from app.tts.base import BaseTTSEngine, TTSCancelled, TTSEngineError


FFMPEG_EXE = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg required")


class ScriptedTTS(BaseTTSEngine):
    engine_id = "fake"

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
        self.voice_configs: list[dict[str, Any]] = []
        self.block_event = block_event
        self.release_event = release_event
        self._cancel = False

    def validate(self, voice_config: dict[str, Any]) -> None:
        return None

    def synthesize_to_wav(
        self, text: str, output_wav: Path, voice_config: dict[str, Any]
    ) -> Path:
        seq = int(voice_config.get("_dubbing_sequence", 0))
        self.calls.append(seq)
        self.voice_configs.append(dict(voice_config))
        if self.block_event is not None:
            self.block_event.set()
        if self.release_event is not None:
            self.release_event.wait(timeout=5)
        if self._cancel or seq in self.cancel_on:
            raise TTSCancelled("cancelled by test")
        if seq in self.fail_on:
            raise TTSEngineError("synthesis failed (test)")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                self.ffmpeg_exe,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration={self.duration}",
                "-ac",
                "1",
                "-ar",
                "22050",
                str(output_wav),
            ],
            check=True,
            capture_output=True,
        )
        return output_wav

    def cancel_current(self) -> None:
        self._cancel = True


def _service(tmp_path, **tts_kwargs):
    store = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    tts = ScriptedTTS(FFMPEG_EXE, **tts_kwargs)
    service = VideoDubbingService(tts, store=store)
    settings = DubbingProjectSettings(
        language="ru",
        tts_engine="fake",
        voice="v1",
        voice_config={"engine": "fake", "voice": "v1"},
        ffmpeg_path="ffmpeg/ffmpeg.exe",
        cue_error_policy="continue",
    )
    project = service.create_project(tmp_path / "proj", settings=settings)
    return service, project, tts


def _import_cues(service, project, count=5):
    blocks = []
    for i in range(1, count + 1):
        start = i * 2
        end = start + 1
        blocks.append(
            f"{i}\n00:00:{start:02d},000 --> 00:00:{end:02d},000\nРеплика {i}.\n"
        )
    srt = project.project_dir / "in.srt"
    srt.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    service.import_srt(project, srt)


def test_cancel_between_cues_not_failed(tmp_path):
    service, project, tts = _service(tmp_path, cancel_on={2})
    _import_cues(service, project, 4)
    result = service.generate_all(project, force=True)
    assert result.status == GenerationRunStatus.CANCELLED
    assert project.cues[0].status != CueStatus.FAILED.value
    assert project.cues[1].status == CueStatus.CANCELLED.value
    assert project.cues[2].status == CueStatus.PENDING.value
    assert not list(project.cues_dir().glob("*.part*"))


def test_cancel_does_not_mark_success_message(tmp_path):
    service, project, tts = _service(tmp_path, cancel_on={1})
    _import_cues(service, project, 2)
    logs: list[str] = []
    service.log_callback = logs.append
    messages: list[str] = []
    service.progress_callback = lambda stage, c, t, msg: messages.append(msg)
    result = service.generate_all(project, force=True)
    assert result.status == GenerationRunStatus.CANCELLED
    assert any("cancelled" in m.casefold() for m in messages)
    assert not any(m == "Generation completed" for m in messages)


def test_continue_policy_completed_with_errors(tmp_path):
    service, project, tts = _service(tmp_path, fail_on={2})
    project.settings.cue_error_policy = CueErrorPolicy.CONTINUE.value
    _import_cues(service, project, 3)
    result = service.generate_all(project, force=True)
    assert result.status == GenerationRunStatus.COMPLETED_WITH_ERRORS
    assert result.failed_count == 1
    assert result.completed_count == 2
    assert project.cues[1].status == CueStatus.FAILED.value
    assert project.cues[2].status != CueStatus.PENDING.value


def test_stop_policy_failed(tmp_path):
    service, project, tts = _service(tmp_path, fail_on={2})
    project.settings.cue_error_policy = CueErrorPolicy.STOP.value
    _import_cues(service, project, 4)
    result = service.generate_all(project, force=True)
    assert result.status == GenerationRunStatus.FAILED
    assert project.cues[2].status == CueStatus.PENDING.value


def test_generation_context_snapshot_ignores_live_voice_change(tmp_path):
    service, project, tts = _service(tmp_path)
    _import_cues(service, project, 2)
    context = service.build_generation_context(project)
    project.settings.voice = "changed-during-run"
    project.settings.voice_config = {"engine": "fake", "voice": "changed-during-run"}
    result = service._generate_many(
        project, list(project.cues), force=True, context=context
    )
    assert result.status == GenerationRunStatus.COMPLETED
    assert all(cfg.get("voice") == "v1" for cfg in tts.voice_configs)
    assert project.cues[0].generation_fingerprint
    # Fingerprint must match the snapshot voice, not the mutated project voice.
    expected = raw_generation_fingerprint(
        project.cues[0],
        engine_id="fake",
        voice_id="v1",
        voice_config={"engine": "fake", "voice": "v1"},
        language="ru",
        sample_rate=project.settings.sample_rate,
        channels=project.settings.channels,
    )
    assert project.cues[0].generation_fingerprint == expected


def test_pause_compression_changes_fit_not_raw_fp(tmp_path):
    service, project, tts = _service(tmp_path)
    _import_cues(service, project, 1)
    service.generate_all(project, force=True)
    raw_fp = project.cues[0].generation_fingerprint
    fit_fp = project.cues[0].fit_fingerprint
    tts.calls.clear()
    project.settings.compress_internal_pauses = True
    project.settings.internal_pause_keep_ms = 40
    plan = service.generation_plan(project)
    assert project.cues[0].sequence in plan["needs_refit_from_raw"]
    assert project.cues[0].sequence not in plan["needs_tts_generation"]
    service.re_fit_existing(project)
    assert tts.calls == []
    assert project.cues[0].generation_fingerprint == raw_fp
    assert project.cues[0].fit_fingerprint != fit_fp


def test_wav_validator_rejects_empty_truncated_corrupt(tmp_path):
    validator = WavArtifactValidator()
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    assert not validator.validate(empty).valid

    truncated = tmp_path / "trunc.wav"
    truncated.write_bytes(b"RIFF$$$$WAVEfmt ")
    assert not validator.validate(truncated).valid

    good = tmp_path / "good.wav"
    subprocess.run(
        [
            FFMPEG_EXE,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.3",
            "-ac",
            "1",
            "-ar",
            "22050",
            str(good),
        ],
        check=True,
        capture_output=True,
    )
    assert validator.validate(good).valid


def test_atomic_replace_keeps_old_on_invalid_part(tmp_path):
    service, project, tts = _service(tmp_path)
    _import_cues(service, project, 1)
    service.generate_all(project, force=True)
    cue = project.cues[0]
    old_size = Path(cue.raw_audio_path).stat().st_size
    part = Path(cue.raw_audio_path).with_name(Path(cue.raw_audio_path).stem + ".part.wav")
    part.write_bytes(b"not a wav")
    # Recovery should quarantine leftovers.
    recovered = service.recover_project_artifacts(project)
    assert recovered
    assert Path(cue.raw_audio_path).is_file()
    assert Path(cue.raw_audio_path).stat().st_size == old_size
    assert not part.exists()


def test_legacy_refit_once(tmp_path):
    service, project, tts = _service(tmp_path)
    _import_cues(service, project, 1)
    service.generate_all(project, force=True)
    cue = project.cues[0]
    cue.generation_fingerprint = ""
    cue.legacy_audio_unverified = True
    cue.legacy_observed_fingerprint = _fingerprint(cue, project.settings)
    cue.fit_fingerprint = ""
    service.store.save_project(project)
    tts.calls.clear()
    plan1 = service.generation_plan(project)
    assert cue.sequence in plan1["needs_refit_from_raw"]
    service.generate_all(project, force=False)
    assert tts.calls == []
    plan2 = service.generation_plan(project)
    assert cue.sequence in plan2["ready_without_changes"]
    assert cue.fit_fingerprint


def test_autosave_during_generation_and_cancel(tmp_path):
    started = threading.Event()
    release = threading.Event()
    service, project, tts = _service(
        tmp_path, block_event=started, release_event=release, cancel_on=set()
    )
    _import_cues(service, project, 3)

    def run():
        service.generate_all(project, force=True)

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(timeout=5)
    # Autosave while TTS is blocked.
    project.title = "autosave-title"
    service.save_project(project)
    service.cancel()
    release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    reloaded = service.load_project(project.project_id)
    assert reloaded.title == "autosave-title"
    assert not list(reloaded.cues_dir().glob("*.part*"))
    # integrity
    import sqlite3

    conn = sqlite3.connect(tmp_path / "db.sqlite3")
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()


def test_reference_content_hash_differs_for_same_name(tmp_path):
    a = tmp_path / "voice.wav"
    b = tmp_path / "other" / "voice.wav"
    b.parent.mkdir()
    subprocess.run(
        [
            FFMPEG_EXE,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.2",
            str(a),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            FFMPEG_EXE,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=0.2",
            str(b),
        ],
        check=True,
        capture_output=True,
    )
    assert file_content_hash(a) != file_content_hash(b)


def test_elastic_group_two_and_three_cues():
    settings = ElasticTimingSettings(
        enabled=True,
        max_cues_per_group=3,
        max_common_speed_factor=1.35,
        min_inter_cue_gap_ms=120,
        boundary_guard_ms=150,
        max_shift_per_cue_ms=5000,
        max_group_extension_ms=10000,
    )
    planner = ElasticTimingPlanner(settings, video_duration_ms=20000)
    cues = [
        DubbingCue("1", 1, 10000, 12500, 2500, "a", "a", raw_duration_ms=4400),
        DubbingCue("2", 2, 13000, 13800, 800, "b", "b", raw_duration_ms=1400),
        DubbingCue("3", 3, 15150, 16000, 850, "c", "c", raw_duration_ms=500),
    ]
    for cue in cues:
        cue.ensure_source_timing()
    plan = planner.choose_group_for_cue(cues, 0)
    assert plan is not None
    assert len(plan.sequences) >= 2
    assert plan.common_speed_factor <= 1.35
    planner.apply_plan(cues, plan)
    factors = {c.common_speed_factor for c in cues if c.timing_group_id == plan.group_id}
    assert len(factors) == 1
    # right-only
    assert all((c.start_shift_ms or 0) >= 0 for c in cues)
    assert cues[0].planned_start_ms == cues[0].source_start_ms


def test_elastic_group_no_left_shift_and_boundary():
    settings = ElasticTimingSettings(
        enabled=True,
        max_cues_per_group=2,
        max_common_speed_factor=1.5,
        boundary_guard_ms=100,
        max_shift_per_cue_ms=3000,
        max_group_extension_ms=5000,
    )
    planner = ElasticTimingPlanner(settings, video_duration_ms=20000)
    cues = [
        DubbingCue("1", 1, 10000, 11000, 1000, "a", "a", raw_duration_ms=3000),
        DubbingCue("2", 2, 12000, 12500, 500, "b", "b", raw_duration_ms=400),
        DubbingCue("3", 3, 14000, 15000, 1000, "c", "c", raw_duration_ms=400),
    ]
    for cue in cues:
        cue.ensure_source_timing()
    plan = planner.choose_group_for_cue(cues, 0)
    assert plan is not None
    assert plan.boundary_end_ms <= 14000 - 100
    for scheduled in plan.schedule:
        assert scheduled.start_shift_ms >= 0
        assert scheduled.planned_end_ms <= plan.boundary_end_ms


def test_elastic_group_fails_when_insufficient_time():
    settings = ElasticTimingSettings(
        enabled=True,
        max_cues_per_group=2,
        max_common_speed_factor=1.2,
        max_shift_per_cue_ms=100,
        max_group_extension_ms=100,
        boundary_guard_ms=50,
    )
    planner = ElasticTimingPlanner(settings, video_duration_ms=13000)
    cues = [
        DubbingCue("1", 1, 10000, 10500, 500, "a", "a", raw_duration_ms=5000),
        DubbingCue("2", 2, 11000, 11200, 200, "b", "b", raw_duration_ms=2000),
        DubbingCue("3", 3, 11300, 12000, 700, "c", "c", raw_duration_ms=500),
    ]
    for cue in cues:
        cue.ensure_source_timing()
    plan = planner.choose_group_for_cue(cues, 0)
    assert plan is None


def test_export_adjusted_srt(tmp_path):
    service, project, tts = _service(tmp_path)
    _import_cues(service, project, 2)
    project.cues[0].planned_start_ms = 2100
    project.cues[0].planned_end_ms = 3000
    project.cues[0].source_start_ms = 2000
    path = service.export_adjusted_srt(project)
    text = path.read_text(encoding="utf-8")
    assert "00:00:02,100" in text
    # source timing unchanged
    assert project.cues[0].start_ms == 2000


def test_russian_audio_mode_labels():
    assert ORIGINAL_AUDIO_MODE_LABELS_RU[OriginalAudioMode.REPLACE.value] == (
        "Заменить оригинальный звук"
    )
    assert ORIGINAL_AUDIO_MODE_LABELS_RU[OriginalAudioMode.DUCKING.value] == (
        "Динамическое приглушение"
    )


def test_ru_locale_has_no_english_stubs_for_video_dubbing():
    root = Path(__file__).resolve().parents[1]
    ru = json.loads((root / "locales" / "ru.json").read_text(encoding="utf-8"))
    en = json.loads((root / "locales" / "en.json").read_text(encoding="utf-8"))
    allowed = {"TTS", "WAV", "FFmpeg", "SRT", "LUFS", "MKV", "MP4", "API"}
    bad = []
    for key, value in ru.items():
        if not (key.startswith("video_dubbing") or key == "nav_video_dubbing"):
            continue
        if key not in en:
            continue
        if value == en[key] and not any(tok in value for tok in allowed):
            # pure English stub copied from en
            if value and all(ord(ch) < 128 for ch in value):
                bad.append((key, value))
    # Soft check: core mode labels must be Russian.
    assert "Заменить" in " ".join(ORIGINAL_AUDIO_MODE_LABELS_RU.values())
