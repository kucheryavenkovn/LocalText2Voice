from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest

from app.core.video_dubbing.models import (
    DubbingProjectSettings,
    SyncMode,
)
from app.core.video_dubbing.service import VideoDubbingService
from app.core.video_dubbing.project_store import DubbingProjectStore
from app.tts.base import BaseTTSEngine

FFMPEG_EXE = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(
    not FFMPEG_EXE,
    reason="ffmpeg not available on PATH",
)


class FakeToneTTS(BaseTTSEngine):
    def __init__(self, ffmpeg_exe: str, durations: dict[int, float]) -> None:
        self.ffmpeg_exe = ffmpeg_exe
        self.durations = durations

    def validate(self, voice_config: dict[str, Any]) -> None:
        return None

    def synthesize_to_wav(
        self, text: str, output_wav: Path, voice_config: dict[str, Any]
    ) -> Path:
        seq = int(voice_config.get("_dubbing_sequence", 1))
        duration = self.durations.get(seq, 1.5)
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
                f"sine=frequency=440:duration={duration:.3f}",
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
        return None


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as audio:
        return int(round(audio.getnframes() / audio.getframerate() * 1000))


@pytest.fixture
def fake_project(tmp_path):
    video = tmp_path / "video.mp4"
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
            "sine=frequency=150:duration=30",
            "-f",
            "lavfi",
            "-i",
            "color=c=navy:s=160x120:d=30",
            "-shortest",
            "-c:a",
            "aac",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    srt = tmp_path / "subs.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nКороткая реплика.\n"
        "2\n00:00:05,000 --> 00:00:07,000\nЧуть длиннее чем окно.\n"
        "3\n00:00:09,000 --> 00:00:10,000\nОчень длинная реплика не влезает.\n",
        encoding="utf-8",
    )
    settings = DubbingProjectSettings(
        language="ru",
        tts_engine="fake",
        ffmpeg_path="ffmpeg/ffmpeg.exe",
        max_speed_factor=1.35,
        sync_mode=SyncMode.BEST_EFFORT,
    )
    settings.voice_config = {"engine": "fake"}
    store = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    project_dir = tmp_path / "proj"
    tts = FakeToneTTS(
        FFMPEG_EXE,
        durations={1: 1.5, 2: 2.4, 3: 2.0},  # cue3 budget=1s, raw=2s -> needs shortening
    )
    service = VideoDubbingService(tts, store=store)
    project = service.create_project(project_dir, settings=settings)
    service.attach_video(project, video)

    project.settings.voice_config = {"engine": "fake"}
    service.import_srt(project, srt)
    return service, project


def test_full_pipeline_generate_render_mix_export(tmp_path, fake_project):
    service, project = fake_project
    # durations already configured per-sequence in the fake engine fixture
    service.generate_all(project)

    c1, c2, c3 = project.cues
    # cue1 short -> no speedup
    assert c1.applied_speed_factor == pytest.approx(1.0)
    # cue2 mild speedup (2.4/2.0=1.2 within 1.35)
    assert c2.applied_speed_factor > 1.0
    assert c2.overflow_ms == 0
    # cue3 budget 1s, raw 2s -> factor 2.0 > 1.35 -> best_effort overflow
    assert c3.overflow_ms > 0

    narration = service.render_narration(project)
    assert narration.is_file()
    assert abs(_wav_duration_ms(narration) - project.duration_ms) <= 40

    mix = service.render_dubbed_mix(project)
    assert mix.is_file()
    assert abs(_wav_duration_ms(mix) - project.duration_ms) <= 40

    final = service.export_video(project)
    assert final.is_file()

    json_report = project.reports_dir() / "timing_report.json"
    csv_report = project.reports_dir() / "timing_report.csv"
    assert json_report.is_file() and csv_report.is_file()


def test_regenerate_single_cue_invalidates_downstream(tmp_path, fake_project):
    service, project = fake_project
    service.generate_all(project)
    narration_v1 = service.render_narration(project)
    v1_size = narration_v1.stat().st_size

    cue = project.cues[0]
    service.update_cue_text(project, cue.sequence, "Изменённый текст реплики.")
    assert cue.is_stale is True
    assert project.stale.narration is True

    # regenerate only the edited cue
    service.generate_selected(project, [cue.sequence])
    assert cue.is_stale is False
    narration_v2 = service.render_narration(project)
    assert narration_v2.is_file()


def test_strict_mode_blocks_export(tmp_path, fake_project):
    service, project = fake_project
    project.settings.sync_mode = SyncMode.STRICT
    service.generate_all(project)
    can, blockers = service.can_export(project)
    assert can is False
    assert any("shortening" in b for b in blockers)


def test_project_round_trip_persistence(tmp_path, fake_project):
    service, project = fake_project
    service.generate_all(project)
    project_id = project.project_id
    reloaded = service.load_project(project_id)
    assert len(reloaded.cues) == 3
    assert reloaded.video_probe is not None
    assert reloaded.cues[0].fitted_audio_path is not None
