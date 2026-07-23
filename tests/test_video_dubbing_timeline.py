from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from app.core.video_dubbing.models import (
    Alignment,
    DubbingCue,
    DubbingProject,
    VideoProbeInfo,
)
from app.core.video_dubbing.timeline_renderer import TimelineRenderer

FFMPEG_EXE = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(
    not FFMPEG_EXE,
    reason="ffmpeg not available on PATH",
)


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as audio:
        return int(round(audio.getnframes() / audio.getframerate() * 1000))


def _make_tone(path: Path, duration_ms: int, freq: int = 440) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            f"sine=frequency={freq}:duration={duration_ms / 1000.0:.3f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-codec:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _project_with_cues(tmp_path: Path, duration_ms: int, cue_specs):
    project = DubbingProject(
        project_id="t",
        project_dir=tmp_path / "proj",
        video_probe=VideoProbeInfo(duration_ms=duration_ms),
    )
    project.ensure_directories()
    for index, (start_ms, end_ms, fitted_ms) in enumerate(cue_specs, start=1):
        fitted_path = project.cue_fitted_path(index)
        _make_tone(fitted_path, fitted_ms, freq=300 + index * 40)
        project.cues.append(
            DubbingCue(
                cue_id=str(index),
                sequence=index,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_budget_ms=end_ms - start_ms,
                source_text=f"text {index}",
                spoken_text=f"text {index}",
                fitted_audio_path=fitted_path,
                fitted_duration_ms=fitted_ms,
                raw_duration_ms=fitted_ms,
                status="fitted",
            )
        )
    return project


def test_render_empty_timeline_produces_exact_silence(tmp_path):
    project = _project_with_cues(tmp_path, duration_ms=6000, cue_specs=[])
    # add no cues; remove all
    project.cues.clear()
    renderer = TimelineRenderer("ffmpeg/ffmpeg.exe", window_seconds=5)
    out = tmp_path / "narration.wav"
    renderer.render(project, out)
    assert out.is_file()
    assert abs(_wav_duration_ms(out) - 6000) <= 25


def test_render_first_cue_not_from_zero(tmp_path):
    # cue at 5s-7.6s, video 10s
    project = _project_with_cues(tmp_path, duration_ms=10_000, cue_specs=[(5000, 7600, 2600)])
    renderer = TimelineRenderer("ffmpeg/ffmpeg.exe", window_seconds=300)
    out = tmp_path / "narration.wav"
    renderer.render(project, out)
    duration = _wav_duration_ms(out)
    assert abs(duration - 10_000) <= 30
    placed = renderer.placed_segments(project.cues)
    assert placed[0].absolute_start_ms == 5000


def test_render_absolute_positioning_no_cumulative_drift(tmp_path):
    project = _project_with_cues(
        tmp_path,
        duration_ms=20_000,
        cue_specs=[(5000, 7600, 2600), (10500, 14000, 3500), (17000, 18800, 1800)],
    )
    renderer = TimelineRenderer("ffmpeg/ffmpeg.exe", window_seconds=300)
    out = tmp_path / "narration.wav"
    renderer.render(project, out)
    assert abs(_wav_duration_ms(out) - 20_000) <= 30
    placed = renderer.placed_segments(project.cues)
    starts = [seg.absolute_start_ms for seg in placed]
    # Third cue start is independent of first two durations.
    assert starts == [5000, 10500, 17000]


def test_render_window_boundaries_split_crossing_cue(tmp_path):
    # window = 5s. cue 1 at 3s-6s crosses window boundary at 5s.
    project = _project_with_cues(
        tmp_path,
        duration_ms=10_000,
        cue_specs=[(3000, 6000, 3000)],
    )
    renderer = TimelineRenderer("ffmpeg/ffmpeg.exe", window_seconds=5)
    out = tmp_path / "narration.wav"
    renderer.render(project, out)
    assert abs(_wav_duration_ms(out) - 10_000) <= 30


def test_render_long_project_multiple_windows(tmp_path):
    specs = [(i * 2000, i * 2000 + 1500, 1400) for i in range(1, 8)]
    project = _project_with_cues(tmp_path, duration_ms=16_000, cue_specs=specs)
    renderer = TimelineRenderer("ffmpeg/ffmpeg.exe", window_seconds=5)
    out = tmp_path / "narration.wav"
    renderer.render(project, out)
    assert abs(_wav_duration_ms(out) - 16_000) <= 40
    windows = renderer._plan_windows(project.duration_ms, renderer.placed_segments(project.cues))
    assert len(windows) >= 3


def test_render_alignment_center_shifts_short_cue(tmp_path):
    project = _project_with_cues(tmp_path, duration_ms=10_000, cue_specs=[(5000, 8000, 2000)])
    renderer = TimelineRenderer("ffmpeg/ffmpeg.exe")
    placed = renderer.placed_segments(project.cues, alignment=Alignment.CENTER)
    assert placed[0].absolute_start_ms == 5500  # 5000 + (3000-2000)/2


def test_render_disabled_cue_excluded(tmp_path):
    project = _project_with_cues(
        tmp_path,
        duration_ms=10_000,
        cue_specs=[(1000, 3000, 2000), (5000, 7000, 2000)],
    )
    project.cues[0].enabled = False
    renderer = TimelineRenderer("ffmpeg/ffmpeg.exe")
    placed = renderer.placed_segments(project.cues)
    assert [seg.sequence for seg in placed] == [2]
