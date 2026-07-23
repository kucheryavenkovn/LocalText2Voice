from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.core.video_dubbing.models import (
    DubbingCue,
    DubbingProject,
    DuckingSettings,
    OriginalAudioMode,
    VideoProbeInfo,
)
from app.core.video_dubbing.preview_renderer import PreviewRenderer
from app.utils import ffprobe_utils

FFMPEG_EXE = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(
    not FFMPEG_EXE,
    reason="ffmpeg not available on PATH",
)


def _make_video(path: Path, duration: float = 10.0) -> Path:
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
            f"sine=frequency=200:duration={duration}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=red:s=160x120:d={duration}",
            "-shortest",
            "-c:a",
            "aac",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _make_fitted(path: Path, duration_ms: int) -> Path:
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
            f"sine=frequency=660:duration={duration_ms / 1000.0:.3f}",
            "-ac",
            "2",
            "-ar",
            "48000",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _project_with_cue(tmp_path: Path):
    video = _make_video(tmp_path / "video.mp4", duration=10.0)
    project = DubbingProject(
        project_id="p",
        project_dir=tmp_path / "proj",
        video_path=video,
        video_probe=VideoProbeInfo(duration_ms=10_000),
    )
    project.ensure_directories()
    fitted = _make_fitted(project.cue_fitted_path(1), 2000)
    project.cues.append(
        DubbingCue(
            cue_id="1",
            sequence=1,
            start_ms=4000,
            end_ms=6000,
            duration_budget_ms=2000,
            source_text="hi",
            spoken_text="hi",
            fitted_audio_path=fitted,
            fitted_duration_ms=2000,
        )
    )
    return project


def test_cue_preview_renders_short_clip(tmp_path):
    project = _project_with_cue(tmp_path)
    renderer = PreviewRenderer("ffmpeg/ffmpeg.exe")
    out = renderer.render_cue_preview(project, project.cues[0])
    assert out.is_file()
    data = ffprobe_utils.probe_media(out, "ffmpeg/ffmpeg.exe")
    duration_ms, video, _audio = ffprobe_utils.parse_video_probe(data)
    assert video is not None
    # window = [4000-1000, 6000+1000] = 4s
    assert 3800 <= duration_ms <= 4200


def test_cue_preview_clamped_to_video_end(tmp_path):
    project = _project_with_cue(tmp_path)
    project.cues[0].start_ms = 9500
    project.cues[0].end_ms = 9800
    project.settings.preview.pre_roll_ms = 1000
    project.settings.preview.post_roll_ms = 1000
    renderer = PreviewRenderer("ffmpeg/ffmpeg.exe")
    out = renderer.render_cue_preview(project, project.cues[0])
    assert out.is_file()


def test_full_preview_uses_mix(tmp_path):
    project = _project_with_cue(tmp_path)
    mix = tmp_path / "mix.wav"
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
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t",
            "10",
            str(mix),
        ],
        check=True,
        capture_output=True,
    )
    project.dubbed_mix_wav = mix
    renderer = PreviewRenderer("ffmpeg/ffmpeg.exe")
    out = renderer.render_full_preview(project)
    assert out.is_file()
    assert out.suffix == ".mkv"
    data = ffprobe_utils.probe_media(out, "ffmpeg/ffmpeg.exe")
    _duration, video, audio = ffprobe_utils.parse_video_probe(data)
    assert video is not None and len(audio) >= 1


def test_full_preview_requires_audio(tmp_path):
    project = _project_with_cue(tmp_path)
    project.dubbed_mix_wav = None
    project.narration_wav = None
    renderer = PreviewRenderer("ffmpeg/ffmpeg.exe")
    with pytest.raises(Exception):
        renderer.render_full_preview(project)


def test_cue_preview_requires_fitted_audio(tmp_path):
    project = _project_with_cue(tmp_path)
    project.cues[0].fitted_audio_path = None
    renderer = PreviewRenderer("ffmpeg/ffmpeg.exe")
    with pytest.raises(Exception):
        renderer.render_cue_preview(project, project.cues[0])
