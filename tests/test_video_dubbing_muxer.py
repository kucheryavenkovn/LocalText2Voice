from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.core.video_dubbing.models import (
    DubbingProject,
    ExportSettings,
    OutputContainer,
    VideoProbeInfo,
)
from app.core.video_dubbing.video_muxer import VideoMuxer
from app.utils import ffprobe_utils

FFMPEG_EXE = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(
    not FFMPEG_EXE,
    reason="ffmpeg not available on PATH",
)


def _make_source_video(path: Path, duration: float = 5.0) -> Path:
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
            f"sine=frequency=220:duration={duration}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=green:s=160x120:d={duration}",
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


def _make_wav(path: Path, duration: float) -> Path:
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
            f"sine=frequency=440:duration={duration}",
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


def _probe_audio_titles(path: Path) -> list[str]:
    data = ffprobe_utils.probe_media(path, "ffmpeg/ffmpeg.exe")
    titles = []
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "audio":
            tags = stream.get("tags", {}) or {}
            # MKV stores title; MP4 stores handler_name. Prefer whichever is set.
            name = str(tags.get("title") or tags.get("handler_name") or "")
            titles.append(name)
    return titles


def _probe_audio_track_count(path: Path) -> int:
    data = ffprobe_utils.probe_media(path, "ffmpeg/ffmpeg.exe")
    return sum(1 for s in data.get("streams", []) if s.get("codec_type") == "audio")


def _probe_default_track_index(path: Path) -> int | None:
    data = ffprobe_utils.probe_media(path, "ffmpeg/ffmpeg.exe")
    for index, stream in enumerate(
        [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    ):
        dispo = stream.get("disposition", {}) or {}
        if dispo.get("default"):
            return index
    return None


def _project(tmp_path: Path, container: OutputContainer) -> DubbingProject:
    video = _make_source_video(tmp_path / "src.mp4")
    mix = _make_wav(tmp_path / "mix.wav", 5.0)
    project = DubbingProject(
        project_id="mux",
        project_dir=tmp_path / "proj",
        video_path=video,
        video_probe=VideoProbeInfo(duration_ms=5000),
    )
    project.settings.language = "ru"
    project.settings.export = ExportSettings(container=container)
    project.dubbed_mix_wav = mix
    project.ensure_directories()
    return project


def test_mux_mkv_two_tracks_default_is_dubbed_mix(tmp_path):
    project = _project(tmp_path, OutputContainer.MKV)
    muxer = VideoMuxer("ffmpeg/ffmpeg.exe")
    result = muxer.mux(project)
    assert result.output_path.is_file()
    titles = _probe_audio_titles(result.output_path)
    assert "Original" in titles
    assert "Dubbed Mix" in titles
    default_idx = _probe_default_track_index(result.output_path)
    assert default_idx == titles.index("Dubbed Mix")
    assert _probe_audio_track_count(result.output_path) == 2


def test_mux_mp4_two_tracks(tmp_path):
    project = _project(tmp_path, OutputContainer.MP4)
    muxer = VideoMuxer("ffmpeg/ffmpeg.exe")
    result = muxer.mux(project)
    assert _probe_audio_track_count(result.output_path) == 2
    titles = _probe_audio_titles(result.output_path)
    assert "Original" in titles and "Dubbed Mix" in titles
    default_idx = _probe_default_track_index(result.output_path)
    assert default_idx == titles.index("Dubbed Mix")


def test_mux_includes_narration_only_track(tmp_path):
    project = _project(tmp_path, OutputContainer.MKV)
    narration = _make_wav(tmp_path / "narration.wav", 5.0)
    project.settings.export.include_narration_only = True
    muxer = VideoMuxer("ffmpeg/ffmpeg.exe")
    result = muxer.mux(project, narration_only_wav=narration)
    titles = _probe_audio_titles(result.output_path)
    assert "Narration Only" in titles
    assert "Dubbed Mix" in titles
    assert _probe_audio_track_count(result.output_path) == 3


def test_mux_video_stream_copied(tmp_path):
    project = _project(tmp_path, OutputContainer.MKV)
    muxer = VideoMuxer("ffmpeg/ffmpeg.exe")
    can_copy, codec = muxer.can_copy_video(Path(project.video_path), OutputContainer.MKV)
    assert can_copy
    assert codec == "h264"


def test_mux_duration_matches_source(tmp_path):
    project = _project(tmp_path, OutputContainer.MKV)
    muxer = VideoMuxer("ffmpeg/ffmpeg.exe")
    result = muxer.mux(project)
    data = ffprobe_utils.probe_media(result.output_path, "ffmpeg/ffmpeg.exe")
    duration_ms, _video, _audio = ffprobe_utils.parse_video_probe(data)
    # no -shortest: should be ~5s
    assert 4800 <= duration_ms <= 5200


def test_mux_embeds_srt(tmp_path):
    project = _project(tmp_path, OutputContainer.MKV)
    srt = tmp_path / "sub.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello.\n", encoding="utf-8")
    project.settings.export.embed_subtitles = True
    muxer = VideoMuxer("ffmpeg/ffmpeg.exe")
    result = muxer.mux(project, srt_path=srt)
    data = ffprobe_utils.probe_media(result.output_path, "ffmpeg/ffmpeg.exe")
    sub_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "subtitle"]
    assert len(sub_streams) >= 1
