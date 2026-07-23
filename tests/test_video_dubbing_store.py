from __future__ import annotations

import json
from pathlib import Path

import pytest

ffmpeg = pytest.importorskip("shutil")  # noqa: E501  placeholder, real check below
from app.utils import ffprobe_utils  # noqa: E402

from app.core.video_dubbing.models import (  # noqa: E402
    DubbingCue,
    DubbingProject,
    DubbingProjectSettings,
    OriginalAudioMode,
    StaleFlags,
    VideoProbeInfo,
)
from app.core.video_dubbing.project_store import (  # noqa: E402
    DUBBING_MANIFEST_NAME,
    DubbingProjectStore,
)
from app.core.video_dubbing.stale_state import (  # noqa: E402
    invalidate_for_container_change,
    invalidate_for_ducking_change,
    invalidate_for_text_change,
    mark_clean,
)


FFMPEG_AVAILABLE = (
    Path(ffprobe_utils.find_ffprobe("ffmpeg/ffmpeg.exe")).is_file()
    if True
    else False
)


def _make_project(tmp_path: Path) -> DubbingProject:
    project = DubbingProject(
        project_id="test-uuid-123",
        project_dir=tmp_path / "proj",
        title="Demo",
        video_path=tmp_path / "video.mp4",
        video_probe=VideoProbeInfo(duration_ms=60_000),
        srt_source_path=tmp_path / "sub.srt",
        srt_source_text="1\n00:00:01,000 --> 00:00:03,000\nHello.\n",
        settings=DubbingProjectSettings(
            language="ru",
            tts_engine="piper",
            max_speed_factor=1.3,
        ),
    )
    project.cues.append(
        DubbingCue(
            cue_id="1",
            sequence=1,
            start_ms=1000,
            end_ms=3000,
            duration_budget_ms=2000,
            source_text="Hello.",
            spoken_text="Hello.",
            raw_duration_ms=1800,
            applied_speed_factor=1.0,
            status="rendered",
        )
    )
    project.ensure_directories()
    return project


def test_store_creates_project(tmp_path):
    store = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    project_id = store.create_project(tmp_path / "proj", title="First")
    assert project_id
    projects = store.list_projects()
    assert len(projects) == 1
    assert projects[0]["title"] == "First"


def test_store_save_and_reload_round_trip(tmp_path):
    store = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    project = _make_project(tmp_path)
    store.save_project(project)

    loaded = store.load_project(project.project_id)
    assert loaded is not None
    assert loaded.title == "Demo"
    assert loaded.video_probe is not None
    assert loaded.video_probe.duration_ms == 60_000
    assert loaded.settings.language == "ru"
    assert len(loaded.cues) == 1
    assert loaded.cues[0].source_text == "Hello."
    assert loaded.cues[0].raw_duration_ms == 1800


def test_store_manifest_written(tmp_path):
    store = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    project = _make_project(tmp_path)
    store.save_project(project)
    manifest = project.project_dir / DUBBING_MANIFEST_NAME
    assert manifest.is_file()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["title"] == "Demo"
    assert data["project_id"] == project.project_id
    assert len(data["cues"]) == 1


def test_store_load_from_manifest(tmp_path):
    store = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    project = _make_project(tmp_path)
    store.save_project(project)
    manifest_path = project.project_dir / DUBBING_MANIFEST_NAME

    fresh_store = DubbingProjectStore(db_path=tmp_path / "db2.sqlite3")
    imported = fresh_store.load_project_from_manifest(manifest_path)
    assert imported.title == "Demo"
    assert imported.settings.language == "ru"
    assert len(imported.cues) == 1


def test_store_update_overwrites_cues(tmp_path):
    store = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    project = _make_project(tmp_path)
    store.save_project(project)
    project.cues[0].spoken_text = "Updated text."
    project.cues.append(
        DubbingCue(
            cue_id="2",
            sequence=2,
            start_ms=4000,
            end_ms=6000,
            duration_budget_ms=2000,
            source_text="Second.",
            spoken_text="Second.",
        )
    )
    store.save_project(project)
    loaded = store.load_project(project.project_id)
    assert loaded is not None
    assert [c.sequence for c in loaded.cues] == [1, 2]
    assert loaded.cues[0].spoken_text == "Updated text."


def test_store_delete_project(tmp_path):
    store = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    project = _make_project(tmp_path)
    store.save_project(project)
    store.delete_project(project.project_id)
    assert store.load_project(project.project_id) is None


def test_stale_state_text_change_invalidates_everything():
    stale = StaleFlags()
    mark_clean(stale)
    invalidate_for_text_change(stale)
    assert stale.cues and stale.narration and stale.mix and stale.preview and stale.video


def test_stale_state_ducking_change_keeps_cues():
    stale = StaleFlags()
    mark_clean(stale)
    invalidate_for_ducking_change(stale)
    assert stale.cues is False
    assert stale.narration is False
    assert stale.mix and stale.preview and stale.video


def test_stale_state_container_change_only_video():
    stale = StaleFlags()
    mark_clean(stale)
    invalidate_for_container_change(stale)
    assert stale.video is True
    assert stale.mix is False
    assert stale.cues is False


def test_stale_flags_round_trip():
    stale = StaleFlags(cues=False, narration=True, mix=False, preview=True, video=False)
    data = stale.to_dict()
    restored = StaleFlags.from_dict(data)
    assert restored == stale


def test_ducking_mode_serialization():
    settings = DubbingProjectSettings()
    settings.ducking.mode = OriginalAudioMode.CONSTANT
    data = settings.to_dict()
    restored = DubbingProjectSettings.from_dict(data)
    assert restored.ducking.mode == OriginalAudioMode.CONSTANT


# ---------------------------------------------------------------------------
# FFprobe integration (skipped when ffprobe unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path(__import__("shutil").which("ffprobe") or "").is_file()
    and not Path(__import__("shutil").which("ffmpeg") or "").is_file(),
    reason="ffprobe/ffmpeg not available on PATH",
)
def test_probe_synthetic_video(tmp_path):
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffprobe not available")
    import subprocess

    video_path = tmp_path / "clip.mp4"
    ffmpeg_exe = Path(__import__("shutil").which("ffmpeg") or "")
    if not ffmpeg_exe.is_file():
        pytest.skip("ffmpeg not available")
    cmd = [
        str(ffmpeg_exe),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=320x240:d=3",
        "-shortest",
        "-c:a",
        "aac",
        "-c:v",
        "libx264",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not video_path.is_file():
        pytest.skip("Could not generate a synthetic test video.")
    data = ffprobe_utils.probe_media(video_path, "ffmpeg/ffmpeg.exe")
    duration_ms, video_stream, audio_streams = ffprobe_utils.parse_video_probe(data)
    assert duration_ms > 0
    assert video_stream is not None
    assert video_stream.codec_type == "video"
    assert len(audio_streams) >= 1
    assert audio_streams[0].channels >= 1
