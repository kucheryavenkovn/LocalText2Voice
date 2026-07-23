from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui.cue_timeline_widget import CueTimelineWidget  # noqa: E402
from app.ui.video_dubbing_page import (  # noqa: E402
    LISTENING_MIX,
    LISTENING_ORIGINAL,
    LISTENING_TRANSLATION,
    VideoDubbingPage,
)
from app.core.video_dubbing.models import (  # noqa: E402
    DubbingCue,
    DubbingProject,
    VideoProbeInfo,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _tr(key, default):
    return default


def test_page_constructs(app):
    page = VideoDubbingPage(_tr, ffmpeg_path="ffmpeg/ffmpeg.exe", default_output_dir=".")
    assert page.play_button.text() == "Play"
    assert page.original_button.text() == "Original"


def test_listening_mode_switch(app):
    page = VideoDubbingPage(_tr, ffmpeg_path="ffmpeg/ffmpeg.exe", default_output_dir=".")
    page._set_listening_mode(LISTENING_MIX)
    assert page._listening_mode == LISTENING_MIX
    assert page.mix_button.isChecked() is True
    assert page.original_button.isChecked() is False


def test_refresh_table_populates_cues(app, tmp_path):
    page = VideoDubbingPage(_tr, ffmpeg_path="ffmpeg/ffmpeg.exe", default_output_dir=".")
    project = DubbingProject(
        project_id="u",
        project_dir=tmp_path,
        video_probe=VideoProbeInfo(duration_ms=10_000),
    )
    project.cues.append(
        DubbingCue(
            cue_id="1",
            sequence=1,
            start_ms=1000,
            end_ms=3000,
            duration_budget_ms=2000,
            source_text="Hi",
            spoken_text="Hi",
            raw_duration_ms=1800,
            applied_speed_factor=1.0,
            status="rendered",
        )
    )
    page._project = project
    page._refresh_ui_from_project()
    assert page.cue_table.rowCount() == 1
    assert page.cue_table.item(0, 0).text() == "1"


def test_timeline_widget_paints(app):
    widget = CueTimelineWidget()
    cue = DubbingCue(
        cue_id="1",
        sequence=1,
        start_ms=1000,
        end_ms=3000,
        duration_budget_ms=2000,
        source_text="x",
        spoken_text="x",
        status="rendered",
    )
    widget.set_cues([cue], 5000)
    widget.set_position_ms(2000)
    widget.set_selected(1)
    assert widget._selected_sequence == 1


def test_build_settings_round_trip(app):
    page = VideoDubbingPage(_tr, ffmpeg_path="ffmpeg/ffmpeg.exe", default_output_dir=".")
    settings = page._build_settings()
    assert settings.preferred_speed_limit == 1.35
    assert settings.hard_speed_limit == 2.50
    assert settings.guard_gap_ms == 20
    assert settings.ducking.original_during_percent == 15
    assert settings.export.container.value == "mkv"


def test_engine_context_injected(app):
    page = VideoDubbingPage(_tr, ffmpeg_path="ffmpeg/ffmpeg.exe", default_output_dir=".")
    page.set_engine_context(None, {"engine": "kokoro", "voice": "af_heart"}, "ffmpeg/ff.exe")
    assert page._voice_config_override["engine"] == "kokoro"
    assert page.voice_combo.currentText() == "af_heart"
    assert page._ffmpeg_path == "ffmpeg/ff.exe"
