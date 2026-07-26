from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui.cue_timeline_widget import CueTimelineWidget  # noqa: E402
from app.ui.video_dubbing_page import (  # noqa: E402
    LISTENING_MIX,
    VideoDubbingPage,
)
from app.core.video_dubbing.models import (  # noqa: E402
    DubbingCue,
    DubbingProject,
    DubbingProjectSettings,
    VideoProbeInfo,
)
from app.core.video_dubbing.project_store import DubbingProjectStore  # noqa: E402
from app.core.video_dubbing.service import VideoDubbingService  # noqa: E402
from app.tts.base import BaseTTSEngine  # noqa: E402


class _NoopTTS(BaseTTSEngine):
    def cancel_current(self) -> None:
        return None

    def validate(self, voice_config: dict[str, Any]) -> None:
        return None

    def synthesize_to_wav(
        self, text: str, output_wav: Path, voice_config: dict[str, Any]
    ) -> Path:
        raise AssertionError("UI round-trip must not synthesize")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _tr(key, default):
    return default


def test_page_constructs(app):
    page = VideoDubbingPage(_tr, ffmpeg_path="ffmpeg/ffmpeg.exe", default_output_dir=".")
    assert page.play_button.icon() and not page.play_button.icon().isNull()
    assert page.pause_button.icon() and not page.pause_button.icon().isNull()
    assert page.prev_cue_button.icon() and not page.prev_cue_button.icon().isNull()
    assert page.next_cue_button.icon() and not page.next_cue_button.icon().isNull()
    assert page.original_button.text()
    assert hasattr(page, "settings_tabs")
    assert page.settings_tabs.count() == 5
    assert hasattr(page, "top_splitter")
    assert page.cancel_button is not None


def test_next_prev_cue_by_selection(app, tmp_path):
    page = VideoDubbingPage(_tr, ffmpeg_path="ffmpeg/ffmpeg.exe", default_output_dir=str(tmp_path))
    project = DubbingProject(
        project_id="nav2",
        project_dir=tmp_path,
        video_probe=VideoProbeInfo(duration_ms=30_000),
    )
    for seq, start in ((1, 1000), (2, 5000), (3, 9000)):
        cue = DubbingCue(
            cue_id=str(seq),
            sequence=seq,
            start_ms=start,
            end_ms=start + 2000,
            duration_budget_ms=2000,
            source_text=f"t{seq}",
            spoken_text=f"t{seq}",
            status="rendered",
        )
        cue.ensure_source_timing()
        project.cues.append(cue)
    page._project = project
    page._refresh_ui_from_project()
    page._selected_sequence = 1
    page._goto_next_cue()
    assert page._selected_sequence == 2
    page._goto_next_cue()
    assert page._selected_sequence == 3
    page._goto_next_cue()
    assert page._selected_sequence == 3
    page._goto_prev_cue()
    assert page._selected_sequence == 2


def test_omnivoice_voice_selection_uses_combo_data_not_stale_reference(app, tmp_path, monkeypatch):
    page = VideoDubbingPage(_tr, ffmpeg_path="ffmpeg/ffmpeg.exe", default_output_dir=str(tmp_path))
    page.tts_engine_combo.setCurrentIndex(page.tts_engine_combo.findData("omnivoice") if page.tts_engine_combo.findData("omnivoice") >= 0 else 0)
    # Simulate stale Pedro reference left over from a previous selection.
    stale = tmp_path / "pedro.wav"
    stale.write_bytes(b"RIFF....WAVEfmt ")
    page._voice_config_override = {
        "engine": "omnivoice",
        "voice": "Pedro",
        "reference_audio_path": str(stale),
        "reference_text": "old pedro text",
    }
    woman = tmp_path / "woman.wav"
    woman.write_bytes(b"RIFF....WAVEfmt ")

    class _Voice:
        name = "Russian Woman"
        ref_text = "female sample"
        language = "ru"

    class _FakeCatalog:
        def resolve_voice_config(self, engine_id, voice_id, base_config=None):
            assert voice_id == "Russian Woman"
            assert "reference_audio_path" not in (base_config or {})
            return {
                "engine": engine_id,
                "voice": "Russian Woman",
                "reference_audio_path": str(woman),
                "reference_text": "female sample",
                "mode": "clone",
            }

        def list_voices(self, engine_id):
            from app.core.video_dubbing.voice_catalog import VoiceDescriptor

            return [
                VoiceDescriptor("Pedro", "Pedro — es", "es"),
                VoiceDescriptor("Russian Woman", "Russian Woman — ru", "ru"),
            ]

        def create_engine(self, engine_id, piper_path=None):
            return _NoopTTS()

    page.voice_catalog = _FakeCatalog()
    page._refresh_voices()
    idx = page.voice_combo.findData("Russian Woman")
    assert idx >= 0
    page.voice_combo.setCurrentIndex(idx)
    # Even if the visible text is the display label, selected id stays clean.
    assert page._selected_voice_id() == "Russian Woman"
    config = page._current_voice_config()
    assert config["reference_audio_path"] == str(woman)
    assert config["voice"] == "Russian Woman"
    assert "pedro" not in config["reference_audio_path"].lower()
    assert page._build_settings().voice == "Russian Woman"


def test_timeline_click_seeks_and_selects(app, tmp_path):
    page = VideoDubbingPage(_tr, ffmpeg_path="ffmpeg/ffmpeg.exe", default_output_dir=str(tmp_path))
    project = DubbingProject(
        project_id="nav",
        project_dir=tmp_path,
        video_probe=VideoProbeInfo(duration_ms=20_000),
    )
    project.video_path = tmp_path / "missing.mp4"  # media optional for selection
    project.cues.append(
        DubbingCue(
            cue_id="1",
            sequence=1,
            start_ms=5000,
            end_ms=7000,
            duration_budget_ms=2000,
            source_text="Hi",
            spoken_text="Hi",
            status="rendered",
            planned_start_ms=5200,
            planned_end_ms=7000,
        )
    )
    project.cues[0].ensure_source_timing()
    page._project = project
    page._refresh_ui_from_project()
    page._on_timeline_cue_selected(1)
    assert page._selected_sequence == 1
    assert page.timeline_widget._selected_sequence == 1
    # Jump uses planned start - pre_roll (1000) => 4200 when media exists;
    # without media path it still selects.
    assert page._cue_start_ms(project.cues[0]) == 5200


def test_full_hd_layout_geometry(app):
    page = VideoDubbingPage(_tr, ffmpeg_path="ffmpeg/ffmpeg.exe", default_output_dir=".")
    page.resize(1920, 1080)
    page.show()
    app.processEvents()
    assert page.cue_table.height() > 80
    sizes = page.top_splitter.sizes()
    assert len(sizes) == 2
    total = sum(sizes) or 1
    ratio = sizes[0] / total
    assert 0.35 <= ratio <= 0.65
    # Action buttons fit in two grid rows.
    assert len(page._action_buttons) <= 12
    page.close()


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
    assert not settings.compress_internal_pauses
    assert settings.ducking.original_during_percent == 15
    assert settings.export.container.value == "mkv"


def test_engine_context_injected(app):
    page = VideoDubbingPage(_tr, ffmpeg_path="ffmpeg/ffmpeg.exe", default_output_dir=".")
    page.set_engine_context(None, {"engine": "kokoro", "voice": "af_heart"}, "ffmpeg/ff.exe")
    assert page._voice_config_override["engine"] == "kokoro"
    assert page._selected_voice_id() == "af_heart"
    assert page._ffmpeg_path == "ffmpeg/ff.exe"


def test_service_can_reset_previous_cancellation(tmp_path):
    from app.core.video_dubbing.generation import GenerationCancelled

    service = VideoDubbingService(
        _NoopTTS(), store=DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    )
    service.cancel()
    try:
        service._check_cancelled()
        assert False, "expected GenerationCancelled"
    except GenerationCancelled:
        pass
    service.reset_cancel()
    service._check_cancelled()


def test_omnivoice_selection_survives_save_close_open(app, tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference fixture")
    store = DubbingProjectStore(db_path=tmp_path / "db.sqlite3")
    project = DubbingProject(
        project_id="omnivoice-round-trip",
        project_dir=tmp_path / "project",
        settings=DubbingProjectSettings(
            tts_engine="omnivoice",
            voice="Russian Woman",
            voice_config={
                "engine": "omnivoice",
                "reference_audio_path": str(reference),
                "reference_text": "Reference text",
            },
        ),
    )
    service = VideoDubbingService(_NoopTTS(), store=store)
    service.save_project(project)

    first = VideoDubbingPage(_tr, ffmpeg_path="ffmpeg", default_output_dir=str(tmp_path))
    first._service = service
    first._tts_engine = service.tts_engine
    first._project = project
    first._apply_project_to_ui()
    first._save_project()
    manifest = project.project_dir / "dubbing_project.json"
    first.close()

    second = VideoDubbingPage(_tr, ffmpeg_path="ffmpeg", default_output_dir=str(tmp_path))
    second._service = VideoDubbingService(_NoopTTS(), store=store)
    second._tts_engine = second._service.tts_engine
    second._load_project_from_manifest(manifest)

    assert second._project is not None
    settings = second._project.settings
    assert settings.tts_engine == "omnivoice"
    assert settings.voice == "Russian Woman"
    assert settings.voice_config["engine"] == "omnivoice"
    assert Path(settings.voice_config["reference_audio_path"]).is_file()
    assert settings.voice_config["reference_text"]
    second.close()
